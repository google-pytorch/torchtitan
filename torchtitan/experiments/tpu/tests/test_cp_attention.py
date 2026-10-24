"""Tests for TPU context parallelism (``torchtitan/experiments/tpu/cp_attention.py``).

The CP attentions run here on CPU, every rank of a CP group on its own thread.
The collectives they issue are emulated across those threads: the functional
collectives of the traced (``torch.compile``) path and ``spmd.redistribute`` of
the eager path. Outputs and gradients are compared against plain single-device
attention over the unsharded sequences.
"""

import concurrent.futures
import contextlib
import sys
import threading
import types
from types import SimpleNamespace
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
# pyformat: disable
# Import torch before spmd_types to avoid circular import with torch_tpu.
import torch
import spmd_types as spmd
# pyformat: enable
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
import torch.nn.functional as F
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.experiments.tpu import cp_attention
from torchtitan.experiments.tpu.activation_checkpoint import TPUSelectiveAC
from torchtitan.experiments.tpu.cp_attention import (
    KVAllGatherCPInnerAttention,
    KVAllGatherCPScaledDotProductInnerAttention,
    KVAllGatherCPVarlenInnerAttention,
    UlyssesCPInnerAttention,
    UlyssesCPScaledDotProductInnerAttention,
    UlyssesCPVarlenInnerAttention,
    reorder_sequence,
    validate_tpu_context_parallel,
)
from torchtitan.experiments.tpu.headtail import headtail_positions
from torchtitan.experiments.tpu.qwen3 import config_registry
from torchtitan.experiments.tpu.unflatten_shim import get_unflatten_seq_len
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common.attention import segment_ids_from_positions

_C10D = torch.ops._c10d_functional
_WAIT = _C10D.wait_tensor.default
_COLLECTIVES = frozenset({
    _C10D.all_to_all_single.default,
    _C10D.all_gather_into_tensor.default,
    _C10D.reduce_scatter_tensor.default,
})


class _Ranks:
  """Runs one function per CP rank on its own thread; emulates collectives."""

  def __init__(self, world_size: int):
    self.world_size = world_size
    self._barrier = threading.Barrier(world_size, timeout=120)
    self._slots = [None] * world_size
    self._local = threading.local()

  @property
  def rank(self) -> int:
    return self._local.rank

  def group(self):
    return SimpleNamespace(size=lambda: self.world_size, rank=lambda: self.rank)

  def exchange(self, value):
    """Returns every rank's ``value`` in rank order (an all-gather)."""
    self._slots[self.rank] = value
    self._barrier.wait()
    gathered = list(self._slots)
    self._barrier.wait()
    return gathered

  def run(self, fn):
    def run_rank(rank):
      self._local.rank = rank
      try:
        return fn(rank)
      except BaseException:
        self._barrier.abort()  # Unblock the other ranks.
        raise

    with concurrent.futures.ThreadPoolExecutor(self.world_size) as pool:
      return list(pool.map(run_rank, range(self.world_size)))


def _all_to_all(ranks: _Ranks, x: torch.Tensor) -> torch.Tensor:
  """Equal-split all-to-all along dim 0: chunk i goes to rank i."""
  chunks = ranks.exchange(x.chunk(ranks.world_size))
  return torch.cat([c[ranks.rank] for c in chunks])


class _AllToAll(torch.autograd.Function):
  """``funcol.all_to_all_single`` (differentiable, as on torch >= 2.11)."""

  @staticmethod
  def forward(ctx, x, ranks):
    ctx.ranks = ranks
    return _all_to_all(ranks, x)

  @staticmethod
  def backward(ctx, grad):
    return _all_to_all(ctx.ranks, grad.contiguous()), None


class _Redistribute(torch.autograd.Function):
  """``spmd.redistribute`` from S(src_dim) to S(dst_dim), or to R if None."""

  @staticmethod
  def forward(ctx, x, ranks, src_dim, dst_dim):
    ctx.ranks, ctx.src_dim, ctx.dst_dim = ranks, src_dim, dst_dim
    full = torch.cat(ranks.exchange(x), dim=src_dim)
    if dst_dim is None:
      return full
    return full.chunk(ranks.world_size, dim=dst_dim)[ranks.rank].contiguous()

  @staticmethod
  def backward(ctx, grad):
    ranks = ctx.ranks
    if ctx.dst_dim is None:  # All-gather: the gradient is a reduce-scatter.
      full = torch.stack(ranks.exchange(grad)).sum(0)
    else:
      full = torch.cat(ranks.exchange(grad), dim=ctx.dst_dim)
    shard = full.chunk(ranks.world_size, dim=ctx.src_dim)[ranks.rank]
    return shard.contiguous(), None, None, None


@contextlib.contextmanager
def _emulated_cp_group(ranks: _Ranks, traced: bool):
  """Routes the CP attentions' collectives through ``ranks``."""

  def all_to_all_single(x, output_split_sizes, input_split_sizes, group):
    del group
    assert output_split_sizes is None and input_split_sizes is None
    return _AllToAll.apply(x, ranks)

  def all_gather_tensor(x, gather_dim, group):
    del group
    return torch.cat(ranks.exchange(x), dim=gather_dim)

  def reduce_scatter_tensor(x, reduce_op, scatter_dim, group):
    del group
    assert reduce_op == "sum"
    full = torch.stack(ranks.exchange(x)).sum(0)
    return full.chunk(ranks.world_size, dim=scatter_dim)[
        ranks.rank
    ].contiguous()

  def redistribute(x, group, *, src, dst, backward_options=None):
    del group, backward_options
    dst_dim = dst.dim if isinstance(dst, spmd.Shard) else None
    return _Redistribute.apply(x, ranks, src.dim, dst_dim)

  with contextlib.ExitStack() as stack:
    stack.enter_context(
        mock.patch.object(
            cp_attention, "spmd_mesh_group", lambda axis: ranks.group()
        )
    )
    if traced:
      # The branch the CP attentions take under torch.compile.
      stack.enter_context(
          mock.patch.object(cp_attention, "_tracing", lambda: True)
      )
      for name, fn in (
          ("all_to_all_single", all_to_all_single),
          ("all_gather_tensor", all_gather_tensor),
          ("reduce_scatter_tensor", reduce_scatter_tensor),
      ):
        stack.enter_context(mock.patch.object(funcol, name, fn))
    else:
      stack.enter_context(mock.patch.object(spmd, "redistribute", redistribute))
    yield


def _shard(x: torch.Tensor, strategy: str, cp: int, rank: int, seq_len: int):
  """Rank ``rank``'s shard of a folded ``[B * S, ...]`` tensor, as cp_shard lays it out."""
  if strategy == "ulysses":
    return x.chunk(cp)[rank]
  rest = list(x.shape[1:])
  bsz = x.shape[0] // seq_len
  x = reorder_sequence(x.reshape([bsz, seq_len] + rest), cp, seq_dim=1)
  x = x.reshape([bsz, cp, seq_len // cp] + rest).movedim(1, 0)
  return x[rank].reshape([bsz * seq_len // cp] + rest)


def _unshard(shards, strategy: str, cp: int, seq_len: int) -> torch.Tensor:
  """Inverse of ``_shard`` over all ranks."""
  x = torch.cat(shards)
  if strategy == "ulysses":
    return x
  rest = list(x.shape[1:])
  bsz = x.shape[0] // seq_len
  x = x.reshape([cp, bsz, seq_len // cp] + rest).movedim(0, 1)
  x = x.reshape([bsz, seq_len] + rest)
  x = reorder_sequence(x, cp, seq_dim=1, to_contiguous=True)
  return x.reshape([bsz * seq_len] + rest)


def _packed_positions(doc_lengths_per_row) -> torch.Tensor:
  """Folded ``[B * S]`` positions that restart at 0 for every packed document."""
  return torch.cat(
      [torch.arange(n) for row in doc_lengths_per_row for n in row]
  )


def _dense_attention(q, k, v, *, seq_len, segment_ids=None, scale=None):
  """Causal (and same-document) attention over folded ``[B * S, H, D]`` inputs."""
  bsz = q.shape[0] // seq_len
  q, k, v = (t.reshape(bsz, seq_len, *t.shape[1:]) for t in (q, k, v))
  rep = q.shape[2] // k.shape[2]
  k, v = (t.repeat_interleave(rep, dim=2) for t in (k, v))
  pos = torch.arange(seq_len)
  mask = (pos[:, None] >= pos[None, :]).expand(bsz, seq_len, seq_len)
  if segment_ids is not None:
    seg = segment_ids.reshape(bsz, seq_len)
    mask = mask & (seg[:, :, None] == seg[:, None, :])
  out = F.scaled_dot_product_attention(
      q.transpose(1, 2),
      k.transpose(1, 2),
      v.transpose(1, 2),
      attn_mask=mask[:, None],
      scale=scale,
  )
  return out.transpose(1, 2).reshape(bsz * seq_len, out.shape[1], -1)


def _dense_local_attention(q, k, v, *, segment_ids=None, scale=None, **unused):
  """A stand-in attention kernel over the full sequence (for Ulysses CP)."""
  del unused
  return _dense_attention(
      q,
      k,
      v,
      seq_len=get_unflatten_seq_len(),
      segment_ids=segment_ids,
      scale=scale,
  )


# Two sequences of 32 tokens, with documents crossing the head-tail chunks.
_SEQ_LEN = 32
_DOCS = ((5, 11, 9, 7), (16, 3, 13))


class CPAttentionParityTest(parameterized.TestCase):
  """CP attention on emulated multi-rank groups vs single-device attention."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    cp_attention._CP_RANK_CACHE.clear()
    self.addCleanup(cp_attention._CP_RANK_CACHE.clear)

  def _run(
      self, strategy, make_attn, cp, n_heads, n_kv_heads, kwargs_fn, traced
  ):
    """Runs CP forward + backward on every rank; returns unsharded results."""
    tokens, head_dim = _SEQ_LEN * len(_DOCS), 8
    q = torch.randn(tokens, n_heads, head_dim)
    k = torch.randn(tokens, n_kv_heads, head_dim)
    v = torch.randn(tokens, n_kv_heads, head_dim)
    grad = torch.randn(tokens, n_heads, head_dim)
    positions = _packed_positions(_DOCS)
    ranks = _Ranks(cp)

    def run_rank(rank):
      local = [
          _shard(t, strategy, cp, rank, _SEQ_LEN).clone().requires_grad_(True)
          for t in (q, k, v)
      ]
      local_positions = _shard(positions, strategy, cp, rank, _SEQ_LEN)
      out = make_attn()(*local, **kwargs_fn(local_positions))
      out.backward(_shard(grad, strategy, cp, rank, _SEQ_LEN))
      return out.detach(), *(t.grad for t in local)

    with set_unflatten_seq_len(_SEQ_LEN), _emulated_cp_group(ranks, traced):
      results = ranks.run(run_rank)
    got = [
        _unshard([r[i] for r in results], strategy, cp, _SEQ_LEN)
        for i in range(4)
    ]
    return (q, k, v, grad, positions), got

  def _check(self, inputs, got, *, use_documents):
    q, k, v, grad, positions = inputs
    q, k, v = (t.clone().requires_grad_(True) for t in (q, k, v))
    segment_ids = (
        segment_ids_from_positions(positions) if use_documents else None
    )
    want = _dense_attention(q, k, v, seq_len=_SEQ_LEN, segment_ids=segment_ids)
    want.backward(grad)
    for name, g, w in zip(
        ("out", "dq", "dk", "dv"), got, (want.detach(), q.grad, k.grad, v.grad)
    ):
      torch.testing.assert_close(g, w, rtol=2e-5, atol=2e-5, msg=name)

  @parameterized.product(traced=(False, True), cp=(2, 4), varlen=(False, True))
  def test_kv_all_gather_matches_dense_attention(self, traced, cp, varlen):
    """KV all-gather CP (eager and traced collectives, B=2, packed docs) matches dense attention."""
    cls = (
        KVAllGatherCPVarlenInnerAttention
        if varlen
        else KVAllGatherCPScaledDotProductInnerAttention
    )

    def kwargs_fn(local_positions):
      # What GQAttention passes. Varlen masks documents through positions;
      # SDPA, like single-device SDPA, masks causally only.
      return dict(
          attention_masks=object() if varlen else None,
          scale=None,
          enable_gqa=True,
          positions=local_positions,
          segment_ids=segment_ids_from_positions(local_positions),
      )

    inputs, got = self._run(
        "all_gather", lambda: cls(cls.Config()), cp, 4, 2, kwargs_fn, traced
    )
    self._check(inputs, got, use_documents=varlen)

  @parameterized.product(
      traced=(False, True), heads=((4, 2, 2), (4, 1, 2), (8, 4, 4))
  )
  def test_ulysses_matches_dense_attention(self, traced, heads):
    """Ulysses CP (eager and traced all-to-alls, KV-head repeat, packed docs) matches dense attention."""
    n_heads, n_kv_heads, cp = heads

    def make_attn():
      cls = UlyssesCPScaledDotProductInnerAttention
      attn = cls(cls.Config())
      # A kernel sees the full sequence and the documents' segment ids.
      attn.set_local_attention(_dense_local_attention)
      return attn

    def kwargs_fn(local_positions):
      return dict(
          attention_masks=None,
          scale=None,
          enable_gqa=n_heads != n_kv_heads,
          positions=local_positions,
          segment_ids=segment_ids_from_positions(local_positions),
      )

    inputs, got = self._run(
        "ulysses", make_attn, cp, n_heads, n_kv_heads, kwargs_fn, traced
    )
    self._check(inputs, got, use_documents=True)


class CPShardTest(parameterized.TestCase):
  """KV all-gather input sharding and the head-tail order."""

  def setUp(self):
    super().setUp()
    cp_attention._CP_RANK_CACHE.clear()  # cp_shard caches the rank tensor.
    self.addCleanup(cp_attention._CP_RANK_CACHE.clear)

  @parameterized.parameters((2, 1), (2, 3), (4, 1), (4, 2))
  def test_kv_all_gather_cp_shard_layout(self, cp, bsz):
    """cp_shard gives each rank the shard forward expects, with B == 1 and B > 1."""
    seq_len = 16
    tokens = torch.arange(bsz * seq_len)
    for rank in range(cp):

      def prepare(input_dict, input_shardings, cp_mesh, lb_type, ptrr_key):
        # Core prepare_context_parallel_input: torch's head-tail balancer (or
        # none), then rank's contiguous chunk along dim 0.
        del input_shardings, cp_mesh, ptrr_key
        for key in ("input", "labels", "positions"):
          t = input_dict[key]
          if lb_type == "headtail":
            lb = _HeadTailLoadBalancer(t.shape[0], cp, "cpu")
            t = t[lb._generate_indices().reshape(-1).long()]
          input_dict[key] = t.chunk(cp)[rank]
        return input_dict

      cp_mesh = SimpleNamespace(
          device_type="cpu", get_local_rank=lambda: rank, size=lambda dim=0: cp
      )
      batch = {
          "input": tokens.clone(),
          "labels": tokens + 1000,
          "positions": tokens % seq_len,
      }
      with set_unflatten_seq_len(seq_len), mock.patch.object(
          cp_attention, "prepare_context_parallel_input", prepare
      ):
        out = KVAllGatherCPInnerAttention.cp_shard(
            batch, None, cp_mesh, "headtail", None
        )
      want = _shard(tokens, "all_gather", cp, rank, seq_len)
      torch.testing.assert_close(out["input"], want)
      torch.testing.assert_close(out["labels"], want + 1000)
      torch.testing.assert_close(out["positions"], want % seq_len)
      # The rank tensor the compiled attention will take as an input.
      self.assertEqual(
          cp_attention._CP_RANK_CACHE[(torch.device("cpu"), rank)].tolist(),
          [rank],
      )

  def test_kv_all_gather_cp_shard_respects_input_shardings(self):
    """When input_shardings is given with B > 1, every CP-sharded key is head-tail reordered."""
    cp, bsz, seq_len = 2, 2, 16
    tokens = torch.arange(bsz * seq_len)
    shardings = {
        "input": spmd.SpmdType(local_type={MeshAxisName.CP: spmd.S(0)}),
        "extra_token_ids": spmd.SpmdType(
            local_type={MeshAxisName.CP: spmd.S(0)}
        ),
        "replicated_aux": spmd.SpmdType(local_type={MeshAxisName.CP: spmd.R}),
    }
    for rank in range(cp):

      def prepare(input_dict, input_shardings, cp_mesh, lb_type, ptrr_key):
        del cp_mesh, lb_type, ptrr_key
        for key in cp_attention._cp_shard_dims(input_shardings):
          input_dict[key] = input_dict[key].chunk(cp)[rank]
        return input_dict

      cp_mesh = SimpleNamespace(
          device_type="cpu", get_local_rank=lambda: rank, size=lambda dim=0: cp
      )
      batch = {
          "input": tokens.clone(),
          "extra_token_ids": tokens + 500,
          "replicated_aux": torch.tensor([1, 2, 3]),
      }
      with set_unflatten_seq_len(seq_len), mock.patch.object(
          cp_attention, "prepare_context_parallel_input", prepare
      ):
        out = KVAllGatherCPInnerAttention.cp_shard(
            batch, shardings, cp_mesh, "headtail", None
        )
      want = _shard(tokens, "all_gather", cp, rank, seq_len)
      torch.testing.assert_close(out["input"], want)
      torch.testing.assert_close(out["extra_token_ids"], want + 500)
      torch.testing.assert_close(out["replicated_aux"], torch.tensor([1, 2, 3]))

    # A CP-sharded key whose shape does not match B * S raises clearly.
    cp_mesh = SimpleNamespace(
        device_type="cpu", get_local_rank=lambda: 0, size=lambda dim=0: cp
    )
    bad_batch = {
        "input": tokens.clone(),
        "extra_token_ids": torch.arange(7),
    }
    with (
        set_unflatten_seq_len(seq_len),
        self.assertRaisesRegex(ValueError, "extra_token_ids"),
    ):
      KVAllGatherCPInnerAttention.cp_shard(
          bad_batch, shardings, cp_mesh, "headtail", None
      )

  def test_cp_rank_cache_device_normalization(self):
    """_CP_RANK_CACHE normalizes device keys across cp_shard, init_weights and forward."""
    self.assertEqual(
        cp_attention._normalize_cp_device("cpu"),
        cp_attention._normalize_cp_device(torch.device("cpu", 0)),
    )
    fake_accel_mod = SimpleNamespace(current_device=lambda: 3)
    with mock.patch.object(torch, "cuda", fake_accel_mod):
      self.assertEqual(
          cp_attention._normalize_cp_device("cuda"),
          torch.device("cuda", 3),
      )
      self.assertEqual(
          cp_attention._normalize_cp_device(torch.device("cuda", 1)),
          torch.device("cuda", 1),
      )

    attn = KVAllGatherCPScaledDotProductInnerAttention(
        KVAllGatherCPScaledDotProductInnerAttention.Config(
            allow_reference_attention=True
        )
    )
    group = SimpleNamespace(rank=lambda: 1, size=lambda: 2)
    with mock.patch.object(cp_attention, "spmd_mesh_group", lambda axis: group):
      attn.init_weights(buffer_device=torch.device("cpu", 0))
    cached = cp_attention._CP_RANK_CACHE[(torch.device("cpu"), 1)]
    self.assertIs(
        attn._get_cp_rank_tensor(torch.device("cpu"), group),
        cached,
    )

  @parameterized.parameters(
      (16, 2), (16, 4), (16, 8), (48, 3), (64, 4), (64, 8)
  )
  def test_headtail_order_matches_torch_load_balancer(self, seq_len, cp):
    """headtail.py (kernels, B > 1 inputs) and torch's balancer (B == 1 inputs) agree."""
    lb = _HeadTailLoadBalancer(seq_len, cp, "cpu")
    self.assertEqual(
        headtail_positions(seq_len, cp).tolist(),
        lb._generate_indices().reshape(-1).tolist(),
    )


@unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
class CompiledCPCollectivesTest(parameterized.TestCase):
  """Traced CP attentions compile to one graph with one wait per collective.

  torch-tpu's split compiler pairs every ``wait_tensor`` with the collective
  that produced its input and rejects a wait on a graph input or a second
  wait. Runs on a fake 2-rank process group, so only graph structure matters.
  """

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if dist.is_initialized():
      raise unittest.SkipTest("a process group is already initialized")
    from torch.testing._internal.distributed.fake_pg import FakeStore  # pylint: disable=g-import-not-at-top

    dist.init_process_group("fake", rank=0, world_size=2, store=FakeStore())

  @classmethod
  def tearDownClass(cls):
    dist.destroy_process_group()
    super().tearDownClass()

  def setUp(self):
    super().setUp()
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    cp_attention._CP_RANK_CACHE.clear()
    self.addCleanup(cp_attention._CP_RANK_CACHE.clear)
    self.enter_context(
        mock.patch.object(
            cp_attention, "spmd_mesh_group", lambda axis: dist.group.WORLD
        )
    )

  def _graphs(self, fn, *args):
    """Compiles ``fn`` into one graph, runs forward + backward; returns the AOT graphs."""
    from functorch.compile import min_cut_rematerialization_partition  # pylint: disable=g-import-not-at-top
    from torch._dynamo.backends.common import aot_autograd  # pylint: disable=g-import-not-at-top

    graphs = []

    def capture(gm, _):
      graphs.append(gm)
      return gm

    backend = aot_autograd(
        fw_compiler=capture,
        bw_compiler=capture,
        partition_fn=min_cut_rematerialization_partition,
    )
    # fullgraph: a graph break fails the test.
    torch.compile(fn, backend=backend, fullgraph=True)(*args).sum().backward()
    return graphs

  def _assert_waits_pair_with_collectives(self, graphs):
    code = "\n".join(gm.code for gm in graphs)
    for gm in graphs:
      waited = []
      n_collectives = 0
      for node in gm.graph.nodes:
        if node.target is _WAIT:
          producer = node.args[0]
          self.assertEqual(producer.op, "call_function", code)
          self.assertIn(producer.target, _COLLECTIVES, code)
          waited.append(producer)
        elif node.target in _COLLECTIVES:
          n_collectives += 1
      # One wait per collective, in every graph.
      self.assertLen(set(waited), len(waited), code)
      self.assertLen(waited, n_collectives, code)
    self.assertTrue(
        any(n.target in _COLLECTIVES for gm in graphs for n in gm.graph.nodes)
    )

  def _inputs(self):
    q = torch.randn(8, 4, 8, requires_grad=True)
    k = torch.randn(8, 2, 8, requires_grad=True)
    v = torch.randn(8, 2, 8, requires_grad=True)
    return q, k, v, torch.tensor([0, 1, 2, 0, 1, 0, 1, 2])

  @parameterized.parameters(False, True)
  def test_ulysses_waits_each_all_to_all_once(self, use_kernel):
    cls = UlyssesCPScaledDotProductInnerAttention
    attn = cls(cls.Config())
    if use_kernel:
      # Consumes the all-gathered segment ids, as Splash does.
      attn.set_local_attention(_dense_local_attention)
    q, k, v, positions = self._inputs()

    def fn(q, k, v):
      return attn(q, k, v, enable_gqa=True, positions=positions)

    with set_unflatten_seq_len(16):
      graphs = self._graphs(fn, q, k, v)
    self._assert_waits_pair_with_collectives(graphs)

  @parameterized.parameters(False, True)
  def test_kv_all_gather_waits_each_collective_once(self, with_positions):
    # cp_shard caches the rank tensor before tracing, so the graphs take it as
    # an input (one graph for all ranks) and the attention indexes with it.
    cp_attention._cache_cp_rank_for_mesh(
        SimpleNamespace(device_type="cpu", get_local_rank=lambda: 0)
    )
    cls = KVAllGatherCPScaledDotProductInnerAttention
    attn = cls(cls.Config())
    q, k, v, positions = self._inputs()

    def fn(q, k, v):
      kwargs = dict(positions=positions) if with_positions else {}
      return attn(q, k, v, enable_gqa=True, **kwargs)

    graphs = self._graphs(fn, q, k, v)
    self._assert_waits_pair_with_collectives(graphs)
    self.assertTrue(
        any(
            node.op == "placeholder"
            and node.meta.get("val") is not None
            and node.meta["val"].dtype == torch.int32
            for node in graphs[0].graph.nodes
        ),
        "the CP rank is not a graph input",
    )

  def test_tpu_selective_ac_recomputes_collectives_with_their_waits(self):
    """Under SAC, core keeps waits on saved tensors; TPUSelectiveAC does not."""

    class Block(torch.nn.Module):

      def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 4))

      def forward(self, x):
        group = dist.group.WORLD
        y = funcol.all_to_all_single(x @ self.w, None, None, group)
        y = torch.sin(y) * y
        y = funcol.all_to_all_single(y.contiguous(), None, None, group)
        return torch.cos(y) @ self.w

    def waits_on_saved_tensors(ac_config):
      block = ac_config.build()._wrap_block(Block())
      graphs = self._graphs(block, torch.randn(8, 4, requires_grad=True))
      return [
          node.format_node()
          for gm in graphs
          for node in gm.graph.nodes
          if node.target is _WAIT and node.args[0].op == "placeholder"
      ]

    self.assertNotEmpty(waits_on_saved_tensors(SelectiveAC.Config()))
    self.assertEmpty(waits_on_saved_tensors(TPUSelectiveAC.Config()))


class CPValidationTest(parameterized.TestCase):
  """Unsupported CP configurations raise up front."""

  def _config(self, strategy, cp=2):
    config = (
        config_registry.qwen3_debugmodel_allgather_cp2()
        if strategy == "all_gather"
        else config_registry.qwen3_debugmodel_ulysses_cp2()
    )
    config.parallelism.context_parallel_degree = cp
    return config

  @parameterized.parameters("all_gather", "ulysses")
  def test_rejects_pipeline_parallelism(self, strategy):
    config = self._config(strategy)
    config.parallelism.pipeline_parallel_degree = 2
    with self.assertRaisesRegex(NotImplementedError, "pipeline parallelism"):
      validate_tpu_context_parallel(config.model_spec.model, config.parallelism)

  @parameterized.named_parameters(
      (
          "all_gather_seq_len",
          "all_gather",
          4,
          2052,
          2052 * 8,
          "max_context_length",
      ),
      ("all_gather_ragged_batch", "all_gather", 2, 2048, 3000, "multiple of"),
      (
          "all_gather_short_stream",
          "all_gather",
          2,
          2048,
          1026,
          "divisible by 4",
      ),
      ("ulysses_tokens", "ulysses", 4, 2048, 16386, "divisible by cp"),
  )
  def test_rejects_unshardable_sequence_lengths(
      self, strategy, cp, seq_len, num_tokens, message
  ):
    config = self._config(strategy, cp)
    config.training.max_context_length = seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = num_tokens
    with self.assertRaisesRegex(ValueError, message):
      config.__post_init__()

  def test_accepts_shardable_sequence_lengths(self):
    for strategy, cp in (("all_gather", 2), ("all_gather", 8), ("ulysses", 8)):
      config = self._config(strategy, cp)
      config.training.max_context_length = 128
      config.training.num_tokens_per_microbatch_per_dp_rank = 256
      config.__post_init__()

  @parameterized.named_parameters(
      ("q_block", dict(sa_block_q=384), "sa_block_q"),
      ("kv_block", dict(sa_block_kv=3072), "sa_block_kv"),
      (
          "dq_block",
          dict(sa_use_fused_bwd_kernel=False, sa_block_q_dq=384),
          "sa_block_q_dq",
      ),
      (
          "tokamax",
          dict(
              use_splash_attention_kernel=False,
              use_tokamax_splash_attention_kernel=True,
              sa_block_kv_compute=3072,
          ),
          "sa_block_kv_compute",
      ),
  )
  def test_rejects_splash_blocks_that_do_not_tile_the_shards(
      self, overrides, message
  ):
    config = self._config("all_gather", cp=4)
    config.training.max_context_length = 4096  # 1024 queries per rank.
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.splash_attention_kernel.use_splash_attention_kernel = True
    for name, value in overrides.items():
      setattr(config.splash_attention_kernel, name, value)
    with self.assertRaisesRegex(ValueError, message):
      config.__post_init__()

  def test_accepts_splash_blocks_capped_at_the_shard_length(self):
    config = self._config("all_gather", cp=4)
    config.training.max_context_length = 1024  # 256 queries per rank < 512.
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.splash_attention_kernel.use_splash_attention_kernel = True
    config.__post_init__()

  def test_rejects_tokamax_diag_skip_with_head_tail_cp(self):
    for flag in ("sa_qk_diag_skip", "sa_sv_diag_skip"):
      config = self._config("all_gather", cp=4)
      config.splash_attention_kernel.use_tokamax_splash_attention_kernel = True
      setattr(config.splash_attention_kernel, flag, True)
      with self.assertRaisesRegex(
          NotImplementedError, "Tokamax \\*_diag_skip assumes natural Q order"
      ):
        config.__post_init__()

  def test_rejects_non_causal_kv_all_gather(self):
    config = self._config("all_gather")
    for layer in config.model_spec.model.layers:
      layer.attention.inner_attention = (
          KVAllGatherCPVarlenInnerAttention.Config(window_size=(-1, -1))
      )
    with self.assertRaisesRegex(NotImplementedError, "causal attention only"):
      validate_tpu_context_parallel(config.model_spec.model, config.parallelism)
    with self.assertRaisesRegex(NotImplementedError, "causal attention only"):
      KVAllGatherCPVarlenInnerAttention(
          KVAllGatherCPVarlenInnerAttention.Config(window_size=(4, 2))
      )
    # Causal sliding windows are fine (with a kernel).
    KVAllGatherCPVarlenInnerAttention(
        KVAllGatherCPVarlenInnerAttention.Config(window_size=(4, 0))
    )

  def test_ulysses_sdpa_rejects_sliding_windows_without_a_kernel(self):
    cls = UlyssesCPScaledDotProductInnerAttention
    attn = cls(cls.Config())
    attn.sliding_window_size = 4
    q = torch.randn(1, 16, 2, 8)
    with self.assertRaisesRegex(NotImplementedError, "sliding windows"):
      attn.local_forward(q, q, q)

  def test_out_transform_is_rejected(self):
    q = torch.randn(8, 2, 4)
    epilogue = lambda out, lse: out
    self.enter_context(
        mock.patch.object(cp_attention, "spmd_mesh_group", return_value=None)
    )
    for cls in (
        UlyssesCPScaledDotProductInnerAttention,
        KVAllGatherCPScaledDotProductInnerAttention,
        KVAllGatherCPVarlenInnerAttention,
    ):
      with self.assertRaisesRegex(NotImplementedError, "out_transform"):
        cls(cls.Config())(q, q, q, out_transform=epilogue)
    # The plain Varlen attention applies it, so Ulysses lets it through (and
    # then stops here at the missing CP mesh).
    attn = UlyssesCPVarlenInnerAttention(UlyssesCPVarlenInnerAttention.Config())
    with self.assertRaisesRegex(RuntimeError, "active CP mesh axis"):
      attn(q, q, q, out_transform=epilogue)
    attn.set_local_attention(lambda *args, **kwargs: None)
    with self.assertRaisesRegex(NotImplementedError, "out_transform"):
      attn(q, q, q, out_transform=epilogue)

  def test_kv_all_gather_rejects_masks_it_cannot_apply(self):
    q = torch.randn(8, 2, 4)
    sdpa = KVAllGatherCPScaledDotProductInnerAttention(
        KVAllGatherCPScaledDotProductInnerAttention.Config()
    )
    with self.assertRaisesRegex(ValueError, "attention_masks"):
      sdpa(q, q, q, attention_masks=object())
    varlen = KVAllGatherCPVarlenInnerAttention(
        KVAllGatherCPVarlenInnerAttention.Config()
    )
    with self.assertRaisesRegex(ValueError, "needs positions"):
      varlen(q, q, q, attention_masks=object())

  @parameterized.parameters(
      "qwen3_1_7b_allgather_cp4", "qwen3_1_7b_ulysses_cp4"
  )
  def test_qwen3_1_7b_cp_recipes_build_and_validate(self, recipe):
    config = getattr(config_registry, recipe)()
    self.assertEqual(config.parallelism.context_parallel_degree, 4)
    self.assertEqual(config.training.max_context_length, 32768)
    self.assertTrue(config.splash_attention_kernel.use_splash_attention_kernel)
    config.__post_init__()  # Core + TPU CP validation, incl. Splash blocks.


class InstallTPUCPPatchesTest(absltest.TestCase):
  """install_tpu_cp_patches hands FSDP2 the flattened (dp_shard, cp) mesh."""

  def test_flattens_dp_shard_and_cp_once(self):
    from torchtitan.distributed import fsdp as fsdp_mod  # pylint: disable=g-import-not-at-top

    class FakeMesh:

      def __getitem__(self, names):
        if isinstance(names, tuple):
          return SimpleNamespace(_flatten=lambda name: ("flat", names, name))
        return ("sub", names)

    def resolve(parallel_dims):
      del parallel_dims
      return FakeMesh(), "dims"

    # A model's parallelize module that bound resolve_fsdp_mesh at import.
    parallelize_mod = types.ModuleType("torchtitan._test_parallelize")
    parallelize_mod.resolve_fsdp_mesh = resolve
    self.enter_context(
        mock.patch.object(fsdp_mod, "resolve_fsdp_mesh", resolve)
    )
    self.enter_context(
        mock.patch.dict(
            sys.modules, {parallelize_mod.__name__: parallelize_mod}
        )
    )
    # Released torch has no spmd_types FSDP (the case the patch is for).
    self.enter_context(
        mock.patch.object(cp_attention, "FSDPParam", SimpleNamespace())
    )
    self.enter_context(mock.patch.object(cp_attention, "_patched", False))
    self.enter_context(
        mock.patch.object(
            DeviceMesh,
            "_concatenate",
            lambda meshes: ("concat", meshes),
            create=True,
        )
    )

    cp_attention.install_tpu_cp_patches()
    patched = parallelize_mod.resolve_fsdp_mesh
    self.assertIsNot(patched, resolve)
    self.assertIs(fsdp_mod.resolve_fsdp_mesh, patched)

    def dims(**overrides):
      fields = dict(
          cp_enabled=True,
          tp_enabled=False,
          ep_enabled=False,
          dp_replicate_enabled=False,
      )
      fields.update(overrides)
      return SimpleNamespace(**fields)

    flat = ("flat", ("dp_shard", "cp"), "dp_shard_cp")
    self.assertEqual(patched(dims()), (flat, None))
    self.assertEqual(
        patched(dims(dp_replicate_enabled=True)),
        (("concat", [("sub", "dp_replicate"), flat]), None),
    )
    for overrides in (
        dict(cp_enabled=False),
        dict(tp_enabled=True),
        dict(ep_enabled=True),
    ):
      mesh, mesh_dims = patched(dims(**overrides))
      self.assertIsInstance(mesh, FakeMesh)
      self.assertEqual(mesh_dims, "dims")

    cp_attention.install_tpu_cp_patches()  # Idempotent.
    self.assertIs(parallelize_mod.resolve_fsdp_mesh, patched)


if __name__ == "__main__":
  absltest.main()
