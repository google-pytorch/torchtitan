"""Tests for Qwen3.5 context parallelism on TPU (``qwen3_5/cp_gated_delta.py``).

Every rank of a CP group runs on its own CPU thread and the collectives are
emulated across the threads (the harness of ``tests/test_cp_attention.py``):
the eager (``spmd.redistribute``) and the traced (functional collectives) paths.
The head-parallel GatedDeltaNet, and the whole hybrid model with CP attention,
are compared against single-device runs over the unsharded sequences, for both
token layouts (KV all-gather's head-tail and Ulysses' contiguous one).
"""

import collections
import contextlib
import copy
import types
import unittest
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
# pyformat: disable
# Import torch before spmd_types to avoid circular import with torch_tpu.
import torch
import spmd_types as spmd  # pylint: disable=unused-import
# pyformat: enable
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.distributed import cp_attention
from torchtitan.experiments.tpu.distributed.cp_attention import tpu_cp_inner_attention, tpu_cp_load_balancer
from torchtitan.experiments.tpu.qwen3_5 import config_registry
from torchtitan.experiments.tpu.qwen3_5 import cp_gated_delta
from torchtitan.experiments.tpu.qwen3_5.cp_gated_delta import CPTpuGatedDeltaNet, CPTpuInnerGatedDeltaNet, GatedDeltaNetContextParallelTransform
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaKernel, TpuGatedDeltaNet, TpuInnerGatedDeltaNet
from torchtitan.experiments.tpu.testing.cp_test_utils import DOCS as _DOCS
from torchtitan.experiments.tpu.testing.cp_test_utils import emulated_cp_group as _emulated_cp_group
from torchtitan.experiments.tpu.testing.cp_test_utils import packed_positions as _packed_positions
from torchtitan.experiments.tpu.testing.cp_test_utils import Ranks as _Ranks
from torchtitan.experiments.tpu.testing.cp_test_utils import reset_cp_state as _reset_cp_state
from torchtitan.experiments.tpu.testing.cp_test_utils import SEQ_LEN as _SEQ_LEN
from torchtitan.experiments.tpu.testing.cp_test_utils import shard as _shard
from torchtitan.experiments.tpu.testing.cp_test_utils import unshard as _unshard
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.qwen3_5.gdn import GatedDeltaNet

_STRATEGIES = ("all_gather", "ulysses")
_GDN_STRATEGIES = ("all_to_all", "allgather_reducescatter")


def _set_gated_delta(cfg, **fields):
  """Sets ``fields`` on every ``TpuGatedDeltaNet`` config of ``cfg``'s model."""
  for _, delta_net, _, _ in cfg.model_spec.model.traverse(
      TpuGatedDeltaNet.Config
  ):
    for name, value in fields.items():
      setattr(delta_net, name, value)
  return cfg


@contextlib.contextmanager
def _forbid_position_all_gathers():
  """Fails if any layer all-gathers positions (``cp_full_positions`` given)."""

  def fail(*args, **kwargs):
    raise AssertionError("positions were all-gathered")

  with mock.patch.object(
      cp_gated_delta, "cp_all_gather", fail
  ), mock.patch.object(cp_attention, "cp_all_gather", fail), mock.patch.object(
      cp_attention, "_full_sequence_segment_ids", fail
  ), mock.patch.object(
      cp_attention.KVAllGatherCPInnerAttention,
      "_full_sequence_segment_ids_all_gather",
      fail,
  ):
    yield


@contextlib.contextmanager
def _patched_cp(ranks: _Ranks, traced: bool):
  """``_emulated_cp_group`` plus the CP group lookup of ``cp_gated_delta``."""
  with _emulated_cp_group(ranks, traced), mock.patch.object(
      cp_gated_delta, "spmd_mesh_group", lambda axis: ranks.group()
  ):
    yield


class CPGatedDeltaNetParityTest(parameterized.TestCase):
  """Head-parallel GatedDeltaNet on emulated CP groups vs a single device."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    _reset_cp_state(self)

  def _inputs(self, n_key_heads, n_value_heads, head_dim, tokens):
    key_dim, value_dim = n_key_heads * head_dim, n_value_heads * head_dim
    acts = [
        torch.randn(tokens, key_dim),  # query
        torch.randn(tokens, key_dim),  # key
        torch.randn(tokens, value_dim),  # value
        torch.randn(tokens, n_value_heads),  # a
        torch.randn(tokens, n_value_heads),  # b
    ]
    params = [
        0.5 * torch.randn(key_dim, 1, 4),  # conv_q
        0.5 * torch.randn(key_dim, 1, 4),  # conv_k
        0.5 * torch.randn(value_dim, 1, 4),  # conv_v
        0.5 * torch.randn(n_value_heads),  # A_log
        0.5 * torch.randn(n_value_heads),  # dt_bias
    ]
    grad = torch.randn(tokens, n_value_heads, head_dim)
    return acts, params, grad

  @parameterized.product(
      strategy=_STRATEGIES,
      cp=(2, 4),
      traced=(False, True),
      documents=(False, True),
  )
  def test_matches_single_device(self, strategy, cp, traced, documents):
    """Output, input and per-head parameter grads match one device (B=2)."""
    n_key_heads, n_value_heads, head_dim = 4, 8, 8
    tokens = _SEQ_LEN * len(_DOCS)
    acts, params, grad = self._inputs(
        n_key_heads, n_value_heads, head_dim, tokens
    )
    positions = _packed_positions(_DOCS) if documents else None
    kernel = TpuGatedDeltaKernel.Config(chunk_size=8)
    load_balancer = tpu_cp_load_balancer(strategy)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32)
    ranks = _Ranks(cp)

    def run_rank(rank):
      local = [
          _shard(t, strategy, cp, rank, _SEQ_LEN).clone().requires_grad_(True)
          for t in acts
      ]
      weights = [p.clone().requires_grad_(True) for p in params]
      module = CPTpuInnerGatedDeltaNet(
          CPTpuInnerGatedDeltaNet.Config(
              kernel=kernel, load_balancer=load_balancer
          )
      )
      out = module(
          *local,
          *weights,
          cu_seqlens,
          key_head_dim=head_dim,
          value_head_dim=head_dim,
          positions=(
              None
              if positions is None
              else _shard(positions, strategy, cp, rank, _SEQ_LEN)
          ),
      )
      out.backward(_shard(grad, strategy, cp, rank, _SEQ_LEN))
      return out.detach(), [t.grad for t in local], [w.grad for w in weights]

    with set_unflatten_seq_len(_SEQ_LEN), _patched_cp(ranks, traced):
      results = ranks.run(run_rank)

    # Single-device reference over the unsharded sequences.
    ref_acts = [t.clone().requires_grad_(True) for t in acts]
    ref_params = [p.clone().requires_grad_(True) for p in params]
    ref = TpuInnerGatedDeltaNet(TpuInnerGatedDeltaNet.Config(kernel=kernel))
    with set_unflatten_seq_len(_SEQ_LEN):
      want = ref(
          *ref_acts,
          *ref_params,
          cu_seqlens,
          key_head_dim=head_dim,
          value_head_dim=head_dim,
          positions=positions,
      )
    want.backward(grad)

    got = _unshard([r[0] for r in results], strategy, cp, _SEQ_LEN)
    torch.testing.assert_close(got, want.detach(), rtol=2e-5, atol=2e-5)
    for i, name in enumerate(("dq", "dk", "dv", "da", "db")):
      got_grad = _unshard([r[1][i] for r in results], strategy, cp, _SEQ_LEN)
      torch.testing.assert_close(
          got_grad, ref_acts[i].grad, rtol=2e-5, atol=2e-5, msg=name
      )
    for i, name in enumerate(
        ("dconv_q", "dconv_k", "dconv_v", "dA_log", "ddt_bias")
    ):
      # Each rank's gradient covers its heads; FSDP sums them over the mesh.
      got_grad = sum(r[2][i] for r in results)
      torch.testing.assert_close(
          got_grad, ref_params[i].grad, rtol=2e-5, atol=2e-5, msg=name
      )

  def test_requires_cp_group(self):
    """Without an active CP axis it raises instead of using token shards."""
    acts, params, _ = self._inputs(2, 4, 8, 32)
    kernel = TpuGatedDeltaKernel.Config(chunk_size=8)
    args = (*acts, *params, torch.tensor([0, 32], dtype=torch.int32))
    kwargs = dict(key_head_dim=8, value_head_dim=8)
    module = CPTpuInnerGatedDeltaNet(
        CPTpuInnerGatedDeltaNet.Config(kernel=kernel, load_balancer="headtail")
    )
    with set_unflatten_seq_len(16):
      with self.assertRaisesRegex(RuntimeError, "active CP mesh axis"):
        module(*args, **kwargs)

  @parameterized.parameters(2, 4)
  def test_headtail_matches_upstream_load_balancer(self, cp):
    """Inverts torch's head-tail balancer on one sequence (no shim, as verl)."""
    tokens = 64
    perm = _HeadTailLoadBalancer(tokens, cp, torch.device("cpu"))
    shards = torch.arange(tokens)[perm._generate_indices().reshape(-1)].chunk(
        cp
    )
    # The GatedDeltaNet all-to-all concatenates the shards in rank order.
    got = cp_gated_delta._headtail_to_sequence_order(torch.cat(shards), cp)
    torch.testing.assert_close(got, torch.arange(tokens))
    torch.testing.assert_close(
        cp_gated_delta._sequence_to_headtail_order(got, cp), torch.cat(shards)
    )

  def test_rejects_heads_not_divisible_by_cp(self):
    """2 key heads cannot be split over 4 ranks."""
    acts, params, _ = self._inputs(2, 4, 8, 64)
    ranks = _Ranks(4)

    def run_rank(rank):
      module = CPTpuInnerGatedDeltaNet(
          CPTpuInnerGatedDeltaNet.Config(kernel=TpuGatedDeltaKernel.Config())
      )
      local = [_shard(t, "ulysses", 4, rank, 16) for t in acts]
      return module(
          *local,
          *params,
          torch.tensor([0, 64], dtype=torch.int32),
          key_head_dim=8,
          value_head_dim=8,
      )

    with set_unflatten_seq_len(16), _patched_cp(ranks, traced=False):
      with self.assertRaisesRegex(ValueError, "n_key_heads"):
        ranks.run(run_rank)


class CompiledCPGatedDeltaNetTest(parameterized.TestCase):
  """The head-parallel GatedDeltaNet under real Dynamo (``fullgraph=True``).

  Runs rank 0 of a fake 2-rank process group, so only tracing matters: the
  per-layer ``torch.compile(fullgraph=True)`` of the TPU trainer must see no
  graph break.
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
    _reset_cp_state(self)
    self.enter_context(
        mock.patch.object(
            cp_gated_delta, "spmd_mesh_group", lambda axis: dist.group.WORLD
        )
    )

  def _compile_and_run(self, load_balancer, init_rank_buffer=False):
    """Compiles one CP GatedDeltaNet call on rank 0; returns the graphs."""
    module, fn, local = self._module_and_call(load_balancer, init_rank_buffer)
    graphs = []

    def count(gm, _):
      graphs.append(gm)
      return gm.forward

    with set_unflatten_seq_len(_SEQ_LEN):
      torch.compile(fn, backend=count, fullgraph=True)(*local).sum().backward()
    del module
    return graphs

  def _module_and_call(self, load_balancer, init_rank_buffer):
    """A rank-0 CP GatedDeltaNet, a call of it and its token-sharded inputs."""
    strategy = "all_gather" if load_balancer == "headtail" else "ulysses"
    n_key_heads, n_value_heads, head_dim = 2, 4, 8
    tokens = _SEQ_LEN * len(_DOCS)
    acts = [
        torch.randn(tokens, n_key_heads * head_dim),
        torch.randn(tokens, n_key_heads * head_dim),
        torch.randn(tokens, n_value_heads * head_dim),
        torch.randn(tokens, n_value_heads),
        torch.randn(tokens, n_value_heads),
    ]
    local = [
        _shard(t, strategy, 2, 0, _SEQ_LEN).requires_grad_(True) for t in acts
    ]
    params = [
        torch.randn(n_key_heads * head_dim, 1, 4),
        torch.randn(n_key_heads * head_dim, 1, 4),
        torch.randn(n_value_heads * head_dim, 1, 4),
        torch.randn(n_value_heads),
        torch.randn(n_value_heads),
    ]
    positions = _shard(_packed_positions(_DOCS), strategy, 2, 0, _SEQ_LEN)
    module = CPTpuInnerGatedDeltaNet(
        CPTpuInnerGatedDeltaNet.Config(
            kernel=TpuGatedDeltaKernel.Config(chunk_size=8),
            load_balancer=load_balancer,
        )
    )
    if init_rank_buffer:
      # As in the trainers: parallelize registers the buffer with the CP mesh,
      # then init_weights fills it with buffer_device=None, outside any SPMD
      # context.
      cp_attention.register_cp_rank_buffer(module, _cp_parallel_dims(rank=0))
      cp_attention.init_cp_rank_buffer(module, None)

    def fn(*local_acts):
      return module(
          *local_acts,
          *params,
          torch.tensor([0, tokens], dtype=torch.int32),
          key_head_dim=head_dim,
          value_head_dim=head_dim,
          positions=positions,
      )

    return module, fn, local

  @staticmethod
  def _has_rank_input(gm) -> bool:
    """Whether a ``(1,)`` int32 graph input (the cached rank tensor) exists."""
    for node in gm.graph.nodes:
      value = node.meta.get("example_value")
      if (
          node.op == "placeholder"
          and isinstance(value, torch.Tensor)
          and value.dtype == torch.int32
          and tuple(value.shape) == (1,)
      ):
        return True
    return False

  @parameterized.parameters("headtail", None)
  def test_compiles_to_one_graph(self, load_balancer):
    """One graph for every rank: the cached rank tensor is a graph input."""
    # cp_shard caches the rank tensor before the compiled forward runs.
    cp_attention._cache_cp_rank_for_mesh(
        types.SimpleNamespace(device_type="cpu", get_local_rank=lambda: 0)
    )
    graphs = self._compile_and_run(load_balancer)
    self.assertLen(graphs, 1)
    self.assertTrue(self._has_rank_input(graphs[0]), graphs[0].code)

  def test_uncached_rank_still_compiles(self):
    """Without cp_shard's cache the rank becomes a graph constant."""
    graphs = self._compile_and_run("headtail")
    self.assertLen(graphs, 1)
    self.assertFalse(self._has_rank_input(graphs[0]), graphs[0].code)

  def test_rank_buffer_is_a_graph_input(self):
    """The filled rank buffer is a graph input, with no cp_shard cache."""
    graphs = self._compile_and_run("headtail", init_rank_buffer=True)
    self.assertLen(graphs, 1)
    self.assertTrue(self._has_rank_input(graphs[0]), graphs[0].code)

  @parameterized.parameters(False, True)
  def test_graph_trainer_trace_takes_rank_as_input(self, init_rank_buffer):
    """Under the graph trainer's make_fx trace only the buffer is an input.

    A cached rank tensor captured from Python becomes a constant of the
    traced graph (one program per rank); the rank buffer is a module input.
    """
    from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer  # pylint: disable=g-import-not-at-top

    cp_attention._cache_cp_rank_for_mesh(
        types.SimpleNamespace(device_type="cpu", get_local_rank=lambda: 0)
    )
    module, fn, local = self._module_and_call("headtail", init_rank_buffer)
    with set_unflatten_seq_len(_SEQ_LEN), mock.patch.object(
        cp_attention, "_tracing", lambda: True
    ):
      traced = minimal_fx_tracer(lambda *x: fn(*x).sum(), module=module)(*local)
    rank_constants = [
        n
        for n in traced.gm.graph.nodes
        if n.op == "get_attr"
        and isinstance(getattr(traced.gm, n.target, None), torch.Tensor)
        and getattr(traced.gm, n.target).dtype == torch.int32
        and tuple(getattr(traced.gm, n.target).shape) == (1,)
    ]
    if init_rank_buffer:
      self.assertIn("_cp_rank", traced.state_fqns)
      self.assertEmpty(rank_constants, traced.gm.code)
    else:
      self.assertNotEmpty(rank_constants, traced.gm.code)


def _cp_parallel_dims(rank: int, mesh: bool = True):
  """``parallel_dims`` with CP on, and a CP mesh at ``rank`` if ``mesh``."""
  cp_mesh = types.SimpleNamespace(get_local_rank=lambda: rank)
  dims = types.SimpleNamespace(cp_enabled=True)
  if mesh:
    dims.get_optional_mesh = lambda name: cp_mesh if name == "cp" else None
  return dims


class CPRankBufferTest(absltest.TestCase):
  """``register_cp_rank_buffer`` / ``init_cp_rank_buffer`` as the trainers call them."""

  def _module(self, parallel_dims):
    module = torch.nn.Module()
    cp_attention.register_cp_rank_buffer(module, parallel_dims)
    return module

  def test_fills_from_saved_mesh_without_spmd_context(self):
    module = self._module(_cp_parallel_dims(rank=1))
    self.assertIsNone(
        cp_attention.spmd_mesh_group(cp_attention.MeshAxisName.CP)
    )
    module.to_empty(device="cpu")  # Uninitialized, as in the trainers.
    cp_attention.init_cp_rank_buffer(module, None)
    self.assertTrue(module._cp_rank_ready)
    torch.testing.assert_close(
        module._cp_rank, torch.tensor([1], dtype=torch.int32)
    )
    self.assertIs(
        cp_attention.module_cp_rank(module, "cpu", None), module._cp_rank
    )

  def test_meta_buffer_stays_unfilled(self):
    with torch.device("meta"):
      module = self._module(_cp_parallel_dims(rank=1))
    cp_attention.init_cp_rank_buffer(module, None)
    self.assertFalse(module._cp_rank_ready)

  def test_falls_back_to_the_spmd_mesh_group(self):
    module = self._module(_cp_parallel_dims(rank=0, mesh=False))
    cp_attention.init_cp_rank_buffer(module, "cpu")
    self.assertFalse(module._cp_rank_ready)
    group = types.SimpleNamespace(rank=lambda: 1)
    with mock.patch.object(
        cp_attention, "spmd_mesh_group", lambda axis: group
    ):
      cp_attention.init_cp_rank_buffer(module, "cpu")
    self.assertTrue(module._cp_rank_ready)
    self.assertEqual(module._cp_rank.item(), 1)

  def test_no_buffer_without_cp(self):
    module = torch.nn.Module()
    cp_attention.register_cp_rank_buffer(
        module, types.SimpleNamespace(cp_enabled=False)
    )
    self.assertNotIn("_cp_rank", module._buffers)
    cp_attention.init_cp_rank_buffer(module, "cpu")
    self.assertFalse(getattr(module, "_cp_rank_ready", False))


def _cp_shard_per_rank(owner, batch, load_balancer, cp):
  """Each rank's output of ``owner.cp_shard`` on a mocked ``cp``-rank mesh.

  ``prepare_context_parallel_input`` (the torch sharding at the end of
  ``cp_shard``) is replaced by taking the rank's contiguous chunk of each
  input: both strategies' ``cp_shard`` ask it for no further reordering
  (Ulysses always; KV all-gather once it has reordered each folded sequence).
  """
  shards = []
  for rank in range(cp):
    mesh = mock.Mock(device_type="cpu")
    mesh.size.return_value = cp
    mesh.get_local_rank.return_value = rank

    def take_chunk(input_dict, shardings, mesh, balancer, ptrr_key, rank=rank):
      del shardings, mesh, ptrr_key
      assert balancer is None, balancer
      for key in ("input", "labels", "positions"):
        input_dict[key] = input_dict[key].chunk(cp)[rank]
      return input_dict

    with mock.patch.object(
        cp_attention, "prepare_context_parallel_input", side_effect=take_chunk
    ):
      shards.append(
          owner.cp_shard(dict(batch), None, mesh, load_balancer, None)
      )
  return shards


def _small_vocab(cfg, vocab_size: int = 512):
  """Shrinks the debug model's 248K vocabulary for fast CPU tests."""
  model = cfg.model_spec.model
  model.vocab_size = vocab_size
  model.tok_embeddings.num_embeddings = vocab_size
  model.lm_head.out_features = vocab_size
  return cfg


def _agrs_delta_net_config(load_balancer, n_key_heads, n_value_heads):
  """The debug model's ``allgather_reducescatter`` layer, other head counts."""
  model_cfg = GatedDeltaNetContextParallelTransform(
      load_balancer=load_balancer, cp_strategy="allgather_reducescatter"
  ).transform(config_registry.qwen35_debugmodel().model_spec.model)
  _, cfg, _, _ = next(iter(model_cfg.traverse(CPTpuGatedDeltaNet.Config)))
  key_dim = n_key_heads * cfg.key_head_dim
  value_dim = n_value_heads * cfg.value_head_dim
  for proj, out in (
      (cfg.in_proj_q, key_dim),
      (cfg.in_proj_k, key_dim),
      (cfg.in_proj_v, value_dim),
      (cfg.in_proj_z, value_dim),
      (cfg.in_proj_a, n_value_heads),
      (cfg.in_proj_b, n_value_heads),
  ):
    proj.out_features = out
  for conv, channels in (
      (cfg.conv_q, key_dim),
      (cfg.conv_k, key_dim),
      (cfg.conv_v, value_dim),
  ):
    conv.in_channels = conv.out_channels = conv.groups = channels
  cfg.out_proj.in_features = value_dim
  cfg.inner_gated_delta_net.kernel = TpuGatedDeltaKernel.Config(chunk_size=8)
  return cfg


class AllGatherReduceScatterParityTest(parameterized.TestCase):
  """``allgather_reducescatter`` layer on emulated CP groups vs one device."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    _reset_cp_state(self)

  @parameterized.product(
      strategy=_STRATEGIES,
      cp=(2, 4),
      traced=(False, True),
      full_positions=(False, True),
  )
  def test_matches_single_device(self, strategy, cp, traced, full_positions):
    """Output, input grad and summed parameter grads match (4 / 8 heads)."""
    cfg = _agrs_delta_net_config(
        tpu_cp_load_balancer(strategy), n_key_heads=4, n_value_heads=8
    )
    with torch.device("cpu"):
      module = cfg.build()
    for param in module.parameters():
      torch.nn.init.normal_(param, std=0.1)
    ref = copy.deepcopy(module)
    # Without a CP group the layer runs as TpuGatedDeltaNet.
    ref.inner_gated_delta_net = TpuInnerGatedDeltaNet.Config(
        kernel=cfg.inner_gated_delta_net.kernel
    ).build()

    tokens = _SEQ_LEN * len(_DOCS)
    x = torch.randn(tokens, module.in_proj_q.in_features)
    grad = torch.randn_like(x)
    positions = _packed_positions(_DOCS)
    ranks = _Ranks(cp)

    def run_rank(rank):
      layer = copy.deepcopy(module)
      local_x = _shard(x, strategy, cp, rank, _SEQ_LEN).requires_grad_(True)
      local_positions = _shard(positions, strategy, cp, rank, _SEQ_LEN)
      out = layer(
          local_x,
          None,
          local_positions,
          full_positions=positions if full_positions else None,
      )
      out.backward(_shard(grad, strategy, cp, rank, _SEQ_LEN))
      return (
          out.detach(),
          local_x.grad,
          {n: p.grad for n, p in layer.named_parameters()},
      )

    with set_unflatten_seq_len(_SEQ_LEN), _patched_cp(ranks, traced):
      results = ranks.run(run_rank)

    ref_x = x.clone().requires_grad_(True)
    with set_unflatten_seq_len(_SEQ_LEN):
      want = ref(ref_x, None, positions)
    want.backward(grad)

    got = _unshard([r[0] for r in results], strategy, cp, _SEQ_LEN)
    torch.testing.assert_close(got, want.detach(), rtol=2e-5, atol=2e-5)
    got_dx = _unshard([r[1] for r in results], strategy, cp, _SEQ_LEN)
    torch.testing.assert_close(got_dx, ref_x.grad, rtol=2e-5, atol=2e-5)
    for name, param in ref.named_parameters():
      if name.startswith("inner_gated_delta_net."):
        continue
      torch.testing.assert_close(
          sum(r[2][name] for r in results),
          param.grad,
          rtol=2e-5,
          atol=2e-5,
          msg=name,
      )


class CPQwen35ModelParityTest(parameterized.TestCase):
  """The CP debug model (attention + GatedDeltaNet) vs a single device."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    _reset_cp_state(self)

  @parameterized.product(
      strategy=_STRATEGIES,
      traced=(False, True),
      gated_delta_strategy=_GDN_STRATEGIES,
      full_positions=(False, True),
  )
  def test_debug_model_matches_single_device(
      self, strategy, traced, gated_delta_strategy, full_positions
  ):
    """Loss and every parameter gradient match (cp=2, B=2, packed documents).

    With ``full_positions``, the model gets ``cp_full_positions`` and no layer
    all-gathers positions.
    """
    self._check_debug_model(
        strategy, traced, gated_delta_strategy, full_positions
    )

  @parameterized.product(
      strategy=_STRATEGIES, gated_delta_strategy=_GDN_STRATEGIES
  )
  def test_debug_model_without_document_resets_matches_single_device(
      self, strategy, gated_delta_strategy
  ):
    """Document resets disabled (``reset_at_documents=False``) in both."""
    self._check_debug_model(
        strategy,
        traced=False,
        gated_delta_strategy=gated_delta_strategy,
        full_positions=True,
        document_resets=False,
    )

  @parameterized.parameters("all_to_all", "allgather_reducescatter")
  def test_debug_model_with_out_proj_bias_matches_single_device(
      self, gated_delta_strategy
  ):
    """Under both GDN CP strategies the ``out_proj`` bias is added once.

    With ``allgather_reducescatter`` it is added after the reduce-scatter.
    """
    self._check_debug_model(
        "ulysses",
        traced=False,
        gated_delta_strategy=gated_delta_strategy,
        full_positions=True,
        out_proj_bias=True,
    )

  def _check_debug_model(
      self,
      strategy,
      traced,
      gated_delta_strategy,
      full_positions,
      document_resets=True,
      out_proj_bias=False,
  ):
    cp = 2
    recipe = (
        config_registry.qwen35_debugmodel_allgather_cp2
        if strategy == "all_gather"
        else config_registry.qwen35_debugmodel_ulysses_cp2
    )
    cp_cfg = _small_vocab(recipe())
    if gated_delta_strategy != "all_to_all":
      cp_cfg = apply_transforms(
          cp_cfg,
          [
              GatedDeltaNetContextParallelTransform(
                  load_balancer=tpu_cp_load_balancer(strategy),
                  cp_strategy=gated_delta_strategy,
              )
          ],
      )
    ref_cfg = _small_vocab(config_registry.qwen35_debugmodel())
    if not document_resets:
      _set_gated_delta(ref_cfg, reset_at_documents=False)
      _set_gated_delta(cp_cfg, reset_at_documents=False)
    with torch.device("cpu"):
      ref_model = ref_cfg.model_spec.model.build()
      ref_model.init_states(buffer_device=torch.device("cpu"))
      cp_model = cp_cfg.model_spec.model.build()
    if out_proj_bias:
      for model in (ref_model, cp_model):
        for module in model.modules():
          if isinstance(module, GatedDeltaNet):
            module.out_proj.bias = torch.nn.Parameter(
                torch.empty(module.out_proj.out_features)
            )
      for module in ref_model.modules():
        if isinstance(module, GatedDeltaNet):
          torch.nn.init.normal_(module.out_proj.bias)
    cp_model.load_state_dict(ref_model.state_dict())

    tokens = _SEQ_LEN * len(_DOCS)
    inputs = torch.randint(0, 512, (tokens,))
    labels = torch.randint(0, 512, (tokens,))
    positions = _packed_positions(_DOCS)
    ranks = _Ranks(cp)
    # The real cp_shard of the recipe's CP attention.
    owner = cp_cfg.model_spec.model.first_full_attention_backend._owner
    with set_unflatten_seq_len(_SEQ_LEN):
      shards = _cp_shard_per_rank(
          owner,
          {"input": inputs, "labels": labels, "positions": positions},
          tpu_cp_load_balancer(strategy),
          cp,
      )
    for rank, shard in enumerate(shards):
      torch.testing.assert_close(
          shard["input"], _shard(inputs, strategy, cp, rank, _SEQ_LEN)
      )

    def run_rank(rank):
      model = copy.deepcopy(cp_model)
      extra = {}
      if full_positions:
        # What TpuQwen35Model.preprocess_inputs passes.
        extra = {
            "cp_full_positions": positions,
            "cp_full_segment_ids": (
                shards[rank]["positions"]._cp_full_segment_ids
            ),
        }
      # The clone drops what cp_shard attached to the positions (the
      # full-sequence segment ids), which compiled blocks do not see either.
      logits = model(
          shards[rank]["input"],
          positions=shards[rank]["positions"].clone(),
          **extra,
      )
      loss = F.cross_entropy(
          logits.float(), shards[rank]["labels"], reduction="sum"
      )
      loss.backward()
      return loss.detach(), {
          n: p.grad.clone() for n, p in model.named_parameters()
      }

    # Patched here, not per rank thread: concurrent mock.patch of the same
    # attribute from several threads would not restore it.
    forbid = (
        _forbid_position_all_gathers()
        if full_positions
        else contextlib.nullcontext()
    )
    with set_unflatten_seq_len(_SEQ_LEN), _patched_cp(ranks, traced), forbid:
      results = ranks.run(run_rank)

    with set_unflatten_seq_len(_SEQ_LEN):
      logits = ref_model(inputs, positions=positions)
    want_loss = F.cross_entropy(logits.float(), labels, reduction="sum")
    want_loss.backward()

    torch.testing.assert_close(
        sum(r[0] for r in results), want_loss.detach(), rtol=1e-5, atol=1e-4
    )
    for name, param in ref_model.named_parameters():
      torch.testing.assert_close(
          sum(r[1][name] for r in results),
          param.grad,
          rtol=1e-4,
          atol=1e-5,
          msg=name,
      )


class CompiledAllGatherReduceScatterGatedDeltaNetTest(parameterized.TestCase):
  """``allgather_reducescatter`` compiles to one graph, one wait per collective.

  Runs on a fake 2-rank process group (as ``test_cp_attention``'s
  ``CompiledCPCollectivesTest``), so only the graph structure matters.
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
    torch.manual_seed(0)
    torch._dynamo.reset()
    self.addCleanup(torch._dynamo.reset)
    cp_attention._CP_RANK_CACHE.clear()
    self.addCleanup(cp_attention._CP_RANK_CACHE.clear)
    world = lambda axis: dist.group.WORLD
    self.enter_context(
        mock.patch.object(cp_attention, "spmd_mesh_group", world)
    )
    self.enter_context(
        mock.patch.object(cp_gated_delta, "spmd_mesh_group", world)
    )

  @parameterized.parameters(None, "headtail")
  def test_one_graph_one_wait_per_collective(self, load_balancer):
    from functorch.compile import min_cut_rematerialization_partition  # pylint: disable=g-import-not-at-top
    from torch._dynamo.backends.common import aot_autograd  # pylint: disable=g-import-not-at-top

    model_cfg = GatedDeltaNetContextParallelTransform(
        load_balancer=load_balancer, cp_strategy="allgather_reducescatter"
    ).transform(config_registry.qwen35_debugmodel().model_spec.model)
    _, delta_cfg, _, _ = next(
        iter(model_cfg.traverse(CPTpuGatedDeltaNet.Config))
    )
    with torch.device("cpu"):
      module = delta_cfg.build()
    for p in module.parameters():
      torch.nn.init.normal_(p, std=0.02)
    tokens = 32
    x = torch.randn(
        tokens // 2, module.in_proj_q.in_features, requires_grad=True
    )
    full_positions = torch.arange(tokens) % 12
    # Cache the rank tensor eagerly, as cp_shard does before tracing.
    cp_attention.cp_rank_tensor(x.device, dist.group.WORLD)

    graphs = []

    def capture(gm, _):
      graphs.append(gm)
      return gm

    backend = aot_autograd(
        fw_compiler=capture,
        bw_compiler=capture,
        partition_fn=min_cut_rematerialization_partition,
    )
    with mock.patch.object(cp_attention, "_tracing", lambda: True):
      fn = lambda x: module(x, None, None, full_positions=full_positions)
      torch.compile(fn, backend=backend, fullgraph=True)(x).sum().backward()

    wait = torch.ops._c10d_functional.wait_tensor.default
    collectives = {
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
    }
    found = collections.Counter()
    for gm in graphs:
      waited = []
      n_collectives = 0
      for node in gm.graph.nodes:
        if node.target is wait:
          self.assertIn(node.args[0].target, collectives, gm.code)
          waited.append(node.args[0])
        elif node.target in collectives:
          n_collectives += 1
          found[node.target] += 1
      self.assertLen(set(waited), len(waited), gm.code)
      self.assertLen(waited, n_collectives, gm.code)
    # Forward all-gather + reduce-scatter and their backward duals only: no
    # all-to-all and no positions all-gather.
    self.assertEqual(found, collections.Counter({c: 2 for c in collectives}))
    code = "\n".join(gm.code for gm in graphs)
    self.assertNotIn("all_to_all", code)


class AllGatherReduceScatterChecksTest(absltest.TestCase):
  """``allgather_reducescatter`` refuses modules it would compute wrongly."""

  def setUp(self):
    super().setUp()
    # The checks run before the group is used.
    self.enter_context(
        mock.patch.object(
            cp_gated_delta, "spmd_mesh_group", lambda axis: object()
        )
    )
    model_cfg = GatedDeltaNetContextParallelTransform(
        load_balancer=None, cp_strategy="allgather_reducescatter"
    ).transform(config_registry.qwen35_debugmodel().model_spec.model)
    _, delta_cfg, _, _ = next(
        iter(model_cfg.traverse(CPTpuGatedDeltaNet.Config))
    )
    with torch.device("cpu"):
      self.module = delta_cfg.build()
    self.x = torch.randn(4, self.module.in_proj_q.in_features)

  def test_rejects_non_cp_inner_module(self):
    self.module.inner_gated_delta_net = TpuInnerGatedDeltaNet.Config(
        kernel=TpuGatedDeltaKernel.Config()
    ).build()
    with self.assertRaisesRegex(TypeError, "CPTpuInnerGatedDeltaNet"):
      self.module(self.x)

  def test_rejects_non_plain_out_proj(self):

    class ScaledLinear(torch.nn.Linear):  # e.g. a converter's linear

      def forward(self, x):
        return 2 * super().forward(x)

    out = self.module.out_proj
    self.module.out_proj = ScaledLinear(
        out.in_features, out.out_features, bias=False
    )
    with self.assertRaisesRegex(ValueError, "slices out_proj"):
      self.module(self.x)

  def test_rejects_non_plain_input_projection(self):

    class ScaledLinear(torch.nn.Linear):  # e.g. a converter's linear

      def forward(self, x):
        return 2 * super().forward(x)

    q = self.module.in_proj_q
    self.module.in_proj_q = ScaledLinear(
        q.in_features, q.out_features, bias=False
    )
    with self.assertRaisesRegex(ValueError, "slices in_proj_q"):
      self.module(self.x)


class CPQwen35ConfigTest(parameterized.TestCase):
  """Recipes, the GatedDeltaNet transform and its validation."""

  @parameterized.parameters(
      ("qwen35_debugmodel_allgather_cp2", "headtail"),
      ("qwen35_debugmodel_ulysses_cp2", None),
      ("qwen35_moe_debugmodel_allgather_cp2", "headtail"),
      ("qwen35_moe_debugmodel_ulysses_cp2", None),
      ("qwen35_debugmodel_ulysses_cp2_agrs", None),
      ("qwen35_moe_35b_split_tokamax_benchmark_allgather_cp2", "headtail"),
      ("qwen35_moe_35b_split_tokamax_benchmark_ulysses_cp2", None),
      ("qwen35_moe_35b_split_tokamax_benchmark_ulysses_cp2_agrs", None),
      ("qwen35_moe_397b_64k_ulysses_cp2", None),
      ("qwen35_moe_397b_64k_allgather_cp2", "headtail"),
  )
  def test_recipes_convert_every_layer(self, recipe, load_balancer):
    cfg = getattr(config_registry, recipe)()
    model = cfg.model_spec.model
    self.assertEqual(cfg.parallelism.context_parallel_degree, 2)
    self.assertEqual(
        cfg.parallelism.context_parallel_load_balancer, load_balancer
    )
    delta_nets = list(model.traverse(GatedDeltaNet.Config))
    self.assertNotEmpty(delta_nets)
    for _, delta_net, _, _ in delta_nets:
      inner = delta_net.inner_gated_delta_net
      self.assertIsInstance(inner, CPTpuInnerGatedDeltaNet.Config)
      self.assertEqual(inner.load_balancer, load_balancer)
    attentions = list(model.traverse(BaseAttention.Config))
    self.assertNotEmpty(attentions)
    mixin = (
        cp_attention.KVAllGatherCPInnerAttention
        if load_balancer == "headtail"
        else cp_attention.UlyssesCPInnerAttention
    )
    for _, attn, _, _ in attentions:
      self.assertTrue(issubclass(attn.inner_attention._owner, mixin))

  @parameterized.parameters(
      "qwen35_moe_397b_64k_ulysses_cp2", "qwen35_moe_397b_64k_allgather_cp2"
  )
  def test_397b_64k_recipes(self, recipe):
    cfg = getattr(config_registry, recipe)()
    self.assertEqual(cfg.training.max_context_length, 65536)
    self.assertEqual(cfg.training.num_tokens_per_microbatch_per_dp_rank, 65536)
    self.assertEqual(cfg.parallelism.expert_parallel_degree, 4)
    self.assertEqual(cfg.tpu_config.ep_token_chunks, 2)
    self.assertEqual(cfg.tpu_config.ep_capacity_factor, 1.5)
    self.assertFalse(cfg.tpu_config.ep_chunk_dw_fp32)
    self.assertFalse(cfg.splash_attention_kernel.sa_use_fused_bwd_kernel)
    self.assertEqual(cfg.splash_attention_kernel.sa_block_q, 2048)
    self.assertEqual(cfg.splash_attention_kernel.sa_block_kv_compute, 1024)
    graph_split = cfg.compile.graph_split
    self.assertEqual(graph_split.forward_layers_per_executable, 2)
    self.assertEqual(graph_split.backward_layers_per_executable, 2)
    cp_gated_delta.validate_gated_delta_context_parallel(
        cfg.model_spec.model, cfg.parallelism
    )

  def _cp_config(self, *, gated_delta_lb="unset", cp=2, strategy="ulysses"):
    cfg = config_registry.qwen35_debugmodel()
    cfg.parallelism.context_parallel_degree = cp
    cfg.parallelism.context_parallel_load_balancer = tpu_cp_load_balancer(
        strategy
    )
    inner = cfg.model_spec.model.first_full_attention_backend
    transforms = [
        ContextParallelTransform(
            inner_attention=tpu_cp_inner_attention(inner._owner, strategy)
        )
    ]
    if gated_delta_lb != "unset":
      transforms.append(
          GatedDeltaNetContextParallelTransform(load_balancer=gated_delta_lb)
      )
    return apply_transforms(cfg, transforms)

  def test_rejects_cp_without_gated_delta_transform(self):
    """CP attention alone would train the GatedDeltaNet on token shards."""
    with self.assertRaisesRegex(ValueError, "is not context-parallel"):
      self._cp_config()

  def test_rejects_mismatched_token_layout(self):
    with self.assertRaisesRegex(ValueError, "token layout"):
      self._cp_config(gated_delta_lb="headtail", strategy="ulysses")

  def test_rejects_heads_not_divisible_by_cp(self):
    # The debug model has 2 key heads.
    with self.assertRaisesRegex(ValueError, "n_key_heads"):
      self._cp_config(gated_delta_lb=None, cp=4)

  def test_rejects_tensor_parallelism(self):
    cfg = self._cp_config(gated_delta_lb=None)
    cfg.parallelism.tensor_parallel_degree = 2
    with self.assertRaisesRegex(NotImplementedError, "tensor parallelism"):
      cp_gated_delta.validate_gated_delta_context_parallel(
          cfg.model_spec.model, cfg.parallelism
      )

  def test_allows_expert_parallelism_with_simple_fsdp(self):
    cfg = self._cp_config(gated_delta_lb=None)
    cfg.parallelism.expert_parallel_degree = 2
    cfg.parallelism.use_simple_fsdp = True
    cp_gated_delta.validate_gated_delta_context_parallel(
        cfg.model_spec.model, cfg.parallelism
    )

  def test_rejects_expert_parallelism_without_simple_fsdp(self):
    cfg = self._cp_config(gated_delta_lb=None)
    cfg.parallelism.expert_parallel_degree = 2
    cfg.parallelism.use_simple_fsdp = False
    with self.assertRaisesRegex(NotImplementedError, "SimpleFSDP"):
      cp_gated_delta.validate_gated_delta_context_parallel(
          cfg.model_spec.model, cfg.parallelism
      )

  def test_rejects_vision_encoder(self):
    cfg = self._cp_config(gated_delta_lb=None)
    model = mock.Mock(wraps=cfg.model_spec.model)
    model.vision_encoder = object()
    model.traverse = cfg.model_spec.model.traverse
    with self.assertRaisesRegex(NotImplementedError, "text-only"):
      cp_gated_delta.validate_gated_delta_context_parallel(
          model, cfg.parallelism
      )

  def test_transform_is_idempotent_and_keeps_the_kernel(self):
    cfg = config_registry.qwen35_debugmodel()
    model = cfg.model_spec.model
    for _ in range(2):
      model = GatedDeltaNetContextParallelTransform(
          load_balancer="headtail"
      ).transform(model)
    for _, delta_net, _, _ in model.traverse(GatedDeltaNet.Config):
      inner = delta_net.inner_gated_delta_net
      self.assertIs(type(inner), CPTpuInnerGatedDeltaNet.Config)
      self.assertIsInstance(inner.kernel, TpuGatedDeltaKernel.Config)
      self.assertEqual(inner.load_balancer, "headtail")

  def test_revert_restores_single_device_gated_delta(self):
    """revert_tpu_context_parallel also reverts the GatedDeltaNet (cp == 1)."""
    cfg = config_registry.qwen35_debugmodel_allgather_cp2()
    model = cfg.model_spec.model
    cp_attention.revert_tpu_context_parallel(model)
    delta_nets = list(model.traverse(GatedDeltaNet.Config))
    self.assertNotEmpty(delta_nets)
    for _, delta_net, _, _ in delta_nets:
      inner = delta_net.inner_gated_delta_net
      self.assertIs(type(inner), TpuInnerGatedDeltaNet.Config)
      self.assertIsInstance(inner.kernel, TpuGatedDeltaKernel.Config)
    for _, attn, _, _ in model.traverse(BaseAttention.Config):
      self.assertFalse(
          issubclass(attn.inner_attention._owner, cp_attention.CPInnerAttention)
      )

  def test_revert_restores_allgather_reducescatter_gated_delta(self):
    """The ``allgather_reducescatter`` outer layer is reverted too."""
    cfg = apply_transforms(
        config_registry.qwen35_debugmodel_ulysses_cp2(),
        [
            GatedDeltaNetContextParallelTransform(
                load_balancer=None, cp_strategy="allgather_reducescatter"
            )
        ],
    )
    model = cfg.model_spec.model
    cp_attention.revert_tpu_context_parallel(model)
    delta_nets = list(model.traverse(GatedDeltaNet.Config))
    self.assertNotEmpty(delta_nets)
    for _, delta_net, _, _ in delta_nets:
      self.assertIs(type(delta_net), TpuGatedDeltaNet.Config)
      self.assertIs(
          type(delta_net.inner_gated_delta_net), TpuInnerGatedDeltaNet.Config
      )

  @parameterized.parameters(
      # (strategy, shim seq_len, tokens)
      ("all_gather", _SEQ_LEN, 2 * _SEQ_LEN),
      ("all_gather", None, 2 * _SEQ_LEN),
      ("ulysses", _SEQ_LEN, 2 * _SEQ_LEN),
  )
  def test_cp_shard_caches_rank_and_segment_ids(
      self, strategy, seq_len, tokens
  ):
    _reset_cp_state(self)
    owner = tpu_cp_inner_attention(
        cp_attention.ScaledDotProductInnerAttention, strategy
    )
    mesh = mock.Mock(device_type="cpu")
    mesh.size.return_value = 2
    mesh.get_local_rank.return_value = 0
    batch = {
        "input": torch.arange(tokens),
        "labels": torch.arange(tokens),
        "positions": torch.arange(tokens) % _SEQ_LEN,
    }
    ctx = (
        set_unflatten_seq_len(seq_len)
        if seq_len is not None
        else contextlib.nullcontext()
    )
    with ctx, mock.patch.object(
        cp_attention,
        "prepare_context_parallel_input",
        side_effect=lambda d, *args: d,
    ):
      # The input sharding TpuQwen35Model.preprocess_inputs passes.
      owner.cp_shard(
          batch,
          {**decoder_input_sharding(), **multimodal_input_sharding()},
          mesh,
          tpu_cp_load_balancer(strategy),
          None,
      )
    self.assertNotEmpty(cp_attention._CP_RANK_CACHE)
    # The whole sequence's segment_ids, for TpuQwen35Model.preprocess_inputs.
    self.assertEqual(batch["positions"]._cp_full_segment_ids.numel(), tokens)

  def test_transform_strategy_converts_gated_delta_nets(self):
    cfg = config_registry.qwen35_debugmodel()
    model = cfg.model_spec.model
    for _ in range(2):
      model = GatedDeltaNetContextParallelTransform(
          load_balancer=None,
          cp_strategy="allgather_reducescatter",
          cp_reduce_dtype="bfloat16",
      ).transform(model)
    delta_nets = list(model.traverse(GatedDeltaNet.Config))
    self.assertNotEmpty(delta_nets)
    for _, delta_net, _, _ in delta_nets:
      self.assertIs(type(delta_net), CPTpuGatedDeltaNet.Config)
      self.assertEqual(delta_net.cp_strategy, "allgather_reducescatter")
      self.assertEqual(delta_net.cp_reduce_dtype, "bfloat16")
      self.assertIs(
          type(delta_net.inner_gated_delta_net), CPTpuInnerGatedDeltaNet.Config
      )

  def test_transform_overrides_an_earlier_strategy(self):
    model = config_registry.qwen35_debugmodel().model_spec.model
    for strategy in ("allgather_reducescatter", "all_to_all"):
      model = GatedDeltaNetContextParallelTransform(
          load_balancer=None, cp_strategy=strategy
      ).transform(model)
    for _, delta_net, _, _ in model.traverse(GatedDeltaNet.Config):
      self.assertEqual(delta_net.cp_strategy, "all_to_all")

  def test_transform_default_strategy_keeps_gated_delta_nets(self):
    model = GatedDeltaNetContextParallelTransform(load_balancer=None).transform(
        config_registry.qwen35_debugmodel().model_spec.model
    )
    for _, delta_net, _, _ in model.traverse(GatedDeltaNet.Config):
      self.assertIs(type(delta_net), TpuGatedDeltaNet.Config)

  def test_transform_rejects_unknown_strategy(self):
    with self.assertRaisesRegex(ValueError, "strategy"):
      GatedDeltaNetContextParallelTransform(
          load_balancer=None, cp_strategy="ring"
      ).transform(config_registry.qwen35_debugmodel().model_spec.model)

  def test_tpu_model_cp_shard_uses_inner_attention(self):
    """TpuQwen35Model shards CP inputs through the CP attention's cp_shard."""
    cfg = _small_vocab(config_registry.qwen35_debugmodel_allgather_cp2())
    with torch.device("cpu"):
      model = cfg.model_spec.model.build()
    owner = cfg.model_spec.model.first_full_attention_backend._owner
    tokens = torch.arange(2 * _SEQ_LEN)
    batch = {"input": tokens, "labels": tokens, "positions": tokens % _SEQ_LEN}
    parallel_dims = mock.Mock(cp_enabled=True)
    rank0 = {k: v.chunk(2)[0] for k, v in batch.items()}
    full_seg = torch.zeros(1, 2 * _SEQ_LEN, dtype=torch.int32)
    rank0["positions"]._cp_full_segment_ids = full_seg
    with mock.patch.object(
        owner, "cp_shard", return_value=dict(rank0)
    ) as cp_shard, mock.patch(
        "torchtitan.experiments.tpu.qwen3_5.model.annotate_input_spmd_types",
        side_effect=lambda dims, b, s: b,
    ):
      inputs, labels, extra = model.preprocess_inputs(
          batch,
          parallel_dims=parallel_dims,
          parallelism=cfg.parallelism,
      )
    cp_shard.assert_called_once()
    self.assertEqual(
        cp_shard.call_args.args[3],
        cfg.parallelism.context_parallel_load_balancer,
    )
    torch.testing.assert_close(inputs, rank0["input"])
    torch.testing.assert_close(extra["positions"], rank0["positions"])
    # The unsharded positions, for the layers' document boundaries.
    torch.testing.assert_close(extra["cp_full_positions"], tokens % _SEQ_LEN)
    # cp_shard's segment_ids as an explicit input, for the attention layers.
    self.assertIs(extra["cp_full_segment_ids"], full_seg)

  @parameterized.named_parameters(
      ("without_positions", "positions", None, ValueError, "RoPE positions"),
      (
          "with_mrope_positions",
          "mrope_positions",
          torch.zeros(2 * _SEQ_LEN, 3, dtype=torch.long),
          NotImplementedError,
          "text-only",
      ),
  )
  def test_tpu_model_cp_rejects_unsupported_inputs(
      self, key, value, error, regex
  ):
    cfg = _small_vocab(config_registry.qwen35_debugmodel_allgather_cp2())
    with torch.device("cpu"):
      model = cfg.model_spec.model.build()
    tokens = torch.arange(2 * _SEQ_LEN)
    batch = {"input": tokens, "labels": tokens, "positions": tokens % _SEQ_LEN}
    batch[key] = value
    with self.assertRaisesRegex(error, regex):
      model.preprocess_inputs(
          batch,
          parallel_dims=mock.Mock(cp_enabled=True),
          parallelism=cfg.parallelism,
      )


if __name__ == "__main__":
  absltest.main()
