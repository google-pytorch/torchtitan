"""Tests for Qwen3.5 context parallelism on TPU (``qwen3_5/cp_gated_delta.py``).

Every rank of a CP group runs on its own CPU thread and the collectives are
emulated across the threads (the harness of ``tests/test_cp_attention.py``):
the eager (``spmd.redistribute``) and the traced (functional collectives) paths.
The head-parallel GatedDeltaNet, and the whole hybrid model with CP attention,
are compared against single-device runs over the unsharded sequences, for both
token layouts (KV all-gather's head-tail and Ulysses' contiguous one).
"""

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
from torchtitan.experiments.tpu.qwen3_5.cp_gated_delta import CPTpuInnerGatedDeltaNet, GatedDeltaNetContextParallelTransform
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaKernel, TpuInnerGatedDeltaNet
from torchtitan.experiments.tpu.tests.test_cp_attention import _DOCS, _emulated_cp_group, _packed_positions, _Ranks, _reset_cp_state, _SEQ_LEN, _shard, _unshard
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.decoder_sharding import decoder_input_sharding
from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding
from torchtitan.models.qwen3_5.gdn import GatedDeltaNet

_STRATEGIES = ("all_gather", "ulysses")


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
    cp_attention._record_cp_input_layout(1, tokens)
    # The GatedDeltaNet all-to-all concatenates the shards in rank order.
    got = cp_gated_delta._headtail_to_sequence_order(torch.cat(shards), cp)
    torch.testing.assert_close(got, torch.arange(tokens))
    torch.testing.assert_close(
        cp_gated_delta._sequence_to_headtail_order(got, cp), torch.cat(shards)
    )

  def test_rejects_layout_that_disagrees_with_cp_shard(self):
    """A different set_unflatten_seq_len at shard and forward time raises."""
    cp_attention._record_cp_input_layout(2, 32)
    with set_unflatten_seq_len(32):
      self.assertEqual(cp_gated_delta._rows(64), (2, 32))
    with self.assertRaisesRegex(RuntimeError, "set_unflatten_seq_len"):
      cp_gated_delta._rows(64)
    with set_unflatten_seq_len(16):
      with self.assertRaisesRegex(RuntimeError, "set_unflatten_seq_len"):
        cp_gated_delta._rows(64)
    # Another token count (e.g. the next micro-batch) is not checked.
    self.assertEqual(cp_gated_delta._rows(128), (1, 128))

  def test_ulysses_rejects_row_sharded_input(self):
    cp_attention._record_cp_input_layout(1, 64)
    cp_gated_delta._check_contiguous_layout(64)
    cp_attention._record_cp_input_layout(2, 32)
    with self.assertRaisesRegex(RuntimeError, "folded"):
      cp_gated_delta._check_contiguous_layout(64)

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

  def _compile_and_run(self, load_balancer):
    """Compiles one CP GatedDeltaNet call on rank 0; returns the graph count."""
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
    graphs = []

    def count(gm, _):
      graphs.append(gm)
      return gm.forward

    def fn(*local_acts):
      return module(
          *local_acts,
          *params,
          torch.tensor([0, tokens], dtype=torch.int32),
          key_head_dim=head_dim,
          value_head_dim=head_dim,
          positions=positions,
      )

    with set_unflatten_seq_len(_SEQ_LEN):
      torch.compile(fn, backend=count, fullgraph=True)(*local).sum().backward()
    return len(graphs)

  @parameterized.parameters("headtail", None)
  def test_compiles_to_one_graph(self, load_balancer):
    # cp_shard caches the rank tensor before the compiled forward runs.
    cp_attention._cache_cp_rank_for_mesh(
        types.SimpleNamespace(device_type="cpu", get_local_rank=lambda: 0)
    )
    self.assertEqual(self._compile_and_run(load_balancer), 1)

  def test_uncached_rank_raises_while_compiling(self):
    # Dynamo re-raises it as a RuntimeError subclass.
    with self.assertRaisesRegex(RuntimeError, "not cached while compiling"):
      self._compile_and_run("headtail")


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


class CPQwen35ModelParityTest(parameterized.TestCase):
  """The CP debug model (attention + GatedDeltaNet) vs a single device."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)
    _reset_cp_state(self)

  @parameterized.product(strategy=_STRATEGIES, traced=(False, True))
  def test_debug_model_matches_single_device(self, strategy, traced):
    """Loss and every parameter gradient match (cp=2, B=2, packed documents)."""
    cp = 2
    recipe = (
        config_registry.qwen35_debugmodel_allgather_cp2
        if strategy == "all_gather"
        else config_registry.qwen35_debugmodel_ulysses_cp2
    )
    cp_cfg = _small_vocab(recipe())
    ref_cfg = _small_vocab(config_registry.qwen35_debugmodel())
    with torch.device("cpu"):
      ref_model = ref_cfg.model_spec.model.build()
      ref_model.init_states(buffer_device=torch.device("cpu"))
      cp_model = cp_cfg.model_spec.model.build()
    cp_model.load_state_dict(ref_model.state_dict())

    tokens = _SEQ_LEN * len(_DOCS)
    inputs = torch.randint(0, 512, (tokens,))
    labels = torch.randint(0, 512, (tokens,))
    positions = _packed_positions(_DOCS)
    ranks = _Ranks(cp)
    # The real cp_shard of the recipe's CP attention (it records the token
    # layout that the GatedDeltaNet checks during the forward below).
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
    self.assertEqual(
        cp_attention.cp_input_layout(tokens)[0],
        1 if strategy == "ulysses" else len(_DOCS),
    )

    def run_rank(rank):
      model = copy.deepcopy(cp_model)
      logits = model(shards[rank]["input"], positions=shards[rank]["positions"])
      loss = F.cross_entropy(
          logits.float(), shards[rank]["labels"], reduction="sum"
      )
      loss.backward()
      return loss.detach(), {
          n: p.grad.clone() for n, p in model.named_parameters()
      }

    with set_unflatten_seq_len(_SEQ_LEN), _patched_cp(ranks, traced):
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


class CPQwen35ConfigTest(parameterized.TestCase):
  """Recipes, the GatedDeltaNet transform and its validation."""

  @parameterized.parameters(
      ("qwen35_debugmodel_allgather_cp2", "headtail"),
      ("qwen35_debugmodel_ulysses_cp2", None),
      ("qwen35_moe_debugmodel_allgather_cp2", "headtail"),
      ("qwen35_moe_debugmodel_ulysses_cp2", None),
      ("qwen35_moe_35b_allgather_cp2", "headtail"),
      ("qwen35_moe_35b_ulysses_cp2", None),
      ("qwen35_moe_35b_split_tokamax_benchmark_allgather_cp2", "headtail"),
      ("qwen35_moe_35b_split_tokamax_benchmark_ulysses_cp2", None),
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

  @parameterized.parameters(
      # (strategy, shim seq_len, tokens, expected (rows, row_len))
      ("all_gather", _SEQ_LEN, 2 * _SEQ_LEN, (2, _SEQ_LEN)),
      ("all_gather", None, 2 * _SEQ_LEN, (1, 2 * _SEQ_LEN)),
      ("ulysses", _SEQ_LEN, 2 * _SEQ_LEN, (1, 2 * _SEQ_LEN)),
  )
  def test_cp_shard_records_the_layout(self, strategy, seq_len, tokens, want):
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
    self.assertEqual(cp_attention.cp_input_layout(tokens), want)
    self.assertIsNone(cp_attention.cp_input_layout(tokens + 1))

  def test_tpu_model_cp_shard_uses_inner_attention(self):
    """TpuQwen35Model shards CP inputs through the CP attention's cp_shard."""
    cfg = _small_vocab(config_registry.qwen35_debugmodel_allgather_cp2())
    with torch.device("cpu"):
      model = cfg.model_spec.model.build()
    owner = cfg.model_spec.model.first_full_attention_backend._owner
    tokens = torch.arange(2 * _SEQ_LEN)
    batch = {"input": tokens, "labels": tokens, "positions": tokens % _SEQ_LEN}
    parallel_dims = mock.Mock(cp_enabled=True)
    with mock.patch.object(
        owner, "cp_shard", return_value=dict(batch)
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
    torch.testing.assert_close(inputs, tokens)
    self.assertIn("positions", extra)

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
