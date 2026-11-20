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
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.experiments.tpu import cp_attention
from torchtitan.experiments.tpu.cp_attention import tpu_cp_inner_attention, tpu_cp_load_balancer
from torchtitan.experiments.tpu.qwen3_5 import config_registry
from torchtitan.experiments.tpu.qwen3_5 import cp_gated_delta
from torchtitan.experiments.tpu.qwen3_5.cp_gated_delta import CPTpuGatedDeltaNet, CPTpuInnerGatedDeltaNet, GatedDeltaNetContextParallelTransform
from torchtitan.experiments.tpu.qwen3_5.gated_delta import TpuGatedDeltaKernel, TpuGatedDeltaNet, TpuInnerGatedDeltaNet
from torchtitan.experiments.tpu.tests.test_cp_attention import _DOCS, _emulated_cp_group, _packed_positions, _Ranks, _SEQ_LEN, _shard, _unshard
from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common.attention import BaseAttention
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
  ), mock.patch.object(
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
    cp_attention._CP_RANK_CACHE.clear()
    self.addCleanup(cp_attention._CP_RANK_CACHE.clear)

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
    """Output and input / per-head parameter gradients match a single device (B=2)."""
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

  def test_without_cp_group_is_single_device(self):
    """Without an active CP axis the module is TpuInnerGatedDeltaNet."""
    acts, params, _ = self._inputs(2, 4, 8, 32)
    kernel = TpuGatedDeltaKernel.Config(chunk_size=8)
    args = (*acts, *params, torch.tensor([0, 32], dtype=torch.int32))
    kwargs = dict(key_head_dim=8, value_head_dim=8)
    with set_unflatten_seq_len(16):
      got = CPTpuInnerGatedDeltaNet(
          CPTpuInnerGatedDeltaNet.Config(
              kernel=kernel, load_balancer="headtail"
          )
      )(*args, **kwargs)
      want = TpuInnerGatedDeltaNet(TpuInnerGatedDeltaNet.Config(kernel=kernel))(
          *args, **kwargs
      )
    torch.testing.assert_close(got, want)

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
    cp_attention._CP_RANK_CACHE.clear()
    self.addCleanup(cp_attention._CP_RANK_CACHE.clear)

  @parameterized.product(
      strategy=_STRATEGIES,
      traced=(False, True),
      gated_delta_strategy=_GDN_STRATEGIES,
      full_positions=(False, True),
      document_resets=(None, False),
  )
  def test_debug_model_matches_single_device(
      self,
      strategy,
      traced,
      gated_delta_strategy,
      full_positions,
      document_resets,
  ):
    """Loss and every parameter gradient match (cp=2, B=2, packed documents).

    With ``full_positions``, the model gets ``cp_full_positions`` and no layer
    all-gathers positions. ``document_resets=False`` passes
    ``gated_delta_document_resets=False``; the reference disables the
    GatedDeltaNet document resets in its config instead.
    """
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
                  strategy=gated_delta_strategy,
              )
          ],
      )
    ref_cfg = _small_vocab(config_registry.qwen35_debugmodel())
    if document_resets is not None:
      _set_gated_delta(ref_cfg, reset_at_documents=document_resets)
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

    def run_rank(rank):
      model = copy.deepcopy(cp_model)
      extra = {"cp_full_positions": positions} if full_positions else {}
      if document_resets is not None:
        extra["gated_delta_document_resets"] = document_resets
      logits = model(
          _shard(inputs, strategy, cp, rank, _SEQ_LEN),
          positions=_shard(positions, strategy, cp, rank, _SEQ_LEN),
          **extra,
      )
      loss = F.cross_entropy(
          logits.float(),
          _shard(labels, strategy, cp, rank, _SEQ_LEN),
          reduction="sum",
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


class CompiledCPGatedDeltaNetTest(parameterized.TestCase):
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

  @parameterized.product(
      load_balancer=(None, "headtail"), fuse_input_projections=(False, True)
  )
  def test_one_graph_one_wait_per_collective(
      self, load_balancer, fuse_input_projections
  ):
    from functorch.compile import min_cut_rematerialization_partition  # pylint: disable=g-import-not-at-top
    from torch._dynamo.backends.common import aot_autograd  # pylint: disable=g-import-not-at-top

    model_cfg = GatedDeltaNetContextParallelTransform(
        load_balancer=load_balancer, strategy="allgather_reducescatter"
    ).transform(config_registry.qwen35_debugmodel().model_spec.model)
    _, delta_cfg, _, _ = next(
        iter(model_cfg.traverse(CPTpuGatedDeltaNet.Config))
    )
    delta_cfg.fuse_input_projections = fuse_input_projections
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
    found = set()
    for gm in graphs:
      waited = []
      n_collectives = 0
      for node in gm.graph.nodes:
        if node.target is wait:
          self.assertIn(node.args[0].target, collectives, gm.code)
          waited.append(node.args[0])
        elif node.target in collectives:
          n_collectives += 1
          found.add(node.target)
      self.assertLen(set(waited), len(waited), gm.code)
      self.assertLen(waited, n_collectives, gm.code)
    # Forward all-gather + reduce-scatter, and their backward duals; no
    # all-to-all and no positions all-gather beyond x's.
    self.assertEqual(found, collectives)
    code = "\n".join(gm.code for gm in graphs)
    self.assertNotIn("all_to_all", code)


class Qwen35LayerInputsTest(parameterized.TestCase):
  """Single-device ``fuse_input_projections`` and document-reset override."""

  def setUp(self):
    super().setUp()
    torch.manual_seed(0)

  def _models(self, **ref_fields):
    cfg = _small_vocab(config_registry.qwen35_debugmodel())
    with torch.device("cpu"):
      model = cfg.model_spec.model.build()
      model.init_states(buffer_device=torch.device("cpu"))
    ref_cfg = _set_gated_delta(
        _small_vocab(config_registry.qwen35_debugmodel()), **ref_fields
    )
    with torch.device("cpu"):
      ref = ref_cfg.model_spec.model.build()
    ref.load_state_dict(model.state_dict())
    return model, ref

  def _loss_and_grads(self, model, **kwargs):
    tokens = _SEQ_LEN * len(_DOCS)
    gen = torch.Generator().manual_seed(1)
    inputs = torch.randint(0, 512, (tokens,), generator=gen)
    labels = torch.randint(0, 512, (tokens,), generator=gen)
    with set_unflatten_seq_len(_SEQ_LEN):
      logits = model(inputs, positions=_packed_positions(_DOCS), **kwargs)
    loss = F.cross_entropy(logits.float(), labels, reduction="sum")
    loss.backward()
    return loss.detach(), {
        n: p.grad.clone() for n, p in model.named_parameters()
    }

  def test_fused_input_projections_match(self):
    model, ref = self._models()
    _set_gated_delta_modules(model, fuse_input_projections=True)
    got_loss, got_grads = self._loss_and_grads(model)
    want_loss, want_grads = self._loss_and_grads(ref)
    torch.testing.assert_close(got_loss, want_loss, rtol=1e-5, atol=1e-4)
    for name, want in want_grads.items():
      torch.testing.assert_close(
          got_grads[name], want, rtol=1e-4, atol=1e-5, msg=name
      )

  def test_fused_input_projections_reject_bias(self):
    model, _ = self._models()
    delta_net = next(
        m for m in model.modules() if isinstance(m, TpuGatedDeltaNet)
    )
    delta_net.fuse_input_projections = True
    delta_net.in_proj_q.bias = torch.nn.Parameter(
        torch.zeros(delta_net.in_proj_q.out_features)
    )
    with self.assertRaisesRegex(ValueError, "bias-free"):
      delta_net.input_projections(
          torch.randn(4, delta_net.in_proj_q.in_features)
      )

  def test_document_resets_override_matches_config(self):
    """``gated_delta_document_resets=False`` == ``document_resets.enabled=False``."""
    model, ref = self._models()
    _set_gated_delta_modules(ref, reset_at_documents=False)
    got_loss, got_grads = self._loss_and_grads(
        model, gated_delta_document_resets=False
    )
    want_loss, want_grads = self._loss_and_grads(ref)
    torch.testing.assert_close(got_loss, want_loss)
    for name, want in want_grads.items():
      torch.testing.assert_close(got_grads[name], want, msg=name)
    # And it does change the result for packed documents.
    model.zero_grad()
    default_loss, _ = self._loss_and_grads(model)
    self.assertFalse(torch.allclose(default_loss, got_loss))


def _set_gated_delta_modules(model, **attrs):
  for module in model.modules():
    if isinstance(module, TpuGatedDeltaNet):
      for name, value in attrs.items():
        setattr(module, name, value)


class CPQwen35ConfigTest(parameterized.TestCase):
  """Recipes, the GatedDeltaNet transform and its validation."""

  @parameterized.parameters(
      ("qwen35_debugmodel_allgather_cp2", "headtail"),
      ("qwen35_debugmodel_ulysses_cp2", None),
      ("qwen35_moe_debugmodel_allgather_cp2", "headtail"),
      ("qwen35_moe_debugmodel_ulysses_cp2", None),
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
      ("qwen35_moe_35b_split_tokamax_ep_allgather_cp2", "headtail"),
      ("qwen35_moe_35b_split_tokamax_ep_ulysses_cp2", None),
  )
  def test_ep_cp_recipes_keep_expert_parallelism(self, recipe, load_balancer):
    """CP on top of the 35B EP benchmark recipe keeps its EP settings."""
    cfg = getattr(config_registry, recipe)()
    self.assertEqual(cfg.parallelism.context_parallel_degree, 2)
    self.assertEqual(
        cfg.parallelism.context_parallel_load_balancer, load_balancer
    )
    self.assertEqual(cfg.parallelism.expert_parallel_degree, 4)
    self.assertEqual(
        cfg.compile.graph_split.backward_partition_mode, "by_layer"
    )
    delta_nets = list(cfg.model_spec.model.traverse(GatedDeltaNet.Config))
    self.assertNotEmpty(delta_nets)
    for _, delta_net, _, _ in delta_nets:
      self.assertIsInstance(
          delta_net.inner_gated_delta_net, CPTpuInnerGatedDeltaNet.Config
      )

  @parameterized.parameters(
      ("qwen35_moe_35b_ep_allgather_cp2", "headtail"),
      ("qwen35_moe_35b_ep_ulysses_cp2", None),
  )
  def test_regular_trainer_ep_cp_recipes_compile_every_block(
      self, recipe, load_balancer
  ):
    """The regular-trainer EP + CP recipes keep EP and compile every block."""
    cfg = getattr(config_registry, recipe)()
    self.assertIs(type(cfg), type(config_registry.qwen35_moe_35b()))
    self.assertEqual(cfg.parallelism.context_parallel_degree, 2)
    self.assertEqual(
        cfg.parallelism.context_parallel_load_balancer, load_balancer
    )
    self.assertEqual(cfg.parallelism.expert_parallel_degree, 4)
    self.assertEqual(cfg.parallelism.spmd_backend, "spmd_types")
    self.assertTrue(cfg.compile.enable)
    self.assertEqual(cfg.compile.backend, "tpu")
    self.assertFalse(cfg.compile.compile_grad_norm)
    self.assertIn(
        config_registry.TOKAMAX_GATED_DELTA_OVERRIDE, cfg.override.imports
    )
    delta_nets = list(cfg.model_spec.model.traverse(GatedDeltaNet.Config))
    self.assertNotEmpty(delta_nets)
    for _, delta_net, _, _ in delta_nets:
      self.assertIsInstance(
          delta_net.inner_gated_delta_net, CPTpuInnerGatedDeltaNet.Config
      )

  def _cp_config(self, *, gated_delta_lb="unset", cp=2, strategy="ulysses"):
    cfg = config_registry.qwen35_debugmodel()
    cfg.parallelism.context_parallel_degree = cp
    cfg.parallelism.spmd_backend = "spmd_types"
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

  def test_transform_strategy_converts_gated_delta_nets(self):
    cfg = config_registry.qwen35_debugmodel()
    model = cfg.model_spec.model
    for _ in range(2):
      model = GatedDeltaNetContextParallelTransform(
          load_balancer=None,
          strategy="allgather_reducescatter",
          reduce_dtype="bfloat16",
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

  def test_transform_default_strategy_keeps_gated_delta_nets(self):
    model = GatedDeltaNetContextParallelTransform(load_balancer=None).transform(
        config_registry.qwen35_debugmodel().model_spec.model
    )
    for _, delta_net, _, _ in model.traverse(GatedDeltaNet.Config):
      self.assertIs(type(delta_net), TpuGatedDeltaNet.Config)

  def test_transform_rejects_unknown_strategy(self):
    with self.assertRaisesRegex(ValueError, "strategy"):
      GatedDeltaNetContextParallelTransform(
          load_balancer=None, strategy="ring"
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
    # The unsharded positions, for the layers' document boundaries.
    torch.testing.assert_close(extra["cp_full_positions"], tokens % _SEQ_LEN)


if __name__ == "__main__":
  absltest.main()
