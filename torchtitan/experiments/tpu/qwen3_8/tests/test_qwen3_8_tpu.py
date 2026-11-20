"""Tests for Qwen3.8 (text-only) on CPU and TPU devices.

Covers:
1. HuggingFace reference parity (``Qwen3_5MoeForCausalLM``, the architecture
   of the Qwen3.8 checkpoints) through ``Qwen35StateDictAdapter`` at the
   2.4T-A95B head and expert ratios, which the Qwen3.5 tests do not reach:
   8 GatedDeltaNet value heads per key head, 16 query heads per KV head,
   top-10 routing over more experts plus the shared expert, 1/4 partial
   rotary with ``mrope_section`` in the 2.4T's proportions. Forward logits,
   loss and every parameter gradient, with negative controls.
2. ``scaled_down``, the 2.4T-A95B flavor's shapes and parameter counts, and
   every recipe in the registry.
"""

import copy
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.tpu.components.expert_parallel import use_expert_parallel
from torchtitan.experiments.tpu.components.grouped_experts import (
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.components.token_dispatcher import TpuAllToAllTokenDispatcher
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.qwen3_5 import use_tpu_modules
import torchtitan.experiments.tpu.qwen3_8 as qwen3_8_tpu
from torchtitan.experiments.tpu.qwen3_8 import config_registry
from torchtitan.experiments.tpu.testing import base_device_test
from torchtitan.experiments.tpu.testing import test_utils
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.moe import MoE
# pylint: disable-next=g-bad-import-order
from torchtitan.models.qwen3_5 import (  # pylint: disable=protected-access
    _build_qwen35_moe_layers,
    _EMBEDDING_INIT,
    _offset_norm,
    _output_linear_init,
    Qwen35Model,
)
from torchtitan.models.qwen3_5.rope import MRoPE
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig


@dataclasses.dataclass(frozen=True)
class Geometry:
  """A Qwen3.8 MoE text geometry, shared by the HF and TorchTitan builds."""

  dim: int
  layer_types: tuple[str, ...]  # "linear_attention" / "full_attention"
  n_heads: int
  n_kv_heads: int
  head_dim: int
  n_key_heads: int
  n_value_heads: int
  key_head_dim: int
  value_head_dim: int
  moe_hidden_dim: int
  shared_expert_hidden_dim: int
  num_experts: int
  top_k: int
  mrope_section: tuple[int, int, int]
  vocab_size: int = 2048

  @property
  def rotary_dim(self) -> int:
    return self.head_dim // 4  # partial_rotary_factor 0.25, as in 2.4T-A95B


# 2.4T-A95B ratios at a CPU-friendly width. 2.4T-A95B: dim 8192, 64 / 4 heads
# of 256, GatedDeltaNet 16 key / 128 value heads of 128, experts 512 x 2048
# top-10, shared expert 2048, mrope_section [11, 11, 10] of rotary_dim 64.
TINY_2_4T_RATIOS = Geometry(
    dim=256,
    layer_types=("linear_attention",) * 3 + ("full_attention",),
    n_heads=16,
    n_kv_heads=1,
    head_dim=64,
    n_key_heads=2,
    n_value_heads=16,
    key_head_dim=32,
    value_head_dim=32,
    moe_hidden_dim=64,
    shared_expert_hidden_dim=64,
    num_experts=32,
    top_k=10,
    mrope_section=(3, 3, 2),
)


def build_hf_reference(geometry: Geometry, seed: int = 42) -> torch.nn.Module:
  """HF ``Qwen3_5MoeForCausalLM`` in fp32 with perturbed unit/zero params."""
  hf_cfg = Qwen3_5MoeTextConfig(
      vocab_size=geometry.vocab_size,
      hidden_size=geometry.dim,
      num_hidden_layers=len(geometry.layer_types),
      layer_types=list(geometry.layer_types),
      num_attention_heads=geometry.n_heads,
      num_key_value_heads=geometry.n_kv_heads,
      head_dim=geometry.head_dim,
      linear_conv_kernel_dim=4,
      linear_key_head_dim=geometry.key_head_dim,
      linear_value_head_dim=geometry.value_head_dim,
      linear_num_key_heads=geometry.n_key_heads,
      linear_num_value_heads=geometry.n_value_heads,
      moe_intermediate_size=geometry.moe_hidden_dim,
      shared_expert_intermediate_size=geometry.shared_expert_hidden_dim,
      num_experts=geometry.num_experts,
      num_experts_per_tok=geometry.top_k,
      rms_norm_eps=1e-6,
      max_position_embeddings=128,
      rope_parameters={
          "rope_type": "default",
          "rope_theta": 10000000.0,
          "partial_rotary_factor": 0.25,
          "mrope_section": list(geometry.mrope_section),
      },
      tie_word_embeddings=False,
      attn_implementation="sdpa",
  )
  torch.manual_seed(seed)
  hf_model = Qwen3_5MoeForCausalLM(hf_cfg).float().eval()
  gen = torch.Generator().manual_seed(1234)
  with torch.no_grad():
    for name, param in hf_model.named_parameters():
      if "norm" in name or "A_log" in name or "dt_bias" in name:
        param.copy_(0.2 * torch.randn(param.shape, generator=gen))
  return hf_model


def build_tt_config(
    geometry: Geometry, *, seq_len: int, use_tpu_grouped_experts: bool
) -> Qwen35Model.Config:
  """The TPU Qwen3.8 model config for ``geometry`` (as ``model_registry``)."""
  rope = MRoPE.Config(
      dim=geometry.rotary_dim,
      max_context_length=seq_len,
      theta=10_000_000.0,
      mrope_section=list(geometry.mrope_section),
  )
  # Build one period of the pattern per needed layer and pick the requested
  # layer types in order (full attention is every 4th layer upstream).
  period = _build_qwen35_moe_layers(
      rope=rope,
      attn_backend="sdpa",
      n_layers=4,
      dim=geometry.dim,
      n_heads=geometry.n_heads,
      n_kv_heads=geometry.n_kv_heads,
      head_dim=geometry.head_dim,
      rotary_dim=geometry.rotary_dim,
      moe_hidden_dim=geometry.moe_hidden_dim,
      num_experts=geometry.num_experts,
      top_k=geometry.top_k,
      shared_expert_hidden_dim=geometry.shared_expert_hidden_dim,
      n_key_heads=geometry.n_key_heads,
      n_value_heads=geometry.n_value_heads,
      key_head_dim=geometry.key_head_dim,
      value_head_dim=geometry.value_head_dim,
  )
  linear_layer, full_layer = period[0], period[3]
  layers = [
      copy.deepcopy(full_layer if t == "full_attention" else linear_layer)
      for t in geometry.layer_types
  ]
  config = Qwen35Model.Config(
      vocab_size=geometry.vocab_size,
      dim=geometry.dim,
      norm=_offset_norm(geometry.dim),
      tok_embeddings=Embedding.Config(
          num_embeddings=geometry.vocab_size,
          embedding_dim=geometry.dim,
          param_init=_EMBEDDING_INIT,
      ),
      lm_head=Linear.Config(
          in_features=geometry.dim,
          out_features=geometry.vocab_size,
          param_init=_output_linear_init(geometry.dim),
      ),
      layers=layers,
  )
  if use_tpu_grouped_experts:
    config = TpuGroupedExpertsConverter.Config().build().convert(config)
  return use_tpu_modules(config)


def relative_l2_error(test: torch.Tensor, reference: torch.Tensor) -> float:
  test_f = test.detach().float().cpu()
  ref_f = reference.detach().float().cpu()
  denom = torch.linalg.vector_norm(ref_f).clamp_min(1e-6)
  return (torch.linalg.vector_norm(test_f - ref_f) / denom).item()


def run_parity(
    test: parameterized.TestCase,
    *,
    geometry: Geometry,
    device: torch.device,
    use_tpu_grouped_experts: bool,
    matmul_precision: str,
    logit_atol: float,
    logit_rtol: float,
    loss_tol: float,
    grad_rel_bound: float,
    batch: int = 2,
    seq_len: int = 16,
) -> dict[str, float]:
  """One forward/backward through HF (CPU, fp32) and TorchTitan (``device``).

  Returns:
    The measured errors: max abs logit error, abs loss error, and the worst
    per-tensor gradient relative L2 error.
  """
  hf_model = build_hf_reference(geometry)
  tt_config = build_tt_config(
      geometry, seq_len=128, use_tpu_grouped_experts=use_tpu_grouped_experts
  )
  adapter = Qwen35StateDictAdapter(tt_config, None)
  tt_model = tt_config.build()
  tt_model.init_states(buffer_device=torch.device("cpu"))

  tt_sd = adapter.from_hf(copy.deepcopy(hf_model.state_dict()))
  missing, unexpected = tt_model.load_state_dict(tt_sd, strict=False)
  test.assertEmpty(missing)
  test.assertEmpty(unexpected)
  if not use_tpu_grouped_experts:
    round_trip = adapter.from_hf(adapter.to_hf(tt_model.state_dict()))
    test.assertEqual(set(round_trip), set(tt_model.state_dict()))
    for k, v in tt_model.state_dict().items():
      torch.testing.assert_close(round_trip[k], v, msg=f"Round-trip key {k}")
  tt_model = tt_model.to(device)

  gen = torch.Generator().manual_seed(99)
  tokens_2d = torch.randint(0, geometry.vocab_size, (batch, seq_len), generator=gen)
  labels_2d = torch.randint(0, geometry.vocab_size, (batch, seq_len), generator=gen)
  positions_1d = torch.arange(seq_len, dtype=torch.long).repeat(batch)

  hf_model.zero_grad()
  hf_logits = hf_model(tokens_2d).logits
  hf_loss = F.cross_entropy(
      hf_logits.view(-1, geometry.vocab_size), labels_2d.view(-1)
  )
  hf_loss.backward()

  tt_model.zero_grad()
  prev_precision = torch.get_float32_matmul_precision()
  torch.set_float32_matmul_precision(matmul_precision)
  try:
    with set_unflatten_seq_len(seq_len):
      tt_logits = tt_model(
          tokens_2d.view(-1).to(device), positions=positions_1d.to(device)
      )
      tt_loss = F.cross_entropy(
          tt_logits.float().view(-1, geometry.vocab_size),
          labels_2d.view(-1).to(device),
      )
      tt_loss.backward()
  finally:
    torch.set_float32_matmul_precision(prev_precision)

  tt_logits_cpu = tt_logits.detach().float().cpu().view(batch, seq_len, -1)
  measured = {
      "logits_max_abs": (tt_logits_cpu - hf_logits.detach()).abs().max().item(),
      "logits_ref_max_abs": hf_logits.detach().abs().max().item(),
      "logits_rel_l2": relative_l2_error(tt_logits_cpu, hf_logits),
      "loss_abs": abs(tt_loss.item() - hf_loss.item()),
  }
  test_utils.check_equivalence(
      tt_logits_cpu,
      hf_logits.detach(),
      atol=logit_atol,
      rtol=logit_rtol,
      check_name="HF Parity Logits",
  )
  test_utils.check_equivalence(
      tt_loss.detach().cpu(),
      hf_loss.detach(),
      atol=loss_tol,
      rtol=loss_tol,
      check_name="HF Parity Loss",
  )

  hf_grads = {
      name: param.grad.detach().clone()
      for name, param in hf_model.named_parameters()
      if param.grad is not None
  }
  expected = adapter.from_hf(hf_grads)
  actual = {
      name: param.grad.detach().float().cpu()
      for name, param in tt_model.named_parameters()
      if param.grad is not None
  }
  test.assertEqual(set(actual), set(expected))
  worst = 0.0
  for name, ref_grad in expected.items():
    rel_err = relative_l2_error(actual[name], ref_grad)
    worst = max(worst, rel_err)
    test.assertLessEqual(
        rel_err,
        grad_rel_bound,
        msg=f"Gradient relative L2 error too high for {name}: {rel_err:.3e}",
    )
    if torch.linalg.vector_norm(ref_grad).item() > 1e-4:
      # Negative controls: the bound rejects zeroed and sign-flipped grads.
      test.assertGreater(
          relative_l2_error(torch.zeros_like(ref_grad), ref_grad),
          grad_rel_bound,
      )
      test.assertGreater(relative_l2_error(-ref_grad, ref_grad), grad_rel_bound)
  measured["grad_rel_l2_worst"] = worst
  return measured


class Qwen38Test(base_device_test.BaseAcceleratorDeviceTest):
  """Tests for Qwen3.8 on the TPU lane, on CPU and TPU."""

  def setUp(self):
    super().setUp()
    use_cpu_safe_histc_patch()

  def _is_tpu(self) -> bool:
    return self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU

  @parameterized.named_parameters(
      dict(testcase_name="_moe", use_tpu_grouped_experts=False),
      dict(testcase_name="_moe_tpu_grouped_experts", use_tpu_grouped_experts=True),
  )
  def test_hf_reference_parity_2_4t_ratios(self, use_tpu_grouped_experts: bool):
    """Logits, loss and every gradient against HF at the 2.4T-A95B ratios."""
    # GroupedExperts run their matmuls in bf16 even in an fp32 model, and
    # libtpu rejects HIGHEST precision for the ragged dot behind
    # torch._grouped_mm, so MoE uses bf16 tolerances (as the Qwen3.5 tests).
    measured = run_parity(
        self,
        geometry=TINY_2_4T_RATIOS,
        device=self.accelerator_device,
        use_tpu_grouped_experts=use_tpu_grouped_experts,
        matmul_precision="high" if self._is_tpu() else "highest",
        logit_atol=5e-2,
        logit_rtol=1e-2,
        loss_tol=5e-3,
        grad_rel_bound=6e-2,
    )
    print(f"measured: {measured}")

  def test_scaled_down(self):
    """`scaled_down` keeps the width and sets every expert count."""
    spec = qwen3_8_tpu.model_registry(
        "2.4T-A95B", seq_len=4096, num_layers=4, num_experts=64
    )
    model = spec.model
    self.assertEqual(len(model.layers), 4)
    self.assertEqual(
        [layer.attention is not None for layer in model.layers],
        [False, False, False, True],
    )
    self.assertEqual(model.dim, 8192)
    moes = [moe for _, moe, _, _ in model.traverse(MoE.Config)]
    self.assertLen(moes, 4)
    for moe in moes:
      self.assertEqual(moe.num_experts, 64)
      self.assertEqual(moe.router.num_experts, 64)
      self.assertEqual(moe.router.top_k, 10)
      self.assertEqual(moe.router.gate.out_features, 64)
      self.assertEqual(moe.routed_experts.inner_experts.num_experts, 64)
      self.assertEqual(moe.routed_experts.token_dispatcher.num_experts, 64)
      self.assertEqual(moe.routed_experts.inner_experts.hidden_dim, 2048)
    with self.assertRaisesRegex(ValueError, "below top_k"):
      qwen3_8_tpu.model_registry("2.4T-A95B", seq_len=4096, num_experts=8)
    with self.assertRaisesRegex(ValueError, "num_layers"):
      qwen3_8_tpu.model_registry("2.4T-A95B", seq_len=4096, num_layers=93)

  def test_2_4t_a95b_flavor(self):
    """2.4T-A95B shapes and parameter counts, built on the meta device."""
    spec = qwen3_8_tpu.model_registry("2.4T-A95B", seq_len=4096)
    self.assertEqual(spec.name, "qwen3_8")
    config = spec.model
    self.assertIsNone(config.vision_encoder)
    self.assertLen(config.layers, 92)
    self.assertEqual(sum(l.attention is not None for l in config.layers), 23)
    self.assertEqual(config.vocab_size, 248320)
    with torch.device("meta"):
      model = config.build()
    nparams, _ = config.get_nparams_and_flops(model, 4096)
    # 26.25B per layer (25.77B routed experts) x 92 + 4.07B embeddings/head.
    self.assertBetween(nparams, 2.40e12, 2.44e12)

  @parameterized.parameters(
      "qwen38_debugmodel",
      "qwen38_moe_debugmodel",
      "qwen38_moe_debugmodel_ep8",
      "qwen38_2_4t_a95b_proxy",
      "qwen38_2_4t_a95b_proxy_tpu7x",
      "qwen38_2_4t_a95b_proxy_tpu7x_8l",
      "qwen38_2_4t_a95b",
  )
  def test_recipe_parses(self, config_name: str):
    cfg = ConfigManager().parse_args([
        "--module",
        "torchtitan.experiments.tpu.qwen3_8",
        "--config",
        config_name,
    ])
    self.assertEqual(cfg.model_spec.name, "qwen3_8")
    self.assertIsNone(cfg.model_spec.model.vision_encoder)
    self.assertTrue(hasattr(config_registry, config_name))
    moes = [m for _, m, _, _ in cfg.model_spec.model.traverse(MoE.Config)]
    for moe in moes:
      # Upstream's expert-bias balancer, not a router aux loss.
      self.assertEqual(moe.load_balance_coeff, 1e-3)
      self.assertIsNone(moe.router.aux_loss)

  def test_ep_recipe_pins_ragged_transport(self):
    """The EP recipe stays on the ragged transport at any EP degree.

    Left unset, the transport becomes upstream's "auto", which picks
    Ring-of-Experts at EP <= top_k when it can run (TPU). At EP = 2 and the
    debug model's top_k = 2 that would switch the recipe off the transport its
    EP parity was measured on.
    """
    cfg = ConfigManager().parse_args([
        "--module",
        "torchtitan.experiments.tpu.qwen3_8",
        "--config",
        "qwen38_moe_debugmodel_ep8",
        "--parallelism.expert_parallel_degree",
        "2",
    ])
    model_config = cfg.model_spec.model
    # What `TpuQwen35Model.Config.update_from_config` does with the EP degree.
    use_expert_parallel(
        model_config, ep_degree=cfg.parallelism.expert_parallel_degree
    )
    dispatchers = [
        d
        for _, d, _, _ in model_config.traverse(
            TpuAllToAllTokenDispatcher.Config
        )
    ]
    self.assertLen(dispatchers, len(list(model_config.traverse(MoE.Config))))
    x_TD = torch.zeros(4, model_config.dim, device=self.accelerator_device)
    for dispatcher_config in dispatchers:
      self.assertLessEqual(2, dispatcher_config.top_k)
      self.assertEqual(dispatcher_config.ep_transport, "ragged_all_to_all")
      dispatcher = dispatcher_config.build()
      # pylint: disable-next=protected-access
      resolved = dispatcher._resolve_ep_transport(2, x_TD)
      self.assertEqual(resolved, "ragged_all_to_all")


  @parameterized.parameters(
      "qwen38_2_4t_a95b_proxy",
      "qwen38_2_4t_a95b_proxy_tpu7x",
      "qwen38_2_4t_a95b_proxy_tpu7x_8l",
  )
  def test_proxies_hold_the_full_models_experts_per_rank(self, config_name):
    """Proxies keep 8 routed experts per EP rank, as EP=64 on 512 experts."""
    cfg = getattr(config_registry, config_name)()
    moes = [m for _, m, _, _ in cfg.model_spec.model.traverse(MoE.Config)]
    self.assertNotEmpty(moes)
    for moe in moes:
      self.assertEqual(
          moe.num_experts // cfg.parallelism.expert_parallel_degree, 8
      )
      self.assertEqual(moe.router.gate.out_features, moe.num_experts)

  # The argument sets of the launch commands in
  # recipes/ironwood/qwen3.8-2.4t-a95b/tpu7x-4x4x4/README.md; keep in sync.
  @parameterized.named_parameters(
      dict(
          testcase_name="_job_a_smoke",
          args=[
              "--config=qwen38_2_4t_a95b_proxy_tpu7x",
              "--dump_folder=/data/qwen38/job-a/smoke",
              "--training.steps=1",
              "--metrics.log_freq=1",
          ],
      ),
      dict(
          testcase_name="_job_a",
          args=[
              "--config=qwen38_2_4t_a95b_proxy_tpu7x",
              "--dump_folder=/data/qwen38/job-a/main",
              "--training.steps=10",
              "--metrics.log_freq=1",
              "--metrics.enable_tensorboard",
              "--profiler.enable_profiling",
              "--profiler.profile_freq=8",
              "--profiler.profiler_warmup=3",
              "--profiler.profiler_active=1",
          ],
      ),
      dict(
          testcase_name="_job_b_point",
          args=[
              "--config=qwen38_2_4t_a95b_proxy_tpu7x_8l",
              "--dump_folder=/data/qwen38/job-b/8l-t2048-float32",
              "--training.steps=10",
              "--metrics.log_freq=1",
              "--metrics.enable_tensorboard",
              "--training.max_context_length=2048",
              "--training.num_tokens_per_microbatch_per_dp_rank=2048",
              "--training.dtype=float32",
          ],
      ),
  )
  def test_launch_configs_parse(self, args):
    cfg = ConfigManager().parse_args(
        ["--module", "torchtitan.experiments.tpu.qwen3_8", *args]
    )
    self.assertEqual(cfg.parallelism.expert_parallel_degree, 64)
    self.assertEqual(
        cfg.training.num_tokens_per_microbatch_per_dp_rank,
        cfg.training.max_context_length,
    )
    # A --training.max_context_length override does not resize the model's
    # RoPE table; positions past it fail at run time.
    self.assertGreaterEqual(
        cfg.model_spec.max_context_length, cfg.training.max_context_length
    )
    self.assertIn("xla_tpu_scoped_vmem_limit_kib", cfg.libtpu_init_flags)
    # The XLA grouped matmul overflows TPU7x VMEM at these shapes.
    for _, experts, _, _ in cfg.model_spec.model.traverse(
        TpuGroupedExperts.Config
    ):
      self.assertTrue(experts.use_pallas_grouped_mm)
      self.assertEqual(experts.pallas_grouped_mm_tile_k, 2048)


if __name__ == "__main__":
  absltest.main()
