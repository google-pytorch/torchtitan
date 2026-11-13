"""CPU tests for HF float8 conversion, QK-norm sizing, and G8-3B configs."""

import json
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
import torchao
from torchao.float8.float8_linear import Float8Linear

if getattr(torchao, "__version__", "") == "unknown":
  torchao.__version__ = "0.17.0"

from torchtitan.experiments.tpu.transformers_modeling_backend import config_registry  # pylint: disable=g-import-not-at-top
from torchtitan.experiments.tpu.transformers_modeling_backend import float8  # pylint: disable=g-import-not-at-top
from torchtitan.experiments.tpu.transformers_modeling_backend import model as tpu_model  # pylint: disable=g-import-not-at-top


class _ToyBlock(nn.Module):

  def __init__(self, hidden: int = 256, mlp: int = 512, bias: bool = False):
    super().__init__()
    self.q_proj = nn.Linear(hidden, hidden, bias=bias)
    self.k_proj = nn.Linear(hidden, hidden, bias=bias)
    self.v_proj = nn.Linear(hidden, hidden, bias=bias)
    self.o_proj = nn.Linear(hidden, hidden, bias=bias)
    self.gate_proj = nn.Linear(hidden, mlp, bias=bias)
    self.up_proj = nn.Linear(hidden, mlp, bias=bias)
    self.down_proj = nn.Linear(mlp, hidden, bias=bias)
    self.gate = nn.Linear(hidden, 128, bias=False)
    self.router = nn.Linear(hidden, 128, bias=False)
    self.unaligned_112 = nn.Linear(hidden, 112, bias=False)
    self.unaligned_10 = nn.Linear(hidden, 10, bias=False)


class _ToyModel(nn.Module):

  def __init__(self, hidden: int = 256, mlp: int = 512, bias: bool = False):
    super().__init__()
    self.tok_embeddings = nn.Embedding(1024, hidden)
    self.embed_proj = nn.Linear(hidden, hidden, bias=False)
    self.layers = nn.ModuleList([_ToyBlock(hidden, mlp, bias=bias)])
    self.lm_head = nn.Linear(hidden, 1024, bias=False)


class Float8ConversionTest(parameterized.TestCase):

  def test_is_float8_candidate_includes_gate_proj_and_skips_moe_gates(self):
    lin = nn.Linear(256, 512, bias=False)
    self.assertTrue(float8.is_float8_candidate(lin, "layers.0.mlp.gate_proj"))
    self.assertTrue(float8.is_float8_candidate(lin, "layers.0.ffn.gate_proj"))
    self.assertFalse(float8.is_float8_candidate(lin, "layers.0.mlp.gate"))
    self.assertFalse(float8.is_float8_candidate(lin, "layers.0.mlp.router"))
    self.assertFalse(float8.is_float8_candidate(lin, "layers.0.router.proj"))
    self.assertFalse(float8.is_float8_candidate(lin, "lm_head"))

  def test_fused_converts_all_128_aligned_block_linears_in_place(self):
    toy = _ToyModel()
    orig_up_weight = toy.layers[0].up_proj.weight
    float8.apply_float8_linears(toy, "fused")

    block = toy.layers[0]
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
      self.assertIsInstance(getattr(block, name), float8.Fp8FusedLinear)
    self.assertIs(block.up_proj.weight, orig_up_weight)
    for skipped in (
        toy.embed_proj,
        toy.lm_head,
        block.gate,
        block.router,
        block.unaligned_112,
        block.unaligned_10,
    ):
      self.assertIs(type(skipped), nn.Linear)

  def test_fused_module_filter_selects_only_matching_fqns(self):
    toy = _ToyModel()
    float8.apply_float8_linears(
        toy,
        "fused",
        only_modules="gate_proj,up_proj,down_proj,q_proj,k_proj,v_proj",
    )

    block = toy.layers[0]
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ):
      self.assertIsInstance(getattr(block, name), float8.Fp8FusedLinear)
    self.assertIs(type(block.o_proj), nn.Linear)
    self.assertIs(type(block.gate), nn.Linear)

  def test_rowwise_converts_16_aligned_block_linears_with_no_fast_accum(self):
    toy = _ToyModel()
    float8.apply_float8_linears(
        toy, "rowwise", only_modules="up_proj,unaligned_112"
    )

    block = toy.layers[0]
    self.assertIsInstance(block.up_proj, Float8Linear)
    self.assertIsInstance(block.unaligned_112, Float8Linear)
    self.assertFalse(block.up_proj.config.gemm_config_output.use_fast_accum)
    self.assertIs(type(block.q_proj), nn.Linear)
    self.assertIs(type(block.unaligned_10), nn.Linear)
    self.assertIs(type(toy.lm_head), nn.Linear)

  def test_raises_when_no_module_matches(self):
    toy = _ToyModel()
    with self.assertRaisesRegex(ValueError, "converted no linear layers"):
      float8.apply_float8_linears(toy, "fused", only_modules="nonexistent")
    with self.assertRaisesRegex(ValueError, "divisible by 128"):
      float8.apply_float8_linears(toy, "fused", only_modules="unaligned_112")

  def test_fp8_fused_linear_forward_with_and_without_bias(self):
    for use_bias in (False, True):
      toy = _ToyModel(bias=use_bias)
      float8.apply_float8_linears(toy, "fused", only_modules="up_proj")
      layer = toy.layers[0].up_proj
      x = torch.randn(2, 128, 256, dtype=torch.bfloat16)
      with mock.patch(
          "torchtitan.experiments.tpu.kernels.fp8_linear.fp8_linear",
          side_effect=lambda a, w: torch.nn.functional.linear(a, w.to(a.dtype)),
      ) as mocked_fn:
        got = layer(x)
      mocked_fn.assert_called_once_with(x, layer.weight)
      expected = torch.nn.functional.linear(x, layer.weight.to(x.dtype))
      if layer.bias is not None:
        expected = expected + layer.bias.to(x.dtype)
      self.assertEqual(got.dtype, torch.bfloat16)
      self.assertTrue(torch.equal(got, expected))

  def test_esmc_size_qk_norms_resizes_mismatched_projection_width(self):
    attn = nn.Module()
    attn.config = SimpleNamespace(
        num_attention_heads=10, num_key_value_heads=10
    )
    attn.head_dim = 256
    attn.q_norm = nn.LayerNorm(2048, eps=1e-6, bias=False).to(torch.bfloat16)
    attn.k_norm = nn.LayerNorm(2048, eps=1e-6, bias=False).to(torch.bfloat16)

    tpu_model._EsmcTPUAttention._size_qk_norms(attn)  # pylint: disable=protected-access
    self.assertEqual(tuple(attn.q_norm.normalized_shape), (2560,))
    self.assertEqual(tuple(attn.k_norm.normalized_shape), (2560,))
    self.assertEqual(attn.q_norm.eps, 1e-6)
    self.assertIsNone(attn.q_norm.bias)
    self.assertEqual(attn.q_norm.weight.dtype, torch.bfloat16)

    q_norm_id = attn.q_norm
    tpu_model._EsmcTPUAttention._size_qk_norms(attn)  # pylint: disable=protected-access
    self.assertIs(attn.q_norm, q_norm_id)

  def test_g8_3b_configs_and_cpu_reference_reset(self):
    bf16_cfg = config_registry.hf_esmc_g8_3b_mlm_uniref50_graph()
    self.assertEqual(
        json.loads(bf16_cfg.hf_config_overrides),
        config_registry._ESMC_G8_3B_OVERRIDES,  # pylint: disable=protected-access
    )
    self.assertEqual(bf16_cfg.hf_float8_recipe, "")
    self.assertEqual(bf16_cfg.hf_float8_modules, "")
    self.assertTrue(bf16_cfg.esmc_fused_qk_norm_rope)

    fp8_cfg = config_registry.hf_esmc_g8_3b_mlm_uniref50_graph_float8()
    self.assertEqual(fp8_cfg.hf_float8_recipe, "fused")
    self.assertEqual(
        fp8_cfg.hf_float8_modules,
        "gate_proj,up_proj,down_proj,q_proj,k_proj,v_proj",
    )

    cpu_ref = config_registry.as_cpu_reference(fp8_cfg)
    self.assertEqual(cpu_ref.hf_float8_recipe, "")
    self.assertEqual(cpu_ref.hf_float8_modules, "")
    self.assertFalse(cpu_ref.esmc_fused_qk_norm_rope)
    self.assertFalse(cpu_ref.esmc_swiglu_saved_sigmoid)
    self.assertFalse(cpu_ref.compile.enable)
    self.assertFalse(cpu_ref.tpu_config.use_graph_trainer)

  def test_real_esmc_g8_shapes_convert_mlp_and_qkv_projections(self):
    try:
      from transformers.models import esmc as hf_esmc  # pylint: disable=g-import-not-at-top
    except ImportError:
      return
    cfg = hf_esmc.EsmcConfig(num_hidden_layers=2)
    for key, value in config_registry._ESMC_G8_3B_OVERRIDES.items():  # pylint: disable=protected-access
      if key != "num_hidden_layers":
        setattr(cfg, key, value)
    cfg.tpu_swiglu_saved_sigmoid = True
    with torch.device("meta"):
      esmc = hf_esmc.EsmcForMaskedLM(cfg)
    tpu_model._use_tpu_esmc_attention(esmc)  # pylint: disable=protected-access
    for layer in esmc.esmc.layers:
      self.assertEqual(tuple(layer.self_attn.q_norm.normalized_shape), (2560,))
      self.assertEqual(tuple(layer.self_attn.k_norm.normalized_shape), (2560,))

    float8.apply_float8_linears(
        esmc,
        "fused",
        only_modules="gate_proj,up_proj,down_proj,q_proj,k_proj,v_proj",
    )
    converted = [
        fqn
        for fqn, m in esmc.named_modules()
        if isinstance(m, float8.Fp8FusedLinear)
    ]
    self.assertLen(converted, 12)
    self.assertEqual(
        sorted({fqn.rsplit(".", 1)[-1] for fqn in converted}),
        ["down_proj", "gate_proj", "k_proj", "q_proj", "up_proj", "v_proj"],
    )


if __name__ == "__main__":
  absltest.main()
