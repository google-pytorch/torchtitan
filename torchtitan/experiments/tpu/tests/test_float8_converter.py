"""Tests for `TpuFloat8LinearConverter`."""

from absl.testing import absltest
import torch
from torch.nn import functional as F
from torchao.float8.config import Float8LinearConfig
from torchao.float8.config import ScalingGranularity
from torchtitan.components.quantization.float8 import Float8Linear
from torchtitan.components.quantization.float8 import Float8LinearConverter
from torchtitan.experiments.tpu.float8_converter import TpuFloat8LinearConverter
from torchtitan.experiments.tpu.llama3 import config_registry
from torchtitan.experiments.tpu.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import model_registry


def _converted(model_config) -> tuple[list[str], list[str]]:
  """FQNs of (Float8Linear, plain Linear) configs in `model_config`."""
  fp8, plain = [], []
  for fqn, linear_config, _, _ in model_config.traverse(Linear.Config):
    (fp8 if isinstance(linear_config, Float8Linear.Config) else plain).append(
        fqn
    )
  return fp8, plain


class TpuFloat8LinearConverterTest(absltest.TestCase):

  def test_rowwise_recipe(self):
    """Verifies that the rowwise recipe builds the upstream torchao config."""
    converter = TpuFloat8LinearConverter.Config(recipe_name="rowwise").build()

    self.assertIsInstance(converter, TpuFloat8LinearConverter)
    expected = Float8LinearConfig.from_recipe_name("rowwise")
    self.assertEqual(converter.torchao_config, expected)
    config = converter.torchao_config
    for cast_config in (
        config.cast_config_input,
        config.cast_config_weight,
        config.cast_config_grad_output,
    ):
      self.assertEqual(
          cast_config.scaling_granularity, ScalingGranularity.AXISWISE
      )
    self.assertTrue(config.round_scales_to_power_of_2)
    self.assertTrue(config.gemm_config_output.use_fast_accum)
    self.assertFalse(config.gemm_config_grad_input.use_fast_accum)
    self.assertFalse(config.gemm_config_grad_weight.use_fast_accum)

  def test_builds_without_cuda(self):
    # The upstream converter refuses to build without NVIDIA SM89+/AMD MI300.
    with self.assertRaisesRegex(ValueError, "only supported on"):
      Float8LinearConverter.Config(recipe_name="rowwise").build()
    self.assertTrue(
        TpuFloat8LinearConverter.Config(recipe_name="rowwise").build().enabled
    )

  def test_rejects_auto_filter(self):
    with self.assertRaisesRegex(ValueError, "auto_filter_small_kn"):
      TpuFloat8LinearConverter.Config(
          filter_fqns=["auto_filter_small_kn"]
      ).build()

  def test_converts_all_linears_except_filtered(self):
    spec = model_registry(
        "debugmodel",
        attn_backend="sdpa",
        converters=[TpuFloat8LinearConverter.Config(filter_fqns=["lm_head"])],
    )

    fp8, plain = _converted(spec.model)
    self.assertEqual(plain, ["lm_head"])
    self.assertNotEmpty(fp8)
    model = spec.model.build()
    for fqn in fp8:
      module = model.get_submodule(fqn)
      self.assertIsInstance(module, Float8Linear)
      self.assertTrue(module.config.gemm_config_output.use_fast_accum)

  def test_emulated_layer_matches_linear(self):
    torch.manual_seed(0)
    converter = TpuFloat8LinearConverter.Config(emulate=True).build()
    layer = converter.convert(
        Linear.Config(in_features=128, out_features=64)
    ).build()
    self.assertIsInstance(layer, Float8Linear)
    torch.nn.init.normal_(layer.weight, std=0.02)
    x = torch.randn(32, 128, requires_grad=True)

    y = layer(x)
    y.sum().backward()

    y_ref = F.linear(x, layer.weight)
    rel_err = (y - y_ref).norm() / y_ref.norm()
    self.assertLess(rel_err.item(), 0.1)
    self.assertIsNotNone(x.grad)
    self.assertIsNotNone(layer.weight.grad)

  def test_llama3_70b_float8_config(self):
    config = config_registry.llama3_70b_float8()

    fp8, plain = _converted(config.model_spec.model)
    self.assertEqual(plain, ["lm_head"])
    # wqkv, wo, w1, w2 and w3 in each of the 80 layers.
    self.assertLen(fp8, 5 * 80)
    self.assertIs(config.model_spec.parallelize_fn, parallelize_llama)


if __name__ == "__main__":
  absltest.main()
