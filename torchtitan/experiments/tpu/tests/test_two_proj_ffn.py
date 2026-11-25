# Copyright 2026 Google LLC

import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from torchtitan.experiments.tpu.components.ffn import (
    TpuPackedFeedForward,
    TpuTwoProjFeedForward,
    TpuTwoProjFeedForwardConverter,
)
from torchtitan.experiments.tpu.examples.gemma4.model import GeGLU, Gemma4FeedForward
from torchtitan.experiments.tpu.graph_trainer.config import (
    apply_tpu_graph_trainer_model_transforms,
)
from torchtitan.experiments.tpu.qwen3.config_registry import (
    graph_trainer_qwen3_debugmodel,
)
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.dist_gemm import AsyncColumnParallelLinear
from torchtitan.models.common.feed_forward import (
    FeedForward,
    SigmoidGatedFeedForward,
)
from torchtitan.models.common.linear import Linear

DIM = 16
HIDDEN_DIM = 32


def _ffn_config(*, bias: bool = False) -> FeedForward.Config:
  init = {"weight": torch.nn.init.normal_}
  cfg = make_ffn_config(
      dim=DIM,
      hidden_dim=HIDDEN_DIM,
      w1_param_init=init,
      w2w3_param_init=init,
  )
  if bias:
    cfg = replace(
        cfg,
        w13=replace(
            cfg.w13,
            bias=True,
            param_init={
                **cast(dict, cfg.w13.param_init),
                "bias": torch.nn.init.normal_,
            },
        ),
        w2=replace(
            cfg.w2,
            bias=True,
            param_init={
                **cast(dict, cfg.w2.param_init),
                "bias": torch.nn.init.normal_,
            },
        ),
    )
  return cfg


def _build_upstream_and_two_proj(*, bias: bool = False, packed: bool = False):
  config = _ffn_config(bias=bias)
  upstream = config.build()
  two_proj = (
      TpuTwoProjFeedForwardConverter.Config(packed=packed)
      .build()
      .convert(copy.deepcopy(config))
      .build()
  )
  for module in (upstream, two_proj):
    torch.manual_seed(1234)
    module.init_states()
  return upstream, two_proj


def test_converter_filters_configs():
  config = _ffn_config()
  converter = TpuTwoProjFeedForwardConverter.Config().build()
  converted = converter.convert(config)
  assert type(converted) is TpuTwoProjFeedForward.Config
  assert isinstance(converted, FeedForward.Config)
  # The stacked projection config is upstream's, untouched.
  assert converted.w13 is config.w13
  assert converted.w13.num_linears == 2
  assert converter.convert(converted) is converted
  assert type(converted.build()) is TpuTwoProjFeedForward

  # Gemma4FeedForward.Config only customizes activation_fn and is converted.
  gemma4_cfg = Gemma4FeedForward.Config(
      w13=config.w13,
      w2=config.w2,
      activation_fn=GeGLU.Config(),
  )
  gemma4_converted = converter.convert(gemma4_cfg)
  assert type(gemma4_converted) is TpuTwoProjFeedForward.Config
  assert isinstance(gemma4_converted.activation_fn, GeGLU.Config)

  # SigmoidGatedFeedForward has an extra `gate` projection and is left alone.
  sigmoid_cfg = SigmoidGatedFeedForward.Config(
      w13=config.w13,
      w2=config.w2,
      gate=Linear.Config(in_features=DIM, out_features=1),
  )
  assert converter.convert(sigmoid_cfg) is sigmoid_cfg

  # A non-Linear w13 (e.g. dist_gemm) owns its GEMM and is left alone.
  async_cfg = replace(
      config,
      w13=AsyncColumnParallelLinear.Config(
          in_features=DIM, out_features=HIDDEN_DIM, num_linears=2
      ),
  )
  assert converter.convert(async_cfg) is async_cfg


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(6, DIM), (2, 3, DIM)])
def test_two_proj_ffn_matches_upstream(bias, shape):
  upstream, two_proj = _build_upstream_and_two_proj(bias=bias)

  # Same parameters and state dict as upstream: the stacked [2, F, D] w13.
  expected_params = ["w13.weight", "w2.weight"]
  if bias:
    expected_params = ["w13.weight", "w13.bias", "w2.weight", "w2.bias"]
  assert [name for name, _ in two_proj.named_parameters()] == expected_params
  assert tuple(two_proj.w13.weight.shape) == (2, HIDDEN_DIM, DIM)
  upstream_state = upstream.state_dict()
  two_proj_state = two_proj.state_dict()
  assert two_proj_state.keys() == upstream_state.keys()
  for key, value in upstream_state.items():
    torch.testing.assert_close(two_proj_state[key], value, rtol=0, atol=0)
  two_proj.load_state_dict(upstream_state)
  upstream.load_state_dict(two_proj.state_dict())

  upstream_input = torch.randn(*shape, requires_grad=True)
  two_proj_input = upstream_input.detach().clone().requires_grad_()
  upstream_output = upstream(upstream_input)
  two_proj_output = two_proj(two_proj_input)
  torch.testing.assert_close(
      two_proj_output, upstream_output, atol=1e-4, rtol=1e-4
  )

  grad = torch.randn_like(upstream_output)
  upstream_output.backward(grad)
  two_proj_output.backward(grad)
  torch.testing.assert_close(
      two_proj_input.grad, upstream_input.grad, atol=1e-4, rtol=1e-4
  )
  for name, param in two_proj.named_parameters():
    torch.testing.assert_close(
        param.grad,
        upstream.get_parameter(name).grad,
        atol=1e-4,
        rtol=1e-4,
    )


def test_two_proj_ffn_reads_the_weight_once():
  _, two_proj = _build_upstream_and_two_proj()
  reads = 0
  weight = two_proj.w13.weight

  class _CountingLinear(type(two_proj.w13)):

    @property
    def weight(self):
      nonlocal reads
      reads += 1
      return weight

  two_proj.w13.__class__ = _CountingLinear
  two_proj(torch.randn(6, DIM))
  # One read is one SimpleFSDP all-gather of w13 per call.
  assert reads == 1


def test_packed_converter_selects_the_packed_class():
  config = _ffn_config()
  converter = TpuTwoProjFeedForwardConverter.Config(packed=True).build()
  converted = converter.convert(config)
  assert type(converted) is TpuPackedFeedForward.Config
  # The config tree keeps upstream's stacked projection config; only the built
  # parameter is packed.
  assert converted.w13 is config.w13
  assert converter.convert(converted) is converted
  module = converted.build()
  assert type(module) is TpuPackedFeedForward
  assert tuple(module.w13.weight.shape) == (2 * HIDDEN_DIM, DIM)
  with pytest.raises(ValueError, match="tensor parallelism"):
    converted.build()._parallelize(SimpleNamespace(tp=2))


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(6, DIM), (2, 3, DIM)])
def test_packed_ffn_matches_upstream(bias, shape):
  upstream, packed = _build_upstream_and_two_proj(bias=bias, packed=True)

  # Physically [2F, D] (FSDP shards dim 0); the state dict is upstream's
  # stacked [2, F, D], with the same values for the same seed.
  expected_params = ["w13.weight", "w2.weight"]
  if bias:
    expected_params = ["w13.weight", "w13.bias", "w2.weight", "w2.bias"]
  assert [name for name, _ in packed.named_parameters()] == expected_params
  assert tuple(packed.w13.weight.shape) == (2 * HIDDEN_DIM, DIM)
  upstream_state = upstream.state_dict()
  packed_state = packed.state_dict()
  assert packed_state.keys() == upstream_state.keys()
  for key, value in upstream_state.items():
    assert packed_state[key].shape == value.shape, key
    torch.testing.assert_close(packed_state[key], value, rtol=0, atol=0)

  # Round trip through upstream's layout, in both directions.
  for value in upstream_state.values():
    torch.nn.init.normal_(value)
  packed.load_state_dict(upstream_state)
  torch.testing.assert_close(
      packed.w13.weight, upstream.w13.weight.flatten(0, 1), rtol=0, atol=0
  )
  upstream.load_state_dict(packed.state_dict())

  upstream_input = torch.randn(*shape, requires_grad=True)
  packed_input = upstream_input.detach().clone().requires_grad_()
  upstream_output = upstream(upstream_input)
  packed_output = packed(packed_input)
  torch.testing.assert_close(
      packed_output, upstream_output, atol=1e-4, rtol=1e-4
  )

  grad = torch.randn_like(upstream_output)
  upstream_output.backward(grad)
  packed_output.backward(grad)
  torch.testing.assert_close(
      packed_input.grad, upstream_input.grad, atol=1e-4, rtol=1e-4
  )
  for name, param in packed.named_parameters():
    torch.testing.assert_close(
        param.grad,
        upstream.get_parameter(name).grad.reshape(param.shape),
        atol=1e-4,
        rtol=1e-4,
    )


def test_graph_trainer_packs_only_non_scan_zero3_without_tp():
  def converted(*, reshard: str, use_scan: bool = False, tp: int = 1):
    cfg = graph_trainer_qwen3_debugmodel()
    cfg.parallelism.fsdp_reshard_after_forward = reshard
    cfg.parallelism.tensor_parallel_degree = tp
    if use_scan:
      cfg.compile.use_scan_layers = True
      cfg.compile.memory_policy = "full_and_scan_offload"
    elif reshard == "always":
      # Non-scan Zero-3 needs an FSDP barrier policy (as qwen3_8b_zero3).
      cfg.compile.memory_policy = "full"
      cfg.compile.enable_remat_barriers = True
      cfg.compile.fsdp_barrier_policy = "interleave"
    apply_tpu_graph_trainer_model_transforms(cfg)
    assert list(cfg.model.traverse(FeedForward.Config))
    return [
        type(c)
        for _, c, _, _ in cfg.model.traverse(
            TpuTwoProjFeedForward.Config
        )
    ]

  zero3 = converted(reshard="always")
  assert zero3 and all(t is TpuPackedFeedForward.Config for t in zero3)
  # Zero-2, scan layers and tensor parallelism keep upstream's FeedForward.
  assert not converted(reshard="default")
  assert not converted(reshard="never")
  assert not converted(reshard="always", use_scan=True)
  assert not converted(reshard="always", tp=2)
