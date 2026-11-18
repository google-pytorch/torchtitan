# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torchtitan.experiments.tpu.components.qkv import (
    TpuPackedQKVConverter,
    TpuPackedQKVLinear,
)
from torchtitan.models.common.attention import FusedQKVLinear, ScaledDotProductInnerAttention
from torchtitan.models.common.config_utils import make_gqa_config
from torchtitan.models.common.rope import CosSinRoPE

DIM = 16
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8


def _attention_config():
  init = {"weight": torch.nn.init.normal_}
  return make_gqa_config(
      dim=DIM,
      n_heads=N_HEADS,
      n_kv_heads=N_KV_HEADS,
      head_dim=HEAD_DIM,
      wqkv_param_init=init,
      wo_param_init=init,
      inner_attention=ScaledDotProductInnerAttention.Config(),
      fuse_qkv=True,
      rope=CosSinRoPE.Config(dim=HEAD_DIM, max_context_length=16, theta=10000.0),
  )


def _build_fused_and_packed():
  config = _attention_config().qkv_linear
  fused = config.build()
  packed = (
      TpuPackedQKVConverter.Config()
      .build()
      .convert(copy.deepcopy(config))
      .build()
  )
  for module in (fused, packed):
    torch.manual_seed(1234)
    module.init_states()
  return fused, packed


def _assert_state_equal(actual, expected):
  assert actual.keys() == expected.keys()
  for key in actual:
    if isinstance(actual[key], torch.Tensor):
      torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    else:
      assert actual[key] == expected[key]


def test_converter_validates_supported_config():
  config = cast(FusedQKVLinear.Config, _attention_config().qkv_linear)
  converter = TpuPackedQKVConverter.Config().build()
  converted = converter.convert(config)
  assert isinstance(converted, TpuPackedQKVLinear.Config)
  assert (converted.n_heads, converted.n_kv_heads, converted.head_dim) == (
      N_HEADS,
      N_KV_HEADS,
      HEAD_DIM,
  )
  assert converter.convert(converted) is converted

  missing_init = replace(config, wqkv=replace(config.wqkv, param_init=None))
  with pytest.raises(
      ValueError, match="explicit fused QKV parameter initializer"
  ):
    converter.convert(missing_init)

  invalid_rows = replace(
      config,
      wqkv=replace(config.wqkv, out_features=config.wqkv.out_features + 1),
  )
  with pytest.raises(ValueError, match="wqkv.out_features"):
    converter.convert(invalid_rows).build()

  with pytest.raises(ValueError, match="supports TP=1 only"):
    converted.build().parallelize(SimpleNamespace(tp=2))


def test_packed_qkv_matches_fused_semantics_and_checkpoint():
  fused, packed = _build_fused_and_packed()
  fused_state = fused.state_dict()
  packed_state = packed.state_dict()
  _assert_state_equal(packed_state, fused_state)
  packed.load_state_dict(fused_state)
  fused.load_state_dict(packed.state_dict())
  assert [name for name, _ in packed.named_parameters()] == [
      "wqkv_packed.weight"
  ]

  fused_input = torch.randn(6, DIM, requires_grad=True)
  packed_input = fused_input.detach().clone().requires_grad_()
  fused_outputs = fused(fused_input)
  packed_outputs = packed(packed_input)
  for actual, expected in zip(packed_outputs, fused_outputs, strict=True):
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected)

  grads = tuple(torch.randn_like(output) for output in fused_outputs)
  torch.autograd.backward(fused_outputs, grads)
  torch.autograd.backward(packed_outputs, grads)
  torch.testing.assert_close(packed_input.grad, fused_input.grad)
  interleaved_grad = fused.wqkv.weight.grad.reshape(
      N_KV_HEADS, N_HEADS // N_KV_HEADS + 2, HEAD_DIM, -1
  )
  expected_grad = torch.cat((
      interleaved_grad[:, : N_HEADS // N_KV_HEADS].reshape(-1, DIM),
      interleaved_grad[:, N_HEADS // N_KV_HEADS].reshape(-1, DIM),
      interleaved_grad[:, N_HEADS // N_KV_HEADS + 1].reshape(-1, DIM),
  ))
  torch.testing.assert_close(packed.wqkv_packed.weight.grad, expected_grad)


def test_packed_qkv_backward_concatenates_one_gradient():
  _, packed = _build_fused_and_packed()
  x = torch.randn(6, DIM)

  def loss(weight):
    outputs = torch.func.functional_call(
        packed, {"wqkv_packed.weight": weight}, (x,)
    )
    return sum(output.sum() for output in outputs)

  graph = make_fx(torch.func.grad(loss))(packed.wqkv_packed.weight.detach())
  targets = [
      node.target for node in graph.graph.nodes if node.op == "call_function"
  ]
  assert targets.count(torch.ops.aten.split_with_sizes.default) == 1
  assert targets.count(torch.ops.aten.cat.default) == 1
  assert not any("slice_backward" in str(target) for target in targets)
