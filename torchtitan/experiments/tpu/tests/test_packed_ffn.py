# Copyright 2026 Google LLC

import copy
from dataclasses import replace
import socket
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import (
    set_current_spmd_mesh,
    set_spmd_meshes,
    spmd_dense_mesh,
)
from torchtitan.experiments.tpu.components.ffn import (
    TpuPackedFeedForward,
    TpuPackedFeedForwardConverter,
    TpuSingleProjPackedFeedForward,
)
from torchtitan.experiments.tpu.examples.gemma4.model import GeGLU, Gemma4FeedForward
from torchtitan.experiments.tpu.graph_trainer.config import (
    apply_tpu_graph_trainer_model_transforms,
)
from torchtitan.experiments.tpu.qwen3.config_registry import (
    graph_trainer_qwen3_debugmodel,
)
from torchtitan.models.common.config_utils import make_ffn_config
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    set_dense_ffn_sharding,
)
from torchtitan.models.common.dist_gemm import AsyncColumnParallelLinear
from torchtitan.models.common.feed_forward import (
    FeedForward,
    SigmoidGatedFeedForward,
)
from torchtitan.models.common.linear import Linear
import spmd_types as spmd

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


def _build_fused_and_packed(*, bias: bool = False):
  config = _ffn_config(bias=bias)
  fused = config.build()
  packed = (
      TpuPackedFeedForwardConverter.Config()
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


def test_converter_validates_and_filters_configs():
  config = _ffn_config()
  converter = TpuPackedFeedForwardConverter.Config().build()
  converted = converter.convert(config)
  assert isinstance(converted, TpuPackedFeedForward.Config)
  assert isinstance(converted, FeedForward.Config)
  assert converter.convert(converted) is converted

  # Gemma4FeedForward.Config only customizes activation_fn and is converted.
  gemma4_cfg = Gemma4FeedForward.Config(
      w13=config.w13,
      w2=config.w2,
      activation_fn=GeGLU.Config(),
  )
  gemma4_converted = converter.convert(gemma4_cfg)
  assert isinstance(gemma4_converted, TpuPackedFeedForward.Config)
  assert isinstance(gemma4_converted.activation_fn, GeGLU.Config)

  # SigmoidGatedFeedForward has an extra `gate` projection and is left alone.
  sigmoid_cfg = SigmoidGatedFeedForward.Config(
      w13=config.w13,
      w2=config.w2,
      gate=Linear.Config(in_features=DIM, out_features=1),
  )
  assert converter.convert(sigmoid_cfg) is sigmoid_cfg

  # Non-Linear w13 (e.g., dist_gemm or Float8Linear) is left alone.
  async_cfg = replace(
      config,
      w13=AsyncColumnParallelLinear.Config(
          in_features=DIM, out_features=2 * HIDDEN_DIM
      ),
  )
  assert converter.convert(async_cfg) is async_cfg

  odd_rows = replace(
      config,
      w13=replace(config.w13, out_features=2 * HIDDEN_DIM + 1),
  )
  with pytest.raises(ValueError, match="w13.out_features"):
    converter.convert(odd_rows).build()

  with pytest.raises(ValueError, match="tp_degree=1"):
    converted.build().parallelize(SimpleNamespace(tp=2))

  tp2_converter = TpuPackedFeedForwardConverter.Config(tp_degree=2).build()
  tp2_converted = tp2_converter.convert(copy.deepcopy(config))
  assert isinstance(tp2_converted, TpuPackedFeedForward.Config)
  assert tp2_converted.tp_degree == 2


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(6, DIM), (2, 3, DIM)])
def test_packed_ffn_matches_fused_semantics_and_checkpoint(bias, shape):
  fused, packed = _build_fused_and_packed(bias=bias)
  fused_state = fused.state_dict()
  packed_state = packed.state_dict()
  _assert_state_equal(packed_state, fused_state)
  packed.load_state_dict(fused_state)
  fused.load_state_dict(packed.state_dict())

  param_names = [name for name, _ in packed.named_parameters()]
  expected_params = ["w13.weight", "w2.weight"]
  if bias:
    expected_params = ["w13.weight", "w13.bias", "w2.weight", "w2.bias"]
  assert param_names == expected_params

  fused_input = torch.randn(*shape, requires_grad=True)
  packed_input = fused_input.detach().clone().requires_grad_()
  fused_output = fused(fused_input)
  packed_output = packed(packed_input)
  assert packed_output.is_contiguous()
  torch.testing.assert_close(packed_output, fused_output)

  grad = torch.randn_like(fused_output)
  fused_output.backward(grad)
  packed_output.backward(grad)
  torch.testing.assert_close(
      packed_input.grad, fused_input.grad, atol=1e-4, rtol=1e-4
  )
  torch.testing.assert_close(packed.w2.weight.grad, fused.w2.weight.grad)

  interleaved_grad = fused.w13.weight.grad.unflatten(0, (-1, 2))
  expected_w13_grad = torch.cat(
      (interleaved_grad[:, 0], interleaved_grad[:, 1]), dim=0
  )
  torch.testing.assert_close(packed.w13.weight.grad, expected_w13_grad)


def test_packed_ffn_backward_concatenates_one_gradient():
  _, packed = _build_fused_and_packed()
  x = torch.randn(6, DIM)

  def loss(weight):
    out = torch.func.functional_call(packed, {"w13.weight": weight}, (x,))
    return out.sum()

  graph = make_fx(torch.func.grad(loss))(packed.w13.weight.detach())
  targets = [
      node.target for node in graph.graph.nodes if node.op == "call_function"
  ]
  assert targets.count(torch.ops.aten.split.Tensor) == 1
  assert targets.count(torch.ops.aten.cat.default) == 1
  assert not any("slice_backward" in str(target) for target in targets)


def test_packed_ffn_tp_degree_sharded_layout_and_checkpoint():
  config = _ffn_config(bias=True)
  fused = config.build()
  tp2_packed = (
      TpuPackedFeedForwardConverter.Config(tp_degree=2)
      .build()
      .convert(copy.deepcopy(config))
      .build()
  )
  for module in (fused, tp2_packed):
    torch.manual_seed(1234)
    module.init_states()

  fused_state = fused.state_dict()
  packed_state = tp2_packed.state_dict()
  _assert_state_equal(packed_state, fused_state)
  tp2_packed.load_state_dict(fused_state)
  _assert_state_equal(tp2_packed.state_dict(), fused_state)

  # Each Shard(0) half of w13.weight holds [w1_shard_r; w3_shard_r].
  w1 = fused_state["w1.weight"]
  w3 = fused_state["w3.weight"]
  half = HIDDEN_DIM // 2
  shard0, shard1 = torch.chunk(tp2_packed.w13.weight, 2, dim=0)
  torch.testing.assert_close(shard0, torch.cat((w1[:half], w3[:half]), dim=0))
  torch.testing.assert_close(shard1, torch.cat((w1[half:], w3[half:]), dim=0))


def test_graph_trainer_transforms_convert_ffn_for_all_tp():
  cfg = graph_trainer_qwen3_debugmodel()
  cfg.parallelism.tensor_parallel_degree = 1
  apply_tpu_graph_trainer_model_transforms(cfg)
  ffns_tp1 = list(cfg.model_spec.model.traverse(TpuPackedFeedForward.Config))
  assert ffns_tp1 and all(c.tp_degree == 1 for _, c, _, _ in ffns_tp1)

  cfg_tp2 = graph_trainer_qwen3_debugmodel()
  cfg_tp2.parallelism.tensor_parallel_degree = 2
  apply_tpu_graph_trainer_model_transforms(cfg_tp2)
  ffns_tp2 = list(
      cfg_tp2.model_spec.model.traverse(TpuPackedFeedForward.Config)
  )
  assert ffns_tp2 and all(c.tp_degree == 2 for _, c, _, _ in ffns_tp2)


def test_graph_trainer_transforms_use_single_proj_packed_ffn_when_scan_enabled():
  cfg_non_scan = graph_trainer_qwen3_debugmodel()
  apply_tpu_graph_trainer_model_transforms(cfg_non_scan)
  assert list(
      cfg_non_scan.model_spec.model.traverse(TpuPackedFeedForward.Config)
  )
  assert not list(
      cfg_non_scan.model_spec.model.traverse(
          TpuSingleProjPackedFeedForward.Config
      )
  )

  cfg_scan = graph_trainer_qwen3_debugmodel()
  cfg_scan.compile.use_scan_layers = True
  cfg_scan.compile.memory_policy = "full_and_scan_offload"
  apply_tpu_graph_trainer_model_transforms(cfg_scan)
  assert list(
      cfg_scan.model_spec.model.traverse(TpuSingleProjPackedFeedForward.Config)
  )
  model = cfg_scan.model_spec.model.build()
  ffns = [m for m in model.modules() if isinstance(m, TpuSingleProjPackedFeedForward)]
  assert ffns and all(type(m) is TpuSingleProjPackedFeedForward for m in ffns)


def test_single_proj_packed_ffn_forward_and_backward_parity():
  config = _ffn_config(bias=True)
  fused = config.build()
  single_proj = (
      TpuPackedFeedForwardConverter.Config(single_projection=True)
      .build()
      .convert(copy.deepcopy(config))
      .build()
  )
  assert type(single_proj) is TpuSingleProjPackedFeedForward
  for module in (fused, single_proj):
    torch.manual_seed(1234)
    module.init_states()

  _assert_state_equal(single_proj.state_dict(), fused.state_dict())

  torch.manual_seed(5678)
  x = torch.randn(2, 6, DIM, requires_grad=True)
  x_single = x.detach().clone().requires_grad_()

  fused_out = fused(x)
  single_out = single_proj(x_single)
  torch.testing.assert_close(single_out, fused_out)

  fused_out.sum().backward()
  single_out.sum().backward()
  torch.testing.assert_close(x_single.grad, x.grad, atol=1e-4, rtol=1e-4)

  w1_grad_ref, w3_grad_ref = fused.w13.weight.grad.view(-1, 2, DIM).unbind(
      dim=1
  )
  w1_grad, w3_grad = torch.chunk(single_proj.w13.weight.grad, 2, dim=0)
  torch.testing.assert_close(w1_grad, w1_grad_ref, atol=1e-4, rtol=1e-4)
  torch.testing.assert_close(w3_grad, w3_grad_ref, atol=1e-4, rtol=1e-4)
  torch.testing.assert_close(
      single_proj.w2.weight.grad, fused.w2.weight.grad, atol=1e-4, rtol=1e-4
  )

  def loss(w13_weight):
    gate_up = torch.nn.functional.linear(x.detach(), w13_weight, single_proj.w13.bias)
    gate, up = torch.chunk(gate_up, 2, dim=-1)
    return single_proj.w2(single_proj.activation_fn(gate, up)).sum()

  graph = make_fx(torch.func.grad(loss))(single_proj.w13.weight.detach())
  targets = [
      node.target for node in graph.graph.nodes if node.op == "call_function"
  ]
  # 1 fwd addmm + 1 bwd mm for w13, plus 1 fwd addmm + 1 bwd mm for w2.
  assert targets.count(torch.ops.aten.addmm.default) == 2
  assert targets.count(torch.ops.aten.mm.default) == 2
  assert targets.count(torch.ops.aten.split.Tensor) == 1
  assert not any("slice_backward" in str(target) for target in targets)


def test_single_proj_packed_ffn_under_layer_stack():
  from torchtitan.experiments.tpu.graph_trainer.layer_stack import (
      LayerStack,
      LayerStackHolder,
  )

  cfg = _ffn_config(bias=True)
  single_cfg = (
      TpuPackedFeedForwardConverter.Config(single_projection=True)
      .build()
      .convert(copy.deepcopy(cfg))
  )
  ref_layers = torch.nn.ModuleList([cfg.build(), cfg.build()])
  exp_layers = torch.nn.ModuleDict(
      {"0": single_cfg.build(), "1": single_cfg.build()}
  )
  holder = LayerStackHolder()
  stack = LayerStack(exp_layers, holder)
  stack.unregister_and_allocate()

  for i, key in enumerate(("0", "1")):
    torch.manual_seed(100 + i)
    ref_layers[i].init_states()
  with stack.unrolled():
    for i, key in enumerate(("0", "1")):
      torch.manual_seed(100 + i)
      exp_layers[key].init_states()

  torch.manual_seed(42)
  x_ref = torch.randn(2, 6, DIM, requires_grad=True)
  x_exp = x_ref.detach().clone().requires_grad_()

  y_ref = x_ref
  for layer in ref_layers:
    y_ref = layer(y_ref)
  y_exp = stack.run(x_exp, use_scan=True)
  torch.testing.assert_close(y_exp, y_ref, atol=1e-4, rtol=1e-4)

  y_ref.sum().backward()
  y_exp.sum().backward()
  torch.testing.assert_close(x_exp.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
  assert all(p.grad is not None for p in holder.parameters())


def _find_free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _tp2_dtensor_worker(rank: int, port: int, x_data: torch.Tensor) -> None:
  dist.init_process_group(
      backend="gloo",
      init_method=f"tcp://127.0.0.1:{port}",
      rank=rank,
      world_size=2,
  )
  try:
    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1,
        cp=1,
        tp=2,
        pp=1,
        ep=1,
        world_size=2,
    )
    init = {"weight": torch.nn.init.normal_}
    cfg = _ffn_config(bias=False)
    torch.manual_seed(1234)
    fused_ref = cfg.build()
    fused_ref.init_states()

    packed_cfg = TpuPackedFeedForwardConverter.Config(tp_degree=2).build().convert(
        copy.deepcopy(cfg)
    )
    set_dense_ffn_sharding(
        packed_cfg,
        attn_x_layout=dense_activation_placement(tp=spmd.I, cp=spmd.S(0)),
        enable_sp=False,
    )
    packed = packed_cfg.build()
    packed.load_state_dict(fused_ref.state_dict())

    parallel_dims.build_mesh()
    set_spmd_meshes(
        dense_mesh=parallel_dims.spmd_dense_mesh(),
        sparse_mesh=parallel_dims.spmd_sparse_mesh(),
    )
    dense_mesh = spmd_dense_mesh()
    with set_current_spmd_mesh(dense_mesh):
      packed.parallelize(parallel_dims)

      x_ref = x_data.detach().clone().requires_grad_()
      x_packed = x_data.detach().clone().requires_grad_()
      ref_out = fused_ref(x_ref)
      packed_out = packed(x_packed)
      torch.testing.assert_close(packed_out, ref_out)

      ref_out.sum().backward()
      packed_out.sum().backward()
      torch.testing.assert_close(x_packed.grad, x_ref.grad, atol=1e-4, rtol=1e-4)
  finally:
    dist.destroy_process_group()


def test_packed_ffn_tp2_dtensor_forward_and_backward():
  torch.manual_seed(7)
  x_data = torch.randn(4, DIM)
  port = _find_free_port()
  mp.spawn(_tp2_dtensor_worker, args=(port, x_data), nprocs=2, join=True)
