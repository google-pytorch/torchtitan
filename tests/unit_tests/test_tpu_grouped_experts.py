from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
# With torch_tpu, torch must be imported before spmd_types to avoid a circular
# import error. Import of torch autoloads torch_tpu which imports
# torch.distributed.fsdp which imports spmd_types still initializing.
import torch  # isort: skip
import spmd_types as spmd

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.experiments.tpu.grouped_experts import (
    _native_weight_sharding,
    _transpose_last_two_dims,
    TpuGroupedExperts,
    TpuGroupedExpertsConverter,
)
from torchtitan.experiments.tpu.graph_trainer import qwen3 as graph_trainer_qwen3
from torchtitan.experiments.tpu.kernels import expert_segment_matmul
from torchtitan.experiments.tpu.qwen3.config_registry import (
    qwen3_moe_30b_fsdp_only_full_ac_benchmark,
    qwen3_moe_30b_fsdp_only_full_ac_profile,
)
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.moe_sharding import expert_param_placement_dense
from torchtitan.protocols.sharding import ShardingConfig


def _build_experts(
    num_experts=1,
    *,
    use_pallas_grouped_mm=False,
    use_pallas_paired_grouped_mm=False,
    use_kernel_native_weight_layout=False,
    param_init=None,
):
    config = TpuGroupedExperts.Config(
        dim=16,
        hidden_dim=24,
        num_experts=num_experts,
        use_pallas_grouped_mm=use_pallas_grouped_mm,
        use_pallas_paired_grouped_mm=use_pallas_paired_grouped_mm,
        pallas_grouped_mm_tgmm_tiling=(
            (128, 0, 0) if use_pallas_paired_grouped_mm else (128, 128, 128)
        ),
        use_kernel_native_weight_layout=use_kernel_native_weight_layout,
        param_init=param_init,
    )
    experts = config.build()
    if param_init is None:
        for param in experts.parameters():
            torch.nn.init.normal_(param)
    return experts


def test_model_converter_preserves_config_fields():
    class RootConfig:
        def __init__(self):
            self.experts = GroupedExperts.Config(
                dim=16,
                hidden_dim=24,
                num_experts=8,
            )

        def traverse(self, config_type):
            if isinstance(self.experts, config_type):
                yield "experts", self.experts, self, "experts"

    root = RootConfig()
    original = root.experts
    with patch("torchtitan.experiments.tpu.grouped_experts.logger.info") as log_info:
        converted = TpuGroupedExpertsConverter.Config().build().convert(root)

    assert converted is root
    assert isinstance(root.experts, TpuGroupedExperts.Config)
    assert root.experts.dim == original.dim
    assert root.experts.hidden_dim == original.hidden_dim
    assert root.experts.num_experts == original.num_experts
    assert not root.experts.use_pallas_paired_grouped_mm
    assert root.experts.pallas_grouped_mm_tile_k == 0
    assert root.experts.pallas_grouped_mm_tile_n == 0
    assert root.experts.pallas_grouped_mm_tgmm_tiling == (128, 128, 128)
    log_info.assert_called_once_with(
        "Replaced %d GroupedExperts configuration(s) for TPU", 1
    )


def test_fsdp_recipe_preserves_global_aux_contract_and_selects_tpu_experts():
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()

    assert config.model_spec is not None
    assert config.model_spec.flavor == "30B-A3B-global-aux"
    assert config.metrics.enable_moe_routing
    assert config.training.mixed_precision_reduce == "float32"
    assert config.optimizer.param_groups[0].optimizer_kwargs["lr"] == 3e-4
    experts = list(config.model_spec.model.traverse(TpuGroupedExperts.Config))
    assert len(experts) == 48
    assert all(expert.use_pallas_grouped_mm for _, expert, _, _ in experts)
    assert all(
        expert.use_pallas_paired_grouped_mm for _, expert, _, _ in experts
    )
    assert all(expert.pallas_grouped_mm_tile_m == 256 for _, expert, _, _ in experts)
    assert all(
        expert.pallas_grouped_mm_tgmm_tiling == (256, 0, 0)
        for _, expert, _, _ in experts
    )
    assert all(expert.use_kernel_native_weight_layout for _, expert, _, _ in experts)

    benchmark = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
    assert config.profiler.enable_profiling
    assert not benchmark.profiler.enable_profiling
    assert benchmark.tpu_config.enable_mfu_benchmark
    assert benchmark.tpu_config.mfu_benchmark_start_step == 10


@pytest.mark.parametrize("weight_dtype", [jnp.bfloat16, jnp.float32])
def test_pallas_dweight_is_stored_in_the_weight_dtype(weight_dtype):
    backward = expert_segment_matmul._grouped_mm_bwd._init_fn.jit_fn
    grad_lhs, grad_rhs = jax.eval_shape(
        backward,
        jax.ShapeDtypeStruct((128, 64), weight_dtype),
        jax.ShapeDtypeStruct((128, 32), weight_dtype),
        jax.ShapeDtypeStruct((2, 32, 64), weight_dtype),
        jax.ShapeDtypeStruct((2,), jnp.int32),
        128,
        0,
        0,
        128,
        32,
        64,
        False,
    )

    assert grad_lhs.shape == (128, 32)
    assert grad_lhs.dtype == weight_dtype
    assert grad_rhs.shape == (2, 32, 64)
    assert grad_rhs.dtype == weight_dtype


@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32])
def test_unpaired_kernels_get_bf16_weights_whatever_the_param_dtype(param_dtype):
    """The dWeight store follows the kernel's weight dtype, which is always BF16.

    Neither ``mixed_precision_param`` nor ``mixed_precision_reduce`` reaches the
    kernel: the experts cast to BF16 themselves, so the gradient autograd wants
    is BF16 in every configuration.
    """
    experts = _build_experts(num_experts=2, use_pallas_grouped_mm=True).to(
        param_dtype
    )
    counts = torch.tensor([3, 5], dtype=torch.int32)
    x = torch.randn(8, experts.w1_EFD.shape[-1], dtype=param_dtype)

    def grouped_mm(x, weight, counts, **kwargs):
        del kwargs
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        return torch._grouped_mm(x, weight, offs=offsets)

    with patch(
        "torchtitan.experiments.tpu.grouped_experts.segment_matmul_pallas_2d",
        side_effect=grouped_mm,
    ) as unpaired:
        out = experts.forward(x, counts)

    assert out.dtype == param_dtype
    assert unpaired.call_count == 3
    for call in unpaired.call_args_list:
        assert call.args[0].dtype == torch.bfloat16
        assert call.args[1].dtype == torch.bfloat16
        assert "dweight_output_dtype" not in call.kwargs


def test_graph_trainer_parallelize_forwards_to_the_graph_trainer():
    experts = _build_experts(
        use_pallas_grouped_mm=True,
    )
    model = torch.nn.Sequential(experts)
    training = SimpleNamespace(
        mixed_precision_param="bfloat16", mixed_precision_reduce="float32"
    )
    parallel_dims = SimpleNamespace(dp_shard=1, cp=1, ep_enabled=False)

    with patch.object(
        graph_trainer_qwen3,
        "graph_trainer_parallelize_qwen3",
        return_value=model,
    ) as delegate:
        result = graph_trainer_qwen3.parallelize_qwen3(
            model,
            training=training,
            parallel_dims=parallel_dims,
            marker="forwarded",
        )

    assert result is model
    delegate.assert_called_once_with(
        model,
        training=training,
        parallel_dims=parallel_dims,
        marker="forwarded",
    )


def test_kernel_native_layout_requires_pallas():
    with pytest.raises(ValueError, match="Kernel-native expert weights require Pallas"):
        _build_experts(use_kernel_native_weight_layout=True)


def test_paired_grouped_mm_validates_supported_configuration():
    with pytest.raises(ValueError, match="requires Pallas"):
        _build_experts(use_pallas_paired_grouped_mm=True)

    config = TpuGroupedExperts.Config(
        dim=16,
        hidden_dim=24,
        num_experts=1,
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=True,
    )
    with pytest.raises(ValueError, match="requires a full N TGMM tile"):
        config.build()

    # A K tile is allowed: the paired dWeight kernel tiles K on its own grid axis.
    TpuGroupedExperts.Config(
        dim=16,
        hidden_dim=24,
        num_experts=1,
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=True,
        pallas_grouped_mm_tgmm_tiling=(128, 128, 0),
    ).build()

    experts = _build_experts(
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=True,
    )

    class ParallelDimsWithEp:
        @staticmethod
        def get_optional_mesh(name):
            return object() if name == "ep" else None

    with pytest.raises(ValueError, match="requires EP=1 exact-row dispatch"):
        experts.parallelize(ParallelDimsWithEp())


def test_paired_grouped_mm_forwards_tgmm_tile_k():
    """The K tile configured on the experts must reach the paired kernel.

    Construction and the kernel's own tests each pass on their own if this
    wiring is lost; only a forward with the kernel stubbed out can see the
    keyword the experts actually hand over.
    """
    experts = TpuGroupedExperts.Config(
        dim=16,
        hidden_dim=24,
        num_experts=2,
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=True,
        pallas_grouped_mm_tgmm_tiling=(128, 8, 0),
    ).build()
    for param in experts.parameters():
        torch.nn.init.normal_(param, std=0.02)
    x = torch.randn(7, 16)
    counts = torch.tensor([3, 4], dtype=torch.int32)

    def grouped_mm(x, weight, counts, **kwargs):
        del kwargs
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        return torch._grouped_mm(x, weight, offs=offsets)

    def paired_grouped_mm(x, w1, w3, counts, **kwargs):
        del kwargs
        return grouped_mm(x, w1, counts), grouped_mm(x, w3, counts)

    with patch(
        "torchtitan.experiments.tpu.grouped_experts.segment_matmul_pallas_2d",
        side_effect=grouped_mm,
    ) as unpaired, patch(
        "torchtitan.experiments.tpu.grouped_experts.paired_segment_matmul_pallas_2d",
        side_effect=paired_grouped_mm,
    ) as paired:
        experts.forward(x, counts)

    paired.assert_called_once()
    assert paired.call_args.kwargs["tgmm_tile_m"] == 128
    assert paired.call_args.kwargs["tgmm_tile_k"] == 8
    # The down projection stays on the unpaired kernel and gets the full tuple.
    unpaired.assert_called_once()
    assert unpaired.call_args.kwargs["tgmm_tiling"] == (128, 8, 0)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("paired", [True, False])
def test_w2_tgmm_tiling_reaches_only_the_w2_dweight(paired, transposed):
    """``pallas_grouped_mm_w2_tgmm_tiling`` tiles W2 and nothing else.

    Its N tile is non-zero here: the full-N rule is the paired kernel's, and W2
    is never paired. The transposed-RHS tiling replaces it only when the W2 RHS
    is transposed.
    """
    shared, w2_own, w2_transposed = (128, 8, 0), (64, 4, 4), (32, 0, 8)
    experts = TpuGroupedExperts.Config(
        dim=16,
        hidden_dim=24,
        num_experts=2,
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=paired,
        pallas_grouped_mm_tgmm_tiling=shared,
        pallas_grouped_mm_w2_tgmm_tiling=w2_own,
        pallas_grouped_mm_w2_transposed_tgmm_tiling=w2_transposed,
    ).build()
    experts._use_transposed_w2_rhs = transposed
    for param in experts.parameters():
        torch.nn.init.normal_(param, std=0.02)
    x = torch.randn(7, 16)
    counts = torch.tensor([3, 4], dtype=torch.int32)

    def grouped_mm(x, weight, counts, **kwargs):
        if kwargs.get("transpose_rhs"):
            weight = weight.transpose(-2, -1)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        return torch._grouped_mm(x, weight, offs=offsets)

    def paired_grouped_mm(x, w1, w3, counts, **kwargs):
        del kwargs
        return grouped_mm(x, w1, counts), grouped_mm(x, w3, counts)

    with patch(
        "torchtitan.experiments.tpu.grouped_experts.segment_matmul_pallas_2d",
        side_effect=grouped_mm,
    ) as unpaired, patch(
        "torchtitan.experiments.tpu.grouped_experts.paired_segment_matmul_pallas_2d",
        side_effect=paired_grouped_mm,
    ) as paired_kernel:
        experts.forward(x, counts)

    if paired:
        paired_kernel.assert_called_once()
        assert paired_kernel.call_args.kwargs["tgmm_tile_m"] == shared[0]
        assert paired_kernel.call_args.kwargs["tgmm_tile_k"] == shared[1]
        unpaired.assert_called_once()
    else:
        paired_kernel.assert_not_called()
        gate, up, _ = unpaired.call_args_list
        assert gate.kwargs["tgmm_tiling"] == shared
        assert up.kwargs["tgmm_tiling"] == shared
    # W2 is always the last unpaired call.
    w2_call = unpaired.call_args_list[-1].kwargs
    assert w2_call["transpose_rhs"] is transposed
    assert w2_call["tgmm_tiling"] == (w2_transposed if transposed else w2_own)


def test_kernel_native_layout_preserves_initialization_and_state_dict():
    param_init = {
        "w1_EFD": lambda x: torch.nn.init.normal_(x, std=0.02),
        "w2_EDF": lambda x: torch.nn.init.normal_(x, std=0.01),
        "w3_EFD": lambda x: torch.nn.init.normal_(x, std=0.03),
    }
    torch.manual_seed(0)
    stock = _build_experts(num_experts=3, param_init=param_init)
    stock.init_states()
    torch.manual_seed(0)
    native = _build_experts(
        num_experts=3,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
        param_init=param_init,
    )
    native.init_states()

    assert native.w1_EDF.is_contiguous()
    assert native.w2_EFD.is_contiguous()
    assert native.w3_EDF.is_contiguous()
    stock_state = stock.state_dict()
    native_state = native.state_dict()
    assert set(native_state) == set(stock_state)
    for name in stock_state:
        torch.testing.assert_close(
            native_state[name], stock_state[name], rtol=0, atol=0
        )

    restored = _build_experts(
        num_experts=3,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
    )
    restored.load_state_dict(stock_state)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, stock_state[name], rtol=0, atol=0)


def test_kernel_native_layout_checkpoint_hook_order():
    native = _build_experts(
        num_experts=2,
        use_pallas_grouped_mm=True,
        use_kernel_native_weight_layout=True,
    )
    save_keys = set()

    def observe_save(module, state_dict, prefix, local_metadata):
        save_keys.update(key.removeprefix(prefix) for key in state_dict)

    native.register_state_dict_post_hook(observe_save)
    state = native.state_dict()
    assert {"w1_EFD", "w2_EDF", "w3_EFD"} <= save_keys
    assert not {"w1_EDF", "w2_EFD", "w3_EDF"} & save_keys

    load_keys = set()

    def observe_load(module, state_dict, prefix, *args):
        load_keys.update(key.removeprefix(prefix) for key in state_dict)

    native.register_load_state_dict_pre_hook(observe_load)
    native.load_state_dict(state)
    assert {"w1_EDF", "w2_EFD", "w3_EDF"} <= load_keys
    assert not {"w1_EFD", "w2_EDF", "w3_EFD"} & load_keys


@pytest.mark.parametrize("use_pair", [False, True])
def test_kernel_native_layout_pallas_forward_and_backward(use_pair):
    param_init = {
        name: lambda x: torch.nn.init.normal_(x, std=0.02)
        for name in ("w1_EFD", "w2_EDF", "w3_EFD")
    }
    torch.manual_seed(0)
    stock = _build_experts(num_experts=2, param_init=param_init)
    stock.init_states()
    native = _build_experts(
        num_experts=2,
        use_pallas_grouped_mm=True,
        use_pallas_paired_grouped_mm=use_pair,
        use_kernel_native_weight_layout=True,
        param_init=param_init,
    )
    native.load_state_dict(stock.state_dict())
    x = torch.randn(7, 16, requires_grad=True)
    x_native = x.detach().clone().requires_grad_(True)
    counts = torch.tensor([3, 4], dtype=torch.int32)

    def grouped_mm(x, weight, counts, **kwargs):
        del kwargs
        assert weight.is_contiguous()
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        return torch._grouped_mm(x, weight, offs=offsets)

    def paired_grouped_mm(x, w1, w3, counts, **kwargs):
        return grouped_mm(x, w1, counts), grouped_mm(x, w3, counts)

    expected = stock.forward(x, counts)
    with patch(
        "torchtitan.experiments.tpu.grouped_experts.segment_matmul_pallas_2d",
        side_effect=grouped_mm,
    ), patch(
        "torchtitan.experiments.tpu.grouped_experts.paired_segment_matmul_pallas_2d",
        side_effect=paired_grouped_mm,
    ):
        actual = native.forward(x_native, counts)
    torch.testing.assert_close(actual, expected)

    actual.float().square().sum().backward()
    expected.float().square().sum().backward()
    torch.testing.assert_close(x_native.grad, x.grad, rtol=0.01, atol=1e-6)
    for name, grad in (
        ("w1_EFD", native.w1_EDF.grad.transpose(-2, -1)),
        ("w2_EDF", native.w2_EFD.grad.transpose(-2, -1)),
        ("w3_EFD", native.w3_EDF.grad.transpose(-2, -1)),
    ):
        expected_param = getattr(stock, name)
        torch.testing.assert_close(grad, expected_param.grad, rtol=0.01, atol=1e-6)


def test_kernel_native_layout_transposes_tp_sharding_axes():
    config = ShardingConfig(
        state_shardings={
            "w1_EFD": expert_param_placement_dense(tp_placement=spmd.S(1)),
            "w2_EDF": expert_param_placement_dense(tp_placement=spmd.S(2)),
            "w3_EFD": expert_param_placement_dense(tp_placement=spmd.S(1)),
        }
    )
    native = _native_weight_sharding(config)
    assert set(native.state_shardings) == {"w1_EDF", "w2_EFD", "w3_EDF"}
    assert native.state_shardings["w1_EDF"].local_type[MeshAxisName.TP] == spmd.S(2)
    assert native.state_shardings["w2_EFD"].local_type[MeshAxisName.TP] == spmd.S(1)
    assert native.state_shardings["w3_EDF"].local_type[MeshAxisName.TP] == spmd.S(2)


def test_kernel_native_layout_transposes_rank_aware_partition_spec():
    # Upstream d771abb0 deleted `protocols.sharding.SpmdLayout` in favour of
    # `spmd_types.SpmdType`, renaming the per-axis field `axis_types` ->
    # `local_type`.
    layout = spmd.SpmdType(
        {
            MeshAxisName.DP_SHARD: spmd.V,
            MeshAxisName.TP: spmd.V,
        },
        partition_spec=(MeshAxisName.DP_SHARD, None, MeshAxisName.TP),
    )

    native = _transpose_last_two_dims(layout)

    assert native.partition_spec == (
        MeshAxisName.DP_SHARD,
        MeshAxisName.TP,
        None,
    )
