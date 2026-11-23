import logging
from dataclasses import dataclass, field, replace

import torch
import spmd_types as spmd
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.config import derive
from torchtitan.distributed.spmd_types import maybe_set_sparse_mesh, spmd_mesh_size
from torchtitan.experiments.tpu.kernels.expert_segment_matmul import (
    segment_matmul_pallas_2d,
)
from torchtitan.experiments.tpu.kernels.paired_segment_matmul import (
    paired_segment_matmul_pallas_2d,
)
from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.base.module import Module
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.models.base.sharding import ShardingConfig

logger = logging.getLogger(__name__)

_CANONICAL_TO_NATIVE_WEIGHT = {
    "w1_EFD": "w1_EDF",
    "w2_EDF": "w2_EFD",
    "w3_EFD": "w3_EDF",
}


def _transpose_last_two_dims(
    layout: spmd.SpmdType, *, ndim: int = 3
) -> spmd.SpmdType:
    """Transpose tensor-dimension references in an SPMD layout.

    Upstream d771abb0 deleted `torchtitan.protocols.sharding.SpmdLayout` in
    favour of `spmd_types.SpmdType`, renaming the per-axis field
    `axis_types` -> `local_type`.

    This reads the RAW fields rather than upstream's `_per_axis_types()`
    helper. That helper flattens `partition_spec` entries into `S(dim)`
    per-axis types, which is fine for inspection but cannot be fed back into
    `local_type`: `SpmdType.__post_init__` rejects a value that carries both
    `S(i)` entries and an explicit `PartitionSpec`. Permuting the raw fields
    independently keeps the two representations mutually exclusive, exactly as
    they arrived.
    """

    if ndim < 2:
        raise ValueError(f"Expected rank of at least 2, got {ndim}")

    permutation = list(range(ndim))
    permutation[-2], permutation[-1] = permutation[-1], permutation[-2]

    local_type = {}
    for axis, axis_type in layout.local_type.items():
        if isinstance(axis_type, spmd.Shard):
            if not 0 <= axis_type.dim < ndim:
                raise ValueError(
                    f"Tensor dimension {axis_type.dim} is invalid for rank {ndim}"
                )
            axis_type = spmd.S(permutation[axis_type.dim])
        local_type[axis] = axis_type

    partition_spec = layout.partition_spec
    if partition_spec is not None:
        if len(partition_spec) != ndim:
            raise ValueError(
                f"Expected rank-{ndim} partition spec, got {len(partition_spec)} dims"
            )
        partition_spec = tuple(partition_spec[dim] for dim in permutation)
    return replace(
        layout,
        local_type=local_type,
        partition_spec=partition_spec,
    )


def _native_weight_sharding(config: ShardingConfig) -> ShardingConfig:
    """Map canonical expert state names/layouts to kernel-native storage."""

    state_shardings = dict(config.state_shardings)
    for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
        state_shardings[native_name] = _transpose_last_two_dims(
            state_shardings.pop(canonical_name)
        )
    return replace(config, state_shardings=state_shardings)


class GroupedExperts(Module):
    """Unfused grouped experts (w1_EFD, w2_EDF, w3_EFD) retained for TPU."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        activation_fn: BinaryActivationFn.Config = field(
            default_factory=SwiGLU.Config
        )

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.activation_fn = config.activation_fn.build()

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Raw expert computation without dispatch/combine."""
        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        gate_RF = self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=self.w1_EFD, offs=offsets_E
        )
        up_RF = self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=self.w3_EFD, offs=offsets_E
        )
        h_RF = self.activation_fn(gate_RF, up_RF)
        return self._grouped_mm(
            A=h_RF, weight_EOI=self.w2_EDF, offs=offsets_E
        ).type_as(x_RD)

    def _grouped_mm(
        self, *, A: torch.Tensor, weight_EOI: torch.Tensor, offs: torch.Tensor
    ) -> torch.Tensor:
        return torch._grouped_mm(
            A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs
        )


class TpuRoutedExperts(RoutedExperts):
    """``RoutedExperts`` subclass that delegates expert compute to ``inner_experts``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoutedExperts.Config):
        w13: GroupedLinear.Config | None = None  # type: ignore[assignment]
        w2: GroupedLinear.Config | None = None  # type: ignore[assignment]
        inner_experts: GroupedExperts.Config | None = None

        @property
        def num_experts(self) -> int:
            if self.inner_experts is not None:
                return self.inner_experts.num_experts
            assert self.w13 is not None
            return self.w13.group_size

        def __post_init__(self) -> None:
            if self.inner_experts is not None:
                if self.token_dispatcher.num_experts != self.inner_experts.num_experts:
                    raise ValueError(
                        "token dispatcher and inner_experts must contain the same "
                        "number of experts"
                    )
                return
            super().__post_init__()

    def __init__(self, config: Config):
        if config.inner_experts is not None:
            Module.__init__(self)
            self.w13 = None  # type: ignore[assignment]
            self.w2 = None  # type: ignore[assignment]
            self.inner_experts = config.inner_experts.build()
            self.activation_fn = config.activation_fn.build()
            self.output_postprocess = (
                config.output_postprocess.build()
                if config.output_postprocess is not None
                else None
            )
            self.token_dispatcher = config.token_dispatcher.build()
        else:
            super().__init__(config)
            self.inner_experts = None

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        routing_map_TE: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.inner_experts is None:
            return super().forward(
                x_TD,
                topk_scores_TK,
                topk_expert_ids_TK,
                routing_map_TE,
                padding_mask_T=padding_mask_T,
            )
        del padding_mask_T
        (
            routed_input_RD,
            num_global_tokens_per_local_expert_e,
            metadata,
        ) = self.token_dispatcher.dispatch(
            x_TD,
            topk_scores_TK,
            topk_expert_ids_TK,
            routing_map_TE,
        )
        with maybe_set_sparse_mesh():
            routed_output_RD = self.inner_experts(
                routed_input_RD, num_global_tokens_per_local_expert_e
            )
            if self.output_postprocess is not None:
                routed_output_RD = self.output_postprocess(routed_output_RD)
        return self.token_dispatcher.combine(routed_output_RD, metadata)


class TpuGroupedExperts(GroupedExperts):
    """TPU expert kernels with opt-in Pallas grouped paths.

    The optional kernel-native layout stores each expert weight in the exact
    ``(E, K, N)`` order consumed by Pallas, where ``K`` is the contracted input
    dimension and ``N`` is the output dimension. Thus ``K=D, N=F`` for w1/w3,
    while ``K=F, N=D`` for w2. State-dict hooks retain the canonical Qwen
    ``w1_EFD`` / ``w2_EDF`` / ``w3_EFD`` checkpoint contract.

    Translation hooks register during construction. Later save hooks on this
    module see canonical keys; later load pre-hooks see native keys.

    The paired path changes only W1/W3 execution: their parameters, FSDP
    collectives, checkpoint keys, and the W2 projection remain independent.
    W2's dWeight may tile differently from W1/W3's: a dWeight is tiled along
    ``K``, the weight's input dimension, which is ``F`` for W2 and ``D`` for
    W1/W3, so one K tile cannot suit both.
    The dWeight kernels accumulate in FP32 and store in the BF16 dtype of the
    weights they are given, whatever the param and reduce dtypes are.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        activation: str = "silu"
        use_pallas_grouped_mm: bool = False
        # Share routed-token loads between the equal-shaped W1/W3 projections.
        use_pallas_paired_grouped_mm: bool = False
        pallas_grouped_mm_tile_m: int = 1024
        pallas_grouped_mm_tile_k: int = 0
        pallas_grouped_mm_tile_n: int = 0
        pallas_grouped_mm_tgmm_tiling: tuple[int, int, int] = (128, 128, 128)
        # TGMM tiling for the W2 (down-projection) dWeight. ``None`` follows
        # ``pallas_grouped_mm_tgmm_tiling``. Set it when W1/W3 want a K tile
        # that W2's shorter K axis cannot hold.
        pallas_grouped_mm_w2_tgmm_tiling: tuple[int, int, int] | None = None
        # W2 dWeight tiling used instead when the parallelizer selects the
        # transposed W2 RHS (W2 FSDP-sharded on its trailing axis). The tuple
        # tiles the same logical dimensions, but the transposed kernel stores
        # them in the other order, so it wants a different shape. ``None``
        # follows ``pallas_grouped_mm_w2_tgmm_tiling``.
        pallas_grouped_mm_w2_transposed_tgmm_tiling: (
            tuple[int, int, int] | None
        ) = None
        use_kernel_native_weight_layout: bool = False

    def __init__(self, config: Config):
        if config.activation not in ("silu", "gelu_tanh"):
            raise ValueError(
                f"Unsupported expert activation: {config.activation!r}"
            )
        if config.use_kernel_native_weight_layout:
            if not config.use_pallas_grouped_mm:
                raise ValueError(
                    "Kernel-native expert weights require Pallas grouped matmul"
                )
            Module.__init__(self)
            self.num_experts = config.num_experts
            self.w1_EDF = torch.nn.Parameter(
                torch.empty(config.num_experts, config.dim, config.hidden_dim)
            )
            self.w2_EFD = torch.nn.Parameter(
                torch.empty(config.num_experts, config.hidden_dim, config.dim)
            )
            self.w3_EDF = torch.nn.Parameter(
                torch.empty(config.num_experts, config.dim, config.hidden_dim)
            )
            # `token_dispatcher` moved to the enclosing `RoutedExperts` in
            # upstream d771abb0; `GroupedExperts` is now dispatch-free.
        else:
            super().__init__(config)

        self.activation = config.activation
        self.use_pallas_grouped_mm = config.use_pallas_grouped_mm
        self.use_pallas_paired_grouped_mm = config.use_pallas_paired_grouped_mm
        self.pallas_grouped_mm_tile_m = config.pallas_grouped_mm_tile_m
        self.pallas_grouped_mm_tile_k = config.pallas_grouped_mm_tile_k
        self.pallas_grouped_mm_tile_n = config.pallas_grouped_mm_tile_n
        self.pallas_grouped_mm_tgmm_tiling = config.pallas_grouped_mm_tgmm_tiling
        self.pallas_grouped_mm_w2_tgmm_tiling = config.pallas_grouped_mm_w2_tgmm_tiling
        self.pallas_grouped_mm_w2_transposed_tgmm_tiling = (
            config.pallas_grouped_mm_w2_transposed_tgmm_tiling
        )
        self.use_kernel_native_weight_layout = config.use_kernel_native_weight_layout
        self._use_transposed_w2_rhs = False
        self._native_sharding_applied = False

        if self.use_pallas_paired_grouped_mm:
            if not self.use_pallas_grouped_mm:
                raise ValueError("Paired grouped matmul requires Pallas")
            if self.pallas_grouped_mm_tgmm_tiling[2] != 0:
                raise ValueError(
                    "Paired grouped matmul requires a full N TGMM tile"
                )

        if not self.use_kernel_native_weight_layout:
            return
        self.register_state_dict_post_hook(self._canonical_weights_on_save)
        self.register_load_state_dict_pre_hook(self._native_weights_on_load)

    def _init_param(self, name: str, param: torch.nn.Parameter) -> None:
        """Initialize native storage through a canonical contiguous tensor."""

        if not self.use_kernel_native_weight_layout:
            return super()._init_param(name, param)

        native_to_canonical = {
            native_name: canonical_name
            for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items()
        }
        canonical_name = native_to_canonical[name]
        canonical = torch.empty_like(param).transpose(-2, -1).contiguous()
        Module._init_param(self, canonical_name, canonical)
        with torch.no_grad():
            param.copy_(canonical.transpose(-2, -1))

    def _distribute_states(self, parallel_dims) -> None:
        if (
            self.use_kernel_native_weight_layout
            and not self._native_sharding_applied
            and self._sharding_config is not None
        ):
            self._sharding_config = _native_weight_sharding(self._sharding_config)
            self._native_sharding_applied = True
        super()._distribute_states(parallel_dims)

    def _parallelize(self, parallel_dims) -> None:
        # The paired GEMM can consume expert-major rows after EP dispatch, but
        # EP produces a ragged row count that is not generally tile-aligned.
        # Supporting it requires masked or padded tail rows, removing any
        # padding before combine, and validating the nested EP/FSDP sharding.
        if (
            self.use_pallas_paired_grouped_mm
            and parallel_dims.get_optional_mesh("ep") is not None
        ):
            raise ValueError("Paired grouped matmul requires EP=1 exact-row dispatch")
        super()._parallelize(parallel_dims)

    parallelize = _parallelize

    @staticmethod
    def _canonical_weights_on_save(module, state_dict, prefix, local_metadata) -> None:
        """Expose kernel-native parameters in canonical checkpoint layout."""

        for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
            native = state_dict.pop(f"{prefix}{native_name}")
            state_dict[f"{prefix}{canonical_name}"] = native.transpose(
                -2, -1
            ).contiguous()

    @staticmethod
    def _native_weights_on_load(module, state_dict, prefix, *args) -> None:
        """Load canonical checkpoint tensors into kernel-native storage."""

        for canonical_name, native_name in _CANONICAL_TO_NATIVE_WEIGHT.items():
            canonical_key = f"{prefix}{canonical_name}"
            if canonical_key in state_dict:
                state_dict[f"{prefix}{native_name}"] = state_dict.pop(
                    canonical_key
                ).transpose(-2, -1).contiguous()

    def _native_experts_forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """``GroupedExperts.forward`` with a configurable activation.

        Mirrors ``GroupedExperts.forward`` in
        ``torchtitan/models/common/moe.py`` (keep in sync); upstream hard-codes
        silu.
        """
        w1_EFD, w2_EDF, w3_EFD = self.w1_EFD, self.w2_EDF, self.w3_EFD

        offsets_E = torch.cumsum(num_tokens_per_expert_E, dim=0, dtype=torch.int32)
        if spmd.is_type_checking() and spmd_mesh_size("ep") == 1:
            for axis in ("dp", "cp"):
                spmd.mutate_type(offsets_E, axis, src=spmd.P, dst=spmd.V)

        h_RF = self._activation_fn(
            self._grouped_mm(A=x_RD.bfloat16(), weight_EOI=w1_EFD, offs=offsets_E)
        )
        h_RF = h_RF * self._grouped_mm(
            A=x_RD.bfloat16(), weight_EOI=w3_EFD, offs=offsets_E
        )
        return self._grouped_mm(A=h_RF, weight_EOI=w2_EDF, offs=offsets_E).type_as(x_RD)

    def _activation_fn(self, gate_RF: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu_tanh":
            return F.gelu(gate_RF, approximate="tanh")
        return F.silu(gate_RF)

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
        row_scale_R: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the experts on expert-sorted rows.

        ``row_scale_R``, if given, multiplies each row's hidden activation
        before the down projection (W2).
        """
        # The Qwen TPU recipe uses Pallas because native torch._grouped_mm mode-2
        # dWeight is incorrect at M=4096. See
        # https://github.com/google-pytorch/torch_tpu/issues/2668.
        if not self.use_pallas_grouped_mm:
            if row_scale_R is not None:
                raise ValueError("row_scale_R requires Pallas grouped matmul")
            if self.activation == "silu":
                return super().forward(x_RD, num_tokens_per_expert_E)
            return self._native_experts_forward(x_RD, num_tokens_per_expert_E)
        if self.use_kernel_native_weight_layout:
            w1_EKN = self.w1_EDF
            w2_EKN = self.w2_EFD
            w3_EKN = self.w3_EDF
        else:
            w1_EKN = self.w1_EFD.transpose(-2, -1)
            w2_EKN = self.w2_EDF.transpose(-2, -1)
            w3_EKN = self.w3_EFD.transpose(-2, -1)

        if self._use_transposed_w2_rhs:
            w2_EKN = w2_EKN.transpose(-2, -1)

        # Match GroupedExperts: expert compute is BF16 even when the surrounding
        # model is FP32, and the final result is restored to the caller's dtype.
        x_bf16 = x_RD.bfloat16()
        tile_m = self.pallas_grouped_mm_tile_m
        # The kernels need a row count divisible by tile_m. Append zero rows to
        # the last expert's group: they add nothing to dWeight and are sliced
        # off the output. A no-op for tile-aligned batches.
        num_rows = x_bf16.shape[0]
        pad_rows = -num_rows % tile_m
        if pad_rows:
            x_bf16 = F.pad(x_bf16, (0, 0, 0, pad_rows))
            num_tokens_per_expert_E = torch.cat(
                [num_tokens_per_expert_E[:-1], num_tokens_per_expert_E[-1:] + pad_rows]
            )
        tile_k = self.pallas_grouped_mm_tile_k
        tile_n = self.pallas_grouped_mm_tile_n
        tgmm_tiling = self.pallas_grouped_mm_tgmm_tiling
        w2_tgmm_tiling = self.pallas_grouped_mm_w2_tgmm_tiling or tgmm_tiling
        if self._use_transposed_w2_rhs:
            w2_tgmm_tiling = (
                self.pallas_grouped_mm_w2_transposed_tgmm_tiling or w2_tgmm_tiling
            )
        if self.use_pallas_paired_grouped_mm:
            # Every expert kernel gets BF16 weights and returns BF16 dWeights.
            # FP32 FSDP reductions widen those gradients after the kernel; BF16
            # reductions consume them as-is.
            gate_RF, up_RF = paired_segment_matmul_pallas_2d(
                x_bf16,
                w1_EKN.bfloat16(),
                w3_EKN.bfloat16(),
                num_tokens_per_expert_E,
                tile_m=tile_m,
                tgmm_tile_m=tgmm_tiling[0],
                tgmm_tile_k=tgmm_tiling[1],
            )
        else:
            gate_RF = segment_matmul_pallas_2d(
                x_bf16,
                w1_EKN.bfloat16(),
                num_tokens_per_expert_E,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                tgmm_tiling=tgmm_tiling,
            )
            up_RF = segment_matmul_pallas_2d(
                x_bf16,
                w3_EKN.bfloat16(),
                num_tokens_per_expert_E,
                tile_m=tile_m,
                tile_k=tile_k,
                tile_n=tile_n,
                tgmm_tiling=tgmm_tiling,
            )
        hidden_RF = self._activation_fn(gate_RF) * up_RF
        if row_scale_R is not None:
            if pad_rows:
                row_scale_R = F.pad(row_scale_R, (0, pad_rows))
            hidden_RF = (hidden_RF.float() * row_scale_R.reshape(-1, 1)).to(
                hidden_RF.dtype
            )
        return segment_matmul_pallas_2d(
            hidden_RF,
            w2_EKN.bfloat16(),
            num_tokens_per_expert_E,
            tile_m=tile_m,
            tile_k=tile_k,
            tile_n=tile_n,
            tgmm_tiling=w2_tgmm_tiling,
            transpose_rhs=self._use_transposed_w2_rhs,
        )[:num_rows].type_as(x_RD)


class TpuGroupedExpertsConverter(ModelConfigConverter):
    """Select the TPU grouped-expert implementation before construction."""

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        pass

    def __init__(self, config: Config):
        pass

    def convert(self, model_config):
        from torchtitan.models.common.moe import RoutedExperts

        count = 0
        matches = list(model_config.traverse(GroupedExperts.Config))
        for _, config, parent, field_name in matches:
            replacement = derive(config, TpuGroupedExperts.Config)
            if parent is None:
                model_config = replacement
            elif isinstance(parent, list):
                assert isinstance(field_name, int)
                parent[field_name] = replacement
            else:
                assert isinstance(field_name, str)
                setattr(parent, field_name, replacement)
            count += 1

        for _, routed_cfg, parent, field_name in list(
            model_config.traverse(RoutedExperts.Config)
        ):
            if getattr(routed_cfg, "inner_experts", None) is not None or routed_cfg.w13 is None:
                continue
            w13 = routed_cfg.w13
            w2 = routed_cfg.w2
            assert w2 is not None
            param_init = {}
            if w13.param_init and "weight" in w13.param_init:
                w13_init = w13.param_init["weight"]
                # `fused_grouped_gate_up_param_init` wraps the per-weight
                # initializers in a picklable `_FusedGroupedGateUpInit` object
                # carrying `w1_init` / `w3_init`; older versions used a closure
                # over the original `param_init` dict.
                w1_init = getattr(w13_init, "w1_init", None)
                w3_init = getattr(w13_init, "w3_init", None)
                if w1_init is not None and w3_init is not None:
                    param_init["w1_EFD"] = w1_init
                    param_init["w3_EFD"] = w3_init
                elif getattr(w13_init, "__closure__", None):
                    for cell in w13_init.__closure__:
                        val = cell.cell_contents
                        if isinstance(val, dict) and {"w1_EFD", "w3_EFD"} <= val.keys():
                            param_init["w1_EFD"] = val["w1_EFD"]
                            param_init["w3_EFD"] = val["w3_EFD"]
            if w2.param_init and "weight" in w2.param_init:
                param_init["w2_EDF"] = w2.param_init["weight"]
            tpu_routed_cfg = TpuRoutedExperts.Config(
                token_dispatcher=routed_cfg.token_dispatcher,
                activation_fn=routed_cfg.activation_fn,
                output_postprocess=routed_cfg.output_postprocess,
                inner_experts=TpuGroupedExperts.Config(
                    dim=w13.in_features,
                    hidden_dim=w13.out_features,
                    num_experts=w13.group_size,
                    activation_fn=routed_cfg.activation_fn,
                    param_init=param_init or None,
                    sharding_config=w13.sharding_config,
                ),
                sharding_config=routed_cfg.sharding_config,
            )
            if parent is None:
                model_config = tpu_routed_cfg
            elif isinstance(parent, list):
                assert isinstance(field_name, int)
                parent[field_name] = tpu_routed_cfg
            else:
                assert isinstance(field_name, str)
                setattr(parent, field_name, tpu_routed_cfg)
            count += 1

        logger.info("Replaced %d GroupedExperts configuration(s) for TPU", count)
        return model_config


__all__ = [
    "GroupedExperts",
    "TpuGroupedExperts",
    "TpuGroupedExpertsConverter",
    "TpuRoutedExperts",
]
