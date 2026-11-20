# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Literal, overload, Protocol

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.flex_shard import (
    BucketConfig,
    ComputeLayout,
    DistMuon as FlexShardDistMuon,
)

from .utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

logger = logging.getLogger(__name__)


__all__ = [
    "Adam",
    "AdamW",
    "BaseOptimizer",
    "DistMuon",
    "OptimizersContainer",
    "ParamGroupConfig",
    "collect_moe_routing_metric_tensors",
    "configure_moe_routing_metrics",
    "default_adamw",
    "get_moe_load_balancing_metrics",
    "register_moe_load_balancing_hook",
    "register_moe_quantile_balancing_hook",
    "set_moe_routing_metric_tensors",
]


@dataclass(kw_only=True, slots=True)
class ParamGroupConfig:
    """Backward-compatible parameter group descriptor."""

    pattern: str
    optimizer_name: str
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


class _MoERouterLike(Protocol):
    tokens_per_expert_E: torch.Tensor  # noqa: N815


class _MoELike(Protocol):
    load_balance_coeff: float | None
    expert_bias_E: torch.Tensor  # noqa: N815
    router: _MoERouterLike

MomentDType = Literal["parameter", "bfloat16"]


class BaseOptimizer(Optimizer, Configurable):
    """Base class for configurable TorchTitan optimizers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pattern: str
        """Regex matched against parameter fully qualified names."""


def _validate_moment_dtype(
    *,
    moment_dtype: MomentDType,
    fused: bool | None,
) -> None:
    if moment_dtype not in ("parameter", "bfloat16"):
        raise ValueError(f"Unsupported Adam moment dtype {moment_dtype!r}")
    if moment_dtype == "bfloat16" and fused is not True:
        raise ValueError("bfloat16 Adam moments require fused=True")


def _register_bfloat16_moment_hooks(optimizer: torch.optim.Optimizer) -> None:
    """Create and restore Adam moment tensors in bfloat16."""

    def initialize_moments(
        optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict
    ) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = optimizer.state[param]
                if state:
                    continue
                state["step"] = (
                    torch.zeros((), dtype=torch.float32, device=param.device)
                    if group.get("capturable") or group.get("fused")
                    else torch.tensor(0.0, dtype=torch.float32)
                )
                state["exp_avg"] = torch.zeros_like(
                    param, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )
                state["exp_avg_sq"] = torch.zeros_like(
                    param, dtype=torch.bfloat16, memory_format=torch.preserve_format
                )
                if group.get("amsgrad"):
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        param,
                        dtype=torch.bfloat16,
                        memory_format=torch.preserve_format,
                    )

    def restore_moment_dtype(optimizer: torch.optim.Optimizer) -> None:
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param, {})
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state:
                        state[key] = state[key].to(dtype=torch.bfloat16)

    optimizer.register_step_pre_hook(initialize_moments)
    optimizer.register_load_state_dict_post_hook(restore_moment_dtype)


class Adam(torch.optim.Adam, BaseOptimizer):
    """Configurable ``torch.optim.Adam``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0.0
        fused: bool | None = True
        moment_dtype: MomentDType = "parameter"

        def __post_init__(self) -> None:
            _validate_moment_dtype(
                moment_dtype=self.moment_dtype,
                fused=self.fused,
            )

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        super().__init__(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            fused=config.fused,
        )
        if config.moment_dtype == "bfloat16":
            _register_bfloat16_moment_hooks(self)


class AdamW(torch.optim.AdamW, BaseOptimizer):
    """Configurable ``torch.optim.AdamW``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        lr: float = 8e-4
        betas: tuple[float, float] = (0.9, 0.95)
        eps: float = 1e-8
        weight_decay: float = 0.1
        foreach: bool | None = None
        fused: bool | None = True
        moment_dtype: MomentDType = "parameter"

        def __post_init__(self) -> None:
            _validate_moment_dtype(
                moment_dtype=self.moment_dtype,
                fused=self.fused,
            )

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        super().__init__(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            foreach=config.foreach,
            fused=config.fused,
        )
        if config.moment_dtype == "bfloat16":
            _register_bfloat16_moment_hooks(self)


class DistMuon(FlexShardDistMuon, BaseOptimizer):
    """Configurable TorchTitan wrapper around FlexShard's ``DistMuon``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        compute_sharding_by_fqn: Mapping[str, ComputeLayout]
        bucket_configs: Sequence[BucketConfig]
        lr: float = 1e-3
        weight_decay: float = 0.1
        momentum: float = 0.95
        nesterov: bool = True
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
        eps: float = 1e-7
        ns_steps: int = 5
        adjust_lr_fn: Literal[
            "original", "match_rms_adamw", "spectral_unclamped"
        ] | None = None

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        super().__init__(
            cast(Iterable[dict[str, Any]], params),
            compute_sharding_by_fqn=config.compute_sharding_by_fqn,
            bucket_configs=config.bucket_configs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_coefficients=config.ns_coefficients,
            eps=config.eps,
            ns_steps=config.ns_steps,
            adjust_lr_fn=config.adjust_lr_fn,
        )


class OptimizersContainer(Optimizer, Stateful, Configurable):
    """A container for multiple optimizers, supporting mixed optimizer types.

    This class wraps multiple optimizers into a single object to simplify the
    training loop. Each configured optimizer selects parameters by regex pattern;
    patterns are checked in order and the first match wins.

    Each model part gets one optimizer instance per matching configuration.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    Args:
        config (Config): Ordered optimizer configurations.
        model_parts (List[nn.Module]): List of model parts to be optimized.
        enable_cuda_graph: Prepare every optimizer for CUDA graph capture.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizers: list[BaseOptimizer.Config] = field(default_factory=list)
        """Optimizer configurations in first-match-wins pattern order."""

        implementation: Literal[
            "for-loop", "foreach", "fused", "fused_opt_states_bf16"
        ] = "fused"
        """Optimizer implementation mode applied when overridden on the container."""

        @property
        def param_groups(self) -> list[ParamGroupConfig]:
            groups: list[ParamGroupConfig] = []
            for opt in self.optimizers:
                opt_name = type(opt).__qualname__.removesuffix(".Config")
                kwargs: dict[str, Any] = {}
                for k in ("lr", "betas", "eps", "weight_decay", "momentum", "nesterov"):
                    if hasattr(opt, k):
                        kwargs[k] = getattr(opt, k)
                groups.append(
                    ParamGroupConfig(
                        pattern=opt.pattern,
                        optimizer_name=opt_name,
                        optimizer_kwargs=kwargs,
                    )
                )
            return groups

    optimizers: list[Optimizer]
    model_parts: list[nn.Module]

    @staticmethod
    def _apply_container_implementation(
        optimizer_config: BaseOptimizer.Config,
        implementation: str,
    ) -> BaseOptimizer.Config:
        if implementation == "fused" or not isinstance(
            optimizer_config, (Adam.Config, AdamW.Config)
        ):
            return optimizer_config
        if implementation == "foreach":
            replace_kwargs: dict[str, Any] = {
                "fused": False,
                "moment_dtype": "parameter",
            }
            if isinstance(optimizer_config, AdamW.Config):
                replace_kwargs["foreach"] = True
            return dataclasses.replace(optimizer_config, **replace_kwargs)
        if implementation == "for-loop":
            replace_kwargs = {
                "fused": False,
                "moment_dtype": "parameter",
            }
            if isinstance(optimizer_config, AdamW.Config):
                replace_kwargs["foreach"] = False
            return dataclasses.replace(optimizer_config, **replace_kwargs)
        if implementation == "fused_opt_states_bf16":
            replace_kwargs = {
                "fused": True,
                "moment_dtype": "bfloat16",
            }
            if isinstance(optimizer_config, AdamW.Config):
                replace_kwargs["foreach"] = None
            return dataclasses.replace(optimizer_config, **replace_kwargs)
        return optimizer_config

    @staticmethod
    def _build_param_group(
        model: nn.Module,
        optimizer_config: BaseOptimizer.Config,
        claimed: set[str],
    ) -> dict[str, Any]:
        """Build one named PyTorch parameter group from an optimizer config."""
        try:
            pattern = re.compile(optimizer_config.pattern)
        except re.error as error:
            raise ValueError(
                f"Invalid optimizer pattern {optimizer_config.pattern!r}: {error}"
            ) from error

        params: list[nn.Parameter] = []
        param_names: list[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad and name not in claimed and pattern.search(name):
                params.append(param)
                param_names.append(canonical_fqn(name))
                claimed.add(name)

        if not params:
            raise ValueError(
                f"Optimizer pattern {optimizer_config.pattern!r} matched no parameters"
            )

        return {"params": params, "param_names": param_names}

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
        enable_cuda_graph: bool = False,
    ) -> None:
        all_params: list[nn.Parameter] = []
        self.optimizers = []
        self.model_parts = model_parts

        for part_idx, model in enumerate(self.model_parts):
            claimed: set[str] = set()
            for raw_optimizer_config in config.optimizers:
                optimizer_config = self._apply_container_implementation(
                    raw_optimizer_config,
                    config.implementation,
                )
                param_group = self._build_param_group(
                    model,
                    optimizer_config,
                    claimed,
                )
                optimizer = optimizer_config.build(params=[param_group])
                if enable_cuda_graph:
                    if any(
                        "capturable" not in group for group in optimizer.param_groups
                    ):
                        raise ValueError(
                            f"Optimizer {type(optimizer).__name__} does not support "
                            "CUDA graph capture."
                        )
                    for group in optimizer.param_groups:
                        group["initial_lr"] = group["lr"]
                        group["capturable"] = True
                        group["lr"] = torch.tensor(
                            group["lr"],
                            dtype=torch.float32,
                            device=group["params"][0].device,
                        )

                    def _save_host_lr(
                        _optimizer: Optimizer, state_dict: dict[str, Any]
                    ) -> dict[str, Any]:
                        for group in state_dict["param_groups"]:
                            if isinstance(group["lr"], torch.Tensor):
                                group["lr"] = float(group["lr"])
                        return state_dict

                    optimizer.register_state_dict_post_hook(_save_host_lr)
                self.optimizers.append(optimizer)
                self._log_optimizer(
                    optimizer,
                    part_idx,
                    optimizer_config.pattern,
                )
                all_params.extend(param_group["params"])

        self._validate_params(all_params)
        self._post_init(all_params)

    def _log_optimizer(self, optimizer: Optimizer, part_idx: int, pattern: str) -> None:
        """Log one optimizer's parameter assignment."""
        _KEY_KWARGS = {
            "lr",
            "weight_decay",
            "betas",
            "eps",
            "momentum",
            "nesterov",
            "fused",
            "foreach",
        }
        opt_name = type(optimizer).__name__
        for group in optimizer.param_groups:
            num_params = len(group["params"])
            kwargs = {k: v for k, v in group.items() if k in _KEY_KWARGS}
            logger.info(
                f"Optimizer {opt_name} (model_part={part_idx}): "
                f"{num_params} params [{pattern}] {kwargs}"
            )

    def _validate_params(self, all_params: list[nn.Parameter]) -> None:
        """Verify every trainable param is assigned to exactly one optimizer."""
        expected = {
            id(p)
            for model in self.model_parts
            for p in model.parameters()
            if p.requires_grad
        }
        actual = {id(p) for p in all_params}
        if len(all_params) != len(actual):
            raise ValueError(
                "A trainable parameter was assigned to multiple optimizers"
            )
        if expected != actual:
            raise ValueError(
                f"Parameter mismatch: {len(expected)} trainable params in model, "
                f"{len(actual)} in optimizers"
            )

    def __iter__(self) -> Iterator[Optimizer]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        assert closure is None, "OptimizersContainer does not support closures"
        for optimizer in self.optimizers:
            optimizer.step()
        return None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        """Return a flat, FQN-keyed optimizer state dict for all optimizers.

        Side effect: if an optimizer's state has not been created yet (no training
        step taken), ``init_optim_state`` materializes it with a zero-gradient,
        zero-lr step before reading. The step leaves parameters unchanged, and the
        call is a no-op once state exists.
        """
        result: dict[str, Any] = {}
        for optim in self.optimizers:
            init_optim_state(optim)
            result.update(get_flat_optim_state_dict(optim))
        return result

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # init_optim_state must run first: the unflatten step reads each
        # optimizer's live state to learn which state tensors to expect.
        for optim in self.optimizers:
            init_optim_state(optim)
            load_flat_optim_state_dict(optim, state_dict)

    def _post_init(self, all_params: list[nn.Parameter]) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks (e.g. register_step_pre_hook for MoE load balancing).
        Optimizer.__init__(self, all_params, {})

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass


def default_adamw(lr: float = 8e-4, **kwargs: Any) -> OptimizersContainer.Config:
    """Create an OptimizersContainer.Config with a catch-all AdamW optimizer.

    Use as a convenience for the common case::

        optimizer=default_adamw(lr=3e-4)
    """
    config_kwargs: dict[str, Any] = {}
    if "implementation" in kwargs:
        config_kwargs["implementation"] = kwargs.pop("implementation")

    return OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(
                pattern=r".*",
                lr=lr,
                **kwargs,
            )
        ],
        **config_kwargs,
    )


def _iter_moe_layers(
    model_parts: list[nn.Module],
) -> Iterator[tuple[nn.Module, _MoELike]]:
    """Yield ``(transformer_block, moe)`` for every MoE layer in the model."""
    for model_part in model_parts:
        # MTP decoder blocks live in ``mtp_layers`` after FSDP wrapping;
        # they are only temporarily inserted into ``layers`` while FSDP
        # is applied. Keep both containers in the runtime hook so MTP
        # expert bias and usage counters receive the same update as the
        # main decoder layers.
        layers = getattr(model_part, "layers", None)
        if layers is None:
            continue
        layer_containers = [layers]
        mtp_layers = getattr(model_part, "mtp_layers", None)
        if mtp_layers is not None:
            layer_containers.append(mtp_layers)
        for layers in layer_containers:
            assert isinstance(layers, (nn.ModuleDict, nn.ModuleList))
            for transformer_block in layers.children():
                if getattr(transformer_block, "moe_enabled", False):
                    yield transformer_block, cast(_MoELike, transformer_block.moe)


def register_moe_load_balancing_hook(
    optimizers: Optimizer,
    model_parts: list[nn.Module],
    parallelism_context: ParallelismContext | None = None,
    *,
    parallel_dims: ParallelismContext | None = None,
) -> None:
    from torchtitan.models.common.moe import (
        register_moe_load_balancing_hook as _register_moe_load_balancing_hook,
    )

    if parallelism_context is None:
        parallelism_context = parallel_dims
    assert parallelism_context is not None
    _register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)


def register_moe_quantile_balancing_hook(
    optimizers: Optimizer,
    model_parts: list[nn.Module],
    parallelism_context: ParallelismContext | None = None,
    *,
    parallel_dims: ParallelismContext | None = None,
) -> None:
    from torchtitan.models.common.moe import (
        register_moe_quantile_balancing_hook as _register_moe_quantile_balancing_hook,
    )

    if parallelism_context is None:
        parallelism_context = parallel_dims
    assert parallelism_context is not None
    _register_moe_quantile_balancing_hook(optimizers, model_parts, parallelism_context)


def _moe_routing_metrics_aux_loss(moe: nn.Module) -> Any:
    """Return the MoE's router aux loss if it produces routing metrics.

    Only ``MoELoadBalanceAuxLoss`` (set as ``router.aux_loss``) produces them;
    it is duck-typed here because this module cannot import the model code.
    """
    aux_loss = getattr(getattr(moe, "router", None), "aux_loss", None)
    if aux_loss is None or not hasattr(aux_loss, "collect_routing_metrics"):
        return None
    return aux_loss


def configure_moe_routing_metrics(
    model_parts: list[nn.Module], enabled: bool
) -> bool:
    """Enable graph-safe routing metric capture on supported MoE layers."""
    found_supported_layer = False
    for _transformer_block, moe in _iter_moe_layers(model_parts):
        aux_loss = _moe_routing_metrics_aux_loss(moe)
        if aux_loss is None:
            continue
        if enabled and aux_loss.type != "batch_wise":
            raise ValueError(
                "MoE routing metrics currently require batch_wise auxiliary loss"
            )
        aux_loss.collect_routing_metrics = enabled
        found_supported_layer = True
    return enabled and found_supported_layer


def _local_metric_tensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.distributed.tensor.DTensor):
        return value.to_local()
    return value


def collect_moe_routing_metric_tensors(
    model_parts: list[nn.Module],
) -> list[torch.Tensor]:
    """Collect the fixed-shape local tensors produced by the latest forward."""
    local_counts = []
    local_normalized_aux_losses = []
    for _transformer_block, moe in _iter_moe_layers(model_parts):
        aux_loss = _moe_routing_metrics_aux_loss(moe)
        if aux_loss is None or not aux_loss.collect_routing_metrics:
            continue
        if not hasattr(aux_loss, "_last_local_tokens_per_expert_E") or not hasattr(
            aux_loss, "_last_local_normalized_aux_loss"
        ):
            raise RuntimeError(
                "MoE routing metrics were enabled but the forward did not produce them"
            )
        local_counts.append(
            _local_metric_tensor(aux_loss._last_local_tokens_per_expert_E)
        )
        local_normalized_aux_losses.append(
            _local_metric_tensor(aux_loss._last_local_normalized_aux_loss)
        )

    if not local_counts:
        return []
    return [
        torch.stack(local_counts).float(),
        torch.stack(local_normalized_aux_losses).float(),
    ]


def set_moe_routing_metric_tensors(
    model_parts: list[nn.Module], metric_tensors: list[torch.Tensor]
) -> None:
    """Restore graph outputs as runtime sidecars consumed by metric logging."""
    if not metric_tensors:
        return
    if len(metric_tensors) != 2:
        raise ValueError(
            f"Expected two MoE routing metric tensors, got {len(metric_tensors)}"
        )
    local_counts_LE, local_normalized_aux_losses_L = metric_tensors
    aux_loss_modules = [
        aux_loss
        for _transformer_block, moe in _iter_moe_layers(model_parts)
        if (aux_loss := _moe_routing_metrics_aux_loss(moe)) is not None
        and aux_loss.collect_routing_metrics
    ]
    if (
        local_counts_LE.size(0) != len(aux_loss_modules)
        or local_normalized_aux_losses_L.size(0) != len(aux_loss_modules)
    ):
        raise ValueError("MoE routing metric output shapes do not match the model")
    for layer_idx, aux_loss in enumerate(aux_loss_modules):
        aux_loss._last_local_tokens_per_expert_E = local_counts_LE[layer_idx]
        aux_loss._last_local_normalized_aux_loss = local_normalized_aux_losses_L[
            layer_idx
        ]


def get_moe_load_balancing_metrics(
    model_parts: list[nn.Module],
) -> dict[str, Any]:
    metric_tensors = collect_moe_routing_metric_tensors(model_parts)
    if not metric_tensors:
        return {}

    local_tokens_per_expert_LE, local_normalized_aux_loss_L = metric_tensors
    aux_loss_modules = [
        aux_loss
        for _transformer_block, moe in _iter_moe_layers(model_parts)
        if (aux_loss := _moe_routing_metrics_aux_loss(moe)) is not None
        and aux_loss.collect_routing_metrics
    ]
    local_sums = torch.cat(
        [local_tokens_per_expert_LE, local_normalized_aux_loss_L.unsqueeze(1)], dim=1
    )
    global_sums = _all_reduce_moe_metric_sums(local_sums, aux_loss_modules[0])
    global_tokens_per_expert_LE = global_sums[:, :-1]
    dp_degree = (
        aux_loss_modules[0].dp_mesh.size()
        if aux_loss_modules[0].dp_mesh is not None
        else 1
    )
    normalized_aux_loss_L = global_sums[:, -1] / dp_degree

    mean_counts_L = global_tokens_per_expert_LE.mean(dim=1)
    max_mean_L = global_tokens_per_expert_LE.amax(dim=1) / mean_counts_L
    route_cv_L = (
        (global_tokens_per_expert_LE - mean_counts_L.unsqueeze(1))
        .square()
        .mean(dim=1)
        .sqrt()
        / mean_counts_L
    )
    assignment_totals_L = global_tokens_per_expert_LE.sum(dim=1)
    # TPU workaround: torch_tpu lacks aten::median; sorting preserves
    # PyTorch's lower-median semantics.
    median_idx = (max_mean_L.numel() - 1) // 2

    metrics = {
        "moe/global_aux_loss_normalized": normalized_aux_loss_L.mean(),
        "moe/route_max_mean_max": max_mean_L.amax(),
        "moe/route_max_mean_mean": max_mean_L.mean(),
        "moe/route_max_mean_median": max_mean_L.sort().values[median_idx],
        "moe/route_cv_max": route_cv_L.amax(),
        "moe/route_cv_mean": route_cv_L.mean(),
        "moe/route_cv_median": route_cv_L.sort().values[median_idx],
        "moe/route_assignments_min": assignment_totals_L.amin(),
        "moe/route_assignments_max": assignment_totals_L.amax(),
    }

    for layer_idx, (layer_loss, layer_counts, layer_max_mean, layer_cv) in enumerate(
        zip(
            normalized_aux_loss_L,
            global_tokens_per_expert_LE,
            max_mean_L,
            route_cv_L,
            strict=True,
        )
    ):
        metrics[f"moe/global_aux_loss_normalized/layer_{layer_idx:03d}"] = layer_loss
        metrics[f"moe/route_max_mean/layer_{layer_idx:03d}"] = layer_max_mean
        metrics[f"moe/route_cv/layer_{layer_idx:03d}"] = layer_cv
        metrics[f"moe/route_assignments/layer_{layer_idx:03d}"] = layer_counts.sum()

    # Preserve exact per-expert counts in one event; thousands of individual
    # scalar writes can make the metrics rank miss the process-group timeout.
    metrics["moe/route_counts_json"] = json.dumps(
        global_tokens_per_expert_LE.to(dtype=torch.int64, device="cpu").tolist(),
        separators=(",", ":"),
    )

    return metrics


def _all_reduce_moe_metric_sums(
    values: torch.Tensor,
    aux_loss: Any,
) -> torch.Tensor:
    with torch.no_grad():
        for reduce_mesh in (aux_loss.tp_mesh, aux_loss.dp_mesh, aux_loss.cp_mesh):
            if reduce_mesh is None:
                continue
            torch.distributed.all_reduce(
                values, group=reduce_mesh.get_group(), op=torch.distributed.ReduceOp.SUM
            )
    return values
