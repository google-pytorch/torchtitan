# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, cast, Generic, Literal, overload, Protocol, TypeVar

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.config import Configurable
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.flex_shard import build_dist_muon
from torchtitan.tools.logging import logger

from .utils import (
    get_flat_optim_state_dict,
    init_optim_state,
    load_flat_optim_state_dict,
)

__all__ = [
    "OptimizersContainer",
    "ParamGroupConfig",
    "collect_moe_routing_metric_tensors",
    "configure_moe_routing_metrics",
    "default_adamw",
    "get_moe_load_balancing_metrics",
    "register_moe_load_balancing_hook",
    "set_moe_routing_metric_tensors",
]


@dataclass(kw_only=True, slots=True)
class ParamGroupConfig:
    """Configuration for a parameter group with its own optimizer.

    Each entry specifies a regex pattern matching parameter FQNs and a
    self-contained optimizer setup. ``optimizer_name`` and ``optimizer_kwargs``
    fully define the optimizer for matched parameters — no implicit inheritance.

    Patterns are checked in order; first match wins. Place specific patterns
    before a broad fallback pattern. Example::

        param_groups=[
            ParamGroupConfig(pattern=r"\\.bias$", ...),
            ParamGroupConfig(pattern=r"\\.router\\.", ...),
            ParamGroupConfig(pattern=r".*", ...),
        ]
    """

    pattern: str
    """Regex pattern matched against parameter fully qualified names (FQNs).
    E.g. '.*bias$', '.*norm.*', '.*\\.embed_tokens\\..*', '.*' (catch-all)"""

    optimizer_name: str
    """Optimizer type for this group."""

    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the optimizer constructor.
    Must include all required kwargs (e.g. ``lr``). No implicit defaults."""


T = TypeVar("T", bound=Optimizer)


class _MoELike(Protocol):
    load_balance_coeff: float | None
    tokens_per_expert_E: torch.Tensor  # noqa: N815
    expert_bias_E: torch.Tensor  # noqa: N815


class OptimizersContainer(Optimizer, Stateful, Configurable, Generic[T]):
    """A container for multiple optimizers, supporting mixed optimizer types.

    This class wraps multiple optimizers into a single object to simplify the
    training loop. Each parameter group is configured via ``ParamGroupConfig``
    with its own optimizer type and kwargs. Parameters are matched to groups
    by regex pattern (first match wins), and groups using the same optimizer
    type are batched into a single optimizer instance for performance.

    Each model part (from pipeline parallelism) may have multiple optimizer
    instances if different parameter groups use different optimizer types.

    **Note**
    Users who want to customize the optimizer behavior can inherit from this class and
    extend the functionality as needed. The following methods must follow the same signature
    as ``torch.optim.Optimizer`` class: ``step()``, ``zero_grad()``, ``state_dict()``,
    ``load_state_dict()``.

    Args:
        config (Config): Optimizer configuration with param group definitions.
        model_parts (List[nn.Module]): List of model parts to be optimized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        param_groups: list[ParamGroupConfig] = field(default_factory=list)
        """Per-parameter-group optimizer configurations. Each entry specifies a
        regex pattern and a self-contained optimizer setup.
        Patterns are checked in order; first match wins."""

        implementation: Literal[
            "for-loop", "foreach", "fused", "fused_opt_states_bf16"
        ] = "fused"
        """
        Optimizer implementation mode applied to all optimizer instances.
        Per-param-group ``optimizer_kwargs`` can override this (e.g.
        ``"fused": False`` for optimizers that don't support fused).

        - 'fused': Use fused implementation (CUDA only) for best performance.
        - 'foreach': Use some horizontal fusion of tensors for better performance.
        - 'for-loop': Use the default implementation for the optimizer (slowest).
        - 'fused_opt_states_bf16': Like 'fused', but initialize Adam/AdamW
          momentum and variance in bfloat16 via a step pre-hook so the fused
          CUDA kernel uses its mixed-precision path (fp32 params + bf16 states).
          Only supported for Adam/AdamW. See docs/bf16_optimizer_states.md.
        - more info: https://pytorch.org/docs/stable/optim.html
        """

        optimizer_factory_kwargs_by_name: dict[str, dict[str, Any]] = field(
            default_factory=dict
        )
        """Arguments passed once per optimizer factory invocation, keyed by name.

        Use this for instance-wide objects such as per-parameter compute
        metadata and communication bucket specs. These arguments are not copied
        into PyTorch parameter groups; group hyperparameters belong in
        ``ParamGroupConfig.optimizer_kwargs``.
        """

    optimizers: list[T]
    model_parts: list[nn.Module]

    @staticmethod
    def _resolve_optimizer_factory(name: str) -> Callable[..., Optimizer]:
        optimizer_factories: dict[str, Callable[..., Optimizer]] = {
            "Adam": torch.optim.Adam,
            "AdamW": torch.optim.AdamW,
            "DistMuon": build_dist_muon,
        }
        if name not in optimizer_factories:
            raise NotImplementedError(f"Optimizer {name} not added.")
        return optimizer_factories[name]

    @staticmethod
    def _build_impl_kwargs(config: Config) -> dict[str, Any]:
        """Build implementation-related kwargs (fused/foreach) from config."""
        assert config.implementation in [
            "fused",
            "foreach",
            "for-loop",
            "fused_opt_states_bf16",
        ]
        fused = config.implementation in ("fused", "fused_opt_states_bf16")
        return {
            "fused": fused,
            "foreach": config.implementation == "foreach",
        }

    @staticmethod
    def _build_param_groups(
        model: nn.Module,
        param_group_configs: list[ParamGroupConfig],
        impl_kwargs: dict[str, Any],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
        """Build PyTorch param groups from model parameters, partitioned by optimizer.

        Each parameter is assigned to the first matching ParamGroupConfig pattern.

        Returns two dicts keyed by optimizer name and aligned by index: the param
        group dicts to pass to the optimizer constructor, and the regex pattern of
        each group. Patterns are returned separately (not stored on the group) so
        they stay out of the saved optimizer state dict; they are logging-only.

        Each param group dict carries a ``param_names`` list (canonical FQNs
        aligned with ``params``) so PyTorch records the names on the group; the
        checkpoint utilities use those names to build FQN-keyed optimizer state.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        patterns: dict[str, list[str]] = defaultdict(list)
        claimed: set[str] = set()  # first-match-wins

        for pg in param_group_configs:
            pattern = re.compile(pg.pattern)
            params: list[nn.Parameter] = []
            param_names: list[str] = []
            for name, param in model.named_parameters():
                if param.requires_grad and name not in claimed and pattern.search(name):
                    params.append(param)
                    param_names.append(canonical_fqn(name))
                    claimed.add(name)

            if not params:
                raise ValueError(
                    f"Optimizer param_groups pattern '{pg.pattern}' "
                    f"matched no parameters"
                )

            groups[pg.optimizer_name].append(
                {
                    "params": params,
                    "param_names": param_names,
                    **impl_kwargs,
                    **pg.optimizer_kwargs,
                }
            )
            patterns[pg.optimizer_name].append(pg.pattern)

        return groups, patterns

    def __init__(self, config: Config, *, model_parts: list[nn.Module]) -> None:
        impl_kwargs = self._build_impl_kwargs(config)
        param_group_configs = config.param_groups
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts

        for part_idx, model in enumerate(self.model_parts):
            groups_by_opt_name, patterns_by_opt_name = self._build_param_groups(
                model, param_group_configs, impl_kwargs
            )
            for opt_name, opt_param_groups in groups_by_opt_name.items():
                optimizer = self._resolve_optimizer_factory(opt_name)(
                    opt_param_groups,
                    **config.optimizer_factory_kwargs_by_name.get(opt_name, {}),
                )
                self.optimizers.append(cast(T, optimizer))
                self._log_optimizer(optimizer, part_idx, patterns_by_opt_name[opt_name])
                for group in opt_param_groups:
                    all_params.extend(group["params"])

        self._validate_params(all_params)

        if config.implementation == "fused_opt_states_bf16":
            self._register_bf16_optimizer_state_hook()
        self._post_init(all_params)

    def _log_optimizer(
        self, optimizer: Optimizer, part_idx: int, patterns: list[str]
    ) -> None:
        """Log one optimizer's param-group assignments (patterns are logging-only)."""
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
        for group, pattern in zip(optimizer.param_groups, patterns):
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
        assert expected == actual, (
            f"Parameter mismatch: {len(expected)} trainable params in model, "
            f"{len(actual)} in optimizers"
        )

    def __iter__(self) -> Iterator[T]:
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

    def _register_bf16_optimizer_state_hook(self) -> None:
        """Create and restore Adam optimizer states in bfloat16.

        The hook pre-populates optimizer state before Adam's lazy initialization
        runs, so that ``_init_group`` finds non-empty state and skips its own
        fp32 allocation. The fused CUDA kernel then sees the dtype mismatch
        between fp32 params and bf16 states, dispatching to the mixed-precision
        kernel (``FusedAdamMathFunctorMP``).

        A load post-hook reapplies the state dtype after PyTorch's
        ``load_state_dict`` casts floating-point states to the parameter dtype.
        """

        def _bf16_state_init_hook(
            optimizer: Optimizer, args: tuple, kwargs: dict
        ) -> None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = optimizer.state[p]
                    if len(state) == 0:
                        state["step"] = (
                            torch.zeros((), dtype=torch.float32, device=p.device)
                            if group.get("capturable") or group.get("fused")
                            else torch.tensor(0.0, dtype=torch.float32)
                        )
                        state["exp_avg"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, dtype=torch.bfloat16, memory_format=torch.preserve_format
                        )
                        if group.get("amsgrad"):
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p,
                                dtype=torch.bfloat16,
                                memory_format=torch.preserve_format,
                            )

        def _bf16_state_load_hook(optimizer: Optimizer) -> None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p, {})
                    # Keep step's dtype/device policy and any other state intact.
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in state:
                            state[key] = state[key].to(dtype=torch.bfloat16)

        for optim in self.optimizers:
            if isinstance(optim, (torch.optim.Adam, torch.optim.AdamW)):
                optim.register_step_pre_hook(_bf16_state_init_hook)
                optim.register_load_state_dict_post_hook(_bf16_state_load_hook)

    def init_cache_state_dict(self) -> None:
        """Initialize cached state dict for TorchFT. No-op for base class."""
        pass


def default_adamw(lr: float = 8e-4, **kwargs: Any) -> OptimizersContainer.Config:
    """Create an OptimizersContainer.Config with a catch-all AdamW param group.

    Use as a convenience for the common case::

        optimizer=default_adamw(lr=3e-4)
    """
    config_kwargs = {}
    if "implementation" in kwargs:
        config_kwargs["implementation"] = kwargs.pop("implementation")

    return OptimizersContainer.Config(
        param_groups=[
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                    **kwargs,
                },
            )
        ],
        **config_kwargs
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
    optimizers: OptimizersContainer,
    model_parts: list[nn.Module],
    parallel_dims: ParallelDims,
) -> None:
    """Register an optimizer step pre-hook for MoE auxiliary-loss-free load balancing.

    This function checks if MoE load balancing is enabled and, if so, registers
    a hook that updates expert biases before each optimizer step.

    Args:
        optimizers: The optimizers container to register the hook on.
        model_parts: List of model parts that may contain MoE layers.
        parallel_dims: Parallel dimensions for distributed communication.
    """

    def _should_register_moe_balancing_hook(model_parts: list[nn.Module]) -> bool:
        moe_layers = list(_iter_moe_layers(model_parts))
        if not moe_layers:
            return False

        load_balance_enabled = moe_layers[0][1].load_balance_coeff is not None
        for _transformer_block, moe in moe_layers[1:]:
            if (moe.load_balance_coeff is not None) != load_balance_enabled:
                raise ValueError(
                    "MoE load_balance_coeff must be configured consistently "
                    "across all MoE layers. Either set it for every MoE layer "
                    "or leave it unset for all MoE layers."
                )
        return load_balance_enabled

    # for MoE auxiliary-loss-free load balancing
    def _is_recomputation_enabled(module):
        return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

    def _update_expert_bias(
        model_parts: list[nn.Module],
        parallel_dims: ParallelDims,
    ):
        loss_mesh = parallel_dims.get_optional_mesh("loss")
        # TODO: Currently this sync is blocking (thus exposed) and happens on the
        # default compute stream. Need to assess if this is OK performance-wise.
        tokens_per_expert_E_list = []
        for transformer_block, moe in _iter_moe_layers(model_parts):
            tokens_per_expert_E = moe.tokens_per_expert_E
            if _is_recomputation_enabled(transformer_block):
                # TODO: This is a hack, we assume with full AC, the tokens_per_expert_E is counted twice.
                # This does not affect to expert choice, but affects the experts usage metrics.
                # We divide by 2 to correct for this double-counting due to recomputation
                # TODO: new API to help determine if AC is enabled https://github.com/pytorch/pytorch/pull/160888
                tokens_per_expert_E = tokens_per_expert_E // 2
            tokens_per_expert_E_list.append(tokens_per_expert_E)

        if not tokens_per_expert_E_list:
            return

        tokens_per_expert_E_by_layer = torch.vstack(tokens_per_expert_E_list)

        if parallel_dims.ep_enabled and parallel_dims.tp > 1:
            torch.distributed.all_reduce(
                tokens_per_expert_E_by_layer,
                group=parallel_dims.get_dense_tp_mesh().get_group(),
            )
        if loss_mesh is not None:
            torch.distributed.all_reduce(
                tokens_per_expert_E_by_layer,
                group=loss_mesh.get_group(),
                op=torch.distributed.ReduceOp.SUM,
            )
        moe_layer_idx = 0
        with torch.no_grad():
            for _transformer_block, moe in _iter_moe_layers(model_parts):
                load_balance_coeff = moe.load_balance_coeff
                assert load_balance_coeff is not None

                tokens_per_expert_E = tokens_per_expert_E_by_layer[
                    moe_layer_idx
                ].float()
                moe_layer_idx += 1

                # update the expert bias
                # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
                expert_bias_delta_E = load_balance_coeff * torch.sign(
                    tokens_per_expert_E.mean() - tokens_per_expert_E
                )
                expert_bias_delta_E = expert_bias_delta_E - expert_bias_delta_E.mean()
                moe.expert_bias_E.add_(expert_bias_delta_E)
                moe.tokens_per_expert_E.zero_()

    if _should_register_moe_balancing_hook(model_parts):
        optimizers.register_step_pre_hook(
            lambda *args, **kwargs: _update_expert_bias(
                model_parts, parallel_dims=parallel_dims
            )
        )


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
