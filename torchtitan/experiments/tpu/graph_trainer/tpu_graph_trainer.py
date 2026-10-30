# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TpuGraphTrainer subclass for TPU with optional optimizer compilation.

Note there are 2 ways of compiling the optimizer graph on TPU:
1. Together with fwd+bwd+loss (when ``optimizer_compile='in_graph'``):
   TpuGraphTrainer directly traces the fused step,
   accessing the functional AdamW math and state buffer helpers from
   `torchtitan.experiments.tpu.compiled_optimizers.functional`.
2. As a separate graph (when ``optimizer_compile='separate_graph'`` on either
   TPU trainer): `apply_compiled_optimizer` handles DTensor subclass
   wrapping/unwrapping and compiles `optimizer.step()` as a standalone graph,
   reusing the same functional AdamW math.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import logging
import os
from typing import Any, cast

import torch
from torch import nn

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer.optimizer import (
    collect_moe_routing_metric_tensors,
    set_moe_routing_metric_tensors,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.experiments.tpu import workarounds
from torchtitan.experiments.tpu.compiled_optimizers.separate_graph import (
    apply_compiled_optimizer,
)
from torchtitan.experiments.tpu.grad_norm import CompiledGradNorm
from torchtitan.experiments.tpu.graph_trainer.config import (
    TpuGraphTrainerCompileConfig,
    apply_tpu_graph_trainer_model_transforms,
    maybe_use_maxtext_flops,
)
from torchtitan.experiments.tpu.graph_trainer.graph_split import (
    GraphSplitter,
)
from torchtitan.experiments.tpu.graph_trainer.mfu_benchmark import MfuBenchmark
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    PASS_PIPELINE_REGISTRY,
    construct_tpu_transform_passes,
    register as register_tpu_pipeline,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import logger


class TpuGraphTrainer(GraphTrainer):
    """Branched specialization of GraphTrainer for TPU/PJRT execution.

    Core Extensions and Overrides over GraphTrainer:
    1. __init__:
       - Registers TPU pass pipeline (PASS_PIPELINE_REGISTRY["tpu"]).
       - Rejects chunked loss until it can preserve sequential chunk lifetimes.
       - Hooks TPU-specific metrics and profiling brackets.
       - Initializes compiled optimizer or optimizer-in-graph state.
    2. _make_fx_forward_backward_step (Overridden):
       - Lowers joint fwd+loss+bwd trace applying PASS_PIPELINE_REGISTRY["tpu"] graph passes.
    3. train_step and forward_backward_step (Overridden conditionally):
       - Delegates to super() when standard tracing is used.
       - Uses the fused optimizer path when optimizer_compile='in_graph'.

    New TPU/PJRT Methods:
    - _compiled_optimizer_step: Executes compiled_optimizer groups across meshes.
    - _register_optimizer_buffers: Registers persistent in-graph optstate buffers.
    - _train_step_optimizer_in_graph and _fwd_bwd_adamw_step: Joint
      fwd+loss+bwd+opt execution loop.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainer.Config, TPUTrainerConfig):
        compile: TpuGraphTrainerCompileConfig = field(
            default_factory=TpuGraphTrainerCompileConfig
        )

        def __post_init__(self) -> None:
            if not isinstance(self.compile, TpuGraphTrainerCompileConfig):
                source_compile = self.compile
                source_fields = {
                    f.name for f in dataclasses.fields(source_compile)
                }
                if isinstance(source_compile, GraphTrainerCompileConfig):
                    names = {
                        f.name
                        for f in dataclasses.fields(TpuGraphTrainerCompileConfig)
                    } & source_fields
                else:
                    names = {"enable", "components", "backend"} & source_fields
                values = {
                    name: getattr(source_compile, name) for name in names
                }
                # A generic GraphTrainer pipeline cannot lower a TPU graph.
                values["pass_pipeline"] = "tpu"
                self.compile = TpuGraphTrainerCompileConfig(**values)
            super(TpuGraphTrainer.Config, self).__post_init__()

        @classmethod
        def from_tpu_trainer_config(
            cls,
            base_config: TPUTrainerConfig,
            model_registry: Callable[[str], ModelSpec],
        ) -> TpuGraphTrainer.Config:
            """Convert a TPU recipe to the final GraphTrainer config type.

            The upstream converter copies shared job fields, replaces the model
            spec, and deliberately drops the regular Trainer compile config so
            this class supplies its GraphTrainer-specific default.
            """
            config = cast(
                TpuGraphTrainer.Config,
                to_graph_trainer_config(
                    base_config,
                    model_registry,
                    target_cls=cls,
                ),
            )
            maybe_use_maxtext_flops(base_config, config)
            return config

    def __init__(self, config: Config) -> None:
        """Override of GraphTrainer.__init__ adding TPU pipeline registration, Pallas loss, and opt-in-graph setup."""
        if isinstance(config.loss, ChunkedLossWrapper.Config):
            raise ValueError(
                "TPU GraphTrainer does not support ChunkedLossWrapper because "
                "FX unrolls its loop and defeats bounded-memory execution; use "
                "CrossEntropyLoss or PallasCrossEntropyLoss"
            )
        apply_tpu_graph_trainer_model_transforms(config)
        assert config.model_spec is not None
        parallelize_fn = config.model_spec.parallelize_fn
        assert parallelize_fn is not None

        def tpu_graph_parallelize(model, *args, **kwargs):
            workarounds.apply_patches(model, config)
            kwargs.pop("tpu_compile_mode", None)
            return parallelize_fn(model, *args, **kwargs)

        config.model_spec = dataclasses.replace(
            config.model_spec, parallelize_fn=tpu_graph_parallelize
        )
        # Ensure the 'tpu' pass pipeline is registered before GraphTrainer traces/compiles
        register_tpu_pipeline()

        super().__init__(config)

        if _split_optimizers_by_mesh(self.optimizers):
            self.lr_schedulers = config.lr_scheduler.build(
                optimizers=self.optimizers,
                training_steps=config.training.steps,
            )
            self.metrics_processor.lr_schedulers = self.lr_schedulers

        self._optimizer_in_graph = (
            config.compile.optimizer_compile == "in_graph"
        )
        if self._optimizer_in_graph:
            for opt in self.optimizers.optimizers:
                for group in opt.param_groups:
                    group["capturable"] = True
        self._compiled_grad_norm = (
            CompiledGradNorm() if config.compile.compile_grad_norm else None
        )

        # Only dump weight/buffer statistics when debug logging is explicitly enabled per review comment.
        if logger.isEnabledFor(logging.DEBUG):
            _dump_weight_stats(self.model_parts)

        if self._optimizer_in_graph:
            self._opt_state_registered = False

        self._optimizer_separate_graph = (
            config.compile.optimizer_compile == "separate_graph"
        )
        self._graph_splitter = GraphSplitter.from_trainer_config(
            config,
            optimizer_in_graph=self._optimizer_in_graph,
        )
        if self._optimizer_separate_graph:
            apply_compiled_optimizer(self)

        self._mfu_benchmark = None
        if config.tpu_config.enable_mfu_benchmark:
            self._mfu_benchmark = MfuBenchmark(
                self.metrics_processor,
                start_step=config.tpu_config.mfu_benchmark_start_step,
                final_step=config.training.steps,
            )

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> None:
        """Run the in-graph optimizer loop when that execution mode is selected."""
        if not self._optimizer_in_graph:
            super().train_step(data_iterator)
        else:
            self._train_step_optimizer_in_graph(data_iterator)
        if self._mfu_benchmark is not None:
            self._mfu_benchmark.record(self.step, self.ntokens_seen)

    def _register_optimizer_buffers(self) -> None:
        """Register persistent optstate buffers for the in-graph optimizer."""
        if self._opt_state_registered:
            return
        assert len(self.model_parts) == 1, (
            "optimizer_compile='in_graph' supports exactly one model part."
        )
        model = self.model_parts[0]
        seen_ids: dict[int, str] = {}
        param_fqns: list[str] = []
        fqn_to_canonical: dict[str, str] = {}
        for name, p in model.named_parameters(remove_duplicate=False):
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen_ids:
                fqn_to_canonical[name] = seen_ids[pid]
                continue
            seen_ids[pid] = name
            fqn_to_canonical[name] = name
            param_fqns.append(name)
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            parent.register_buffer(
                f"_optstate_{leaf}_exp_avg",
                torch.zeros_like(p, requires_grad=False),
                persistent=True,
            )
            parent.register_buffer(
                f"_optstate_{leaf}_exp_avg_sq",
                torch.zeros_like(p, requires_grad=False),
                persistent=True,
            )
        model.register_buffer(
            "_optstate_step",
            torch.zeros((), dtype=torch.float32, device=self.device),
            persistent=True,
        )
        lr0 = float(self.lr_schedulers.schedulers[0].get_last_lr()[0])
        model.register_buffer(
            "_optstate_lr",
            torch.tensor(lr0, dtype=torch.float32, device=self.device),
            persistent=False,
        )
        self._optstate_param_fqns = param_fqns
        self._optstate_fqn_to_canonical = fqn_to_canonical
        self._opt_state_registered = True

    def _train_step_optimizer_in_graph(
        self,
        data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]],
    ) -> None:
        """Run the TPU/PJRT joint fwd+loss+bwd+optimizer training loop."""
        from torchtitan.components.loss import IGNORE_INDEX
        from torchtitan.distributed import utils as dist_utils
        from torchtitan.models.common.aux_loss import AuxLoss

        self._register_optimizer_buffers()

        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        parallel_dims = self.parallel_dims

        model = self.model_parts[0]
        model._optstate_lr.fill_(lr)

        should_log = self.metrics_processor.should_log(self.step)
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            batch = next(data_iterator)
            if isinstance(batch, tuple):
                input_dict, labels = batch
                input_dict = dict(input_dict)
                input_dict["labels"] = labels
            else:
                input_dict = dict(batch)
            num_valid = input_dict.pop("num_valid_tokens", None)
            if num_valid is not None:
                local_valid_tokens += int(num_valid)
            else:
                local_valid_tokens += (input_dict["labels"] != IGNORE_INDEX).sum()
            microbatches.append(input_dict)
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens, batch_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens.float()
        AuxLoss.set_step_denominator(global_valid_tokens)

        accumulated_losses = []
        for input_dict in microbatches:
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            loss = self.forward_backward_step(
                input_dict=input_dict,
                global_valid_tokens=global_valid_tokens,  # type: ignore[arg-type]
            )
            accumulated_losses.append(loss.detach())

        self.lr_schedulers.step()
        loss = torch.sum(torch.stack(accumulated_losses))

        if not should_log:
            return

        if parallel_dims.dp_cp_enabled:
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss = dist_utils.dist_sum(loss.detach(), loss_mesh)
            global_max_loss = global_avg_loss
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            0.0,
            extra_metrics={"lr": lr, "n_tokens_seen": self.ntokens_seen},
        )

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, Any] | list[dict[str, Any]],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Include AdamW when the in-graph optimizer mode is selected."""
        from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len

        with set_unflatten_seq_len(self.config.training.max_context_length):
            if not self._optimizer_in_graph:
                return super().forward_backward_step(
                    input_dict=input_dict,
                    global_valid_tokens=global_valid_tokens,
                )
            assert isinstance(input_dict, dict)
            return self._fwd_bwd_adamw_step(
                input_dict=input_dict,
                global_valid_tokens=global_valid_tokens,
            )

    def _make_fx_forward_backward_step(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
        params: list[torch.Tensor],
        extra_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """Override of GraphTrainer._make_fx_forward_backward_step applying TPU pipeline graph passes."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            maybe_register_blockmask_pytree_node,
        )
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            run_traced,
            minimal_fx_tracer,
        )
        from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step
        from torchtitan.experiments.graph_trainer.passes import (
            apply_graph_passes,
        )
        from torchtitan.experiments.graph_trainer.remove_noop_passes import (
            remove_parameter_gradient_markers_pass,
        )
        if self._traced_step is None:
            if self.config.compile.precompile_artifact_dir:
                self._load_precompiled_fx_trace(model)
            else:
                fwd_bwd_fn = make_fwd_bwd_step(
                    model,
                    self.loss_fn,
                    collect_moe_routing_metrics=self._collect_moe_routing_metrics,
                )
                maybe_register_blockmask_pytree_node()
                with self.train_context():
                    self._traced_step = minimal_fx_tracer(
                        fwd_bwd_fn,
                        module=model,
                        prepare_inputs=self._prepare_trace_inputs,
                        prepare_call_inputs=self._prepare_trace_call_inputs,
                    )(
                        inputs,
                        labels,
                        global_valid_tokens,
                        extra_kwargs,
                    )
            if self.config.compile.enable_passes:
                if self._graph_splitter.enabled:
                    # Remat and FSDP transforms annotate the joint graph and must
                    # run before partitioning. The terminal compiler is omitted
                    # here because each child graph is compiled independently below.
                    passes = construct_tpu_transform_passes(
                        self._traced_step, self.config
                    )
                else:
                    pipeline_fn = PASS_PIPELINE_REGISTRY[
                        self.config.compile.pass_pipeline
                    ]
                    passes = pipeline_fn(self._traced_step, self.config)
                self._traced_step.gm = apply_graph_passes(
                    self._traced_step.gm,
                    self._traced_step.example_inputs,
                    passes,
                    compile_config=self.config.compile,
                )
            else:
                self._traced_step.gm = remove_parameter_gradient_markers_pass(
                    self._traced_step.gm, self._traced_step.example_inputs
                )
            if self._graph_splitter.enabled:
                self._traced_step.gm = self._graph_splitter.split_and_compile(
                    self._traced_step
                )
        with self.train_context():
            outputs = run_traced(self._traced_step, module=model)(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
            )
        loss = outputs[0]
        grads = outputs[1 : 1 + len(params)]
        if self._collect_moe_routing_metrics:
            set_moe_routing_metric_tensors(
                [model], list(outputs[1 + len(params) :])
            )

        for param, grad in zip(params, grads, strict=True):
            if param.requires_grad and grad is not None:
                param.grad = grad

        return loss

    def _clip_grad_norm(self) -> torch.Tensor:
        if self._compiled_grad_norm is None:
            return super()._clip_grad_norm()
        parameters = [p for m in self.model_parts for p in m.parameters()]
        return self._compiled_grad_norm.clip_grad_norm_(
            parameters,
            self.config.training.max_norm,
        )

    def _fwd_bwd_adamw_step(
        self,
        *,
        input_dict: dict[str, Any],
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """TPU/PJRT addition: lowers and runs joint forward, backward, and AdamW optimizer FX graph step."""
        from torchtitan.experiments.graph_trainer.common_utils import (
            maybe_register_blockmask_pytree_node,
        )
        from torchtitan.experiments.graph_trainer.make_fx_tracer import (
            run_traced,
            minimal_fx_tracer,
        )
        from torchtitan.experiments.graph_trainer.passes import apply_graph_passes
        from torchtitan.protocols.model import BaseModel
        from typing import cast

        assert len(self.model_parts) == 1
        model = self.model_parts[0]

        inputs, labels, extra_kwargs = cast(BaseModel, model).preprocess_inputs(
            input_dict,
            parallel_dims=self.parallel_dims,
            parallelism=self.config.parallelism,
            max_num_documents=self.dataloader.max_num_documents,
            max_context_length=self.config.training.max_context_length,
        )
        self.ntokens_seen += (
            labels[0].numel() if isinstance(labels, tuple) else labels.numel()
        )

        if self._traced_step is None:
            maybe_register_blockmask_pytree_node()
            param_fqns = list(self._optstate_param_fqns)
            betas = tuple(self.optimizers.optimizers[0].param_groups[0]["betas"])
            wd = float(self.optimizers.optimizers[0].param_groups[0]["weight_decay"])
            eps = float(self.optimizers.optimizers[0].param_groups[0]["eps"])
            max_norm_val = getattr(self.config.training, "max_norm", None)
            max_norm = float(max_norm_val) if max_norm_val is not None else None

            fwd_bwd_adamw = _make_fwd_bwd_adamw_fn(
                model,
                loss_fn=self.loss_fn,
                param_fqns=param_fqns,
                beta1=float(betas[0]),
                beta2=float(betas[1]),
                eps=eps,
                weight_decay=wd,
                max_norm=max_norm,
                collect_moe_routing_metrics=self._collect_moe_routing_metrics,
            )

            with self.train_context():
                self._traced_step = minimal_fx_tracer(
                    fwd_bwd_adamw,
                    module=model,
                )(
                    inputs,
                    labels,
                    global_valid_tokens,
                    extra_kwargs,
                )
            pipeline_fn = PASS_PIPELINE_REGISTRY[
                self.config.compile.pass_pipeline
            ]
            passes = pipeline_fn(self._traced_step, self.config)
            self._traced_step.gm = apply_graph_passes(
                self._traced_step.gm,
                self._traced_step.example_inputs,
                passes,
                compile_config=self.config.compile,
            )

        with self.train_context():
            outputs = run_traced(self._traced_step, module=model)(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
            )

        loss = outputs[0]
        n = len(self._optstate_param_fqns)
        new_params = outputs[1 : 1 + n]
        new_exp_avgs = outputs[1 + n : 1 + 2 * n]
        new_exp_avg_sqs = outputs[1 + 2 * n : 1 + 3 * n]
        new_step = outputs[1 + 3 * n]
        if self._collect_moe_routing_metrics:
            set_moe_routing_metric_tensors(
                [model], list(outputs[2 + 3 * n :])
            )

        from torchtitan.experiments.tpu.compiled_optimizers.functional import update_model_and_optstate_buffers

        update_model_and_optstate_buffers(
            model,
            self._optstate_param_fqns,
            new_params,
            new_exp_avgs,
            new_exp_avg_sqs,
        )
        model._optstate_step.copy_(
            new_step.detach() if isinstance(new_step, torch.Tensor) else new_step
        )

        return loss


_GRAD_CLIP_EPS = 1e-6
_GRAD_CLIP_MAX_SCALE = 1.0


def _make_fwd_bwd_adamw_fn(
    model: nn.Module,
    *,
    loss_fn,
    param_fqns: list[str],
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    max_norm: float | None = None,
    collect_moe_routing_metrics: bool = False,
    capturable: bool = True,
):
    from torchtitan.experiments.graph_trainer.common_utils import (
        compute_annotated_loss,
        compute_parameter_gradients,
    )
    from torchtitan.experiments.tpu.compiled_optimizers.functional import (
        _get_param_by_fqn,
        _get_optstate_buffer,
        functional_adamw_step,
    )

    def fwd_bwd_adamw(
        inputs, labels, global_valid_tokens, extra_kwargs
    ):
        pred = model(inputs, **extra_kwargs)
        routing_metrics = (
            collect_moe_routing_metric_tensors([model])
            if collect_moe_routing_metrics
            else []
        )
        loss = compute_annotated_loss(
            loss_fn,
            pred,
            labels,
            {"global_valid_tokens": global_valid_tokens},
        )

        canonical_params: list[torch.Tensor] = [
            _get_param_by_fqn(model, fqn) for fqn in param_fqns
        ]
        grads = compute_parameter_gradients(
            loss, zip(param_fqns, canonical_params, strict=True)
        )

        if max_norm is not None and max_norm > 0.0:
            total_norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g) for g in grads]),
                2.0,
            )
            clip_coef = max_norm / (total_norm + _GRAD_CLIP_EPS)
            clip_coef_clamped = torch.clamp(clip_coef, max=_GRAD_CLIP_MAX_SCALE)
            grads = tuple(g * clip_coef_clamped for g in grads)

        exp_avgs = [
            _get_optstate_buffer(model, fqn, "exp_avg") for fqn in param_fqns
        ]
        exp_avg_sqs = [
            _get_optstate_buffer(model, fqn, "exp_avg_sq") for fqn in param_fqns
        ]
        step_t = model._optstate_step
        lr_t = model._optstate_lr

        n = len(canonical_params)
        steps = [step_t for _ in range(n)]
        new_params, new_eas, new_eass, new_steps = functional_adamw_step(
            canonical_params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            steps,
            lr_t,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            maximize=False,
            foreach=False,
            capturable=capturable,
        )
        return [
            loss,
            *new_params,
            *new_eas,
            *new_eass,
            new_steps[0],
            *routing_metrics,
        ]

    return fwd_bwd_adamw


def _split_optimizers_by_mesh(optimizers_container) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        DTensor = None

    def _mesh_key(p):
        if DTensor is None or not isinstance(p, DTensor):
            return None
        try:
            return id(p._spec.mesh)
        except Exception:
            return None

    did_split = False
    for opt in optimizers_container.optimizers:
        new_groups: list[dict] = []
        for grp in opt.param_groups:
            params = list(grp["params"])
            base_kwargs = {k: v for k, v in grp.items() if k != "params"}
            buckets: dict = {}
            for p in params:
                key = _mesh_key(p)
                buckets.setdefault(key, []).append(p)
            if len(buckets) > 1:
                did_split = True
            for bucket_params in buckets.values():
                new_groups.append({"params": bucket_params, **base_kwargs})
        if len(new_groups) == len(opt.param_groups):
            continue
        opt.param_groups = []
        opt.state.clear()
        for grp in new_groups:
            opt.add_param_group(grp)
    return did_split


def _dump_weight_stats(model_parts) -> None:
  rank = int(os.environ.get("RANK", "0"))
  if rank != 0:
    return
  try:
    from torch.distributed.tensor import DTensor
  except Exception:
    DTensor = None
  logger.debug("[debug_weights] === post-init weight stats (rank 0) ===")
  for model in model_parts:
    for name, p in model.named_parameters():
      t = p
      if DTensor is not None and isinstance(p, DTensor):
        t = p._local_tensor
      try:
        t_cpu = t.detach().to("cpu").float()
        has_nan = bool(torch.isnan(t_cpu).any().item())
        has_inf = bool(torch.isinf(t_cpu).any().item())
        lo = float(t_cpu.min().item())
        hi = float(t_cpu.max().item())
        me = float(t_cpu.mean().item())
        st = float(t_cpu.std().item())
        logger.debug(
                    f"[debug_weights]   {name}: shape={tuple(t.shape)} "
                    f"min={lo:+.4f} max={hi:+.4f} mean={me:+.5f} std={st:.5f} "
                    f"nan={has_nan} inf={has_inf}"
                )
      except Exception as e:
        logger.debug(f"[debug_weights]   {name}: <error {type(e).__name__}: {e}>")
  for model in model_parts:
    for name, buf in model.named_buffers():
      t = buf
      if DTensor is not None and isinstance(buf, DTensor):
        t = buf._local_tensor
      try:
        t_cpu = t.detach().to("cpu").float()
        has_nan = bool(torch.isnan(t_cpu).any().item())
        lo = float(t_cpu.min().item())
        hi = float(t_cpu.max().item())
        logger.debug(
                    f"[debug_weights]   buf {name}: shape={tuple(t.shape)} "
                    f"min={lo:+.4f} max={hi:+.4f} nan={has_nan}"
                )
      except Exception as e:
        logger.debug(f"[debug_weights]   buf {name}: <error {type(e).__name__}: {e}>")
  logger.debug("[debug_weights] === end ===")
