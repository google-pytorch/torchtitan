"""Compiled optimizer as a separate static XLA graph for PyTorch-on-TPU.

This module compiles `optimizer.step()` into its own standalone static XLA graph
executable via `StaticCompiler`, eliminating 300+ host dispatch roundtrips per
step without merging the optimizer into the forward/backward graph.

Used by:
1. Regular TPUTrainer when `--compile.optimizer-compile=separate_graph`.
2. TpuGraphTrainer when `--compile.optimizer-compile=separate_graph`.
"""

from typing import Any, Optional
import torch
from torch_tpu._internal.compile.compiler import StaticCompiler  # pytype: disable=import-error

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _unwrap_subclasses,
    _wrap_subclasses,
    minimal_fx_tracer,
)
from torchtitan.experiments.tpu.compiled_optimizers.functional import functional_adamw_step
from torchtitan.tools.logging import logger


def _tpu_backend_compile_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
) -> torch.fx.GraphModule:
  output_node = next(n for n in gm.graph.nodes if n.op == "output")
  original_outs = output_node.args[0]
  tensor_outs: list = []
  non_tensor_slots: list[tuple[int, object]] = []
  for i, out in enumerate(original_outs):
    if out is None:
      tensor_outs.append(out)
    elif isinstance(out, torch.fx.Node):
      val = out.meta.get("val")
      if val is None or isinstance(val, torch.Tensor):
        tensor_outs.append(out)
      else:
        non_tensor_slots.append((i, val))
    else:
      non_tensor_slots.append((i, out))
  if non_tensor_slots:
    output_node.args = (tuple(tensor_outs),)
    gm.graph.lint()
    gm.recompile()

  executable = StaticCompiler()(gm, list(example_inputs))

  if non_tensor_slots:

    def _compiled_forward(*flat_inputs):
      outs = list(executable(*flat_inputs))
      for pos, val in non_tensor_slots:
        outs.insert(pos, val)
      return tuple(outs)

  else:

    def _compiled_forward(*flat_inputs):
      return executable(*flat_inputs)

  gm.forward = _compiled_forward
  return gm


def _build_compiled_optimizer_groups(opt: torch.optim.Optimizer) -> list:
  out: list = []
  for group_idx, group in enumerate(opt.param_groups):
    params = [p for p in group["params"] if p.requires_grad]
    if not params:
      continue
    for p in params:
      st = opt.state[p]
      if "exp_avg" not in st:
        st["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
      if "exp_avg_sq" not in st:
        st["exp_avg_sq"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )
      if "step" not in st:
        st["step"] = torch.zeros((), dtype=torch.float32, device=p.device)

    lr_ref = torch.tensor(
        float(group["lr"]), dtype=torch.float32, device=params[0].device
    )
    lr_ref._group_idx = group_idx

    beta1, beta2 = group["betas"]
    wd = group["weight_decay"]
    eps = group["eps"]
    amsgrad = group.get("amsgrad", False)
    assert not amsgrad, "compiled-optimizer path: amsgrad not supported"
    maximize = group.get("maximize", False)
    n_params = len(params)
    is_capturable = params[0].device.type != "cpu"

    # StaticCompiler/minimal_fx_tracer traces a single callable taking a flat
    # dictionary state. `_adamw_step` acts as the unpacking adapter: it unpacks
    # the dictionary keys into ordered lists, delegates the numerical update to
    # `functional_adamw_step`, and closes over group hyperparameters so they are
    # baked as static constants into the compiled XLA graph.
    def _adamw_step(
        state,
        _n=n_params,
        _b1=beta1,
        _b2=beta2,
        _wd=wd,
        _eps=eps,
        _maximize=maximize,
        _capturable=is_capturable,
    ):
      """Tracing adapter: unpacks dict state and calls functional_adamw_step."""
      params_c = [state[f"p_{i}"] for i in range(_n)]
      grads_c = [state[f"g_{i}"] for i in range(_n)]
      exp_avgs_c = [state[f"ea_{i}"] for i in range(_n)]
      exp_avg_sqs_c = [state[f"eas_{i}"] for i in range(_n)]
      steps_c = [state[f"st_{i}"] for i in range(_n)]
      lr_t = state["lr"]
      # Delegate numerical AdamW step math to functional_adamw_step kernel.
      new_params, new_eas, new_eass, new_steps = functional_adamw_step(
          params_c,
          grads_c,
          exp_avgs_c,
          exp_avg_sqs_c,
          steps_c,
          lr_t,
          beta1=_b1,
          beta2=_b2,
          eps=_eps,
          weight_decay=_wd,
          maximize=_maximize,
          foreach=True,
          capturable=True,
      )
      return (*new_params, *new_eas, *new_eass, *new_steps)

    trace_state: dict = {}
    for i, p in enumerate(params):
      trace_state[f"p_{i}"] = p
      g = p.grad if p.grad is not None else torch.zeros_like(p)
      trace_state[f"g_{i}"] = g
      trace_state[f"ea_{i}"] = opt.state[p]["exp_avg"]
      trace_state[f"eas_{i}"] = opt.state[p]["exp_avg_sq"]
      trace_state[f"st_{i}"] = opt.state[p]["step"]
    trace_state["lr"] = lr_ref

    logger.info(
        f"[compiled_opt] tracing AdamW group {group_idx}: {n_params} params"
    )
    traced = minimal_fx_tracer(_adamw_step)(trace_state)
    traced.gm = _tpu_backend_compile_pass(traced.gm, traced.example_inputs)

    state_fqns = list(trace_state.keys())

    def _run(
        live_params,
        live_grads,
        live_exp_avgs,
        live_exp_avg_sqs,
        live_steps,
        live_lr,
        _gm=traced.gm,
        _n=n_params,
        _state_fqns=state_fqns,
        _layouts=traced.output_subclass_layouts,
        _num_flat=traced.num_flat_outputs,
    ):
      d: dict = {}
      for i in range(_n):
        d[f"p_{i}"] = live_params[i]
        d[f"g_{i}"] = live_grads[i]
        d[f"ea_{i}"] = live_exp_avgs[i]
        d[f"eas_{i}"] = live_exp_avg_sqs[i]
        d[f"st_{i}"] = live_steps[i]
      d["lr"] = live_lr

      ordered = [d[k] for k in _state_fqns]
      unwrapped, _ = _unwrap_subclasses(ordered)
      with torch.no_grad():
        flat_out = _gm(*unwrapped)
      wrapped = _wrap_subclasses(flat_out, _num_flat, _layouts)
      return wrapped

    out.append((_run, params, [opt.state[p] for p in params], lr_ref))
  return out


def apply_compiled_optimizer(trainer: Any) -> None:
  """Wraps trainer.optimizers.step() to run AdamW in 1 compiled XLA graph.

  1. Intercepts optimizer.step().
  2. Extracts weight, grad, and optimizer states into flat tensors.
  3. Traces functionalized AdamW math via minimal_fx_tracer into an FX graph.
  4. Lowers the FX graph via StaticCompiler into 1 static XLA binary.
  5. Executes the binary per step and copies updated tensors back in-place.
  """
  compiled_optimizers: list[Optional[list]] = [
      None for _ in trainer.optimizers.optimizers
  ]
  _container = trainer.optimizers

  def _patched_step(closure: Any = None) -> None:
    assert closure is None, "compiled optimizer step does not support closures"
    for i, opt in enumerate(_container.optimizers):
      if not isinstance(opt, torch.optim.AdamW):
        raise TypeError(
            f"apply_compiled_optimizer currently only supports torch.optim.AdamW, "
            f"got {type(opt).__name__}"
        )
      if compiled_optimizers[i] is None:
        compiled_optimizers[i] = _build_compiled_optimizer_groups(opt)
      for fn, params, state_refs, lr_ref in compiled_optimizers[i]:
        grads = [p.grad for p in params]
        if any(g is None for g in grads):
          opt.step()
          continue
        exp_avgs = [s["exp_avg"] for s in state_refs]
        exp_avg_sqs = [s["exp_avg_sq"] for s in state_refs]
        steps = [s["step"] for s in state_refs]
        lr_ref.fill_(float(opt.param_groups[lr_ref._group_idx]["lr"]))
        # Execute compiled XLA executable binary for all params simultaneously
        outs = fn(params, grads, exp_avgs, exp_avg_sqs, steps, lr_ref)
        n = len(params)
        new_params = outs[:n]
        new_exp_avgs = outs[n : 2 * n]
        new_exp_avg_sqs = outs[2 * n : 3 * n]
        new_steps = outs[3 * n : 4 * n]
        # Copy functional outputs back in-place to update model weights
        for p, np_ in zip(params, new_params, strict=True):
          p.data.copy_(np_)
        for ea, nea in zip(exp_avgs, new_exp_avgs, strict=True):
          ea.copy_(nea)
        for eas, neas in zip(exp_avg_sqs, new_exp_avg_sqs, strict=True):
          eas.copy_(neas)
        for s, ns in zip(steps, new_steps, strict=True):
          s.copy_(ns)

  _container.step = _patched_step
  logger.info(
      "[compiled_opt] Separate optimizer-graph compile enabled — patched OptimizersContainer.step"
      f" ({len(_container.optimizers)} inner optimizer(s))."
  )


__all__ = [
    "apply_compiled_optimizer",
]
