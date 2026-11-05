"""Utility for automatically annotating PyTorch model submodules for profiling."""

from jax import profiler as jax_profiler
import torch


def wrap_model(model: torch.nn.Module) -> None:
  """Wraps all submodules with hooks to add JAX profiler annotations.

  Args:
    model: The model to wrap.
  """
  for name, module in model.named_modules():
    module_name = name if name else type(module).__name__

    if not hasattr(module, "trace_stack"):
      module.trace_stack = []

    def make_pre_hook(mod_name, scope):
      def pre_hook(mod, *args):  # pylint: disable=unused-argument
        cm = jax_profiler.TraceAnnotation(f"{scope}/{mod_name}")
        cm.__enter__()
        mod.trace_stack.append(cm)
        return None

      return pre_hook

    def make_post_hook():
      def post_hook(mod, *args):  # pylint: disable=unused-argument
        if mod.trace_stack:
          cm = mod.trace_stack.pop()
          cm.__exit__(None, None, None)
        return None

      return post_hook

    module.register_forward_pre_hook(make_pre_hook(module_name, "forward"))
    module.register_forward_hook(make_post_hook())

    module.register_full_backward_pre_hook(make_pre_hook(module_name, "backward"))
    module.register_full_backward_hook(make_post_hook())
