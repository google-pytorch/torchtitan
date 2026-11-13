"""Float8 linear conversion for HuggingFace transformer blocks on TPU."""

import torch
from torch import nn

from torchtitan.tools.logging import logger

_BLOCK_CONTAINER_NAMES = frozenset({"layers", "layer", "blocks", "block", "h"})
_FLOAT8_SKIP_NAME_PARTS = ("head", "embed", "classifier")
_FLOAT8_EXACT_SKIP_PARTS = frozenset({"gate", "router"})
_FUSED_FLOAT8_RECIPE = "fused"


def is_float8_candidate(module: nn.Module, fqn: str) -> bool:
  """Returns True for 16-aligned `nn.Linear` layers inside transformer blocks."""
  if type(module) is not nn.Linear:  # pylint: disable=unidiomatic-typecheck
    return False
  parts = fqn.split(".")
  if not _BLOCK_CONTAINER_NAMES.intersection(parts):
    return False
  if _FLOAT8_EXACT_SKIP_PARTS.intersection(parts):
    return False
  for part in parts:
    if any(skip in part for skip in _FLOAT8_SKIP_NAME_PARTS):
      return False
  return module.in_features % 16 == 0 and module.out_features % 16 == 0


class Fp8FusedLinear(nn.Linear):
  """`nn.Linear` executing `fp8_linear.fp8_linear` in `forward`."""

  def forward(self, input: torch.Tensor) -> torch.Tensor:  # pylint: disable=redefined-builtin
    from torchtitan.experiments.tpu.kernels import fp8_linear  # pylint: disable=g-import-not-at-top

    output = fp8_linear.fp8_linear(input, self.weight)
    if self.bias is not None:
      output = output + self.bias.to(output.dtype)
    return output


def apply_float8_linears(
    model: nn.Module, recipe_name: str, only_modules: str = ""
) -> None:
  """Converts eligible transformer block `nn.Linear` layers to FP8."""
  converted: list[str] = []
  only = tuple(part.strip() for part in only_modules.split(",") if part.strip())
  fused = recipe_name == _FUSED_FLOAT8_RECIPE

  def module_filter_fn(module: nn.Module, fqn: str) -> bool:
    if only and not any(part in fqn for part in only):
      return False
    if not is_float8_candidate(module, fqn):
      return False
    if fused and (module.in_features % 128 or module.out_features % 128):
      return False
    converted.append(fqn)
    return True

  if fused:
    for fqn, module in model.named_modules():
      if module_filter_fn(module, fqn):
        module.__class__ = Fp8FusedLinear
  else:
    from torchao.float8 import convert_to_float8_training  # pylint: disable=g-import-not-at-top
    from torchtitan.experiments.tpu.float8_converter import tpu_float8_linear_config  # pylint: disable=g-import-not-at-top

    convert_to_float8_training(
        model,
        module_filter_fn=module_filter_fn,
        config=tpu_float8_linear_config(recipe_name),
    )

  if not converted:
    align = 128 if fused else 16
    raise ValueError(
        f"hf_float8_recipe={recipe_name!r} converted no linear layers in "
        f"{type(model).__name__}: none sit under a block container "
        f"({sorted(_BLOCK_CONTAINER_NAMES)}) with both dims divisible by "
        f"{align}"
        + (f" and a name containing one of {list(only)}." if only else ".")
    )
  leaves = sorted({fqn.rsplit(".", 1)[-1] for fqn in converted})
  logger.info(
      "Float8 (%s): converted %d linear layers to %s (%s)",
      recipe_name,
      len(converted),
      "Fp8FusedLinear" if fused else "torchao Float8Linear",
      ", ".join(leaves),
  )
