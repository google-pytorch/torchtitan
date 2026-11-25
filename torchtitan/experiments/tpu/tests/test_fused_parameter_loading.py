# Copyright 2026 Google LLC

"""Tests for loading logical checkpoint entries into fused parameters by hand.

``state_dict()`` of a module that fuses parameters (wq/wk/wv in wqkv, w1/w3 in
w13) exposes the logical names, built by a save hook. A loader that copies
tensors into the model itself, like ``rl_nc_ray``'s CPU loader, must merge the
checkpoint into the fused layout and write into ``named_parameters()``.
"""

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.experiments.tpu.tests import test_packed_ffn
from torchtitan.experiments.tpu.tests import test_packed_qkv
from torchtitan.experiments.tpu.utils import merge_into_fused_parameters


def _fused_and_packed_pairs():
  """(source, target) modules: both layouts of the FFN and of the QKV linear."""
  pairs = {}
  for kind, build in (
      ("ffn", lambda: test_packed_ffn._build_fused_and_packed(bias=True)),
      ("qkv", test_packed_qkv._build_fused_and_packed),
  ):
    source_fused, source_packed = build()
    target_fused, target_packed = build()
    pairs[f"{kind}_fused"] = (source_fused, target_fused)
    pairs[f"{kind}_packed"] = (source_packed, target_packed)
  return pairs


def _logical_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
  return {key: value.clone() for key, value in module.state_dict().items()}


def _zero_parameters(module: torch.nn.Module) -> None:
  with torch.no_grad():
    for parameter in module.parameters():
      parameter.zero_()


def _copy_into_parameters(
    module: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> set[str]:
  """Copies as ``load_initial_hf_weights_on_cpu`` does; returns keys copied."""
  targets = {
      **module.state_dict(),
      **{canonical_fqn(name): p for name, p in module.named_parameters()},
  }
  copied = set()
  with torch.no_grad():
    for key, value in state_dict.items():
      if key in targets:
        targets[key].copy_(value)
        copied.add(key)
  return copied


@pytest.mark.parametrize(
    "kind", ["ffn_fused", "ffn_packed", "qkv_fused", "qkv_packed"]
)
def test_merged_checkpoint_loads_every_fused_parameter(kind):
  source, target = _fused_and_packed_pairs()[kind]
  checkpoint = _logical_state(source)
  _zero_parameters(target)

  merged = merge_into_fused_parameters(target, checkpoint)
  # The checkpoint itself keeps its logical keys.
  assert checkpoint.keys() == _logical_state(source).keys()
  # Every merged entry is a physical parameter, and every parameter has one.
  assert merged.keys() == dict(target.named_parameters()).keys()

  copied = _copy_into_parameters(target, merged)
  assert copied == set(merged)
  test_packed_ffn._assert_state_equal(_logical_state(target), checkpoint)


@pytest.mark.parametrize("kind", ["ffn_fused", "qkv_fused"])
def test_copying_logical_entries_into_state_dict_misses_fused_parameter(kind):
  """Why the merge is needed: upstream's logical entries are copies."""
  source, target = _fused_and_packed_pairs()[kind]
  checkpoint = _logical_state(source)
  _zero_parameters(target)

  logical = target.state_dict()
  with torch.no_grad():
    for key, value in checkpoint.items():
      logical[key].copy_(value)

  fused = [
      parameter
      for name, parameter in target.named_parameters()
      if name not in checkpoint
  ]
  assert fused
  assert all(not parameter.any() for parameter in fused)


def test_merge_uses_canonical_names_under_checkpoint_wrapper():
  source, target = _fused_and_packed_pairs()["ffn_fused"]
  checkpoint = {
      f"layers.0.{key}": value for key, value in _logical_state(source).items()
  }
  _zero_parameters(target)
  model = torch.nn.Module()
  model.layers = torch.nn.ModuleDict({"0": checkpoint_wrapper(target)})
  assert any(
      "_checkpoint_wrapped_module" in name
      for name, _ in model.named_parameters()
  )

  merged = merge_into_fused_parameters(model, checkpoint)
  assert merged.keys() == {
      canonical_fqn(name) for name, _ in model.named_parameters()
  }
  copied = _copy_into_parameters(model, merged)
  assert copied == set(merged)
  test_packed_ffn._assert_state_equal(
      _logical_state(target), _logical_state(source)
  )
