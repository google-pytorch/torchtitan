"""HF <-> TorchTitan state-dict conversion for Flint.

The released checkpoint (``RecursiveSI/flint``) stores experts per expert
(``mlp.experts.<e>.{gate,up,down}_proj.weight``) and the sinks as
``self_attn.softmax_offset``. transformers >= 5.21 renames those on load to
fused ``mlp.experts.gate_up_proj`` / ``down_proj`` tensors and
``self_attn.sinks``. ``from_hf`` accepts both layouts; ``to_hf`` emits the
on-disk layout so DCP can load the released safetensors.
"""

import re
from typing import Any

import torch

from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter

_FUSED_EXPERTS = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)")


class FlintStateDictAdapter(Qwen3StateDictAdapter):
  """Qwen3 MoE mapping plus Flint's sinks and ungated shared expert."""

  def __init__(self, model_config, hf_assets_path: str | None) -> None:
    super().__init__(model_config, hf_assets_path)
    self.from_hf_map.update({
        "model.layers.{}.self_attn.softmax_offset": "layers.{}.attention.sinks",
        "model.layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
        "model.layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
        "model.layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
    })

  def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    remaining: dict[str, Any] = {}
    for key, value in hf_state_dict.items():
      match = _FUSED_EXPERTS.fullmatch(key)
      if match is not None:
        layer, name = match.groups()
        prefix = f"layers.{layer}.moe.routed_experts.inner_experts"
        if name == "gate_up_proj":  # [E, 2F, D]: gate rows, then up rows
          w1_EFD, w3_EFD = torch.chunk(value, 2, dim=1)
          state_dict[f"{prefix}.w1_EFD"] = w1_EFD.contiguous()
          state_dict[f"{prefix}.w3_EFD"] = w3_EFD.contiguous()
        else:  # down_proj: [E, D, F]
          state_dict[f"{prefix}.w2_EDF"] = value
        continue
      if key.endswith(".self_attn.sinks"):
        key = key.removesuffix(".sinks") + ".softmax_offset"
      remaining[key] = value
    state_dict.update(super().from_hf(remaining))
    return state_dict
