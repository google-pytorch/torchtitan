# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Mamba-2 Neural Architecture for TorchTitan TPU experiments."""

# pylint: disable=invalid-name

import dataclasses
from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.examples.mamba2.kernels import mamba2_pallas
from torchtitan.protocols import model as model_protocol
from torchtitan.protocols import module


class GatedRMSNorm(module.Module):

  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps) * F.silu(z)


@dataclasses.dataclass(kw_only=True, slots=True)
class Mamba2Config(model_protocol.BaseModel.Config):
  """Configuration for Mamba-2 architecture."""

  vocab_size: int = 128256
  d_model: int = 768
  n_layers: int = 24
  n_heads: int = 24
  n_groups: Optional[int] = None
  d_state: int = 16
  d_conv: int = 4
  expand: int = 2
  chunk_size: int = 64
  rms_norm_eps: float = 1e-5
  tie_embeddings: bool = True
  enable_weight_tying: bool = True
  ssm_backend: str = "pallas"

  @property
  def d_inner(self) -> int:
    return self.expand * self.d_model

  @property
  def head_dim(self) -> int:
    return self.d_inner // self.n_heads

  def update_from_config(self, *, config, **kwargs) -> None:
    pass

  def build(self):
    return Mamba2LM(self)

  def get_nparams_and_flops(
      self, model: nn.Module, seq_len: int
  ) -> Tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())
    num_flops_per_token = 6 * nparams
    return nparams, num_flops_per_token


class Mamba2Mixer(module.Module):
  """Mamba-2 SSD Mixer block powered by Pallas custom TPU kernel."""

  def __init__(self, config: Mamba2Config):
    super().__init__()
    self.d_model = config.d_model
    self.d_inner = config.d_inner
    self.n_heads = config.n_heads
    self.n_groups = (
        config.n_groups if config.n_groups is not None else config.n_heads
    )
    self.head_dim = config.head_dim
    self.d_state = config.d_state
    self.d_conv = config.d_conv
    self.chunk_size = config.chunk_size

    # In-projections for [z, x, B, C, dt]
    self.in_proj = nn.Linear(
        self.d_model,
        2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads,
        bias=False,
    )

    # 1D Causal Depthwise Convolution
    self.conv1d = nn.Conv1d(
        in_channels=self.d_inner + 2 * self.n_groups * self.d_state,
        out_channels=self.d_inner + 2 * self.n_groups * self.d_state,
        kernel_size=self.d_conv,
        groups=self.d_inner + 2 * self.n_groups * self.d_state,
        padding=self.d_conv - 1,
        bias=True,
    )

    # Learnable continuous-time decay parameters
    self.A_log = nn.Parameter(
        torch.log(torch.arange(1, self.n_heads + 1).float())
    )
    self.D = nn.Parameter(torch.ones(self.n_heads, self.head_dim))

    # Fused Gated RMSNorm
    self.norm = GatedRMSNorm(self.d_inner, eps=config.rms_norm_eps)
    self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    b, l, _ = u.shape

    # 1. Project inputs
    projected = self.in_proj(u)
    z, xBC, dt = torch.split(
        projected,
        [
            self.d_inner,
            self.d_inner + 2 * self.n_groups * self.d_state,
            self.n_heads,
        ],
        dim=-1,
    )

    # 2. Causal 1D Convolution & SiLU activation
    xBC_conv = self.conv1d(xBC.transpose(1, 2))[:, :, :l].transpose(1, 2)
    xBC_act = F.silu(xBC_conv)

    x, B, C = torch.split(
        xBC_act,
        [
            self.d_inner,
            self.n_groups * self.d_state,
            self.n_groups * self.d_state,
        ],
        dim=-1,
    )

    x = x.view(b, l, self.n_heads, self.head_dim)
    B = B.view(b, l, self.n_groups, self.d_state)
    C = C.view(b, l, self.n_groups, self.d_state)
    dt = F.softplus(dt)

    # 3. Custom Fused Pallas SSD Scan
    y = mamba2_pallas.ssd_chunk_scan_pallas(
        x,
        dt,
        self.A_log,
        B,
        C,
        d=self.D,
        chunk_size=self.chunk_size,
    )

    # 4. Fused Gated Normalization & Output Projection
    y_flat = y.reshape(b, l, self.d_inner)
    y_norm = self.norm(y_flat, z)
    return self.out_proj(y_norm)


class Mamba2Block(module.Module):
  """A single residual block containing Mamba-2 mixer."""

  def __init__(self, config: Mamba2Config):
    super().__init__()
    self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
    self.mixer = Mamba2Mixer(config)

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return x + self.mixer(self.norm(x))


class Mamba2LM(model_protocol.BaseModel):
  """Full Language Model with Mamba-2 backbone."""

  def __init__(self, config: Mamba2Config):
    super().__init__()
    self.config = config
    self.enable_weight_tying = getattr(
        config, "enable_weight_tying", config.tie_embeddings
    )
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
    self.embedding = self.tok_embeddings
    self.layers = nn.ModuleDict(
        {str(i): Mamba2Block(config) for i in range(config.n_layers)}
    )
    self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
    self.norm_f = self.norm
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    if self.enable_weight_tying:
      self.lm_head.weight = self.tok_embeddings.weight

  def verify_module_protocol(self) -> None:
    """Verify module protocol compatibility."""
    pass

  def init_weights(self, buffer_device=None, **kwargs) -> None:
    """Initialize model weights with DTensor support."""
    if self.enable_weight_tying:
      self.lm_head.weight = self.tok_embeddings.weight

    with torch.no_grad():
      for name, param in self.named_parameters():
        target = param.to_local() if hasattr(param, "to_local") else param.data
        if name.endswith(".A_log") or name == "A_log":
          init_val = torch.log(
              torch.arange(1, self.config.n_heads + 1).float()
          ).to(target.device)
          if target.numel() == init_val.numel():
            target.copy_(init_val.view_as(target))
          else:
            target.fill_(0.0)
        elif name.endswith(".D") or name == "D":
          target.fill_(1.0)
        elif "norm" in name and "weight" in name:
          target.fill_(1.0)
        elif "bias" in name:
          target.zero_()
        elif "weight" in name:
          nn.init.normal_(target, mean=0.0, std=0.02)

  def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # TT-UNFLATTEN-SHIM:
    input_ids = unflatten_shim.unflatten_tensor(input_ids, expected_flat_ndim=1)

    h = self.tok_embeddings(input_ids)
    for layer in self.layers.values():
      h = layer(h)
    h = self.norm(h)
    out = self.lm_head(h)

    # TT-UNFLATTEN-SHIM:
    return unflatten_shim.flatten_tensor(out, original_flat_ndim=2)


# Aliases for backwards compatibility
PallasMamba2Config = Mamba2Config
PallasMamba2LM = Mamba2LM
PallasMamba2Block = Mamba2Block
PallasMamba2Mixer = Mamba2Mixer
