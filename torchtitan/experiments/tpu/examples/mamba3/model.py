# Copyright 2026 Google LLC

"""Mamba-3 Neural Architecture for TorchTitan TPU experiments."""

# pylint: disable=invalid-name,g-import-not-at-top

import dataclasses
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.experiments.tpu.examples.mamba3.kernels import mamba3_pallas

try:
  from torchtitan.protocols import model as model_protocol
  from torchtitan.protocols import module
  _BaseModel = model_protocol.BaseModel
  _BaseConfig = model_protocol.BaseModel.Config
  _Module = module.Module
except (ImportError, ModuleNotFoundError):
  _BaseConfig = object
  _Module = nn.Module

  class _BaseModel(nn.Module):
    pass


@dataclasses.dataclass(kw_only=True, slots=True)
class Mamba3Config(_BaseConfig):
  """Configuration for Mamba-3 architecture."""

  vocab_size: int = 128256
  d_model: int = 2560
  n_layers: int = 64
  n_heads: int = 80
  d_state: int = 128
  expand: int = 2
  chunk_size: int = 256
  rms_norm_eps: float = 1e-5
  tie_embeddings: bool = True
  enable_weight_tying: bool = True
  ssm_backend: str = "pallas"
  is_mimo: bool = False
  mimo_rank: int = 4
  rope_fraction: float = 1.0
  A_floor: float = 1e-4
  dt_min: float = 0.001
  dt_max: float = 0.1
  dt_init_floor: float = 1e-4

  @property
  def d_inner(self) -> int:
    return self.expand * self.d_model

  @property
  def head_dim(self) -> int:
    return self.d_inner // self.n_heads

  def update_from_config(self, *, config, **kwargs) -> None:
    pass

  def build(self):
    return Mamba3LM(self)

  def get_nparams_and_flops(
      self, model: nn.Module, seq_len: int
  ) -> Tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())
    num_flops_per_token = 6 * nparams
    return nparams, num_flops_per_token


class Mamba3Mixer(_Module):
  """Mamba-3 SSM Mixer block powered by Pallas custom TPU kernel."""

  def __init__(self, config: Mamba3Config):
    super().__init__()
    self.config = config
    self.d_model = config.d_model
    self.d_inner = config.d_inner
    self.n_heads = config.n_heads
    self.head_dim = config.head_dim
    self.d_state = config.d_state
    self.chunk_size = getattr(config, "chunk_size", 256)
    self.is_mimo = getattr(config, "is_mimo", False)
    self.A_floor = getattr(config, "A_floor", 1e-4)

    dt_min = getattr(config, "dt_min", 0.001)
    dt_max = getattr(config, "dt_max", 0.1)
    dt_init_floor = getattr(config, "dt_init_floor", 1e-4)

    if self.is_mimo:
      self.mimo_rank = config.mimo_rank
      self.rope_fraction = getattr(config, "rope_fraction", 0.5)
      self.num_rope_angles = int(self.d_state // 2 * self.rope_fraction)
      d_in_proj = (
          2 * self.d_inner
          + 2 * self.mimo_rank * self.d_state
          + 3 * self.n_heads
          + self.num_rope_angles
      )
      self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
      self.B_bias = nn.Parameter(
          torch.ones(self.n_heads, self.mimo_rank, self.d_state)
      )
      self.C_bias = nn.Parameter(
          torch.ones(self.n_heads, self.mimo_rank, self.d_state)
      )
      self.Psi = nn.Parameter(
          torch.ones(self.n_heads, self.mimo_rank, self.head_dim)
          / self.mimo_rank
      )
      self.Phi = nn.Parameter(
          torch.ones(self.n_heads, self.mimo_rank, self.head_dim)
          / self.mimo_rank
      )
      self.Zeta = nn.Parameter(
          torch.ones(self.n_heads, self.mimo_rank, self.head_dim)
      )
    else:
      self.num_rope_angles = self.d_state // 2
      d_in_proj = (
          2 * self.d_inner
          + 2 * self.d_state
          + 3 * self.n_heads
          + self.num_rope_angles
      )
      self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False)
      self.B_bias = nn.Parameter(torch.ones(self.n_heads, self.d_state))
      self.C_bias = nn.Parameter(torch.ones(self.n_heads, self.d_state))

    _dt = torch.exp(
        torch.rand(self.n_heads)
        * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min)))
        + torch.log(torch.tensor(dt_min))
    )
    _dt = torch.clamp(_dt, min=dt_init_floor)
    self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))

    self.B_norm_weight = nn.Parameter(torch.ones(self.d_state))
    self.C_norm_weight = nn.Parameter(torch.ones(self.d_state))
    self.D = nn.Parameter(torch.ones(self.n_heads))
    self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

  @staticmethod
  def _rmsnorm(
      x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
  ) -> torch.Tensor:
    x32 = x.float()
    rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x32 * rms).to(x.dtype) * weight

  def forward(self, u: torch.Tensor) -> torch.Tensor:
    b, t, _ = u.shape

    # 1. Project inputs
    projected = self.in_proj(u)
    if self.is_mimo:
      z, v, B, C, dd_dt, dd_A, trap_raw, angle_raw = torch.split(
          projected,
          [
              self.d_inner,
              self.d_inner,
              self.mimo_rank * self.d_state,
              self.mimo_rank * self.d_state,
              self.n_heads,
              self.n_heads,
              self.n_heads,
              self.num_rope_angles,
          ],
          dim=-1,
      )
      v = v.reshape(b, t, self.n_heads, self.head_dim)
      B = self._rmsnorm(
          B.reshape(b, t, self.mimo_rank, self.d_state), self.B_norm_weight
      )
      C = self._rmsnorm(
          C.reshape(b, t, self.mimo_rank, self.d_state), self.C_norm_weight
      )
      B = B[:, :, None, :, :].expand(
          b, t, self.n_heads, self.mimo_rank, self.d_state
      )
      C = C[:, :, None, :, :].expand(
          b, t, self.n_heads, self.mimo_rank, self.d_state
      )
    else:
      z, v, B, C, dd_dt, dd_A, trap_raw, angle_raw = torch.split(
          projected,
          [
              self.d_inner,
              self.d_inner,
              self.d_state,
              self.d_state,
              self.n_heads,
              self.n_heads,
              self.n_heads,
              self.num_rope_angles,
          ],
          dim=-1,
      )
      v = v.reshape(b, t, self.n_heads, self.head_dim)
      B = self._rmsnorm(B, self.B_norm_weight)[:, :, None, :].expand(
          b, t, self.n_heads, self.d_state
      )
      C = self._rmsnorm(C, self.C_norm_weight)[:, :, None, :].expand(
          b, t, self.n_heads, self.d_state
      )

    angle_raw = angle_raw[:, :, None, :].expand(
        b, t, self.n_heads, self.num_rope_angles
    )
    A = -F.softplus(dd_A.float())
    A = torch.clamp(A, max=-self.A_floor)
    dt = F.softplus(dd_dt + self.dt_bias)
    adt = A * dt

    # 2. Custom Fused Pallas SSD-3 Scan
    y = mamba3_pallas.ssd3_chunk_scan_pallas(
        v=v,
        b=B,
        c=C,
        b_bias=self.B_bias,
        c_bias=self.C_bias,
        adt=adt,
        dt=dt,
        trap_raw=trap_raw,
        angle_raw=angle_raw,
        d=self.D,
        chunk_size=self.chunk_size,
        psi=self.Psi if self.is_mimo else None,
        phi=self.Phi if self.is_mimo else None,
        zeta=self.Zeta if self.is_mimo else None,
        z=z if self.is_mimo else None,
        is_mimo=self.is_mimo,
    )

    # 3. Gating & Output Projection
    if not self.is_mimo:
      y = y.reshape(b, t, self.d_inner) * F.silu(z)
    else:
      y = y.reshape(b, t, self.d_inner)
    return self.out_proj(y)


class Mamba3Block(_Module):
  """A single residual block containing Mamba-3 mixer."""

  def __init__(self, config: Mamba3Config):
    super().__init__()
    self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
    self.mixer = Mamba3Mixer(config)

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return x + self.mixer(self.norm(x))


class Mamba3LM(_BaseModel):
  """Full Language Model with Mamba-3 backbone."""

  def __init__(self, config: Mamba3Config):
    super().__init__()
    self.config = config
    self.enable_weight_tying = getattr(
        config, "enable_weight_tying", config.tie_embeddings
    )
    self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
    self.embedding = self.tok_embeddings
    self.layers = nn.ModuleDict(
        {str(i): Mamba3Block(config) for i in range(config.n_layers)}
    )
    self.norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
    self.norm_f = self.norm
    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    if self.enable_weight_tying:
      self.lm_head.weight = self.tok_embeddings.weight

  def verify_module_protocol(self) -> None:
    pass

  def init_weights(self, buffer_device=None, **kwargs) -> None:
    if self.enable_weight_tying:
      self.lm_head.weight = self.tok_embeddings.weight

    with torch.no_grad():
      for name, param in self.named_parameters():
        target = param.to_local() if hasattr(param, "to_local") else param.data
        if "norm" in name and "weight" in name:
          target.fill_(1.0)
        elif "B_bias" in name or "C_bias" in name:
          target.fill_(1.0)
        elif "dt_bias" in name:
          dt_min = getattr(self.config, "dt_min", 0.001)
          dt_max = getattr(self.config, "dt_max", 0.1)
          dt_floor = getattr(self.config, "dt_init_floor", 1e-4)
          _dt = torch.exp(
              torch.rand(self.config.n_heads, device=target.device)
              * (
                  torch.log(torch.tensor(dt_max, device=target.device))
                  - torch.log(torch.tensor(dt_min, device=target.device))
              )
              + torch.log(torch.tensor(dt_min, device=target.device))
          )
          _dt = torch.clamp(_dt, min=dt_floor)
          init_val = _dt + torch.log(-torch.expm1(-_dt))
          target.copy_(init_val.view_as(target))
        elif "bias" in name:
          target.zero_()
        elif name.endswith(".D") or name == "D":
          target.fill_(1.0)
        elif "Psi" in name or "Phi" in name:
          mimo_r = getattr(self.config, "mimo_rank", 1)
          target.fill_(1.0 / mimo_r)
        elif "Zeta" in name:
          target.fill_(1.0)
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
