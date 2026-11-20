"""Conformer model for TorchTitan."""

import logging
from dataclasses import dataclass
import math
import einops
import torch
import torch.nn as nn
from torchtitan.experiments.tpu.kernels.splash_attention import splash_sdpa
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
from torchtitan.protocols.model import BaseModel

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ConformerModelArgs(BaseModel.Config):
  """Arguments for Conformer model configuration."""

  vocab_size: int = 64
  hidden_dim: int = 512
  num_layers: int = 17
  num_heads: int = 8
  kernel_size: int = 31
  max_seq_len: int = 2048  # Added to match train_minimal expectations
  use_splash: bool = False
  use_vmap_bwd: bool = False

  def update_from_config(self, train_config, **kwargs):
    # Update from train_config if needed
    if hasattr(train_config, "training"):
      # Upstream d771abb0 renamed training.seq_len ->
      # training.max_context_length.
      self.max_seq_len = train_config.training.max_context_length
    if isinstance(train_config, TPUTrainerConfig):
      self.use_splash = (
          train_config.splash_attention_kernel.use_splash_attention_kernel
      )
      self.use_vmap_bwd = (
          train_config.splash_attention_kernel.use_vmap_bwd
      )

  def get_nparams_and_flops(
      self, model: nn.Module, seq_len: int
  ) -> tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())

    t = seq_len
    d = self.hidden_dim
    k = self.kernel_size
    ffn_dim = 4 * d  # Based on ffn_dim=4*config.hidden_dim
    l = self.num_layers

    # 1. Feed Forward Networks (one in beginning, one at end of each layer)
    # 2 linear layers per FFN (D -> FFN_dim, and FFN_dim -> D).
    # MatMul FLOPs = 2 * T * in_dim * out_dim
    ffn_flops = 2 * (2 * t * d * ffn_dim) * 2

    # 2. Attention Module (PatchedMHA)
    # QKV Projections + Output Projection: 4 * (2 * T * D^2)
    # QK^T and AV Matrix Multiplications: 2 * (2 * T^2 * D)
    attn_flops = (8 * t * (d**2)) + (4 * (t**2) * d)

    # 3. Convolution Module
    # Pointwise 1 (D -> 2D) + Pointwise 2 (D -> D) = 6 * T * D^2
    # Depthwise Conv: 2 * T * D * K
    conv_flops = (6 * t * (d**2)) + (2 * t * d * k)

    # Total FLOPs for all Conformer layers
    flops_per_layer = ffn_flops + attn_flops + conv_flops
    total_layer_flops = l * flops_per_layer

    # Final FC classifier layer
    fc_flops = 2 * t * d * self.vocab_size

    # Total Forward FLOPs
    forward_flops = total_layer_flops + fc_flops

    # Total Training FLOPs = Forward (1x) + Backward (2x) = 3x total
    total_training_flops = forward_flops * 3

    # Return FLOPs normalized per token to match downstream metric processor
    flops_per_token = total_training_flops // t

    return nparams, int(flops_per_token)


# --- TPU-compatible GLU Fix ---
def tpu_glu(input, dim=-1):
  chunks = input.chunk(2, dim)
  return chunks[0] * torch.sigmoid(chunks[1])


class TPUGLU(torch.nn.Module):

  def __init__(self, dim=-1):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    return tpu_glu(x, self.dim)


# TPU Patch: PatchedMHA to force Flash Attention
class PatchedMHA(nn.Module):

  def __init__(self, embed_dim, num_heads, dropout=0.0, use_splash=False, use_vmap_bwd=False):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.head_dim = embed_dim // num_heads
    self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
    self.use_splash = use_splash
    self.use_vmap_bwd = use_vmap_bwd

  def forward(
      self,
      query,
      key,
      value,
      key_padding_mask=None,
      need_weights=True,
      attn_mask=None,
      average_attn_weights=True,
      is_causal=False,
  ):
    _, _, hidden_dim = query.shape
    w_q, w_k, w_v = self.in_proj.weight.chunk(3, dim=0)
    w_q_flat = w_q.reshape(self.embed_dim, hidden_dim)
    w_k_flat = w_k.reshape(self.embed_dim, hidden_dim)
    w_v_flat = w_v.reshape(self.embed_dim, hidden_dim)

    q = einops.rearrange(
        torch.matmul(query, w_q_flat.t()),
        "b t (h d) -> b h t d",
        h=self.num_heads,
    ).contiguous()
    k = einops.rearrange(
        torch.matmul(query, w_k_flat.t()),
        "b t (h d) -> b h t d",
        h=self.num_heads,
    ).contiguous()
    v = einops.rearrange(
        torch.matmul(query, w_v_flat.t()),
        "b t (h d) -> b h t d",
        h=self.num_heads,
    ).contiguous()

    if self.use_splash:
      attn_output = splash_sdpa(q, k, v, is_causal=is_causal, use_vmap_bwd=self.use_vmap_bwd)
    else:
      attn_output = torch.nn.functional.scaled_dot_product_attention(
          q,
          k,
          v,
          attn_mask=None,
          dropout_p=self.dropout if self.training else 0.0,
          is_causal=is_causal,
      )

    w = self.out_proj.weight.view(self.out_proj.out_features, self.num_heads, self.head_dim)

    # Vectorized mh projection via einsum
    attn_output = torch.einsum("bhtd,ohd->bto", attn_output, w)
    if self.out_proj.bias is not None:
      attn_output = attn_output + self.out_proj.bias
    return attn_output, None


class _FeedForwardModule(nn.Module):

  def __init__(self, dim, hidden_dim, dropout_p=0.0):
    super().__init__()
    self.sequential = nn.Sequential(
        nn.LayerNorm(dim, eps=1e-05),
        nn.Linear(dim, hidden_dim, bias=True),
        nn.SiLU(),
        nn.Dropout(p=dropout_p),
        nn.Linear(hidden_dim, dim, bias=True),
        nn.Dropout(p=dropout_p),
    )

  def forward(self, x):
    return self.sequential(x)


class _ConvolutionModule(nn.Module):

  def __init__(self, dim, kernel_size=31, dropout_p=0.0):
    super().__init__()
    padding = kernel_size // 2
    self.layer_norm = nn.LayerNorm(dim, eps=1e-05)
    self.sequential = nn.Sequential(
        nn.Conv1d(dim, 2 * dim, kernel_size=1, stride=1),
        TPUGLU(dim=1),
        nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
        ),
        nn.BatchNorm1d(dim, eps=1e-05, momentum=0.1),
        nn.SiLU(),
        nn.Conv1d(dim, dim, kernel_size=1, stride=1),
        nn.Dropout(p=dropout_p),
    )

  def forward(self, x):
    x = self.layer_norm(x)
    x = x.transpose(1, 2)
    x = self.sequential(x)
    x = x.transpose(1, 2)
    return x


class ConformerLayer(nn.Module):

  def __init__(
      self,
      dim,
      num_heads,
      ffn_dim,
      kernel_size=31,
      dropout_p=0.0,
      use_splash=False,
      use_vmap_bwd=False,
  ):
    super().__init__()
    self.ffn1 = _FeedForwardModule(dim, ffn_dim, dropout_p)
    self.self_attn_layer_norm = nn.LayerNorm(dim, eps=1e-05)
    # Use PatchedMHA instead of nn.MultiheadAttention
    self.self_attn = PatchedMHA(
        dim, num_heads, dropout=dropout_p, use_splash=use_splash, use_vmap_bwd=use_vmap_bwd
    )
    self.self_attn_dropout = nn.Dropout(p=dropout_p)
    self.conv_module = _ConvolutionModule(dim, kernel_size, dropout_p)
    self.ffn2 = _FeedForwardModule(dim, ffn_dim, dropout_p)
    self.final_layer_norm = nn.LayerNorm(dim, eps=1e-05)

  def forward(self, x):
    x = x + 0.5 * self.ffn1(x)
    residual = x
    x = self.self_attn_layer_norm(x)
    x, _ = self.self_attn(x, x, x)
    x = residual + self.self_attn_dropout(x)
    x = x + self.conv_module(x)
    x = x + 0.5 * self.ffn2(x)
    x = self.final_layer_norm(x)
    return x


class Conformer(BaseModel):
  """Build a conformer model matching torchaudio.models.Conformer architecture."""

  Config = ConformerModelArgs

  def __init__(self, config: ConformerModelArgs) -> None:
    super().__init__()
    self.config = config
    self._model_args = config
    self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

    self.conformer_layers = nn.ModuleList([
        ConformerLayer(
            dim=config.hidden_dim,
            num_heads=config.num_heads,
            ffn_dim=4 * config.hidden_dim,
            kernel_size=config.kernel_size,
            use_splash=config.use_splash,
            use_vmap_bwd=config.use_vmap_bwd,
        )
        for _ in range(config.num_layers)
    ])
    self.fc = nn.Linear(config.hidden_dim, config.vocab_size)
    logger.info("Conformer model built successfully.")

  def forward(self, inputs: torch.Tensor, **kwargs):
    x = self.embedding(inputs)
    for layer in self.conformer_layers:
      x = layer(x)
    logits = self.fc(x)
    return logits

  def init_weights(self, buffer_device: torch.device | None = None) -> None:
    logger.info("Initializing Conformer weights mimicking PyTorch defaults ...")
    with torch.no_grad():
      for module in self.modules():
        if isinstance(module, nn.Linear):
          fan_in = module.in_features
          bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
          for param in module.parameters():
            local_tensor = (
                param.to_local() if hasattr(param, "to_local") else param
            )
            cpu_tensor = torch.empty_like(local_tensor, device="cpu")
            cpu_tensor.uniform_(-bound, bound)
            local_tensor.copy_(cpu_tensor)
        elif isinstance(module, nn.Conv1d):
          fan_in = module.in_channels * module.kernel_size[0]
          bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
          for param in module.parameters():
            local_tensor = (
                param.to_local() if hasattr(param, "to_local") else param
            )
            cpu_tensor = torch.empty_like(local_tensor, device="cpu")
            cpu_tensor.uniform_(-bound, bound)
            local_tensor.copy_(cpu_tensor)
        elif isinstance(module, nn.Embedding):
          local_tensor = (
              module.weight.to_local()
              if hasattr(module.weight, "to_local")
              else module.weight
          )
          cpu_tensor = torch.empty_like(local_tensor, device="cpu")
          cpu_tensor.normal_(0.0, 1.0)
          local_tensor.copy_(cpu_tensor)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
          if module.weight is not None:
            local_tensor = (
                module.weight.to_local()
                if hasattr(module.weight, "to_local")
                else module.weight
            )
            local_tensor.fill_(1.0)
          if module.bias is not None:
            local_tensor = (
                module.bias.to_local()
                if hasattr(module.bias, "to_local")
                else module.bias
            )
            local_tensor.fill_(0.0)
      for buffer in self.buffers():
        buffer.fill_(0)
    logger.info("Conformer weights initialized.")
