"""TPU GatedDeltaNet for Qwen3.5: chunked gated delta rule and causal conv1d.

Upstream ``torchtitan.models.qwen3_5.gdn`` runs the gated delta rule and the
varlen causal ``conv1d`` with CUDA/Triton ``attn_gym`` kernels (see
``attn_gym_stub``), and its fallback ``conv1d`` treats the folded ``[T]`` token
axis as one stream. This module provides the TPU replacements:

1. ``chunked_gated_delta``: the chunked gated delta rule (``chunk_gdn`` math)
   in PyTorch, as batched matmuls that XLA compiles.
2. ``TpuGatedDeltaKernel`` and ``TpuInnerGatedDeltaNet``: drop-in modules that
   unflatten ``[T]`` to ``[B, S]`` once via ``unflatten_shim``, so the causal
   ``conv1d`` and the recurrence run on each ``[B, S]`` row independently
   without dynamic-shape ``cu_seqlens`` offsets (upstream ``GatedDeltaNet``
   only passes ``cu_seqlens``, not ``positions`` / ``segment_ids``).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu import unflatten_shim

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel, InnerGatedDeltaNet


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _causal_conv(x_BSC: torch.Tensor, weight_C1W: torch.Tensor) -> torch.Tensor:
  """Depthwise causal ``conv1d`` on ``[B, S, C]`` via shifted slices.

  ``y[s] = sum_k w[k] * x[s + k - (W - 1)]``, with zeros before the sequence.
  Slices of the front-padded input avoid transposing ``[B, S, C]`` to
  ``[B, C, S]`` and torch_tpu's grouped 1D ``convolution_backward`` shape bug
  under GraphTrainer.

  Args:
    x_BSC: ``[B, S, C]`` input.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight.

  Returns:
    ``[B, S, C]`` convolution output.
  """
  seq_len = x_BSC.shape[1]
  kernel_size = weight_C1W.shape[-1]
  x_pad = F.pad(x_BSC, (0, 0, kernel_size - 1, 0))
  out_BSC = x_pad[:, :seq_len, :] * weight_C1W[:, 0, 0]
  for k in range(1, kernel_size):
    out_BSC = out_BSC + x_pad[:, k : k + seq_len, :] * weight_C1W[:, 0, k]
  return out_BSC


def chunked_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
  """Chunked gated delta rule in PyTorch (FLA ``chunk_gdn`` math).

  Work within each chunk is expressed as batched matrix multiplications over
  all chunks at once, with the UT transform as one batched triangular solve;
  only the inter-chunk state recurrence is sequential over
  ``seq_len / chunk_size`` steps, and each step is three matmuls. Applies Q/K
  L2 normalization and ``1 / sqrt(key_head_dim)`` query scaling in fp32.

  Args:
    q: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized queries.
    k: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized keys.
    v: ``(bs, seq_len, n_heads, value_head_dim)`` values.
    g: ``(bs, seq_len, n_heads)`` log-space decay (negative).
    beta: ``(bs, seq_len, n_heads)`` update gate in ``(0, 1)``.
    chunk_size: Tokens per chunk, a power of 2.

  Returns:
    ``(bs, seq_len, n_heads, value_head_dim)`` output in ``q.dtype``.
  """
  out_dtype = q.dtype
  d_k = q.shape[-1]
  q, k, v = [x.transpose(1, 2).float() for x in (q, k, v)]
  g, beta = [x.transpose(1, 2).float() for x in (g, beta)]
  q = _l2norm(q) * (d_k**-0.5)
  k = _l2norm(k)

  bs, n_heads, seq_len, _ = k.shape
  d_v = v.shape[-1]
  chunk = chunk_size
  if 0 < seq_len < chunk_size:
    # Short sequences use one chunk, rounded up to a power of 2.
    chunk = 1 << (seq_len - 1).bit_length()
  if chunk & (chunk - 1):
    raise ValueError(f"chunk_size must be a power of 2: {chunk}")
  pad = (chunk - seq_len % chunk) % chunk
  if pad:
    # Padded tokens have beta = 0 (no state update) and g = 0 (no decay).
    q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    g, beta = [F.pad(x, (0, pad)) for x in (g, beta)]
  num_chunks = (seq_len + pad) // chunk

  q, k, v = [
      x.reshape(bs, n_heads, num_chunks, chunk, x.shape[-1]) for x in (q, k, v)
  ]
  beta = beta.reshape(bs, n_heads, num_chunks, chunk, 1)
  g = g.reshape(bs, n_heads, num_chunks, chunk).cumsum(dim=-1)

  # decay[i, j] = exp(g_i - g_j) for i >= j and 0 otherwise. Masking before
  # exp keeps the positive upper-triangle exponents from overflowing.
  decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

  # UT transform for all chunks at once:
  #   [W | U] = T [diag(beta) diag(exp(g)) K | diag(beta) V],
  #   T = (I + tril_{-1}(diag(beta) (K K^T * decay)))^-1.
  # `unitriangular=True` takes the unit diagonal of `I + a` as given.
  k_beta = k * beta
  a = (k_beta @ k.transpose(-1, -2) * decay).tril(-1)
  rhs = torch.cat([k_beta * g.exp().unsqueeze(-1), v * beta], dim=-1)
  wu = torch.linalg.solve_triangular(a, rhs, upper=False, unitriangular=True)
  w, u = wu[..., :d_k], wu[..., d_k:]

  # State-independent per-chunk terms, batched over all chunks.
  qk = q @ k.transpose(-1, -2) * decay  # Causal, including the diagonal.
  g_last = g[..., -1:]
  # Rows [:chunk] give W S (the state's prediction of each value) and rows
  # [chunk:] give exp(g) Q S (the inter-chunk output), in one matmul.
  wq = torch.cat([w, q * g.exp().unsqueeze(-1)], dim=-2)
  k_decayed_t = (k * (g_last - g).exp().unsqueeze(-1)).transpose(-1, -2)
  chunk_decay = g_last.exp().unsqueeze(-1)

  state = torch.zeros(bs, n_heads, d_k, d_v, dtype=q.dtype, device=q.device)
  outs = []
  for i in range(num_chunks):
    wq_state = wq[:, :, i] @ state
    v_new = u[:, :, i] - wq_state[..., :chunk, :]
    outs.append(wq_state[..., chunk:, :] + qk[:, :, i] @ v_new)
    state = state * chunk_decay[:, :, i] + k_decayed_t[:, :, i] @ v_new

  out = torch.stack(outs, dim=2).reshape(bs, n_heads, -1, d_v)[:, :, :seq_len]
  return out.transpose(1, 2).contiguous().to(out_dtype)


class TpuGatedDeltaKernel(GatedDeltaKernel):
  """``GatedDeltaKernel`` running ``chunked_gated_delta`` on ``[B, S]`` rows."""

  @dataclass(kw_only=True, slots=True)
  class Config(GatedDeltaKernel.Config):
    chunk_size: int = 64

  def __init__(self, config: Config):
    super().__init__(config)
    self.chunk_size = config.chunk_size

  def forward(
      self,
      xq_BSHK: torch.Tensor,
      xk_BSHK: torch.Tensor,
      xv_BSHV: torch.Tensor,
      g_BSH: torch.Tensor,
      beta_BSH: torch.Tensor,
      *,
      cu_seqlens: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if cu_seqlens is not None:
      raise NotImplementedError(
          "TpuGatedDeltaKernel does not support varlen cu_seqlens."
      )
    # `chunked_gated_delta` needs one key head per value head.
    repeats = xv_BSHV.shape[2] // xq_BSHK.shape[2]
    if repeats > 1:
      xq_BSHK = xq_BSHK.repeat_interleave(repeats, dim=2)
      xk_BSHK = xk_BSHK.repeat_interleave(repeats, dim=2)
    return chunked_gated_delta(
        xq_BSHK, xk_BSHK, xv_BSHV, g_BSH, beta_BSH, chunk_size=self.chunk_size
    )


class TpuInnerGatedDeltaNet(InnerGatedDeltaNet):
  """``InnerGatedDeltaNet`` that convolves each ``[B, S]`` row on its own."""

  @dataclass(kw_only=True, slots=True)
  class Config(InnerGatedDeltaNet.Config):
    pass

  def forward(
      self,
      query_TC: torch.Tensor,
      key_TC: torch.Tensor,
      value_TC: torch.Tensor,
      a_TH: torch.Tensor,
      b_TH: torch.Tensor,
      conv_q_weight_C1W: torch.Tensor,
      conv_k_weight_C1W: torch.Tensor,
      conv_v_weight_C1W: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_H: torch.Tensor,
      cu_seqlens: torch.Tensor,
      *,
      key_head_dim: int,
      value_head_dim: int,
  ) -> torch.Tensor:
    del cu_seqlens  # Rows come from set_unflatten_seq_len instead.
    g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
    beta_TH = torch.sigmoid(b_TH)
    # TT-UNFLATTEN-SHIM: unflatten [T, ...] to [B, S, ...] once for conv + recurrence.
    query_BSC, key_BSC, value_BSC, g_BSH, beta_BSH = (
        unflatten_shim.unflatten_tensor(x, expected_flat_ndim=2)
        for x in (query_TC, key_TC, value_TC, g_TH, beta_TH)
    )

    def causal_conv(
        x_BSC: torch.Tensor,
        weight_C1W: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
      return F.silu(_causal_conv(x_BSC, weight_C1W)).unflatten(
          -1, (-1, head_dim)
      )

    xq_BSHK = causal_conv(query_BSC, conv_q_weight_C1W, key_head_dim)
    xk_BSHK = causal_conv(key_BSC, conv_k_weight_C1W, key_head_dim)
    xv_BSHV = causal_conv(value_BSC, conv_v_weight_C1W, value_head_dim)
    out_BSHV = self.kernel(
        xq_BSHK,
        xk_BSHK,
        xv_BSHV,
        g_BSH,
        beta_BSH,
        cu_seqlens=None,
    )
    return unflatten_shim.flatten_tensor(out_BSHV, original_flat_ndim=3)
