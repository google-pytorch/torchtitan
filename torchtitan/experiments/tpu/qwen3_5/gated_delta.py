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
   ``tokamax_gated_delta.TokamaxGatedDeltaKernel`` swaps the recurrence for
   Tokamax's Mosaic TPU kernels.
3. ``TpuGatedDeltaNet``: ``GatedDeltaNet`` that passes ``positions`` down, so
   the causal ``conv1d`` and the recurrence also restart at every document
   packed into a row (``positions == 0``), as upstream's ``cu_seqlens`` kernels
   do. The document boundaries are static-shape ``[B, S]`` segment-id masks.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.experiments.tpu import unflatten_shim

# Before `torchtitan.models.qwen3_5`, which imports the CUDA-only `attn_gym`.
from torchtitan.experiments.tpu.qwen3_5 import attn_gym_stub  # pylint: disable=unused-import
from torchtitan.models.common.attention import VarlenMetadata, segment_ids_from_positions
from torchtitan.models.qwen3_5.gdn import GatedDeltaKernel, GatedDeltaNet, InnerGatedDeltaNet


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _same_segment(segment_ids_BS: torch.Tensor, shift: int) -> torch.Tensor:
  """``[B, S]``: whether token ``s - shift`` is in the document of token ``s``.

  ``shift`` may be negative (a later token). Where ``s - shift`` is outside the
  row, the result is arbitrary; callers read zero padding there.
  """
  seq_len = segment_ids_BS.shape[1]
  if shift >= 0:
    other_BS = F.pad(segment_ids_BS, (shift, 0), value=-1)[:, :seq_len]
  else:
    other_BS = F.pad(segment_ids_BS, (0, -shift), value=-1)[:, -shift:]
  return other_BS == segment_ids_BS


def _causal_conv(
    x_BSC: torch.Tensor,
    weight_C1W: torch.Tensor,
    segment_ids_BS: torch.Tensor | None = None,
) -> torch.Tensor:
  """Depthwise causal ``conv1d`` on ``[B, S, C]`` via shifted slices.

  ``y[s] = sum_k w[k] * x[s + k - (W - 1)]``, with zeros before the sequence.
  Slices of the front-padded input avoid transposing ``[B, S, C]`` to
  ``[B, C, S]`` and torch_tpu's grouped 1D ``convolution_backward`` shape bug
  under GraphTrainer. As in ``F.conv1d``, the taps add in fp32 and the sum is
  rounded once to the input dtype.

  Args:
    x_BSC: ``[B, S, C]`` input.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight.
    segment_ids_BS: Optional ``[B, S]`` document index of each token (see
      ``segment_ids_from_positions``). Taps that reach into an earlier document
      read zeros, as they do before the start of the row.

  Returns:
    ``[B, S, C]`` convolution output in ``x_BSC.dtype``.
  """
  seq_len = x_BSC.shape[1]
  kernel_size = weight_C1W.shape[-1]
  x_pad = F.pad(x_BSC, (0, 0, kernel_size - 1, 0))
  # At-least-fp32 weights promote bf16/fp16 products (and their sum) to fp32
  # while preserving fp64 and saving the input slices in the input dtype.
  acc_dtype = torch.promote_types(weight_C1W.dtype, torch.float32)
  weight_CW = weight_C1W[:, 0, :].to(acc_dtype)
  out_BSC = None
  for k in range(kernel_size):
    x_shifted = x_pad[:, k : k + seq_len, :]
    shift = kernel_size - 1 - k
    if segment_ids_BS is not None and shift > 0:
      same_BS1 = _same_segment(segment_ids_BS, shift).unsqueeze(-1)
      x_shifted = torch.where(same_BS1, x_shifted, 0.0)
    term = x_shifted * weight_CW[:, k]
    out_BSC = term if out_BSC is None else out_BSC + term
  return out_BSC.to(x_BSC.dtype)


class _CausalConvSilu(torch.autograd.Function):
  """``silu(_causal_conv(x, w))`` with a backward that reads shifted slices.

  Autograd of the slice-based forward writes each tap's input gradient into a
  zero buffer of the padded input (``slice_backward``) and sums the buffers.
  The row offsets are not tile-aligned, so XLA emits slow
  ``dynamic-update-slice`` adds (~0.19 TB/s on TPU7x). This backward reads
  instead: ``dx[t] = sum_k w[k] * dy[t + W - 1 - k]`` is ``W`` shifted slices
  of the end-padded output gradient, which XLA fuses into one pass.

  The convolution is recomputed for ``silu``'s derivative, gradients stay
  within ``segment_ids_BS`` documents, and the ``dx`` taps and ``dw`` token
  sums accumulate in at least fp32 and are rounded once.
  """

  @staticmethod
  def forward(
      ctx,
      x_BSC: torch.Tensor,
      weight_C1W: torch.Tensor,
      segment_ids_BS: torch.Tensor | None,
  ):
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(x_BSC, weight_C1W, segment_ids_BS)
    return F.silu(_causal_conv(x_BSC, weight_C1W, segment_ids_BS))

  @staticmethod
  def backward(ctx, grad_BSC: torch.Tensor | None):
    need_dx, need_dw, _ = ctx.needs_input_grad
    if grad_BSC is None or not (need_dx or need_dw):
      return None, None, None
    x_BSC, weight_C1W, segment_ids_BS = ctx.saved_tensors
    seq_len = x_BSC.shape[1]
    kernel_size = weight_C1W.shape[-1]
    grad_y_BSC = torch.ops.aten.silu_backward(
        grad_BSC, _causal_conv(x_BSC, weight_C1W, segment_ids_BS)
    )
    grad_pad = F.pad(grad_y_BSC, (0, 0, 0, kernel_size - 1))
    # dw sums over all B * S tokens; torch_tpu disallows XLA excess precision,
    # so bf16 sums would round after every add. Accumulate in at least fp32.
    acc_dtype = torch.promote_types(
        torch.promote_types(x_BSC.dtype, weight_C1W.dtype), torch.float32
    )
    weight_CW = weight_C1W[:, 0, :].to(acc_dtype)
    x_acc_BSC = x_BSC.to(acc_dtype) if need_dw else None
    grad_x_BSC = None
    grad_w = []
    for k in range(kernel_size):
      shift = kernel_size - 1 - k
      if shift == 0:
        grad_shifted = grad_y_BSC
      else:
        # grad_shifted[t] = dy[t + shift], the output that w[k] links to x[t],
        # zeroed where that output is in a later document.
        grad_shifted = grad_pad[:, shift : shift + seq_len, :].contiguous()
        if segment_ids_BS is not None:
          same_BS1 = _same_segment(segment_ids_BS, -shift).unsqueeze(-1)
          grad_shifted = torch.where(same_BS1, grad_shifted, 0.0)
      if need_dx:
        term = grad_shifted * weight_CW[:, k]
        grad_x_BSC = term if grad_x_BSC is None else grad_x_BSC + term
      if not need_dw:
        continue
      if shift >= seq_len:
        grad_w.append(x_acc_BSC.new_zeros(weight_C1W.shape[0]))
      elif shift == 0 or segment_ids_BS is not None:
        grad_w.append((x_acc_BSC * grad_shifted).sum(dim=(0, 1)))
      else:
        # Without documents, skip the zero padding: t < seq_len - shift.
        grad_w.append(
            (x_acc_BSC[:, : seq_len - shift, :] * grad_y_BSC[:, shift:, :]).sum(
                dim=(0, 1)
            )
        )
    grad_x = grad_x_BSC.to(x_BSC.dtype) if need_dx else None
    grad_w_C1W = None
    if need_dw:
      grad_w_C1W = torch.stack(grad_w, dim=-1).unsqueeze(1).to(weight_C1W.dtype)
    return grad_x, grad_w_C1W, None


def causal_conv_silu(
    x_BSC: torch.Tensor,
    weight_C1W: torch.Tensor,
    segment_ids_BS: torch.Tensor | None = None,
) -> torch.Tensor:
  """``silu`` of a depthwise causal ``conv1d`` over ``[B, S, C]``.

  The backward is ``_CausalConvSilu``'s: shifted slices of the output gradient
  instead of autograd's scattered adds.

  Args:
    x_BSC: ``[B, S, C]`` input.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight.
    segment_ids_BS: Optional ``[B, S]`` document index of each token; the
      convolution restarts at every document, as at the start of the row.

  Returns:
    ``[B, S, C]`` output.
  """
  return _CausalConvSilu.apply(x_BSC, weight_C1W, segment_ids_BS)


def chunked_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    segment_ids: torch.Tensor | None = None,
) -> torch.Tensor:
  """Chunked gated delta rule in PyTorch (FLA ``chunk_gdn`` math).

  Work within each chunk is expressed as batched matrix multiplications over
  all chunks at once, with the UT transform as one batched triangular solve;
  only the inter-chunk state recurrence is sequential over
  ``seq_len / chunk_size`` steps, and each step is three matmuls. Applies Q/K
  L2 normalization and ``1 / sqrt(key_head_dim)`` query scaling in fp32.

  With ``segment_ids``, the state restarts from zero at the first token of
  every document, as in upstream's ``cu_seqlens`` kernels, through masks of
  static shape: within a chunk, a token only sees the earlier tokens of its own
  document; the state entering a chunk only reaches the document that continues
  from the previous chunk; and only the chunk's last document leaves state.

  Args:
    q: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized queries.
    k: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized keys.
    v: ``(bs, seq_len, n_heads, value_head_dim)`` values.
    g: ``(bs, seq_len, n_heads)`` log-space decay (negative).
    beta: ``(bs, seq_len, n_heads)`` update gate in ``(0, 1)``.
    chunk_size: Tokens per chunk, a power of 2.
    segment_ids: Optional ``(bs, seq_len)`` document index of each token,
      non-decreasing along the sequence (see ``segment_ids_from_positions``).

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
    if segment_ids is not None:
      # ... and continue the last document.
      segment_ids = torch.cat(
          [segment_ids, segment_ids[:, -1:].expand(-1, pad)], dim=1
      )
  num_chunks = (seq_len + pad) // chunk

  q, k, v = [
      x.reshape(bs, n_heads, num_chunks, chunk, x.shape[-1]) for x in (q, k, v)
  ]
  beta = beta.reshape(bs, n_heads, num_chunks, chunk, 1)
  g = g.reshape(bs, n_heads, num_chunks, chunk).cumsum(dim=-1)
  g_last = g[..., -1:]

  # decay[i, j] = exp(g_i - g_j) for i >= j and 0 otherwise. Masking before
  # exp keeps the positive upper-triangle exponents from overflowing.
  decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
  # Decay of the incoming state up to each token, of each token's update up
  # to the end of the chunk, and of the incoming state over the whole chunk.
  state_decay = g.exp()
  update_decay = (g_last - g).exp()
  chunk_decay = g_last.exp()
  if segment_ids is not None:
    seg = segment_ids.reshape(bs, 1, num_chunks, chunk)
    decay = decay * (seg.unsqueeze(-1) == seg.unsqueeze(-2))
    # Document of the previous chunk's last token (the first chunk's incoming
    # state is zero anyway).
    seg_before = torch.cat([seg[:, :, :1, :1], seg[:, :, :-1, -1:]], dim=2)
    continues = seg == seg_before
    state_decay = state_decay * continues
    chunk_decay = chunk_decay * continues[..., -1:]
    update_decay = update_decay * (seg == seg[..., -1:])

  # UT transform for all chunks at once:
  #   [W | U] = T [diag(beta) diag(exp(g)) K | diag(beta) V],
  #   T = (I + tril_{-1}(diag(beta) (K K^T * decay)))^-1.
  # `unitriangular=True` takes the unit diagonal of `I + a` as given.
  k_beta = k * beta
  a = (k_beta @ k.transpose(-1, -2) * decay).tril(-1)
  rhs = torch.cat([k_beta * state_decay.unsqueeze(-1), v * beta], dim=-1)
  wu = torch.linalg.solve_triangular(a, rhs, upper=False, unitriangular=True)
  w, u = wu[..., :d_k], wu[..., d_k:]

  # State-independent per-chunk terms, batched over all chunks.
  qk = q @ k.transpose(-1, -2) * decay  # Causal, including the diagonal.
  # Rows [:chunk] give W S (the state's prediction of each value) and rows
  # [chunk:] give exp(g) Q S (the inter-chunk output), in one matmul.
  wq = torch.cat([w, q * state_decay.unsqueeze(-1)], dim=-2)
  k_decayed_t = (k * update_decay.unsqueeze(-1)).transpose(-1, -2)
  chunk_decay = chunk_decay.unsqueeze(-1)

  state = torch.zeros(bs, n_heads, d_k, d_v, dtype=q.dtype, device=q.device)
  outs = []
  for i in range(num_chunks):
    wq_state = wq[:, :, i] @ state
    v_new = u[:, :, i] - wq_state[..., :chunk, :]
    outs.append(wq_state[..., chunk:, :] + qk[:, :, i] @ v_new)
    state = state * chunk_decay[:, :, i] + k_decayed_t[:, :, i] @ v_new

  out = torch.stack(outs, dim=2).reshape(bs, n_heads, -1, d_v)[:, :, :seq_len]
  return out.transpose(1, 2).contiguous().to(out_dtype)


def per_row_segment_ids(
    segment_ids_BS: torch.Tensor, max_num_segments: int
) -> torch.Tensor:
  """Per-row document ids, keeping at most ``max_num_segments`` apart.

  Args:
    segment_ids_BS: ``[B, S]`` document index of each token, non-decreasing
      along each row in steps of one (``segment_ids_from_positions``, possibly
      counting on from earlier rows).
    max_num_segments: Documents per row kept apart.

  Returns:
    ``[B, S]`` int32 ids that start at 1 in every row. Documents beyond
    ``max_num_segments`` share the last id, so they run as one sequence, as
    without document ids.
  """
  ids = segment_ids_BS - segment_ids_BS[:, :1] + 1
  return ids.clamp(max=max_num_segments).to(torch.int32)


class TpuGatedDeltaKernel(GatedDeltaKernel):
  """``GatedDeltaKernel`` running ``chunked_gated_delta`` on ``[B, S]`` rows."""

  @dataclass(kw_only=True, slots=True)
  class Config(GatedDeltaKernel.Config):
    chunk_size: int = 64

  def __init__(self, config: Config):
    super().__init__(config)
    self.chunk_size = config.chunk_size

  def document_ids(self, segment_ids_BS: torch.Tensor) -> torch.Tensor:
    """The document ids that the causal convolution and the recurrence use.

    ``TpuInnerGatedDeltaNet`` passes the same ids to both, so a kernel that
    keeps only some documents apart (``TokamaxGatedDeltaKernel``) merges the
    others in the convolution too. This one keeps every document apart.

    Args:
      segment_ids_BS: ``[B, S]`` document index of each token.

    Returns:
      ``[B, S]`` document ids.
    """
    return segment_ids_BS

  def recurrence(
      self,
      q_BSHK: torch.Tensor,
      k_BSHK: torch.Tensor,
      v_BSHV: torch.Tensor,
      g_BSH: torch.Tensor,
      beta_BSH: torch.Tensor,
      segment_ids_BS: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """The gated delta rule on ``[B, S]`` rows; subclasses swap the kernel.

    Args:
      q_BSHK: Queries, one key head per value head.
      k_BSHK: Keys, one key head per value head.
      v_BSHV: Values.
      g_BSH: Log-space decay.
      beta_BSH: Update gate.
      segment_ids_BS: Optional ``[B, S]`` document index of each token; the
        recurrence restarts at every document.

    Returns:
      ``[B, S, H, V]`` output.
    """
    return chunked_gated_delta(
        q_BSHK,
        k_BSHK,
        v_BSHV,
        g_BSH,
        beta_BSH,
        chunk_size=self.chunk_size,
        segment_ids=segment_ids_BS,
    )

  def forward(
      self,
      xq_BSHK: torch.Tensor,
      xk_BSHK: torch.Tensor,
      xv_BSHV: torch.Tensor,
      g_BSH: torch.Tensor,
      beta_BSH: torch.Tensor,
      *,
      cu_seqlens: torch.Tensor | None = None,
      segment_ids_BS: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Runs the recurrence, restarting it at every ``segment_ids_BS`` change."""
    if cu_seqlens is not None:
      raise NotImplementedError(
          "TpuGatedDeltaKernel does not support varlen cu_seqlens; pass"
          " segment_ids_BS."
      )
    # `chunked_gated_delta` needs one key head per value head.
    repeats = xv_BSHV.shape[2] // xq_BSHK.shape[2]
    if repeats > 1:
      xq_BSHK = xq_BSHK.repeat_interleave(repeats, dim=2)
      xk_BSHK = xk_BSHK.repeat_interleave(repeats, dim=2)
    return self.recurrence(
        xq_BSHK,
        xk_BSHK,
        xv_BSHV,
        g_BSH,
        beta_BSH,
        segment_ids_BS=segment_ids_BS,
    )


class TpuInnerGatedDeltaNet(InnerGatedDeltaNet):
  """``InnerGatedDeltaNet`` that convolves each ``[B, S]`` row on its own.

  Given ``positions``, the convolution and the recurrence also restart at
  every document packed into a row (``positions == 0``), as upstream's varlen
  kernels do at ``cu_seqlens``.
  """

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
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    del cu_seqlens  # Rows come from set_unflatten_seq_len instead.
    num_tokens = query_TC.shape[0]
    g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
    beta_TH = torch.sigmoid(b_TH)
    # TT-UNFLATTEN-SHIM: unflatten [T, ...] to [B, S, ...] once for conv + recurrence.
    query_BSC, key_BSC, value_BSC, g_BSH, beta_BSH = (
        unflatten_shim.unflatten_tensor(x, expected_flat_ndim=2)
        for x in (query_TC, key_TC, value_TC, g_TH, beta_TH)
    )
    if query_BSC.ndim == 2:
      # Without set_unflatten_seq_len, the [T] token stream is one row.
      query_BSC, key_BSC, value_BSC, g_BSH, beta_BSH = (
          x.unsqueeze(0)
          for x in (query_BSC, key_BSC, value_BSC, g_BSH, beta_BSH)
      )
    segment_ids_BS = None
    if positions is not None:
      if positions.numel() != num_tokens:
        raise ValueError(
            "TpuInnerGatedDeltaNet needs one position per token, got"
            f" positions of shape {tuple(positions.shape)} for"
            f" {num_tokens} tokens."
        )
      # Document index of each token: documents start at positions == 0.
      segment_ids_BS = self.kernel.document_ids(
          segment_ids_from_positions(positions.reshape(-1)).reshape(
              query_BSC.shape[:2]
          )
      )

    def causal_conv(
        x_BSC: torch.Tensor,
        weight_C1W: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
      return causal_conv_silu(x_BSC, weight_C1W, segment_ids_BS).unflatten(
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
        segment_ids_BS=segment_ids_BS,
    )
    # TT-UNFLATTEN-SHIM: fold [B, S, ...] back to [T, ...].
    return out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:])


class TpuGatedDeltaNet(GatedDeltaNet):
  """``GatedDeltaNet`` that passes ``positions`` to its inner module.

  Identical to upstream ``GatedDeltaNet.forward`` except that it also takes
  ``positions`` (``Qwen35Model.forward`` hands them to every layer) and passes
  them to ``TpuInnerGatedDeltaNet``, which restarts the convolution and the
  recurrence at every packed document. Upstream instead passes ``cu_seqlens``
  document offsets, which ``Qwen35Model`` only builds for the flex and varlen
  attention backends and whose shape changes with the number of documents.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(GatedDeltaNet.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: VarlenMetadata | None = None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_tokens = x_TD.shape[0]
    if attention_masks is not None:
      cu_seqlens = attention_masks.cu_seq_q
    else:
      cu_seqlens = torch.arange(
          0,
          num_tokens + 1,
          num_tokens,
          dtype=torch.int32,
          device=x_TD.device,
      )

    query_TC = self.in_proj_q(x_TD)
    key_TC = self.in_proj_k(x_TD)
    value_TC = self.in_proj_v(x_TD)
    gate_TC = self.in_proj_z(x_TD)
    a_TH = self.in_proj_a(x_TD)
    b_TH = self.in_proj_b(x_TD)

    output_THV = self.inner_gated_delta_net(
        query_TC,
        key_TC,
        value_TC,
        a_TH,
        b_TH,
        self.conv_q.weight,
        self.conv_k.weight,
        self.conv_v.weight,
        self.A_log,
        self.dt_bias,
        cu_seqlens,
        key_head_dim=self.key_head_dim,
        value_head_dim=self.value_head_dim,
        # TPU: pass `positions`; packed documents restart conv + recurrence.
        positions=positions,
    )
    gate_THV = gate_TC.view(num_tokens, -1, self.value_head_dim)
    output_THV = self.norm(output_THV, gate_THV)
    out_TD = output_THV.reshape(num_tokens, -1)
    return self.out_proj(out_TD)
