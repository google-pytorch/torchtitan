"""Context parallelism (Ulysses and Load-Balanced KV All-Gather) for the TPU lane.

As on GPU, a recipe turns CP on explicitly: it sets
``parallelism.context_parallel_degree``, ``spmd_backend="spmd_types"`` and the
load balancer, then applies core ``ContextParallelTransform`` with one of the
TPU CP inner attentions below (see ``qwen3/config_registry.py``). Two
strategies are provided:

1. ``all_gather`` (MaxText-style load-balanced KV All-Gather): Inputs
   are reordered into head-tail load-balanced order
   (`[0, 2*cp-1, 1, 2*cp-2, ...]`) and sharded along the sequence dimension.
   Each attention layer keeps ``Q`` sharded (`S / cp` tokens, all ``H``
   heads), all-gathers ``K`` and ``V`` to the full sequence ``S`` (with fp32
   reduce-scatter in backward), reorders ``K`` and ``V`` back to contiguous
   order, and runs SplashAttention with its causal / local mask in head-tail
   ``Q`` order (`q_seq_shards=cp`). A kernel must be plugged in on TPU (see
   ``KVAllGatherCPInnerAttention.forward``).
2. ``ulysses``: Each attention layer all-to-alls q/k/v from token-sharded to
   head-sharded, runs full-sequence attention on ``H / cp`` heads, then
   all-to-alls the output back. When ``n_kv_heads < cp`` and
   ``cp % n_kv_heads == 0``, KV heads are repeated before the all-to-all.

Tensor suffixes: ``T`` tokens, ``H`` heads, ``K`` qk head dim, ``V`` v head dim.
"""

from collections.abc import Callable
from dataclasses import dataclass, fields
import math
import sys
from typing import Any, Literal

# pyformat: disable
# Import torch before spmd_types to avoid circular import with torch_tpu.
import torch
import spmd_types as spmd
# pyformat: enable
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import fsdp as fsdp_mod
from torchtitan.distributed.context_parallel.api import prepare_context_parallel_input
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.tpu.headtail import headtail_chunk_order
from torchtitan.experiments.tpu.unflatten_shim import get_unflatten_seq_len
from torchtitan.models.common.attention import (
    BaseAttention,
    ScaledDotProductInnerAttention,
    VarlenInnerAttention,
    segment_ids_from_positions,
)
from torchtitan.models.common.cp_attention import CPInnerAttention

__all__ = [
    "UlyssesCPInnerAttention",
    "UlyssesCPScaledDotProductInnerAttention",
    "UlyssesCPVarlenInnerAttention",
    "KVAllGatherCPInnerAttention",
    "KVAllGatherCPScaledDotProductInnerAttention",
    "KVAllGatherCPVarlenInnerAttention",
    "reorder_sequence",
    "tpu_cp_inner_attention",
    "tpu_cp_load_balancer",
    "validate_tpu_context_parallel",
    "revert_tpu_context_parallel",
    "install_tpu_cp_patches",
]

_TOKEN_DIM = 0
_HEAD_DIM = 1

# Per-(device, rank) cache for the local CP rank tensor (shape (1,), int32),
# populated during cp_shard / eager setup so torch.compile lifts it as a dynamic
# graph input instead of baking a constant rank into the traced graph.
_CP_RANK_CACHE: dict[tuple[torch.device, int], torch.Tensor] = {}


def _cache_cp_rank_for_mesh(cp_mesh: DeviceMesh) -> None:
  dev_type = cp_mesh.device_type
  dev = torch.device(f"{dev_type}:0" if dev_type != "cpu" else "cpu")
  r = int(cp_mesh.get_local_rank())
  key = (dev, r)
  if key not in _CP_RANK_CACHE:
    _CP_RANK_CACHE[key] = torch.tensor([r], dtype=torch.int32, device=dev)


def _type_checking() -> bool:
  return get_spmd_backend() == "spmd_types" and spmd.is_type_checking()


def _tracing() -> bool:
  # spmd.redistribute is an autograd.Function around dist.* collectives,
  # which Dynamo skips; use traceable functional collectives instead.
  return (
      torch.compiler.is_compiling() or torch.compiler._is_non_strict_tracing()
  )


def _wait(x: torch.Tensor) -> torch.Tensor:
  """Waits on a functional collective's result exactly once.

  While tracing, funcol already applies ``wait_tensor`` to its result, and
  TorchTPU's split compiler requires every ``wait_tensor`` to consume a
  collective, so a second wait fails to compile. In eager mode funcol returns
  an ``AsyncCollectiveTensor`` that still needs its one wait.
  """
  if isinstance(x, funcol.AsyncCollectiveTensor):
    return x.wait()
  return x


def _all_to_all(
    x: torch.Tensor, group: Any, *, src_dim: int, dst_dim: int
) -> torch.Tensor:
  """Reshards ``x`` from ``S(src_dim)`` to ``S(dst_dim)`` over ``group``."""
  if not _tracing():
    return spmd.redistribute(x, group, src=spmd.S(src_dim), dst=spmd.S(dst_dim))
  n = group.size()
  shape = list(x.shape)
  if shape[dst_dim] % n:
    raise ValueError(
        f"dim {dst_dim} (size {shape[dst_dim]}) is not divisible by cp={n}."
    )
  # [.., D, ..] -> [n, .., D / n, ..]: chunk i goes to rank i.
  chunked = shape[:dst_dim] + [n, shape[dst_dim] // n] + shape[dst_dim + 1 :]
  x = x.reshape(chunked).movedim(dst_dim, 0).contiguous()
  x = _wait(funcol.all_to_all_single(x, None, None, group))
  # [n, ..] -> concatenate the received chunks along src_dim.
  out_shape = list(shape)
  out_shape[dst_dim] //= n
  out_shape[src_dim] *= n
  return x.movedim(0, src_dim).reshape(out_shape)


class _AllGatherKV(torch.autograd.Function):
  """Traceable All-Gather in forward and dtype-controlled Reduce-Scatter in backward."""

  @staticmethod
  def forward(
      ctx,
      x: torch.Tensor,
      group: Any,
      seq_dim: int,
      reduce_dtype: torch.dtype,
  ) -> torch.Tensor:
    ctx.group = group
    ctx.seq_dim = seq_dim
    ctx.reduce_dtype = reduce_dtype
    ctx.orig_dtype = x.dtype
    return _wait(funcol.all_gather_tensor(x, seq_dim, group))

  @staticmethod
  def backward(ctx, grad_out: torch.Tensor):
    g = grad_out.to(ctx.reduce_dtype)
    g = _wait(funcol.reduce_scatter_tensor(g, "sum", ctx.seq_dim, ctx.group))
    return g.to(ctx.orig_dtype), None, None, None


def _all_gather_kv(
    x: torch.Tensor,
    group: Any,
    *,
    seq_dim: int = _TOKEN_DIM,
    reduce_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  """All-gathers ``x`` from ``S(seq_dim)`` to ``R`` with ``reduce_dtype`` backward RS."""
  if not _tracing():
    return spmd.redistribute(
        x,
        group,
        src=spmd.S(seq_dim),
        dst=spmd.R,
        backward_options={"op_dtype": reduce_dtype},
    )
  return _AllGatherKV.apply(x, group, seq_dim, reduce_dtype)


def reorder_sequence(
    tensor: torch.Tensor,
    cp_size: int,
    seq_dim: int = 1,
    to_contiguous: bool = False,
) -> torch.Tensor:
  """Reorders ``tensor`` along ``seq_dim`` between contiguous and head-tail CP order.

  Matches MaxText's ``reorder_sequence`` (``maxtext/utils/max_utils.py``):
  splits the sequence into ``2 * cp_size`` chunks and pairs chunk ``i`` from
  the front with chunk ``2 * cp_size - 1 - i`` from the back so each rank's
  ``S / cp_size`` shard has identical causal attention work.
  """
  if cp_size <= 1:
    return tensor
  seq_len = tensor.shape[seq_dim]
  if seq_len % (2 * cp_size) != 0:
    raise ValueError(
        f"Sequence length ({seq_len}) must be divisible by"
        f" 2 * cp_size ({2 * cp_size})."
    )
  chunk_size = seq_len // (2 * cp_size)
  orig_shape = list(tensor.shape)
  new_shape = (
      orig_shape[:seq_dim]
      + [2 * cp_size, chunk_size]
      + orig_shape[seq_dim + 1 :]
  )
  reshaped = tensor.reshape(new_shape)
  indices = headtail_chunk_order(cp_size)
  if to_contiguous:
    # Inverse permutation: the head-tail index of each contiguous chunk.
    indices = sorted(range(2 * cp_size), key=indices.__getitem__)
  idx_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
  reordered = torch.index_select(reshaped, seq_dim, idx_tensor)
  return reordered.reshape(orig_shape)


def _repeat_kv_heads(x: torch.Tensor, cp: int) -> torch.Tensor:
  """Repeats KV heads so that each of the ``cp`` head shards gets one.

  Needed when ``n_kv_heads < cp``, e.g. Qwen3.5-397B (2 KV heads) at cp=4.
  The result keeps the token sharding of ``x``.
  """
  n_kv = x.shape[_HEAD_DIM]
  if n_kv >= cp:
    return x
  if cp % n_kv:
    raise ValueError(f"cp ({cp}) must be a multiple of n_kv_heads ({n_kv}).")
  with spmd.no_typecheck():
    out = x.repeat_interleave(cp // n_kv, dim=_HEAD_DIM)
  if _type_checking():
    spmd.assert_type(out, spmd.get_local_type(x), spmd.get_partition_spec(x))
  return out


@spmd.no_typecheck()
def _full_sequence_segment_ids(
    positions: torch.Tensor, group: Any
) -> torch.Tensor:
  """Packed-document ``segment_ids`` of the whole (unsharded) sequence."""
  if _tracing():
    positions = _wait(funcol.all_gather_tensor(positions, _TOKEN_DIM, group))
  else:
    positions = spmd.redistribute(
        positions, group, src=spmd.S(_TOKEN_DIM), dst=spmd.R
    )
  return segment_ids_from_positions(positions)


class UlyssesCPInnerAttention(CPInnerAttention):
  """Ulysses (all-to-all) CP mixin for any single-device inner attention."""

  def __init__(self, config: Any) -> None:
    super().__init__(config)  # pyrefly: ignore[bad-argument-count]
    # A full-sequence attention kernel, plugged in with set_local_attention;
    # None runs the wrapped inner attention (local_forward).
    self.set_local_attention(None)

  @classmethod
  def cp_shard(
      cls,
      input_dict: dict[str, Any],
      input_shardings: dict[str, Any] | None,
      cp_mesh: Any,
      load_balancer_type: str | None,
      ptrr_mask_key: str | None,
  ) -> dict[str, Any]:
    # Attention sees the full sequence, so masks stay unsharded, and there is
    # no per-rank causal imbalance to load-balance.
    attention_masks = input_dict.pop("attention_masks", None)
    prepare_context_parallel_input(
        input_dict, input_shardings, cp_mesh, None, None
    )
    if attention_masks is not None:
      input_dict["attention_masks"] = attention_masks
    return input_dict

  def set_local_attention(
      self, local_attn: Callable[..., torch.Tensor] | None
  ) -> None:
    """Replaces the full-sequence attention, e.g. with TPU splash attention."""
    object.__setattr__(self, "_local_attn_override", local_attn)

  def local_forward(self, q_THK, k_THK, v_THV, **kwargs) -> torch.Tensor:
    """The wrapped single-device inner attention."""
    return super().forward(  # pyrefly: ignore[missing-attribute]
        q_THK, k_THK, v_THV, **kwargs
    )

  def forward(
      self,
      q_THK: torch.Tensor,
      k_THK: torch.Tensor,
      v_THV: torch.Tensor,
      **kwargs,
  ) -> torch.Tensor:
    group = spmd_mesh_group(MeshAxisName.CP)
    if group is None:
      raise RuntimeError("CP attention requires an active CP mesh axis.")
    cp = group.size()
    k_THK, v_THV = _repeat_kv_heads(k_THK, cp), _repeat_kv_heads(v_THV, cp)
    # (T/cp, H, *) -> (T, H/cp, *).
    q_THK, k_THK, v_THV = (
        _all_to_all(x, group, src_dim=_TOKEN_DIM, dst_dim=_HEAD_DIM)
        for x in (q_THK, k_THK, v_THV)
    )
    if "enable_gqa" in kwargs:
      kwargs["enable_gqa"] = q_THK.shape[_HEAD_DIM] != k_THK.shape[_HEAD_DIM]
    # Decoder.forward derived segment_ids from the token-sharded positions;
    # the local attention needs them for the full sequence.
    positions = kwargs.pop("positions", None)
    if positions is not None:
      kwargs["segment_ids"] = _full_sequence_segment_ids(positions, group)

    local_attn = self._local_attn_override or self.local_forward
    type_check = _type_checking()
    if type_check:
      q_type = spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)
    # Kernels below (SDPA shim, Pallas) carry no spmd_types annotations.
    with spmd.no_typecheck():
      out_THV = local_attn(q_THK, k_THK, v_THV, **kwargs)
    if type_check and not spmd.has_local_type(out_THV):
      spmd.assert_type(out_THV, *q_type)
    # (T, H/cp, V) -> (T/cp, H, V).
    return _all_to_all(out_THV, group, src_dim=_HEAD_DIM, dst_dim=_TOKEN_DIM)


class KVAllGatherCPInnerAttention(CPInnerAttention):
  """MaxText-style load-balanced KV All-Gather CP mixin for inner attention."""

  def __init__(self, config: Any) -> None:
    super().__init__(config)  # pyrefly: ignore[bad-argument-count]
    self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]
    self.allow_reference_attention = config.allow_reference_attention
    # The local attention kernel, plugged in with set_local_attention.
    self.set_local_attention(None)

  def init_weights(self, **kwargs) -> None:
    super().init_weights(**kwargs)  # pyrefly: ignore[missing-attribute]
    group = spmd_mesh_group(MeshAxisName.CP)
    buffer_device = kwargs.get("buffer_device")
    if (
        group is not None
        and buffer_device is not None
        and buffer_device.type != "meta"
    ):
      r = int(group.rank())
      _CP_RANK_CACHE[(buffer_device, r)] = torch.tensor(
          [r], dtype=torch.int32, device=buffer_device
      )

  def _get_cp_rank_tensor(
      self, device: torch.device, group: Any
  ) -> torch.Tensor:
    r = int(group.rank())
    key = (device, r)
    cached = _CP_RANK_CACHE.get(key)
    if cached is not None:
      return cached
    t = torch.tensor([r], dtype=torch.int32, device=device)
    if not _tracing():
      _CP_RANK_CACHE[key] = t
    return t

  @classmethod
  def cp_shard(
      cls,
      input_dict: dict[str, Any],
      input_shardings: dict[str, Any] | None,
      cp_mesh: Any,
      load_balancer_type: str | None,
      ptrr_mask_key: str | None,
  ) -> dict[str, Any]:
    # Populate the per-device CP rank tensor in eager mode before tracing.
    _cache_cp_rank_for_mesh(cp_mesh)

    if load_balancer_type != "headtail":
      raise ValueError(
          "KV all-gather CP shards inputs in head-tail order, so"
          " context_parallel_load_balancer must be 'headtail', got"
          f" {load_balancer_type!r}."
      )
    attention_masks = input_dict.pop("attention_masks", None)
    seq_len = get_unflatten_seq_len()
    cp = cp_mesh.size(0)
    inp = input_dict.get("input")

    # When multiple sequences (B > 1) are folded into 1D [B * S], head-tail
    # reordering and CP sharding must happen per sequence of length S, not
    # across the folded B * S axis.
    if (
        seq_len is not None
        and isinstance(inp, torch.Tensor)
        and inp.ndim >= 1
        and inp.shape[0] > seq_len
        and inp.shape[0] % seq_len == 0
    ):
      bsz = inp.shape[0] // seq_len
      for key in ("input", "labels", "positions"):
        val = input_dict.get(key)
        if (
            isinstance(val, torch.Tensor)
            and val.ndim >= 1
            and val.shape[0] == bsz * seq_len
        ):
          rest = list(val.shape[1:])
          v_bs = val.reshape([bsz, seq_len] + rest)
          v_reord = reorder_sequence(
              v_bs, cp_size=cp, seq_dim=1, to_contiguous=False
          )
          # Place the cp shard axis first so dim-0 chunking gives rank r
          # its [bsz, seq_len // cp] slice across all batch elements.
          v_grouped = v_reord.reshape([bsz, cp, seq_len // cp] + rest).movedim(
              1, 0
          )
          input_dict[key] = v_grouped.reshape([bsz * seq_len] + rest)
      prepare_context_parallel_input(
          input_dict, input_shardings, cp_mesh, None, ptrr_mask_key
      )
    else:
      prepare_context_parallel_input(
          input_dict, input_shardings, cp_mesh, "headtail", ptrr_mask_key
      )
    if attention_masks is not None:
      input_dict["attention_masks"] = attention_masks
    return input_dict

  def set_local_attention(
      self, local_attn: Callable[..., torch.Tensor] | None
  ) -> None:
    """Replaces the local attention kernel, e.g. with TPU splash attention."""
    object.__setattr__(self, "_local_attn_override", local_attn)

  def local_forward(
      self,
      q: torch.Tensor,
      k: torch.Tensor,
      v: torch.Tensor,
      *,
      scale: float | None = None,
      enable_gqa: bool = False,
      is_causal: bool = True,
      segment_ids: torch.Tensor | None = None,
      cp_rank: torch.Tensor | None = None,
      **kwargs,
  ) -> torch.Tensor:
    """Reference attention for CPU, tests and short sequences.

    Computes explicit masked attention in fp32 over the full (S / cp, S) score
    matrix (avoiding the torch_tpu compiled SDPA backward NaN issue when
    ``q_seq_len != kv_seq_len``) using tensor indexing with ``cp_rank`` so it is
    traceable without graph breaks. It is the Splash wrapper's fallback for CPU
    and short sequences; ``forward`` refuses it on TPU unless
    ``allow_reference_attention`` is set. Sliding windows are not supported.
    """
    if self._uses_sliding_window():
      raise NotImplementedError(
          "The KV all-gather CP reference attention does not support sliding"
          " windows; plug in the Splash kernel with set_local_attention."
      )
    # Unpatched ScaledDotProductInnerAttention on CPU ignores segment_ids
    # (F.scaled_dot_product_attention only masks via is_causal), so match its
    # behavior when called from GQAttention (which passes attention_masks=None).
    use_segment_ids = segment_ids is not None and not (
        isinstance(self, ScaledDotProductInnerAttention)
        and "attention_masks" in kwargs
    )
    del kwargs
    was_3d = q.ndim == 3
    if was_3d:
      q = q.unsqueeze(0)
      k = k.unsqueeze(0)
      v = v.unsqueeze(0)
    bsz, q_len, n_heads, head_dim = q.shape
    kv_len = k.shape[1]
    n_kv_heads = k.shape[2]
    cp = kv_len // q_len

    if enable_gqa and n_heads != n_kv_heads:
      k = k.repeat_interleave(n_heads // n_kv_heads, dim=2)
      v = v.repeat_interleave(n_heads // n_kv_heads, dim=2)

    q_bh = q.transpose(1, 2).float()
    k_bh = k.transpose(1, 2).float()
    v_bh = v.transpose(1, 2).float()
    sm_scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q_bh, k_bh.transpose(-2, -1)) * sm_scale

    if cp > 1 and cp_rank is None:
      raise ValueError("cp_rank is required when Q is a CP shard (cp > 1).")
    kv_pos = torch.arange(kv_len, device=q.device, dtype=torch.int32)
    if cp > 1:
      r = cp_rank.reshape(())
      q_pos_full = reorder_sequence(kv_pos, cp_size=cp, seq_dim=0)
      q_pos = q_pos_full.reshape(cp, q_len)[r]
    else:
      q_pos = kv_pos

    mask = torch.ones(
        (bsz, 1, q_len, kv_len), dtype=torch.bool, device=q.device
    )
    if is_causal:
      mask = mask & (q_pos[None, None, :, None] >= kv_pos[None, None, None, :])
    if use_segment_ids and segment_ids is not None:
      seg_2d = (
          segment_ids.unsqueeze(0) if segment_ids.ndim == 1 else segment_ids
      )
      if cp > 1:
        q_seg_full = reorder_sequence(seg_2d, cp_size=cp, seq_dim=1)
        q_seg = q_seg_full.reshape(bsz, cp, q_len)[:, r, :]
      else:
        q_seg = seg_2d
      mask = mask & (q_seg[:, None, :, None] == seg_2d[:, None, None, :])

    scores = scores.masked_fill(~mask, -1e30)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v_bh).to(q.dtype).transpose(1, 2)
    return out.squeeze(0) if was_3d else out

  def _uses_sliding_window(self) -> bool:
    if isinstance(self, VarlenInnerAttention):
      return self.window_size[0] >= 0
    # SDPA-based models set the window on the module, not in its config.
    return getattr(self, "sliding_window_size", None) is not None

  @spmd.no_typecheck()
  def _full_sequence_segment_ids_all_gather(
      self,
      positions: torch.Tensor,
      group: Any,
      bsz: int,
      q_seq_len: int,
  ) -> torch.Tensor:
    """All-gather sharded ``positions`` and return contiguous ``(B, S)`` ``segment_ids``."""
    cp = group.size()
    if _tracing():
      pos_ag = _wait(funcol.all_gather_tensor(positions, _TOKEN_DIM, group))
    else:
      pos_ag = spmd.redistribute(
          positions, group, src=spmd.S(_TOKEN_DIM), dst=spmd.R
      )
    pos_bs = (
        pos_ag.reshape(cp, bsz, q_seq_len)
        .movedim(0, 1)
        .reshape(bsz, cp * q_seq_len)
    )
    pos_bs = reorder_sequence(pos_bs, cp_size=cp, seq_dim=1, to_contiguous=True)
    return segment_ids_from_positions(pos_bs)

  def forward(
      self,
      q_THK: torch.Tensor,
      k_THK: torch.Tensor,
      v_THV: torch.Tensor,
      **kwargs,
  ) -> torch.Tensor:
    local_attn = self._local_attn_override
    if local_attn is None:
      if q_THK.device.type != "cpu" and not self.allow_reference_attention:
        raise RuntimeError(
            "KV all-gather CP has no attention kernel. Set"
            " splash_attention_kernel.use_splash_attention_kernel=True: the"
            " Splash patch plugs the kernel in via set_local_attention. The"
            " plain-PyTorch reference (local_forward) builds the full"
            " (S / cp, S) score matrix and does not support sliding windows;"
            " set allow_reference_attention=True to run it on TPU anyway"
            " (tests, tiny models)."
        )
      local_attn = self.local_forward
    group = spmd_mesh_group(MeshAxisName.CP)
    if group is None:
      raise RuntimeError("CP attention requires an active CP mesh axis.")
    cp = group.size()
    cp_rank = self._get_cp_rank_tensor(q_THK.device, group)

    type_check = _type_checking()
    if type_check:
      q_type = spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)

    was_3d = q_THK.ndim == 3
    if was_3d:
      total_q = q_THK.shape[0]
      full_seq_len = get_unflatten_seq_len()
      if (
          full_seq_len is not None
          and full_seq_len % cp == 0
          and total_q % (full_seq_len // cp) == 0
      ):
        q_seq_len = full_seq_len // cp
        bsz = total_q // q_seq_len
      else:
        bsz = 1
        q_seq_len = total_q
    else:
      bsz, q_seq_len = q_THK.shape[0], q_THK.shape[1]
    kv_seq_len = q_seq_len * cp

    # All-Gather K and V across the CP group (with fp32 Reduce-Scatter in backward).
    k_ag = _all_gather_kv(
        k_THK, group, seq_dim=_TOKEN_DIM, reduce_dtype=self.reduce_dtype
    )
    v_ag = _all_gather_kv(
        v_THV, group, seq_dim=_TOKEN_DIM, reduce_dtype=self.reduce_dtype
    )

    with spmd.no_typecheck():
      q_4d = q_THK.reshape(bsz, q_seq_len, q_THK.shape[-2], q_THK.shape[-1])
      k_4d = (
          k_ag.reshape(cp, bsz, q_seq_len, k_THK.shape[-2], k_THK.shape[-1])
          .movedim(0, 1)
          .reshape(bsz, kv_seq_len, k_THK.shape[-2], k_THK.shape[-1])
      )
      v_4d = (
          v_ag.reshape(cp, bsz, q_seq_len, v_THV.shape[-2], v_THV.shape[-1])
          .movedim(0, 1)
          .reshape(bsz, kv_seq_len, v_THV.shape[-2], v_THV.shape[-1])
      )
      k_4d = reorder_sequence(k_4d, cp_size=cp, seq_dim=1, to_contiguous=True)
      v_4d = reorder_sequence(v_4d, cp_size=cp, seq_dim=1, to_contiguous=True)

      if "enable_gqa" in kwargs:
        kwargs["enable_gqa"] = q_4d.shape[2] != k_4d.shape[2]

      positions = kwargs.pop("positions", None)
      if positions is not None:
        kwargs["segment_ids"] = self._full_sequence_segment_ids_all_gather(
            positions, group, bsz, q_seq_len
        )
      elif kwargs.get("segment_ids") is not None:
        seg = kwargs["segment_ids"]
        if seg.numel() == bsz * q_seq_len:
          seg_ag = (
              _wait(funcol.all_gather_tensor(seg, _TOKEN_DIM, group))
              if _tracing()
              else spmd.redistribute(
                  seg, group, src=spmd.S(_TOKEN_DIM), dst=spmd.R
              )
          )
          seg_bs = (
              seg_ag.reshape(cp, bsz, q_seq_len)
              .movedim(0, 1)
              .reshape(bsz, kv_seq_len)
          )
          kwargs["segment_ids"] = reorder_sequence(
              seg_bs, cp_size=cp, seq_dim=1, to_contiguous=True
          )

      kwargs["cp_rank"] = cp_rank
      out_4d = local_attn(q_4d, k_4d, v_4d, **kwargs)
      out_THV = (
          out_4d.reshape(bsz * q_seq_len, out_4d.shape[-2], out_4d.shape[-1])
          if was_3d
          else out_4d
      )

    if type_check and not spmd.has_local_type(out_THV):
      spmd.assert_type(out_THV, *q_type)
    return out_THV


class UlyssesCPScaledDotProductInnerAttention(
    UlyssesCPInnerAttention, ScaledDotProductInnerAttention
):

  @dataclass(kw_only=True, slots=True)
  class Config(ScaledDotProductInnerAttention.Config):
    pass


class UlyssesCPVarlenInnerAttention(
    UlyssesCPInnerAttention, VarlenInnerAttention
):

  @dataclass(kw_only=True, slots=True)
  class Config(VarlenInnerAttention.Config):
    pass


class KVAllGatherCPScaledDotProductInnerAttention(
    KVAllGatherCPInnerAttention, ScaledDotProductInnerAttention
):

  @dataclass(kw_only=True, slots=True)
  class Config(ScaledDotProductInnerAttention.Config):
    reduce_dtype: Literal["float32", "bfloat16"] = "float32"
    allow_reference_attention: bool = False
    """Run ``local_forward`` on TPU when no kernel is plugged in (tests, tiny models)."""


class KVAllGatherCPVarlenInnerAttention(
    KVAllGatherCPInnerAttention, VarlenInnerAttention
):

  @dataclass(kw_only=True, slots=True)
  class Config(VarlenInnerAttention.Config):
    reduce_dtype: Literal["float32", "bfloat16"] = "float32"
    allow_reference_attention: bool = False
    """Run ``local_forward`` on TPU when no kernel is plugged in (tests, tiny models)."""


CPStrategy = Literal["ulysses", "all_gather"]

# (single-device inner attention, Ulysses CP, KV All-Gather CP)
_CP_INNER_ATTENTIONS = (
    (
        ScaledDotProductInnerAttention,
        UlyssesCPScaledDotProductInnerAttention,
        KVAllGatherCPScaledDotProductInnerAttention,
    ),
    (
        VarlenInnerAttention,
        UlyssesCPVarlenInnerAttention,
        KVAllGatherCPVarlenInnerAttention,
    ),
)


def tpu_cp_inner_attention(base: type[Any], strategy: CPStrategy) -> type[Any]:
  """Returns the TPU CP inner attention for ``base`` under ``strategy``.

  ``base`` is a single-device inner attention class (e.g.
  ``ScaledDotProductInnerAttention``). The result is meant for core
  ``ContextParallelTransform(inner_attention=...)``.
  """
  if strategy not in ("ulysses", "all_gather"):
    raise ValueError(
        f"Unknown CP strategy {strategy!r}; expected 'ulysses' or 'all_gather'."
    )
  for single, ulysses, all_gather in _CP_INNER_ATTENTIONS:
    if base is single:
      return all_gather if strategy == "all_gather" else ulysses
  raise ValueError(
      f"No TPU CP inner attention for {base.__qualname__}; TPU CP supports"
      " ScaledDotProductInnerAttention and VarlenInnerAttention (Flex"
      " attention does not run on TPU)."
  )


def tpu_cp_load_balancer(strategy: CPStrategy) -> str | None:
  """Returns the ``context_parallel_load_balancer`` that ``strategy`` needs.

  KV All-Gather keeps Q sequence-sharded in head-tail order (see
  ``headtail.py``), which its Splash masks are built for; Ulysses attends over
  the full sequence, so there is no causal imbalance to balance.
  """
  return "headtail" if strategy == "all_gather" else None


def _validate_heads(
    fqn: str,
    attn: Any,
    tp: int,
    cp: int,
    strategy: CPStrategy,
) -> None:
  n_heads = attn.n_heads
  n_kv = getattr(attn, "n_kv_heads", None) or n_heads
  if strategy == "all_gather":
    if n_heads % tp:
      raise ValueError(
          f"{fqn}: n_heads ({n_heads}) must be divisible by tp ({tp})."
      )
    if n_kv % tp:
      raise ValueError(
          f"{fqn}: n_kv_heads ({n_kv}) must be divisible by tp ({tp})."
      )
    return
  if n_heads % (tp * cp):
    raise ValueError(
        f"{fqn}: n_heads ({n_heads}) must be divisible by tp * cp ({tp * cp})."
    )
  # KV heads are TP-sharded, then repeated up to cp if there are fewer.
  local_kv = n_kv // tp
  if n_kv % tp or (local_kv % cp and cp % local_kv):
    raise ValueError(
        f"{fqn}: n_kv_heads ({n_kv}) / tp ({tp}) must divide or be divisible"
        f" by cp ({cp})."
    )


def validate_tpu_context_parallel(model: Any, parallelism: Any) -> None:
  """Checks TPU CP attention against the parallelism config.

  Complements core ``validate_context_parallel`` (which checks the SPMD
  backend and that every attention is a ``CPInnerAttention``) with the
  TPU-specific rules: head divisibility per strategy, the load balancer each
  strategy expects, and no CP + TP/EP under FSDP2 on torch builds that
  ``install_tpu_cp_patches`` cannot fix. Does not modify anything.
  """
  cp = parallelism.context_parallel_degree
  if cp <= 1:
    return
  if (
      not hasattr(FSDPParam, "_resolve_spmd_types_for_storage")
      and not parallelism.use_simple_fsdp
      and (
          parallelism.tensor_parallel_degree > 1
          or parallelism.expert_parallel_degree > 1
      )
  ):
    raise NotImplementedError(
        "CP with TP or EP under FSDP2 needs a torch release with"
        " FSDPParam._resolve_spmd_types_for_storage; this one has none, and"
        " install_tpu_cp_patches only covers CP with FSDP/HSDP. Use"
        " parallelism.use_simple_fsdp, or turn off TP/EP."
    )
  for fqn, attn, _, _ in model.traverse(BaseAttention.Config):
    owner = attn.inner_attention._owner
    if owner is None:
      continue
    if issubclass(owner, KVAllGatherCPInnerAttention):
      strategy = "all_gather"
      expected_lb = "headtail"
    elif issubclass(owner, UlyssesCPInnerAttention):
      strategy = "ulysses"
      expected_lb = None
    else:
      continue
    if parallelism.context_parallel_load_balancer != expected_lb:
      raise ValueError(
          f"{fqn}.inner_attention uses {owner.__qualname__}, so"
          f" context_parallel_load_balancer must be {expected_lb!r}, got"
          f" {parallelism.context_parallel_load_balancer!r}."
      )
    _validate_heads(fqn, attn, parallelism.tensor_parallel_degree, cp, strategy)


def revert_tpu_context_parallel(model: Any) -> None:
  """Swaps TPU CP inner attentions back to their single-device versions.

  Lets a CP recipe be rerun on one device (cp == 1), e.g. as the CPU
  reference in the distributed parity tests; core validation rejects CP
  attention when cp == 1.
  """
  for _, attn, _, _ in model.traverse(BaseAttention.Config):
    owner = attn.inner_attention._owner
    for single, ulysses, all_gather in _CP_INNER_ATTENTIONS:
      if owner in (ulysses, all_gather):
        attn.inner_attention = single.Config(**{
            f.name: getattr(attn.inner_attention, f.name)
            for f in fields(single.Config)
        })


_patched = False


def install_tpu_cp_patches() -> None:
  """Lets FSDP2 shard over ``(dp_shard, cp)`` on torch without spmd_types FSDP.

  Released torch (e.g. 2.13.0) has no ``FSDPParam._resolve_spmd_types_for_
  storage`` and rejects the multi-axis ``DataParallelMeshDims`` that
  ``resolve_fsdp_mesh`` returns when CP is on; give it the flattened
  ``dp_shard_cp`` mesh instead. No-op on torch versions that have it. It does
  not cover CP with TP or EP, which ``validate_tpu_context_parallel`` rejects
  on such torch versions.

  TODO: Remove once the TPU stack moves to a torch release with
  ``FSDPParam._resolve_spmd_types_for_storage``.
  """
  global _patched
  if _patched:
    return
  _patched = True
  if hasattr(FSDPParam, "_resolve_spmd_types_for_storage"):
    return
  resolve = fsdp_mod.resolve_fsdp_mesh

  def resolve_flattened(parallel_dims):
    mesh, dims = resolve(parallel_dims)
    if (
        dims is None
        or not parallel_dims.cp_enabled
        or parallel_dims.tp_enabled
        or parallel_dims.ep_enabled
    ):
      return mesh, dims
    shard = mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
    if parallel_dims.dp_replicate_enabled:
      shard = DeviceMesh._concatenate([mesh["dp_replicate"], shard])
    return shard, None

  # Model parallelize modules bind the function at import time.
  for name, mod in list(sys.modules.items()):
    if (
        name.startswith("torchtitan.")
        and vars(mod).get("resolve_fsdp_mesh") is resolve
    ):
      mod.resolve_fsdp_mesh = resolve_flattened
