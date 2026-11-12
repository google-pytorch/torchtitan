# Copyright 2026 Google LLC
# TT-UNFLATTEN-SHIM:
# Temporary compatibility shim to unflatten 1D [T] token streams back to [B, S, ...]
# for legacy TPU kernels that have not yet been migrated to natively support [T] shapes.
#
# REMOVAL INSTRUCTIONS:
# When the TPU kernels are updated to support [T] sizes directly:
# 1. Delete this file entirely.
# 2. Grep for "TT-UNFLATTEN-SHIM" across experiments/tpu/** and delete the wrappers
#    in workarounds.py, splash_bias_dropout.py, mamba2/model.py, train.py, etc.
# 3. Quilt delete 0006_unflatten_shim.patch.

import contextlib
import torch


# NOTE: This is a mutable module-level global set by a context manager.
# It is NOT safe across threads or concurrent trainers. 
# This is an intentional architectural concession for a temporary shim to avoid
# invasive plumbing into models that do not natively carry seq_len on their configs.
_ACTIVE_SEQ_LEN: int | None = None

@contextlib.contextmanager
def set_unflatten_seq_len(seq_len: int):
    """Context manager to activate unflatten mapping with the exact S dimension."""
    global _ACTIVE_SEQ_LEN
    old = _ACTIVE_SEQ_LEN
    _ACTIVE_SEQ_LEN = seq_len
    try:
        yield
    finally:
        _ACTIVE_SEQ_LEN = old

def get_unflatten_seq_len() -> int | None:
    return _ACTIVE_SEQ_LEN

def unflatten_tensor(t: torch.Tensor, expected_flat_ndim: int = 3) -> torch.Tensor:
    """
    Unflattens dim 0 from [T] to [B, S].
    - Safe bypass: Returns `t` untouched if _ACTIVE_SEQ_LEN is None or `t.ndim != expected_flat_ndim`.
    - Loud failure: Asserts T % S == 0 naming the shim.
    """
    if not isinstance(t, torch.Tensor) or t.ndim != expected_flat_ndim:
        return t
    
    seq_len = get_unflatten_seq_len()
    if seq_len is None:
        # Safe default bypass if not inside a trainer.train() loop (e.g. kernel tests)
        return t

    num_tokens = t.shape[0]
    if num_tokens % seq_len != 0:
        raise ValueError(
            f"TT-UNFLATTEN-SHIM: num_tokens ({num_tokens}) is not cleanly divisible "
            f"by seq_len ({seq_len}). Varlen/packed sequences are not supported by "
            "the legacy block unflattening constraints. To fix, either migrate "
            "kernels to native [T] routing or disable sequence packing."
        )
    
    return t.unflatten(0, (num_tokens // seq_len, seq_len))

def flatten_tensor(t: torch.Tensor, original_flat_ndim: int = 3) -> torch.Tensor:
    """
    Flattens [B, S] back to [T] if and only if tensor length is original_flat_ndim + 1.
    """
    if not isinstance(t, torch.Tensor) or t.ndim != (original_flat_ndim + 1):
        return t
    
    seq_len = get_unflatten_seq_len()
    if seq_len is None:
        return t

    return t.flatten(0, 1)

class UnflattenShim(torch.nn.Module):
    """
    TT-UNFLATTEN-SHIM: nn.Module wrapper acting as a kernel boundary unflatten/flatten mechanism.
    """
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        
    def forward(self, *args, **kwargs):
        # Unflatten primary arguments (usually q, k, v) - explicitly limit to indices 0, 1, 2
        # to avoid silently reshaping arbitrary 3D tensors like biases or masks.
        target_indices = {0, 1, 2}
        new_args = [
            unflatten_tensor(a, expected_flat_ndim=3) if i in target_indices and isinstance(a, torch.Tensor) else a 
            for i, a in enumerate(args)
        ]
        
        # Unflatten relevant keyword tensors
        new_kwargs = {}
        target_keys = {"q", "k", "v", "query", "key", "value"}
        target_1d_keys = {"positions", "segment_ids"}
        
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if k in target_keys:
                    new_kwargs[k] = unflatten_tensor(v, expected_flat_ndim=3)
                elif k in target_1d_keys:
                    new_kwargs[k] = unflatten_tensor(v, expected_flat_ndim=1)
                else:
                    new_kwargs[k] = v
            else:
                new_kwargs[k] = v
                
        out = self.inner(*new_args, **new_kwargs)
        
        # Re-flatten the output only if an input was actually unflattened.
        # Models that already pass [B, S, H, D] (e.g. flux) must get the same
        # layout back.
        unflattened = any(
            n is not o
            for n, o in zip(
                list(new_args) + list(new_kwargs.values()),
                list(args) + list(kwargs.values()),
            )
        )
        if not unflattened:
            return out
        if isinstance(out, tuple):
            return tuple(flatten_tensor(o, original_flat_ndim=3) if isinstance(o, torch.Tensor) else o for o in out)
        elif isinstance(out, torch.Tensor):
            return flatten_tensor(out, original_flat_ndim=3)
        return out
