"""TPU JAX based kernel for fill_indices."""

import jax
import jax.numpy as jnp
import torch
import torch_tpu  # pylint: disable=unused-import
from torch.tpu import pallas as torch_pallas  # pyrefly: ignore[missing-import]

_fill_indices_kernel_cache = {}

def fill_indices(
    tokens_per_expert_group: torch.Tensor,
    start_index_values: torch.Tensor,
    write_offsets: torch.Tensor,
    experts_per_rank: int,
    num_ranks: int,
    max_len: int,
) -> torch.Tensor:
  """Prepare permutation indices and the number of tokens for each expert.

  This is a TPU JAX implementation of the `fill_indices` kernel, ported from
  `models/moe/kernels.py`. It leverages JAX primitives for loop representation,
  which translates to efficient scans on TPU, avoiding CPU/TPU data transfers
  and offering better performance than the original CPU version.

  Args:
      tokens_per_expert_group: number of tokens for each expert from all ranks.
      start_index_values: start index values for each expert from all ranks.
      write_offsets: write offsets for each expert.
      experts_per_rank: number of experts per rank.
      num_ranks: number of ranks.
      max_len: maximum length of the output index vector.

  Returns:
    A tensor of permutation indices.
  """

  cache_key = (experts_per_rank, num_ranks, max_len)
  if cache_key not in _fill_indices_kernel_cache:

    @jax.jit
    def fill_indices_jax(
        tpeg: jax.Array, siv: jax.Array, wo: jax.Array
    ) -> jax.Array:
      tpeg = tpeg.astype(jnp.int32)
      siv = siv.astype(jnp.int32)
      wo = wo.astype(jnp.int32)

      # Reshape to (experts_per_rank, num_ranks)
      # Original order in tpeg/siv is rank-major: (r0e0, r0e1, ..., r1e0, r1e1, ...)
      # We want to process in expert order: (e0r0, e0r1, ..., e1r0, e1r1, ...)
      tpeg_e = tpeg.reshape(num_ranks, experts_per_rank).T
      siv_e = siv.reshape(num_ranks, experts_per_rank).T

      # Calculate write starts for each (expert, rank) pair
      # Shifted cumsum along ranks for each expert
      offsets_within_expert = jnp.cumsum(tpeg_e, axis=1) - tpeg_e
      write_starts = wo[:, jnp.newaxis] + offsets_within_expert

      # Fully vectorized JAX approach:
      total_elements = num_ranks * experts_per_rank

      lengths = tpeg_e.reshape(-1)
      starts = siv_e.reshape(-1)
      w_starts = write_starts.reshape(-1)

      # Use fori_loop to process each segment sequentially
      def segment_body(i, current_out):
        length = lengths[i]
        start = starts[i]
        w_start = w_starts[i]

        def atom_body(j, atom_out):
          target_idx = w_start + j
          # Using at[idx].set(val) in a loop is safe in JAX JIT
          return jax.lax.cond(
              target_idx < max_len,
              lambda o: o.at[target_idx].set(start + j),
              lambda o: o,
              atom_out,
          )

        return jax.lax.fori_loop(0, length, atom_body, current_out)

      out = jnp.full((max_len,), -1, dtype=jnp.int32)
      out = jax.lax.fori_loop(0, total_elements, segment_body, out)
      return out

    op_name = (
        f"pallas::fill_indices_tpu_{experts_per_rank}_{num_ranks}_{max_len}"
    )
    _fill_indices_kernel_cache[cache_key] = torch_pallas.jax_op(
        op_name,
        fill_indices_jax,
    )

  return _fill_indices_kernel_cache[cache_key](
      tokens_per_expert_group.to(torch.int32),
      start_index_values.to(torch.int32),
      write_offsets.to(torch.int32),
  )
