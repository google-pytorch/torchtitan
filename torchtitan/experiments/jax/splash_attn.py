"""Tokamax Splash Attention kernel setup."""

import jax

try:
  from tokamax._src.ops.splash_attention import splash_attention_kernel
  from tokamax._src.ops.splash_attention import splash_attention_mask
except ImportError:
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_kernel
  from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask
import torchtitan.tools.logging

logger = torchtitan.tools.logging.logger

P = jax.sharding.PartitionSpec


def _build_splash_config(config, query_shape, key_shape, attn_logits_soft_cap):
  return splash_attention_kernel.SplashConfig(
      block_q=min(config.sa_block_q, query_shape[2]),
      block_kv=min(config.sa_block_kv, key_shape[2]),
      block_kv_compute=min(config.sa_block_kv_compute, key_shape[2]),
      block_q_dkv=min(config.sa_block_q_dkv, query_shape[2]),
      block_kv_dkv=min(config.sa_block_kv_dkv, key_shape[2]),
      block_kv_dkv_compute=min(config.sa_block_kv_dkv_compute, query_shape[2]),
      block_q_dq=None
      if config.sa_use_fused_bwd_kernel
      else min(config.sa_block_q_dq, query_shape[2]),
      block_kv_dq=None
      if config.sa_use_fused_bwd_kernel
      else min(config.sa_block_kv_dq, query_shape[2]),
      use_fused_bwd_kernel=config.sa_use_fused_bwd_kernel,
      q_layout=splash_attention_kernel.QKVLayout[config.sa_q_layout],
      k_layout=splash_attention_kernel.QKVLayout[config.sa_k_layout],
      v_layout=splash_attention_kernel.QKVLayout[config.sa_v_layout],
      attn_logits_soft_cap=attn_logits_soft_cap,
      use_base2_exp=False,
  )


def build_splash_attention_callable(
    mesh,
    q_sharding,
    config,
    *,
    apply_shard_map: bool = True,
    attn_logits_soft_cap: float | None = None,
    local_window: int | None = None,
):
  """Return a jitted, sharded splash-attention callable.

  Signature of the returned callable:
      attn_fn(query, key, value, segment_ids) -> jax.Array
  where tensors are [batch, n_heads, seq, head_dim].

  When ``local_window`` is None the mask is full causal. When it is an int
  ``W``, the mask is causal-with-sliding-window: each query at position q can
  attend to keys at positions [q-W+1, q]. Models that mix local + global
  attention need separate callables (one per mask) by calling
  ``make_splash_attention_fn`` twice.
  """

  def _wrap_flash_attention(query, key, value, segment_ids):
    if segment_ids is not None:
      segment_ids = splash_attention_kernel.SegmentIds(segment_ids, segment_ids)
      assert (
          query.shape[2] == segment_ids.q.shape[1]
      ), 'Sharding along sequence dimension not allowed in tpu kernel attention'
    splash_config = _build_splash_config(
        config, query.shape, key.shape, attn_logits_soft_cap
    )
    if local_window is not None:
      # Causal local-window: each q at position p attends to keys in
      # [p - local_window + 1, p]. Splash's LocalMask uses the
      # ``window_size = (left, right)`` convention with offset=0:
      # ``left = local_window - 1`` past tokens included on the left,
      # ``right = 0`` no future tokens. ``shape = (S, S)``.
      mask = splash_attention_mask.LocalMask(
          shape=(query.shape[2], query.shape[2]),
          window_size=(local_window - 1, 0),
          offset=0,
      )
    else:
      mask = splash_attention_mask.CausalMask(
          shape=(query.shape[2], query.shape[2])
      )
    kernel = splash_attention_kernel.make_splash_mha(
        mask=mask,
        q_seq_shards=1,
        config=splash_config,
    )
    return jax.vmap(kernel)(query, key, value, segment_ids=segment_ids)

  fn = _wrap_flash_attention
  if apply_shard_map:
    fn = jax.experimental.shard_map.shard_map(
        fn,
        mesh=mesh,
        in_specs=(q_sharding, q_sharding, q_sharding, None),
        out_specs=q_sharding,
        check_rep=False,
    )
  return jax.jit(fn)


def make_splash_attention_fn(
        mesh, splash_cfg, *, local_window: int | None = None):
    """Build a splash-attention callable for the jax experiment.
    """
    try:
        q_sharding = P('fsdp', 'tp', None, None)
        attn_fn = build_splash_attention_callable(
            mesh, q_sharding, splash_cfg, apply_shard_map=True,
            local_window=local_window
        )
        if local_window is not None:
            logger.info(
                'Splash attention (local window=%d) enabled.', local_window)
        else:
            logger.info('Splash attention kernel enabled.')
        return attn_fn
    except Exception as e:
        logger.error(
            'Failed to build splash attention: %s — falling back to SDPA.', e
        )
        return None
