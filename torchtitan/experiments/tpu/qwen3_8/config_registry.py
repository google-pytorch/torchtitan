"""TPU configuration registry for Qwen3.8 (text-only) models.

Recipe helpers come from the Qwen3.5 TPU registry: Qwen3.8 is the same
architecture (see the package docstring).

Generation portability: the debug recipes and `qwen38_2_4t_a95b_proxy` set no
libtpu flags and run on TPU v6e and TPU7x alike. The `*_tpu7x` proxies and
``qwen38_2_4t_a95b``, which target TPU7x, add TPU7x flags (``_TPU7X_FLAGS``).
"""

from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.components.datasets import c4_test_dataloader
from torchtitan.experiments.tpu.components.grouped_experts import TpuGroupedExperts
from torchtitan.experiments.tpu.components.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
    TpuRoutedExpertsConverter,
)
from torchtitan.experiments.tpu.qwen3_5 import config_registry as qwen35_recipes
from torchtitan.experiments.tpu.qwen3_8 import model_registry
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig

# The Qwen3.5 TPU MoE converters plus `TpuRoutedExperts`, which binds the
# EP rank that the ragged EP transport needs (a no-op otherwise, with
# `scale_before_down_proj` off). As in the Qwen3 EP recipes.
# pylint: disable-next=protected-access
_TPU_MOE_CONVERTERS = qwen35_recipes._TPU_MOE_CONVERTERS + [
    TpuRoutedExpertsConverter.Config()
]

# TPU7x only. The largest scoped VMEM limit TPU7x accepts (as in
# `qwen35_moe_35b`); the Splash backward at head_dim 256 needs more than the
# 32 MiB default.
_TPU7X_FLAGS = tpu_flags.merge_flags({"xla_tpu_scoped_vmem_limit_kib": 65472})


def _use_ragged_ep_transport(cfg: TPUTrainerConfig) -> None:
  """Selects the dropless `ragged_all_to_all` EP transport, bounded.

  torch_tpu's `all_to_all` rejects asymmetric output buffers, which
  upstream's variable-size EP transport produces. The ragged transport's
  receive buffer would otherwise be sized for the worst case (`ep_size * T *
  top_k` rows per rank: 64x the routed rows at EP=64, 10.7 GB bf16 per buffer
  at dim 8192 and T = 1024). `ep_capacity_factor=2.0` with
  `ep_capacity_overflow="grow"` sizes it at twice this rank's routed rows,
  doubling as needed so that no route is dropped (the default "drop" would
  drop the routes past the cap). The buffer then grows with routing
  imbalance: on the TPU7x proxies from random initialization it reached 8x
  the initial size at 8 layers.
  """
  assert cfg.model_spec is not None
  dispatchers = list(
      cfg.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
  )
  if not dispatchers:
    raise ValueError("Expected TpuAllToAllTokenDispatcher (_TPU_MOE_CONVERTERS)")
  for _, dispatcher, _, _ in dispatchers:
    dispatcher.ep_transport = "ragged_all_to_all"
    dispatcher.ep_capacity_factor = 2.0
    dispatcher.ep_capacity_overflow = "grow"


def qwen38_debugmodel() -> TPUTrainerConfig:
  """Debug config for hybrid Qwen3.8 (dense) on TPU (= `qwen35_debugmodel`)."""
  cfg = qwen35_recipes.qwen35_debugmodel()
  cfg.model_spec = model_registry("debugmodel", seq_len=256)
  return cfg


def qwen38_moe_debugmodel() -> TPUTrainerConfig:
  """Debug config for 4-layer hybrid Qwen3.8 MoE on TPU.

  TPU token dispatcher and grouped experts, with upstream's expert-bias load
  balancer (`load_balance_coeff`), as upstream's `qwen38_debugmodel_moe`.
  """
  cfg = qwen38_debugmodel()
  cfg.model_spec = model_registry(
      "debugmodel_moe", seq_len=256, converters=_TPU_MOE_CONVERTERS
  )
  return cfg


def qwen38_moe_debugmodel_ep8() -> TPUTrainerConfig:
  """`qwen38_moe_debugmodel` with EP=8 on 8 ranks (FSDP 8)."""
  cfg = qwen38_moe_debugmodel()
  cfg.parallelism.expert_parallel_degree = 8
  _use_ragged_ep_transport(cfg)
  return cfg


def _real_shape_proxy(
    *,
    num_layers: int,
    num_experts: int,
    expert_parallel_degree: int,
    seq_len: int,
    max_seq_len: int | None = None,
) -> TPUTrainerConfig:
  """A Qwen3.8-2.4T-A95B proxy: full width, fewer layers and experts.

  The full width (dim 8192, 64 query / 4 KV heads of 256, GatedDeltaNet 16
  key / 128 value heads, expert hidden size 2048, vocab 248320), the first
  `num_layers` layers and `num_experts` routed experts. With
  `num_experts / expert_parallel_degree == 8` every rank holds 8 experts, as
  EP=64 does on the full model's 512. bf16 parameters and optimizer state
  (`training.dtype`), full activation checkpointing, plain SDPA and the
  reference GatedDeltaNet: rung 1, for shapes, memory and numerics, not
  speed. One sequence of `seq_len` tokens per rank, on the bounded dropless
  EP transport (`_use_ragged_ep_transport`). The
  `qwen35_moe_35b` LR schedule (200 warmup steps, clamped to
  `training.steps` in short runs); a 2-step warmup to 3e-4 spiked the loss.
  `max_seq_len` (default `seq_len`) sizes the model's RoPE table, the upper
  bound for `--training.max_context_length` overrides.
  """
  cfg = qwen38_moe_debugmodel()
  cfg.hf_assets_path = "./tests/assets/tokenizer"
  cfg.model_spec = model_registry(
      "2.4T-A95B",
      seq_len=max_seq_len or seq_len,
      converters=_TPU_MOE_CONVERTERS,
      num_layers=num_layers,
      num_experts=num_experts,
  )
  cfg.dataloader = c4_test_dataloader()
  cfg.training.max_context_length = seq_len
  cfg.training.num_tokens_per_microbatch_per_dp_rank = seq_len
  cfg.training.dtype = "bfloat16"
  cfg.parallelism = ParallelismConfig(
      data_parallel_shard_degree=-1,
      expert_parallel_degree=expert_parallel_degree,
      fsdp_reshard_after_forward="always",
      use_simple_fsdp=True,
  )
  _use_ragged_ep_transport(cfg)
  cfg.activation_checkpoint = FullAC.Config(preserve_rng_state=False)
  cfg.optimizer = default_adamw(lr=3e-4, implementation="foreach")
  cfg.lr_scheduler = qwen35_recipes.qwen35_moe_35b().lr_scheduler
  cfg.checkpoint.enable = False
  return cfg


def qwen38_2_4t_a95b_proxy() -> TPUTrainerConfig:
  """2.4T-A95B proxy for one TPU v6e-8 host: 4 layers, 64 experts, EP=8.

  4 layers are one period of the 3:1 GatedDeltaNet / attention pattern. 2K
  tokens per rank peak at 30.8 of the 32 GiB per v6e chip. See
  `_real_shape_proxy`.
  """
  return _real_shape_proxy(
      num_layers=4, num_experts=64, expert_parallel_degree=8, seq_len=2048
  )


def _use_tpu7x_expert_tiles(cfg: TPUTrainerConfig) -> None:
  """Runs the routed experts with the Pallas grouped matmul in 512x2048x1024 tiles.

  On TPU7x, XLA's lowering of `torch._grouped_mm` for these expert shapes
  (dim 8192, hidden 2048) needs 64.9 MB of scoped VMEM at the 63.9 MB limit
  `_TPU7X_FLAGS` sets, and the Pallas grouped matmul's default tiles (whole
  K x N weight panels) need 108-120 MB. These tiles fit, ran on a TPU7x 4x4x4
  slice, and on TPU v6e match `torch._grouped_mm` at the per-rank shapes
  (8 experts, empty leading experts included) to a relative error of about
  1e-4 in the output and every gradient. The paired variant needs EP=1, so
  this is the plain one.
  """
  for _, experts, _, _ in cfg.model_spec.model.traverse(TpuGroupedExperts.Config):
    experts.use_pallas_grouped_mm = True
    experts.pallas_grouped_mm_tile_m = 512
    experts.pallas_grouped_mm_tile_k = 2048
    experts.pallas_grouped_mm_tile_n = 1024


def qwen38_2_4t_a95b_proxy_tpu7x() -> TPUTrainerConfig:
  """2.4T-A95B proxy for one TPU7x 4x4x4 slice: 4 layers, 512 experts, EP=64.

  The full model's 512 experts at its EP degree (8 experts per rank) on 64
  chips / 128 ranks, with the TPU7x libtpu flags and the Pallas grouped
  matmul experts (`_use_tpu7x_expert_tiles`). 1K tokens per rank by default
  (39 GiB peak per rank at 4 layers, 50 GiB at 8). Raise
  `--training.max_context_length` and
  `--training.num_tokens_per_microbatch_per_dp_rank` together to sweep it, up
  to 8192 (the RoPE table's length).
  See `_real_shape_proxy` and
  `recipes/ironwood/qwen3.8-2.4t-a95b/tpu7x-4x4x4/README.md`.
  """
  cfg = _real_shape_proxy(
      num_layers=4,
      num_experts=512,
      expert_parallel_degree=64,
      seq_len=1024,
      max_seq_len=8192,
  )
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags, _TPU7X_FLAGS
  )
  _use_tpu7x_expert_tiles(cfg)
  return cfg


def qwen38_2_4t_a95b_proxy_tpu7x_8l() -> TPUTrainerConfig:
  """`qwen38_2_4t_a95b_proxy_tpu7x` with 8 layers (two pattern periods).

  The second point of the layer sweep, from which per-layer memory and step
  time extrapolate to 92 layers.
  """
  cfg = _real_shape_proxy(
      num_layers=8,
      num_experts=512,
      expert_parallel_degree=64,
      seq_len=1024,
      max_seq_len=8192,
  )
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags, _TPU7X_FLAGS
  )
  _use_tpu7x_expert_tiles(cfg)
  return cfg


def qwen38_2_4t_a95b() -> TPUTrainerConfig:
  """Qwen3.8-2.4T-A95B (92 hybrid layers, 512 experts) on TPU7x.

  Starting point for the full model, not yet run: EP=64, FSDP over the rest,
  TP=1, 4K context, FP32 parameters, gradients and AdamW state with bf16
  compute, full activation checkpointing, the expert-bias balancer, and the
  `qwen35_moe_35b` kernels (Pallas paired grouped matmuls, Splash with
  1024 blocks for head_dim 256, Pallas fused linear cross-entropy). At 92
  layers the regular trainer's single step graph is not expected to compile
  in reasonable time; the graph-split or scan variant comes with Phase 4.
  """
  cfg = qwen35_recipes.qwen35_moe_35b()
  seq_len = 4096
  cfg.hf_assets_path = "./assets/hf/Qwen3.8-2.4T-A95B"
  cfg.model_spec = model_registry(
      "2.4T-A95B", seq_len=seq_len, converters=_TPU_MOE_CONVERTERS
  )
  # pylint: disable-next=protected-access
  qwen35_recipes._use_tpu_moe_kernels(cfg, tile_m=512)
  cfg.training.max_context_length = seq_len
  cfg.training.num_tokens_per_microbatch_per_dp_rank = 4 * seq_len
  cfg.parallelism = ParallelismConfig(
      data_parallel_shard_degree=-1,
      expert_parallel_degree=64,
      fsdp_reshard_after_forward="always",
      use_simple_fsdp=True,
  )
  # The compiled gradient norm handles one gradient list; EP splits dense and
  # expert gradients across meshes (`TPUTrainerConfig.__post_init__`).
  cfg.compile.compile_grad_norm = False
  cfg.libtpu_init_flags = tpu_flags.merge_flags(
      cfg.libtpu_init_flags, _TPU7X_FLAGS
  )
  return cfg
