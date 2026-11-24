"""TPU configuration registry for Qwen3.8 (text-only) models.

Recipe helpers come from the Qwen3.5 TPU registry: Qwen3.8 is the same
architecture (see the package docstring).

Generation portability: the debug and proxy recipes set no libtpu flags and
run on TPU v6e and TPU7x alike. Only ``qwen38_2_4t_a95b``, which targets TPU7x,
adds TPU7x flags (``_TPU7X_FLAGS``).
"""

from torchtitan.components.optimizer.optimizer import default_adamw
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.components.datasets import c4_test_dataloader
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
  top_k` rows per rank: about 60 of the 91 GiB peak per rank of the TPU7x
  proxy, and 64x the routed rows at EP=64). `ep_capacity_factor=2.0` with
  `ep_capacity_overflow="grow"` sizes it at twice this rank's routed rows,
  doubling as needed so that no route is dropped (the default "drop" would
  drop the routes past the cap).
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


def qwen38_2_4t_a95b_proxy() -> TPUTrainerConfig:
  """Real-per-rank-shape proxy of Qwen3.8-2.4T-A95B for one TPU v6e-8 host.

  The full width (dim 8192, 64 query / 4 KV heads of 256, GatedDeltaNet 16
  key / 128 value heads, expert hidden size 2048, vocab 248320), the first 4
  layers (3 GatedDeltaNet + 1 full attention, one period of the layer
  pattern) and 64 routed experts, so EP=8 holds 8 experts per rank as EP=64
  does on the full model's 512. bf16 parameters and optimizer state
  (`training.dtype`), full activation checkpointing, plain SDPA and the
  reference GatedDeltaNet: rung 1, for shapes and numerics, not speed.

  2K tokens per rank: the ragged EP transport's worst-case receive buffer is
  `ep * T * top_k` rows (8 * 4096 * 10 x 8192 bf16 = 5.4 GB at 4K, which ran
  out of HBM on v6e). The `qwen35_moe_35b` LR schedule (200 warmup steps,
  clamped to `training.steps` in short runs): the debug recipe's 2-step
  warmup to 3e-4 spiked the loss (10.2 -> 14.0 at step 5, grad_norm 69).
  """
  cfg = qwen38_moe_debugmodel()
  seq_len = 2048
  cfg.hf_assets_path = "./tests/assets/tokenizer"
  cfg.model_spec = model_registry(
      "2.4T-A95B",
      seq_len=seq_len,
      converters=_TPU_MOE_CONVERTERS,
      num_layers=4,
      num_experts=64,
  )
  cfg.dataloader = c4_test_dataloader()
  cfg.training.max_context_length = seq_len
  cfg.training.num_tokens_per_microbatch_per_dp_rank = seq_len
  cfg.training.dtype = "bfloat16"
  cfg.parallelism = ParallelismConfig(
      data_parallel_shard_degree=-1,
      expert_parallel_degree=8,
      fsdp_reshard_after_forward="always",
      use_simple_fsdp=True,
  )
  _use_ragged_ep_transport(cfg)
  cfg.activation_checkpoint = FullAC.Config(preserve_rng_state=False)
  cfg.optimizer = default_adamw(lr=3e-4, implementation="foreach")
  cfg.lr_scheduler = qwen35_recipes.qwen35_moe_35b().lr_scheduler
  cfg.checkpoint.enable = False
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
