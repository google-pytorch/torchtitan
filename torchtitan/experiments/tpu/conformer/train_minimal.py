r"""Conformer minimal training script using torchtitan infra.

Example:
torchrun --nproc_per_node=4 \
  -m torchtitan.experiments.tpu.conformer.train_minimal \
  --module=torchtitan.experiments.tpu.conformer \
  --config=conformer_test \
  --tpu_config.use_simple_fsdp \
  --compile.mode=layer \
  --metrics.log_freq=1 \
  --compile.enable
"""

import os
import time
import typing
from typing import Tuple

from jax import profiler as jax_profiler
import torch
import torch.nn.functional as F
from torchtitan.components import metrics
import torchtitan.components.tokenizer
import torchtitan.config
import torchtitan.distributed
from torchtitan.distributed import utils as dist_utils
import torchtitan.experiments.tpu.conformer  # trigger conformer_tpu model registration
import torchtitan.experiments.tpu.gmain as gmain
import torchtitan.experiments.tpu.model_annotator as model_annotator
from torchtitan.experiments.tpu.tpu_job_config import TPUTrainerConfig
import torchtitan.experiments.tpu.utils as tpu_utils
from torchtitan.tools import utils
import torchtitan.tools.logging
import torchtitan.experiments.tpu.profiler_workaround as profiler_workaround
import torchtitan.tools.profiler as torchtitan_profiler

TORCH_DTYPE_MAP = torchtitan.config.TORCH_DTYPE_MAP
ParallelDims = torchtitan.distributed.ParallelDims
logger = torchtitan.tools.logging.logger


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------


def train_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    materialize_forward_before_backward: bool = False,
    use_ctc: bool = False,
    use_jax_ctc: bool = False,
    output_lengths: torch.Tensor = None,
    target_lengths: torch.Tensor = None,
) -> torch.Tensor:
  """Single training step."""
  if use_ctc and use_jax_ctc:
    raise ValueError(
        "Only one of use_ctc and use_jax_ctc can be enabled at the same time."
    )
  with jax_profiler.TraceAnnotation("optimizer.zero_grad"):
    optimizer.zero_grad()

  if materialize_forward_before_backward:
    import torch_tpu._internal.sync  # pylint: disable=g-import-not-at-top

  logits = model(inputs.to(torch.long))
  with jax_profiler.TraceAnnotation("loss"):
    if use_jax_ctc:
      from torchtitan.experiments.tpu.conformer import jax_ctc_loss  # pylint: disable=g-import-not-at-top
      # JAX CTC loss expects batch-first layouts:
      #   logits: (B, T, C) where B=batch, T=time/frames, C=vocabulary
      #   targets: (B, L) where L=target text label length
      # Conformer outputs logits as (B, S, C) where sequence length S is identical to T.
      loss = jax_ctc_loss.jax_ctc_loss(logits, targets)
    elif use_ctc:
      # PyTorch CTC loss expects time-first layouts:
      #   log_probs: (T, B, C) where T=time/frames, B=batch, C=vocabulary
      # We transpose Conformer's batch-first logits (B, S, C) to time-first (T, B, C).
      logits_f32 = logits.float()
      log_probs = torch.nn.functional.log_softmax(logits_f32, dim=-1).transpose(
          0, 1
      )
      # TODO: b/513607161 - change to torch.nn.CTCLoss when ready.
      loss, _ = torch.ops.aten._ctc_loss.Tensor(
          log_probs,
          targets.to(torch.long),
          output_lengths,
          target_lengths,
          blank=0,
          zero_infinity=False,
      )
      loss = loss.sum()
    else:
      loss = F.cross_entropy(
          logits.reshape(-1, logits.shape[-1]),
          targets.to(torch.long).reshape(-1),
          reduction="sum",
      )

  if materialize_forward_before_backward:
    torch_tpu._internal.sync.synchronize(loss, wait=False)
  loss.backward()

  with jax_profiler.TraceAnnotation("optimizer.step"):
    optimizer.step()

  return loss


def build_random_dataloader(train_config, vocab_size, device):
  # Upstream d771abb0 renamed training.seq_len -> training.max_context_length
  # and replaced training.local_batch_size with the token-based
  # training.num_tokens_per_microbatch_per_dp_rank.
  seq_len = train_config.training.max_context_length
  local_batch_size = (
      train_config.training.num_tokens_per_microbatch_per_dp_rank // seq_len)
  # Inputs should be token IDs for the embedding layer
  x = torch.randint(0, vocab_size, (local_batch_size, seq_len), device=device)

  is_any_ctc_loss_enabled = hasattr(train_config, "conformer") and (
      train_config.conformer.use_ctc_loss
      or train_config.conformer.use_jax_ctc_loss
  )

  if is_any_ctc_loss_enabled:
    # Make targets shorter for CTC loss (input length must be >= target length)
    target_len = getattr(train_config.conformer, "ctc_loss_target_len", 30)
    targets = torch.randint(
        1, vocab_size, (local_batch_size, target_len), device=device
    )
  else:
    targets = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )

  while True:
    yield {"input": x}, targets


def start_trainer(train_config: TPUTrainerConfig) -> None:
  rank = int(os.environ.get("RANK", 0))
  world_size = int(os.environ.get("WORLD_SIZE", 1))

  if rank == 0:
    torchtitan.tools.logging.init_logger()
    train_config.maybe_log()

  device = utils.get_local_device()

  if world_size == 1 and device.type == "tpu":
    from torch_tpu._internal.device._device_module import TpuDeviceModule

    if not TpuDeviceModule.is_initialized():
      logger.info("Initializing PjRt runtime for single chip.")
      try:
        TpuDeviceModule._init_runtime_options()
      except RuntimeError as e:
        if "PrivateUse1HooksInterface only could be registered once" in str(e):
          logger.warning("PjRt hooks already registered, ignoring error.")
        else:
          raise

  if world_size > 1:
    dist_utils.init_distributed(
        train_config.comm,
        enable_cpu_backend=train_config.training.enable_cpu_offload,
        base_folder=train_config.dump_folder,
    )
    torch.distributed.barrier()

  dp_replicate = train_config.parallelism.data_parallel_replicate_degree
  dp_shard = train_config.parallelism.data_parallel_shard_degree

  if dp_replicate == -1:
    dp_shard = 1
    cp = train_config.parallelism.context_parallel_degree
    tp = train_config.parallelism.tensor_parallel_degree
    pp = train_config.parallelism.pipeline_parallel_degree
    dp_replicate = world_size // (dp_shard * cp * tp * pp)

  parallel_dims = ParallelDims(
      dp_shard=dp_shard,
      dp_replicate=dp_replicate,
      cp=train_config.parallelism.context_parallel_degree,
      tp=train_config.parallelism.tensor_parallel_degree,
      pp=train_config.parallelism.pipeline_parallel_degree,
      ep=train_config.parallelism.expert_parallel_degree,
      world_size=world_size,
  )
  logger.info("parallel_dims: %s", parallel_dims)
  train_config.maybe_log()

  batch_degree, batch_rank = 1, 0
  if world_size > 1:
    if parallel_dims.dp_enabled:
      batch_mesh = parallel_dims.get_mesh("batch")
      batch_degree, batch_rank = (
          batch_mesh.size(),
          batch_mesh.get_local_rank(),
      )
    else:
      batch_degree, batch_rank = 1, 0

  seed = train_config.debug.seed or 42
  if utils.device_type == "tpu":
    torch.manual_seed(seed)
    logger.info("Set manual seed to %d", seed)
  else:
    if world_size > 1:
      dist_utils.set_determinism(
          parallel_dims,
          device,
          train_config.debug,
          distinct_seed_mesh_dims=["pp"],
      )

  materialize_forward_before_backward = (
      isinstance(train_config, TPUTrainerConfig)
      and train_config.compile.materialize_forward_before_backward
  )
  log_freq = train_config.metrics.log_freq

  assert train_config.model_spec is not None
  model_spec = train_config.model_spec
  model_args = model_spec.model
  model_args.update_from_config(train_config=train_config)

  tokenizer = typing.cast(
      torchtitan.components.tokenizer.HuggingFaceTokenizer,
      train_config.tokenizer.build(tokenizer_path=train_config.hf_assets_path),
  )

  if train_config.dataloader.dataset == "random":
    logger.info("Using random data loader instead of real dataset.")
    dataloader = build_random_dataloader(
        train_config, model_args.vocab_size, device  # pytype: disable=attribute-error
    )
  else:
    # Upstream d771abb0 renamed training.seq_len -> training.max_context_length
    # and replaced training.local_batch_size with the token-based
    # training.num_tokens_per_microbatch_per_dp_rank; the dataloader is now
    # components.data.GrainDataLoader, which takes max_context_length /
    # num_tokens_per_batch instead of seq_len / local_batch_size.
    dataloader = train_config.dataloader.build(
        dp_world_size=batch_degree,
        dp_rank=batch_rank,
        tokenizer=tokenizer,
        max_context_length=train_config.training.max_context_length,
        num_tokens_per_batch=(
            train_config.training.num_tokens_per_microbatch_per_dp_rank),
    )

  logger.info(
      "Building %s %s with %s",
      model_spec.name,
      model_spec.flavor,
      model_args,
  )

  metrics_processor = train_config.metrics.build(
      parallel_dims=parallel_dims,
      dump_folder=train_config.dump_folder,
      pp_schedule=train_config.parallelism.pipeline_parallel_schedule,
      config_dict=train_config.to_dict(),
      has_quantization=False,
  )

  with (
      torch.device("meta"),
      utils.set_default_dtype(TORCH_DTYPE_MAP[train_config.training.dtype]),
  ):
    model = model_args.build()

  # Upstream d771abb0 renamed training.seq_len -> training.max_context_length.
  model_param_count, _ = model_args.get_nparams_and_flops(
      model, train_config.training.max_context_length
  )
  logger.info(
      "Model %s %s size: %d total parameters",
      model_spec.name,
      model_spec.flavor,
      model_param_count,
  )

  if train_config.training.enable_cpu_offload:
    logger.info("Materializing model on CPU for CPU offloading")
    model = model.to_empty(device="cpu")

  use_simple_fsdp = (
      isinstance(train_config, TPUTrainerConfig)
      and train_config.tpu_config.use_simple_fsdp
  )

  if use_simple_fsdp:
    # SimpleFSDP requires weights to be materialized and initialized BEFORE
    # wrapping/parallelization. This is because SimpleFSDP replaces module
    # parameters with properties, which breaks in-place initialization (like
    # model.init_weights()) if called after wrapping.
    logger.info(
        "Moving model to device %s before parallelization (SimpleFSDP flow).",
        device,
    )
    if not train_config.training.enable_cpu_offload:
      model = model.to_empty(device=device)
    with torch.no_grad():
      model.init_weights()

  model_compile_enabled = (
      train_config.compile.enable and "model" in train_config.compile.components
  )
  if world_size > 1 or model_compile_enabled:
    if model_spec.parallelize_fn is not None:
      compiled_model = model_spec.parallelize_fn(
          model, parallel_dims, train_config
      )
      if compiled_model is not None:
        model = compiled_model
    else:
      logger.warning("No parallelize_fn provided in ModelSpec.")

  if not use_simple_fsdp:
    # For FSDP2 and other strategies, we prefer to parallelize (and optionally
    # compile) FIRST while the model is still on the meta device, and then
    # materialize and initialize weights on the correct device afterwards.
    logger.info(
        "Moving model to device %s after parallelization (Standard flow).",
        device,
    )
    if not train_config.training.enable_cpu_offload:
      model = model.to_empty(device=device)
    with torch.no_grad():
      model.init_weights()

  loss_fn = None  # Handled in train_step

  if not train_config.compile.enable and train_config.profiler.enable_profiling:
    model_annotator.wrap_model(model)

  trainable_params = [p for p in model.parameters() if p.requires_grad]
  logger.info(
      "Trainable parameters: %d / %d",
      sum(p.numel() for p in trainable_params),
      model_param_count,
  )

  if train_config.optimizer.implementation == "fused":
    logger.warning(
        "Fused optimizer is not supported on TPU, changing to foreach."
    )
    train_config.optimizer.implementation = "foreach"

  optimizer = train_config.optimizer.build(model_parts=[model])

  if train_config.compile.optimizer_compile == "separate_graph":
    logger.info("Applying torch.compile to optimizer.step")
    optimizer.step = torch.compile(
        optimizer.step,
        backend=train_config.compile.backend,
        fullgraph=True,
        dynamic=False,
    )

  if torch.distributed.is_initialized():
    torch.distributed.barrier()

  steps = train_config.training.steps
  # Upstream d771abb0 replaced training.local_batch_size * training.seq_len
  # with the single token-based training.num_tokens_per_microbatch_per_dp_rank.
  num_tokens_per_step = (
      train_config.training.num_tokens_per_microbatch_per_dp_rank)

  # Upstream d771abb0 renamed training.seq_len -> training.max_context_length.
  (
      model_param_count,
      num_flops_per_token,
  ) = model_args.get_nparams_and_flops(
      model, train_config.training.max_context_length
  )
  metrics_processor.num_flops_per_token = num_flops_per_token

  device_name = utils.device_module.get_device_name()
  peak_flops = utils.get_peak_flops(device_name)
  logger.info(
      "Peak FLOPS for %s used for computing MFU: %d",
      device_name,
      peak_flops,
  )

  total_tokens = 0
  total_time = 0.0
  accumulated_tokens = 0
  accumulated_time = 0.0
  accumulated_steps = 0

  warmup_steps = train_config.lr_scheduler.warmup_steps

  profiler_workaround.init(train_config)

  ntokens_seen = 0
  with torchtitan_profiler.Profiler(
      train_config.profiler,
      global_step=0,
      base_folder=train_config.dump_folder,
  ) as profiler:
    data_iterator = iter(dataloader)
    for step in range(steps):
      step_start = time.perf_counter()
      step_tokens = num_tokens_per_step
      ntokens_seen += step_tokens
      metrics_processor.ntokens_since_last_log += step_tokens

      t0 = time.perf_counter()
      batch = next(data_iterator)
      metrics_processor.data_loading_times.append(time.perf_counter() - t0)

      if parallel_dims.fsdp_enabled or (
          isinstance(train_config, TPUTrainerConfig)
          and train_config.tpu_config.use_simple_fsdp
      ):
        compute_dtype = TORCH_DTYPE_MAP[
            train_config.training.mixed_precision_param
        ]
        inputs = batch[0]["input"].to(device, dtype=compute_dtype)
      else:
        inputs = batch[0]["input"].to(
            device, dtype=next(model.parameters()).dtype
        )
      targets = batch[1].to(device)

      with jax_profiler.TraceAnnotation("train", step_num=step):
        use_ctc = (
            hasattr(train_config, "conformer")
            and train_config.conformer.use_ctc_loss
        )
        use_jax_ctc = (
            hasattr(train_config, "conformer")
            and train_config.conformer.use_jax_ctc_loss
        )
        loss = train_step(
            model,
            inputs,
            targets,
            optimizer,
            loss_fn,
            materialize_forward_before_backward=(
                materialize_forward_before_backward
            ),
            use_ctc=use_ctc,
            use_jax_ctc=use_jax_ctc,
            output_lengths=torch.full(
                (inputs.size(0),),
                inputs.size(1),
                dtype=torch.long,
                device=device,
            ),
            target_lengths=torch.full(
                (targets.size(0),),
                targets.size(1),
                dtype=torch.long,
                device=device,
            ),
        )

      should_log_step = (
          metrics_processor.should_log(step + 1)
          or (step + 1) % log_freq == 0
          or step == (steps - 1)
      )
      if should_log_step:
        with jax_profiler.TraceAnnotation("D2H", step_num=step):
          loss_cpu = loss.cpu().item()
        should_log = True
      else:
        if materialize_forward_before_backward:
          import torch_tpu._internal.sync

          torch_tpu._internal.sync.synchronize(loss, wait=False)
        loss_cpu = float("nan")
        should_log = False

      if profiler:
        profiler.step()

      step_end = time.perf_counter()
      step_time = step_end - step_start

      accumulated_tokens += step_tokens
      accumulated_time += step_time
      accumulated_steps += 1

      if step >= warmup_steps:
        total_tokens += step_tokens
        total_time += step_time

      if should_log:
        avg_step_time = accumulated_time / accumulated_steps
        accumulated_tokens = 0
        accumulated_time = 0.0
        accumulated_steps = 0

        extra_metrics = {
            "lr": optimizer.optimizers[0].param_groups[0]["lr"],
            "n_tokens_seen": ntokens_seen,
            "avg_step_time": avg_step_time,
        }
        metrics_processor.log(
            step + 1,
            loss_cpu / step_tokens,
            loss_cpu / step_tokens,
            float("nan"),
            extra_metrics=extra_metrics,
        )

  avg_tps = total_tokens / total_time if total_time > 0 else 0.0
  avg_tflops = num_flops_per_token * avg_tps / 1e12
  avg_mfu = 100 * num_flops_per_token * avg_tps / peak_flops
  if rank == 0:
    logger.info(
        "Average TPS (i.e., Frames Per Second in this case, excl. %d warmup"
        " steps): %.0f  avg TFlops: %.2f  avg MFU: %.2f%%",
        warmup_steps,
        avg_tps,
        avg_tflops,
        avg_mfu,
    )

  if torch.distributed.is_initialized():
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
  logger.info("Process group destroyed")


if __name__ == "__main__":
  gmain.handle_main(start_trainer)
