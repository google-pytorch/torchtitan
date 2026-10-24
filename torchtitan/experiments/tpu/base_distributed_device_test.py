"""Base class and helper functions for distributed accelerator tests."""

import enum
import importlib
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, Tuple, Type

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.experiments.tpu import gdist
import torchtitan.config
import torchtitan.trainer
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import device_env
from torchtitan.experiments.tpu import distributed_utils
from torchtitan.experiments.tpu import numerical_validation
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import train_minimal
from torchtitan.experiments.tpu.cp_attention import revert_tpu_context_parallel
import torchtitan.experiments.tpu.tpu_job_config as tpu_job_config_module
from torchtitan.tools import utils


ParallelismFunc = Callable[[nn.Module], nn.Module | None]


class InputDistribution(enum.Enum):
  """Input distribution for distributed tests."""

  REPLICATE = "replicate"  # All ranks see all data (e.g. TP)
  SPLIT_BATCH = (  # Rank i only sees batch[i::world_size] (e.g. FSDP)
      "split_batch"
  )


# Gathering helper functions (shared by unit test runner and training state
# validator)
def _gather_distributed_tensor(
    tensor: torch.Tensor | DTensor,
) -> torch.Tensor:
  """Unshards a tensor (DTensor) purely on device."""

  # Handle DTensor (automatic stitching)
  if isinstance(tensor, DTensor):
    return tensor.full_tensor().detach()

  # Should be one of the above, otherwise its a standard tensor and
  # replicated shard so we can just take local copy.
  return tensor.detach()


class DistributedUnitTestRunner:
  """Runs distributed parity tests between a Single-Device CPU model and a Parallel Device model.

  Works with DTensor-based parallelism set ups

  This class manages:
  - Creating single-device CPU model as a reference.
  - Creating parallelized device model to test against reference.
  - Ensuring test/ref models start with identical weights.
  - Applying arbitrary parallelization transforms.
  - Running Forward/Backward/Step checks.
    - For forward checks, we run the reference model on Rank 0 and the
      parallel model on all ranks and compare outputs.
    - For backward/step checks, Rank 0 pre-computes the reference model's
      loss and gradients for all steps, and then all ranks synchronize and
      compare against the cached values. This is to avoid potential
      deadlocks from running the reference model in lockstep with the
      parallel model.
  """

  def __init__(
      self,
      device: torch.device,
      rank: int,
      world_size: int,
      model_class: Type[nn.Module],
      model_args: Any,
      parallelism_func: ParallelismFunc,
      input_distribution: InputDistribution = InputDistribution.REPLICATE,
      use_meta_init: bool = True,
  ):
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.model_class = model_class
    self.model_args = model_args
    self.input_distribution = input_distribution
    self.use_meta_init = use_meta_init

    # Apply parallelism to the model.
    self.parallel_model = None
    self._create_parallel_model(parallelism_func)

    # Create CPU reference model.
    self.reference_model = None
    self._create_reference_from_parallel()

    # Track current step for parity checks.
    self.current_step = 0

  def run_forward_parity(
      self,
      batch_size: int,
      seq_len: int,
      atol: float = 1e-3,
      rtol: float = 1e-3,
  ):
    """Verifies that reference and parallel model produce equivalent outputs on forward pass."""
    if self.reference_model is None:
      raise ValueError(
          "Reference model not initialized. To run test, call "
          "`apply_parallelism`to parallelize the device model "
          "and create a reference on CPU."
      )

    global_input, global_target = self._generate_global_batch(
        batch_size, seq_len
    )
    local_input, _ = self._get_local_batch_view(global_input, global_target)

    reference_out = None

    # Run reference model (Rank 0 only)
    if self.rank == 0:
      with torch.no_grad():
        reference_out, _ = self._run_step(
            self.reference_model, global_input, backward=False
        )

    with torch.no_grad():
      parallel_out, _ = self._run_step(
          self.parallel_model, local_input.to(self.device), backward=False
      )

    # Helper to move tensor/dtensor to CPU
    parallel_out_cpu = self._gather_output_to_cpu(parallel_out)

    if self.rank == 0:
      test_utils.check_equivalence(
          tensor_test=parallel_out_cpu,
          tensor_reference=reference_out,
          check_name=f"Forward Pass (Rank {self.rank})",
          step=self.current_step,
          atol=atol,
          rtol=rtol,
          test_label="Parallel",
          ref_label="Single Device",
      )
      logging.info("[Rank %s] Forward parity passed.", self.rank)

  def _run_reference_model_pre_run(self, batches, num_steps):
    """Executes the reference model steps on CPU and caches artifacts."""
    reference_records = []
    logging.info(
        "[Rank 0]: Executing %d step Reference Model Pre-run...", num_steps
    )
    for step in range(num_steps):
      inputs_global, targets_global = batches[step]
      _, reference_loss = self._run_step(
          self.reference_model, inputs_global, targets_global, backward=True
      )

      ref_grads = {}
      for name, param in self.reference_model.named_parameters():
        if param.grad is not None:
          ref_grads[name] = param.grad.clone()

      reference_records.append({
          "loss": (
              reference_loss.detach().clone()
              if reference_loss is not None
              else None
          ),
          "grads": ref_grads,
      })
    logging.info("[Rank 0]: Reference Model sequence compiled natively.")
    return reference_records

  def run_backward_parity(
      self,
      num_steps: int = 1,
      batch_size: int = 1,
      seq_len: int = 1,
      atol_loss: float = 1e-3,
      rtol_loss: float = 1e-3,
      atol_grad: float = 1e-3,
      rtol_grad: float = 1e-3,
  ):
    """Verifies equivalence of loss and gradients across models."""
    if self.reference_model is None:
      raise ValueError(
          "Reference model not initialized. To run test, call "
          "`apply_parallelism`to parallelize the device model "
          "and create a reference on CPU."
      )

    logging.info(
        "[Rank %d]: Generating all %d shared batches upfront.",
        self.rank,
        num_steps,
    )
    start_step = self.current_step
    batches = []
    for step in range(num_steps):
      self.current_step = start_step + step
      batches.append(self._generate_global_batch(batch_size, seq_len))

    # CPU Pre-Run: Rank 0 completely pre-computes the reference evaluation
    reference_records = []
    if self.rank == 0:
      reference_records = self._run_reference_model_pre_run(batches, num_steps)

    # Universal barrier to ensure Rank 0 has cached everything correctly
    dist.barrier()

    for step in range(num_steps):
      self.current_step = start_step + step
      inputs_global, targets_global = batches[step]
      inputs_local, targets_local = self._get_local_batch_view(
          inputs_global, targets_global
      )
      inputs_device = inputs_local.to(self.device)
      targets_device = targets_local.to(self.device)

      _, parallel_loss = self._run_step(
          self.parallel_model, inputs_device, targets_device, backward=True
      )

      # If we split the batch, the local loss is just a slice. We must
      # average across ranks to match the global reference loss.
      if self.input_distribution == InputDistribution.SPLIT_BATCH:
        loss_to_check = parallel_loss.detach().clone()
        dist.all_reduce(loss_to_check, op=dist.ReduceOp.AVG)
      else:
        loss_to_check = parallel_loss

      # Force ALL ranks to extract `.cpu()` identically here.
      loss_to_check_cpu = loss_to_check.cpu()

      if self.rank == 0:
        test_utils.check_equivalence(
            tensor_test=loss_to_check_cpu,
            tensor_reference=reference_records[step]["loss"],
            check_name=f"Backward Loss (Rank {self.rank})",
            step=step,
            atol=atol_loss,
            rtol=rtol_loss,
            test_label="Parallel",
            ref_label="Single Device",
        )

      self._check_gradients_against_recorded(
          reference_records[step]["grads"] if self.rank == 0 else {},
          atol_grad,
          rtol_grad,
      )

      dist.barrier()

    logging.info("[Rank %s] Backward parity passed.", self.rank)

    self.current_step = start_step + num_steps

  def _create_parallel_model(self, func: Callable[[nn.Module], None]):
    """Creates a parallel model and applies the parallelism function."""
    if self.use_meta_init:
      with torch.device("meta"):
        self.parallel_model = self.model_class(self.model_args)
    else:
      self.parallel_model = self.model_class(self.model_args).to(self.device)

    # Capture return value if parallelism_func does not make change in place.
    modified_model = func(self.parallel_model)
    if modified_model is not None:
      self.parallel_model = modified_model

    # Materialize weights if they are still on Meta device
    if any(p.device.type == "meta" for p in self.parallel_model.parameters()):
      self.parallel_model.to_empty(device=self.device)
      with torch.device(self.device):
        self.parallel_model.init_weights()

  def _create_reference_from_parallel(self):
    """Creates a CPU reference model and loads weights from the parallel model."""
    self.reference_model = self.model_class(self.model_args).to("cpu")

    parallel_state = self.parallel_model.state_dict()
    cpu_state_dict = {}

    with torch.no_grad():
      for key, param in parallel_state.items():
        cpu_state_dict[key] = self._gather_weights_wrapper(param).cpu()

    self.reference_model.load_state_dict(cpu_state_dict)
    # Ensure gradients are enabled on reference model params.
    for p in self.reference_model.parameters():
      p.requires_grad_(True)

  def _generate_global_batch(
      self, batch_size: int, seq_len: int
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates the full global batch on CPU (deterministically)."""
    torch.manual_seed(1 + self.current_step)

    global_input = torch.randint(
        0, self.model_args.vocab_size, (batch_size, seq_len), device="cpu"
    )
    global_target = torch.randn(
        batch_size, seq_len, self.model_args.vocab_size, device="cpu"
    )
    return global_input, global_target

  def _get_local_batch_view(
      self, global_input: torch.Tensor, global_target: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slices or replicates the global batch for the current rank."""

    if self.input_distribution == InputDistribution.REPLICATE:
      # TP: Everyone sees the full global batch
      return global_input, global_target

    elif self.input_distribution == InputDistribution.SPLIT_BATCH:
      # FSDP: Slice the global batch for this rank
      batch_size = global_input.size(0)
      assert batch_size % self.world_size == 0

      local_bs = batch_size // self.world_size
      start = self.rank * local_bs
      end = start + local_bs

      return global_input[start:end], global_target[start:end]

    raise ValueError(f"Unknown distribution: {self.input_distribution}")

  def _run_step(
      self,
      model: nn.Module,
      inputs: torch.Tensor,
      targets: torch.Tensor = None,
      backward: bool = False,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs a forward pass and optionally a backward pass."""
    if backward:
      model.zero_grad()

    import contextlib
    from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len
    from torchtitan.models.common.decoder import Decoder

    loss = None
    if isinstance(model, Decoder) and inputs.ndim == 2:
      bs, seq_len = inputs.shape
      positions = torch.arange(
          seq_len, dtype=torch.long, device=inputs.device
      ).repeat(bs)
      ctx = set_unflatten_seq_len(seq_len)
      call_fn = lambda: model(inputs.reshape(-1), positions).reshape(
          bs, seq_len, -1
      )
    else:
      ctx = contextlib.nullcontext()
      call_fn = lambda: model(inputs)

    with ctx:
      output = call_fn()
      if backward and targets is not None:
        if targets.dtype != output.dtype:
          targets = targets.to(output.dtype)
        loss = nn.MSELoss()(output, targets)
        loss.backward()

    return output, loss

  def _check_gradients_against_recorded(
      self, reference_grads_recorded, atol, rtol
  ):
    """Standard DTensor gradient check."""
    gathered_gradients = {}

    # Pass 1: Enqueue all collectives asynchronously without calling `.cpu()` yet.
    for name, par_param in self.parallel_model.named_parameters():
      if par_param.grad is None:
        continue

      # Gather unshard natively on device without implicit `.cpu()`
      gathered_gradients[name] = self._gather_weights_wrapper(par_param.grad)

    # Force synchronization by evaluating a scalar dependent on all gathered gradients.
    if gathered_gradients:
      dummy_sync = sum(t.view(-1)[0] for t in gathered_gradients.values())
      _ = dummy_sync.cpu()

    dist.barrier()

    # Pass 2: Rank 0 fetches the fully evaluated tensors to CPU for checking.
    if self.rank == 0:
      for name, par_tensor in gathered_gradients.items():
        if name in reference_grads_recorded:
          full_grad = par_tensor.cpu()
          test_utils.check_equivalence(
              tensor_test=full_grad,
              tensor_reference=reference_grads_recorded[name],
              atol=atol,
              rtol=rtol,
              check_name=f"Gradient {name}",
              step=self.current_step,
          )

  def _gather_output_to_cpu(
      self,
      output: torch.Tensor | DTensor,
  ) -> torch.Tensor:
    """Standardizes gathering outputs from DTensor or Tensor to CPU."""
    if isinstance(output, DTensor):
      # DTensor knows how to unshard itself
      return output.full_tensor().cpu()

    # If it's local tensor, behavior depends on the test
    # If TP output is sharded on sequence dim (SP) but is not DTensor,
    # we might need manual gathering logic here.
    return output.cpu()

  def _gather_weights_wrapper(
      self, tensor: torch.Tensor | DTensor
  ) -> torch.Tensor:
    """Wraps module-level gather function with instance config."""
    return _gather_distributed_tensor(tensor)


def _capture_distributed_state(
    model: nn.Module,
    loss: torch.Tensor,
    world_size: int,
    input_distribution: InputDistribution = InputDistribution.REPLICATE,
) -> Dict[str, Any]:
  """Captures a static snapshot of the distributed model's parameters, gradients, and loss."""

  # Unwrap DTensor if present (Keep on device)
  if isinstance(loss, DTensor):
    loss_local = loss.full_tensor()
  else:
    loss_local = loss

  # Detach and clone (on device)
  loss_on_device = loss_local.detach().clone()

  # Perform reduction if needed (on device)
  if input_distribution == InputDistribution.SPLIT_BATCH and world_size > 1:
    # Sum across ranks (loss is already globally scaled by tokens)
    dist.all_reduce(loss_on_device, op=dist.ReduceOp.SUM)

  # Move result
  loss_cpu = loss_on_device.cpu()

  state = {
      "loss": loss_cpu,
      "params": {},
      "grads": {},
  }

  for name, param in model.named_parameters():
    gathered_param = _gather_distributed_tensor(param).cpu()
    state["params"][name] = gathered_param

    if param.grad is not None:
      gathered_grad = _gather_distributed_tensor(param.grad).cpu()
      state["grads"][name] = gathered_grad

  return state


def config_to_input_distribution(
    config: torchtitan.trainer.Trainer.Config,
) -> InputDistribution:
  """Determines the input distribution based on the config.

  Args:
    config: The job configuration.

  Returns:
    The determined InputDistribution based on the data/context parallel degrees.
  """
  dp_degree = config.parallelism.data_parallel_shard_degree
  rep_degree = config.parallelism.data_parallel_replicate_degree
  cp_degree = config.parallelism.context_parallel_degree
  if dp_degree > 1 or rep_degree > 1 or cp_degree > 1:
    return InputDistribution.SPLIT_BATCH
  else:
    return InputDistribution.REPLICATE


def _apply_cpu_splash_reference_patch(model: nn.Module) -> None:
  """Makes CPU ScaledDotProductInnerAttention apply segment_ids/window masks like Splash."""
  from torchtitan.models.common.attention import (  # pylint: disable=g-import-not-at-top
      ScaledDotProductInnerAttention,
      segment_ids_from_positions,
  )

  for module in model.modules():
    if not isinstance(module, ScaledDotProductInnerAttention):
      continue
    window = getattr(module, "sliding_window_size", None)

    def _splash_ref_forward(
        q,
        k,
        v,
        *,
        scale=None,
        enable_gqa=False,
        is_causal=True,
        segment_ids=None,
        positions=None,
        _orig=module.forward,
        _window=window,
        **kwargs,
    ):
      if segment_ids is None and positions is not None:
        segment_ids = segment_ids_from_positions(positions)
      if segment_ids is None and _window is None:
        return _orig(
            q,
            k,
            v,
            scale=scale,
            enable_gqa=enable_gqa,
            is_causal=is_causal,
            **kwargs,
        )
      was_3d = q.ndim == 3
      if was_3d:
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
      bsz, seq_len = q.shape[0], q.shape[1]
      pos = torch.arange(seq_len, device=q.device)
      mask = torch.ones(
          (bsz, 1, seq_len, seq_len), dtype=torch.bool, device=q.device
      )
      if is_causal:
        mask = mask & (pos[None, None, :, None] >= pos[None, None, None, :])
      if _window is not None:
        mask = mask & (
            pos[None, None, :, None] - pos[None, None, None, :] <= _window
        )
      if segment_ids is not None:
        seg = (
            segment_ids.unsqueeze(0)
            if segment_ids.ndim == 1
            else segment_ids.reshape(bsz, seq_len)
        )
        mask = mask & (seg[:, None, :, None] == seg[:, None, None, :])
      q_bh = q.transpose(1, 2)
      k_bh = k.transpose(1, 2)
      v_bh = v.transpose(1, 2)
      out = torch.nn.functional.scaled_dot_product_attention(
          q_bh,
          k_bh,
          v_bh,
          attn_mask=mask,
          is_causal=False,
          scale=scale,
          enable_gqa=enable_gqa,
      ).transpose(1, 2)
      return out.squeeze(0) if was_3d else out

    module.forward = _splash_ref_forward


def _run_cpu_verification(
    temp_path: str,
    config_args: list[str],
    loss_atol: float,
    loss_rtol: float,
    grad_atol: float,
    grad_rtol: float,
    param_atol: float,
    param_rtol: float,
    cpu_config_fn: str | None = None,
):
  """Runs the CPU verification in a separate process.

  ``cpu_config_fn`` ("module:function") is applied to the parsed CPU config
  after the generic single-device overrides, for lanes whose accelerator-only
  choices are not CLI flags (e.g. the HF backend's attention implementation
  and fused loss live on the model spec).
  """
  logging.info("[Child Process] Starting CPU Training (Verifying)...")

  # The device is CPU because run_in_subprocess launched this process with a
  # CPU-only environment (see device_env), so detection reports CPU here.

  # Load recorded data from the file
  if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
    raise RuntimeError(f"Recorded history file missing or empty: {temp_path}")

  recorded_history = torch.load(temp_path)
  logging.info(
      "Retrieved history for %s steps (including init).",
      len(recorded_history),
  )

  # Extract initialization state
  if "init" not in recorded_history:
    raise RuntimeError("Recorded history missing 'init' state.")
  init_state = recorded_history.pop("init")

  # Prepare CPU Configuration
  config_manager = torchtitan.config.ConfigManager()
  cpu_config = config_manager.parse_args(config_args)

  # Force overrides for single-device CPU execution
  cpu_config.parallelism.tensor_parallel_degree = 1
  cpu_config.parallelism.data_parallel_shard_degree = 1
  cpu_config.parallelism.data_parallel_replicate_degree = 1
  cpu_config.parallelism.context_parallel_degree = 1
  cpu_config.compile.enable = False
  if (
      isinstance(cpu_config, tpu_job_config_module.BaseTPUTrainerConfig)
      and cpu_config.model_spec is not None
  ):
    # A CP recipe's attention is a CP inner attention; the single-device
    # reference needs the plain one.
    revert_tpu_context_parallel(cpu_config.model_spec.model)

  # Disable splash attention patch on CPU to avoid ValueError, while recording
  # whether the device run used Splash/Tokamax so the CPU reference applies the
  # same causal + segment_ids (+ sliding window) mask.
  had_splash_kernel = False
  if hasattr(cpu_config, "splash_attention_kernel"):
    had_splash_kernel = bool(
        cpu_config.splash_attention_kernel.use_splash_attention_kernel
        or cpu_config.splash_attention_kernel.use_tokamax_splash_attention_kernel
    )
    cpu_config.splash_attention_kernel.use_splash_attention_kernel = False
    cpu_config.splash_attention_kernel.use_tokamax_splash_attention_kernel = False

  if cpu_config_fn is not None:
    module_name, function_name = cpu_config_fn.split(":")
    cpu_config = getattr(importlib.import_module(module_name), function_name)(
        cpu_config
    )

  # Setup validator callback
  validator = numerical_validation.StateValidatorCallback(
      recorded_history=recorded_history,
      capture_fn=numerical_validation.capture_local_state,
      init_state=init_state,
      loss_atol=loss_atol,
      loss_rtol=loss_rtol,
      grad_atol=grad_atol,
      grad_rtol=grad_rtol,
      param_atol=param_atol,
      param_rtol=param_rtol,
  )

  with validator:
    # Initialize CPU Trainer
    trainer_cpu = train_minimal.TrainerMinimal(
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
        job_config=cpu_config,
        step_callback=validator.on_step,
    )
    if had_splash_kernel:
      _apply_cpu_splash_reference_patch(trainer_cpu.model)

    # Load the captured distributed weights into the CPU model
    current_cpu_dict = dict(trainer_cpu.model.named_parameters())

    for name, param in init_state["params"].items():
      # Strip wrapper prefixes (e.g. checkpointing) to match CPU reference model
      cleaned_name = numerical_validation.clean_parameter_name(name)
      if cleaned_name in current_cpu_dict:
        cpu_param = current_cpu_dict[cleaned_name]
        with torch.no_grad():
          cpu_param.copy_(param)
        after_diff = torch.max(torch.abs(cpu_param - param))
        if after_diff > 0.0:
          logging.error(
              "INIT COPY FAILED for %s! Remaining Diff: %e",
              cleaned_name,
              after_diff,
          )
      else:
        logging.warning(
            "Param %s (cleaned: %s) from distributed run not found in CPU"
            " model.",
            name,
            cleaned_name,
        )

    logging.info("CPU model initialized with distributed weights.")

    # Build mock dataloader from recorded inputs/labels to ensure exact data parity
    mock_dataloader = []
    for step in sorted(recorded_history.keys()):
      if "inputs" in recorded_history[step] and "labels" in recorded_history[step]:
        input_dict = {"input": recorded_history[step]["inputs"]}
        if "positions" in recorded_history[step]:
          input_dict["positions"] = recorded_history[step]["positions"]
        mock_dataloader.append((input_dict, recorded_history[step]["labels"]))
      else:
        raise RuntimeError(f"Step {step} missing recorded inputs/labels!")

    trainer_cpu.dataloader = mock_dataloader

    trainer_cpu.train()

  logging.info("[Child Process] Parity Test Passed Successfully.")

# Chose for wrapper to be class so that output_path can written to/read from
# outside the function.
# (alternative is to use functools.partial)
class _DistributedInitWrapper:
  """Bootstraps the TPU distributed environment, then runs the real entry point.

  gdist.torchrun() invokes the trainer directly, bypassing gmain, which is what
  performs this bootstrap in production. It has to run inside the worker:
  maybe_init_distributed() derives per-worker values (TPU_VISIBLE_CHIPS,
  CLOUD_TPU_TASK_ID) from LOCAL_RANK / WORLD_SIZE, which only torchrun-spawned
  workers have, so the spawning process cannot do it on their behalf.

  A class rather than a closure because elastic_launch pickles the entry point,
  and it carries __name__ because gdist.torchrun uses it for the launch role.
  """

  def __init__(self, fn):
    self.fn = fn
    self.__name__ = getattr(fn, "__name__", "start_trainer")
    self.__qualname__ = self.__name__

  def __call__(self, *args, **kwargs):
    distributed_utils.maybe_init_distributed()
    return self.fn(*args, **kwargs)


class DistributedTrainWithRecorder:
  """Runs a distributed training job while recording model state history to a temporary file."""

  def __init__(self, output_path: str):
    self.output_path = output_path
    self.__name__ = "DistributedTrainWithRecorder"
    self.__qualname__ = "DistributedTrainWithRecorder"

  def __call__(self, config):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = utils.get_local_device()

    dist_type = config_to_input_distribution(config)

    # Define how to capture state in current distributed context
    def dist_capture_fn(model, loss):
      return _capture_distributed_state(model, loss, world_size, dist_type)

    # Setup recorder callback
    recorder = numerical_validation.StateRecorderCallback(
        capture_fn=dist_capture_fn
    )

    trainer = train_minimal.TrainerMinimal(
        device=device,
        rank=rank,
        world_size=world_size,
        job_config=config,
        step_callback=recorder.on_step,
    )

    # Init State Capture (Manual call to prevent deadlock)
    init_loss = torch.tensor(0.0, device=device)
    init_state = dist_capture_fn(trainer.model, init_loss)

    if rank == 0:
      recorder.history["init"] = init_state

    trainer.train()

    if rank == 0:
      torch.save(recorder.get_history(), self.output_path)

    # Final cleanup inside the spawned process
    if dist.is_initialized():
      dist.destroy_process_group()


class BaseDistributedDeviceTest(parameterized.TestCase):
  """Base class for tests that need to run on different distributed devices.

  This class sets up `self.num_devices` based on the `--num_devices` flag and
  `self.accelerator_device_type` based on the `--device` flag.
  """

  def setUp(self):
    super().setUp()
    torch.compiler.reset()
    self.num_devices = test_utils._NUM_DEVICES.value
    self.accelerator_device_type = test_utils.resolve_device_type()

    if not test_utils.flags.FLAGS["num_devices"].present:
      if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
        from torch_tpu._internal.utils import hardware  # pylint: disable=g-import-not-at-top
        logging.info(
            "--num_devices flag not set, attempting to detect number of TPU"
            " chips..."
        )
        self.num_devices = hardware.get_tpu_device_count()
        logging.info("Found %d TPU chips.", self.num_devices)

    # Ensure Hugging Face datasets use a directory with enough storage
    # (e.g. /tmp). The default cache in `~/.cache` is too small in some
    # environments.
    os.environ["HF_DATASETS_CACHE"] = os.path.join(
        tempfile.gettempdir(), "huggingface_cache"
    )

  def test_gpu_available(self):
    if self.accelerator_device_type == "cuda":
      if not torch.cuda.is_available():
        self.fail("GPU requested but CUDA not available.")

  def _test_train_distributed(
      self,
      config_args: list[str],
      data_parallel_shard_degree: int | None = None,
      tensor_parallel_degree: int | None = None,
      skip_devices: list[device_type.AcceleratorDeviceType] | None = None,
      start_trainer: Callable[
          [
              torchtitan.trainer.Trainer.Config
              | tpu_job_config_module.TPUTrainerConfig,
          ],
          Any,
      ] = None,
      enable_compile: bool = False,
      data_parallel_replicate_degree: int | None = None,
      context_parallel_degree: int | None = None,
  ):
    """Tests the minimal distributed training setup for torchtitan models."""
    if skip_devices and self.accelerator_device_type in skip_devices:
      self.skipTest(
          f"Skipping test for device type: {self.accelerator_device_type}"
      )
      return

    global_flags = []
    if enable_compile:
      if self.accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
        self.skipTest("Skipping CPU test for torch.compile: b/327271919")
        return
      global_flags.append("--compile.enable")
      if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
        global_flags.append("--compile.backend=tpu")

    if tensor_parallel_degree is not None:
      if tensor_parallel_degree == -1:
        tensor_parallel_degree = self.num_devices
      global_flags.append(
          f"--parallelism.tensor_parallel_degree={tensor_parallel_degree}"
      )
    if data_parallel_shard_degree is not None:
      if data_parallel_shard_degree == -1:
        data_parallel_shard_degree = self.num_devices
      global_flags.append(
          f"--parallelism.data_parallel_shard_degree={data_parallel_shard_degree}"
      )
    if data_parallel_replicate_degree is not None:
      if data_parallel_replicate_degree == -1:
        data_parallel_replicate_degree = self.num_devices
      global_flags.append(
          f"--parallelism.data_parallel_replicate_degree={data_parallel_replicate_degree}"
      )
    if context_parallel_degree is not None:
      if context_parallel_degree == -1:
        context_parallel_degree = self.num_devices
      global_flags.append(
          f"--parallelism.context_parallel_degree={context_parallel_degree}"
      )

    # Prepend flags collected in global_flags to config_args so they are
    # not swallowed by positional subcommands introduced in the tyro upgrade.
    # Namely, activation-checkpoint:none. This means global_flags should only
    # contain global flags and not subcommands.
    config_args[:] = global_flags + config_args

    config_manager = torchtitan.config.ConfigManager()

    config = config_manager.parse_args(config_args)
    config.dump_folder = tempfile.gettempdir()

    if self.accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
      # Workers are spawned and re-detect their device, so hide the accelerator
      # in the environment they inherit rather than setting a global here.
      os.environ.update(device_env.cpu_only_env())
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU:
      # Construct the environment before spawning workers.
      distributed_utils.maybe_init_distributed(self.num_devices)
      start_trainer = _DistributedInitWrapper(start_trainer)

    gdist.torchrun(start_trainer, nproc_per_node=self.num_devices)(config)

  def _run_trainer_distributed_parity_test(
      self,
      config_args: list[str],
      tensor_parallel_degree: int | None = None,
      data_parallel_shard_degree: int | None = None,
      skip_devices: list[device_type.AcceleratorDeviceType] | None = None,
      loss_atol: float = 1e-3,
      loss_rtol: float = 1e-3,
      grad_atol: float = 5e-3,  # grads can be noisier in distributed setting
      grad_rtol: float = 5e-3,
      param_atol: float = 5e-3,
      param_rtol: float = 5e-3,
      enable_compile: bool = False,
      data_parallel_replicate_degree: int | None = None,
      context_parallel_degree: int | None = None,
      cpu_config_fn: str | None = None,
  ):
    """Orchestrates a distributed parity test: Record Distributed -> Verify on CPU."""
    if skip_devices and self.accelerator_device_type in skip_devices:
      self.skipTest(
          f"Skipping test for device type: {self.accelerator_device_type}"
      )
      return

    # Setup temporary file for data transfer.
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
      temp_path = tmp_file.name

    try:
      logging.info(
          "[Phase 1] Running Distributed Training on device (Recording)..."
      )

      # Use wrapper class to run the trainer and record the state to the file.
      self._test_train_distributed(
          config_args=config_args,
          data_parallel_shard_degree=data_parallel_shard_degree,
          tensor_parallel_degree=tensor_parallel_degree,
          skip_devices=skip_devices,
          start_trainer=DistributedTrainWithRecorder(output_path=temp_path),
          enable_compile=enable_compile,
          data_parallel_replicate_degree=data_parallel_replicate_degree,
          context_parallel_degree=context_parallel_degree,
      )

      # Verify the distributed run produced data
      if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
        self.fail(
            "Distributed run finished but returned no data (File"
            " empty/missing)."
        )

      # Setup CPU Verification Run
      logging.info("[Phase 2] Running CPU Training (Verifying)...")

      try:
        test_utils.run_in_subprocess(
            module_name="torchtitan.experiments.tpu.base_distributed_device_test",
            function_name="_run_cpu_verification",
            env=device_env.cpu_only_env(),
            args=[
                temp_path,
                config_args,
                loss_atol,
                loss_rtol,
                grad_atol,
                grad_rtol,
                param_atol,
                param_rtol,
            ],
            kwargs={"cpu_config_fn": cpu_config_fn},
        )
      except subprocess.CalledProcessError as e:
        self.fail(f"Isolated subprocess failed with exit code {e.returncode}.\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
      except Exception as e:
        self.fail(f"Isolated subprocess failed: {e}")

    finally:
      # Cleanup temporary file
      if os.path.exists(temp_path):
        os.remove(temp_path)
