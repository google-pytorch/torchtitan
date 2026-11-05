"""Tests for the minimal distributed training setup for MLP models."""

from typing import Callable, Optional

from absl import logging
from absl.testing import absltest
from torch.distributed.tensor import DeviceMesh
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
import torch.distributed.tensor
from torch.distributed.tensor import parallel
from torch.distributed.tensor import DTensor, Replicate, Shard
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchtitan.experiments.tpu.distributed import accelerator_device_type as device_type
from torchtitan.experiments.tpu.testing import base_distributed_device_test
from torchtitan.experiments.tpu.testing import test_utils
from torchtitan.experiments.tpu.distributed import distributed
import torch.compiler



BATCH_SIZE = 2
SEQ_LEN = 128
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = HIDDEN_SIZE * 4
NUM_TRAINING_STEPS = 20
NUM_HEADS = 16


class MLP(nn.Module):
  """Standard MLP module (single-device version)."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: Optional[int] = None,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size or 4 * hidden_size

    self.dense_h_to_4h = nn.Linear(
        hidden_size, self.intermediate_size, bias=True
    )
    self.dense_4h_to_h = nn.Linear(
        self.intermediate_size, hidden_size, bias=True
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through MLP."""
    intermediate = self.dense_h_to_4h(x)
    intermediate = F.silu(intermediate)
    output = self.dense_4h_to_h(intermediate)
    return output


class Attention(nn.Module):
  """Multi-head attention with proper tensor parallelism design.

  Key design choices for tensor parallelism:
  1. Separate q, k, v projections (each ColwiseParallel) - shards heads across
  devices
  2. Attention computation stays local on each device
  3. Output projection (RowwiseParallel) - gathers head results with all-reduce
  """

  def __init__(
      self,
      hidden_size: int,
      num_heads: int = 12,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    assert hidden_size % num_heads == 0
    self.head_dim = hidden_size // num_heads

    # Separate projections for Q, K, V
    # Each projects hidden -> hidden (for all heads combined)
    self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
    self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
    self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    # Output projection
    self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with proper sharding semantics..."""
    batch_size, seq_len, hidden_size = x.shape

    # Project to Q, K, V
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    # Use -1 instead of `n_heads` to infer the actual
    # local heads from sizes of xq, xk, and xv as TP may have sharded them
    # after the above linear ops.

    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

    # Compute attention
    scale = self.head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)

    # Reshape back to (batch_size, seq_len, hidden_size)
    attn_out = (
        attn_out.transpose(1, 2)
        .contiguous()
        .view(batch_size, seq_len, -1)
    )

    # Output projection with RowwiseParallel
    output = self.out_proj(attn_out)

    return output


class TransformerBlock(nn.Module):
  """Transformer block with attention and MLP, designed for tensor parallelism."""

  def __init__(
      self,
      hidden_size: int,
      intermediate_size: Optional[int] = None,
      num_heads: int = 12,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads

    # Attention block
    self.ln_1 = nn.LayerNorm(hidden_size)
    self.attention = Attention(hidden_size, num_heads)

    # MLP block
    self.ln_2 = nn.LayerNorm(hidden_size)
    self.mlp = MLP(hidden_size, intermediate_size)

  def init_weights(self):
    """Initialize weights for the module."""
    def _init_weights_layer_logic(module):
      if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

    self.apply(_init_weights_layer_logic)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass using pre-norm configuration."""
    # Attention with residual
    residual = x
    x = self.ln_1(x)
    x = self.attention(x)
    x = residual + x

    # MLP with residual
    residual = x
    x = self.ln_2(x)
    x = self.mlp(x)
    x = residual + x

    return x


def parallelize_transformer_block_tp(
    model: TransformerBlock,
    device_mesh: dist.DeviceMesh,
) -> TransformerBlock:
  """Parallelize TransformerBlock using tensor parallelism patterns.

  Sharding strategy:
  - Q, K, V projections: ColwiseParallel (output features sharded)
  - Output projection: RowwiseParallel (input features sharded, all-reduce on
  output)
  - MLP: ColwiseParallel -> RowwiseParallel (Megatron pattern)
  - LayerNorm: Default (replicated)

  Args:
      model: The TransformerBlock module to parallelize.
      device_mesh: The DeviceMesh for tensor parallelism.
  """
  plan = {
      # Attention projections - each ColwiseParallel shards the output heads
      "attention.q_proj": parallel.ColwiseParallel(),
      "attention.k_proj": parallel.ColwiseParallel(),
      "attention.v_proj": parallel.ColwiseParallel(),
      # Output projection - RowwiseParallel with automatic all-reduce
      "attention.out_proj": parallel.RowwiseParallel(),
      # MLP layers
      "mlp.dense_h_to_4h": parallel.ColwiseParallel(),
      "mlp.dense_4h_to_h": parallel.RowwiseParallel(),
  }

  parallelized_model = parallel.parallelize_module(
      model,
      device_mesh,
      parallelize_plan=plan,  # pyrefly: ignore[bad-argument-type]
  )

  return parallelized_model  # pyrefly: ignore[bad-return]


def parallelize_transformer_block_tp_sp(
    model: TransformerBlock,
    device_mesh: dist.DeviceMesh,
) -> TransformerBlock:
  """Parallelize TransformerBlock using proper TP+SP.

  Sharding strategy:
  - Q, K, V projections: ColwiseParallel (output features sharded)
  - Output projection: RowwiseParallel (input features sharded, all-reduce on
  output)
  - MLP: ColwiseParallel -> RowwiseParallel (Megatron pattern)
  - LayerNorm: Default (replicated)

  Args:
      model: The TransformerBlock module to parallelize.
      device_mesh: The DeviceMesh for tensor and sequence parallelism.

  Returns:
      The parallelized TransformerBlock module.
  """
  plan = {
      "ln_1": parallel.SequenceParallel(),
      "ln_2": parallel.SequenceParallel(),
      "attention": parallel.PrepareModuleInput(
          input_layouts=(torch.distributed.tensor.Shard(1),),
          desired_input_layouts=(torch.distributed.tensor.Replicate(),),
      ),
      "attention.q_proj": parallel.ColwiseParallel(),
      "attention.k_proj": parallel.ColwiseParallel(),
      "attention.v_proj": parallel.ColwiseParallel(),
      "attention.out_proj": parallel.RowwiseParallel(
          output_layouts=torch.distributed.tensor.Shard(1),
      ),
      "mlp": parallel.PrepareModuleInput(
          input_layouts=(torch.distributed.tensor.Shard(1),),
          desired_input_layouts=(torch.distributed.tensor.Replicate(),),
      ),
      "mlp.dense_h_to_4h": parallel.ColwiseParallel(),
      "mlp.dense_4h_to_h": parallel.RowwiseParallel(
          output_layouts=torch.distributed.tensor.Shard(1)
      ),
  }

  parallelized_model = parallel.parallelize_module(
      model,
      device_mesh,
      parallelize_plan=plan,  # pyrefly: ignore[bad-argument-type]
  )

  return parallelized_model  # pyrefly: ignore[bad-return]


def parallelize_mlp(
    model: MLP,
    device_mesh: dist.DeviceMesh,
) -> MLP:
  """Parallelize MLP using Megatron-style tensor parallelism.

  Parallelization strategy:
      - dense_h_to_4h: ColwiseParallel (shard output features)
      - dense_4h_to_h: RowwiseParallel (shard input features)

  This creates the same sharding pattern as Megatron-LM:
      x (replicated) -> ColwiseLinear -> activation (sharded)
      -> RowwiseLinear -> output (replicated via all-reduce)

  Arguments:
      model: MLP module to parallelize
      device_mesh: DeviceMesh for tensor parallelism

  Returns:
      Parallelized model with DTensor parameters
  """
  plan = {
      "dense_h_to_4h": parallel.ColwiseParallel(),
      "dense_4h_to_h": parallel.RowwiseParallel(),
  }

  parallelized_model = parallel.parallelize_module(
      model,
      device_mesh,
      parallelize_plan=plan,  # pyrefly: ignore[bad-argument-type]
  )

  return parallelized_model  # pyrefly: ignore[bad-return]


def train_model(
    model: nn.Module,
    num_steps: int = 10,
    batch_size: int = 2,
    seq_len: int = 128,
    hidden_size: int = 128,
    learning_rate: float = 1e-4,
    device: torch.device = torch.device("cpu"),
):
  """Simple training loop for the MLP model.

  Arguments:
      model: Model to train
      num_steps: Number of training steps
      batch_size: Batch size for training
      seq_len: Sequence length
      hidden_size: Hidden dimension
      learning_rate: Learning rate for optimizer
      device: Device to train on

  Returns:
      The trained model.
  """
  # Setup optimizer
  # TODO: b/45811390 - fused=False is required for torch_tpu
  optimizer = torch.optim.AdamW(
      model.parameters(), lr=learning_rate, fused=False
  )

  # Simple loss function (MSE for demonstration)
  criterion = nn.MSELoss()

  logging.info("\nTraining Configuration:")
  logging.info("  Steps: %d", num_steps)
  logging.info("  Batch size: %d", batch_size)
  logging.info("  Sequence length: %d", seq_len)
  logging.info("  Learning rate: %f", learning_rate)
  logging.info("  Device: %s", device)

  logging.info("\n%s", "-" * 60)
  logging.info("Training Progress:")
  logging.info("%s", "-" * 60)

  model.train()

  for step in range(num_steps):
    # Generate random input and target
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    target = torch.randn(batch_size, seq_len, hidden_size, device=device)

    # Forward pass
    output = model(x)

    # Compute loss
    loss = criterion(output, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Print progress
    if (step + 1) % max(1, num_steps // 10) == 0 or step == 0:
      logging.info(
          "  Step %3d/%d: loss = %.6f", step + 1, num_steps, loss.item()
      )

  logging.info("%s", "-" * 60)
  logging.info("✓ Training completed!\n")

  return model


def setup_logger():
  """Sets up the logger for the test."""
  if not dist.is_initialized() or dist.get_rank() == 0:
    print("Setting logger verbosity to INFO")
    logging.set_verbosity(logging.INFO)
  else:
    print("Setting logger verbosity to ERROR")
    logging.set_verbosity(logging.ERROR)


def set_device(device: torch.device, rank: int):
  if device.type == "cuda":
    # Set the current process's default device to its rank index.
    # This is a common pattern to ensure uniqueness.
    torch.cuda.set_device(rank)
    # The 'device' object might still be generic, but the context is now set.
    logging.info("  Rank %d successfully set to CUDA device %d", rank, rank)


def _get_dtensor_hierarchy_string(model: nn.Module, indent: int = 0) -> str:
  """Recursively builds the model's structure string with DTensor details."""

  output_lines = []
  prefix = " " * indent
  sub_prefix = " " * (indent + 2)

  # 1. Module Header
  module_header = model.__class__.__name__
  if isinstance(model, nn.Linear):
    module_header += (
        f"(in_features={model.in_features}, out_features={model.out_features},"
        f" bias={model.bias is not None})"
    )
  elif isinstance(model, nn.LayerNorm):
    module_header += (
        f"(normalized_shape={model.normalized_shape}, eps={model.eps})"
    )

  output_lines.append(f"{prefix}{module_header}(")

  # 2. Inspect Direct Parameters for DTensor Sharding
  for name, param in model.named_parameters(recurse=False):
    if hasattr(param, "to_local"):
      placements_str = ", ".join(str(p) for p in param.placements)
      local_shape = param.to_local().shape
      global_shape = list(param.shape)

      output_lines.append(
          f"{sub_prefix}(DTensor Param '{name}': "
          f"Global={global_shape}, "
          f"Local={list(local_shape)}, "
          f"Placements=[{placements_str}])"
      )

  # 3. Recursively Process Submodules
  for name, submodule in model.named_children():
    submodule_repr = _get_dtensor_hierarchy_string(submodule, indent=indent + 4)
    submodule_lines = submodule_repr.split("\n")
    if submodule_lines:
      # Append the submodule's name and the first line of its repr (stripped of its indent)
      output_lines.append(
          f"{sub_prefix}({name}): {submodule_lines[0].lstrip()}"
      )
      # Append the rest of the submodule's lines (already indented)
      output_lines.extend(submodule_lines[1:])

  # 4. Module Footer
  output_lines.append(f"{prefix})")

  return "\n".join(output_lines)


def _test_parallel_mlp(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Tests parallel MLP training."""
  setup_logger()
  set_device(device, rank)

  logging.info("Megatron MLP with parallelize_module")

  # Configuration
  # Create single-device model first
  logging.info("\nCreating single-device model...")
  model = MLP(hidden_size=HIDDEN_SIZE, intermediate_size=INTERMEDIATE_SIZE).to(
      device
  )

  logging.info("\nModel: %s", model)

  logging.info("\nModel Configuration:")
  logging.info("  Hidden size: %d", HIDDEN_SIZE)
  logging.info("  Intermediate size: %d", INTERMEDIATE_SIZE)

  logging.info("\nOriginal model parameters:")
  for name, param in model.named_parameters():
    logging.info(
        "  %s: %s (%s parameters)",
        name,
        param.shape,
        format(param.numel(), ","),
    )

  total_params = sum(p.numel() for p in model.parameters())
  logging.info("\nTotal parameters: %s", format(total_params, ","))

  # Test single-device forward pass
  logging.info("\nTesting single-device forward pass...")
  x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).to(device)

  output_single = model(x)
  logging.info("  Input shape: %s", x.shape)
  logging.info("  Output shape: %s", output_single.shape)
  assert output_single.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
  logging.info("  ✓ Single-device test passed!")

  # Create device mesh for parallelization
  logging.info("\nSetting up distributed environment...")
  device_mesh = dist.DeviceMesh(device.type, list(range(world_size)))
  logging.info("  Device mesh: %s", device_mesh)

  # Parallelize the model
  logging.info("\nParallelizing model with parallelize_module...")
  model_parallel = parallelize_mlp(model, device_mesh)
  logging.info(
      "sharded model: %s",
      _get_dtensor_hierarchy_string(model_parallel))
  model_parallel = model_parallel.to(device)

  logging.info("\nParallelized model parameters:")
  for name, param in model_parallel.named_parameters():
    if hasattr(param, "_local_tensor"):
      # It's a DTensor
      logging.info(
          f"  {name}: global={param.shape}, local={param._local_tensor.shape}, "
          f"placements={param.placements}"
      )
    else:
      logging.info(f"  {name}: {param.shape}")

  # Test parallelized forward pass
  logging.info("\nTesting parallelized forward pass...")
  output_parallel = model_parallel(x)
  logging.info(f"  Input shape: {x.shape}")
  logging.info(f"  Output shape: {output_parallel.shape}")
  assert output_parallel.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
  logging.info("  ✓ Parallel test passed!")

  # Train the parallelized model
  logging.info("\n Training parallelized MLP...")
  train_model(
      model_parallel,
      num_steps=NUM_TRAINING_STEPS,
      batch_size=BATCH_SIZE,
      seq_len=SEQ_LEN,
      hidden_size=HIDDEN_SIZE,
      learning_rate=1e-3,
      device=device,
  )


def _test_parallel_transformer_block(
    device: torch.device,
    rank: int,
    world_size: int,
    parallelize_fn: Callable[
        [TransformerBlock, dist.DeviceMesh], TransformerBlock
    ],
):
  setup_logger()
  set_device(device, rank)

  # Demonstrate TransformerBlock parallelization
  logging.info("TransformerBlock Parallelization test")

  logging.info("\nCreating and parallelizing TransformerBlock...")
  transformer_block = TransformerBlock(
      hidden_size=HIDDEN_SIZE,
      intermediate_size=INTERMEDIATE_SIZE,
      num_heads=NUM_HEADS,
  )

  logging.info("\nTransformerBlock model: %s", transformer_block)

  # Create device mesh for parallelization
  logging.info("\nSetting up distributed environment...")
  device_mesh = dist.DeviceMesh(device.type, list(range(world_size)))
  logging.info("  Device mesh: %s", device_mesh)

  transformer_parallel = parallelize_fn(
      transformer_block,
      device_mesh,
  )

  logging.info(
      "sharded model: %s",
      _get_dtensor_hierarchy_string(transformer_parallel))

  logging.info("\n Copying TransformerBlock to device: %s", device)
  transformer_parallel = transformer_parallel.to(device)

  logging.info("\nTransformerBlock parallelized parameters:")
  for name, param in transformer_parallel.named_parameters():
    if hasattr(param, "_local_tensor"):
      logging.info(
          "  %s: global=%s, placements=%s", name, param.shape, param.placements
      )
    else:
      logging.info("  %s: %s", name, param.shape)

  # Test transformer forward pass
  x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).to(device)
  output_transformer = transformer_parallel(x)
  logging.info("\n  Input shape: %s", x.shape)
  logging.info("  Output shape: %s", output_transformer.shape)
  assert output_transformer.shape == (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
  logging.info("  ✓ TransformerBlock test passed!")

  # Train the transformer block
  logging.info("\nTraining TransformerBlock...")
  train_model(
      transformer_parallel,
      num_steps=NUM_TRAINING_STEPS,
      batch_size=BATCH_SIZE,
      seq_len=SEQ_LEN,
      hidden_size=HIDDEN_SIZE,
      learning_rate=1e-4,
      device=device,
  )


def _test_fully_shard_transformer_block(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Tests DTensor fully_shard on the TransformerBlock."""
  # Init logic used: [Meta -> Shard -> Materialize].
  # (matches that of train_minimal and torch_tpu fsdp example.)
  setup_logger()
  set_device(device, rank)

  logging.info("Testing FSDP2 (fully_shard) on TransformerBlock.")

  logging.info("Setting up distributed environment...")
  device_mesh = dist.DeviceMesh(device.type, list(range(world_size)))

  # init on meta (matches trainer_minimal)
  logging.info("Initializing model on meta device...")
  with torch.device("meta"):
    model = TransformerBlock(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_heads=NUM_HEADS,
    )

  # apply fully_shard
  # FSDP wrapping happens while tensors are still on 'meta'
  logging.info("Applying fully_shard to meta model...")
  fully_shard(model, mesh=device_mesh)

  # move model from 'meta' to the actual device using to_empty.
  logging.info("Materializing model on %s via to_empty...", device)
  model.to_empty(device=device)

  # init weights
  logging.info("Initializing weights...")
  with torch.no_grad():
    model.init_weights()

  # verify sharding (using existing helper)
  logging.info("Verifying sharding...")
  logging.info(
      "Sharded model structure:\n%s",
      _get_dtensor_hierarchy_string(model)
  )

  # train
  logging.info("Starting FSDP2 training...")
  train_model(
      model,
      num_steps=NUM_TRAINING_STEPS,
      batch_size=BATCH_SIZE,
      seq_len=SEQ_LEN,
      hidden_size=HIDDEN_SIZE,
      learning_rate=1e-4,
      device=device,
  )


def _test_transformer_block_compile_view_behavior(
    device: torch.device,
    rank: int,
    world_size: int,
    acc_type: device_type.AcceleratorDeviceType,
):
  """Worker function to test sharded view within a compiled Attention module."""
  setup_logger()
  set_device(device, rank)

  logging.info(
      "Rank %d: Testing TransformerBlock TP on %s for view issue",
      rank,
      acc_type,
  )

  device_mesh = DeviceMesh(device.type, list(range(world_size)))

  # Initialize model on the target device
  model = TransformerBlock(
      hidden_size=HIDDEN_SIZE,
      intermediate_size=INTERMEDIATE_SIZE,
      num_heads=NUM_HEADS,
  ).to(device)

  # Parallelize the model
  model_parallel = parallelize_transformer_block_tp(model, device_mesh)

  # Input tensor
  x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device)

  # Create a Replicated DTensor for input
  x_dtensor = DTensor.from_local(
      x, device_mesh, [Replicate()], run_check=True
  )

  # Determine the backend for torch.compile
  if acc_type == device_type.AcceleratorDeviceType.TPU:
    backend = "tpu"
  else:
    backend = "inductor"

  logging.info(
      "Rank %d: Compiling model_parallel.attention with backend='%s'",
      rank,
      backend,
  )

  # Compile ONLY the attention block
  model_parallel.attention = torch.compile(  # pyrefly: ignore[bad-assignment]
      model_parallel.attention, backend=backend, fullgraph=True
  )
  logging.info("Rank %d: Compilation finished.", rank)

  # Forward pass  (calling attention directly to bypass un-parallelized
  # LayerNorms)
  output = model_parallel.attention(x_dtensor)

  expected_shape = torch.Size([BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE])
  assert (
      output.shape == expected_shape
  ), f"Expected {expected_shape}, got {output.shape}"
  logging.info(
      "Rank %d: Output shape assertion passed for backend %s", rank, backend
  )

def _test_fsdp_complex_bf16_mismatch(
    device: torch.device,
    rank: int,
    world_size: int,
):
  """Replicates issue with Llama3 when using BF16 Parameters + Complex64 Buffer during FSDP."""
  setup_logger()
  set_device(device, rank)

  device_mesh = dist.DeviceMesh(device.type, list(range(world_size)))

  class LlamaRoPERepro(nn.Module):
    def __init__(self):
      super().__init__()
      self.param = nn.Parameter(torch.ones(HIDDEN_SIZE, HIDDEN_SIZE))
      cpu_complex = torch.complex(torch.randn(SEQ_LEN, HIDDEN_SIZE // 2),
                                  torch.randn(SEQ_LEN, HIDDEN_SIZE // 2))
      # register buffer analogous to freqs_cis in Llama3
      self.register_buffer("complex_buf", cpu_complex)

    def forward(self, x):
      # bf16 linear projection (fsdp will wrap this)
      x = F.linear(x, self.param)
      # group to complex (f32)
      x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
      # complex multiplication with the buffer
      res = x_c * self.complex_buf[:x.shape[1], :]
      return torch.view_as_real(res).flatten(-2).type_as(x)

  with torch.device("meta"):
    model = LlamaRoPERepro()

  # apply FSDP with mp_policy that uses bfloat16 for parameters.
  from torch.distributed.fsdp import MixedPrecisionPolicy
  mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
  fully_shard(model, mesh=device_mesh, mp_policy=mp_policy)

  model.to_empty(device=device)

  # setup inputs
  x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

  logging.info("Trying FSDP forward/backward (The 13 vs 15 crash site)...")
  try:
    # b/489119669 - on TPU, this will trigger the crash
    # Check failed: result_buf.element_type() == tensor_element_type (13 vs. 15)
    out = model(x)
    out.sum().backward()
    logging.info("Test passed!")
  except Exception as e:
    logging.error("Test failed: %s", e)
    raise


class TrainMLPDTensorTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Tests the minimal distributed training setup for MLP models."""

  def test_parallel_mlp(self):
    distributed.run_distributed(
        self.num_devices, self.accelerator_device_type, _test_parallel_mlp
    )

  def test_parallel_transformer_block_tp(self):
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _test_parallel_transformer_block,
        parallelize_transformer_block_tp,
    )

  def test_parallel_transformer_block_tp_sp(self):
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _test_parallel_transformer_block,
        parallelize_transformer_block_tp_sp,
    )

  # NOTE: Flaky OOM on TPU (known issue with FSDP2 + TPU)
  def test_fully_shard_transformer_block(self):
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _test_fully_shard_transformer_block,
    )

  def test_transformer_block_compile_view_error(self):
    acc_type = self.accelerator_device_type

    if acc_type == device_type.AcceleratorDeviceType.CPU:
      self.skipTest(
          "torch.compile not fully supported on CPU in g3 (b/327271919)"
      )
      return

    if acc_type == device_type.AcceleratorDeviceType.TPU:
      self.skipTest(
          "Bug with torch.compile on TPU (b/501466433)"
      )
      return

    distributed.run_distributed(
        self.num_devices,
        acc_type,
        _test_transformer_block_compile_view_behavior,
        acc_type,
    )

  def test_fsdp_complex_bf16_mismatch(self):
    distributed.run_distributed(
        self.num_devices,
        self.accelerator_device_type,
        _test_fsdp_complex_bf16_mismatch,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
