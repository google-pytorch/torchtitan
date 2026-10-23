"""Distributed model test for torchtitan models."""

import os

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchtitan.config
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu import local_tpu_device
import torchtitan.experiments.tpu.examples.gemma4  # trigger model registration
import torchtitan.experiments.tpu.llama3  # trigger model registration
import torchtitan.experiments.tpu.qwen3   # trigger model registration
import torchtitan.experiments.tpu.tpu_job_config
from torchtitan.tools import utils
from torchtitan.experiments.tpu.workarounds import use_cpu_safe_histc_patch




def setup_logger():
  """Sets up the logger for the test."""
  if not dist.is_initialized() or dist.get_rank() == 0:
    print("Setting logger verbosity to INFO")
    logging.set_verbosity(logging.INFO)
  else:
    print("Setting logger verbosity to ERROR")
    logging.set_verbosity(logging.ERROR)


def _run_model_distributed(
    train_config: torchtitan.experiments.tpu.tpu_job_config.TPUTrainerConfig,
):
  use_cpu_safe_histc_patch()
  device = utils.get_local_device()
  world_size = int(os.environ.get("WORLD_SIZE", 1))
  if world_size > 1:
    dist_utils.init_distributed(
        train_config.comm,
        enable_cpu_backend=train_config.training.enable_cpu_offload,
        base_folder=train_config.dump_folder,
    )
  setup_logger()

  assert train_config.model_spec is not None
  model_spec = train_config.model_spec
  model_args = model_spec.model
  model_args.update_from_config(config=train_config)

  parallelism_config = train_config.parallelism
  parallel_dims = torchtitan.distributed.ParallelDims(
      dp_shard=parallelism_config.data_parallel_shard_degree,
      dp_replicate=parallelism_config.data_parallel_replicate_degree,
      cp=parallelism_config.context_parallel_degree,
      tp=parallelism_config.tensor_parallel_degree,
      pp=parallelism_config.pipeline_parallel_degree,
      ep=parallelism_config.expert_parallel_degree,
      world_size=world_size,
  )

  # Initializing model consistently with the way TorchTitan does it.
  logging.info("Initializing model with dtype %s", train_config.training.dtype)
  with (
      torch.device("meta"),
      utils.set_default_dtype(
          torchtitan.config.TORCH_DTYPE_MAP[train_config.training.dtype]
      ),
  ):
    model = model_args.build()
  logging.info("Parallelizing model")
  model = model_spec.parallelize_fn(
      model,
      parallel_dims=parallel_dims,
      training=train_config.training,
      parallelism=train_config.parallelism,
      compile_config=train_config.compile,
      ac_config=train_config.activation_checkpoint,
      dump_folder=train_config.dump_folder,
  )
  logging.info("Calling model.to_empty()")
  model.to_empty(device=device)
  with torch.no_grad():
    logging.info("Calling model.init_weights()")
    model.init_weights(buffer_device=None)

  # TODO: b/45811390 - fused=False is required for torch_tpu
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, fused=False)

  loss_fn = train_config.loss.build(compile_config=train_config.compile)
  if isinstance(loss_fn, torchtitan.components.loss.ChunkedLossWrapper):
    loss_fn.set_lm_head(model.lm_head)
    model._skip_lm_head = True

  train_context = dist_utils.get_spmd_context(
      parallel_dims=parallel_dims, spmd_typechecking=False
  )

  n_steps = train_config.training.steps
  # Upstream replaced seq_len/local_batch_size with a token budget:
  # local_batch_size = num_tokens_per_microbatch_per_dp_rank // max_context_length.
  seq_len = train_config.training.max_context_length
  batch_size = (
      train_config.training.num_tokens_per_microbatch_per_dp_rank // seq_len
  )
  for step in range(n_steps):
    logging.info("Step %s / %s", step, n_steps)
    # Generate new inputs for each step
    if model_spec.name == "flux":
      # Use batch_size=1 to avoid OOM issues in tests.
      batch_size = 1
      img = torch.randn(
          batch_size, seq_len, model_args.in_channels, device=device
      )
      img_ids = torch.zeros(batch_size, seq_len, 3, device=device)
      txt = torch.randn(
          batch_size, seq_len, model_args.context_in_dim, device=device
      )
      txt_ids = torch.zeros(batch_size, seq_len, 3, device=device)
      timesteps = torch.randn(batch_size, device=device)
      y = torch.randn(batch_size, model_args.vec_in_dim, device=device)

      optimizer.zero_grad()
      with train_context():  # pytype: disable=not-callable
        logging.info("calling model()")
        out_device = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            y=y,
        )
        loss = loss_fn(out_device, img)
        if isinstance(loss, tuple):
          loss, _ = loss
        logging.info("calling loss.backward()")
        loss.backward()
    else:
      from torchtitan.experiments.tpu.unflatten_shim import set_unflatten_seq_len  # pylint: disable=g-import-not-at-top

      tokens_cpu = torch.randint(
          0, model_args.vocab_size, (batch_size * seq_len,), device="cpu"
      )
      tokens_device = tokens_cpu.to(device)
      positions_device = (
          torch.arange(seq_len, device=device).repeat(batch_size)
      )
      optimizer.zero_grad()
      with train_context(), set_unflatten_seq_len(seq_len):  # pytype: disable=not-callable
        logging.info("calling model()")
        out_device = model(tokens_device, positions=positions_device)
        loss = loss_fn(out_device, tokens_device)
        if isinstance(loss, tuple):
          loss, _ = loss
        logging.info("calling loss.backward()")
        loss.backward()
    logging.info("loss: %s", loss.item())

    optimizer.step()


class Qwen3DTensorParallelizeTest(base_distributed_device_test.BaseDistributedDeviceTest):
  """Tests the DTensor-based `apply_non_moe_tp` in a distributed environment."""

  @parameterized.named_parameters([
      dict(
          testcase_name="llama3_tp_dtensor",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="llama3_tp_dtensor_compile",
          module="torchtitan.experiments.tpu.llama3",
          config="llama3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
      ),
      dict(
          testcase_name="qwen3_tp_dtensor",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="qwen3_tp_dtensor_compile",
          module="torchtitan.experiments.tpu.qwen3",
          config="qwen3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          # TODO(jerryx99): Failing only on TPU7x b/568843473
          skip_on_tpu7x=True,
          skip_devices=[
              # TODO(jerryx99): Flaky in TAP on Viperlite: rank 0 is aborted
              # with SLICE_FAILURE_CHIP_DRIVER_ERROR during teardown after
              # training completes. Re-enable once the flake is resolved.
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_tp_dtensor",
          module="torchtitan.experiments.tpu.deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          skip_devices=[
              # b/540049946R Re-enable once test is fixed.
              # TODO: ValueError: Linear: output DTensor has placements (Shard(dim=1),), but out_src_shardings expects (Shard(dim=2),).
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="deepseek_v3_tp_dtensor_compile",
          module="torchtitan.experiments.tpu.deepseek_v3",
          config="deepseek_v3_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          skip_devices=[
              # b/501466433 - Bug with torch.compile
              device_type.AcceleratorDeviceType.TPU,
          ],
      ),
      dict(
          testcase_name="flux_tp_dtensor",
          module="torchtitan.experiments.tpu.flux",
          config="flux_testmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
      ),
      dict(
          testcase_name="flux_tp_dtensor_compile",
          module="torchtitan.experiments.tpu.flux",
          config="flux_testmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          dataset_path="tests/assets/cc12m_test",
          dataset="cc12m-test",
          enable_compile=True,
      ),
      dict(
          testcase_name="gemma4_tp",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
      ),
      dict(
          testcase_name="gemma4_tp_compile",
          module="torchtitan.experiments.tpu.examples.gemma4",
          config="gemma4_debugmodel",
          data_parallel_shard_degree=1,
          tensor_parallel_degree=-1,
          enable_compile=True,
          # TODO: Same TPU7x-only XLA fusion_emitter.cc CHECK failure
          # ("Scaled window bound is not a power of 2") as qwen3_tp_compile.
          # b/568843473
          skip_on_tpu7x=True,
      ),
  ])
  def test_model_distributed(
      self,
      module,
      config,
      data_parallel_shard_degree,
      tensor_parallel_degree,
      skip_devices=None,
      enable_compile=False,
      dataset_path="tests/assets/c4_test",
      dataset="c4_test",
      skip_on_tpu7x=False,
  ):
    if (
        skip_on_tpu7x
        and self.accelerator_device_type == device_type.AcceleratorDeviceType.TPU
        and local_tpu_device.get_local_chips()[0]
        == local_tpu_device.TpuChip.V7X
    ):
      self.skipTest("Failing only on TPU7x: b/568843473")
    args = [
        f"--module={module}",
        f"--config={config}",
        "--optimizer.implementation=foreach",
        "--hf_assets_path=tests/assets/tokenizer",
        f"--dataloader.dataset_path={dataset_path}",
        f"--dataloader.dataset={dataset}",
        "--training.max_context_length=32",
        "--training.steps=3",
    ]
    # Activation checkpoint configuration refactored to use tyro subcommands.
    # This must go last as tyro expects all subsequent flags to be part of the
    # subcommand.
    args.append("activation-checkpoint:none")

    self._test_train_distributed(
        args,
        data_parallel_shard_degree,
        tensor_parallel_degree,
        skip_devices,
        start_trainer=_run_model_distributed,
        enable_compile=enable_compile,
    )

if __name__ == "__main__":
  mp.set_start_method("spawn")
  absltest.main()
