"""Distributed parity of the HF transformers backend on the TPU lane.

Trains a few steps with the lane's production choices -- splash attention via
``attn_implementation=splash_tpu``, the fused Pallas cross entropy, SimpleFSDP
over every chip -- and checks loss, gradients and parameters step for step
against a single-device CPU run of the same config on the recorded batches
(``base_distributed_device_test._run_trainer_distributed_parity_test``). The
CPU side is the same wrapper with the portable choices substituted
(``config_registry.as_cpu_reference``: sdpa, plain cross entropy, no compile).

Run on the chips of this host, or on CPU with a gloo world (sharded CPU vs
single CPU):

    python -m torchtitan.experiments.tpu.transformers_modeling_backend.tests.test_hf_backend_distributed
    python -m ... --device=cpu --num_devices=2

The single-device check against the raw ``transformers`` model class is
``test_hf_reference_parity``.
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import accelerator_device_type as device_type
from torchtitan.experiments.tpu import base_distributed_device_test

from torchtitan.experiments.tpu.transformers_modeling_backend.tests.lane_args import (
    CPU_REFERENCE,
    lane_config_args,
)


class HFBackendDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """SimpleFSDP over all devices vs a single-device CPU run, per model family."""

  # Tolerances are set from measurement on this host (3 steps, fp32 params,
  # lr 8e-4 with 2 warmup steps). The portable path (sdpa + plain CE) agrees
  # with the CPU reference to 1e-4 on the loss; the kernels' bf16 internals
  # move the loss by 6-9e-3 (fused-CE logits in bf16) from step 0 on, with
  # gradients within 2e-3. Parameters drift to 4e-3 by step 2 on EVERY path,
  # including the portable control, so that bound is the optimizer trajectory
  # in fp32, not the kernels.
  _KERNEL_TOL = dict(loss_atol=2e-2, loss_rtol=1e-3, grad_atol=5e-3, grad_rtol=5e-3,
                     param_atol=1e-2, param_rtol=5e-3)
  _PORTABLE_TOL = dict(loss_atol=1e-3, loss_rtol=1e-3, grad_atol=5e-3, grad_rtol=5e-3,
                       param_atol=1e-2, param_rtol=5e-3)

  @parameterized.named_parameters([
      # Control: the kernel-shaped two-layer model on the portable path, so a
      # kernel regression shows as a gap between this case and the next two.
      dict(
          testcase_name="qwen3_debug_sdpa",
          config="hf_debugmodel_kernels_sdpa",
          hf_model="Qwen/Qwen3-0.6B",
          # On CPU (--device=cpu --num_devices=N: SimpleFSDP over gloo vs a
          # single CPU process) the config's compile(backend="tpu") has no
          # meaning; everything else is portable.
          cpu_args=["--compile.no-enable"],
          **_PORTABLE_TOL,
      ),
      # Splash only (plain CE), then the frontier feature set (splash with
      # document segment ids, fused CE, AC keeping the attention output,
      # per-layer compile). Two-layer debug shapes: the recorded history holds
      # fp32 params + grads for every step, so the full 0.6B / 1B checkpoints
      # would write ~17 GB per test.
      dict(
          testcase_name="qwen3_debug_splash",
          config="hf_debugmodel_kernels_splash",
          hf_model="Qwen/Qwen3-0.6B",
          skip_devices=[device_type.AcceleratorDeviceType.CPU],  # splash / fused CE are TPU kernels
          **_KERNEL_TOL,
      ),
      dict(
          testcase_name="qwen3_debug_max",
          config="hf_debugmodel_max",
          hf_model="Qwen/Qwen3-0.6B",
          args=["--loss.b_block_size=256"],  # the CE tile must divide the 256 tokens
          skip_devices=[device_type.AcceleratorDeviceType.CPU],  # splash / fused CE are TPU kernels
          **_KERNEL_TOL,
      ),
      dict(
          testcase_name="llama3_debug_max",
          config="hf_debugmodel_max",
          hf_model="unsloth/Llama-3.2-1B-Instruct",
          args=["--loss.b_block_size=256"],
          skip_devices=[device_type.AcceleratorDeviceType.CPU],  # splash / fused CE are TPU kernels
          **_KERNEL_TOL,
      ),
      # GPT-2 at full size: learned positions (wpe reaches named_children),
      # tied lm_head, attention module named `attn` rather than `self_attn`.
      dict(
          testcase_name="distilgpt2_splash",
          config="hf_full_max_plain_ce",
          hf_model="distilbert/distilgpt2",
          args=['--hf_config_overrides={"attn_pdrop": 0, "resid_pdrop": 0, "embd_pdrop": 0}'],
          skip_devices=[device_type.AcceleratorDeviceType.CPU],  # splash / fused CE are TPU kernels
          **_KERNEL_TOL,
      ),
      # ESM-1b encoder (bidirectional splash, learned positions, EsmLMHead); see
      # test_hf_reference_parity. Its 33-token vocabulary is widened to the
      # test tokenizer's, which TrainerMinimal does by itself but Trainer does not.
      dict(
          testcase_name="esm1b_debug_splash",
          config="hf_debugmodel_kernels_splash",
          hf_model="facebook/esm1b_t33_650M_UR50S",
          args=['--hf_config_overrides={"vocab_size": 2020}'],
          skip_devices=[device_type.AcceleratorDeviceType.CPU],  # splash / fused CE are TPU kernels
          **_KERNEL_TOL,
      ),
      # MoE: the Titan MoE is swapped in by parallelize_fn, which TrainerMinimal
      # only runs for world_size > 1 -- so the single-device CPU reference keeps
      # HF's own MoE block and the recorded Titan expert weights (w1_EFD, ...)
      # have nowhere to go: 16 parameters are skipped at load and the two runs
      # start from different experts (measured: loss off by 0.17 at step 0).
      # Kept as a documented skip until the reference can swap too.
      dict(
          testcase_name="tiny_mixtral_moe",
          config="hf_moe_auto",
          hf_model="hf-internal-testing/tiny-random-MixtralForCausalLM",
          skip_reason=(
              "CPU reference (world_size 1) does not run the Titan MoE swap, so"
              " its expert weights are not the recorded ones"
          ),
          skip_devices=[device_type.AcceleratorDeviceType.CPU],  # splash / fused CE are TPU kernels
          **_KERNEL_TOL,
      ),
  ])
  def test_simple_fsdp_parity(
      self,
      config,
      hf_model,
      args=None,
      cpu_args=None,
      skip_devices=None,
      skip_reason=None,
      loss_atol=1e-3,
      loss_rtol=1e-3,
      grad_atol=5e-3,
      grad_rtol=5e-3,
      param_atol=5e-3,
      param_rtol=5e-3,
  ):
    if skip_reason:
      self.skipTest(skip_reason)
    if self.accelerator_device_type == device_type.AcceleratorDeviceType.CPU:
      args = (args or []) + (cpu_args or [])
    self._run_trainer_distributed_parity_test(
        config_args=lane_config_args(
            config, hf_model, steps=3, extra=["--parallelism.use_simple_fsdp"] + (args or [])
        ),
        tensor_parallel_degree=1,
        data_parallel_shard_degree=-1,
        skip_devices=skip_devices,
        loss_atol=loss_atol,
        loss_rtol=loss_rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        param_atol=param_atol,
        param_rtol=param_rtol,
        cpu_config_fn=CPU_REFERENCE,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
