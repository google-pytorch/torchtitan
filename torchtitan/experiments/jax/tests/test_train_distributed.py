"""Tests for the JAX distributed training setup.
"""

from absl.testing import absltest
from absl.testing import parameterized

from torchtitan.experiments.tpu.config_manager import ConfigManager
from torchtitan.experiments.jax import train_minimal


class TestJaxDistributed(parameterized.TestCase):

  @parameterized.named_parameters([
      dict(
          testcase_name="llama3_debug_scan",
          module="torchtitan.experiments.jax.llama3",
          config="llama3_debugmodel",
          use_scan=True,
      ),
      dict(
          testcase_name="llama3_debug_no_scan",
          module="torchtitan.experiments.jax.llama3",
          config="llama3_debugmodel",
          use_scan=False,
      ),
  ])
  def test_train_jax(self, module, config, use_scan):
    scan_flag = (
        "--jax_config.use_scan"
        if use_scan
        else "--jax_config.no-use-scan"
    )
    args = [
        "--module",
        module,
        "--config",
        config,
        scan_flag,
        "--jax_config.model_layer_override=1",
        "--dataloader.dataset_path=tests/assets/c4_test",
        "--hf_assets_path=tests/assets/tokenizer",
        "--training.max_context_length=128",
        "--training.steps=3",
        # Upstream d771abb0 renamed `--training.local_batch_size` to the
        # token-denominated `--training.num_tokens_per_microbatch_per_dp_rank`:
        # 8 samples x 128 tokens.
        "--training.num_tokens_per_microbatch_per_dp_rank=1024",
        "activation-checkpoint:full",
    ]

    job_config = ConfigManager().parse_args(args)
    success = train_minimal.main_train_loop(job_config)
    self.assertTrue(success)


if __name__ == "__main__":
  absltest.main()
