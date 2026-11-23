"""The command line the lane's parity tests train with.

Both suites run the same small recipe -- 2 x 128 tokens per rank, fp32
parameters, foreach AdamW -- so that a difference between them is the
device / parallelism axis and nothing else.
"""

MODULE = "torchtitan.experiments.tpu.transformers_modeling_backend"
CPU_REFERENCE = f"{MODULE}.config_registry:as_cpu_reference"


def lane_config_args(config: str, hf_model: str, *, steps: int, extra=None) -> list[str]:
  return [
      f"--module={MODULE}",
      f"--config={config}",
      f"--hf_model={hf_model}",
      "--optimizer.foreach=true",
      "--hf_assets_path=tests/assets/tokenizer",
      "--model.max_context_length=128",
      f"--training.steps={steps}",
      "--precision.param_dtype=float32",
      "--precision.reduce_dtype=float32",
      "--training.num_tokens_per_microbatch_per_dp_rank=256",  # 2 x 128
  ] + list(extra or [])
