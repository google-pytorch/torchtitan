"""Config registry for the JAX-lane Llama3 model."""

from torchtitan.experiments.jax import llama3 as jax_llama3
from torchtitan.experiments.jax.jax_job_config import JaxJobConfig
from torchtitan.experiments.tpu.llama3 import config_registry as tpu_llama3


def llama3_debugmodel() -> JaxJobConfig:
  """Llama3 debug model on JAX (smoke test)."""
  cfg = JaxJobConfig.derive_from(tpu_llama3.llama3_debugmodel())
  assert cfg.model is not None
  cfg.model = jax_llama3.args["debugmodel"]
  cfg.jax_config.use_scan = False
  return cfg
