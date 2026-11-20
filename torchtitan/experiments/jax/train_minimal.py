r"""Pure JAX training for LLM models on TPU.

Parallel to torchtitan/experiments/torchax/train_minimal.py but uses pure
JAX / Flax NNX instead of torchax (the PyTorch-on-JAX wrapper).

Run llama3 debugmodel on v6e-4:
```
python -m torchtitan.experiments.jax.train_minimal \
    --module=torchtitan.experiments.jax.llama3 \
    --config=llama3_debugmodel
```
"""

import logging
import sys
import typing
from typing import Any

from absl import app, flags
import jax
import torchtitan.observability.logging
import torchtitan.config
from torchtitan.experiments.tpu.config_manager import ConfigManager
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.jax import jax_job_config
from torchtitan.experiments.jax import trainer as jax_trainer
from torchtitan.experiments.jax import utils as jax_utils


P = jax.sharding.PartitionSpec
logger = logging.getLogger(__name__)


def _maybe_init_jax_distributed():
  """Initialise JAX's multi-host runtime when running under a multi-host
  launcher (GKE/XPK, SLURM, etc.). No-op on a single-host VM — libtpu's
  local discovery already gives us all chips.

  Multi-host is detected via the ``TPU_WORKER_HOSTNAMES`` env var exported
  by Cloud-TPU-on-GKE; it holds a comma-separated list of worker hostnames
  (len > 1 ⇒ multi-host). Rank and host count are inferred from
  ``TPU_WORKER_ID`` / the same hostnames list, which are guaranteed to be
  set on every TPU GKE pod.
  """
  import os
  hostnames = os.environ.get('TPU_WORKER_HOSTNAMES')
  if not hostnames:
    return
  hosts = hostnames.split(',')
  if len(hosts) <= 1:
    return
  process_id = int(os.environ.get('TPU_WORKER_ID', '0'))
  coordinator = f"{hosts[0]}:8476"
  logger.info(
      'Initialising jax.distributed: num_processes=%d process_id=%d '
      'coordinator=%s',
      len(hosts), process_id, coordinator,
  )
  jax.distributed.initialize(
      coordinator_address=coordinator,
      num_processes=len(hosts),
      process_id=process_id,
  )


def main_train_loop(job_config: Any):
  torchtitan.observability.logging.init_logger()

  # Must apply XLA and libtpu flags before the JAX backend is initialized.
  tpu_flags.apply_init_flags(job_config)

  # Must run before any jax.devices() / jit call so the TPU runtime starts
  # with the full multi-host mesh instead of a single-host slice.
  _maybe_init_jax_distributed()

  logger.info('Running with config: %s', job_config)

  devices = jax.devices()
  if not devices:
    raise RuntimeError('No JAX devices found.')
  platform = devices[0].platform
  device_kind = devices[0].device_kind
  accelerator = jax_utils.get_accelerator_short_name(device_kind)
  logger.info(
      "Detected JAX device '%s', accelerator type '%s'",
      device_kind,
      accelerator,
  )

  num_global_devices = jax.device_count()
  num_local_devices = jax.local_device_count()
  logger.info(
      'Devices: %d global, %d local', num_global_devices, num_local_devices
  )

  tp = job_config.parallelism.tensor_parallel_degree
  if tp <= 0:
    tp = 1
  fsdp = num_global_devices // tp

  if platform == 'tpu' and job_config.jax_config.tpu_num_slices > 1:
    dev_array = jax.experimental.mesh_utils.create_hybrid_device_mesh(
        (fsdp // job_config.jax_config.tpu_num_slices, tp),
        (job_config.jax_config.tpu_num_slices, 1),
        devices,
        process_is_granule=False,
        allow_split_physical_axes=True,
    )
    mesh = jax.sharding.Mesh(dev_array, ('fsdp', 'tp'))
  else:
    mesh = jax.make_mesh(
        (fsdp, tp),
        ('fsdp', 'tp'),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )

  logger.info('Mesh: fsdp=%d tp=%d', fsdp, tp)

  with mesh:
    t = jax_trainer.JaxTrainer(
        mesh=mesh,
        job_config=job_config,
        accelerator=accelerator,
        num_global_devices=num_global_devices,
        num_local_devices=num_local_devices,
    )
    return t.train()


def main(argv):
  del argv
  config = typing.cast(
      jax_job_config.JaxJobConfig,
      ConfigManager().parse_args(sys.argv[1:]),
  )
  assert main_train_loop(config), 'training failed'


if __name__ == '__main__':
  app.run(
      main,
      flags_parser=(
          lambda args: flags.FLAGS(args, known_only=True)
      ),
  )
