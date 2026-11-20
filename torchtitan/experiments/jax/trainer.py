"""JAX trainer for Llama3 and other models.

Mirrors the structure of torchax/trainer.py but uses pure JAX / Flax NNX
instead of the torchax PyTorch-on-JAX wrapper.

Training loop:
  1. Model defined in Flax NNX; parameters split into (graphdef, state).
  2. Optimizer state managed by nnx.Optimizer (integrates optax).
  3. Train step compiled with nnx.jit + nnx.value_and_grad.
  4. Inputs sharded on FSDP axis; parameters sharded per sharding_map.
"""

import logging
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.experiments.jax import data_utils
from torchtitan.experiments.jax import distributed
from torchtitan.experiments.jax import jax_profiling
from torchtitan.experiments.jax import metrics as jax_metrics
from torchtitan.experiments.jax import splash_attn
from torchtitan.experiments.jax import llama3 as jax_llama3


logger = logging.getLogger(__name__)

P = jax.sharding.PartitionSpec


def _scale_by_adam_bf16_state(b1, b2, eps):
    """Adam with mu and nu stored in bf16; math runs in fp32 per-leaf.

    Stock `optax.scale_by_adam` only exposes `mu_dtype`; nu stays in params'
    dtype (fp32 here). This helper casts both moments back to bf16 after the
    update, cutting opt-state footprint in half without changing numerics
    visibly (the reduction is the same as MaxText's `mu_dtype=bf16` recipe
    extended to nu).
    """
    from optax._src import transform as _optax_transform
    ScaleByAdamState = _optax_transform.ScaleByAdamState

    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, dtype=jnp.bfloat16), params)
        nu = jax.tree.map(lambda t: jnp.zeros_like(t, dtype=jnp.bfloat16), params)
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        count_inc = state.count + 1
        bc1 = 1.0 - b1 ** count_inc.astype(jnp.float32)
        bc2 = 1.0 - b2 ** count_inc.astype(jnp.float32)

        def _mu(g, m):
            if g is None:
                return m
            return (b1 * m.astype(jnp.float32)
                    + (1.0 - b1) * g.astype(jnp.float32)).astype(jnp.bfloat16)

        def _nu(g, n):
            if g is None:
                return n
            g32 = g.astype(jnp.float32)
            return (b2 * n.astype(jnp.float32)
                    + (1.0 - b2) * (g32 * g32)).astype(jnp.bfloat16)

        def _upd(g, m, n):
            if g is None:
                return None
            mhat = m.astype(jnp.float32) / bc1
            nhat = n.astype(jnp.float32) / bc2
            return (mhat / (jnp.sqrt(nhat) + eps)).astype(g.dtype)

        new_mu = jax.tree.map(_mu, updates, state.mu,
                              is_leaf=lambda x: x is None)
        new_nu = jax.tree.map(_nu, updates, state.nu,
                              is_leaf=lambda x: x is None)
        upds = jax.tree.map(_upd, updates, new_mu, new_nu,
                            is_leaf=lambda x: x is None)
        return upds, ScaleByAdamState(count=count_inc, mu=new_mu, nu=new_nu)

    return optax.GradientTransformation(init_fn, update_fn)


class JaxTrainer:

  def __init__(
      self,
      mesh: jax.sharding.Mesh,
      job_config: Any,
      accelerator: str,
      num_global_devices: int,
      num_local_devices: int,
  ):
    self.mesh = mesh
    self.job_config = job_config
    self.accelerator = accelerator
    self.num_global_devices = num_global_devices
    self.num_local_devices = num_local_devices

  # ------------------------------------------------------------------
  # Data
  # ------------------------------------------------------------------

  def setup_dataloader(self, job_config):
    """Build a dataloader that returns (inputs, labels) numpy arrays."""
    # Upstream d771abb0 replaced the sample-denominated
    # ``training.seq_len`` / ``training.local_batch_size`` /
    # ``training.global_batch_size`` knobs with the token-denominated
    # ``training.max_context_length`` /
    # ``training.num_tokens_per_microbatch_per_dp_rank`` /
    # ``training.num_tokens_per_train_step``. The JAX trainer reasons in
    # samples, so divide the token counts back out by the context length.
    seq_len = job_config.training.max_context_length
    expected_global_batch_size = (
        job_config.training.num_tokens_per_train_step // seq_len
    )
    if expected_global_batch_size <= 0:
      expected_global_batch_size = (
          job_config.training.num_tokens_per_microbatch_per_dp_rank // seq_len
      )

    dataset = job_config.dataloader.dataset
    if not dataset or dataset.startswith('fake'):
      train_loader = data_utils.fake_dataloader(
          job_config.training.steps,
          seq_len,
          expected_global_batch_size,
      )
      logger.warning('Using fake data loader.')
      return train_loader

    # Real dataset via the new config-driven builders (the per-train-spec
    # ``build_tokenizer_fn`` / ``build_dataloader_fn`` callables were
    # removed in the 2026-05-04 OSS sync; ``config_registry`` factories
    # now pin the concrete tokenizer / dataloader configs on the job
    # config). Upstream d771abb0 also dropped
    # ``HuggingFaceTextDataLoader`` in favour of
    # ``torchtitan.components.data.GrainDataLoader``, whose build kwargs
    # are ``max_context_length`` / ``num_tokens_per_batch`` rather than
    # ``seq_len`` / ``local_batch_size``; both are still build kwargs so we
    # don't need to mutate ``job_config.training``.
    tokenizer = job_config.tokenizer.build(
        tokenizer_path=job_config.hf_assets_path,
    )
    torch_loader = job_config.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=tokenizer,
        max_context_length=seq_len,
        num_tokens_per_batch=expected_global_batch_size * seq_len,
    )
    return data_utils.torch_loader_to_jax(torch_loader, seq_len=seq_len)

  # ------------------------------------------------------------------
  # Model
  # ------------------------------------------------------------------

  def setup_model(self, job_config):
    """Instantiate and shard the Flax NNX model."""
    model_name = job_config.model_spec.name  # e.g. 'llama3'
    model_flavor = job_config.model_spec.flavor  # e.g. '8B'
    jax_config = job_config.jax_config
    use_scan = jax_config.use_scan

    if model_name == 'llama3':
      jax_model_mod = jax_llama3
      layer_attr = 'n_layers'
    else:
      raise ValueError(f'Unsupported model: {model_name}')

    model_args = jax_model_mod.args[model_flavor]
    # ``max_seq_len`` is a JAX-side dataclass field; only the job config
    # side was renamed (``training.seq_len`` -> ``training.
    # max_context_length``) by upstream d771abb0.
    model_args.max_seq_len = job_config.training.max_context_length

    if jax_config.model_layer_override is not None:
      old = getattr(model_args, layer_attr)
      new = jax_config.model_layer_override
      logger.warning('Overriding %s from %d to %d.', layer_attr, old, new)
      setattr(model_args, layer_attr, new)

    # Choose activation checkpoint policy.
    from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC

    ac_config = job_config.activation_checkpoint
    if isinstance(ac_config, FullAC.Config):
      checkpoint_policy = jax.checkpoint_policies.nothing_saveable
    elif isinstance(ac_config, SelectiveAC.Config):
      checkpoint_policy = jax.checkpoint_policies.dots_saveable
    elif ac_config is None:
      checkpoint_policy = None
    else:
      checkpoint_policy = jax.checkpoint_policies.nothing_saveable

    # Build splash attention if on TPU and the flag is enabled.
    splash_cfg = job_config.splash_attention_kernel
    devices = jax.devices()
    platform = devices[0].platform if devices else 'cpu'
    attn_fn = None
    if platform == 'tpu' and splash_cfg.use_splash_attention_kernel:
      attn_fn = splash_attn.make_splash_attention_fn(self.mesh, splash_cfg)
    elif platform == 'tpu':
      logger.info(
          'Splash attention disabled via '
          '--splash_attention_kernel.use_splash_attention_kernel=False; '
          'using fallback scaled-dot-product attention.'
      )

    sharding_map = (
        jax_model_mod.sharding_map_scan
        if use_scan
        else jax_model_mod.sharding_map_original
    )

    logger.info(
        'Building model %s-%s (scan=%s) ...', model_name, model_flavor, use_scan
    )

    # Map job_config.training.dtype (Literal['bfloat16','float32']) to
    # the model's master-weight dtype. Compute dtype stays bf16.
    _param_dtype_map = {'float32': jnp.float32, 'bfloat16': jnp.bfloat16}
    param_dtype = _param_dtype_map[job_config.training.dtype]
    logger.info(
        'Model param_dtype=%s (compute dtype=bfloat16)',
        jnp.dtype(param_dtype).name,
    )

    # Instantiate on the *host-local* CPU so we don't OOM before
    # sharding. Using `jax.devices('cpu')[0]` on multi-host picks a
    # CPU device that may not be addressable from every host, which
    # later breaks `jax.device_get(param)` with "Cannot reshard an
    # input that is not fully addressable". `local_devices('cpu')[0]`
    # is always host-local.
    with jax.default_device(jax.local_devices(backend='cpu')[0]):
      model = jax_model_mod.Transformer(
          model_args,  # pyrefly: ignore[bad-argument-type]
          use_scan=use_scan,
          attn_fn=attn_fn,
          checkpoint_policy=checkpoint_policy,
          param_dtype=param_dtype,
          rngs=nnx.Rngs(0),
      )

    # Extract state, apply sharding, merge back.
    graphdef, state = nnx.split(model)
    logger.info(
        'Applying sharding to %d parameter tensors ...',
        len(jax.tree_util.tree_leaves(state)),
    )
    state = distributed.apply_sharding_to_state(state, sharding_map, self.mesh)
    model = nnx.merge(graphdef, state)

    self.model_args = model_args
    return model

  # ------------------------------------------------------------------
  # Optimizer
  # ------------------------------------------------------------------

  def setup_optimizer(self, job_config, model):
    # Adapt to OptimizersContainer.Config
    optim_config = job_config.optimizer
    if (
        hasattr(optim_config, 'param_groups')
        and len(optim_config.param_groups) > 0
    ):
      # Assume first group is representative for simple JAX setups
      pg = optim_config.param_groups[0]
      opt_name = pg.optimizer_name.lower()
      kwargs = pg.optimizer_kwargs
      lr = kwargs.get('lr', 8e-4)
      eps = kwargs.get('eps', 1e-8)
      wd = kwargs.get('weight_decay', 0.1)
      betas = kwargs.get('betas', (0.9, 0.95))
      b1, b2 = betas
    else:
      opt_name = optim_config.name.lower()
      lr = optim_config.lr
      b1 = optim_config.beta1 or 0.9
      b2 = optim_config.beta2 or 0.95
      eps = optim_config.eps or 1e-8
      wd = optim_config.weight_decay or 0.1

    if opt_name == 'sgd':
      tx = optax.sgd(lr)
    elif opt_name == 'adam':
      tx = optax.adam(lr, b1=b1, b2=b2, eps=eps)
    elif opt_name == 'adamw':
      if job_config.jax_config.adamw_bf16_state:
        # Custom Adam with both mu and nu stored in bf16 (upcast to
        # fp32 per-leaf inside the update math). Stock optax exposes
        # mu_dtype but not nu_dtype; on fp32 master weights the f32
        # stacked-MLP nu can be 3.5 GiB/chip and XLA schedules multiple
        # live copies at optimizer peak. Opt in with
        # --jax_config.adamw_bf16_state on memory-tight configs.
        logger.info('adamw: using custom bf16 mu+nu state')
        tx = optax.chain(
            _scale_by_adam_bf16_state(b1=b1, b2=b2, eps=eps),
            optax.add_decayed_weights(wd),
            optax.scale_by_learning_rate(lr),
        )
      else:
        tx = optax.adamw(lr, b1=b1, b2=b2, eps=eps, weight_decay=wd)
    else:
      raise ValueError(f'Unsupported optimizer: {opt_name}')

    return nnx.Optimizer(model, tx, wrt=nnx.Param)

  # ------------------------------------------------------------------
  # Training
  # ------------------------------------------------------------------

  def train(self):
    job_config = self.job_config
    model = self.setup_model(job_config)
    optimizer = self.setup_optimizer(job_config, model)

    train_loader = self.setup_dataloader(job_config)

    metrics_processor = jax_metrics.JaxMetricsProcessor(
        job_config,
        self.accelerator,
        self.num_global_devices,
        tpu_megacore=job_config.jax_config.tpu_megacore,
        log_freq=job_config.metrics.log_freq,
    )
    # Approximate flops: 6 * n_params * seq_len (dense model estimate).
    n_params = sum(
        x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )
    metrics_processor.num_flops_per_token = 6 * n_params

    # donate model+optimizer buffers so XLA can alias param-in with
    # param-out and opt-state-in with opt-state-out, halving peak fp32
    # copies in the optimizer update section (observed 7+ f32[32,1024,
    # 28672] live copies in the memory report without donate).

    @nnx.jit(donate_argnames=('model', 'optimizer'))
    def train_step(model, optimizer: nnx.Optimizer, inputs, labels):
      def loss_fn(model):
        logits = model(inputs)  # [B, S, vocab] — bf16, vocab-sharded
        B, S, V = logits.shape
        logits_2d = logits.reshape(B * S, V)
        labels_1d = labels.reshape(B * S)
        log_z = jax.nn.logsumexp(logits_2d, axis=-1)
        target = jnp.take_along_axis(
            logits_2d, labels_1d[..., None], axis=-1
        ).squeeze(-1)
        loss = (log_z - target).astype(jnp.float32).sum()
        return loss

      loss, grads = nnx.value_and_grad(
          loss_fn,
          argnums=nnx.DiffState(0, nnx.Param),
      )(model)
      optimizer.update(model, grads)
      return loss

    logger.info('Starting training loop ...')

    with jax_profiling.maybe_enable_profiling(
        job_config.profiler,
        global_step=0,
        base_folder=job_config.dump_folder,
    ) as profiler:
      step = -1
      for inputs_np, labels_np in train_loader:
        step += 1
        if step >= job_config.training.steps:
          break

        data_load_start = time.perf_counter()
        # Keep the batch as numpy through sharding: jnp.array
        # commits to host 0's TPU, which would then require an
        # unsupported cross-host reshard on multi-host runs.
        # shard_input calls sharded_device_put, which indexes the
        # numpy array per-addressable-device.
        inputs_np = np.asarray(inputs_np, dtype=np.int32)
        labels_np = np.asarray(labels_np, dtype=np.int32)

        # Shard batch on FSDP axis.
        inputs = distributed.shard_input(
            inputs_np,
            self.mesh,
            self.num_global_devices,
            self.num_local_devices,
        )
        labels = distributed.shard_input(
            labels_np,
            self.mesh,
            self.num_global_devices,
            self.num_local_devices,
        )

        ntokens = labels.size
        metrics_processor.ntokens_since_last_log += ntokens
        metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )

        with jax.named_scope('train_step'):
          loss = train_step(model, optimizer, inputs, labels)

        if profiler is not None:
          jax.block_until_ready(loss)
          profiler.step()

        if metrics_processor.should_log(step + 1):
          jax.block_until_ready(loss)
          metrics_processor.log(step + 1, float(loss) / ntokens)

    return True
