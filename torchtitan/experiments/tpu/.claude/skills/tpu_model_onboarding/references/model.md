# Model code: layout, the weight mapping, registration

`tpu/` means `torchtitan/experiments/tpu/`. Wrapping an upstream model
(`tpu/muse_glimmer/`, commit 8be9902531) reuses upstream's model, adapter and
sharding; only a port (`tpu/examples/<name>/`, the Gemma 4 port 66d411eed5)
writes the files below.

## Port layout

```
tpu/examples/<name>/
  __init__.py            flavors {str: Model.Config} + model_registry(flavor, ...) -> ModelSpec
  model.py               Model(Decoder) + nested Model.Config; get_nparams_and_flops()
  state_dict_adapter.py  the weight mapping (below)
  sharding.py            set_<name>_sharding_config(...) with the decoder_sharding helpers
  infra/parallelize.py   model.parallelize(parallel_dims): MoE fsdp -> AC -> compile -> FSDP
  config_registry.py     <name>_debugmodel() -> TPUTrainerConfig, then the real sizes
  tests/
```

Build from `torchtitan/models/common/` (`Decoder`, `TransformerBlock`,
`AttentionBase`, `FeedForward`, `MoE`, `RotaryEmbedding`, `RMSNorm`); subclass
for a deviation, do not copy. Model files are architecture only: TPU-only
behaviour goes through converters (`ModelConfigConverter` subclasses passed as
`converters=[...]`), `override.imports` factories under `tpu/overrides/`, or
`workarounds.apply_patches`, never `if tpu:` in the model. Hyperparameters live
on `Model.Config`; `get_nparams_and_flops()` must count what the model does
(active experts, GQA, tied embeddings) or the recipe reports wrong MFU.

## The weight mapping (state_dict_adapter.py)

What it is: a `StateDictAdapter` subclass (`torchtitan/protocols/state_dict_adapter.py`)
with `from_hf(hf_state_dict) -> titan_state_dict` and `to_hf(titan_state_dict)
-> hf_state_dict`. The names are legacy: for any source the job is the same
key-and-layout translation, and the class is what `ModelSpec.state_dict_adapter`
points at.

Who calls it: the checkpointer, when the recipe sets
`--checkpoint.initial_load_in_hf` with `--hf_assets_path=<dir>` holding the
safetensors files and `model.safetensors.index.json` (the adapter reads the
index to build `fqn_to_index_mapping`; `base.py` asserts both are present).
Loading is `to_hf` of the instantiated state dict to know which HF keys to
read, then `from_hf` of what was read. Saving in HF format
(`--checkpoint.last_save_in_hf`) is `to_hf` alone. The parity test calls
`from_hf` directly on the reference's state dict.

Examples in the tree, by what they show (`torchtitan/models/<name>/state_dict_adapter.py`):

```
qwen3          the from_hf_map pattern; per-expert HF weights stacked into w1_EFD / w2_EDF /
               w3_EFD; tied lm_head handled in both directions
llama3         dense; the q/k permutation between HF's and the native RoPE layout
               (_permute / _reverse_permute), the case the KL test in the conversion
               README was written for
muse_glimmer   the most recent; a vision tower mapped or dropped alongside the text
               decoder, and a module docstring that states the mapping and its decisions
               (the form to copy for the PR's key table)
```

How to write it, from `torchtitan/models/qwen3/state_dict_adapter.py`:

- A `from_hf_map` dict of abstract keys with `{}` for the layer index (and
  the expert index for per-expert HF weights), HF name -> torchtitan name,
  `None` for HF entries to drop (`rotary_emb.inv_freq`, non-text towers).
  `to_hf` inverts it.
- Layout differences are code, not map entries: per-expert HF tensors stacked
  into `w1_EFD` / `w2_EDF` / `w3_EFD` (`MoEStateDictAdapter` in
  `torchtitan/models/utils.py` has the split and concatenate helpers,
  DTensor-aware), fused vs separate QKV, tied `lm_head`
  (when `enable_weight_tying`, `from_hf` copies `embed_tokens` into `lm_head`
  if HF omits it and `to_hf` drops the duplicate), RoPE conventions
  (`_validate_hf_rope_config`, and the Llama-style permutation of q/k when the
  rotary layouts differ; the conversion README shows the KL test that caught it).
- Other sources: Megatron/NeMo adds TP shard concatenation order (column vs
  row parallel), PP stage offsets into layer indices, fused QKV interleave
  (per-head-grouped vs contiguous), fused gate/up, norm naming and bias
  presence; gather to one unsharded state dict on the reference side first,
  the adapter never handles sharding. MaxText/JAX adds pytree paths to module
  names, `[in, out]` vs `[out, in]` kernels, scanned stacks split per layer.
  Write the mapping table into the PR before writing the code. No Megatron or
  MaxText adapter exists in the tree yet; the first one is the template.
- Kernel-native layouts (`TpuGroupedExperts`) are translated in load/save
  hooks, not in the adapter, so checkpoint keys stay canonical; those hooks
  cover the model state dict only, optimizer state is keyed by live parameter
  names and shapes.

How it is checked, in this order:

1. The parity test (`tests.md`): `from_hf(reference.state_dict())` into
   `load_state_dict(strict=False)` with `missing` and `unexpected` both empty,
   then the forward/backward comparison. Wrong mapping, wrong layout and
   wrong model all fail here, together.
2. The round-trip test: `from_hf(to_hf(sd))` equals `sd` key for key and
   value for value on the tiny flavor (the Gemma 4 port's
   `TestGemma4StateDictAdapter`).
3. Before Step 3, on the real checkpoint:
   `scripts/checkpoint_conversion/numerical_tests_*.py` style KL divergence
   against the HF model (the README documents the method and the Llama RoPE
   permutation it caught), and a 10-step run with `initial_load_in_hf` whose
   step-1 loss is a sane pretrained loss, not an initialized one. Quote the
   KL in the PR.

## Registration

- Recipes return `TPUTrainerConfig`; derive variants from the parent
  (`cfg = <name>_debugmodel(); cfg.x = ...`). `c4_test_dataloader()` for debug
  recipes.
- `tpu/train.py`: import the package for side effects and add
  `model_spec.name` to the tuple that pops `tpu_compile_mode` under
  GraphTrainer; `train_minimal.py` needs the import too.
- GraphTrainer (rung 2): `train.py` needs a GraphTrainer registry per model
  (the `elif model_name == "<name>"` route). Reject the parallelisms the model
  does not support in `Model.Config.update_from_config`, which the trainer
  calls before either parallelize path, not in a parallelize wrapper; then a
  dense `Decoder` reuses the llama3 parallelization on both paths (the TPU
  llama3 `parallelize_llama` eager, upstream GraphTrainer's llama3
  `parallelize_llama` under GraphTrainer) and the registry is
  `dataclasses.replace(model_registry(flavor), parallelize_fn=<GraphTrainer
  parallelize>, pipelining_fn=None)`, a function next to `model_registry`.
  Only GraphTrainer-specific logic (uneven-vocab shard dims, split meshes)
  earns a `tpu/graph_trainer/<name>.py` like `qwen3.py` / `muse_glimmer.py`.
  Do not add a `GraphTrainer<Model>` subclass whose `init_states` runs under
  `disable_active_parametrization()`, although upstream's GraphTrainer models
  carry one: `init_states` writes parameters through `named_parameters()`,
  which bypasses SimpleFSDP's parametrization, so the wrapper changes nothing
  when every parameterized module has `param_init` (OLMo 2: bit-identical
  20-step losses with and without it). A module that falls back to
  `reset_parameters()` (it reads `self.weight`, which goes through the
  parametrization) is the case where it can matter; give that module a
  `param_init` instead.
- `bld_layout.py` (`[T, D]` viewed as `[B, L, D]` inside blocks) is on by
  default but applies only to `Llama3TransformerBlock` / `Qwen3TransformerBlock`
  running the stock forward (`_BLOCK_FORWARDS`); any other block class,
  including a subclass with its own forward, runs the `[T]` path silently
  (OLMo 2 does). Say which path the model runs in the PR; extending the match
  is a lever for later rungs, not rung 1. A non-stock `Decoder.forward` must
  make `tpu_model_class` raise rather than silently skip (`tpu/layer_loop.py`).
