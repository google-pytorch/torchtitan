"""Single-chip parity of the TPU lane's HF backend against the raw HF model.

For each model family the lane builds the wrapped model on the accelerator
with its production choices -- splash attention (``attn_implementation=
splash_tpu``) and the fused Pallas cross entropy -- runs one forward/backward
on a real batch, and compares the loss and every parameter gradient against
the *original* ``transformers`` model class on CPU: same weights (copied from
the device), same tokens, positions expressed as a dense document mask, plain
cross entropy over the same valid tokens. This is the check that the lane
computes what HF computes -- mask semantics, positions, head, loss -- with
nothing from torchtitan on the reference side but the batch.

    python -m torchtitan.experiments.tpu.transformers_modeling_backend.tests.test_hf_reference_parity

Single-device and in-process, like the kernel tests: ``BaseAcceleratorDeviceTest``
owns the device (its ``setUp`` brings the TPU runtime up in this process).
The backend's Llama-like init patch is installed on ``PreTrainedModel``
itself, so ``tearDown`` restores HF's init hooks between model cases. The
multi-chip check (SimpleFSDP vs a CPU run of the same wrapper) is
``test_hf_backend_distributed``.
"""

import copy
import tempfile

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import torch
import torch.nn.functional as F

import torchtitan.config
from transformers.modeling_utils import PreTrainedModel
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.experiments.tpu import base_device_test
from torchtitan.experiments.tpu import test_utils
from torchtitan.experiments.tpu import train_minimal
from torchtitan.experiments.tpu.transformers_modeling_backend import (
    attention as splash_adapter,
)
from torchtitan.experiments.tpu.transformers_modeling_backend.model import (
    TPUHFTransformerModel,
)
from torchtitan.experiments.tpu.transformers_modeling_backend.tests.lane_args import (
    lane_config_args,
)


def _document_mask(positions: torch.Tensor, causal: bool, window: int | None = None) -> torch.Tensor:
  """Dense ``[1, 1, T, T]`` additive mask: same document, causal if asked,
  and within ``window`` keys of the query (HF's causal convention, q - k < W).

  ``positions`` restart at 0 at every packed-sample boundary; this is the
  mask the lane hands splash as segment ids (plus its causal flag), written
  out for an attention implementation that takes a 4D mask as-is.
  """
  doc = torch.cumsum((positions == 0).int(), dim=0)
  allowed = doc[:, None] == doc[None, :]
  idx = torch.arange(positions.shape[0])
  if causal:
    allowed &= idx[:, None] >= idx[None, :]
  if window is not None:
    allowed &= (idx[:, None] - idx[None, :]) < window
  mask = torch.zeros(allowed.shape, dtype=torch.float32)
  mask.masked_fill_(~allowed, float("-inf"))
  return mask[None, None]


def _reference_logits(reference, wrapper, input_ids, kwargs):
  """The logits the raw HF model produces for the wrapper's inputs.

  Mirrors what the wrapper trains through: the model's own head when it maps
  to the vocabulary, otherwise the trunk's hidden states through the head the
  lane attached (tied to the token embedding).
  """
  if wrapper._added_lm_head is None:
    return reference(input_ids=input_ids, **kwargs).logits
  hidden = reference.base_model(input_ids=input_ids, **kwargs).last_hidden_state
  return hidden @ reference.get_input_embeddings().weight.t()


def one_step_vs_reference(config_args: list[str], device: torch.device) -> dict:
  """One step on ``device``, the same step on the raw HF model on CPU."""
  config = torchtitan.config.ConfigManager().parse_args(config_args)
  config.dump_folder = tempfile.gettempdir()
  torch.manual_seed(config.debug.seed or 42)
  trainer = train_minimal.TrainerMinimal(device=device, rank=0, world_size=1, job_config=config)
  wrapper, hf = trainer.model, trainer.model.model
  assert isinstance(wrapper, TPUHFTransformerModel), type(wrapper)

  batch = dict(next(iter(trainer.dataloader)))
  valid = batch.pop("num_valid_tokens", None)
  batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
  tokens, labels, extra = wrapper.preprocess_inputs(
      batch, parallel_dims=trainer.parallel_dims, parallelism=config.parallelism
  )
  if valid is None:
    valid = (labels != IGNORE_INDEX).sum()
  valid = torch.as_tensor(valid, dtype=torch.int64, device=device)

  wrapper.train()
  splash_adapter._reported.clear()  # per-process report set; start this case clean
  pred = wrapper(tokens, **extra)
  loss, _ = trainer.loss_fn(pred, labels, valid)
  loss.backward()
  # The adapter reports once per process: "splash_tpu fired" when the kernel
  # ran, a reason string for every distinct fallback to sdpa. Eager mode
  # (compile is off in this test) is what makes these reports fire.
  reports = set(splash_adapter._reported)

  # The reference: the same HF class on CPU with the portable attention.
  ref_config = copy.deepcopy(hf.config)
  # HF's SDPA integration silently ignores an attention-logit soft-cap
  # (Gemma-2), so a model with one is referenced against HF's eager attention,
  # which applies it.
  text_ref = ref_config.get_text_config() if hasattr(ref_config, "get_text_config") else ref_config
  # Attention sinks (gpt-oss, Flint) likewise exist only in HF's eager path.
  has_sinks = any(n.endswith(".sinks") for n, _ in hf.named_parameters())
  reference_attn = (
      "eager" if getattr(text_ref, "attn_logit_softcapping", None) or has_sinks else "sdpa"
  )
  TPUHFTransformerModel._set_attn_everywhere(ref_config, reference_attn)
  reference = type(hf)(config=ref_config)
  missing, unexpected = reference.load_state_dict(
      {k: v.detach().float().cpu() for k, v in hf.state_dict().items()}, strict=False
  )
  tied = set(getattr(reference, "_tied_weights_keys", None) or [])
  missing = [m for m in missing if not any(m.endswith(t) for t in tied)]
  assert not unexpected and not missing, (unexpected[:5], missing[:5])
  reference.float().eval()  # dropout is zero on the device side; eval() is that with no RNG

  positions = extra.get("positions")
  positions = positions.cpu() if positions is not None else torch.arange(tokens.shape[0])
  causal = any(getattr(m, "is_causal", False) for m in hf.modules())
  ref_mask = _document_mask(positions, causal)
  # HF uses an explicit 4D mask as-is in every layer, so a model with
  # sliding-window layers (Flint, ...) gets HF's per-layer-type mapping instead:
  # without it the reference silently drops the window the device applies.
  window = getattr(text_ref, "sliding_window", None)
  if isinstance(window, int) and "sliding_attention" in (getattr(text_ref, "layer_types", None) or ()):
    ref_mask = {"full_attention": ref_mask,
                "sliding_attention": _document_mask(positions, causal, window)}
  ref_kwargs = {"position_ids": positions[None], "attention_mask": ref_mask}
  if wrapper._forward_kwargs is not None:
    ref_kwargs = {k: v for k, v in ref_kwargs.items() if k in wrapper._forward_kwargs}
  logits = _reference_logits(reference, wrapper, tokens.cpu()[None], ref_kwargs)
  ref_loss = F.cross_entropy(
      logits.float().squeeze(0), labels.cpu(), ignore_index=IGNORE_INDEX, reduction="sum"
  ) / valid.cpu()
  ref_loss.backward()

  return {
      "model_class": type(hf).__name__,
      "attn_impl": getattr(hf.config, "_attn_implementation", None),
      "splash_fired": "splash_tpu fired" in reports,
      "fallbacks": sorted(r for r in reports if r not in splash_adapter.INFO_REPORTS),
      "fused_loss": wrapper._fused_loss,
      "fused_operands": isinstance(pred, tuple),  # (hidden, head weight) for the fused CE
      "n_valid": int(valid.cpu()),
      "device_loss": loss.detach().float().cpu(),
      "ref_loss": ref_loss.detach(),
      "device_grads": {n: p.grad.detach().float().cpu() for n, p in hf.named_parameters() if p.grad is not None},
      "ref_grads": {n: p.grad.detach().float() for n, p in reference.named_parameters() if p.grad is not None},
  }


# HF's own init hooks, before any model case installs the backend's Llama-like
# patch on PreTrainedModel (process-wide); restored between cases.
_HF_INIT_HOOKS = {
    name: getattr(PreTrainedModel, name)
    for name in ("_init_weights", "_initialize_weights")
}


class HFReferenceParityTest(base_device_test.BaseAcceleratorDeviceTest):
  """One chip, splash + fused CE, vs the raw transformers model on CPU."""

  def setUp(self):
    super().setUp()
    if self.accelerator_device.type != "tpu":
      self.skipTest("This parity test targets the TPU kernels.")

  def tearDown(self):
    for name, hook in _HF_INIT_HOOKS.items():
      setattr(PreTrainedModel, name, hook)
    super().tearDown()

  @parameterized.named_parameters([
      # Decoders: GQA + RoPE (Qwen3), the lane's frontier family (Llama).
      dict(testcase_name="qwen3", config="hf_debugmodel_max",
           hf_model="Qwen/Qwen3-0.6B", args=["--loss.b_block_size=256"]),
      dict(testcase_name="llama3", config="hf_debugmodel_max",
           hf_model="unsloth/Llama-3.2-1B-Instruct", args=["--loss.b_block_size=256"]),
      # A soft-capped head on the fused CE: the model hands the cap to the loss,
      # which re-applies it in-kernel, so loss and gradients must match HF's own
      # capped logits. The checkpoint's cap (30) is overridden to 0.5 on BOTH
      # sides: at random init logits are small and 30 * tanh(x / 30) ~= x, so a
      # cap of 30 would let a dropped cap pass unnoticed.
      # Gemma-2 caps both the attention logits (checkpoint: 50) and the final
      # logits (30). At random init both caps are near-identity (logits are
      # small and cap * tanh(x / cap) ~= x), which would let a dropped cap pass:
      # so on both sides the caps are tightened AND query_pre_attn_scalar is
      # shrunk (Gemma-2 scales q by its -1/2 power: 0.01 -> x10) to push the
      # attention logits far into the capped range. The attention cap reaches
      # the kernel through HF's `softcap` kwarg.
      dict(testcase_name="gemma2_attn_and_final_softcap", config="hf_debugmodel_max",
           hf_model="google/gemma-2-2b",
           args=["--loss.b_block_size=256",
                 '--hf_config_overrides={"attn_logit_softcapping": 2.0, "final_logit_softcapping": 0.5, "query_pre_attn_scalar": 0.01}'],
           # bf16 attention through a deliberately saturated cap moves the tied
           # embedding's gradient by ~1.7e-2 (the kernel-level soft-cap test pins
           # the arithmetic itself in fp32). Detection does not rest on grads:
           # with the cap dropped the LOSS is off by 6.2e-2 (tolerance 1e-2).
           grad_atol=2e-2, grad_rtol=5e-2),
      # Flint: attention sinks (s_aux) take the eager fallback; the shrunk
      # config keeps one dense and one MoE layer (sigmoid router + shared
      # expert, HF's own MoE block under this dense config), a sliding window,
      # and one RoPE / one no-RoPE layer.
      dict(testcase_name="flint_attention_sinks", config="hf_debugmodel_max",
           hf_model="RecursiveSI/flint",
           args=["--loss.b_block_size=256",
                 '--hf_config_overrides={"num_experts": 8, "num_experts_per_tok": 2, '
                 '"moe_intermediate_size": 64, "shared_expert_intermediate_size": 128, '
                 '"mlp_only_layers": [0], "no_rope_layer_indices": [0], "sliding_window": 33, '
                 '"layer_types": ["sliding_attention", "full_attention"]}'],
           expected_fallbacks=("attention sinks (s_aux): the kernel has no sink term",),
           # At init the sinks are small (std 0.02): dropping them moves the
           # worst gradient by ~1.4e-3, inside the default 5e-3. The correct
           # path differs by ~1.5e-4, so a tighter bound catches a wrong sink term.
           grad_atol=5e-4, min_transformers="5.21"),
      dict(testcase_name="gemma3_softcap_fused", config="hf_debugmodel_max",
           hf_model="google/gemma-3-1b-it",
           args=["--loss.b_block_size=256",
                 '--hf_config_overrides={"final_logit_softcapping": 0.5}']),
      # GPT-2 at full size: learned positions, tied head, `attn` block naming.
      dict(testcase_name="distilgpt2", config="hf_full_max_plain_ce",
           hf_model="distilbert/distilgpt2",
           args=["--compile.no-enable",
                 '--hf_config_overrides={"attn_pdrop": 0, "resid_pdrop": 0, "embd_pdrop": 0}']),
      # MoE checkpoint through splash. The Titan MoE swap happens in
      # parallelize_fn, which TrainerMinimal runs only for world_size > 1, so
      # on one chip HF's own MixtralSparseMoeBlock runs on the device too:
      # splash + HF MoE vs HF on CPU, not the Titan MoE. Routing is top-k on a
      # random-init model, so a bf16 attention difference can flip a near-tie
      # assignment and move a handful of gradient entries by ~1e-2 (measured 4
      # of 2M embedding entries); the loss itself agrees to ~2e-3.
      dict(testcase_name="tiny_mixtral_moe", config="hf_moe_auto",
           hf_model="hf-internal-testing/tiny-random-MixtralForCausalLM",
           grad_atol=2e-2, grad_rtol=1e-1),
      # GPT-2 with its shipped attention dropout KEPT (attn_pdrop=0.1; Granite
      # and Seed-Coder also train with 0.1): the in-kernel PRNG dropout path,
      # where the adapter used to fall back to sdpa. The reference runs
      # undropped, so this case checks the path -- the kernel fired, no
      # fallback, loss finite and near the undropped value -- not the numbers;
      # dropout's own parity is the kernel test
      # (test_splash_bias_dropout.test_segment_ids_with_dropout).
      dict(testcase_name="distilgpt2_with_dropout", config="hf_full_max_plain_ce",
           hf_model="distilbert/distilgpt2",
           args=["--compile.no-enable",
                 '--hf_config_overrides={"resid_pdrop": 0, "embd_pdrop": 0}'],
           loss_atol=0.5, loss_rtol=0.05, check_grads=False),
  ])
  def test_one_step_matches_hf(
      self, config, hf_model, args=None, loss_atol=1e-2, loss_rtol=1e-3,
      grad_atol=5e-3, grad_rtol=5e-3, check_grads=True, expected_fallbacks=(),
      min_transformers=None,
  ):
    if min_transformers is not None:
      import transformers
      from packaging import version
      if version.parse(transformers.__version__) < version.parse(min_transformers):
        self.skipTest(f"{hf_model} needs transformers>={min_transformers}")
    # Eager: the adapter's fallback report fires outside Dynamo tracing, so a
    # kernel that did not run is visible rather than silently measured as sdpa.
    result = one_step_vs_reference(
        lane_config_args(config, hf_model, steps=1, extra=["--compile.no-enable"] + (args or [])),
        self.accelerator_device,
    )

    self.assertEqual(result["attn_impl"], "splash_tpu", "the device run did not select splash")
    # A model the kernel cannot express (attention sinks) must take the
    # adapter's eager fallback and say so; every other case must fire the kernel.
    self.assertEqual(result["fallbacks"], sorted(expected_fallbacks),
                     "unexpected splash adapter fallbacks")
    if not expected_fallbacks:
      self.assertTrue(result["splash_fired"], "the splash kernel never ran")
    self.assertEqual(result["fused_operands"], result["fused_loss"],
                     "fused-loss config and the model's output disagree")
    logging.info(
        "%s: device loss %.6f, reference loss %.6f (fused_loss=%s, %d valid tokens)",
        result["model_class"], result["device_loss"], result["ref_loss"],
        result["fused_loss"], result["n_valid"],
    )
    labels = dict(test_label="TPU", ref_label=f"raw {result['model_class']} on CPU")
    test_utils.check_equivalence(
        result["device_loss"], result["ref_loss"], atol=loss_atol, rtol=loss_rtol,
        check_name="Loss", **labels,
    )
    if not check_grads:
      return
    device_grads, ref_grads = result["device_grads"], result["ref_grads"]
    self.assertEqual(set(device_grads), set(ref_grads), "gradient sets differ")
    worst = max((float((device_grads[n] - ref_grads[n]).abs().max()), n) for n in ref_grads)
    logging.info("largest gradient difference %.3e on %s", *worst)
    for name in sorted(ref_grads):
      test_utils.check_equivalence(
          device_grads[name], ref_grads[name], atol=grad_atol, rtol=grad_rtol,
          check_name=f"Gradient ({name})", **labels,
      )


if __name__ == "__main__":
  absltest.main()
