"""Tests for the TPU [B, L] block layout (`experiments/tpu/components/bld_layout.py`)."""

import copy
import pickle
import types

from absl.testing import absltest
from absl.testing import parameterized
import torch

from torchtitan.experiments.tpu.components.bld_layout import apply_bld_layout
from torchtitan.experiments.tpu.components.bld_layout import bld_skip_reason
from torchtitan.experiments.tpu.components.bld_layout import cossin_rotate
from torchtitan.experiments.tpu.overrides.bf16_rope import tpu_activation_dtype_cossin_rope
from torchtitan.experiments.tpu.components.qkv import TpuPackedQKVConverter
from torchtitan.experiments.tpu.components.unflatten_shim import set_unflatten_seq_len
from torchtitan.models.common import ComplexRoPE
from torchtitan.models.common import CosSinRoPE
from torchtitan.models.llama3 import _build_llama3_layers
from torchtitan.models.qwen3 import _build_qwen3_layers

_DIM = 128
_HEAD_DIM = 32
_MAX_LEN = 64
_SEQ_LEN = 16
_BATCH = 2


def _cossin_rope():
  return CosSinRoPE.Config(
      dim=_HEAD_DIM, max_context_length=_MAX_LEN, theta=1000000.0
  )


def _qwen3_block(fuse_qkv: bool, rope=None):
  return _build_qwen3_layers(
      n_layers=1,
      dim=_DIM,
      n_heads=4,
      n_kv_heads=2,
      head_dim=_HEAD_DIM,
      hidden_dim=256,
      fuse_qkv=fuse_qkv,
      attn_backend="sdpa",
      rope=rope or _cossin_rope(),
  )[0]


def _llama3_block(fuse_qkv: bool):
  rope = ComplexRoPE.Config(
      dim=_HEAD_DIM, max_context_length=_MAX_LEN, theta=500000.0
  )
  return _build_llama3_layers(
      n_layers=1,
      dim=_DIM,
      n_heads=4,
      n_kv_heads=2,
      hidden_dim=256,
      fuse_qkv=fuse_qkv,
      attn_backend="sdpa",
      rope=rope,
  )[0]


def _packed_qwen3_block():
  return TpuPackedQKVConverter.Config().build().convert(_qwen3_block(True))


_BLOCKS = {
    "qwen3_qkv": lambda: _qwen3_block(False),
    "qwen3_fused": lambda: _qwen3_block(True),
    "qwen3_packed": _packed_qwen3_block,
    "qwen3_bf16_rope": lambda: _qwen3_block(
        False, tpu_activation_dtype_cossin_rope(_cossin_rope())
    ),
    "llama3_fused": lambda: _llama3_block(True),
    "llama3_qkv": lambda: _llama3_block(False),
}


def _build_pair(name: str):
  torch.manual_seed(0)
  upstream = _BLOCKS[name]().build()
  for p in upstream.parameters():
    torch.nn.init.normal_(p, std=0.05)
  bld = copy.deepcopy(upstream)
  return upstream, bld


class BLDLayoutTest(parameterized.TestCase):

  def test_cossin_rotate_is_bit_exact(self):
    torch.manual_seed(0)
    rope = _cossin_rope().build()
    positions = torch.cat([torch.arange(30), torch.arange(18)])
    for dtype in (torch.bfloat16, torch.float32):
      query = (torch.randn(48, 4, _HEAD_DIM) * 3).to(dtype)
      key = (torch.randn(48, 2, _HEAD_DIM) * 3).to(dtype)
      cache = rope._reshape_cache(query, positions)
      expected_q, expected_k = CosSinRoPE.apply_rotary_emb(query, key, cache)
      cos_h = cache[..., : _HEAD_DIM // 2]
      sin_h = cache[..., _HEAD_DIM : _HEAD_DIM + _HEAD_DIM // 2]
      torch.testing.assert_close(
          cossin_rotate(query, cos_h, sin_h), expected_q, rtol=0, atol=0
      )
      torch.testing.assert_close(
          cossin_rotate(key, cos_h, sin_h), expected_k, rtol=0, atol=0
      )

  @parameterized.named_parameters((name, name) for name in _BLOCKS)
  def test_block_matches_upstream(self, name):
    upstream, bld = _build_pair(name)
    self.assertEqual(apply_bld_layout(bld), 1)
    self.assertEqual(set(bld.state_dict()), set(upstream.state_dict()))

    x = torch.randn(_BATCH * _SEQ_LEN, _DIM, requires_grad=True)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)

    # Outside the trainer (no seq_len) the patched block is the upstream path.
    torch.testing.assert_close(
        bld(x, None, positions), upstream(x, None, positions), rtol=0, atol=0
    )

    with set_unflatten_seq_len(_SEQ_LEN):
      expected = upstream(x, None, positions)
      grads_expected = torch.autograd.grad(
          expected.square().sum(), [x, *upstream.parameters()]
      )
      actual = bld(x, None, positions)
      grads_actual = torch.autograd.grad(
          actual.square().sum(), [x, *bld.parameters()]
      )
    self.assertEqual(actual.shape, expected.shape)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for g_actual, g_expected in zip(grads_actual, grads_expected):
      torch.testing.assert_close(g_actual, g_expected, rtol=1e-4, atol=1e-4)

  def test_idempotent(self):
    _, bld = _build_pair("qwen3_qkv")
    self.assertEqual(apply_bld_layout(bld), 1)
    cls, attn_cls = type(bld), type(bld.attention)
    self.assertEqual(apply_bld_layout(bld), 1)
    self.assertIs(type(bld), cls)
    self.assertIs(type(bld.attention), attn_cls)

  def test_ineligible_block_is_untouched(self):
    _, block = _build_pair("qwen3_qkv")

    class CustomRoPE(type(block.attention.rope)):

      def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    block.attention.rope.__class__ = CustomRoPE
    cls = type(block)
    self.assertEqual(apply_bld_layout(block), 0)
    self.assertIs(type(block), cls)

  @parameterized.named_parameters(
      ("qkv", "qwen3_qkv"), ("fused", "llama3_fused"), ("packed", "qwen3_packed")
  )
  def test_qkv_subclass_swapped_later(self, name):
    # SimpleFSDP swaps parametrized modules to generated subclasses after the
    # patch runs (e.g. `SimpleFSDPQKVLinear_7`), as in the regular trainer.
    upstream, bld = _build_pair(name)
    self.assertEqual(apply_bld_layout(bld), 1)
    qkv = bld.attention.qkv_linear
    qkv.__class__ = type(f"Wrapped{type(qkv).__name__}", (type(qkv),), {})
    x = torch.randn(_BATCH * _SEQ_LEN, _DIM)
    positions = torch.arange(_SEQ_LEN).repeat(_BATCH)
    with set_unflatten_seq_len(_SEQ_LEN):
      torch.testing.assert_close(
          bld(x, None, positions),
          upstream(x, None, positions),
          rtol=1e-5,
          atol=1e-5,
      )

  def test_instance_forward_wrapper_is_untouched(self):
    # e.g. the graph trainer's `annotate_module_fqns`: a class swap after it
    # would not take effect, so the block must be left alone.
    _, block = _build_pair("qwen3_qkv")
    block.attention.forward = block.attention.forward
    cls = type(block)
    self.assertEqual(apply_bld_layout(block), 0)
    self.assertIs(type(block), cls)

  def test_generated_classes_pickle(self):
    _, bld = _build_pair("llama3_fused")
    apply_bld_layout(bld)
    # Classes pickle by qualified name; the graph trainer's GraphPickler needs
    # generated ones to resolve.
    for cls in (type(bld), type(bld.attention)):
      self.assertIs(pickle.loads(pickle.dumps(cls)), cls)


class BLDSkipReasonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("default", {}, False),
      ("context_parallel", {"context_parallel_degree": 2}, True),
      ("tensor_parallel", {"tensor_parallel_degree": 2}, True),
  )
  def test_skip_reason(self, parallelism_overrides, skipped):
    parallelism = types.SimpleNamespace(
        context_parallel_degree=1, tensor_parallel_degree=1
    )
    vars(parallelism).update(parallelism_overrides)

    reason = bld_skip_reason(parallelism)

    self.assertEqual(reason is not None, skipped, reason)


if __name__ == "__main__":
  absltest.main()
