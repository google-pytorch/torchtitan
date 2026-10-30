"""The scan offload has to engage on a real model, not just on a built graph.

A test that builds the fwd/bwd scan pair by hand passes whether or not a real
run offloads anything, because it builds the graph it wants to see. This one
trains llama3 for one step on TPU and asserts on the graph the trainer ended up
with.

That is the gap a real bug went through: a change to how `scan`'s backward is
patched left the backward scan node labelled with a stand-in object, so the pass
matched nothing, offloaded nothing, and the run ran out of memory with no error
anywhere.

The assertions run inside the spawned worker, because `gdist.torchrun` returns
nothing to the parent.
"""

from absl.testing import absltest
import torch
import torch.multiprocessing as mp
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.tpu import base_distributed_device_test
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import TpuGraphTrainer
import torchtitan.experiments.tpu.train


_SCAN = torch.ops.higher_order.scan
_OFFLOAD = torch.ops.ao.offload.default
_RELOAD = torch.ops.ao.reload.default


def _ao_targets(graph):
  """Every offload/reload op in one graph, in order."""
  return [n.target for n in graph.nodes if n.target in (_OFFLOAD, _RELOAD)]


def _body(gm, scan_node):
  """The loop body of a scan node, which the tracer stores as a submodule."""
  return getattr(gm, scan_node.args[0].target)


def _scans(gm):
  """The scan nodes, split into forward and backward."""
  forward, backward = [], []
  for node in gm.graph.nodes:
    if node.target is _SCAN:
      target = backward if node.meta.get("autograd_backward", False) else forward
      target.append(node)
  return forward, backward


def _check_offload_engaged(gm):
  """The pass offloaded the layer stack, one slice per iteration."""
  forward, backward = _scans(gm)
  assert forward and backward, (
      f"expected a forward and a backward scan, saw {len(forward)} and"
      f" {len(backward)} -- the pass has nothing to match"
  )

  # Recomputing a scan node reruns every iteration of the loop in backward:
  # exact loss, a better memory number, and twice the work.
  assert forward[0].meta.get("recompute") is CheckpointPolicy.MUST_SAVE, (
      f"the forward scan is tagged {forward[0].meta.get('recompute')}"
  )

  fwd_body = _body(gm, forward[0]).graph
  bwd_body = _body(gm, backward[0]).graph
  assert _OFFLOAD in _ao_targets(fwd_body), "the forward body does not offload"
  assert _RELOAD in _ao_targets(bwd_body), "the backward body does not reload"

  # Reloading the placeholder is legal and pulls the whole stack back on every
  # iteration -- worse than not offloading.
  reload_node = next(n for n in bwd_body.nodes if n.target is _RELOAD)
  assert reload_node.args[0].target is torch.ops.aten.index_select.default, (
      f"the reload is on {reload_node.args[0].target}, not on the slice"
  )

  # The selector has to prefer the layer stack over the loss logits, which are
  # the larger tensor at this size. Offloading those would ship a tensor the
  # backward reloads immediately.
  offload_node = next(n for n in fwd_body.nodes if n.target is _OFFLOAD)
  assert offload_node.meta["val"].dim() >= 2, (
      f"offloaded {tuple(offload_node.meta['val'].shape)}, which is not a stack"
  )

  # Around the loop rather than inside it means every row lands on device at
  # once, which costs the stack instead of saving it. The epilogue's read of the
  # last layer is the one legitimate reload out here, and it is a single row.
  row = offload_node.meta["val"].numel()
  assert _OFFLOAD not in _ao_targets(gm.graph), "the offload landed outside the loop"
  for node in gm.graph.nodes:
    if node.target is _RELOAD:
      assert node.meta["val"].numel() <= row, (
          f"a reload outside the loop takes {tuple(node.meta['val'].shape)},"
          " which is more than one row"
      )


def _train_and_check(config, check):
  """Trains, then checks the graph the trainer produced.

  `start_trainer` builds the trainer and drops it, so the only way to reach
  `_traced_step` is to catch the instance on its way past.
  """
  captured = []
  original = TpuGraphTrainer.train

  def train(self, *args, **kwargs):
    captured.append(self)
    return original(self, *args, **kwargs)

  TpuGraphTrainer.train = train
  try:
    torchtitan.experiments.tpu.train.start_trainer(config)
  finally:
    TpuGraphTrainer.train = original

  assert captured, "the graph trainer never ran"
  check(captured[0]._traced_step.gm)  # pylint: disable=protected-access


# Module level, not closures: the workers are spawned, so whatever is passed as
# `start_trainer` has to pickle by name.
def _start_and_expect_offload(config):
  """Trains with the policy and scan layers both on."""
  _train_and_check(config, _check_offload_engaged)


class ScanOffloadDistributedTest(
    base_distributed_device_test.BaseDistributedDeviceTest
):
  """Trains llama3 with offloading on, and reads the graph it produced."""

  def test_scan_offload(self):
    # Upstream replaced seq_len/local_batch_size with a token budget:
    # num_tokens_per_microbatch_per_dp_rank = local_batch_size * seq_len = 4 * 128.
    self._test_train_distributed(
        config_args=[
            "--module=torchtitan.experiments.tpu.llama3",
            "--config=graph_trainer_llama3_debugmodel",
            "--optimizer.implementation=foreach",
            "--dataloader.dataset_path=tests/assets/c4_test",
            "--dataloader.dataset=c4_test",
            "--hf_assets_path=tests/assets/tokenizer",
            "--training.max_context_length=128",
            "--training.steps=1",
            "--training.mixed_precision_param=float32",
            "--training.mixed_precision_reduce=float32",
            "--training.num_tokens_per_microbatch_per_dp_rank=512",
            "--splash_attention_kernel.use_splash_attention_kernel",
            "--parallelism.use_simple_fsdp",
            "--compile.use_scan_layers",
            "--compile.memory_policy=full_and_scan_offload",
        ],
        data_parallel_shard_degree=-1,
        tensor_parallel_degree=1,
        start_trainer=_start_and_expect_offload,  # pyrefly: ignore[bad-argument-type]
        enable_compile=True,
    )


if __name__ == "__main__":
  mp.set_start_method("spawn")  # pyrefly: ignore[missing-attribute]
  absltest.main()
