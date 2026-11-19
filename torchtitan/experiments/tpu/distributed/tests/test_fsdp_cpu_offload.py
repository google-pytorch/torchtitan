# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU unit tests for TPU FSDP2 CPU offload policy, shard cache, and grad accumulation."""

import importlib.abc
import importlib.machinery
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

from absl.testing import absltest
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

from torchtitan.config import ParallelismConfig, TrainingConfig
import torchtitan.distributed.fsdp as tt_fsdp
from torchtitan.experiments.tpu.distributed import fsdp_cpu_offload as offload_mod
from torchtitan.experiments.tpu.distributed.fsdp_cpu_offload import (
    TPUCPUOffloadPolicy,
    configure_tpu_fsdp_cpu_offload_from_job_config,
    get_cpu_offload_stats,
    install_tpu_cpu_offload_policy,
    install_tpu_fsdp_cpu_offload,
    is_tpu_fsdp_cpu_offload_installed,
    materialize_host_grads,
    reset_cpu_offload_stats,
    uninstall_tpu_cpu_offload_policy,
    uninstall_tpu_fsdp_cpu_offload,
    verify_tpu_fsdp_cpu_offload_no_pin_memory,
    wire_trainer_cpu_offload,
)
from torchtitan.experiments.tpu.tpu_job_config import (
    TPUConfig,
    TPUTrainerConfig,
    TpuTrainerCompileConfig,
)
from torchtitan.models.llama3.config_registry import model_registry


class _TorchTpuStubModule(ModuleType):
  """Minimal package stub so pure-CPU envs without torch_tpu can import train.py."""

  def __init__(self, name: str):
    super().__init__(name)
    self.__path__ = []

  def __getattr__(self, name: str) -> Any:
    if name.startswith("__"):
      raise AttributeError(name)
    val = mock.MagicMock(name=f"{self.__name__}.{name}")
    setattr(self, name, val)
    return val


class _TorchTpuStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):

  def find_spec(self, fullname: str, path: Any = None, target: Any = None):
    del path, target
    if fullname == "torch_tpu" or fullname.startswith("torch_tpu."):
      return importlib.machinery.ModuleSpec(
          fullname, self, is_package=True
      )
    return None

  def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
    return _TorchTpuStubModule(spec.name)

  def exec_module(self, module: ModuleType) -> None:
    del module


def _get_tpu_train_module() -> Any:
  """Imports ``torchtitan.experiments.tpu.train`` on TPU or CPU-only envs."""
  try:
    from torchtitan.experiments.tpu import train as tpu_train  # pylint: disable=g-import-not-at-top

    return tpu_train
  except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("torch_tpu"):
      raise
    finder = _TorchTpuStubFinder()
    saved_tpu_mods = {
        k: v for k, v in sys.modules.items() if k.startswith("torch_tpu")
    }
    sys.meta_path.append(finder)
    try:
      from torchtitan.experiments.tpu import train as tpu_train  # pylint: disable=g-import-not-at-top

      return tpu_train
    finally:
      sys.meta_path.remove(finder)
      for k in [m for m in sys.modules if m.startswith("torch_tpu")]:
        if k not in saved_tpu_mods:
          del sys.modules[k]


def _ensure_single_rank_gloo() -> DeviceMesh:
  os.environ.setdefault("TEST_TARGET", "1")
  if not dist.is_initialized():
    dist.init_process_group(
        backend="gloo",
        store=dist.HashStore(),
        rank=0,
        world_size=1,
    )
  return DeviceMesh("cpu", [0])


class _ResidualBlock(nn.Module):

  def __init__(self, dim: int, odd_hidden: int):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fc1 = nn.Linear(dim, odd_hidden, bias=True)
    self.fc2 = nn.Linear(odd_hidden, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))


class _ToyFSDPModel(nn.Module):

  def __init__(self, dim: int = 32, num_layers: int = 3):
    super().__init__()
    self.in_proj = nn.Linear(dim, dim, bias=False)
    self.layers = nn.ModuleList(
        [_ResidualBlock(dim, dim * 2 + 3) for _ in range(num_layers)]
    )
    self.out_norm = nn.LayerNorm(dim)
    self.head = nn.Linear(dim, dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.in_proj(x)
    for layer in self.layers:
      h = layer(h)
    return self.head(self.out_norm(h))


def _build_sharded_model(
    mesh: DeviceMesh,
    init_state: dict[str, torch.Tensor],
    *,
    reshard_after_forward: bool = True,
    freeze_in_proj: bool = False,
) -> _ToyFSDPModel:
  model = _ToyFSDPModel()
  model.load_state_dict(init_state)
  if freeze_in_proj:
    model.in_proj.weight.requires_grad_(False)
  mp_policy = MixedPrecisionPolicy(
      param_dtype=torch.bfloat16,
      reduce_dtype=torch.float32,
  )
  offload_policy = CPUOffloadPolicy(pin_memory=False)
  for layer in model.layers:
    fully_shard(
        layer,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
    )
  fully_shard(
      model,
      mesh=mesh,
      mp_policy=mp_policy,
      offload_policy=offload_policy,
      reshard_after_forward=reshard_after_forward,
  )
  return model


class FSDPCPUOffloadTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mesh = _ensure_single_rank_gloo()
    uninstall_tpu_fsdp_cpu_offload()
    reset_cpu_offload_stats()

  def tearDown(self):
    uninstall_tpu_fsdp_cpu_offload()
    reset_cpu_offload_stats()
    super().tearDown()

  def test_12_step_4_microbatch_bitwise_parity_and_counters(self):
    torch.manual_seed(2026)
    base_model = _ToyFSDPModel()
    init_state = {k: v.clone() for k, v in base_model.state_dict().items()}
    num_params = sum(1 for _ in base_model.parameters())

    num_steps = 12
    grad_accum = 4
    batches = [
        [
            (
                torch.randn(8, 32),
                torch.randn(8, 32),
            )
            for _ in range(grad_accum)
        ]
        for _ in range(num_steps)
    ]

    def _run_training(*, use_optimized: bool) -> list[dict[str, Any]]:
      if use_optimized:
        install_tpu_fsdp_cpu_offload(
            enable_param_cache=True,
            enable_device_grad_accum=True,
        )
      else:
        uninstall_tpu_fsdp_cpu_offload()
      reset_cpu_offload_stats()

      model = _build_sharded_model(
          self.mesh, init_state, reshard_after_forward=True
      )
      opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
      if use_optimized:
        wire_trainer_cpu_offload(opt)

      history: list[dict[str, Any]] = []
      for step in range(num_steps):
        opt.zero_grad(set_to_none=True)
        mb_losses: list[torch.Tensor] = []
        for x, target in batches[step]:
          pred = model(x)
          loss = torch.nn.functional.mse_loss(pred.float(), target) / grad_accum
          loss.backward()
          mb_losses.append(loss.detach().clone())

        if use_optimized:
          materialize_host_grads()
        grads_before_clip = {
            name: p.grad.to_local().detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        opt.step()
        params_after_step = {
            name: p.to_local().detach().clone()
            for name, p in model.named_parameters()
        }
        history.append(
            {
                "mb_losses": mb_losses,
                "grad_norm": grad_norm.to_local().detach().clone()
                if hasattr(grad_norm, "to_local")
                else grad_norm.detach().clone(),
                "grads": grads_before_clip,
                "params": params_after_step,
            }
        )
      return history

    ref_hist = _run_training(use_optimized=False)
    opt_hist = _run_training(use_optimized=True)
    stats = get_cpu_offload_stats()

    self.assertEqual(stats.param_cache_misses, num_params * num_steps)
    self.assertGreaterEqual(
        stats.param_cache_hits,
        num_params * num_steps * (2 * grad_accum - 1),
    )
    self.assertEqual(stats.d2h_sync_calls, num_steps)

    for step in range(num_steps):
      ref_s = ref_hist[step]
      opt_s = opt_hist[step]
      for mb in range(grad_accum):
        self.assertTrue(
            torch.equal(opt_s["mb_losses"][mb], ref_s["mb_losses"][mb]),
            f"Loss mismatch at step {step} mb {mb}",
        )
      self.assertTrue(
          torch.equal(opt_s["grad_norm"], ref_s["grad_norm"]),
          f"grad_norm mismatch at step {step}",
      )
      for name in ref_s["grads"]:
        self.assertTrue(
            torch.equal(opt_s["grads"][name], ref_s["grads"][name]),
            f"Grad mismatch for {name} at step {step}",
        )
      for name in ref_s["params"]:
        self.assertTrue(
            torch.equal(opt_s["params"][name], ref_s["params"][name]),
            f"Param mismatch for {name} at step {step}",
        )
      step_loss = sum(float(t.item()) for t in opt_s["mb_losses"])
      self.assertTrue(torch.isfinite(torch.tensor(step_loss)).item())

    # Verify same-batch training converges cleanly when trained on fixed data.
    uninstall_tpu_fsdp_cpu_offload()
    install_tpu_fsdp_cpu_offload(
        enable_param_cache=True,
        enable_device_grad_accum=True,
    )
    conv_model = _build_sharded_model(self.mesh, init_state)
    conv_opt = torch.optim.AdamW(conv_model.parameters(), lr=1e-2)
    wire_trainer_cpu_offload(conv_opt)
    x_fixed, y_fixed = batches[0][0]
    fixed_losses: list[float] = []
    for _ in range(10):
      conv_opt.zero_grad()
      for _ in range(2):
        loss = (
            torch.nn.functional.mse_loss(conv_model(x_fixed).float(), y_fixed)
            / 2
        )
        loss.backward()
      materialize_host_grads()
      torch.nn.utils.clip_grad_norm_(conv_model.parameters(), 1.0)
      conv_opt.step()
      fixed_losses.append(float(loss.detach().item()))
    self.assertLess(fixed_losses[-1], fixed_losses[0])

  def test_reshard_after_forward_false_bitwise_parity(self):
    torch.manual_seed(77)
    init_state = {
        k: v.clone() for k, v in _ToyFSDPModel().state_dict().items()
    }
    batches = [
        [(torch.randn(6, 32), torch.randn(6, 32)) for _ in range(3)]
        for _ in range(6)
    ]

    def _run(*, opt_mode: bool):
      if opt_mode:
        install_tpu_fsdp_cpu_offload(
            enable_param_cache=True,
            enable_device_grad_accum=True,
        )
      else:
        uninstall_tpu_fsdp_cpu_offload()
      model = _build_sharded_model(
          self.mesh, init_state, reshard_after_forward=False
      )
      opt = torch.optim.AdamW(model.parameters(), lr=3e-2)
      if opt_mode:
        wire_trainer_cpu_offload(opt)
      for step_batches in batches:
        opt.zero_grad()
        for x, y in step_batches:
          (torch.nn.functional.mse_loss(model(x).float(), y) / 3).backward()
        opt.step()
      return {k: p.to_local().clone() for k, p in model.named_parameters()}

    ref_params = _run(opt_mode=False)
    opt_params = _run(opt_mode=True)
    for name in ref_params:
      self.assertTrue(torch.equal(opt_params[name], ref_params[name]), name)

  def test_zero_grad_set_to_none_false_and_mid_step_materialize(self):
    torch.manual_seed(88)
    init_state = {
        k: v.clone() for k, v in _ToyFSDPModel().state_dict().items()
    }
    batches = [
        [(torch.randn(4, 32), torch.randn(4, 32)) for _ in range(4)]
        for _ in range(5)
    ]

    def _run(*, opt_mode: bool):
      if opt_mode:
        install_tpu_fsdp_cpu_offload(
            enable_param_cache=True,
            enable_device_grad_accum=True,
        )
      else:
        uninstall_tpu_fsdp_cpu_offload()
      model = _build_sharded_model(self.mesh, init_state)
      opt = torch.optim.AdamW(model.parameters(), lr=4e-2)
      if opt_mode:
        wire_trainer_cpu_offload(opt)
      for step_batches in batches:
        opt.zero_grad(set_to_none=False)
        for mb_idx, (x, y) in enumerate(step_batches):
          (torch.nn.functional.mse_loss(model(x).float(), y) / 4).backward()
          if opt_mode and mb_idx == 1:
            # Mid-step materialization followed by further microbatches.
            materialize_host_grads()
        opt.step()
      return {k: p.to_local().clone() for k, p in model.named_parameters()}

    ref_params = _run(opt_mode=False)
    opt_params = _run(opt_mode=True)
    for name in ref_params:
      self.assertTrue(torch.equal(opt_params[name], ref_params[name]), name)

  def test_cache_invalidation_on_load_state_dict_checkpointer_and_zero_grad(
      self,
  ):
    torch.manual_seed(101)
    init_a = {k: v.clone() for k, v in _ToyFSDPModel().state_dict().items()}
    torch.manual_seed(202)
    init_b = {k: v.clone() for k, v in _ToyFSDPModel().state_dict().items()}
    x = torch.randn(4, 32)
    y = torch.randn(4, 32)

    install_tpu_fsdp_cpu_offload(
        enable_param_cache=True,
        enable_device_grad_accum=True,
    )
    model = _build_sharded_model(self.mesh, init_a)
    inner_opt = torch.optim.SGD(model.parameters(), lr=0.25)

    # Wrap inner_opt in a container to test what happens if a caller steps
    # inner_opt directly (bypassing optimizers.step) before calling zero_grad().
    class _OptContainer:

      def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        inner_opt.zero_grad(*args, **kwargs)

      def step(self, *args: Any, **kwargs: Any) -> Any:
        return inner_opt.step(*args, **kwargs)

    opt_container = _OptContainer()

    # Populate parameter shard cache with init_a.
    _ = model(x)

    uninstall_tpu_fsdp_cpu_offload()
    ref_b = _build_sharded_model(self.mesh, init_b)
    expected_out_b = ref_b(x).detach().clone()

    install_tpu_fsdp_cpu_offload(
        enable_param_cache=True,
        enable_device_grad_accum=True,
    )
    ref_b_state = ref_b.state_dict()

    class _DummyCheckpointer:

      def load(self, step: int = -1) -> None:
        del step
        model.load_state_dict(ref_b_state)

    ckpt = _DummyCheckpointer()
    wire_trainer_cpu_offload(opt_container, ckpt)
    ckpt.load()
    actual_out_b = model(x).detach().clone()
    self.assertTrue(torch.equal(actual_out_b, expected_out_b))

    # Defensive zero_grad() invalidation: step inner_opt directly without going
    # through opt_container.step(), then call opt_container.zero_grad() and
    # confirm cached shards are invalidated before the next forward pass.
    loss = torch.nn.functional.mse_loss(model(x).float(), y)
    loss.backward()
    materialize_host_grads()
    inner_opt.step()  # bypasses opt_container.step()
    opt_container.zero_grad()

    uninstall_tpu_fsdp_cpu_offload()
    ref_after_direct_step = _build_sharded_model(self.mesh, init_b)
    ref_after_direct_step.load_state_dict(model.state_dict())
    expected_after_direct = ref_after_direct_step(x).detach().clone()

    install_tpu_fsdp_cpu_offload(
        enable_param_cache=True,
        enable_device_grad_accum=True,
    )
    actual_after_direct = model(x).detach().clone()
    self.assertFalse(torch.equal(actual_after_direct, expected_out_b))
    self.assertTrue(torch.equal(actual_after_direct, expected_after_direct))

    # Also test reset_sharded_param invalidation directly via assign=True reload.
    uninstall_tpu_fsdp_cpu_offload()
    ref_a = _build_sharded_model(self.mesh, init_a)
    expected_out_a = ref_a(x).detach().clone()
    install_tpu_fsdp_cpu_offload(
        enable_param_cache=True,
        enable_device_grad_accum=True,
    )
    model.load_state_dict(ref_a.state_dict(), assign=True)
    actual_out_a = model(x).detach().clone()
    self.assertTrue(torch.equal(actual_out_a, expected_out_a))

  def test_frozen_params_and_post_accumulate_grad_hooks_bypass(self):
    torch.manual_seed(303)
    init_state = {
        k: v.clone() for k, v in _ToyFSDPModel().state_dict().items()
    }
    batches = [(torch.randn(4, 32), torch.randn(4, 32)) for _ in range(3)]

    def _run(*, opt_mode: bool) -> tuple[int, dict[str, torch.Tensor]]:
      if opt_mode:
        install_tpu_fsdp_cpu_offload(
            enable_param_cache=True,
            enable_device_grad_accum=True,
        )
      else:
        uninstall_tpu_fsdp_cpu_offload()
      model = _build_sharded_model(
          self.mesh, init_state, freeze_in_proj=True
      )
      hook_calls = 0

      def _grad_hook(_param: torch.Tensor) -> None:
        nonlocal hook_calls
        hook_calls += 1

      handle = model.head.weight.register_post_accumulate_grad_hook(_grad_hook)
      try:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-2
        )
        if opt_mode:
          wire_trainer_cpu_offload(opt)
        opt.zero_grad()
        for x, y in batches:
          (torch.nn.functional.mse_loss(model(x).float(), y) / 3).backward()
        opt.step()
      finally:
        handle.remove()
      return (
          hook_calls,
          {k: p.to_local().clone() for k, p in model.named_parameters()},
      )

    ref_hooks, ref_params = _run(opt_mode=False)
    opt_hooks, opt_params = _run(opt_mode=True)
    self.assertEqual(opt_hooks, 3)
    self.assertEqual(opt_hooks, ref_hooks)
    for name in ref_params:
      self.assertTrue(torch.equal(opt_params[name], ref_params[name]), name)

  def test_policy_injection_opt_in_defaults_and_num_threads(self):
    self.assertFalse(TPUCPUOffloadPolicy().pin_memory)
    self.assertTrue(TPUCPUOffloadPolicy(pin_memory=True).pin_memory)
    self.assertIsInstance(TPUCPUOffloadPolicy(), CPUOffloadPolicy)

    # TPUConfig defaults both persistent-HBM optimizations to False (opt-in).
    default_tpu_cfg = TPUConfig()
    self.assertFalse(default_tpu_cfg.enable_cpu_offload_param_cache)
    self.assertFalse(default_tpu_cfg.enable_cpu_offload_device_grad_accum)

    orig_cls = tt_fsdp.CPUOffloadPolicy
    install_tpu_cpu_offload_policy(force=True)
    self.assertIs(tt_fsdp.CPUOffloadPolicy, TPUCPUOffloadPolicy)
    self.assertFalse(tt_fsdp.CPUOffloadPolicy().pin_memory)
    uninstall_tpu_cpu_offload_policy()
    self.assertIs(tt_fsdp.CPUOffloadPolicy, orig_cls)

    # SimpleFSDP disables enable_cpu_offload; FSDP2 preserves it and installs
    # TPUCPUOffloadPolicy(pin_memory=False) while keeping cache/accum off by default.
    simple_cfg = SimpleNamespace(
        training=SimpleNamespace(enable_cpu_offload=True),
        parallelism=SimpleNamespace(use_simple_fsdp=True),
        tpu_config=TPUConfig(),
    )
    self.assertFalse(configure_tpu_fsdp_cpu_offload_from_job_config(simple_cfg))
    self.assertFalse(simple_cfg.training.enable_cpu_offload)
    self.assertFalse(is_tpu_fsdp_cpu_offload_installed())

    default_fsdp2_cfg = SimpleNamespace(
        training=SimpleNamespace(enable_cpu_offload=True),
        parallelism=SimpleNamespace(use_simple_fsdp=False),
        tpu_config=None,
    )
    self.assertTrue(
        configure_tpu_fsdp_cpu_offload_from_job_config(default_fsdp2_cfg)
    )
    self.assertIs(tt_fsdp.CPUOffloadPolicy, TPUCPUOffloadPolicy)
    self.assertFalse(offload_mod._PARAM_CACHE_ENABLED)  # pylint: disable=protected-access
    self.assertFalse(offload_mod._DEVICE_GRAD_ACCUM_ENABLED)  # pylint: disable=protected-access

    prev_threads = torch.get_num_threads()
    try:
      opt_in_cfg = SimpleNamespace(
          training=SimpleNamespace(enable_cpu_offload=True),
          parallelism=SimpleNamespace(use_simple_fsdp=False),
          tpu_config=TPUConfig(
              enable_cpu_offload_param_cache=True,
              enable_cpu_offload_device_grad_accum=True,
              cpu_offload_num_threads=2,
          ),
      )
      self.assertTrue(configure_tpu_fsdp_cpu_offload_from_job_config(opt_in_cfg))
      self.assertTrue(offload_mod._PARAM_CACHE_ENABLED)  # pylint: disable=protected-access
      self.assertTrue(offload_mod._DEVICE_GRAD_ACCUM_ENABLED)  # pylint: disable=protected-access
      self.assertEqual(torch.get_num_threads(), 2)
    finally:
      torch.set_num_threads(prev_threads)

  def test_separate_graph_optimizer_compile_rejected_with_cpu_offload(self):
    # 1) Rejected at TPUTrainerConfig validation time.
    with self.assertRaisesRegex(
        ValueError, "optimizer_compile='separate_graph'"
    ):
      TPUTrainerConfig(
          model_spec=model_registry("debugmodel", attn_backend="sdpa"),
          training=TrainingConfig(enable_cpu_offload=True),
          parallelism=ParallelismConfig(use_simple_fsdp=False),
          compile=TpuTrainerCompileConfig(optimizer_compile="separate_graph"),
      )

    # 2) Also rejected at TPUTrainer.__init__ if mutated post-construction.
    tpu_train = _get_tpu_train_module()
    cfg = TPUTrainerConfig(
        model_spec=model_registry("debugmodel", attn_backend="sdpa"),
        training=TrainingConfig(enable_cpu_offload=True),
        parallelism=ParallelismConfig(use_simple_fsdp=False),
        compile=TpuTrainerCompileConfig(optimizer_compile="eager"),
    )
    cfg.compile.optimizer_compile = "separate_graph"
    with self.assertRaisesRegex(
        ValueError, "optimizer_compile='separate_graph'"
    ):
      tpu_train.TPUTrainer(cfg)

    # 3) Allowed when use_simple_fsdp=True (which disables enable_cpu_offload).
    simple_cfg = TPUTrainerConfig(
        model_spec=model_registry("debugmodel", attn_backend="sdpa"),
        training=TrainingConfig(enable_cpu_offload=True),
        parallelism=ParallelismConfig(use_simple_fsdp=True),
        compile=TpuTrainerCompileConfig(optimizer_compile="separate_graph"),
    )
    self.assertEqual(simple_cfg.compile.optimizer_compile, "separate_graph")

  def test_verify_tpu_fsdp_cpu_offload_no_pin_memory(self):
    good_model = _build_sharded_model(
        self.mesh, _ToyFSDPModel().state_dict()
    )
    verify_tpu_fsdp_cpu_offload_no_pin_memory([good_model])

    # Simulate a parallelize module calling fully_shard with stock
    # CPUOffloadPolicy(pin_memory=True) on a meta-device model before to_empty.
    with torch.device("meta"):
      bad_model = _ToyFSDPModel()
    fully_shard(
        bad_model,
        mesh=self.mesh,
        offload_policy=CPUOffloadPolicy(pin_memory=True),
    )
    with self.assertRaisesRegex(RuntimeError, "pin_memory=True"):
      verify_tpu_fsdp_cpu_offload_no_pin_memory([bad_model])

  def test_compiled_grad_norm_logs_once_under_cpu_offload(self):
    tpu_train = _get_tpu_train_module()
    dummy_model = nn.Linear(4, 4, bias=False)
    dummy_model.weight.grad = torch.ones_like(dummy_model.weight)
    fake_compiled = mock.Mock()
    trainer = object.__new__(tpu_train.TPUTrainer)
    trainer._compiled_grad_norm = fake_compiled  # pylint: disable=protected-access
    trainer._logged_cpu_offload_grad_norm_fallback = False  # pylint: disable=protected-access
    trainer.model_parts = [dummy_model]
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        training=SimpleNamespace(enable_cpu_offload=True, max_norm=1.0)
    )
    with (
        mock.patch.object(offload_mod.logger, "info"),
        mock.patch.object(tpu_train.logger, "info") as mock_train_info,
        mock.patch(
            "torchtitan.train.Trainer._clip_grad_norm",
            return_value=torch.tensor(1.5),
        ),
    ):
      norm1 = trainer._clip_grad_norm()  # pylint: disable=protected-access
      norm2 = trainer._clip_grad_norm()  # pylint: disable=protected-access
    self.assertEqual(float(norm1.item()), 1.5)
    self.assertEqual(float(norm2.item()), 1.5)
    fake_compiled.clip_grad_norm_.assert_not_called()
    self.assertEqual(mock_train_info.call_count, 1)

  def test_missing_cast_and_view_falls_back_gracefully_with_warning(self):
    from torch.distributed.fsdp._fully_shard import (  # pylint: disable=g-import-not-at-top
        _fsdp_collectives as fc,
    )

    torch.manual_seed(404)
    init_state = {
        k: v.clone() for k, v in _ToyFSDPModel().state_dict().items()
    }
    batches = [
        [(torch.randn(5, 32), torch.randn(5, 32)) for _ in range(3)]
        for _ in range(4)
    ]

    uninstall_tpu_fsdp_cpu_offload()
    ref_model = _build_sharded_model(self.mesh, init_state)
    ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=3e-2)
    for step_batches in batches:
      ref_opt.zero_grad()
      for x, y in step_batches:
        (torch.nn.functional.mse_loss(ref_model(x).float(), y) / 3).backward()
      ref_opt.step()
    ref_params = {k: p.to_local().clone() for k, p in ref_model.named_parameters()}

    orig_cav = getattr(fc, "_cast_and_view_sharded_grads", None)
    try:
      if orig_cav is not None:
        delattr(fc, "_cast_and_view_sharded_grads")
      with mock.patch.object(offload_mod.logger, "warning") as mock_warn:
        install_tpu_fsdp_cpu_offload(
            enable_param_cache=True,
            enable_device_grad_accum=True,
        )
      if orig_cav is not None:
        setattr(fc, "_cast_and_view_sharded_grads", orig_cav)
      mock_warn.assert_called_once()
      self.assertFalse(offload_mod._DEVICE_GRAD_ACCUM_ENABLED)  # pylint: disable=protected-access
      reset_cpu_offload_stats()
      opt_model = _build_sharded_model(self.mesh, init_state)
      opt_opt = torch.optim.AdamW(opt_model.parameters(), lr=3e-2)
      wire_trainer_cpu_offload(opt_opt)
      for step_batches in batches:
        opt_opt.zero_grad()
        for x, y in step_batches:
          (torch.nn.functional.mse_loss(opt_model(x).float(), y) / 3).backward()
        opt_opt.step()
      self.assertEqual(get_cpu_offload_stats().d2h_sync_calls, 0)
      opt_params = {
          k: p.to_local().clone() for k, p in opt_model.named_parameters()
      }
      for name in ref_params:
        self.assertTrue(torch.equal(opt_params[name], ref_params[name]), name)
    finally:
      uninstall_tpu_fsdp_cpu_offload()
      if orig_cav is not None:
        setattr(fc, "_cast_and_view_sharded_grads", orig_cav)

  def test_per_param_foreach_add_fallback_without_cat_and_four_arg_stub(self):
    model = _build_sharded_model(self.mesh, _ToyFSDPModel().state_dict())
    install_tpu_fsdp_cpu_offload(
        enable_param_cache=True,
        enable_device_grad_accum=True,
    )
    reset_cpu_offload_stats()

    called_with_four_args = False
    orig_hook = offload_mod._ORIG_CAST_AND_VIEW_SHARDED_GRADS  # pylint: disable=protected-access
    try:
      # Return mixed-dtype sharded grads with bfloat16 reduce_output so
      # _accumulate_sharded_grads_on_device exercises _PerParamGradAccum +
      # torch._foreach_add_ across multiple microbatches without torch.cat.
      def _four_arg_mixed_dtype_stub(
          reduce_output: torch.Tensor,
          fsdp_params: list[Any],
          padded_sharded_numels: list[int],
          consumer_stream: Any,
      ) -> list[torch.Tensor]:
        nonlocal called_with_four_args
        del consumer_stream
        called_with_four_args = True
        out: list[torch.Tensor] = []
        offset = 0
        for idx, (p, span) in enumerate(
            zip(fsdp_params, padded_sharded_numels)
        ):
          numel = p.sharded_size.numel()
          target_dtype = torch.float32 if idx % 2 == 0 else torch.bfloat16
          out.append(
              reduce_output.narrow(0, offset, numel)
              .to(target_dtype)
              .view(p.sharded_size)
          )
          offset += span
        return out

      offload_mod._ORIG_CAST_AND_VIEW_SHARDED_GRADS = (  # pylint: disable=protected-access
          _four_arg_mixed_dtype_stub
      )
      fsdp_state = model._get_fsdp_state()  # pylint: disable=protected-access
      fsdp_params = fsdp_state._fsdp_param_group.fsdp_params  # pylint: disable=protected-access
      for idx, p in enumerate(fsdp_params):
        if idx % 2 == 1 and hasattr(p.sharded_param, "grad_dtype"):
          p.sharded_param.grad_dtype = torch.bfloat16
      numels = [p._sharded_param_data.numel() for p in fsdp_params]  # pylint: disable=protected-access
      fake_reduce = torch.ones(sum(numels), dtype=torch.bfloat16)

      with mock.patch("torch.cat", side_effect=AssertionError("unexpected cat")):
        for _ in range(3):
          ret = offload_mod._patched_cast_and_view_sharded_grads(  # pylint: disable=protected-access
              fake_reduce, fsdp_params, numels, None
          )
          self.assertEqual(ret, [])

      self.assertTrue(called_with_four_args)
      materialize_host_grads()
      for idx, p in enumerate(fsdp_params):
        expected_dtype = torch.float32 if idx % 2 == 0 else torch.bfloat16
        self.assertIsNotNone(p.sharded_param.grad)
        self.assertTrue(
            torch.equal(
                p.sharded_param.grad.to_local(),
                torch.full(p.sharded_size, 3.0, dtype=expected_dtype),
            )
        )
    finally:
      offload_mod._ORIG_CAST_AND_VIEW_SHARDED_GRADS = orig_hook  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
