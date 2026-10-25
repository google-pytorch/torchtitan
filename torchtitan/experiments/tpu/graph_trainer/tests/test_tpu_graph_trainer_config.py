# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for TPU GraphTrainer config conversion."""

import dataclasses
import typing
from types import SimpleNamespace

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import nn
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import apply_overrides
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.graph_trainer import llama3 as llama3_gt
from torchtitan.experiments.graph_trainer import qwen3 as qwen3_gt
from torchtitan.experiments.graph_trainer.qwen3.model import GraphTrainerQwen3Model
from torchtitan.experiments.tpu import tpu_flags
from torchtitan.experiments.tpu.graph_trainer.config import (
    to_tpu_graph_trainer_config,
)
from torchtitan.experiments.tpu.graph_trainer.graph_split import GraphSplitter
from torchtitan.experiments.tpu.graph_trainer.tpu_backend_pass import (
    construct_tpu_pass_pipeline,
)
from torchtitan.experiments.tpu.graph_trainer.tpu_graph_trainer import (
    TpuGraphTrainer,
)
from torchtitan.experiments.tpu.graph_trainer.qwen3 import (
    MaxTextTpuGraphTrainerQwen3Model,
    model_registry as tpu_qwen3_graph_registry,
)
from torchtitan.experiments.tpu.layer_loop import (
    builds_layer_loop,
    runs_layer_loop,
)
from torchtitan.experiments.tpu.llama3.config_registry import (
    llama3_1b,
    llama3_8b,
    llama3_8b_scan,
    llama3_70b,
    llama3_70b_scan,
    llama3_405b,
    llama3_405b_scan,
    llama3_debugmodel,
    sft_llama3_8b_metamath,
)
from torchtitan.experiments.tpu.loss import PallasCrossEntropyLoss
from torchtitan.experiments.tpu.overrides.bf16_rope import (
    TpuActivationDtypeCosSinRoPE,
)
from torchtitan.experiments.tpu.overrides.fused_loss import (
    FusedLossLinearWrapper,
    fused_loss_lm_head,
)
from torchtitan.experiments.tpu.overrides.real_rope import (
    TpuRealArithmeticComplexRoPE,
)
from torchtitan.experiments.tpu.overrides.discarded_token_counter import (
    TpuDiscardedTokenCounterMoE,
)
from torchtitan.experiments.tpu.qkv import TpuPackedQKVLinear
from torchtitan.experiments.tpu.qwen3 import config_registry as qwen3_config_registry
from torchtitan.experiments.tpu.qwen3.config_registry import (
    qwen3_8b,
    qwen3_8b_maxtext_parity,
    qwen3_8b_zero3,
    qwen3_32b,
    qwen3_32b_4x8x8,
    qwen3_32b_maxtext_parity,
    qwen3_moe_30b,
    qwen3_moe_30b_fsdp_only_attention_parity_benchmark,
    qwen3_moe_30b_fsdp_only_full_ac_benchmark,
    qwen3_moe_30b_fsdp_only_full_ac_profile,
    qwen3_moe_30b_fsdp_only_maxtext_benchmark,
    qwen3_moe_235b_fsdp_activation_offload_4x4x4,
    qwen3_moe_235b_fsdp_activation_offload_maxtext_4x4x4,
    qwen3_moe_235b_scan,
    qwen3_moe_235b_scan_maxtext,
    qwen3_moe_235b_scan_proxy,
)
from torchtitan.experiments.tpu.token_dispatcher import (
    TpuAllToAllTokenDispatcher,
)
from torchtitan.experiments.tpu.tpu_job_config import TPUConfig, TPUTrainerConfig
from torchtitan.models.common import ComplexRoPE, Linear
from torchtitan.models.common.moe import MoE, MoELoadBalanceAuxLoss
from torchtitan.models.llama3.config_registry import (
    model_registry as llama3_model_registry,
)


class TpuGraphTrainerConfigTest(parameterized.TestCase):

  def _qwen3_graph_flops(self, config) -> int:
    """GraphTrainer FLOPs/token at sequence length 2048."""
    graph_config = to_tpu_graph_trainer_config(
        config, tpu_qwen3_graph_registry, target_cls=TpuGraphTrainer.Config
    )
    with torch.device("meta"):
      model = graph_config.model_spec.model.build()
    self.assertIsInstance(model, GraphTrainerQwen3Model)
    return graph_config.model_spec.model.get_nparams_and_flops(
        model, seq_len=2048
    )[1]

  def test_qwen3_graph_config_reports_torchtitan_flops_by_default(
      self,
  ) -> None:
    self.assertEqual(self._qwen3_graph_flops(qwen3_moe_30b()), 23_083_044_864)

  def test_maxtext_recipe_reports_maxtext_flops_on_graph_trainer(self) -> None:
    """A ``*maxtext*`` recipe keeps MaxText FLOPs through the conversion.

    Same model as the test above, so the values differ only by convention.
    """
    self.assertEqual(
        self._qwen3_graph_flops(qwen3_moe_30b_fsdp_only_maxtext_benchmark()),
        20_665_860_096,
    )

  def test_maxtext_scan_recipe_reports_maxtext_flops_on_graph_trainer(
      self,
  ) -> None:
    """The scan rewrite keeps the MaxText class picked before it."""
    graph_config = self._qwen3_graph_config(qwen3_moe_235b_scan_maxtext())
    self.assertIsInstance(
        graph_config.model_spec.model,
        MaxTextTpuGraphTrainerQwen3Model.Config,
    )
    self.assertTrue(builds_layer_loop(graph_config.model_spec.model))

  @parameterized.named_parameters(
      ("torchtitan_flops", qwen3_moe_235b_scan_proxy),
      ("maxtext_flops", qwen3_moe_235b_scan_maxtext),
  )
  def test_scanned_moe_reports_the_same_flops_as_unstacked(self, recipe):
    """Stacking the layer weights must not change the FLOP estimate.

    Scan moves expert weights out of their MoE modules; upstream's active-
    parameter count would then miss the top-k discount and count every expert.
    """
    config = recipe()
    unstacked = config.model_spec.model
    scanned = self._qwen3_graph_config(config).model_spec.model
    self.assertTrue(builds_layer_loop(scanned))
    with torch.device("meta"):
      unstacked_model, scanned_model = unstacked.build(), scanned.build()
    self.assertEqual(
        scanned.get_nparams_and_flops(scanned_model, seq_len=4096),
        unstacked.get_nparams_and_flops(unstacked_model, seq_len=4096),
    )

  def _qwen3_graph_config(self, config):
    return to_tpu_graph_trainer_config(
        config, tpu_qwen3_graph_registry, target_cls=TpuGraphTrainer.Config
    )

  @parameterized.parameters(False, True)
  def test_scanning_converts_the_model_and_nothing_else_does(
      self, scanned: bool
  ) -> None:
    """Scanning needs a model that runs its layers container.

    The conversion has to happen here and not on the recipe: the graph-trainer
    registry rebuilds the model config from a fixed class, so a converter the
    recipe applied is already gone by this point.
    """
    config = qwen3_8b()
    config.tpu_config.use_scan_layers = scanned
    graph_config = self._qwen3_graph_config(config)

    plain = self._qwen3_graph_config(qwen3_8b()).model_spec.model
    with torch.device("meta"):
      model = graph_config.model_spec.model.build()
      plain_model = plain.build()

    self.assertEqual(runs_layer_loop(model), scanned)
    # The MaxText FLOP accounting the registry installed must survive it.
    self.assertEqual(
        graph_config.model_spec.model.get_nparams_and_flops(model, seq_len=2048),
        plain.get_nparams_and_flops(plain_model, seq_len=2048),
    )

  def test_scanning_allows_the_moe_recipe(self) -> None:
    """Scanning refuses MoE load balancing, which this recipe does not use.

    The only MoE recipe on this lane is the global-aux one, so it sets no
    `load_balance_coeff`: there is no `expert_bias_E` and nothing reads
    `tokens_per_expert_E`, and sharing layer 0's buffers changes nothing. The
    refusal itself is covered in ``test_layer_stack.py`` -- it cannot be reached
    from here, because turning the coefficient on for this recipe trips the
    aux-loss mutual exclusion first.
    """
    config = qwen3_moe_30b()
    config.tpu_config.use_scan_layers = True
    # The recipe runs expert parallel, which is refused on its own; turn it off
    # so this reaches the MoE check.
    config.parallelism.expert_parallel_degree = 1

    with torch.device("meta"):
      self._qwen3_graph_config(config).model_spec.model.build()

  def test_scanning_refuses_expert_parallelism(self) -> None:
    config = qwen3_8b()
    config.tpu_config.use_scan_layers = True
    config.parallelism.expert_parallel_degree = 2

    with self.assertRaisesRegex(ValueError, "expert-parallel-degree"):
      self._qwen3_graph_config(config)

  def test_scanning_needs_the_graph_trainer(self) -> None:
    """Off the graph-trainer lane nothing applies the flag, so it must not be
    silently accepted."""
    # Upstream d771abb0 made `model_spec` required; the flag check under test
    # does not look at it.
    with self.assertRaisesRegex(ValueError, "use-graph-trainer"):
      TPUTrainerConfig(
          model_spec=llama3_model_registry("debugmodel", attn_backend="sdpa"),
          tpu_config=TPUConfig(use_scan_layers=True, use_graph_trainer=False),
      )

  @parameterized.parameters("sac_and_offload", "full_and_offload")
  def test_scanning_rejects_generic_offload_policy(self, policy: str) -> None:
    config = qwen3_8b()
    config.tpu_config.use_scan_layers = True
    config.tpu_config.graph_trainer_memory_policy = typing.cast(
        typing.Any, policy
    )

    with self.assertRaisesRegex(ValueError, "full_and_scan_offload"):
      self._qwen3_graph_config(config)

  @parameterized.parameters(
      "default", "full", "eager", "sac_and_offload", "full_and_offload"
  )
  def test_cli_memory_policy_is_propagated(self, policy: str) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            f"--tpu_config.graph_trainer_memory_policy={policy}",
        ]),
    )
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(config.tpu_config.graph_trainer_memory_policy, policy)
    self.assertEqual(graph_config.compile.memory_policy, policy)

  def test_cpu_offload_controls_are_propagated(self) -> None:
    config = qwen3_8b()
    config.tpu_config.graph_trainer_memory_policy = "full_and_offload"
    config.tpu_config.graph_trainer_cpu_offload_budget_gb = 4.5

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(graph_config.compile.memory_policy, "full_and_offload")
    self.assertEqual(graph_config.compile.cpu_offload_budget_gb, 4.5)
    passes = construct_tpu_pass_pipeline(
        typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
        graph_config,
    )
    pass_names = [
        getattr(graph_pass, "func", graph_pass).__name__
        for graph_pass in passes
    ]
    self.assertLess(
        pass_names.index("tag_with_memory_policy_pass"),
        pass_names.index("apply_cpu_offload_pass"),
    )
    self.assertLess(
        pass_names.index("apply_cpu_offload_pass"),
        pass_names.index("selective_activation_remat_pass"),
    )

  def test_cpu_offload_prefetch_requires_backward_split(self) -> None:
    config = qwen3_8b()
    config.tpu_config.graph_split.cpu_offload_prefetch_executables = 1

    with self.assertRaisesRegex(ValueError, "requires a split backward graph"):
      GraphSplitter.from_trainer_config(
          self._qwen3_graph_config(config),
          optimizer_in_graph=False,
      )

    config.tpu_config.use_graph_split = True
    config.tpu_config.graph_split.forward_layers_per_executable = 1
    with self.assertRaisesRegex(ValueError, "requires a split backward graph"):
      GraphSplitter.from_trainer_config(
          self._qwen3_graph_config(config),
          optimizer_in_graph=False,
      )

    config.tpu_config.graph_split.backward_partition_mode = "by_layer"
    splitter = GraphSplitter.from_trainer_config(
        self._qwen3_graph_config(config),
        optimizer_in_graph=False,
    )
    self.assertTrue(splitter.enabled)

  def test_negative_cpu_offload_budget_is_rejected(self) -> None:
    config = qwen3_8b()
    config.tpu_config.graph_trainer_cpu_offload_budget_gb = -0.1

    with self.assertRaisesRegex(ValueError, "must be nonnegative"):
      to_tpu_graph_trainer_config(
          config,
          qwen3_gt.model_registry,
          target_cls=TpuGraphTrainer.Config,
      )

  def test_tpu_selective_cli_is_propagated(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.graph_trainer_memory_policy=tpu_selective",
            "--tpu_config.graph_trainer_save_regions",
            "moe.routing",
            "--tpu_config.graph_trainer_save_last_n_layers",
            "attention.inner_attention",
            "4",
        ]),
    )
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(graph_config.compile.memory_policy, "tpu_selective")
    self.assertEqual(
        graph_config.tpu_config.graph_trainer_save_regions, ("moe.routing",)
    )
    self.assertEqual(
        graph_config.tpu_config.graph_trainer_save_last_n_layers,
        {"attention.inner_attention": 4},
    )

  @parameterized.named_parameters(
      ("no_regions", "tpu_selective", (), {}, False, "needs a region"),
      ("unknown_region", "tpu_selective", ("moe.nope",), {}, False, "Unknown"),
      (
          "region_in_both_fields",
          "tpu_selective",
          ("attention.inner_attention",),
          {"attention.inner_attention": 2},
          False,
          "in both",
      ),
      (
          "zero_layers",
          "tpu_selective",
          (),
          {"attention.inner_attention": 0},
          False,
          "positive integer",
      ),
      (
          "scan",
          "tpu_selective",
          ("moe.routing",),
          {},
          True,
          "use-scan-layers",
      ),
      (
          "regions_without_policy",
          "full",
          ("moe.routing",),
          {},
          False,
          "only apply",
      ),
      (
          "limits_without_policy",
          "full",
          (),
          {"attention.inner_attention": 2},
          False,
          "only apply",
      ),
  )
  def test_tpu_selective_rejects_bad_settings(
      self, policy, save_regions, last_n_layers, scanned, message
  ) -> None:
    config = qwen3_8b()
    config.tpu_config.graph_trainer_memory_policy = policy
    config.tpu_config.graph_trainer_save_regions = save_regions
    config.tpu_config.graph_trainer_save_last_n_layers = last_n_layers
    config.tpu_config.use_scan_layers = scanned

    with self.assertRaisesRegex(ValueError, message):
      self._qwen3_graph_config(config)

  def test_negative_cpu_offload_executable_prefetch_is_rejected(self) -> None:
    config = qwen3_8b()
    config.tpu_config.graph_split.cpu_offload_prefetch_executables = -1

    with self.assertRaisesRegex(ValueError, "must be nonnegative"):
      GraphSplitter.from_trainer_config(
          self._qwen3_graph_config(config),
          optimizer_in_graph=False,
      )

  def test_cli_remat_schedule_is_propagated(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.enable_graph_trainer_remat_schedule",
        ]),
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertTrue(
        graph_config.tpu_config.enable_graph_trainer_remat_schedule
    )

  def test_cli_fsdp_schedule_is_propagated(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.graph_trainer_fsdp_schedule=interleave",
        ]),
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(
        graph_config.tpu_config.graph_trainer_fsdp_schedule,
        "interleave",
    )

  def test_cli_fsdp_prefetch_distance_is_propagated(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.graph_trainer_fsdp_prefetch_distance=2",
        ]),
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(
        graph_config.tpu_config.graph_trainer_fsdp_prefetch_distance,
        2,
    )

  def test_backward_layer_split_is_propagated(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
            "--tpu_config.use_graph_split",
            "--tpu_config.graph_split.backward_partition_mode=by_layer",
            "--tpu_config.graph_split.backward_layers_per_executable=2",
        ]),
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(
        graph_config.tpu_config.graph_split.backward_partition_mode,
        "by_layer",
    )
    self.assertEqual(
        graph_config.tpu_config.graph_split.backward_layers_per_executable,
        2,
    )

  def test_qwen3_8b_explicitly_uses_default_graph_policy(self) -> None:
    config = qwen3_8b()

    self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)
    self.assertTrue(config.parallelism.enable_tpu_topo_mesh)
    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy,
        "default",
    )
    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "never")
    splash = config.splash_attention_kernel
    self.assertEqual(splash.sa_q_layout, "SEQ_MINOR")
    self.assertEqual(splash.sa_k_layout, "SEQ_MINOR")
    self.assertEqual(splash.sa_v_layout, "HEAD_DIM_MINOR")
    self.assertEqual(splash.sa_block_kv_compute, 512)
    self.assertEqual(splash.sa_block_q_dkv, 2048)
    self.assertEqual(splash.sa_block_kv_dkv, 2048)

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    self.assertIsInstance(graph_config.loss, PallasCrossEntropyLoss.Config)
    self.assertEqual(graph_config.compile.memory_policy, "default")

  def test_qwen3_8b_zero3_recipe_uses_whole_graph_fsdp_schedule(self) -> None:
    config = qwen3_8b_zero3()

    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "always")
    self.assertEqual(config.tpu_config.graph_trainer_memory_policy, "full")
    self.assertTrue(config.tpu_config.enable_graph_trainer_remat_schedule)
    self.assertEqual(config.tpu_config.graph_trainer_fsdp_schedule, "interleave")
    self.assertTrue(config.tpu_config.optimizer_in_graph)
    self.assertFalse(config.tpu_config.use_graph_split)
    self.assertTrue(config.splash_attention_kernel.use_vmap_bwd)
    self.assertFalse(
        config.splash_attention_kernel.sa_use_document_aware_block_skipping
    )

  def test_unprotected_graph_trainer_zero3_is_rejected(self) -> None:
    config = qwen3_8b()
    config.parallelism.fsdp_reshard_after_forward = "always"

    with self.assertRaisesRegex(ValueError, "GraphTrainer ZeRO-3 requires"):
      self._qwen3_graph_config(config)

  @parameterized.named_parameters(
      ("fsdp_barriers", False, False, "prefetch"),
      ("graph_split", True, False, "none"),
      ("scan", False, True, "none"),
  )
  def test_graph_trainer_zero3_accepts_lifetime_boundary(
      self,
      use_graph_split: bool,
      use_scan_layers: bool,
      fsdp_schedule: str,
  ) -> None:
    config = qwen3_8b()
    config.parallelism.fsdp_reshard_after_forward = "always"
    config.tpu_config.use_graph_split = use_graph_split
    config.tpu_config.use_scan_layers = use_scan_layers
    config.tpu_config.graph_trainer_fsdp_schedule = fsdp_schedule

    self._qwen3_graph_config(config)

  def test_qwen3_8b_maxtext_parity_recipe(self) -> None:
    config = qwen3_8b_maxtext_parity()

    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "always")
    self.assertEqual(config.training.mixed_precision_reduce, "bfloat16")
    self.assertFalse(config.tpu_config.use_scan_layers)
    self.assertIn(
        "torchtitan.experiments.tpu.overrides.bf16_rope"
        ".tpu_activation_dtype_cossin_rope",
        config.override.imports,
    )
    splash = config.splash_attention_kernel
    self.assertTrue(splash.use_vmap_bwd)
    self.assertFalse(splash.sa_use_document_aware_block_skipping)
    self.assertEqual(splash.sa_q_layout, "SEQ_MINOR")
    self.assertEqual(splash.sa_k_layout, "SEQ_MINOR")
    self.assertEqual(splash.sa_v_layout, "HEAD_DIM_MINOR")
    self.assertEqual(splash.sa_block_kv_compute, 512)
    self.assertEqual(splash.sa_block_q_dkv, 2048)
    self.assertEqual(splash.sa_block_kv_dkv, 2048)

    graph_config = self._qwen3_graph_config(config)
    apply_overrides(graph_config.override, graph_config)
    assert graph_config.model_spec is not None
    ropes = [layer.attention.rope for layer in graph_config.model_spec.model.layers]
    self.assertTrue(ropes)
    self.assertTrue(
        all(
            isinstance(rope, TpuActivationDtypeCosSinRoPE.Config)
            for rope in ropes
        )
    )

  @parameterized.named_parameters(
      ("fp32_reduce", qwen3_32b, "float32", False),
      ("maxtext_parity", qwen3_32b_maxtext_parity, "bfloat16", True),
  )
  def test_qwen3_32b_recipes_reuse_8b_zero3_policy(
      self, recipe, reduce_dtype: str, bf16_rope: bool
  ) -> None:
    config = recipe()
    zero3 = qwen3_8b_zero3()

    assert config.model_spec is not None
    self.assertEqual(config.model_spec.flavor, "32B")
    self.assertFalse(config.model_spec.model.enable_weight_tying)
    self.assertLen(config.model_spec.model.layers, 64)
    self.assertLen(
        list(config.model_spec.model.traverse(TpuPackedQKVLinear.Config)), 64
    )
    self.assertEqual(config.training.max_context_length, 4096)
    self.assertEqual(
        config.training.num_tokens_per_microbatch_per_dp_rank, 8 * 4096
    )
    self.assertEqual(config.training.num_tokens_per_train_step, 1024 * 4096)
    self.assertEqual(config.training.mixed_precision_reduce, reduce_dtype)
    self.assertEqual(config.parallelism, zero3.parallelism)
    self.assertEqual(
        config.tpu_config,
        dataclasses.replace(
            zero3.tpu_config, graph_trainer_fsdp_schedule="prefetch"
        ),
    )
    self.assertEqual(
        config.splash_attention_kernel, zero3.splash_attention_kernel
    )
    self.assertEqual(
        "torchtitan.experiments.tpu.overrides.bf16_rope"
        ".tpu_activation_dtype_cossin_rope"
        in config.override.imports,
        bf16_rope,
    )
    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.HIERARCHICAL_REDUCE_SCATTER
        },
        tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
    )
    self.assertEqual(config.libtpu_init_flags["xla_tpu_dvfs_p_state"], "7")

    graph_config = self._qwen3_graph_config(config)
    assert graph_config.model_spec is not None
    self.assertFalse(graph_config.model_spec.model.enable_weight_tying)
    self.assertLen(
        list(
            graph_config.model_spec.model.traverse(TpuPackedQKVLinear.Config)
        ),
        64,
    )
    self.assertEqual(graph_config.compile.memory_policy, "full")

  def test_qwen3_32b_4x8x8_reduces_ffn_gradients_with_allreduce(self) -> None:
    config = qwen3_32b_4x8x8()
    base = qwen3_32b()

    self.assertEqual(config.training.num_tokens_per_train_step, 4096 * 4096)
    self.assertEqual(
        config.tpu_config,
        dataclasses.replace(
            base.tpu_config,
            graph_trainer_fsdp_schedule="interleave",
            fsdp_allreduce_slice_parameter_patterns=(
                "layers.*.feed_forward.w1.weight",
                "layers.*.feed_forward.w3.weight",
            ),
            fsdp_allreduce_slice_max_input_bytes=25600 * 5120 * 4,
        ),
    )
    self.assertEqual(config.libtpu_init_flags, base.libtpu_init_flags)
    self._qwen3_graph_config(config)

  def test_fsdp_only_recipe_uses_fused_adamw_and_attention_tuning(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_benchmark()

    self.assertEqual(config.optimizer.implementation, "fused")
    self.assertTrue(
        config.splash_attention_kernel.use_tokamax_splash_attention_kernel
    )
    self.assertTrue(config.splash_attention_kernel.use_vmap_bwd)
    self.assertFalse(
        config.splash_attention_kernel.sa_use_document_aware_block_skipping
    )
    self.assertEqual(config.splash_attention_kernel.sa_q_layout, "SEQ_MINOR")
    self.assertEqual(config.splash_attention_kernel.sa_k_layout, "SEQ_MINOR")
    self.assertEqual(
        config.splash_attention_kernel.sa_v_layout,
        "HEAD_DIM_MINOR",
    )

    self.assertEqual(config.splash_attention_kernel.sa_block_kv_compute, 512)
    self.assertEqual(config.splash_attention_kernel.sa_block_q_dkv, 2048)
    self.assertEqual(config.splash_attention_kernel.sa_block_kv_dkv, 2048)
    self.assertEqual(config.splash_attention_kernel.sa_block_kv_dkv_compute, 1024)
    assert config.model_spec is not None
    dispatchers = list(
        config.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
    )
    self.assertTrue(dispatchers)
    self.assertTrue(
        all(
            not dispatcher.fuse_route_score_backward
            for _, dispatcher, _, _ in dispatchers
        )
    )

  def test_fsdp_only_recipe_selects_attention_gradients_for_allreduce(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_benchmark()

    self.assertSequenceEqual(
        config.tpu_config.fsdp_allreduce_slice_parameter_patterns,
        (
            "layers.*.attention.qkv_linear.wqkv_packed.weight",
            "layers.*.attention.wo.weight",
        ),
    )
    self.assertEqual(
        config.tpu_config.fsdp_allreduce_slice_max_input_bytes,
        40 * 1024**2,
    )
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    pipeline = construct_tpu_pass_pipeline(
        typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
        graph_config,
    )
    pipeline_names = [
        getattr(getattr(step, "func", step), "__name__", "")
        for step in pipeline
    ]
    self.assertIn("apply_selected_reduce_scatter_allreduce_pass", pipeline_names)

  @parameterized.named_parameters(
      ("requires_simple_fsdp", "use_simple_fsdp", False, "SimpleFSDP"),
      (
          "rejects_dp_replication",
          "data_parallel_replicate_degree",
          2,
          "full-world FSDP",
      ),
      (
          "rejects_fixed_fsdp_degree",
          "data_parallel_shard_degree",
          64,
          "full-world FSDP",
      ),
      (
          "rejects_tensor_parallel",
          "tensor_parallel_degree",
          2,
          "full-world FSDP",
      ),
      (
          "rejects_context_parallel",
          "context_parallel_degree",
          2,
          "full-world FSDP",
      ),
      (
          "rejects_pipeline_parallel",
          "pipeline_parallel_degree",
          2,
          "full-world FSDP",
      ),
      (
          "rejects_expert_parallel",
          "expert_parallel_degree",
          2,
          "full-world FSDP",
      ),
  )
  def test_fsdp_allreduce_slice_rejects_unsupported_parallelism(
      self,
      field: str,
      value: bool | int,
      message: str,
  ) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
    setattr(config.parallelism, field, value)

    with self.assertRaisesRegex(ValueError, message):
      to_tpu_graph_trainer_config(
          config,
          qwen3_gt.model_registry,
          target_cls=TpuGraphTrainer.Config,
      )

  def test_fsdp_allreduce_slice_allows_bf16_reductions(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_benchmark()
    config.training.mixed_precision_reduce = "bfloat16"

    to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

  def test_attention_parity_recipe_adds_bf16_rope(self) -> None:
    config = qwen3_moe_30b_fsdp_only_attention_parity_benchmark()
    self.assertIn(
        "torchtitan.experiments.tpu.overrides.bf16_rope"
        ".tpu_activation_dtype_cossin_rope",
        config.override.imports,
    )
    assert config.model_spec is not None
    dispatchers = list(
        config.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
    )
    self.assertTrue(dispatchers)
    self.assertTrue(
        all(
            dispatcher.fuse_route_score_backward
            for _, dispatcher, _, _ in dispatchers
        )
    )

    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    apply_overrides(graph_config.override, graph_config)
    assert graph_config.model_spec is not None
    rope_configs = [
        layer.attention.rope for layer in graph_config.model_spec.model.layers
    ]
    self.assertTrue(rope_configs)
    self.assertTrue(
        all(
            isinstance(rope, TpuActivationDtypeCosSinRoPE.Config)
            for rope in rope_configs
        )
    )

  def test_235b_recipe_stays_stackable_under_scan(self) -> None:
    config = qwen3_moe_235b_scan()

    assert config.model_spec is not None
    self.assertEqual(config.model_spec.flavor, "235B-A22B")
    # Bias-based load balancing (#22) and the routing metrics (#28) both keep
    # per-layer state a scan body cannot hold.
    moes = list(config.model_spec.model.traverse(MoE.Config))
    self.assertLen(moes, 94)
    self.assertTrue(all(moe.load_balance_coeff is None for _, moe, _, _ in moes))
    self.assertTrue(
        all(moe.router.aux_loss is not None for _, moe, _, _ in moes)
    )
    self.assertFalse(config.metrics.enable_moe_routing)
    # Clearing the coeff leaves a dead in-place write scan also rejects, so the
    # override has to reach every MoE.
    apply_overrides(config.override, config)
    self.assertTrue(
        all(
            isinstance(moe, TpuDiscardedTokenCounterMoE.Config)
            for _, moe, _, _ in config.model_spec.model.traverse(MoE.Config)
        )
    )

  def test_235b_splitter_uses_global_aux_loss(self) -> None:
    config = qwen3_moe_235b_fsdp_activation_offload_4x4x4()

    assert config.model_spec is not None
    moes = list(config.model_spec.model.traverse(MoE.Config))
    self.assertLen(moes, 94)
    for _, moe, _, _ in moes:
      self.assertIsNone(moe.load_balance_coeff)
      aux_loss = moe.router.aux_loss
      self.assertIsInstance(aux_loss, MoELoadBalanceAuxLoss.Config)
      assert isinstance(aux_loss, MoELoadBalanceAuxLoss.Config)
      self.assertEqual(aux_loss.type, "batch_wise")
      self.assertEqual(aux_loss.coeff, 1e-3)
      self.assertEqual(aux_loss.top_k, moe.router.top_k)

  def test_30b_global_aux_recipe_uses_global_batch_aux_loss(self) -> None:
    # 30B-A3B-global-aux (and the fsdp_only recipes derived from it) must get
    # the fork's global-batch loss, not upstream's microbatch-wise one.
    config = qwen3_moe_30b_fsdp_only_full_ac_benchmark()

    assert config.model_spec is not None
    self.assertEqual(config.model_spec.flavor, "30B-A3B-global-aux")
    self.assertTrue(config.metrics.enable_moe_routing)
    moes = list(config.model_spec.model.traverse(MoE.Config))
    self.assertLen(moes, 48)
    for _, moe, _, _ in moes:
      self.assertIsNone(moe.load_balance_coeff)
      aux_loss = moe.router.aux_loss
      self.assertIsInstance(aux_loss, MoELoadBalanceAuxLoss.Config)
      assert isinstance(aux_loss, MoELoadBalanceAuxLoss.Config)
      self.assertEqual(aux_loss.type, "batch_wise")
      self.assertEqual(aux_loss.coeff, 1e-3)
      self.assertEqual(aux_loss.top_k, moe.router.top_k)

  def test_235b_splitter_uses_tokamax_splash_attention(self) -> None:
    for config in (
        qwen3_moe_235b_fsdp_activation_offload_4x4x4(),
        qwen3_moe_235b_fsdp_activation_offload_maxtext_4x4x4(),
    ):
      with self.subTest(reduce_dtype=config.training.mixed_precision_reduce):
        splash = config.splash_attention_kernel
        self.assertTrue(splash.use_tokamax_splash_attention_kernel)
        self.assertFalse(splash.sa_use_document_aware_block_skipping)

  def test_235b_4x4x4_uses_hierarchical_reduce_scatter(self) -> None:
    config = qwen3_moe_235b_fsdp_activation_offload_4x4x4()

    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.HIERARCHICAL_REDUCE_SCATTER
        },
        tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
    )

  def test_235b_flavor_resolves_through_the_graph_trainer(self) -> None:
    # The graph trainer discards the recipe's ModelSpec and re-resolves the
    # flavor string against upstream's registry. A flavor that only the TPU
    # registry knows dies there on a KeyError, after every other test passes.
    config = qwen3_moe_235b_scan()
    assert config.model_spec is not None

    spec = tpu_qwen3_graph_registry(config.model_spec.flavor)

    self.assertIsInstance(spec.model, GraphTrainerQwen3Model.Config)

  def test_235b_recipe_matches_the_maxtext_shape(self) -> None:
    config = qwen3_moe_235b_scan()

    # local_batch_size is now
    # num_tokens_per_microbatch_per_dp_rank // max_context_length.
    self.assertEqual(
        config.training.num_tokens_per_microbatch_per_dp_rank
        // config.training.max_context_length,
        16,
    )
    self.assertEqual(config.training.max_context_length, 4096)
    self.assertTrue(config.tpu_config.use_scan_layers)
    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy, "full_and_scan_offload"
    )
    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "always")
    for group in (
        tpu_flags.SELECTIVE_ICI_RS_PIPELINING,
        tpu_flags.ICI_AG_PIPELINING,
        tpu_flags.SPARSECORE_ASYNC_ALL_GATHER,
    ):
      self.assertEqual(
          {name: config.libtpu_init_flags[name] for name in group}, group
      )
    self.assertEqual(
        config.libtpu_init_flags[
            "xla_tpu_layer_scheduler_min_compute_latency_for_backward_pass"
        ],
        "25000000",
    )
    self.assertEqual(
        config.libtpu_init_flags["xla_tpu_copy_insertion_use_region_analysis"],
        "true",
    )
    self.assertEqual(
        config.libtpu_init_flags[
            "xla_tpu_copy_insertion_use_region_analysis_limit"
        ],
        "1000",
    )

  def test_235b_proxy_keeps_the_layer_shape_and_drops_the_count(self) -> None:
    proxy = qwen3_moe_235b_scan_proxy()

    assert proxy.model_spec is not None
    self.assertLen(proxy.model_spec.model.layers, 12)
    # The 32x global-to-local ratio, now in token counts: both sides scale by
    # the same max_context_length, so the ratio carries over unchanged.
    self.assertEqual(
        proxy.training.num_tokens_per_train_step,
        32 * proxy.training.num_tokens_per_microbatch_per_dp_rank,
    )

  def test_235b_maxtext_twin_changes_only_the_two_numerics_knobs(self) -> None:
    base = qwen3_moe_235b_scan()
    maxtext = qwen3_moe_235b_scan_maxtext()

    self.assertEqual(base.training.mixed_precision_reduce, "float32")
    self.assertEqual(maxtext.training.mixed_precision_reduce, "bfloat16")
    self.assertEqual(
        set(maxtext.override.imports) - set(base.override.imports),
        {
            "torchtitan.experiments.tpu.overrides.bf16_rope"
            ".tpu_activation_dtype_cossin_rope"
        },
    )
    assert base.model_spec is not None and maxtext.model_spec is not None
    # The FP32 route-score fusion is not a numerics knob: every Q235 recipe
    # uses it.
    splitter = qwen3_moe_235b_fsdp_activation_offload_4x4x4()
    assert splitter.model_spec is not None
    for config in (base, maxtext, splitter):
      dispatchers = list(
          config.model_spec.model.traverse(TpuAllToAllTokenDispatcher.Config)
      )
      self.assertLen(dispatchers, 94)
      self.assertTrue(all(d.fuse_route_score_backward for _, d, _, _ in dispatchers))
    # Everything the base recipe works around is inherited, not re-derived.
    self.assertEqual(maxtext.model_spec.flavor, base.model_spec.flavor)
    self.assertEqual(
        maxtext.training.num_tokens_per_microbatch_per_dp_rank
        // maxtext.training.max_context_length,
        16,
    )
    self.assertTrue(maxtext.tpu_config.use_scan_layers)

  def test_optimizer_implementation_cli_override_is_authoritative(self) -> None:
    config = ConfigManager().parse_args([
        "--module=torchtitan.experiments.tpu.qwen3",
        "--config=qwen3_8b",
        "--optimizer.implementation=fused",
    ])

    self.assertEqual(config.optimizer.implementation, "fused")

  def test_qwen3_30b_fsdp_profile_explicitly_uses_zero2(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()

    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy, "tpu_selective"
    )
    self.assertEqual(config.parallelism.fsdp_reshard_after_forward, "never")

  def test_qwen3_30b_fsdp_recipe_saves_routing_and_splash(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_benchmark()

    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy, "tpu_selective"
    )
    self.assertEqual(
        config.tpu_config.graph_trainer_save_regions,
        ("moe.routing", "attention.inner_attention"),
    )
    self.assertEqual(config.tpu_config.graph_trainer_save_last_n_layers, {})
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    self.assertEqual(graph_config.compile.memory_policy, "tpu_selective")
    self.assertEqual(
        graph_config.tpu_config.graph_trainer_save_regions,
        ("moe.routing", "attention.inner_attention"),
    )

  def test_qwen3_30b_fsdp_recipe_opts_out_to_full_from_cli(self) -> None:
    # The opt-out documented in the recipe README.
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_moe_30b_fsdp_only_full_ac_benchmark",
            "--tpu_config.graph_trainer_memory_policy=full",
            "--tpu_config.graph_trainer_save_regions",
        ]),
    )
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(graph_config.compile.memory_policy, "full")
    self.assertEqual(graph_config.tpu_config.graph_trainer_save_regions, ())

  def test_qwen3_30b_fsdp_recipe_limits_splash_to_last_layers_from_cli(
      self,
  ) -> None:
    # The partial save documented in the recipe README.
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_moe_30b_fsdp_only_full_ac_benchmark",
            "--tpu_config.graph_trainer_save_regions",
            "moe.routing",
            "--tpu_config.graph_trainer_save_last_n_layers",
            "attention.inner_attention",
            "24",
        ]),
    )
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertEqual(graph_config.compile.memory_policy, "tpu_selective")
    self.assertEqual(
        graph_config.tpu_config.graph_trainer_save_regions, ("moe.routing",)
    )
    self.assertEqual(
        graph_config.tpu_config.graph_trainer_save_last_n_layers,
        {"attention.inner_attention": 24},
    )

  @parameterized.parameters(
      name
      for name in dir(qwen3_config_registry)
      if name.startswith("qwen3_moe_235b")
  )
  def test_qwen3_235b_recipes_keep_their_own_memory_policy(self, name) -> None:
    # The 235B recipes start from the 30B benchmark recipe, whose
    # tpu_selective saves were measured only on 30B.
    config = getattr(qwen3_config_registry, name)()

    self.assertNotEqual(
        config.tpu_config.graph_trainer_memory_policy, "tpu_selective"
    )
    self.assertEqual(config.tpu_config.graph_trainer_save_regions, ())
    if config.tpu_config.use_graph_trainer:
      # The conversion validates the memory settings.
      self._qwen3_graph_config(config)

  def test_qwen3_moe_fsdp_uses_pallas_cross_entropy(self) -> None:
    config = qwen3_moe_30b_fsdp_only_full_ac_profile()

    self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)

  @parameterized.parameters(
      llama3_8b,
      llama3_70b,
      llama3_405b,
      sft_llama3_8b_metamath,
  )
  def test_llama3_recipe_uses_pallas_cross_entropy(self, recipe) -> None:
    """`loss_kernel.use_loss_kernel` is the route that does not survive here.

    It patches `lm_head` a second time, behind the graph trainer's back.
    """
    config = recipe()

    self.assertIsInstance(config.loss, PallasCrossEntropyLoss.Config)
    self.assertFalse(config.loss_kernel.use_loss_kernel)

  @parameterized.parameters(
      llama3_8b_scan,
      llama3_70b_scan,
      llama3_405b_scan,
  )
  def test_scan_recipe_sets_the_whole_lane(self, recipe) -> None:
    """Every field here is load-bearing and silent when missing.

    Without `full_and_scan_offload` 405B does not compile, and f32 Adam moments
    cost 12.7 GiB that sequence 8192 needs.
    """
    config = recipe()

    self.assertTrue(config.tpu_config.use_graph_trainer)
    self.assertTrue(config.tpu_config.use_scan_layers)
    self.assertEqual(config.tpu_config.compile_mode, "aot_fx_trace")
    self.assertEqual(
        config.tpu_config.graph_trainer_memory_policy, "full_and_scan_offload"
    )
    self.assertEqual(config.optimizer.implementation, "fused_opt_states_bf16")
    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.SELECTIVE_ICI_RS_PIPELINING
        },
        tpu_flags.SELECTIVE_ICI_RS_PIPELINING,
    )
    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.ICI_AG_PIPELINING
        },
        tpu_flags.ICI_AG_PIPELINING,
    )
    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.HIERARCHICAL_REDUCE_SCATTER
        },
        tpu_flags.HIERARCHICAL_REDUCE_SCATTER,
    )
    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.SPARSECORE_ASYNC_ALL_GATHER
        },
        tpu_flags.SPARSECORE_ASYNC_ALL_GATHER,
    )
    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER
        },
        tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
    )
    self.assertEqual(
        {
            name: config.libtpu_init_flags[name]
            for name in tpu_flags.CONVOLUTION_UNFUSING
        },
        tpu_flags.CONVOLUTION_UNFUSING,
    )
    self.assertEqual(
        config.libtpu_init_flags["xla_tpu_copy_insertion_use_region_analysis"],
        "true",
    )
    self.assertEqual(
        config.libtpu_init_flags[
            "xla_tpu_copy_insertion_use_region_analysis_limit"
        ],
        "1000",
    )

  @parameterized.parameters(
      llama3_8b,
      llama3_70b,
      llama3_405b,
  )
  def test_layerwise_recipe_stays_off_the_graph_trainer(self, recipe) -> None:
    """The production lane is layerwise; adding a `_scan` twin must not move it."""
    self.assertFalse(recipe().tpu_config.use_graph_trainer)

  @parameterized.parameters(
      llama3_debugmodel,
      llama3_1b,
      llama3_8b,
      llama3_8b_scan,
      llama3_70b,
      llama3_70b_scan,
      llama3_405b,
      llama3_405b_scan,
      sft_llama3_8b_metamath,
  )
  def test_llama3_recipe_overrides_complex_rope(self, recipe) -> None:
    """A recipe without it cannot run its decoder loop as a `scan`."""
    graph_config = to_tpu_graph_trainer_config(
        recipe(),
        llama3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertIn(
        "torchtitan.experiments.tpu.overrides.real_rope"
        ".tpu_real_arithmetic_complex_rope",
        graph_config.override.imports,
    )
    apply_overrides(graph_config.override, graph_config)
    assert graph_config.model_spec is not None
    with torch.device("meta"):
      model = graph_config.model_spec.model.build()
    ropes = [m for m in model.modules() if isinstance(m, ComplexRoPE)]
    self.assertNotEmpty(ropes)
    for rope in ropes:
      self.assertIsInstance(rope, TpuRealArithmeticComplexRoPE)

  @parameterized.parameters(
      (qwen3_8b, qwen3_gt),
      (qwen3_moe_30b_fsdp_only_full_ac_profile, qwen3_gt),
      (llama3_8b, llama3_gt),
      (llama3_70b, llama3_gt),
      (llama3_405b, llama3_gt),
      (sft_llama3_8b_metamath, llama3_gt),
  )
  def test_pallas_loss_recipe_overrides_lm_head(self, recipe, module) -> None:
    config = recipe()
    graph_config = to_tpu_graph_trainer_config(
        config,
        module.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    self.assertIn(
        "torchtitan.experiments.tpu.overrides.fused_loss.fused_loss_lm_head",
        graph_config.override.imports,
    )
    apply_overrides(graph_config.override, graph_config)
    assert graph_config.model_spec is not None
    lm_head_config = graph_config.model_spec.model.lm_head
    self.assertIsInstance(lm_head_config, FusedLossLinearWrapper.Config)

    with torch.device("meta"):
      model = graph_config.model_spec.model.build()
    self.assertIsInstance(model.lm_head, FusedLossLinearWrapper)
    self.assertEqual(
        [name for name in model.state_dict() if name.endswith("lm_head.weight")],
        ["lm_head.weight"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    self.assertTrue(
        any(parameter is model.lm_head.weight for parameter in optimizer_params)
    )

  def test_fused_loss_linear_wrapper_contract(self) -> None:
    param_init = {"weight": lambda weight: nn.init.constant_(weight, 0.25)}
    sharding_config = object()
    linear_config = Linear.Config(
        in_features=4,
        out_features=8,
        param_init=param_init,
        sharding_config=sharding_config,
    )
    config = fused_loss_lm_head(linear_config)

    self.assertIs(config.param_init, linear_config.param_init)
    self.assertIs(config.sharding_config, linear_config.sharding_config)
    lm_head = config.build()
    lm_head.init_states()
    hidden = torch.randn(2, 4)
    output, output_weight = lm_head(hidden)

    self.assertIs(output, hidden)
    self.assertEqual(list(lm_head.state_dict()), ["weight"])
    torch.testing.assert_close(
        lm_head.weight,
        torch.full_like(lm_head.weight, 0.25),
    )
    torch.testing.assert_close(output_weight, lm_head.weight.t())

  def test_qwen3_moe_ep_keeps_plain_cross_entropy(self) -> None:
    config = qwen3_moe_30b()

    self.assertIsInstance(config.loss, CrossEntropyLoss.Config)

  def test_chunked_loss_is_rejected(self) -> None:
    config = typing.cast(
        TPUTrainerConfig,
        ConfigManager().parse_args([
            "--module=torchtitan.experiments.tpu.qwen3",
            "--config=qwen3_debugmodel",
        ]),
    )
    config.loss = ChunkedLossWrapper.Config()
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    with self.assertRaisesRegex(
        ValueError, "TPU GraphTrainer does not support ChunkedLossWrapper"
    ):
      TpuGraphTrainer(graph_config)

  def test_remat_schedule_supports_optimizer_in_graph(self) -> None:
    config = qwen3_8b()
    config.tpu_config.enable_graph_trainer_remat_schedule = True
    config.tpu_config.graph_trainer_fsdp_schedule = "interleave"
    config.tpu_config.graph_trainer_fsdp_prefetch_distance = 2
    config.tpu_config.optimizer_in_graph = True
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )

    passes = construct_tpu_pass_pipeline(
        typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
        graph_config,
    )
    pass_names = [
        getattr(graph_pass, "func", graph_pass).__name__ for graph_pass in passes
    ]

    self.assertIn("apply_tpu_remat_schedule", pass_names)
    self.assertEqual(pass_names[0], "remove_parameter_gradient_markers_pass")
    self.assertLess(
        pass_names.index("apply_tpu_remat_schedule"),
        pass_names.index("apply_tpu_remat_fsdp_all_gather_schedule"),
    )
    self.assertLess(
        pass_names.index("schedule_fsdp_comms_to_dense_regions_pass"),
        pass_names.index("apply_tpu_fsdp_rs_deadline_schedule"),
    )
    fsdp_schedule = next(
        graph_pass
        for graph_pass in passes
        if getattr(graph_pass, "func", graph_pass).__name__
        == "apply_tpu_remat_fsdp_all_gather_schedule"
    )
    self.assertEqual(fsdp_schedule.keywords["prefetch_distance"], 2)

  @parameterized.parameters(
      ("none", False, False),
      ("prefetch", True, False),
      ("interleave", True, True),
  )
  def test_fsdp_schedule_selects_passes(
      self,
      schedule: str,
      has_prefetch: bool,
      has_interleave: bool,
  ) -> None:
    config = qwen3_8b()
    config.tpu_config.enable_graph_trainer_remat_schedule = True
    config.tpu_config.graph_trainer_fsdp_schedule = schedule
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    passes = construct_tpu_pass_pipeline(
        typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
        graph_config,
    )
    pass_names = {
        getattr(graph_pass, "func", graph_pass).__name__ for graph_pass in passes
    }

    self.assertEqual(
        "apply_tpu_remat_fsdp_all_gather_schedule" in pass_names,
        has_prefetch,
    )
    self.assertEqual(
        "schedule_fsdp_comms_to_dense_regions_pass" in pass_names,
        has_prefetch,
    )
    self.assertEqual(
        "apply_tpu_fsdp_rs_deadline_schedule" in pass_names,
        has_interleave,
    )

  def test_fsdp_schedule_requires_rematerialization(self) -> None:
    config = qwen3_8b()
    config.tpu_config.graph_trainer_fsdp_schedule = "interleave"
    graph_config = to_tpu_graph_trainer_config(
        config,
        qwen3_gt.model_registry,
        target_cls=TpuGraphTrainer.Config,
    )
    with self.assertRaisesRegex(ValueError, "rematerialization"):
      construct_tpu_pass_pipeline(
          typing.cast(typing.Any, SimpleNamespace(num_static_inputs=0)),
          graph_config,
      )


if __name__ == "__main__":
  absltest.main()
