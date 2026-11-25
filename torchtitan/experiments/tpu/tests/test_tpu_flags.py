"""Tests for the XLA / libtpu init flags carried by TPU configs."""

import dataclasses
from unittest import mock

from absl.testing import absltest
import tyro

from torchtitan.config.manager import custom_registry
from torchtitan.experiments.tpu import tpu_flags


@dataclasses.dataclass
class _FakeParallelism:
  expert_parallel_degree: int = 1


@dataclasses.dataclass
class _FakeConfig:
  """The fields apply_init_flags() reads off a BaseTPUTrainerConfig."""

  libtpu_init_flags: dict[str, str] = dataclasses.field(default_factory=dict)
  xla_flags: dict[str, str] = dataclasses.field(default_factory=dict)
  extra_libtpu_init_flags: dict[str, str] = dataclasses.field(
      default_factory=dict
  )
  extra_xla_flags: dict[str, str] = dataclasses.field(default_factory=dict)
  parallelism: _FakeParallelism = dataclasses.field(
      default_factory=_FakeParallelism
  )


class MergeFlagsTest(absltest.TestCase):

  def test_later_bundles_and_overrides_win(self) -> None:
    merged = tpu_flags.merge_flags(
        {"a": "1", "b": "2"},
        {"b": "3"},
        c="4",
    )
    self.assertEqual(merged, {"a": "1", "b": "3", "c": "4"})

  def test_bundles_are_not_mutated(self) -> None:
    bundle = {"a": "1"}
    tpu_flags.merge_flags(bundle, a="2")
    self.assertEqual(bundle, {"a": "1"})

  def test_values_render_as_libtpu_expects(self) -> None:
    merged = tpu_flags.merge_flags(
        enabled=True,
        disabled=False,
        count=65536,
        ratio=0.5,
        text="SEQ_MINOR",
    )
    self.assertEqual(
        merged,
        {
            "enabled": "true",
            "disabled": "false",
            "count": "65536",
            "ratio": "0.5",
            "text": "SEQ_MINOR",
        },
    )


class FormatAndParseTest(absltest.TestCase):

  def test_format_renders_command_line_flags(self) -> None:
    self.assertEqual(
        tpu_flags.format_flags({"xla_tpu_dvfs_p_state": "7", "verbose": ""}),
        "--xla_tpu_dvfs_p_state=7 --verbose",
    )

  def test_parse_accepts_launcher_flag_strings(self) -> None:
    self.assertEqual(
        tpu_flags.parse_flags(
            "--xla_tpu_scoped_vmem_limit_kib=65536"
            "   --xla_tpu_dvfs_p_state=7 --verbose"
        ),
        {
            "xla_tpu_scoped_vmem_limit_kib": "65536",
            "xla_tpu_dvfs_p_state": "7",
            "verbose": "",
        },
    )

  def test_parse_of_empty_string_is_empty(self) -> None:
    self.assertEqual(tpu_flags.parse_flags(""), {})

  def test_round_trip(self) -> None:
    flags = tpu_flags.merge_flags(
        tpu_flags.SPARSECORE_LATENCY_HIDING_SCHEDULER,
        tpu_flags.TPU7X_THROUGHPUT_P_STATE,
        xla_tpu_scoped_vmem_limit_kib=65472,
    )
    self.assertEqual(tpu_flags.parse_flags(tpu_flags.format_flags(flags)), flags)


class ApplyInitFlagsTest(absltest.TestCase):

  def test_writes_both_environment_variables(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={"xla_tpu_dvfs_p_state": "7"},
        xla_flags={"xla_dump_to": "/tmp/hlo"},
    )
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(
        environ,
        {
            "LIBTPU_INIT_ARGS": "--xla_tpu_dvfs_p_state=7",
            "XLA_FLAGS": "--xla_dump_to=/tmp/hlo",
        },
    )

  def test_environment_wins_over_config_for_the_same_flag(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={
            "xla_tpu_scoped_vmem_limit_kib": "65536",
            "xla_tpu_dvfs_p_state": "7",
        }
    )
    environ = {
        "LIBTPU_INIT_ARGS": (
            "--xla_tpu_scoped_vmem_limit_kib=131072 --xla_tpu_extra_flag=1"
        )
    }

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(
        tpu_flags.parse_flags(environ["LIBTPU_INIT_ARGS"]),
        {
            # From the environment, overriding the config.
            "xla_tpu_scoped_vmem_limit_kib": "131072",
            # Only in the environment, so preserved.
            "xla_tpu_extra_flag": "1",
            # Only in the config, so applied.
            "xla_tpu_dvfs_p_state": "7",
        },
    )

  def test_empty_config_flags_leave_the_environment_alone(self) -> None:
    environ = {"LIBTPU_INIT_ARGS": "--xla_tpu_extra_flag=1"}

    tpu_flags.apply_init_flags(_FakeConfig(), environ=environ)

    self.assertEqual(environ, {"LIBTPU_INIT_ARGS": "--xla_tpu_extra_flag=1"})
    self.assertNotIn("XLA_FLAGS", environ)


class ExtraFlagsTest(absltest.TestCase):
  """The additive layer: adds flags to a recipe and retunes the ones it sets."""

  def test_extra_flags_add_to_and_override_the_config(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={
            "xla_tpu_scoped_vmem_limit_kib": "65472",
            "xla_tpu_dvfs_p_state": "7",
        },
        extra_libtpu_init_flags={
            "xla_tpu_dvfs_p_state": "5",
            "xla_dump_to": "/tmp/hlo",
        },
    )
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(
        tpu_flags.parse_flags(environ["LIBTPU_INIT_ARGS"]),
        {
            # Untouched by the launch command.
            "xla_tpu_scoped_vmem_limit_kib": "65472",
            # Retuned by it.
            "xla_tpu_dvfs_p_state": "5",
            # Added by it.
            "xla_dump_to": "/tmp/hlo",
        },
    )

  def test_environment_still_wins_over_extra_flags(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={"xla_tpu_dvfs_p_state": "7"},
        extra_libtpu_init_flags={"xla_tpu_dvfs_p_state": "5"},
    )
    environ = {"LIBTPU_INIT_ARGS": "--xla_tpu_dvfs_p_state=3"}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(
        tpu_flags.parse_flags(environ["LIBTPU_INIT_ARGS"]),
        {"xla_tpu_dvfs_p_state": "3"},
    )

  def test_extra_flags_alone_reach_the_environment(self) -> None:
    # A config that declares nothing still gets the launch command's flags.
    config = _FakeConfig(extra_xla_flags={"xla_dump_to": "/tmp/hlo"})
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(environ["XLA_FLAGS"], "--xla_dump_to=/tmp/hlo")


class FlagMapCommandLineTest(absltest.TestCase):
  """The tyro rule that lets one string set any flag, whatever the default."""

  @dataclasses.dataclass
  class _Config:
    libtpu_init_flags: tpu_flags.FlagMap = dataclasses.field(
        default_factory=tpu_flags.FlagMap
    )

  def _parse(self, default: dict[str, str], *args: str) -> dict[str, str]:
    config = tyro.cli(
        self._Config,
        args=list(args),
        default=self._Config(tpu_flags.FlagMap(default)),
        registry=custom_registry,
    )
    return dict(config.libtpu_init_flags)

  def test_sets_flags_the_default_does_not_have(self) -> None:
    # The whole point of the FlagMap rule: a plain dict field would only expose
    # the keys already in its default.
    self.assertEqual(
        self._parse(
            {"xla_tpu_dvfs_p_state": "7"},
            "--libtpu-init-flags",
            "xla_dump_to=/tmp/hlo",
        ),
        {"xla_dump_to": "/tmp/hlo"},
    )

  def test_accepts_several_flags_in_one_value(self) -> None:
    self.assertEqual(
        self._parse({}, "--libtpu-init-flags", "a=1 b=2"),
        {"a": "1", "b": "2"},
    )

  def test_accepts_the_environment_variable_spelling(self) -> None:
    # A value that starts with "--" has to be attached with "=", or argparse
    # reads it as the next option.
    self.assertEqual(
        self._parse({}, "--libtpu-init-flags=--a=1 --b=2"),
        {"a": "1", "b": "2"},
    )


class RemovingFlagsTest(absltest.TestCase):
  """"!flag" drops a flag without naming the value libtpu would fall back to."""

  def test_removes_a_flag_the_config_sets(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={
            "xla_tpu_dvfs_p_state": "7",
            "xla_tpu_scoped_vmem_limit_kib": "65472",
        },
        extra_libtpu_init_flags=tpu_flags.parse_flags("!xla_tpu_dvfs_p_state"),
    )
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(
        environ["LIBTPU_INIT_ARGS"], "--xla_tpu_scoped_vmem_limit_kib=65472"
    )

  def test_removing_a_flag_nothing_sets_is_a_no_op(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={"xla_tpu_dvfs_p_state": "7"},
        extra_libtpu_init_flags=tpu_flags.parse_flags("!xla_dump_to"),
    )
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(environ["LIBTPU_INIT_ARGS"], "--xla_tpu_dvfs_p_state=7")

  def test_removal_and_addition_in_one_value(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={"xla_tpu_dvfs_p_state": "7"},
        extra_libtpu_init_flags=tpu_flags.parse_flags(
            "!xla_tpu_dvfs_p_state xla_dump_to=/tmp/hlo"
        ),
    )
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(environ["LIBTPU_INIT_ARGS"], "--xla_dump_to=/tmp/hlo")

  def test_environment_can_put_back_a_removed_flag(self) -> None:
    # The environment is still the outermost layer, removals included.
    config = _FakeConfig(
        libtpu_init_flags={"xla_tpu_dvfs_p_state": "7"},
        extra_libtpu_init_flags=tpu_flags.parse_flags("!xla_tpu_dvfs_p_state"),
    )
    environ = {"LIBTPU_INIT_ARGS": "--xla_tpu_dvfs_p_state=3"}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(environ["LIBTPU_INIT_ARGS"], "--xla_tpu_dvfs_p_state=3")


class ReplacingTheDeclaredSetTest(absltest.TestCase):
  """Setting the plain field replaces a recipe's whole set, empty included."""

  def test_emptying_the_config_flags_applies_none(self) -> None:
    # The A/B for "was it the flags?": --libtpu-init-flags "" on a recipe that
    # declares a bundle runs with no libtpu flags at all.
    config = _FakeConfig(libtpu_init_flags={})
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertNotIn("LIBTPU_INIT_ARGS", environ)

  def test_replacement_set_is_applied_verbatim(self) -> None:
    config = _FakeConfig(libtpu_init_flags={"xla_tpu_dvfs_p_state": "5"})
    environ: dict[str, str] = {}

    tpu_flags.apply_init_flags(config, environ=environ)

    self.assertEqual(environ["LIBTPU_INIT_ARGS"], "--xla_tpu_dvfs_p_state=5")


_RAGGED_A2A_SC = "xla_tpu_enable_sparse_core_collective_offload_ragged_all_to_all"


class ExpertParallelFlagsTest(absltest.TestCase):
  """EP > 1 keeps ragged_all_to_all on the TensorCore on libtpu < 0.0.51."""

  def setUp(self):
    super().setUp()
    # Hermetic: don't depend on the libtpu installed where the test runs.
    self.enter_context(
        mock.patch.object(
            tpu_flags, "_installed_libtpu_version", return_value="0.0.50"
        )
    )

  def _libtpu_flags(
      self, config, environ: dict[str, str] | None = None
  ) -> dict[str, str]:
    environ = {} if environ is None else environ
    tpu_flags.apply_init_flags(config, environ=environ)
    return tpu_flags.parse_flags(environ.get("LIBTPU_INIT_ARGS", ""))

  def test_ep_default_gating(self) -> None:
    ep_config = _FakeConfig(
        libtpu_init_flags={"xla_tpu_dvfs_p_state": "7"},
        parallelism=_FakeParallelism(expert_parallel_degree=2),
    )
    self.assertEqual(
        self._libtpu_flags(ep_config),
        {_RAGGED_A2A_SC: "false", "xla_tpu_dvfs_p_state": "7"},
    )

    environ: dict[str, str] = {}
    tpu_flags.apply_init_flags(_FakeConfig(), environ=environ)
    self.assertNotIn("LIBTPU_INIT_ARGS", environ)

    with mock.patch.object(
        tpu_flags, "_installed_libtpu_version", return_value="0.0.51"
    ):
      self.assertEqual(
          self._libtpu_flags(ep_config), {"xla_tpu_dvfs_p_state": "7"}
      )

  def test_recipe_value_wins_over_ep_default(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={_RAGGED_A2A_SC: "true"},
        parallelism=_FakeParallelism(expert_parallel_degree=2),
    )
    self.assertEqual(self._libtpu_flags(config), {_RAGGED_A2A_SC: "true"})

  def test_harness_variant_applies_only_the_ep_flag(self) -> None:
    config = _FakeConfig(
        libtpu_init_flags={"xla_tpu_dvfs_p_state": "7"},
        xla_flags={"xla_dump_to": "/tmp/hlo"},
        extra_libtpu_init_flags={"xla_tpu_scoped_vmem_limit_kib": "65472"},
        parallelism=_FakeParallelism(expert_parallel_degree=2),
    )
    environ = {"LIBTPU_INIT_ARGS": "--xla_tpu_extra_flag=1"}

    tpu_flags.apply_expert_parallel_init_flags(config, environ=environ)

    self.assertEqual(
        environ,
        {
            "LIBTPU_INIT_ARGS": (
                f"--{_RAGGED_A2A_SC}=false --xla_tpu_extra_flag=1"
            )
        },
    )

  def test_libtpu_versions_that_need_the_tensorcore_lowering(self) -> None:
    for version, needed in (
        ("0.0.50", True),
        ("0.0.51.dev20261015", True),
        ("0.0.51rc1", True),
        (None, True),
        ("unknown", True),
        ("0.0.51", False),
        ("0.0.51.post1", False),
        ("0.0.52.dev3", False),
        ("0.1.0", False),
    ):
      with self.subTest(version=version):
        self.assertEqual(
            tpu_flags.libtpu_needs_tensorcore_ragged_all_to_all(version),
            needed,
        )


if __name__ == "__main__":
  absltest.main()
