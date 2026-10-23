"""Discover test modules across the whole repository and shard them for CI.

Every ``test_*.py`` file in the repository is a candidate; files that define no
tests (shared helpers such as ``test_utils.py``) are skipped. Each discovered
test is classified along two independent axes:

* hardware -- ``distributed`` (multi-chip TPU), ``single_tpu``, or ``cpu``,
  derived from the base classes the test's TestCase inherits from.
* runner -- how the file is invoked:

  - ``module``: absltest/unittest files with a ``__main__`` block, run as
    ``python -m <module>``. These are the tests written in G3.
  - ``pytest``: bare pytest-style files with no entry point, plus everything
    under ``PYTEST_ONLY_DIRS``, run as ``pytest <path>``. These are the tests
    from upstream under tests/unit_tests.

Shards are balanced by measured runtime rather than by test count, so one slow
module does not stack up behind another. The resulting shards are emitted as
JSON for the GitHub Actions matrix.
"""

import ast
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Measured wall-clock seconds per test target, used to balance the shards.
# Refresh from a nightly run's logs when shard times drift; targets absent from
# the file are treated as median-cost rather than free.
DURATIONS_PATH = Path(__file__).with_name("test_durations.json")

NUM_DISTRIBUTED_TPU_SHARDS = 4
NUM_SINGLE_TPU_SHARDS = 2
NUM_CPU_SHARDS = 2
NUM_PYTEST_SHARDS = 2

# Trees run under pytest even when a file has a unittest entry point. pytest
# collects unittest.TestCase natively, so routing the whole unit test suite
# through one runner keeps behavior uniform. Paths are relative to REPO_ROOT.
PYTEST_ONLY_DIRS = ("tests/unit_tests",)

# Directory names pruned during the walk. These hold no importable tests.
EXCLUDED_DIR_NAMES = {
    ".ci",
    ".claude",
    ".git",
    ".github",
    ".pytest_cache",
    "__pycache__",
    "assets",
    "build",
    "dist",
    "outputs",
    "node_modules",
}

# Tests that are temporarily disabled.
DISABLED_TESTS = {
    "torchtitan.experiments.tpu.kernels.tests.test_segment_matmul",
    # RL Ray non-colocated tests (require multi-host Ray TPU cluster)
    "torchtitan.experiments.tpu.rl_nc_ray.tests.test_mini_rl",
    "torchtitan.experiments.tpu.rl_nc_ray.tests.test_varlen_attn_tpu",
    # TODO: add copybara handling for this test
    "torchtitan.experiments.tpu.kernels.tests.test_kernel_name_determinism_distributed",
    # TODO: RuntimeError: create_process_group(): PjRtClient is not initialized.
    "torchtitan.experiments.tpu.graph_trainer.tests.test_layer_stack_distributed",
    # TODO: imports torchtitan.scripts; this fork keeps scripts/ at the repo root.
    "tests.unit_tests.cpu.test_download_hf_assets",
    "tests.unit_tests.cpu.test_tokenizer",
    # TODO: mark_dynamic() no longer sets _dynamo_dynamic_range, so the copy assertion is stale.
    "torchtitan.experiments.graph_trainer.tests.test_trace_module",
    # Missing optional dependency renderers
    "torchtitan.experiments.rl.tests.test_alphabet_sort",
    "torchtitan.experiments.rl.tests.test_async_controller",
    "torchtitan.experiments.rl.tests.test_rollout_recorder",
    "torchtitan.experiments.rl.tests.test_rollout_utils",
    "torchtitan.experiments.rl.tests.test_search_r1",
    "torchtitan.experiments.rl.tests.test_weight_sync",
    "torchtitan.experiments.rl.tests.test_dapo_math",
    "torchtitan.experiments.rl.tests.test_muse_glimmer_renderer",
    "torchtitan.experiments.rl.tests.test_renderer",
    "torchtitan.experiments.rl.tests.test_rollouter_pool",
    "torchtitan.experiments.rl.tests.test_rollout_worker",
    "torchtitan.experiments.rl.tests.test_token_env",
    # Missing optional dependency torchstore
    "torchtitan.experiments.rl.tests.test_engine_loop",
    # Missing optional dependency monarch
    "torchtitan.experiments.rl.tests.test_shutdown",
    "torchtitan.experiments.rl.tests.test_inter_generator_router",
    "torchtitan.experiments.rl.tests.test_intra_generator_router",
    # Missing optional dependency vllm
    "torchtitan.experiments.rl.tests.test_generator",
    "torchtitan.experiments.rl.tests.test_bitwise_parity",
    "torchtitan.experiments.rl.tests.test_vllm_otel_stat_logger",
    "torchtitan.experiments.rl.tests.test_vllm_wrapper",
    # Missing optional dependency verifiers
    "torchtitan.experiments.rl.tests.test_verifiers",
    "torchtitan.experiments.rl.tests.test_verifiers_example",
    # Missing optional dependency torch_checkpointing
    "tests.unit_tests.cpu.test_torch_checkpointing",
    # Requires CUDA; upstream runs these tests only on CUDA
    "torchtitan.experiments.graph_trainer.tests.test_passes",
    "torchtitan.experiments.graph_trainer.tests.test_profiler",
    "torchtitan.experiments.graph_trainer.tests.test_cpu_offload",
    "torchtitan.experiments.graph_trainer.tests.test_numerics",
    # Requires optional dependency fla from requirements-vlm.txt
    "tests.unit_tests.cpu.test_rope",
    "tests.unit_tests.cpu.test_qwen3_5",
    # Missing optional dependency torchvision
    "tests.unit_tests.cpu.test_mm_dataset_checkpointing",
    "tests.unit_tests.cpu.components.data.test_qwen_multimodal_data",
    "tests.unit_tests.cpu.test_packed_vision",
    "tests.unit_tests.cpu.test_qwen3_6",
    "tests.unit_tests.cpu.test_qwen3_8",
    "tests.unit_tests.cpu.test_mm_dataset_preprocessing",
    "tests.unit_tests.cpu.test_debug_config_defaults",
    "tests.unit_tests.cpu.test_no_new_cli_options",
    # Upstream test assumes spmd_backend defaults to "spmd_types"; this fork
    # pins "partial_dtensor" (copy.bara.sky), so the CP backend gate fires first.
    "tests.unit_tests.cpu.test_context_parallel_validation",
    # TPUGrainDataLoader.Config overrides `dataset` as a plain str.
    "tests.unit_tests.cpu.test_dataset_flux",
    # set_gqa_inner_attention_local_map also registers the SDPA q_BLHK/k_BLHK/
    # v_BLHV keys; the test asserts the exact varlen key set.
    "tests.unit_tests.cpu.test_varlen_attention",
    # Image torch rejects 'reorder_for_locality' in nested compile-region options.
    "torchtitan.experiments.graph_trainer.tests.test_subgraph_regions",
    # Missing optional dependency autoparallel
    "torchtitan.experiments.graph_trainer.tests.test_autoparallel",
    "torchtitan.experiments.jax.tests.test_distributed",
    # TODO: get_peak_flops() rejects 'cpu', so MFU setup raises; make MFU optional.
    "torchtitan.experiments.jax.tests.test_train_distributed",
}

# Upstream GPU-only tests; upstream runs these only on CUDA runners.
GPU_ONLY_DIRS = ("tests/unit_tests/gpu",)

# Individual upstream pytest tests that are disabled, as node-ID prefixes passed
# to `pytest --deselect`. Use this instead of DISABLED_TESTS when the rest of
# the file passes.
DESELECTED_TESTS = (
    # Upstream tests; MXFP8 needs torchao's 32x32 swizzled cast kernels, which
    # are not in any torchao release the image installs.
    "tests/unit_tests/cpu/test_quantization.py::test_mxfp8_converter_replaces_a_root_linear_config",
    "tests/unit_tests/cpu/test_quantization.py::test_mxfp8_converter_applies_mxfp8_saved_input_fqns",
    "tests/unit_tests/cpu/test_quantization.py::test_mxfp8_converter_rejects_unmatched_saved_input_fqns",
    # Upstream tests assume spmd_backend defaults to "spmd_types"; this fork
    # pins "partial_dtensor" (copy.bara.sky).
    "tests/unit_tests/cpu/test_config_manager.py::TestConfigManager::test_suppressed_model_spec_is_opaque_to_tyro",
    "tests/unit_tests/cpu/test_router_gate_linear.py::TestRouterGateLinearSPMD::test_autograd_function_propagates_router_types",
    # Upstream test; the fork's Qwen3 Config.update_from_config reads
    # config.training, which the test's SimpleNamespace config lacks.
    "tests/unit_tests/cpu/test_expert_parallel.py::TestExpertParallelConfigValidation::test_all_moe_layers_divisible",
    # Upstream tests; the fork's train_step calls self._clip_grad_norm()
    # (_GOOGLE_PATCHES), which the tests' SimpleNamespace trainer lacks.
    "tests/unit_tests/cpu/test_trainer.py::test_train_step_replay_checks_only_first_forward_backward",
    "tests/unit_tests/cpu/test_trainer.py::test_trainer_accumulates_reused_cuda_graph_losses",
    "tests/unit_tests/cpu/test_trainer.py::test_pure_fsdp_does_not_toggle_requires_all_reduce",
    "tests/unit_tests/cpu/test_trainer.py::test_hsdp_skips_replicate_all_reduce_until_last_accum_group",
    "tests/unit_tests/cpu/test_trainer.py::test_hsdp_keeps_all_reduce_on_single_accum_group",
    "tests/unit_tests/cpu/test_trainer.py::test_hsdp_does_not_toggle_requires_all_reduce_under_cuda_graphs",
    # Upstream test; the fork passes cache_dir= to load_dataset (copy.bara.sky),
    # so the exact-call assertion fails.
    "tests/unit_tests/cpu/components/data/test_grain_data.py::test_hf_explicit_fields_are_passed_to_load_dataset",
)


def _iter_test_files():
    """Yield every ``test_*.py`` path in the repository, pruning excluded dirs."""
    stack = [REPO_ROOT]
    while stack:
        directory = stack.pop()
        for entry in sorted(directory.iterdir()):
            if entry.is_dir():
                if entry.name not in EXCLUDED_DIR_NAMES:
                    stack.append(entry)
            elif entry.name.startswith("test_") and entry.suffix == ".py":
                yield entry


def _is_test_node(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name.startswith("test")
    if isinstance(node, ast.ClassDef):
        return node.name.startswith("Test") or any(
            _is_test_node(child) for child in node.body
        )
    return False


def _base_name(base):
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def classify_test(tree):
    """Return the hardware category a parsed test module needs."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _base_name(base)
            if base_name is None:
                continue
            if "BaseDistributedDeviceTest" in base_name:
                return "distributed"
            if "BaseAcceleratorDeviceTest" in base_name:
                return "single_tpu"
    return "cpu"


def _is_pytest_only(relative):
    """True if the path lives in a tree that always runs under pytest."""
    return str(relative).startswith(PYTEST_ONLY_DIRS)


def has_main_block(tree):
    """True if the module has a top-level ``if __name__ == "__main__":`` guard."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Name) and sub.id == "__name__":
                return True
    return False


def discover_tests():
    """Bucket every discovered test by hardware category and runner.

    The distributed, single_tpu and cpu buckets hold dotted module paths; the
    pytest bucket holds file paths, matching how each runner is invoked.
    """
    categorized_tests = {
        "distributed": [],
        "single_tpu": [],
        "cpu": [],
        "pytest": [],
    }

    for path in _iter_test_files():
        relative = path.relative_to(REPO_ROOT)
        if str(relative).startswith(GPU_ONLY_DIRS):
            continue
        module = str(relative.with_suffix("")).replace("/", ".")
        if module in DISABLED_TESTS:
            continue

        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            print(f"Warning: skipping unparsable {relative}: {exc}", file=sys.stderr)
            continue

        # Skip helpers (e.g. test_utils.py) that match the glob but define no tests.
        if not any(_is_test_node(node) for node in ast.walk(tree)):
            continue

        category = classify_test(tree)
        if category == "distributed" and not path.name.endswith("_distributed.py"):
            sys.exit(
                f"Error: Test '{relative}' is classified as distributed (inherits from "
                "BaseDistributedDeviceTest) but its filename does not end with '_distributed.py'. "
                "Please rename it to end with '_distributed.py'."
            )

        if category == "cpu":
            # Categorize tests without a `__main__` entry point to run under pytest.
            if _is_pytest_only(relative) or not has_main_block(tree):
                categorized_tests["pytest"].append(str(relative))
            else:
                categorized_tests["cpu"].append(module)
        else:
            categorized_tests[category].append(module)

    # Sort for deterministic behavior
    for category in categorized_tests:
        categorized_tests[category].sort()
        
    return categorized_tests

def chunk_tests(tests, num_shards, durations):
    """Split tests into shards of roughly equal estimated runtime.

    Longest-processing-time-first: walk the targets slowest to fastest, adding
    each to whichever shard is lightest so far.
    """
    if not tests:
        return []

    known = [durations[test] for test in tests if test in durations]
    fallback = statistics.median(known) if known else 1
    weights = {test: durations.get(test, fallback) for test in tests}
    # Name breaks ties so the same input always yields the same layout.
    ordered = sorted(tests, key=lambda test: (-weights[test], test))

    buckets = [[] for _ in range(num_shards)]
    loads = [0] * num_shards
    for test in ordered:
        lightest = min(range(num_shards), key=lambda index: (loads[index], index))
        buckets[lightest].append(test)
        loads[lightest] += weights[test]

    shards = []
    for bucket in buckets:
        if bucket:
            # Store tests as a space-separated string so bash can iterate easily
            shards.append({
                "shard": f"shard_{len(shards) + 1}",
                "modules": " ".join(sorted(bucket)),
            })

    return shards


def main():
    if not (REPO_ROOT / "torchtitan").is_dir():
        sys.exit(f"Error: {REPO_ROOT} is not a torchtitan checkout")

    tests = discover_tests()
    if not any(tests.values()):
        sys.exit("Error: No tests found")


    durations = json.loads(DURATIONS_PATH.read_text())

    matrix = {
        "distributed_shards": chunk_tests(
            tests["distributed"], NUM_DISTRIBUTED_TPU_SHARDS, durations
        ),
        "single_tpu_shards": chunk_tests(
            tests["single_tpu"], NUM_SINGLE_TPU_SHARDS, durations
        ),
        "cpu_shards": chunk_tests(tests["cpu"], NUM_CPU_SHARDS, durations),
        "pytest_shards": chunk_tests(tests["pytest"], NUM_PYTEST_SHARDS, durations),
        "pytest_deselect": " ".join(f"--deselect={test}" for test in DESELECTED_TESTS),
    }
    
    # We output a JSON object of shard arrays to stdout
    print(json.dumps(matrix))

if __name__ == "__main__":
    main()
