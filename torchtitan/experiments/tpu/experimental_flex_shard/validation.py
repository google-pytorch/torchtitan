"""Actual dense Qwen3 parity and joint-graph replay validation.

B: document batch, L: tokens per single document, V: vocabulary. Each rank owns
one document; the CPU global-batch reference evaluates documents separately so
causal attention never crosses a document or rank boundary.
"""

import copy
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torch.distributed.device_mesh import init_device_mesh
from torch.fx import GraphModule

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.experiments.tpu.experimental_flex_shard.dependency import (
    load_executor,
    verify_core,
)
from torchtitan.experiments.tpu.experimental_flex_shard.trace import (
    apply_sgd_outputs,
    capture_sgd_step,
)
from torchtitan.models.qwen3 import model_registry


def build_initialized_model(*, device: str = "cpu"):
    """Build the fork's unmodified three-layer dense Qwen3 testmodel."""
    from functools import partial

    config = model_registry(
        "testmodel", seq_len=16, attn_backend="sdpa", fuse_qkv=False
    ).model
    config.enable_weight_tying = False
    config.tok_embeddings.param_init = {
        "weight": partial(torch.nn.init.normal_, std=0.02)
    }
    torch.manual_seed(42)
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device=device)
    model.init_weights(buffer_device=torch.device(device))
    return model


def _local_slice(tensor, parameter, rank, world):
    placement = parameter._placements[0]
    if placement.dim is None:
        return tensor
    chunk = math.ceil(tensor.shape[placement.dim] / world)
    start = min(rank * chunk, tensor.shape[placement.dim])
    size = min(chunk, tensor.shape[placement.dim] - start)
    return tensor.narrow(placement.dim, start, size)


def _op_counts(graph):
    counts = {
        "all_gather": 0,
        "reduce_scatter": 0,
        "all_reduce": 0,
        "wait": 0,
        "native_fsdp_copy": 0,
    }
    for module in graph.modules():
        if not isinstance(module, GraphModule):
            continue
        for node in module.graph.nodes:
            name = str(node.target)
            if "all_gather_into_tensor" in name:
                counts["all_gather"] += 1
            if "reduce_scatter_tensor" in name:
                counts["reduce_scatter"] += 1
            if "_c10d_functional.all_reduce" in name:
                counts["all_reduce"] += 1
            if "_c10d_functional.wait_tensor" in name:
                counts["wait"] += 1
            if "fsdp." in name and "copy" in name:
                counts["native_fsdp_copy"] += 1
    return counts


def validate_worker(*, device: str, checkpointed: bool, output: str, steps: int = 10):
    """Prove real-rank gradients, updates, and changed-input joint graph replay."""
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2:
        raise ValueError("Distributed validation needs at least two actual ranks")
    reference = build_initialized_model()
    model = copy.deepcopy(reference).to(device)
    executor = load_executor()
    mesh = init_device_mesh(device, (world,), mesh_dim_names=("fsdp",))
    executor.data_parallel(model, mesh, mode="fully_shard")
    if checkpointed:
        from functools import partial

        from torch.utils.checkpoint import checkpoint

        for layer in model.layers.values():
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
    output_path = (
        Path(output) / ("checkpoint" if checkpointed else "plain") / f"rank{rank}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    loss_fn = CrossEntropyLoss.Config().build()
    lr = torch.tensor(0.01, device=device)
    global_valid_tokens = torch.tensor(world * 16, device=device)
    cpu_global_valid_tokens = global_valid_tokens.cpu()
    generator = torch.Generator().manual_seed(42)
    # Changed input replay, fixed deterministic documents across runs.
    all_inputs_BL = torch.randint(2048, (steps, world, 16), generator=generator)
    all_labels_BL = torch.randint(2048, (steps, world, 16), generator=generator)
    traced, runner = capture_sgd_step(
        model,
        loss_fn,
        all_inputs_BL[0, rank].to(device),
        all_labels_BL[0, rank].to(device),
        global_valid_tokens,
        lr,
    )
    (output_path / "joint.fx.txt").write_text(
        traced.gm.print_readable(print_output=False, include_device=True)
    )
    counts = _op_counts(traced.gm)
    assert counts["all_gather"] > 0 and counts["reduce_scatter"] > 0
    assert counts["native_fsdp_copy"] == 0
    assert all(node.op != "call_module" for node in traced.gm.graph.nodes)
    # Keep an inspectable FX graph; TPU execution uses the same graph directly
    # through StaticCompiler, not Dynamo's fused-backward remat route.
    if device == "tpu":
        from torch_tpu._internal.compile.compiler import StaticCompiler

        compiled = StaticCompiler()(traced.gm, traced.example_inputs)
        traced.gm = compiled
    parameter_ids = [id(parameter) for parameter in model.parameters()]
    results = []
    for step in range(steps):
        reference.zero_grad(set_to_none=True)
        cpu_losses = []
        for document_rank in range(world):
            logits_LV = reference(all_inputs_BL[step, document_rank])
            loss, _ = loss_fn(
                logits_LV, all_labels_BL[step, document_rank], cpu_global_valid_tokens
            )
            loss.backward()
            cpu_losses.append(loss.detach())
        inputs_L = all_inputs_BL[step, rank].to(device)
        labels_L = all_labels_BL[step, rank].to(device)
        loss, gradients, updates = runner(inputs_L, labels_L, global_valid_tokens, lr)
        if device == "tpu":
            torch.accelerator.synchronize()
        torch.testing.assert_close(loss.cpu(), cpu_losses[rank], rtol=1e-5, atol=1e-6)
        max_grad_error = 0.0
        max_update_error = 0.0
        references = dict(reference.named_parameters())
        for (name, parameter), gradient, update in zip(
            model.named_parameters(), gradients, updates, strict=True
        ):
            expected_grad = _local_slice(references[name].grad, parameter, rank, world)
            torch.testing.assert_close(
                gradient.cpu(), expected_grad, rtol=1e-4, atol=1e-6
            )
            expected_update = -0.01 * expected_grad
            actual_update = update.cpu() - parameter.detach().cpu()
            torch.testing.assert_close(
                actual_update, expected_update, rtol=1e-4, atol=1e-7
            )
            max_grad_error = max(
                max_grad_error, float((gradient.cpu() - expected_grad).abs().max())
            )
            max_update_error = max(
                max_update_error, float((actual_update - expected_update).abs().max())
            )
        apply_sgd_outputs(model, gradients, updates)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.add_(parameter.grad, alpha=-0.01)
        for name, parameter in model.named_parameters():
            expected = _local_slice(references[name], parameter, rank, world)
            torch.testing.assert_close(parameter.cpu(), expected, rtol=1e-5, atol=2e-6)
        assert parameter_ids == [id(parameter) for parameter in model.parameters()]
        global_loss = funcol.all_reduce(loss.detach(), "sum", dist.group.WORLD)
        if device == "tpu":
            torch.accelerator.synchronize()
        torch.testing.assert_close(
            global_loss.cpu(), sum(cpu_losses), rtol=1e-5, atol=1e-6
        )
        results.append(
            {
                "step": step,
                "loss": float(global_loss.cpu()),
                "max_grad_error": max_grad_error,
                "max_update_error": max_update_error,
            }
        )
    summary = {
        "status": "PASS",
        "device": device,
        "rank": rank,
        "world": world,
        "model": "Qwen3 testmodel, 3 layers, dim=128, vocab=2048, CosSinRoPE, SDPA",
        "core_identity": verify_core(),
        "torch": torch.__version__,
        "checkpointed": checkpointed,
        "steps": results,
        "joint_graph_op_counts": counts,
        "explicit_parameter_inputs": len(list(model.parameters())),
        "capture": "TorchTitan minimal_fx_tracer, forward/loss/autograd/functional SGD",
        "parameter_identities_preserved": True,
        "not_validated": [
            "full FlexShard CUDA runtime",
            "HSDP/TP/EP/CP/PP",
            "offload",
            "performance",
            "peak memory",
            "BF16 accuracy",
        ],
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        "TRACEABLE_QWEN_PASS "
        + json.dumps({k: v for k, v in summary.items() if k != "steps"}),
        flush=True,
    )
    dist.barrier()
