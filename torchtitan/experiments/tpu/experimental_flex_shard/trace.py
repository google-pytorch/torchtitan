"""Capture and replay the actual TorchTitan model/loss/backward with local SGD.

SGD math is captured in the joint FX graph. Installation and update copies are
outside capture, preserving the byte-backed Parameter storage identities.
"""

import torch
from torch import nn

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.trainer import make_fwd_bwd_step


def capture_sgd_step(
    model: nn.Module, loss_fn, inputs, labels, global_valid_tokens, lr
):
    """Return the unchanged TorchTitan tracer result and its state-lifting runner."""
    trainable = tuple(model.named_parameters())
    all_parameters = tuple(model.named_parameters(remove_duplicate=False))
    if len(trainable) != len(all_parameters) or any(
        not parameter.requires_grad for _, parameter in trainable
    ):
        raise ValueError(
            "Joint-SGD demonstration requires untied, trainable parameters only"
        )
    forward_backward = make_fwd_bwd_step(model, loss_fn)
    num_parameters = len(trainable)

    def forward_backward_sgd(inputs, labels, global_valid_tokens, lr):
        with torch.fx.traceback.annotate({"spmd_safe": True}):
            values = forward_backward(inputs, labels, global_valid_tokens, {})
            loss, *gradients = values
            parameters = tuple(model.parameters())
            updates = tuple(
                parameter - lr * gradient
                for parameter, gradient in zip(parameters, gradients, strict=True)
            )
            return loss, tuple(gradients), updates

    traced = minimal_fx_tracer(forward_backward_sgd, module=model)(
        inputs, labels, global_valid_tokens, lr
    )
    runner = run_traced(traced, module=model, _validate_runtime=True)
    assert num_parameters > 0
    return traced, runner


def apply_sgd_outputs(model: nn.Module, gradients, updates) -> None:
    """Copy graph outputs into the existing shards without replacing Parameters."""
    with torch.no_grad():
        for parameter, gradient, updated in zip(
            model.parameters(), gradients, updates, strict=True
        ):
            parameter.grad = gradient
            parameter.copy_(updated)
