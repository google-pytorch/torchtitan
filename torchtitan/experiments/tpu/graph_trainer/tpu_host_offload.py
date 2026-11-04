"""Keep the scan's saved layer input on host between its two loops.

The forward scan stacks the layer input it saves for backward into one
``[n_layers, ...]`` tensor, live until the backward loop ends. This pass adds
one host copy per iteration inside each loop body, so the device holds one
row at a time:

    forward body                      backward body
      x     = layer input               i     = loop counter
      out   = layer(x)                  s     = index_select(stack, i)  # host
      saved = ao.offload(clone(x))      s     = ao.reload(s)            # device
      return (out, saved, ...)          use(s)

The saved copy is the one ``scan_remat_policy`` tags ``layer_input``. Other
saved values (attention output, MoE counts) stay in HBM.

``cpu_offload.apply_cpu_offload_pass`` can't do this: it reloads before the
first backward user, which is the backward scan node, so all rows would land
on device at once.
"""

import operator

import torch

# Registering ao::offload / ao::reload / ao::wait_tensor is an import side
# effect, and this module inserts them without going through cpu_offload.py.
import torch._functorch._activation_offloading.offload_ops  # noqa: F401
import torch.fx
from torch.fx import Node

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.cpu_offload import _is_view, _tensor_bytes
from torchtitan.experiments.graph_trainer.memory_policy import (
    _full_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.registry import register_memory_policy
from torchtitan.experiments.tpu.remat_regions import LAYER_INPUT, REMAT_REGION
from torchtitan.tools.logging import logger

_SCAN_HOP = torch.ops.higher_order.scan


def _scan_body(gm: torch.fx.GraphModule, scan_node: Node) -> torch.fx.GraphModule:
    # `trace_scan` registers the body as a submodule of the root graph.
    return getattr(gm, scan_node.args[0].target)


def _outputs(body: torch.fx.GraphModule) -> list:
    return list(next(iter(body.graph.find_nodes(op="output"))).args[0])


def _placeholder(body: torch.fx.GraphModule, index: int) -> Node:
    return body.graph.find_nodes(op="placeholder")[index]


def _is_layer_input(node: object) -> bool:
    custom = node.meta.get("custom") if isinstance(node, Node) else None
    return (custom or {}).get(REMAT_REGION) == LAYER_INPUT


def _only_read_by(node: Node, scan: Node) -> bool:
    """Whether ``scan`` is the only live reader of ``node`` or a view of it.

    Any other reader would silently get host memory. Dead users don't count:
    tracing ``scan`` leaves unused element-0 slices of its operands.
    """
    pending, seen = [node], set()
    while pending:
        for user in pending.pop().users:
            if user is scan or user in seen or not user.users:
                continue
            seen.add(user)
            if not _is_view(user):
                return False
            pending.append(user)
    return True


def _backward_reader(stack: Node) -> tuple[Node, int]:
    """The backward scan that reads ``stack``, and its body placeholder index.

    Body placeholders are ``(*init, *xs, *additional_inputs)``, the same order
    as the scan's operands.
    """
    pending, seen = [stack], set()
    while pending:
        node = pending.pop()
        for user in node.users:
            if user in seen:
                continue
            seen.add(user)
            if user.target is _SCAN_HOP and _is_backward_node(user):
                operands = [*user.args[1], *user.args[2], *user.args[3]]
                if node in operands:
                    return user, operands.index(node)
            elif _is_view(user):
                pending.append(user)
    raise RuntimeError(f"scan offload: no backward scan reads {stack.name}")


def _offload_body_output(body: torch.fx.GraphModule, index: int) -> None:
    """Emit body output ``index`` on host instead of on device."""
    output = next(iter(body.graph.find_nodes(op="output")))
    outs = _outputs(body)
    produced = outs[index]
    with body.graph.inserting_after(produced):
        offload = body.graph.call_function(
            torch.ops.ao.offload.default, args=(produced,)
        )
        offload.meta["val"] = produced.meta["val"].to("cpu")
    with body.graph.inserting_after(offload):
        # The keepalive argument holds the device storage until the copy is done.
        wait = body.graph.call_function(
            torch.ops.ao.wait_tensor.default, args=(offload, produced, produced)
        )
        wait.meta["val"] = offload.meta["val"]
    outs[index] = wait
    output.args = (tuple(outs),)
    body.graph.lint()
    body.recompile()


def _reload_body_slice(body: torch.fx.GraphModule, index: int) -> None:
    """Reload the row that the body indexes out of host placeholder ``index``.

    The reload goes after the index, not on the placeholder: reloading the
    placeholder would copy the whole stack back every iteration.
    """
    param = _placeholder(body, index)
    whole = param.meta["val"].numel()
    users = list(param.users)
    if not users or any(
        u.meta.get("val") is None or u.meta["val"].numel() >= whole for u in users
    ):
        raise RuntimeError(
            f"scan offload: the backward body reads {param.name} whole, not one "
            "row per iteration"
        )
    for user in users:
        val = user.meta["val"]
        with body.graph.inserting_after(user):
            reload = body.graph.call_function(
                torch.ops.ao.reload.default,
                args=(user, val.device, list(val.size()), list(val.stride())),
            )
            reload.meta["val"] = val
        with body.graph.inserting_after(reload):
            wait = body.graph.call_function(
                torch.ops.ao.wait_tensor.default, args=(reload, user)
            )
            wait.meta["val"] = val
        for consumer in list(user.users):
            if consumer not in (reload, wait):
                consumer.replace_input_with(user, wait)
        user.meta["val"] = val.to("cpu")
    param.meta["val"] = param.meta["val"].to("cpu")
    body.graph.lint()
    body.recompile()


def _offload_one(gm: torch.fx.GraphModule, fwd_scan: Node, index: int) -> None:
    # A scan node returns what its body returns, so getitem index = body output.
    stacks = [
        u for u in fwd_scan.users
        if u.target is operator.getitem and u.args[1] == index
    ]
    if len(stacks) != 1:
        raise RuntimeError(
            f"scan offload: expected one reader of {fwd_scan.name}[{index}], "
            f"saw {len(stacks)}"
        )
    (stack,) = stacks
    bwd_scan, param_index = _backward_reader(stack)
    if not _only_read_by(stack, bwd_scan):
        raise RuntimeError(
            f"scan offload: {stack.name} is read outside the backward scan, "
            "which would then read host memory"
        )
    _reload_body_slice(_scan_body(gm, bwd_scan), param_index)
    _offload_body_output(_scan_body(gm, fwd_scan), index)
    # Keep `meta["val"]` of the stack and its views in step with the backend.
    pending = [stack]
    while pending:
        node = pending.pop()
        node.meta["val"] = node.meta["val"].to("cpu")
        pending.extend(u for u in node.users if _is_view(u))
    logger.info(
        "scan offload: %s %s -> host inside the loop bodies, %.2f GiB",
        stack.name,
        tuple(stack.meta["val"].shape),
        _tensor_bytes(stack.meta["val"]) / (1024**3),
    )


def offload_scan_activations(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Move each forward scan's saved ``layer_input`` to host, inside the loops.

    Raises if there is none: asking for the offload and silently not getting
    it gives a correct run at the wrong memory.
    """
    found = 0
    for node in gm.graph.find_nodes(op="call_function", target=_SCAN_HOP):
        if _is_backward_node(node):
            continue
        outs = _outputs(_scan_body(gm, node))
        tagged = [i for i, out in enumerate(outs) if _is_layer_input(out)]
        if len(tagged) > 1:
            raise RuntimeError(f"scan offload: {node.name} saves its carry twice")
        for index in tagged:
            _offload_one(gm, node, index)
            found += 1
    if not found:
        raise RuntimeError(
            "scan offload found no saved layer input to offload. It needs "
            "--compile.use-scan-layers, and a scan body tagged by "
            "scan_remat_policy."
        )
    return gm


@register_memory_policy("full_and_scan_offload")
def _full_and_scan_offload_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Full recompute, then keep the scan's saved layer input on host."""
    _full_memory_policy_pass(gm, config=config)
    offload_scan_activations(gm)
    return gm
