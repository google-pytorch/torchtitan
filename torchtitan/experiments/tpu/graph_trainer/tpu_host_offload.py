"""Keep named scan saves on host between the scan's two loops.

The forward scan stacks each value it saves for backward into one
``[n_layers, ...]`` tensor, live until the backward loop ends. This pass adds
one host copy per iteration inside each loop body, so the device holds one
row at a time:

    forward body                      backward body
      x     = layer input               i     = loop counter
      out   = layer(x)                  s     = index_select(stack, i)  # host
      saved = ao.offload(clone(x))      s     = ao.reload(s)            # device
      return (out, saved, ...)          use(s)

Which saves go to host is chosen by name (``compile.scan_offload``):

* ``layer_input``: the body output ``clone(x)``, where ``x`` is the carry
  placeholder. Scan saves a body input this way, so no region label can name it.
* a save region, e.g. ``attention.inner_attention``: the body outputs whose
  op carries that ``remat_region`` label. ``scan_with_remat_policy`` saves it.

Saves not named stay in HBM.

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
from torchtitan.experiments.tpu.components.remat_regions import REMAT_REGION
from torchtitan.tools.logging import logger

_SCAN_HOP = torch.ops.higher_order.scan
# The ``compile.scan_offload`` name of the saved carry.
LAYER_INPUT = "layer_input"


def _scan_body(gm: torch.fx.GraphModule, scan_node: Node) -> torch.fx.GraphModule:
    # `trace_scan` registers the body as a submodule of the root graph.
    return getattr(gm, scan_node.args[0].target)


def _outputs(body: torch.fx.GraphModule) -> list:
    return list(next(iter(body.graph.find_nodes(op="output"))).args[0])


def _placeholder(body: torch.fx.GraphModule, index: int) -> Node:
    return body.graph.find_nodes(op="placeholder")[index]


def _is_saved_carry(body: torch.fx.GraphModule, node: object) -> bool:
    # How scan returns a body input that min-cut saved; the carry is input 0.
    return (
        isinstance(node, Node)
        and node.target is torch.ops.aten.clone.default
        and node.args[0] is _placeholder(body, 0)
    )


def _is_named_save(body: torch.fx.GraphModule, node: object, name: str) -> bool:
    if name == LAYER_INPUT:
        return _is_saved_carry(body, node)
    # Scan returns a saved op output as is, with the op's region label.
    return (
        isinstance(node, Node)
        and (node.meta.get("custom") or {}).get(REMAT_REGION) == name
    )


def _only_read_by(node: Node, scan: Node) -> bool:
    """Whether ``scan`` is the only live reader of ``node`` or a view of it.

    Any other reader would silently get host memory. Dead users don't count:
    tracing ``scan`` leaves unused element-0 slices of its operands.
    """
    pending, seen = [node], set()
    while pending:
        for user in pending.pop().users:
            # The graph output has no users but is a real reader.
            if user.op == "output":
                return False
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


def _offload_one(
    gm: torch.fx.GraphModule, fwd_scan: Node, index: int, name: str
) -> None:
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
        "scan offload: %s %s %s -> host inside the loop bodies, %.2f GiB",
        name,
        stack.name,
        tuple(stack.meta["val"].shape),
        _tensor_bytes(stack.meta["val"]) / (1024**3),
    )


def offload_scan_activations(
    gm: torch.fx.GraphModule, names: tuple[str, ...] = (LAYER_INPUT,)
) -> torch.fx.GraphModule:
    """Move each forward scan's saves named in ``names`` to host, inside the loops.

    Raises if a name matches no save: asking for the offload and silently not
    getting it gives a correct run at the wrong memory.
    """
    found = dict.fromkeys(names, 0)
    forward_scans = 0
    for node in gm.graph.find_nodes(op="call_function", target=_SCAN_HOP):
        if _is_backward_node(node):
            continue
        forward_scans += 1
        body = _scan_body(gm, node)
        outs = _outputs(body)
        for name in names:
            saved = [i for i, o in enumerate(outs) if _is_named_save(body, o, name)]
            if name == LAYER_INPUT and len(saved) > 1:
                raise RuntimeError(f"scan offload: {node.name} saves its carry twice")
            for index in saved:
                _offload_one(gm, node, index, name)
                found[name] += 1
    if not forward_scans:
        raise RuntimeError(
            "scan offload found no forward scan; it needs "
            "--compile.use-scan-layers."
        )
    if found.get(LAYER_INPUT) == 0:
        raise RuntimeError(
            f"scan offload: {forward_scans} forward scans, none saves its "
            "carry (is the layer input recomputed instead of saved?)"
        )
    missing = [name for name, count in found.items() if not count]
    if missing:
        raise RuntimeError(
            f"scan offload: no forward scan saves region(s) {missing} "
            "(is each one in compile.save_regions?)"
        )
    return gm


@register_memory_policy("full_and_scan_offload")
def _full_and_scan_offload_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """Full recompute, then keep the scan's saved layer input on host.

    ``compile.scan_offload`` can only name ``layer_input`` here: full recompute
    saves no region.
    """
    _full_memory_policy_pass(gm, config=config)
    offload_scan_activations(gm, tuple(config.compile.scan_offload))
    return gm
