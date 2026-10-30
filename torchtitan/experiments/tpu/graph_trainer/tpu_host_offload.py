"""Offload the one activation a scan over decoder layers keeps alive.

Under FullAC every layer saves its own input, so backward can reuse it.

Unrolled, that is ``n_layers`` separate tensors. They all pile up during the
forward, but backward then frees them one at a time, as each layer's backward
finishes.

Under scan there is only one loop, so those saves come out as one stacked
tensor. One tensor cannot be freed a row at a time, so all ``n_layers`` rows
stay alive until the backward loop ends:

    forward scan                    backward scan
    +--------------------+          +--------------------+
    | iter 0   saves x0  |          | iter 0   reads xN  |
    | iter 1   saves x1  |          | iter 1   reads x.. |
    |    ...             |          |    ...             |
    | iter n-1 saves xN  |          | iter n-1 reads x0  |
    +--------------------+          +--------------------+
              |                                ^
              v                                |
              +--> [n_layers, b, s, d] -------+
                   one tensor, live across both loops

At 405B seq 8192 that is 33.8 GB -- the whole of the model's sequence-length
dependence, and on its own the reason seq 8192 does not fit at any batch size.

This pass inserts one ``ao.offload`` inside the forward body and one
``ao.reload`` inside the backward body. The loop supplies the repetition: two
inserted ops become ``2 * n_layers`` transfers of one slice each, and the
device never holds more than one slice.

    forward body                      backward body
      x     = layer input               i     = loop counter
      out   = layer(x)                  s     = index_select(stack, i)  # host
      saved = ao.offload(x)             s     = ao.reload(s)            # device
      return (out, saved)               use(s)

The carry and the save are two different outputs holding the same value. Only
the save is offloaded; ``out`` stays on device for the next iteration, so no
iteration ever has to wait for a host round trip.

``cpu_offload.apply_cpu_offload_pass`` cannot do this. It places ``ao.reload``
immediately before the first backward consumer, and here that consumer *is the
loop node*, so one reload lands all ``n_layers`` slices on device before
iteration 0 -- a regression, not a missed win, since the run then pays the host
round trip and holds the whole stack anyway. Nor can the XLA scheduler recover
it: splitting one transfer into ``n_layers`` transfers changes which ops exist,
not when they run.

Two asymmetries decide where the ops go, and both are the opposite of the
obvious guess:

  * The forward's save is a ``ys`` output, so the body already emits one slice
    per iteration. The offload goes on that body output.
  * The backward does *not* receive it as ``xs``. It arrives in
    ``additional_inputs`` as the whole stack, and the body indexes it with the
    counter ``xs`` carries. So the reload goes on the output of that in-body
    index, not on the placeholder -- reloading the placeholder would pull all
    ``n_layers`` slices back every iteration.

Offloading moves the stack to host without telling anyone else who reads it, so
the backward loop has to be its only reader. Anything else and the pass declines
and says so.

Picking the tensor needs no layer names, only what the two loops do with it: the
forward scan produced it and the backward scan reads it. That pair is what a
per-layer save *is*. It rules out the loss logits, which no backward scan reads,
and it rules out anything computed outside the loop and handed to both -- a RoPE
table, a mask. Size then only breaks ties between genuine saves, rather than
being the thing that separates the layer input from a lookup table.
"""

import operator

import torch

# Registering ao::offload / ao::reload / ao::wait_tensor is an import side
# effect, and this module inserts them without going through cpu_offload.py.
import torch._functorch._activation_offloading.offload_ops  # noqa: F401
import torch.fx
from torch.fx import Node

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _can_offload_node,
    _has_recompute_consumer,
    _is_view,
    _tensor_bytes,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _full_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.registry import (
    register_memory_policy,
)
from torchtitan.tools.logging import logger

_SCAN_HOP = torch.ops.higher_order.scan
_POLICY = "full_and_scan_offload"


def _scan_body(gm: torch.fx.GraphModule, scan_node: Node) -> torch.fx.GraphModule:
    """The loop body of ``scan_node``.

    ``trace_scan`` registers the traced ``combine_fn`` as a submodule of the
    root graph and passes it as argument 0, so the body is reachable -- and
    mutable -- from a pass that only holds the parent graph.
    """
    return getattr(gm, scan_node.args[0].target)


def _only_read_by(node: Node, scan: Node) -> bool:
    """Whether ``scan`` is the only thing that reads ``node`` or a view of it.

    Offloading makes ``node`` host-resident and does not tell its readers, so a
    second one would silently be handed host memory. Dead nodes do not count:
    tracing a ``scan`` slices element 0 off every operand to build the body, and
    those slices stay in the graph with nothing reading them.
    """
    pending = [node]
    seen: set[Node] = set()
    while pending:
        for user in pending.pop().users:
            if user is scan or user in seen:
                continue
            seen.add(user)
            if not user.users:
                continue
            if not _is_view(user):
                return False
            pending.append(user)
    return True


def _find_stack_producer(node: Node) -> tuple[Node, int] | None:
    """The forward scan that produced ``node``, and which output it was.

    A scan node returns ``(*carry, *ys)`` and its body returns the same tuple,
    so the ``getitem`` index is already the body's output position.
    """
    seen: set[Node] = set()
    cur: object = node
    while isinstance(cur, Node) and cur not in seen:
        seen.add(cur)
        if cur.target is operator.getitem and isinstance(cur.args[0], Node):
            parent = cur.args[0]
            if parent.target is _SCAN_HOP and not _is_backward_node(parent):
                return parent, cur.args[1]
        if _is_view(cur) and cur.args and isinstance(cur.args[0], Node):
            cur = cur.args[0]
            continue
        return None
    return None


def _find_stack_consumer(node: Node) -> tuple[Node, int] | None:
    """The backward scan that reads ``node``, and its body placeholder index.

    ``scan``'s signature is ``(combine_fn, init, xs, additional_inputs)`` and
    the body's placeholders are ``(*carry, *xs_slices, *additional_inputs)``,
    so an operand's position in either list maps straight onto a placeholder.
    Both lists are searched because the stack arrives in ``additional_inputs``
    while ``xs`` carries only the loop counter.
    """
    seen: set[Node] = set()

    def walk(n: Node) -> tuple[Node, int] | None:
        for user in n.users:
            if user in seen:
                continue
            seen.add(user)
            if user.target is _SCAN_HOP and _is_backward_node(user):
                n_init, xs, additional = len(user.args[1]), user.args[2], user.args[3]
                for i, x in enumerate(xs):
                    if x is n:
                        return user, n_init + i
                for i, x in enumerate(additional):
                    if x is n:
                        return user, n_init + len(xs) + i
                continue
            if _is_view(user):
                found = walk(user)
                if found is not None:
                    return found
        return None

    return walk(node)


def _select_stacked_activation(gm: torch.fx.GraphModule) -> Node | None:
    """The largest value the forward loop saved and the backward loop reads.

    Both loop tests are needed. Without the producer test a tensor computed
    *outside* the loop and passed into both scans -- a RoPE table, a mask -- is
    a candidate, and only its size keeps it from being picked. Requiring that
    the forward scan produced it leaves just the per-layer saves, so size is a
    tiebreak between those rather than the thing doing the work.
    """
    best: Node | None = None
    best_bytes = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        if not _can_offload_node(node):
            continue
        if _has_recompute_consumer(node):
            # AC will regenerate it; there is nothing to keep alive.
            continue
        consumer = _find_stack_consumer(node)
        if consumer is None:
            continue
        if _find_stack_producer(node) is None:
            continue
        if not _only_read_by(node, consumer[0]):
            logger.warning(
                "scan offload: %s is read by something other than the backward "
                "scan, which would then read host memory. Leaving it on device",
                node.name,
            )
            continue
        nbytes = _tensor_bytes(node.meta["val"])
        if nbytes > best_bytes:
            best, best_bytes = node, nbytes
    return best


def _offload_body_output(body: torch.fx.GraphModule, index: int) -> None:
    """Emit body output ``index`` on host instead of on device."""
    output = next(n for n in body.graph.nodes if n.op == "output")
    outs = list(output.args[0])
    produced = outs[index]
    val = produced.meta["val"]

    with body.graph.inserting_after(produced):
        offload = body.graph.call_function(
            torch.ops.ao.offload.default, args=(produced,)
        )
        offload.meta["val"] = val.to("cpu")
    with body.graph.inserting_after(offload):
        # The keepalive argument holds the device storage until the D2H
        # completes. Inside the body the producer is its own last consumer.
        wait = body.graph.call_function(
            torch.ops.ao.wait_tensor.default, args=(offload, produced, produced)
        )
        wait.meta["val"] = offload.meta["val"]

    outs[index] = wait
    output.args = (tuple(outs),)
    body.graph.lint()
    body.recompile()


def _body_placeholder(body: torch.fx.GraphModule, index: int) -> Node:
    return [n for n in body.graph.nodes if n.op == "placeholder"][index]


def _reads_one_slice(body: torch.fx.GraphModule, index: int) -> bool:
    """Whether body placeholder ``index`` is read a slice at a time.

    The placeholder is the whole stack. A body that uses it any other way would
    have the whole thing reloaded every iteration, which is worse than not
    offloading -- so the caller declines and says so rather than doing that
    silently. A user with no ``val`` is not a slice we can reason about.
    """
    param = _body_placeholder(body, index)
    users = list(param.users)
    if not users:
        return False
    whole = param.meta["val"].numel()
    return all(
        u.meta.get("val") is not None and u.meta["val"].numel() < whole
        for u in users
    )


def _reload_body_slice(body: torch.fx.GraphModule, index: int) -> None:
    """Reload the per-iteration slice of body placeholder ``index``.

    The reload goes on the output of the op that indexes the stack, not on the
    placeholder itself, which is the whole stack.
    """
    param = _body_placeholder(body, index)
    users = list(param.users)

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


def offload_scan_activations(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Keep the scan's stacked layer input on host between the two loops."""
    node = _select_stacked_activation(gm)
    if node is None:
        # Asking for this policy is asking for the stacked save to go to host.
        # Warning instead would leave a run that is correct, uses exactly the
        # memory it would have used with the policy off, and reports a
        # plausible number -- so the mistake surfaces as a perf result nobody
        # can explain. The scan counts say which mistake it is: zero means the
        # run has no scan to offload from, nonzero means the selector could not
        # reach the save and this pass is at fault.
        scans = [n for n in gm.graph.nodes if n.target is _SCAN_HOP]
        raise RuntimeError(
            f"memory_policy={_POLICY!r} found nothing to offload. Saw "
            f"{len(scans)} scan nodes, "
            f"{sum(1 for n in scans if _is_backward_node(n))} in the backward. "
            "With no scan layers this policy has no stacked activation to move; "
            "run with --compile.use-scan-layers or pick another policy."
        )

    fwd_scan, out_index = _find_stack_producer(node)
    bwd_scan, param_index = _find_stack_consumer(node)

    bwd_body = _scan_body(gm, bwd_scan)
    if not _reads_one_slice(bwd_body, param_index):
        logger.warning(
            "scan offload: the backward body uses %s whole rather than one "
            "slice per iteration, so offloading it would reload the entire "
            "stack every step. Leaving it on device.",
            node.name,
        )
        return gm

    _offload_body_output(_scan_body(gm, fwd_scan), out_index)
    _reload_body_slice(bwd_body, param_index)

    # The stacked tensor now lives on host, so retype it and every view of it
    # to keep ``meta["val"]`` agreeing with what the backend will lower. Every
    # view, not one chain: a view is another name for the same storage, so one
    # left saying device is one stale label.
    pending = [node]
    while pending:
        cur = pending.pop()
        cur.meta["val"] = cur.meta["val"].to("cpu")
        pending.extend(u for u in cur.users if _is_view(u))

    logger.info(
        "scan offload: %s %s -> host inside the loop bodies, %.2f GiB",
        node.name,
        tuple(node.meta["val"].shape),
        _tensor_bytes(node.meta["val"]) / (1024**3),
    )
    return gm


@register_memory_policy(_POLICY)
def _full_and_scan_offload_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: "GraphTrainer.Config",
) -> torch.fx.GraphModule:
    """FullAC, then keep the largest saved activation on the host.

    FullAC first because it decides *what* that activation is: with anything
    less, the forward saves more per layer and no single tensor dominates.
    """
    _full_memory_policy_pass(gm, config=config)
    offload_scan_activations(gm)
    return gm
