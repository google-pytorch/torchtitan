"""Keep in-place writes to graph inputs that the TPU compiler would drop.

The traced step can write a module buffer in place, e.g. ``MoE.forward``'s
``tokens_per_expert_E.add_(counts)``. The TPU compiler lowers that write as a
pure op whose result nothing reads, so it is dead code and the buffer never
changes. With load balancing on, the counts stay zero and the expert bias never
moves -- with no error.

``functionalize_input_mutations`` turns each such write into a pure op whose
result becomes an extra graph output. ``CopyBackInputMutations`` wraps the
compiled graph: it copies those outputs into the inputs and returns the rest.
"""

import torch
from torch import fx

__all__ = ["CopyBackInputMutations", "functionalize_input_mutations"]


def _writes_first_arg(node: fx.Node) -> bool:
    target = node.target
    if not isinstance(target, torch._ops.OpOverload):
        return False
    args = target._schema.arguments
    return bool(args) and args[0].alias_info is not None and args[0].alias_info.is_write


def _pure_twin(target: torch._ops.OpOverload) -> torch._ops.OpOverload:
    """``aten.add_.Tensor`` -> ``aten.add.Tensor``."""
    name = target.overloadpacket.__name__
    if not name.endswith("_"):
        raise NotImplementedError(f"no pure form for in-place {target}")
    return getattr(getattr(torch.ops.aten, name[:-1]), target._overloadname)


def functionalize_input_mutations(gm: fx.GraphModule) -> list[int]:
    """Rewrite in-place writes to placeholders as outputs; return their positions.

    Returns the flat input index of each written placeholder, in the order its
    final value was appended to the graph's outputs.

    Raises:
        NotImplementedError: if an input is written through a view, which this
            pass cannot follow.
    """
    graph = gm.graph
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    index = {p: i for i, p in enumerate(placeholders)}
    latest: dict[fx.Node, fx.Node] = {}
    stands_for: dict[fx.Node, fx.Node] = {}
    for node in list(graph.nodes):
        if node.op != "call_function" or not _writes_first_arg(node):
            continue
        dst = node.args[0]
        if not isinstance(dst, fx.Node):
            continue
        # A second write to the same input lands on the first write's result.
        written = dst if dst.op == "placeholder" else stands_for.get(dst)
        if written is None:
            base = _view_base(dst)
            if base.op == "placeholder" and base is not dst:
                raise NotImplementedError(
                    f"{node.target} writes graph input {base.name} through a view"
                )
            continue
        with graph.inserting_after(node):
            pure = graph.call_function(_pure_twin(node.target), node.args, node.kwargs)
        pure.meta = dict(node.meta)
        # Later readers of the input, and of the in-place op's result, see the
        # new value.
        node.replace_all_uses_with(pure)
        for user in list(dst.users):
            if user is not pure and _after(pure, user):
                user.replace_input_with(dst, pure)
        graph.erase_node(node)
        latest[written] = pure
        stands_for[pure] = written
    if not latest:
        return []
    output = next(n for n in graph.nodes if n.op == "output")
    outs = output.args[0]
    output.args = (type(outs)([*outs, *latest.values()]),)
    graph.lint()
    gm.recompile()
    return [index[p] for p in latest]


def _view_base(node: fx.Node) -> fx.Node:
    """Follow view ops (``select``, ``view``, ...) back to the tensor they alias."""
    while node.op == "call_function" and isinstance(node.target, torch._ops.OpOverload):
        returns = node.target._schema.returns
        alias = returns[0].alias_info if returns else None
        if alias is None or alias.is_write or not isinstance(node.args[0], fx.Node):
            break
        node = node.args[0]
    return node


def _after(a: fx.Node, b: fx.Node) -> bool:
    node = a.next
    while node.op != "root":
        if node is b:
            return True
        node = node.next
    return False


class CopyBackInputMutations:
    """Run the compiled graph, then write its extra outputs back to the inputs."""

    def __init__(self, compiled, input_positions: list[int]):
        self.compiled = compiled
        self.input_positions = input_positions

    def __call__(self, *flat_inputs):
        outs = list(self.compiled(*flat_inputs))
        n = len(outs) - len(self.input_positions)
        with torch.no_grad():
            for position, value in zip(self.input_positions, outs[n:], strict=True):
                flat_inputs[position].copy_(value)
        return outs[:n]
