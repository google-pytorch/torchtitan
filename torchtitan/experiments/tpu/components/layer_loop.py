"""Lets a TPU recipe replace the decoder layer loop, without touching ``Decoder``.

``Decoder.forward`` loops over the layers inline, so running them any other way
-- as one ``scan``, or with a gather or an offload around each layer -- means
owning that loop. :class:`TpuLayerLoop` is ``model.layers`` as a container that
runs its own layers, and :func:`tpu_model_class` builds the model subclass whose
``forward`` calls it. Recipes that skip the rewrite are untouched.
Design: the unmerged ``tpu-layer-loop-converter`` branch (7cde56bca0).
"""

import functools
from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import (
    AttentionMasksType,
    segment_ids_from_positions,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.protocols.module import ModuleDict


__all__ = [
    "TpuLayerLoop",
    "builds_layer_loop",
    "model_subclass",
    "runs_layer_loop",
    "tpu_model_class",
]


class TpuLayerLoop(ModuleDict):
    """``model.layers``, as a container that runs its own layers.

    Subclass to run them differently. Still a ``ModuleDict``, and the layer
    modules inside are the same objects, so FQNs and the optimizer are
    unaffected.
    """

    def forward(self, h: torch.Tensor, *args) -> torch.Tensor:
        for layer in self.values():
            h = layer(h, *args)
        return h


def _forward(
    self,
    tokens: torch.Tensor,
    positions: torch.Tensor | None = None,
    attention_masks: AttentionMasksType | None = None,
):
    """Runs the decoder: embeddings, the layers, the final norm, the LM head.

    A copy of ``Decoder.forward`` (``models/common/decoder.py``) that hands the
    layers to ``model.layers`` when that container runs them itself, and
    otherwise loops as upstream does. Only that hunk differs, so upstream's
    inline comments stay there, and ``tests/test_layer_loop.py`` diffs the two
    ASTs to catch an upstream edit.
    """
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

    segment_ids = (
        segment_ids_from_positions(positions) if positions is not None else None
    )

    # The only change from `Decoder.forward`: let the container run the layers.
    # The `else` is upstream's loop, kept for the plain, empty `ModuleDict` a
    # layer-less pipeline stage gets -- that one has no `forward`.
    #
    # A TpuLayerLoop itself cannot be split across pipeline stages.
    # TODO: support pipeline parallelism.
    if isinstance(self.layers, TpuLayerLoop):
        h = self.layers(h, attention_masks, positions, segment_ids)
    else:
        for layer in self.layers.values():
            h = layer(h, attention_masks, positions, segment_ids)

    h = self.norm(h) if self.norm is not None else h

    if self._skip_lm_head:
        return h
    output = self.lm_head(h) if self.lm_head is not None else h
    return output


def runs_layer_loop(model: torch.nn.Module) -> bool:
    """Whether ``model`` hands its layers to ``model.layers`` to run."""
    return type(model).forward is _forward


def builds_layer_loop(model_config: Decoder.Config) -> bool:
    """Whether ``model_config.build()`` would satisfy :func:`runs_layer_loop`.

    Lets a caller check the rewrite landed without building the model.
    """
    return model_config._owner.forward is _forward


def model_subclass(
    model_cls: type,
    config_cls: type,
    name: str,
    *,
    config_body: dict | None = None,
    **body,
) -> type:
    """Subclasses ``model_cls``, giving the subclass a ``Config`` of its own.

    ``build()`` instantiates ``Config._owner``, so a rewrite needs both halves.
    ``config_cls`` is passed in rather than read off ``model_cls`` because the
    recipe may have substituted one (qwen3 does, for FLOP accounting).
    ``config_body`` adds attributes, such as method overrides, to the new
    ``Config``; ``body`` adds them to the model subclass.
    """
    subclass_config = dataclass(kw_only=True, slots=True)(
        type(
            "Config",
            (config_cls,),
            {"__qualname__": f"{name}.Config", **(config_body or {})},
        )
    )
    return type(name, (model_cls,), {**body, "Config": subclass_config})


@functools.cache
def tpu_model_class(config_cls: type) -> type:
    """Builds the ``Tpu<Model>`` subclass whose ``forward`` is ``_forward``.

    ``config_cls`` is a model's nested ``Config``, and ``Config._owner`` the
    model class that declares it.

    Put the recipe on the new ``Config`` with ``derive``. Do it last: any later
    rewrite that rebuilds the root config from a fixed class undoes this one,
    which is also why it cannot go on a recipe -- ``experiments/tpu/train.py``
    rebuilds ``model_spec`` from the graph-trainer registry.

    One class per config class, cached.
    """
    model_cls = config_cls._owner
    if model_cls.forward is _forward:
        return model_cls  # already rewritten
    if not issubclass(model_cls, Decoder) or model_cls.forward is not Decoder.forward:
        raise TypeError(
            f"{model_cls.__name__} is not a Decoder running the stock forward"
        )
    return model_subclass(
        model_cls, config_cls, f"Tpu{model_cls.__name__}", forward=_forward
    )
