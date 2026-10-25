"""Llama 3 on the TPU graph trainer: upstream's model registry, the TPU parallelize."""

import dataclasses

from torch import nn

from torchtitan.experiments.graph_trainer.llama3 import (
    model_registry as graph_trainer_model_registry,
)
from torchtitan.experiments.graph_trainer.llama3.parallelize import annotate_llama
from torchtitan.experiments.tpu.graph_trainer.parallelize import (
    parallelize_tpu_graph_trainer,
)
from torchtitan.protocols.model_spec import ModelSpec


def parallelize_llama3(model: nn.Module, **kwargs) -> nn.Module:
    return parallelize_tpu_graph_trainer(model, annotate=annotate_llama, **kwargs)


def model_registry(flavor: str) -> ModelSpec:
    return dataclasses.replace(
        graph_trainer_model_registry(flavor), parallelize_fn=parallelize_llama3
    )
