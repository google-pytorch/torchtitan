"""Data utilities for JAX experiment."""

import numpy as np


def fake_dataloader(size, seqlen, batch_size):
    """Generates synthetic data (numpy arrays, no PyTorch dependency)."""
    for _ in range(size):
        x = np.random.randint(0, 32000, (batch_size, seqlen), dtype=np.int32)
        yield x, (x + 1) % 32000


def torch_loader_to_jax(train_loader, seq_len=None):
    """Wrap a torchtitan dataloader to yield numpy arrays."""
    for batch in train_loader:
        if hasattr(batch, 'as_input_dict'):
            # Upstream ec953b36: dataloaders yield TrainingMicrobatch objects.
            inputs, labels = batch.as_input_dict()['input'], batch.labels
        elif isinstance(batch, dict):
            inputs, labels = batch['input'], batch['labels']
        else:
            inputs, labels = batch[0]['input'], batch[1]
        inputs_np = inputs.numpy().astype(np.int32)
        labels_np = labels.numpy().astype(np.int32)
        if inputs_np.ndim == 1 and seq_len is not None:
            inputs_np = inputs_np.reshape(-1, seq_len)
            labels_np = labels_np.reshape(-1, seq_len)
        yield inputs_np, labels_np
