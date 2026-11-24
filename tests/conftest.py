# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Import torch before any test module does. Some upstream tests import
# ``spmd_types`` first; ``spmd_types`` imports torch, torch auto-loads the
# ``torch_tpu`` backend, and that pulls in ``torch.distributed.fsdp``, which
# registers with the still partially initialized ``spmd_types`` and fails with
# "partially initialized module 'spmd_types' has no attribute
# 'register_local_autograd_function'".
import torch  # noqa: F401
