# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 attention backend Triton kernels.

These kernels replace the per-seq Python state-write logic in
`atom/models/deepseek_v4.py` (PR-A: kill .item() / unlock CUDAGraph). All
take batched tensors (positions, slot_per_token, cu_seqlens_q) — nothing is
derived from device data via `.item()`.

P7-r6 (#37 issue #60): also exports pure-PyTorch capture-friendly variants
``update_compressor_states_pytorch`` / ``swa_write_pytorch``. Used to
validate / fix the cudagraph capture variance hypothesis (Triton state-
mutation kernels racing with downstream PyTorch reads).
"""

from atom.model_ops.v4_kernels.state_writes import (
    swa_write,
    swa_write_pytorch,
    update_compressor_states,
    update_compressor_states_pytorch,
)

__all__ = [
    "update_compressor_states",
    "update_compressor_states_pytorch",
    "swa_write",
    "swa_write_pytorch",
]
