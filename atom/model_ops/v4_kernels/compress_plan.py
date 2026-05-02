# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang-style packed compression plan for V4 batched compressor kernels.

Each plan slot is a 16-byte struct of 4 int32 fields:
    [ragged_id, batch_id, position, window_len]

  - ragged_id:  token's row index in the ragged input stream (kv_in / score_in)
  - batch_id:   sequence index (→ state_slot_mapping[batch_id], block_table[batch_id])
  - position:   absolute token position (→ RoPE)
  - window_len: number of leading K-loop iterations that read from state cache
                instead of the ragged input. K = STATE_SIZE = (1+overlap)*ratio.

Two plan tensors are produced per `compress_ratio`:
  - compress_plan: rows for tokens whose `(position+1) % ratio == 0`
                   (= compression boundaries). One row per fused-compress kernel
                   program. Grid = `num_compress`.
  - write_plan:    rows for tokens whose `position` falls in the per-seq
                   "last STATE_SIZE positions" window. One row per
                   `update_compressor_states` kernel program. Grid = `num_write`.

Caller (per-seq loop) gets `cu_compress_cpu` for slicing the kernel's flat
output `[num_compress, head_dim]` back to per-seq chunks.

P7-r20a — capture-stable worst-case grids
-----------------------------------------
Under HIP cudagraph capture, the kernel launch grid `(num_compress,)` /
`(num_write,)` is BAKED at warmup-time into the captured graph. Warmup
sees `extend_lens=1, start_pos=1` → `num_compress=0` for ratio=4/128
→ grid=(0,) baked → at replay every captured kernel launches with grid 0
regardless of the live token count. The Indexer's compressor never writes
at replay → silent-incorrect topk → gibberish completion.

Fix shape (this module's contribution): the plan tensor itself is sized to
`compress_capacity` / `write_capacity` (worst-case per (ratio, max_seqlen)),
caller passes that capacity as the kernel grid, and excess plan rows are
filled with `position=-1` sentinel rows. Both kernels early-return on
`position < 0` so the over-launched programs no-op cleanly.

`compress_capacity` / `write_capacity` are OPTIONAL fields (default None);
when None, the kernel falls back to `num_compress` / `num_write` (legacy
non-capture behaviour, preserved for unit tests + eager paths that don't
need capture stability).
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch

# P7-r20a: per-row "invalid" sentinel for capture-stable worst-case grids.
# Excess plan rows past the live `num_compress`/`num_write` get this sentinel
# so the kernel can `if position < 0: return` early. Other fields are 0 to
# avoid undefined-behaviour pre-loads before the early-return.
_INVALID_POSITION_SENTINEL: int = -1


@dataclass
class CompressPlan:
    """Packed metadata for one (compress_ratio, overlap) variant of one fwd."""

    compress_plan_gpu: (
        torch.Tensor
    )  # [num_compress, 4] int32 (or [capacity, 4] when capture-stable)
    write_plan_gpu: (
        torch.Tensor
    )  # [num_write, 4]    int32 (or [capacity, 4] when capture-stable)
    num_compress: int  # CPU scalar — fused_compress kernel grid (legacy path)
    num_write: int  # CPU scalar — update_compressor_states kernel grid (legacy path)
    cu_compress_cpu: (
        np.ndarray
    )  # [bs+1] int32 — per-seq slice into out[num_compress, D]
    # P7-r20a capture-stable additions. When set, kernel callers use these
    # as the launch grid (instead of num_compress / num_write), giving a
    # constant grid that bakes safely under cudagraph capture. Tail rows of
    # the plan tensor MUST be filled with `position=-1` sentinels so the
    # kernel early-returns cleanly. Both default to None for backward compat
    # with eager / unit-test paths that build plans on-the-fly.
    compress_capacity: Optional[int] = None
    write_capacity: Optional[int] = None


def make_compress_plans(
    extend_lens_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    unique_ratios_overlap: Iterable[Tuple[int, bool]],
    device: torch.device,
) -> dict[int, CompressPlan]:
    """Build a CompressPlan per (ratio, overlap) variant.

    Args:
      extend_lens_cpu:  np[bs] int — number of tokens this fwd processes per seq.
      seq_lens_cpu:     np[bs] int — absolute seq_len per seq (= prefix + extend).
      unique_ratios_overlap: iterable of (ratio, is_overlap) pairs. Typically
                             {(4, True), (128, False)} for V4-Pro; a subset for
                             models with only CSA or only HCA layers.
      device: target GPU device for the GPU-side plan tensors.

    Returns:
      dict[ratio] -> CompressPlan. Empty dict if `extend_lens_cpu.sum() == 0`.

    NOTE: This builder produces TIGHT plan tensors (`shape = (num_compress, 4)`),
    NOT capture-stable worst-case plans. It's used by the eager + non-capture
    paths. Capture-stable construction lives in
    `atom/model_ops/attentions/deepseek_v4_attn.py:_build_compress_plans_into_buffers`
    and `Indexer._build_bs1_compress_plan` (P7-r20a).
    """
    bs = len(extend_lens_cpu)
    extend_lens_cpu = np.ascontiguousarray(extend_lens_cpu, dtype=np.int32)
    seq_lens_cpu = np.ascontiguousarray(seq_lens_cpu, dtype=np.int32)
    total = int(extend_lens_cpu.sum())
    out: dict[int, CompressPlan] = {}
    if total == 0 or bs == 0:
        return out

    # Per-token columns shared across ratios.
    batch_ids = np.repeat(np.arange(bs, dtype=np.int32), extend_lens_cpu)
    ragged_ids = np.arange(total, dtype=np.int32)
    cu_extend = np.empty(bs + 1, dtype=np.int32)
    cu_extend[0] = 0
    np.cumsum(extend_lens_cpu, out=cu_extend[1:])
    j_in_seq = ragged_ids - cu_extend[batch_ids]
    prefix_lens = seq_lens_cpu - extend_lens_cpu
    positions = prefix_lens[batch_ids] + j_in_seq

    for ratio, is_overlap in unique_ratios_overlap:
        K = ratio * (2 if is_overlap else 1)
        # window_len = K - min(j_in_seq + 1, K)
        # Number of leading K-loop iterations that go to state cache.
        window_lens = np.maximum(0, K - np.minimum(j_in_seq + 1, K)).astype(np.int32)
        plan_rows = np.stack(
            [ragged_ids, batch_ids, positions, window_lens], axis=1
        ).astype(np.int32)

        # compress: token at a compression boundary
        compress_mask = (positions + 1) % ratio == 0
        compress_plan = plan_rows[compress_mask]
        # cu_compress: per-seq prefix-sum of boundary counts (for caller slicing).
        # bincount preserves seq order because compress_plan rows are already
        # sorted by ragged_id (and ragged_id increases monotonically with batch_id).
        compress_counts = np.bincount(compress_plan[:, 1], minlength=bs).astype(
            np.int32
        )
        cu_compress = np.empty(bs + 1, dtype=np.int32)
        cu_compress[0] = 0
        np.cumsum(compress_counts, out=cu_compress[1:])

        # write: tokens whose absolute position falls in the per-seq
        # "last STATE_SIZE positions" window. STATE_SIZE = K.
        # write_start[i] = max(0, seq_lens[i] - K) — uniform across overlap/non-overlap;
        # the SGLang formula `(seq_len // ratio) * ratio - (ratio if overlap else 0)`
        # is a stricter bound that includes only ratio-aligned writes; the looser
        # `seq_len - K` is what ATOM's update_compressor_states already uses
        # (state_writes.py:152-154 docstring) and what the fused kernel's
        # state-cache reader expects.
        write_starts = np.maximum(0, seq_lens_cpu - K).astype(np.int32)
        write_mask = positions >= write_starts[batch_ids]
        write_plan = plan_rows[write_mask]

        out[ratio] = CompressPlan(
            compress_plan_gpu=torch.from_numpy(np.ascontiguousarray(compress_plan)).to(
                device, non_blocking=True
            ),
            write_plan_gpu=torch.from_numpy(np.ascontiguousarray(write_plan)).to(
                device, non_blocking=True
            ),
            num_compress=int(compress_plan.shape[0]),
            num_write=int(write_plan.shape[0]),
            cu_compress_cpu=cu_compress,
        )
    return out


def make_single_seq_plan(
    *,
    token_num: int,
    start_pos: int,
    ratio: int,
    is_overlap: bool,
    device: torch.device,
) -> CompressPlan:
    """Build a 1-seq CompressPlan. Used by Indexer.forward (which still loops
    per-seq from the outer dispatch)."""
    extend_lens = np.array([token_num], dtype=np.int32)
    seq_lens = np.array([start_pos + token_num], dtype=np.int32)
    plans = make_compress_plans(extend_lens, seq_lens, [(ratio, is_overlap)], device)
    if not plans:
        # token_num == 0 — return an empty plan with consistent shapes.
        empty = torch.empty((0, 4), dtype=torch.int32, device=device)
        return CompressPlan(
            compress_plan_gpu=empty,
            write_plan_gpu=empty,
            num_compress=0,
            num_write=0,
            cu_compress_cpu=np.array([0, 0], dtype=np.int32),
        )
    return plans[ratio]


# === P7-r20a: capture-stable plan helpers ================================
#
# These helpers build worst-case-shape plans into CALLER-PROVIDED stable
# pinned/gpu buffers, so the kernel launch sees a constant grid + a stable
# input pointer across capture+replay. Excess rows past the live count are
# filled with `position = _INVALID_POSITION_SENTINEL` so the kernel
# early-returns cleanly (see fused_compress.py / state_writes.py top-of-
# kernel checks).
#
# Caller pattern (Indexer):
#   * compress_plan_pin / write_plan_pin: pinned-CPU [capacity, 4] int32 staging
#   * compress_plan_buf / write_plan_buf: GPU [capacity, 4] int32 destination
#   * call `fill_invalid_plan_rows_inplace_np(compress_plan_pin_np, ...)`
#     once at __init__ to set the sentinel for ALL rows.
#   * each forward writes valid rows into the front of the pinned buffer,
#     then `compress_plan_buf.copy_(compress_plan_pin, non_blocking=True)`
#     to publish to GPU.
#
# Shared invariants asserted by the kernel:
#   * Plan dtype is int32 — required for the `tl.load(plan_base + i)` 4xi32
#     loads in fused_compress.py.
#   * Position sentinel is signed (= -1), so int32 plans naturally encode it.


def fill_invalid_plan_rows_inplace_np(
    plan_np: np.ndarray, start: int, end: int
) -> None:
    """Fill `plan_np[start:end, :]` with the invalid-row sentinel.

    Each tail row becomes (ragged_id=0, batch_id=0, position=-1, window_len=0).
    The kernel only inspects `position` for the early-return; the other
    columns are zeroed for safety against any speculative pre-load.

    Args:
      plan_np: [capacity, 4] int32 — pinned-CPU view of the plan staging buffer.
      start:   first invalid row index (= live num_compress / num_write).
      end:     one-past-last invalid row index (= capacity).
    """
    if end <= start:
        return
    plan_np[start:end, 0] = 0
    plan_np[start:end, 1] = 0
    plan_np[start:end, 2] = _INVALID_POSITION_SENTINEL
    plan_np[start:end, 3] = 0


def fill_all_invalid_plan_rows_inplace_np(plan_np: np.ndarray) -> None:
    """Convenience: fill the ENTIRE plan_np with invalid-row sentinels.

    Used at __init__ time so a pre-allocated capture-stable plan buffer
    starts in a "no-op everywhere" state. Subsequent forwards overwrite
    the front rows with live data and leave the tail at sentinel.
    """
    fill_invalid_plan_rows_inplace_np(plan_np, 0, plan_np.shape[0])


# === P7-r20b: capture-stable batched plan builder ========================


def make_compress_plans_into_buffers(
    extend_lens_cpu: np.ndarray,
    seq_lens_cpu: np.ndarray,
    unique_ratios_overlap: Iterable[Tuple[int, bool]],
    *,
    compress_pin_per_ratio: "dict[int, np.ndarray]",
    write_pin_per_ratio: "dict[int, np.ndarray]",
    compress_buf_per_ratio: "dict[int, torch.Tensor]",
    write_buf_per_ratio: "dict[int, torch.Tensor]",
    compress_capacity_per_ratio: "dict[int, int]",
    write_capacity_per_ratio: "dict[int, int]",
) -> dict[int, CompressPlan]:
    """Capture-stable batched build: writes plan rows into CALLER-PROVIDED
    pinned-CPU + GPU buffers, then publishes to GPU via `.copy_()` (a
    capture-safe memcpy node). Returns CompressPlans that wrap the FULL
    capacity slice so the kernel wrapper launches with the worst-case grid
    (`grid = (capacity,)`) — invariant across forwards / replays.

    Rationale: legacy `make_compress_plans` does
    `torch.from_numpy(plan_np).to(device, non_blocking=True)` which allocates
    a FRESH GPU tensor every forward. Under HIP cudagraph capture the kernel
    launch records the warmup-time tensor pointer, and replay launches with
    that stale pointer (or, if the warmup plan had num_compress=0, with grid
    0 even when the live plan has many rows). This builder side-steps both
    issues: stable buffer addresses + worst-case grid.

    Args:
      extend_lens_cpu, seq_lens_cpu: same as `make_compress_plans` (np int32).
      unique_ratios_overlap: same.
      compress_pin_per_ratio:    {ratio: np.ndarray} — pinned-CPU staging,
                                 each shape [compress_capacity, 4] int32.
                                 Tail rows are pre-filled with sentinel by
                                 the caller (typically once at __init__).
      write_pin_per_ratio:       {ratio: np.ndarray} — pinned-CPU staging
                                 for write plans.
      compress_buf_per_ratio:    {ratio: torch.Tensor} — GPU dest, same
                                 shape as the corresponding pin buffer.
      write_buf_per_ratio:       {ratio: torch.Tensor} — GPU dest for writes.
      compress_capacity_per_ratio:  {ratio: int} — worst-case row counts.
      write_capacity_per_ratio:     {ratio: int} — worst-case row counts.

    Returns:
      dict[ratio] -> CompressPlan with `compress_capacity` / `write_capacity`
      set; `compress_plan_gpu` / `write_plan_gpu` are FULL slices of the
      caller's GPU buffers (same data_ptr across calls).

    Empty dict when `extend_lens_cpu.sum() == 0` — caller should fall back
    to the warmup-sentinel state of the buffers (or skip the kernel call).
    """
    bs = len(extend_lens_cpu)
    extend_lens_cpu = np.ascontiguousarray(extend_lens_cpu, dtype=np.int32)
    seq_lens_cpu = np.ascontiguousarray(seq_lens_cpu, dtype=np.int32)
    total = int(extend_lens_cpu.sum())
    out: dict[int, CompressPlan] = {}
    if bs == 0 or total == 0:
        # Even with no live tokens, emit capture-stable plans whose buffers
        # are all-sentinel (kernel will no-op) so downstream callers get
        # the right grid + pointer. This matches the warmup invariant.
        for ratio, _is_overlap in unique_ratios_overlap:
            cap_c = compress_capacity_per_ratio[ratio]
            cap_w = write_capacity_per_ratio[ratio]
            # Re-stamp every row with sentinel (defensive against earlier
            # forwards leaving live data in front rows).
            fill_invalid_plan_rows_inplace_np(compress_pin_per_ratio[ratio], 0, cap_c)
            fill_invalid_plan_rows_inplace_np(write_pin_per_ratio[ratio], 0, cap_w)
            compress_buf_per_ratio[ratio].copy_(
                torch.from_numpy(compress_pin_per_ratio[ratio]),
                non_blocking=True,
            )
            write_buf_per_ratio[ratio].copy_(
                torch.from_numpy(write_pin_per_ratio[ratio]),
                non_blocking=True,
            )
            out[ratio] = CompressPlan(
                compress_plan_gpu=compress_buf_per_ratio[ratio],
                write_plan_gpu=write_buf_per_ratio[ratio],
                num_compress=0,
                num_write=0,
                cu_compress_cpu=np.zeros(max(bs + 1, 2), dtype=np.int32),
                compress_capacity=cap_c,
                write_capacity=cap_w,
            )
        return out

    # Per-token columns shared across ratios (mirrors the legacy builder).
    batch_ids = np.repeat(np.arange(bs, dtype=np.int32), extend_lens_cpu)
    ragged_ids = np.arange(total, dtype=np.int32)
    cu_extend = np.empty(bs + 1, dtype=np.int32)
    cu_extend[0] = 0
    np.cumsum(extend_lens_cpu, out=cu_extend[1:])
    j_in_seq = ragged_ids - cu_extend[batch_ids]
    prefix_lens = seq_lens_cpu - extend_lens_cpu
    positions = prefix_lens[batch_ids] + j_in_seq

    for ratio, is_overlap in unique_ratios_overlap:
        K = ratio * (2 if is_overlap else 1)
        compress_pin = compress_pin_per_ratio[ratio]
        write_pin = write_pin_per_ratio[ratio]
        compress_buf = compress_buf_per_ratio[ratio]
        write_buf = write_buf_per_ratio[ratio]
        cap_c = compress_capacity_per_ratio[ratio]
        cap_w = write_capacity_per_ratio[ratio]

        window_lens = np.maximum(0, K - np.minimum(j_in_seq + 1, K)).astype(np.int32)

        # compress: token at a compression boundary
        compress_mask = (positions + 1) % ratio == 0
        compress_idx = np.where(compress_mask)[0]
        n_compress = compress_idx.size
        # Capacity overflow guards (defensive — caller should size capacity
        # to true worst-case but throw a clear error if it underflows).
        if n_compress > cap_c:
            raise RuntimeError(
                f"compress plan overflow for ratio={ratio}: "
                f"n_compress={n_compress} > capacity={cap_c}"
            )
        if n_compress > 0:
            compress_pin[:n_compress, 0] = ragged_ids[compress_idx]
            compress_pin[:n_compress, 1] = batch_ids[compress_idx]
            compress_pin[:n_compress, 2] = positions[compress_idx]
            compress_pin[:n_compress, 3] = window_lens[compress_idx]
        # P7-r20b: re-stamp the tail (rows [n_compress:capacity]) with
        # sentinel — guards against shrink: a previous forward's larger
        # n_compress could have left live data in rows we now consider
        # invalid (replay would launch them as real boundaries -> double
        # write into kv_cache). Same invariant as the bs=1 Indexer path.
        fill_invalid_plan_rows_inplace_np(compress_pin, n_compress, cap_c)

        # cu_compress: per-seq prefix-sum of boundary counts (unchanged).
        compress_counts = np.bincount(batch_ids[compress_idx], minlength=bs).astype(
            np.int32
        )
        cu_compress = np.empty(bs + 1, dtype=np.int32)
        cu_compress[0] = 0
        np.cumsum(compress_counts, out=cu_compress[1:])

        # write_plan: rows where positions >= max(0, seq_len - K)
        write_starts = np.maximum(0, seq_lens_cpu - K).astype(np.int32)
        write_mask = positions >= write_starts[batch_ids]
        write_idx = np.where(write_mask)[0]
        n_write = write_idx.size
        if n_write > cap_w:
            raise RuntimeError(
                f"write plan overflow for ratio={ratio}: "
                f"n_write={n_write} > capacity={cap_w}"
            )
        if n_write > 0:
            write_pin[:n_write, 0] = ragged_ids[write_idx]
            write_pin[:n_write, 1] = batch_ids[write_idx]
            write_pin[:n_write, 2] = positions[write_idx]
            write_pin[:n_write, 3] = window_lens[write_idx]
        fill_invalid_plan_rows_inplace_np(write_pin, n_write, cap_w)

        # H2D copy_ of FULL capacity (front-live + tail-sentinel) into the
        # stable GPU buffers (capture-safe memcpy node; pinned source).
        # `torch.from_numpy(pin)` on a pinned-memory numpy view returns a
        # tensor that shares the pinned storage — the H2D copy stays
        # capture-stream-legal per SOP-6.
        compress_buf.copy_(torch.from_numpy(compress_pin), non_blocking=True)
        write_buf.copy_(torch.from_numpy(write_pin), non_blocking=True)

        out[ratio] = CompressPlan(
            compress_plan_gpu=compress_buf,
            write_plan_gpu=write_buf,
            num_compress=int(n_compress),
            num_write=int(n_write),
            cu_compress_cpu=cu_compress,
            compress_capacity=cap_c,
            write_capacity=cap_w,
        )
    return out
