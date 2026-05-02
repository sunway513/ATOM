# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""State-write Triton kernels for V4 attention backend.

Replaces the per-seq Python state writes in `deepseek_v4.py` (PR-A Phase 1).
Inputs are flat batched tensors; per-token slot/position lookups happen
inside the kernel — no `.item()` syncs.

Currently implemented:
- `swa_write`: writes `swa_kv[slot_per_token[t], position[t] % win, :] = kv[t, :]`
  for each token in the flat batch. Equivalent semantics to the legacy
  prefill (lines 1429-1437) and decode (line 1446) writes.
- `update_compressor_states`: unified in-place update of Compressor's
  per-request `kv_state` + `score_state` ring buffers, covering both prefill
  (B-side overlap context + tail) and decode (every token at `pos % STATE_SIZE`
  in a single ring). Layout follows paper §3.6.1 (per-request fixed-size state
  cache) but indexes the buffer as ONE ring of size `STATE_SIZE = 2*ratio`
  (CSA overlap) or `ratio` (HCA). Token at absolute `pos` always lands at
  `kv_state[slot, pos % STATE_SIZE]` — no segment switching, no roll. The
  Compressor's softmax-pool consumer reads two halves whose A-side / B-side
  identity alternates by block-id parity; see `Compressor.forward` for that
  consumer-side logic.

Caller contract (`swa_write`):
- `kv`            [num_tokens, head_dim] flat — same layout as the model's
                  per-seq slice `seq_kv.squeeze(0)` concatenated across seqs.
- `positions`    [num_tokens] int — absolute token positions, model already has these.
- `slot_per_token` [num_tokens] int — built once per step in the metadata builder
                  by repeating `slot_per_seq` along `cu_seqlens_q` deltas.
- `swa_kv`       [num_slots, win, head_dim] in-place buffer.
- `win`          int sliding-window size (e.g. 128).

For long-prefill (`seqlen > win`), the caller must pre-filter `kv` and the
position/slot tensors so that only the last `win` tokens per seq are passed
in (otherwise multiple tokens in the same seq map to the same `pos % win`
and the write race is non-deterministic). The metadata builder produces a
`swa_keep_mask` for this purpose. Decode steps (seqlen=1) and short prefill
(seqlen<=win) bypass the mask.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_write_kernel(
    kv_ptr,  # [num_tokens, head_dim]
    positions_ptr,  # [num_tokens] int
    slot_ptr,  # [num_tokens] int
    swa_kv_ptr,  # [num_slots, win, head_dim]
    swa_kv_slot_stride,  # = win * head_dim
    swa_kv_pos_stride,  # = head_dim
    head_dim,
    win,
    BLOCK_D: tl.constexpr,
):
    tok_id = tl.program_id(0)
    pos = tl.load(positions_ptr + tok_id)
    slot = tl.load(slot_ptr + tok_id)
    ring_idx = pos % win

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    src = tl.load(
        kv_ptr + tok_id * head_dim + d_offsets,
        mask=d_mask,
    )
    dst = (
        swa_kv_ptr
        + slot * swa_kv_slot_stride
        + ring_idx * swa_kv_pos_stride
        + d_offsets
    )
    tl.store(dst, src, mask=d_mask)


def swa_write(
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_per_token: torch.Tensor,
    swa_kv: torch.Tensor,
    win: int,
) -> None:
    """In-place write `swa_kv[slot_per_token[t], pos[t] % win, :] = kv[t, :]`.

    Args:
        kv: [num_tokens, head_dim] flat batched KV (BF16).
        positions: [num_tokens] int absolute positions.
        slot_per_token: [num_tokens] int per-request slot ids.
        swa_kv: [num_slots, win, head_dim] in-place ring buffer.
        win: sliding-window size.
    """
    assert kv.dim() == 2, f"kv must be [N, D], got {kv.shape}"
    assert positions.dim() == 1
    assert slot_per_token.dim() == 1
    assert swa_kv.dim() == 3, f"swa_kv must be [S, W, D], got {swa_kv.shape}"
    num_tokens, head_dim = kv.shape
    assert positions.shape[0] == num_tokens
    assert slot_per_token.shape[0] == num_tokens
    assert swa_kv.shape[1] == win
    assert swa_kv.shape[2] == head_dim
    assert kv.is_contiguous() and swa_kv.is_contiguous()

    if num_tokens == 0:
        return

    # head_dim is small (e.g. 64-128 for V4 SWA layer), so a single Triton
    # block per token covers it. Round up to the next power of two for tl.
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (num_tokens,)
    _swa_write_kernel[grid](
        kv,
        positions,
        slot_per_token,
        swa_kv,
        swa_kv.stride(0),
        swa_kv.stride(1),
        head_dim,
        win,
        BLOCK_D=BLOCK_D,
    )


def swa_write_reference(
    kv: torch.Tensor,
    positions: torch.Tensor,
    slot_per_token: torch.Tensor,
    swa_kv: torch.Tensor,
    win: int,
) -> None:
    """Pure-PyTorch reference equivalent of `swa_write`. For tests / dump-bisect."""
    ring_idx = positions % win
    swa_kv[slot_per_token, ring_idx] = kv


# === Unified Compressor state save (plan path) ==========================
# Paper §3.6.1: per-request fixed-size state cache for "uncompressed tail
# tokens + previous block as overlap context (B-side, eq 11)". ATOM keeps
# this as a single ring of size `STATE_SIZE = 2*ratio` (CSA overlap) or
# `ratio` (HCA). Each token at absolute `pos` writes to slot
# `pos % STATE_SIZE`; the consumer (`fused_compress.*` kernel) reads its K
# source rows per-source-position, dispatching INPUT vs state cache by the
# `k_static >= window_len` plan field (where `window_len` is the count of
# leading K-loop iterations that go to state cache, encoded per-boundary in
# `compress_plan`).
#
# Write window selection (HOST side, in compress_plan.make_compress_plans):
#   write_plan rows = tokens whose absolute `pos >= max(0, seq_len - STATE_SIZE)`.
#   This preserves the last STATE_SIZE absolute positions of this forward
#   regardless of how it was scheduled (fresh prefill, chunked prefill,
#   single decode, MTP-N). The kernel below writes those rows
#   unconditionally — no in-kernel mask.


@triton.jit
def _update_compressor_states_kernel(
    kv_ptr,  # [N, dim]
    score_ptr,  # [N, dim]
    ape_ptr,  # [RATIO, dim]
    write_plan_ptr,  # [num_write, 4] int32 (ragged_id, batch_id, position, _)
    state_slot_mapping_ptr,  # [bs] int32 — per-seq state cache slot
    kv_state_ptr,
    kv_state_slot_stride,
    kv_state_pos_stride,
    score_state_ptr,
    score_state_slot_stride,
    score_state_pos_stride,
    dim,
    STATE_SIZE: tl.constexpr,  # = 2*RATIO if OVERLAP else RATIO
    OVERLAP: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """SGLang plan-style write: one program per row in `write_plan_ptr`.

    Each plan row = (ragged_id, batch_id, position, _). The plan was
    pre-filtered on the host to include only tokens whose `position` falls in
    the per-seq "last STATE_SIZE absolute positions" window — so the kernel
    writes unconditionally (no in-kernel mask), keeping it minimal.

    Destination (uniform):
      dst = position % STATE_SIZE
      slot = state_slot_mapping[batch_id]

    Score write fuses ape lookup: `score + ape[position % RATIO]`.

    Capture-stable mode (P7-r20a): when the host launches with grid =
    (write_capacity,) instead of (num_write,) (so the captured grid is a
    constant, replay-safe value), excess plan rows are filled with
    `position = -1` sentinels by the host plan builder. This kernel checks
    `position < 0` immediately after the position load and returns; no state
    cache writes happen for sentinel rows. Legacy callers that launch with
    grid = (num_write,) and tight plans see no behavioural change.
    """
    pid = tl.program_id(0)
    plan_base = write_plan_ptr + pid * 4
    # P7-r20a: load position FIRST so the sentinel early-return fires before
    # we touch ragged_id/batch_id (those are zero on sentinel rows but the
    # state_slot_mapping[batch_id] load still needs valid memory).
    position = tl.load(plan_base + 2)
    if position < 0:
        # Sentinel: invalid row in a worst-case-grid plan; no-op cleanly.
        return
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)

    slot = tl.load(state_slot_mapping_ptr + batch_id)
    dst = position % STATE_SIZE
    ring_idx_ape = position % RATIO

    d = tl.arange(0, BLOCK_D)
    m = d < dim

    kv_v = tl.load(kv_ptr + ragged_id * dim + d, mask=m).to(tl.float32)
    sc_v = tl.load(score_ptr + ragged_id * dim + d, mask=m).to(tl.float32)
    ape_v = tl.load(ape_ptr + ring_idx_ape * dim + d, mask=m).to(tl.float32)

    tl.store(
        kv_state_ptr + slot * kv_state_slot_stride + dst * kv_state_pos_stride + d,
        kv_v,
        mask=m,
    )
    tl.store(
        score_state_ptr
        + slot * score_state_slot_stride
        + dst * score_state_pos_stride
        + d,
        sc_v + ape_v,
        mask=m,
    )


def update_compressor_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    write_plan: torch.Tensor,  # [num_write or write_capacity, 4] int32
    num_write: int,
    state_slot_mapping: torch.Tensor,  # [bs] int32 — per-seq state slot
    ratio: int,
    overlap: bool,
    write_capacity: "int | None" = None,
) -> None:
    """In-place update of Compressor's per-request `kv_state`/`score_state`
    ring buffer (size `2*ratio` for overlap CSA, `ratio` for HCA), driven by
    a SGLang-style packed `write_plan`.

    The plan is pre-filtered on the host to include only tokens whose
    `position` falls in the per-seq "last STATE_SIZE absolute positions"
    window — the kernel writes unconditionally, no in-kernel mask.

    Args:
      kv:           [N, dim] flat batched KV (typically fp32 or bf16, cast inside).
      score:        [N, dim] flat batched score (NOT pre-added with ape;
                    kernel fuses ape addition).
      ape:          [ratio, dim] absolute position embedding.
      kv_state:     [num_slots, S, dim] in-place ring buffer.
                    S = 2*ratio if overlap else ratio.
      score_state:  same shape as kv_state.
      write_plan:   [num_write, 4] int32 (legacy) or [write_capacity, 4]
                    int32 (P7-r20a capture-stable). Excess rows past
                    num_write MUST be filled with `position=-1` sentinels.
      num_write:    Live row count. In legacy mode this is also the grid
                    size; in capture-stable mode this is informational only
                    (used for shape assertion when capacity is None) and the
                    grid comes from `write_capacity`.
      state_slot_mapping: [bs] int32 — per-seq state cache slot.
      ratio, overlap: compress geometry.
      write_capacity: P7-r20a — when set, kernel grid = (write_capacity,)
                    INSTEAD of (num_write,). The captured grid is a constant
                    (== capacity) so replay-time `num_write` variation
                    doesn't affect the captured launch parameters. Excess
                    program slots no-op via `position=-1` sentinel
                    early-return inside the kernel.
    """
    assert kv.dim() == 2 and score.dim() == 2
    assert kv.shape == score.shape, f"{kv.shape} vs {score.shape}"
    assert ape.dim() == 2 and ape.shape[0] == ratio
    state_size = (2 if overlap else 1) * ratio
    assert (
        kv_state.shape[1] == state_size
    ), f"kv_state.shape[1]={kv_state.shape[1]}, expected {state_size}"
    # Capture-stable: grid is constant capacity; legacy: grid is live count
    # and we short-circuit when there's nothing to do.
    if write_capacity is not None:
        grid_size = int(write_capacity)
        if grid_size == 0:
            return
    else:
        if num_write == 0:
            return
        grid_size = int(num_write)
    dim = kv.shape[1]
    assert write_plan.dim() == 2 and write_plan.shape[1] == 4
    assert write_plan.dtype == torch.int32
    assert (
        write_plan.shape[0] == grid_size
    ), f"write_plan rows {write_plan.shape[0]} != grid_size {grid_size}"
    assert state_slot_mapping.dim() == 1 and state_slot_mapping.dtype == torch.int32

    BLOCK_D = triton.next_power_of_2(dim)
    _update_compressor_states_kernel[(grid_size,)](
        kv if kv.is_contiguous() else kv.contiguous(),
        score if score.is_contiguous() else score.contiguous(),
        ape,
        write_plan,
        state_slot_mapping,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        dim,
        STATE_SIZE=state_size,
        OVERLAP=int(overlap),
        RATIO=ratio,
        BLOCK_D=BLOCK_D,
    )


def update_compressor_states_reference(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    *,
    write_plan: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    ratio: int,
    overlap: bool,
) -> None:
    """Pure-PyTorch reference equivalent of `update_compressor_states` (plan path).

    `write_plan[i] = (ragged_id, batch_id, position, _)` — each row is one
    token to write.  No mask (host filtered).
    """
    state_size = (2 if overlap else 1) * ratio
    plan_cpu = write_plan.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    for i in range(plan_cpu.shape[0]):
        ragged_id, batch_id, position, _ = plan_cpu[i].tolist()
        slot = int(slot_map_cpu[batch_id].item())
        dst = position % state_size
        kv_state[slot, dst] = kv[ragged_id]
        score_state[slot, dst] = score[ragged_id] + ape[position % ratio]
