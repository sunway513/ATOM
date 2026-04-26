# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 ForwardBatch metadata (Path 3 / W4.1 — issue #37).

This module defines `DSV4ForwardBatch`, the per-step metadata object the
DSV4 model layers will consume in the SGLang/vLLM-isomorphic refactor
(see issue #37 for the architectural rationale and PR-split plan).

W4.1 SCOPE: structure only. We define the dataclass, validate field
shapes/dtypes/devices, and provide a `from_attn_metadata()` adapter that
constructs a DSV4ForwardBatch from ATOM's existing
`forward_context.attn_metadata` so the runtime can be plumbed through
later stages without changing model behavior in this PR.

W4.2-W4.6 will progressively wire this object into:
- engine-owned KV/state pools (W4.2)
- Compressor/Indexer state owned by pools, not nn.Module register_buffer (W4.4)
- DeepseekV4Attention.forward consumption (W4.3)
- correctness validation conc=4 (W4.5)
- perf optimization: in-graph metadata, fused kernels (W4.6)

Reference design: SGLang `python/sglang/srt/models/deepseek_v4.py:819`
(`MQALayer.forward(x, positions, forward_batch)`) and vLLM
`vllm/model_executor/layers/deepseek_v4_attention.py:650`
(`forward(self, q, kv, positions, output)`).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


class DSV4ForwardMode(enum.Enum):
    """High-level dispatch mode for one model step."""

    PREFILL = "prefill"
    """Mixed or pure prefill batch — at least one seq has > 1 token."""
    DECODE = "decode"
    """All seqs have exactly 1 token (lockstep decode)."""
    IDLE = "idle"
    """Warmup / dummy step. No real seqs to attend to."""


@dataclass
class DSV4ForwardBatch:
    """Per-step metadata for a DSV4 forward pass.

    All tensor fields are 1-D unless documented otherwise. Validation in
    `__post_init__` checks dtype/device/shape consistency so model layers
    can rely on invariants without re-checking.

    Fields
    ------
    forward_mode: which dispatch path to take (see `DSV4ForwardMode`).
    positions: `[num_tokens]` long tensor of absolute positions per token.
        For prefill, each seq's positions are sequential starting at the
        seq's previous KV length (0 for fresh requests). For decode, each
        token's position equals `seq_lens[seq_idx_of_token] - 1` (token
        is being generated AT that position).
    seq_lens: `[num_seqs]` long tensor — total length each seq has after
        this step (= context length + new tokens generated this step).
    extend_seq_lens: `[num_seqs]` long tensor — number of NEW tokens this
        seq is processing this step. For prefill = seq_len; for decode = 1.
    cu_seqlens_q: `[num_seqs + 1]` int32 tensor — cumulative `extend_seq_lens`
        prefix sum, 0-prefixed. Equivalent to ATOM's existing
        `attn_metadata.cu_seqlens_q`.
    req_pool_indices: `[num_seqs]` long tensor — opaque per-seq id stable
        across steps. **W4.1 PLACEHOLDER**: `from_attn_metadata()` populates
        this with `block_tables[:, 0]` (first physical block id). This is
        NOT stable-unique under prefix caching / block reuse: two
        prefix-sharing seqs can share a first block. Do not depend on
        uniqueness — call `assert_pool_slot_unique()` if you need that
        guarantee at the call site. W4.2 replaces the adapter with the
        engine KV-pool's true slot allocator, which guarantees a stable
        unique id per active seq regardless of prefix-cache state.
    out_cache_loc: `[num_tokens]` long tensor — absolute slot in the engine
        KV pool where each token's KV should be written. W4.2 fills this
        with `req_pool_indices[seq] * ring_size + (positions[t] %
        ring_size)`. W4.1 leaves it None (allocators not wired yet).
    block_tables: pass-through reference to ATOM's existing
        `attn_metadata.block_tables` (`[num_seqs, max_blocks_per_seq]`).
        Kept here so a single ForwardBatch carries everything the model
        layer needs without re-fetching from the global forward_context.
    """

    forward_mode: DSV4ForwardMode
    positions: torch.Tensor
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    req_pool_indices: torch.Tensor

    # Optional in W4.1 (will become required in W4.2 once pools are wired).
    out_cache_loc: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    # Reserved for W4.2: link to the engine-owned DSV4 KV/state pool.
    # Kept Optional + Any to avoid a hard import on the (not-yet-existing)
    # dsv4_pool module.
    kv_pool: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # ---- shape invariants ----
        assert (
            self.positions.dim() == 1
        ), f"positions must be 1-D [num_tokens], got shape {tuple(self.positions.shape)}"
        assert self.seq_lens.dim() == 1, "seq_lens must be 1-D [num_seqs]"
        assert self.extend_seq_lens.dim() == 1, "extend_seq_lens must be 1-D [num_seqs]"
        assert self.cu_seqlens_q.dim() == 1, "cu_seqlens_q must be 1-D [num_seqs + 1]"
        assert (
            self.req_pool_indices.dim() == 1
        ), "req_pool_indices must be 1-D [num_seqs]"

        num_seqs = self.seq_lens.numel()
        num_tokens = self.positions.numel()
        assert self.extend_seq_lens.numel() == num_seqs, (
            f"extend_seq_lens.numel()={self.extend_seq_lens.numel()} "
            f"!= num_seqs={num_seqs}"
        )
        assert self.cu_seqlens_q.numel() == num_seqs + 1, (
            f"cu_seqlens_q.numel()={self.cu_seqlens_q.numel()} "
            f"!= num_seqs+1={num_seqs + 1}"
        )
        assert self.req_pool_indices.numel() == num_seqs, (
            f"req_pool_indices.numel()={self.req_pool_indices.numel()} "
            f"!= num_seqs={num_seqs}"
        )
        # cu_seqlens_q's last entry must equal total tokens
        if num_seqs > 0:
            tail = int(self.cu_seqlens_q[-1].item())
            assert (
                tail == num_tokens
            ), f"cu_seqlens_q[-1]={tail} != num_tokens={num_tokens}"

        # ---- dtype invariants ----
        assert self.positions.dtype in (
            torch.long,
            torch.int32,
            torch.int64,
        ), f"positions must be int dtype, got {self.positions.dtype}"
        assert self.seq_lens.dtype in (torch.long, torch.int32, torch.int64)
        assert self.extend_seq_lens.dtype in (torch.long, torch.int32, torch.int64)
        assert self.cu_seqlens_q.dtype in (torch.long, torch.int32, torch.int64)
        assert self.req_pool_indices.dtype in (torch.long, torch.int32, torch.int64)

        # ---- device invariants ----
        device = self.positions.device
        for fname in (
            "seq_lens",
            "extend_seq_lens",
            "cu_seqlens_q",
            "req_pool_indices",
        ):
            t = getattr(self, fname)
            assert (
                t.device == device
            ), f"{fname}.device={t.device} != positions.device={device}"

        # ---- forward_mode consistency ----
        if self.forward_mode == DSV4ForwardMode.DECODE and num_seqs > 0:
            # In decode mode, every seq emits exactly 1 new token.
            ext = self.extend_seq_lens
            assert bool(
                torch.all(ext == 1).item()
            ), f"DECODE mode requires extend_seq_lens == 1 for all seqs, got {ext.tolist()}"
        elif self.forward_mode == DSV4ForwardMode.PREFILL and num_seqs > 0:
            # In prefill mode, at least one seq has > 1 new token.
            ext = self.extend_seq_lens
            assert bool(torch.any(ext > 1).item()), (
                f"PREFILL mode requires at least one seq with extend_seq_len > 1, "
                f"got {ext.tolist()}"
            )

    @property
    def num_seqs(self) -> int:
        return self.seq_lens.numel()

    @property
    def num_tokens(self) -> int:
        return self.positions.numel()

    @property
    def is_decode(self) -> bool:
        return self.forward_mode == DSV4ForwardMode.DECODE

    @property
    def is_prefill(self) -> bool:
        return self.forward_mode == DSV4ForwardMode.PREFILL

    def assert_pool_slot_unique(self) -> None:
        """Runtime check: `req_pool_indices` has no duplicates.

        W4.1 callers depending on per-seq isolation (e.g., per-seq
        kv_cache row indexing) MUST call this before using
        `req_pool_indices` as a unique seq tag — the W4.1 placeholder
        adapter (`from_attn_metadata`) sets it to `block_tables[:, 0]`
        which can collide under prefix caching. W4.2 makes this
        unconditional, but until then this is the explicit gate.
        """
        if self.req_pool_indices.numel() == 0:
            return
        unique = torch.unique(self.req_pool_indices)
        if unique.numel() != self.req_pool_indices.numel():
            raise ValueError(
                "req_pool_indices is not unique — likely a prefix-cache "
                "first-block collision (W4.1 placeholder limitation). "
                "Got "
                f"{self.req_pool_indices.tolist()}. Wait for W4.2 "
                "engine KV-pool slot allocator to land before relying on "
                "uniqueness, or disable prefix caching for this run."
            )

    @classmethod
    def from_attn_metadata(
        cls,
        attn_metadata: Any,
        positions: torch.Tensor,
    ) -> "DSV4ForwardBatch":
        """Adapter: construct DSV4ForwardBatch from ATOM's
        `forward_context.attn_metadata` + positions tensor.

        Used during W4.1-W4.3 transition where the runtime still produces
        ATOM's standard AttentionMetaData but the DSV4 model is being
        migrated to consume DSV4ForwardBatch. Once W4.2 wires the engine-
        owned pool and out_cache_loc, this adapter is upgraded to
        `from_engine_state()` (W4.4).

        ATOM normally builds attn_metadata tensors on CUDA, but a few
        paths (warmup, CPU-fallback unit tests) construct them on CPU
        while `positions` lands on the model's GPU device. Normalize
        ALL fields to `positions.device` here so __post_init__'s
        device-uniformity invariant holds regardless of where the
        upstream tensors started.
        """
        device = positions.device
        cu = attn_metadata.cu_seqlens_q.to(torch.long).to(device)
        num_seqs = cu.numel() - 1
        # extend_seq_lens computed on the unified target device
        extend_seq_lens = cu[1:] - cu[:-1]

        # context_lens may not be populated in pure-prefill warmup paths;
        # fall back to extend_seq_lens (no prior KV).
        context_lens = getattr(attn_metadata, "context_lens", None)
        if context_lens is not None and context_lens.numel() == num_seqs:
            seq_lens = context_lens.to(torch.long).to(device) + extend_seq_lens
        else:
            seq_lens = extend_seq_lens.clone()

        # req_pool_indices: see field docstring — W4.1 placeholder fills
        # with block_tables[:, 0]; NOT stable-unique under prefix caching.
        # Call assert_pool_slot_unique() if uniqueness is required.
        block_tables = getattr(attn_metadata, "block_tables", None)
        if (
            block_tables is not None
            and block_tables.dim() == 2
            and block_tables.shape[0] == num_seqs
        ):
            req_pool_indices = block_tables[:, 0].to(torch.long).to(device)
        else:
            req_pool_indices = torch.arange(num_seqs, dtype=torch.long, device=device)

        # forward_mode: if any seq has extend > 1, it's prefill (mixed batch
        # is also classified as PREFILL — the runtime will decide internal
        # dispatch via cu_seqlens_q). Otherwise decode. Empty batch = idle.
        if num_seqs == 0:
            mode = DSV4ForwardMode.IDLE
        elif bool(torch.any(extend_seq_lens > 1).item()):
            mode = DSV4ForwardMode.PREFILL
        else:
            mode = DSV4ForwardMode.DECODE

        # block_tables is kept as a pass-through reference; if it lives
        # on a different device that's fine (model layers will move it
        # when consumed in W4.3+). Only the metadata tensors that
        # __post_init__ asserts on are guaranteed to share device.
        return cls(
            forward_mode=mode,
            positions=positions.to(torch.long),
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            cu_seqlens_q=cu,
            req_pool_indices=req_pool_indices,
            block_tables=block_tables,
        )
