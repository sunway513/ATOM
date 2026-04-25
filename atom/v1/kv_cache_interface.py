# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV-cache spec interface, vLLM-isomorphic.

Each stateful KV-class layer in a model declares its cache requirement by
returning a :class:`KVCacheSpec` (or one of its subclasses) from a
``get_kv_cache_spec(config)`` method. The block manager collects these specs
from the model tree, coalesces them into physical block pools by
``(page_shape, dtype, spec_kind)``, and serves per-request slot mapping +
block-table metadata to the model on every forward pass.

Field names and semantics mirror upstream vLLM
(``vllm/v1/kv_cache_interface.py``) byte-for-byte at the spec level so that
ATOM models can be lifted to vLLM with mechanical work and so that vLLM's
DSV4 model can be ported into ATOM with the same cache contract.

The pinned vLLM commit is recorded in :data:`_VLLM_AUDIT_COMMIT` and the
in-tree audit (``tests/audit/_vllm_spec_snapshot.py`` +
``tests/audit/test_spec_alignment_with_vllm.py``) keeps the field set in
sync without requiring per-PR network access.

References:
- RFC `docs/rfcs/2026-04-25-dsv4-kvcache-reform.md` v0.2.6 §6.2 / §6.2.4
- vLLM `vllm/v1/kv_cache_interface.py`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

# Pinned vLLM commit for ATOM ↔ vLLM spec alignment audit. Refreshed by a
# scheduled job (NOT per-PR) per RFC §9.5.5 — keeps per-PR CI offline.
_VLLM_AUDIT_COMMIT = "e8e38e1686c3ca0835b9556fc1f9b28b9e1a455f"  # zyongye/vllm:dsv4


@dataclass(frozen=True)
class KVCacheSpec:
    """Base spec describing one logical cache's per-block layout.

    Logical caches with identical ``(block_size, page_size_bytes)`` AND the
    same concrete subclass are coalesced into one physical block pool by the
    block manager (see RFC §6.2.1).

    Attributes:
        block_size: Number of native tokens per logical block. For DSV4
            compressed layers the canonical value is 256 (matches vLLM).
        page_size_bytes: Bytes consumed by one block in physical memory.
            Used by the block-budget allocator; computed by the model when
            wiring the spec.
    """

    block_size: int
    page_size_bytes: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(
                f"block_size must be positive, got {self.block_size}"
            )
        if self.page_size_bytes <= 0:
            raise ValueError(
                f"page_size_bytes must be positive, got {self.page_size_bytes}"
            )


@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    """Attention-style cache: K (and possibly V) over an attention block.

    Subclassed by :class:`FullAttentionSpec` (standard K+V), and by
    :class:`MLAAttentionSpec` (DeepSeek-style multi-head latent attention,
    single shared latent vector). MLA vs standard is distinguished by
    isinstance checks on the spec subclass — matches vLLM's convention
    and keeps the field set minimal.

    Attributes:
        num_kv_heads: KV-head count (1 for MLA, num_attention_heads for
            standard).
        head_size: Per-head latent / KV dim.
        dtype: Storage dtype for the cache (e.g. ``torch.bfloat16``,
            ``torch.float8_e4m3fn``).
    """

    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_kv_heads <= 0:
            raise ValueError(
                f"num_kv_heads must be positive, got {self.num_kv_heads}"
            )
        if self.head_size <= 0:
            raise ValueError(
                f"head_size must be positive, got {self.head_size}"
            )


@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    """Standard attention with full-context K and V (no compression, no
    sliding window). Used by every non-DSV4 model in ATOM today.

    Backwards-compatible default: when no DSV4 layer is present, every
    layer registers ``FullAttentionSpec`` and the block manager runs a
    single-pool path identical to pre-reform behavior.
    """


@dataclass(frozen=True)
class MLAAttentionSpec(AttentionSpec):
    """DeepSeek-V4 (and V3.2) MLA attention cache, optionally compressed.

    ``compress_ratio`` controls dynamic-sequence compression of the latent
    KV stream:

    - ``1``: standard MLA (one slot per native token, dense).
    - ``4``: c4a compression — one stored entry per 4 native tokens.
      ``storage_block_size = block_size // compress_ratio = 64`` for the
      canonical ``block_size=256`` page.
    - ``128``: c128a compression — one stored entry per 128 native tokens.
      ``storage_block_size = 2`` for ``block_size=256``.

    ``compress_ratio`` MUST divide ``block_size`` evenly.
    """

    compress_ratio: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.compress_ratio <= 0:
            raise ValueError(
                f"compress_ratio must be positive, got {self.compress_ratio}"
            )
        if self.block_size % self.compress_ratio != 0:
            raise ValueError(
                f"compress_ratio={self.compress_ratio} does not divide "
                f"block_size={self.block_size} evenly"
            )

    @property
    def storage_block_size(self) -> int:
        """Number of stored compressed entries per logical block.

        For ``block_size=256`` this is 256, 64, or 2 for compress_ratio
        1, 4, 128 respectively. Mirrors vLLM's
        ``MLAAttentionSpec.storage_block_size``.
        """
        return self.block_size // self.compress_ratio


@dataclass(frozen=True)
class SlidingWindowMLASpec(MLAAttentionSpec):
    """Sliding-window flavor of :class:`MLAAttentionSpec` used for DSV4
    Compressor state.

    Per the vLLM DSV4 blog and ``vllm/model_executor/layers/deepseek_compressor.py``,
    Compressor state is registered as sliding-window MLA-shaped KV. The
    sliding window length is ``coff * compress_ratio`` natively
    (8 for C4 with overlap=True, 128 for C128).

    ``sliding_window`` MUST be a multiple of ``compress_ratio``.
    """

    sliding_window: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sliding_window <= 0:
            raise ValueError(
                "SlidingWindowMLASpec requires sliding_window > 0, got "
                f"sliding_window={self.sliding_window}"
            )
        if self.sliding_window % self.compress_ratio != 0:
            raise ValueError(
                f"sliding_window={self.sliding_window} must be a multiple of "
                f"compress_ratio={self.compress_ratio}"
            )


# ---------------------------------------------------------------------------
# Pool-key derivation
# ---------------------------------------------------------------------------


def physical_pool_key(spec: KVCacheSpec) -> tuple:
    """Derive the physical-pool key under which the block manager coalesces
    logical caches. Two specs sharing this key share one underlying
    ``BlockPool`` (RFC §6.2.1 logical→physical coalescence).

    The key intentionally includes the concrete subclass, so for example a
    full-attention spec and an MLA spec with otherwise identical fields do
    NOT collide.
    """
    base = (type(spec).__name__, spec.block_size, spec.page_size_bytes)
    if isinstance(spec, AttentionSpec):
        base = base + (
            spec.num_kv_heads,
            spec.head_size,
            str(spec.dtype),
        )
    if isinstance(spec, MLAAttentionSpec):
        base = base + (spec.compress_ratio,)
    if isinstance(spec, SlidingWindowMLASpec):
        base = base + (spec.sliding_window,)
    return base
