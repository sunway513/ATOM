# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Engine-owned DSV4 KV / state pool (W4.2 — issue #37).

REPLACES the W4.1 placeholder in ``atom/utils/dsv4_forward_batch.py``::

    req_pool_indices = block_tables[:, 0]   # NOT stable-unique under prefix caching

with a stable per-request slot allocator (``admit_request`` / ``finish_request``)
plus a ring-encoding helper for ``out_cache_loc``.

Patterns mirrored:
- Free-list allocator: ``atom/model_engine/block_manager.py:215`` (free_mamba_slots).
- Ring-loc formula: SGLang
  ``python/sglang/srt/mem_cache/compress_state.py:147-153`` (CompressStatePool).
- Three-pool grouping: SGLang
  ``python/sglang/srt/mem_cache/deepseekv4_memory_pool.py:356`` (DeepSeekV4TokenToKVPool).
- Per-request token pool API: SGLang ReqToTokenPool.

W4.2 SCOPE
----------
- This file (NEW), the ``DSV4ForwardBatch.from_engine_state`` ctor, unit tests.
- NO model changes (``atom/models/deepseek_v4.py`` untouched). That's W4.3+.
- Scheduler ``admit/finish`` wiring is deferred to W4.2b / W4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import torch

# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

RingName = Literal["main", "compressor", "indexer"]


@dataclass
class DSV4KVPoolConfig:
    """Static config for the pool. Built from atom.config.Config in W4.2 wire-up.

    Fields
    ------
    max_active_seqs: ``Config.max_num_seqs`` (today default 512). Hard cap on
        concurrently-admitted DSV4 requests.
    num_layers: total transformer layers (target + draft if MTP).
    num_c4_layers / num_c128_layers: subset of ``num_layers`` whose
        ``compress_ratio`` is 4 / 128 respectively. Drives Compressor/Indexer
        pool sizing. From ``DeepseekV4Args.compress_ratios``.
    head_dim, rope_head_dim: from ``DeepseekV4Args``.
    window_size: main attention sliding window (``args.window_size``).
    max_seq_len: ``args.max_seq_len`` for indexer cache sizing.
    ring_size_main: ring length for the main attention sliding-window cache.
    ring_size_compressor_c4 / ring_size_compressor_c128: PER-RATIO ring lengths
        for the Compressor state slabs. Mirrors SGLang's
        ``get_compress_state_ring_size(compress_ratio)``: c4 layers (Indexer's
        inner Compressor, ratio=4, overlap=True) → 2*ratio = 8; c128 layers
        (main Compressor, ratio=128, overlap=False) → ratio = 128. Single
        unified ``ring_size_compressor`` from W4.4 was a Sprint-1 bug
        (Evidence K Bug #1).
    ring_size_indexer: ring length for the Indexer KV cache (separate from
        the inner Compressor's state — that lives in
        ``ring_size_compressor_c4``).
    compress_ratio_per_layer: one entry per layer (0 / 4 / 128). Drives
        ``view_for_layer`` dispatch — layers with ratio 0 only have a main
        kv view; ratio 4 layers also have compressor + indexer; ratio 128
        layers have compressor but no indexer.
    state_inner_dim_c4 / state_inner_dim_c128: PER-RATIO inner feature dim
        for the Compressor state slabs. c4 layer's Compressor uses
        ``head_dim = index_head_dim`` and ``coff = 2`` (overlap=True) so its
        inner = ``2 * index_head_dim``; c128 layer's Compressor uses
        ``head_dim = main attention head_dim`` and ``coff = 1`` (overlap=False)
        so its inner = ``head_dim``. Single unified ``state_inner_dim`` from
        W4.4 was a Sprint-1 bug (Evidence K Bug #2 — the c4 Compressor's
        ``[T, 2*index_head_dim]`` write didn't match the slab's
        ``[T, head_dim]`` shape).
    dtype: storage dtype (typically ``torch.bfloat16`` for kv).
    state_dtype: dtype for compressor ``kv_state`` / ``score_state`` (fp32
        per ``deepseek_v4.py:662-684``).
    device: target device (``cuda`` in production, ``cpu`` in unit tests).

    Backward compat: ``ring_size_compressor`` and ``state_inner_dim`` (the
    pre-Sprint-2 single-valued fields) are still accepted as a shorthand
    that fills BOTH per-ratio fields with the same value. Unit tests built
    in Sprint 1 still work; production callers should pass per-ratio values.
    """

    max_active_seqs: int
    num_layers: int
    num_c4_layers: int
    num_c128_layers: int
    head_dim: int
    rope_head_dim: int
    window_size: int
    max_seq_len: int
    ring_size_main: int
    # Per-ratio Compressor slab sizing (Sprint 2 — Evidence K Bug #1+#2 fix).
    ring_size_compressor_c4: int = 0
    ring_size_compressor_c128: int = 0
    ring_size_indexer: int = 0
    state_inner_dim_c4: int = 0
    state_inner_dim_c128: int = 0
    # Indexer KV cache last-dim (Sprint 2 Bug #5 fix). The Indexer's
    # compress-boundary write `kv_cache[slot, row] = kv_seq` uses
    # `index_head_dim` (typically 128), not main attention's `head_dim`
    # (typically 512). Sprint-1 sized the indexer slab with `head_dim`
    # which broke the Indexer's kv_cache write.
    index_head_dim: int = 0
    # Sprint-1 single-value fields — kept as backward-compat shortcut. If set
    # AND the per-ratio fields are not, fan out to both.
    ring_size_compressor: int = 0
    state_inner_dim: int = 0
    compress_ratio_per_layer: List[int] = field(default_factory=list)
    dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        # ---- Backward compat: fan out single-value fields to per-ratio ----
        # If caller used the Sprint-1 ``ring_size_compressor`` / ``state_inner_dim``
        # shortcut and didn't set the per-ratio variants, mirror the value to
        # both slabs. This keeps ``tests/test_dsv4_pool.py`` and other Sprint-1
        # UTs working without churn.
        if self.ring_size_compressor_c4 == 0 and self.ring_size_compressor > 0:
            self.ring_size_compressor_c4 = self.ring_size_compressor
        if self.ring_size_compressor_c128 == 0 and self.ring_size_compressor > 0:
            self.ring_size_compressor_c128 = self.ring_size_compressor
        if self.state_inner_dim_c4 == 0 and self.state_inner_dim > 0:
            self.state_inner_dim_c4 = self.state_inner_dim
        if self.state_inner_dim_c128 == 0 and self.state_inner_dim > 0:
            self.state_inner_dim_c128 = self.state_inner_dim
        # Default per-ratio inner dims to head_dim if still unset (matches
        # the Sprint-1 single-value default).
        if self.state_inner_dim_c4 == 0:
            self.state_inner_dim_c4 = self.head_dim
        if self.state_inner_dim_c128 == 0:
            self.state_inner_dim_c128 = self.head_dim
        # Default index_head_dim to head_dim if unset — preserves Sprint-1 UT
        # behavior where there's no separate index head. Production callers
        # (model_runner.py) pass the model's actual ``args.index_head_dim``.
        if self.index_head_dim == 0:
            self.index_head_dim = self.head_dim
        # Mirror back to the legacy single-value field for any reader that
        # still consults it (none in production after Sprint 2; some Sprint-1
        # UTs assert on it).
        if self.state_inner_dim == 0:
            self.state_inner_dim = self.state_inner_dim_c128
        if self.ring_size_compressor == 0:
            # Pick the larger so any legacy reader sees a valid worst-case.
            self.ring_size_compressor = max(
                self.ring_size_compressor_c4, self.ring_size_compressor_c128
            )

        if len(self.compress_ratio_per_layer) != self.num_layers:
            raise ValueError(
                f"compress_ratio_per_layer length {len(self.compress_ratio_per_layer)} "
                f"!= num_layers {self.num_layers}"
            )
        # Sanity: declared c4/c128 counts must match the per-layer list.
        actual_c4 = sum(1 for r in self.compress_ratio_per_layer if r == 4)
        actual_c128 = sum(1 for r in self.compress_ratio_per_layer if r == 128)
        if actual_c4 != self.num_c4_layers:
            raise ValueError(
                f"num_c4_layers={self.num_c4_layers} but compress_ratio_per_layer "
                f"contains {actual_c4} layers with ratio==4"
            )
        if actual_c128 != self.num_c128_layers:
            raise ValueError(
                f"num_c128_layers={self.num_c128_layers} but compress_ratio_per_layer "
                f"contains {actual_c128} layers with ratio==128"
            )


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class DSV4KVPool:
    """Engine-owned per-request KV/state pool for DSV4.

    Responsibilities
    ----------------
    1. Stable slot allocator: ``admit_request(seq_id) → slot_idx`` invariant
       across the seq's lifetime, freed on ``finish_request(seq_id)``. Slots
       are dense in ``[0, max_active_seqs)``.
    2. Owns the on-device tensors for the three logical caches: main
       sliding-window KV, Compressor overlap state, Indexer compressed-KV.
       (W4.2 declares them; W4.4 has the model start consuming them.)
    3. Provides ``compute_out_cache_loc`` so the per-step ForwardBatch can
       compute write positions without the model touching pool internals.

    Invariants
    ----------
    - ``admit_request`` is idempotent; re-admit returns the same slot until
      ``finish_request`` is called.
    - ``finish_request`` is idempotent; finishing an unknown seq is a no-op
      (matches Scheduler's defensive double-deallocate pattern at
      ``scheduler.py:578`` and ``:730``).
    - Two seqs with colliding ``block_tables[:, 0]`` get DIFFERENT slots.
      This is the bug W4.2 fixes; covered by
      ``test_prefix_cache_first_block_collision``.

    Cudagraph / TP notes
    --------------------
    - ``get_slots`` returns a fresh device tensor per call; the model
      consumes it via ``DSV4ForwardBatch.req_pool_indices`` which the graph
      treats as an input tensor (not a pool query).
    - Slot allocation runs Python-side BEFORE graph capture. All tensor ops
      after that point (``compute_out_cache_loc``, ``view_for_layer``) are
      graph-safe (``torch.bucketize`` + gather + modulo).
    - TP/PP: pool is one per TP rank; same Scheduler ⇒ same slot map ⇒ no
      cross-rank sync needed. (Cross-process Scheduler split is out of
      scope; if introduced, slot map needs broadcasting — open Q3.)
    """

    # ---- construction ----

    def __init__(self, config: DSV4KVPoolConfig) -> None:
        self.cfg = config
        self.max_active_seqs = config.max_active_seqs

        # Slot allocator: LIFO stack so the warmest slot is reused first
        # (cudagraph-friendly memory locality). Mirrors
        # ``BlockManager.free_mamba_slots`` pattern in
        # ``atom/model_engine/block_manager.py:215``.
        # Storing as a list with pop()/append() gives LIFO semantics.
        # Initial order [0, 1, ..., N-1] means first admit returns 0,
        # second admit returns 1, etc. (slot N-1 sits at the bottom of
        # the stack and is the last-recycled).
        self._free: List[int] = list(range(self.max_active_seqs - 1, -1, -1))
        self._seq_to_slot: Dict[int, int] = {}

        # Per-cache tensors. Layouts mirror the model's existing
        # ``register_buffer`` shapes (``deepseek_v4.py:662, :949, :1215``)
        # so W4.4 can rebind without resizing.
        self._main_kv: torch.Tensor
        # Sprint 2 (Evidence K Bug #1+#2): Compressor state is split into
        # per-ratio slabs. The unified ``_compressor_state`` / `_score` are
        # backward-compat views that exist only when both slabs happen to
        # share the same shape (legacy single-value config path).
        self._compressor_state_c4: Optional[torch.Tensor]
        self._compressor_score_c4: Optional[torch.Tensor]
        self._compressor_state_c128: Optional[torch.Tensor]
        self._compressor_score_c128: Optional[torch.Tensor]
        self._compressor_state: Optional[torch.Tensor]
        self._compressor_score: Optional[torch.Tensor]
        self._indexer_kv: Optional[torch.Tensor]

        # Map global layer_id → per-pool layer index for compressor / indexer.
        # Mirrors SGLang's _init_compressed_layer_mapping (deepseekv4_memory_pool.py).
        # Sprint 2: split into per-ratio maps; ``_compressor_layer_idx`` kept
        # as a union for any Sprint-1 caller still referencing it.
        self._compressor_layer_idx: Dict[int, int] = {}
        self._c4_layer_idx: Dict[int, int] = {}
        self._c128_layer_idx: Dict[int, int] = {}
        self._indexer_layer_idx: Dict[int, int] = {}

        self._build_buffers()

    def _build_buffers(self) -> None:
        """Allocate the logical caches once, sized by ``cfg``.

        Returns nothing; populates ``self._main_kv`` etc. Respects
        ``cfg.device`` (cpu in unit tests) and ``cfg.dtype`` /
        ``cfg.state_dtype``.

        Layouts (must match model side at W4.4 rebinding):
          - main_kv:                ``[num_layers, N, ring_main, head_dim]`` dtype
          - compressor_state_c4:    ``[num_c4_layers, N, ring_comp_c4, inner_c4]``
                                    state_dtype (zero-init); None if no c4 layers
          - compressor_score_c4:    same shape; fill -inf
          - compressor_state_c128:  ``[num_c128_layers, N, ring_comp_c128, inner_c128]``
                                    state_dtype (zero-init); None if no c128 layers
          - compressor_score_c128:  same shape; fill -inf
          - indexer_kv:             ``[num_c4_layers, N, ring_idx, head_dim]`` dtype
                                    None if no c4 layers

        Sprint 2 (Evidence K Bug #1+#2): the Sprint-1 design used a single
        ``_compressor_state`` slab keyed by global-layer-idx, which forced
        c4 and c128 layers to share both ring_size and inner_dim — neither
        is correct for both ratios. Splitting into per-ratio slabs lets each
        match SGLang's per-layer get_compress_state_ring_size + the per-ratio
        Compressor coff math.
        """
        cfg = self.cfg
        N = cfg.max_active_seqs

        self._main_kv = torch.zeros(
            (cfg.num_layers, N, cfg.ring_size_main, cfg.head_dim),
            dtype=cfg.dtype,
            device=cfg.device,
        )

        # ---- Compressor pool: SPLIT into c4 + c128 slabs (Sprint 2) ----
        if cfg.num_c4_layers > 0:
            self._compressor_state_c4 = torch.zeros(
                (
                    cfg.num_c4_layers,
                    N,
                    cfg.ring_size_compressor_c4,
                    cfg.state_inner_dim_c4,
                ),
                dtype=cfg.state_dtype,
                device=cfg.device,
            )
            self._compressor_score_c4 = torch.full(
                (
                    cfg.num_c4_layers,
                    N,
                    cfg.ring_size_compressor_c4,
                    cfg.state_inner_dim_c4,
                ),
                float("-inf"),
                dtype=cfg.state_dtype,
                device=cfg.device,
            )
        else:
            self._compressor_state_c4 = None
            self._compressor_score_c4 = None

        if cfg.num_c128_layers > 0:
            self._compressor_state_c128 = torch.zeros(
                (
                    cfg.num_c128_layers,
                    N,
                    cfg.ring_size_compressor_c128,
                    cfg.state_inner_dim_c128,
                ),
                dtype=cfg.state_dtype,
                device=cfg.device,
            )
            self._compressor_score_c128 = torch.full(
                (
                    cfg.num_c128_layers,
                    N,
                    cfg.ring_size_compressor_c128,
                    cfg.state_inner_dim_c128,
                ),
                float("-inf"),
                dtype=cfg.state_dtype,
                device=cfg.device,
            )
        else:
            self._compressor_state_c128 = None
            self._compressor_score_c128 = None

        # Per-ratio global-layer → slab-row maps. The unified
        # ``_compressor_layer_idx`` is kept as a backward-compat union that
        # maps to whichever slab the layer belongs to (c4 or c128). New code
        # should consult ``_c4_layer_idx`` / ``_c128_layer_idx`` directly.
        c4_idx = 0
        c128_idx = 0
        for layer_id, ratio in enumerate(cfg.compress_ratio_per_layer):
            if ratio == 4:
                self._c4_layer_idx[layer_id] = c4_idx
                self._compressor_layer_idx[layer_id] = c4_idx
                c4_idx += 1
            elif ratio == 128:
                self._c128_layer_idx[layer_id] = c128_idx
                self._compressor_layer_idx[layer_id] = c128_idx
                c128_idx += 1

        # Backward-compat union view for any Sprint-1 caller that reads
        # ``self._compressor_state`` / ``_compressor_score``. The union
        # only exists if BOTH slabs share the same shape (legacy single-value
        # config path); under per-ratio config the shapes diverge so the
        # union is None and old callers must migrate to view_for_layer.
        if (
            self._compressor_state_c4 is not None
            and self._compressor_state_c128 is not None
            and self._compressor_state_c4.shape[2:] == self._compressor_state_c128.shape[2:]
        ):
            self._compressor_state = torch.cat(
                [self._compressor_state_c4, self._compressor_state_c128], dim=0
            )
            self._compressor_score = torch.cat(
                [self._compressor_score_c4, self._compressor_score_c128], dim=0
            )
        elif self._compressor_state_c4 is not None and self._compressor_state_c128 is None:
            self._compressor_state = self._compressor_state_c4
            self._compressor_score = self._compressor_score_c4
        elif self._compressor_state_c128 is not None and self._compressor_state_c4 is None:
            self._compressor_state = self._compressor_state_c128
            self._compressor_score = self._compressor_score_c128
        else:
            self._compressor_state = None
            self._compressor_score = None

        # Indexer pool: one slab per c4 layer. Last dim is `index_head_dim`,
        # NOT main attention `head_dim` (Sprint 2 Bug #5 fix).
        if cfg.num_c4_layers > 0:
            self._indexer_kv = torch.zeros(
                (cfg.num_c4_layers, N, cfg.ring_size_indexer, cfg.index_head_dim),
                dtype=cfg.dtype,
                device=cfg.device,
            )
            idx_idx = 0
            for layer_id, ratio in enumerate(cfg.compress_ratio_per_layer):
                if ratio == 4:
                    self._indexer_layer_idx[layer_id] = idx_idx
                    idx_idx += 1
        else:
            self._indexer_kv = None

    # ---- lifecycle (called by Scheduler) ----

    def admit_request(self, seq_id: int) -> int:
        """Assign a stable slot to ``seq_id``. Idempotent.

        Raises ``RuntimeError`` if no free slot (caller is the scheduler,
        which already gates on ``max_num_seqs`` — this is a defensive
        guard).

        Returns the slot index in ``[0, max_active_seqs)``.
        """
        slot = self._seq_to_slot.get(seq_id)
        if slot is not None:
            return slot
        if not self._free:
            raise RuntimeError(
                f"DSV4KVPool: no free slot (max_active_seqs={self.max_active_seqs}). "
                "Scheduler should have gated this admit on max_num_seqs."
            )
        slot = self._free.pop()
        self._seq_to_slot[seq_id] = slot
        return slot

    def finish_request(self, seq_id: int) -> None:
        """Free the slot held by ``seq_id``. Idempotent (no-op if unknown).

        Note on Compressor/Indexer state lifetime: the slot's pool rows are
        NOT proactively zeroed here (cudagraph compat — would require a
        sync). Next admit overwrites them on first prefill. If a future
        debug build wants zero-on-free, gate it behind an env var.

        Preemption semantics (per design § open Q4): when Scheduler
        preempts a seq, it MUST call ``finish_request`` so the slot is
        reusable. The seq will be re-admitted on resume and get a fresh
        slot — Compressor/Indexer state is therefore stale, but that's
        fine because preemption already invalidates KV (the seq
        re-prefills from scratch).
        """
        slot = self._seq_to_slot.pop(seq_id, None)
        if slot is None:
            return
        self._free.append(slot)

    def get_slot(self, seq_id: int) -> int:
        """Return the slot for an admitted seq.

        Raises ``KeyError`` if ``seq_id`` is not currently admitted.
        """
        return self._seq_to_slot[seq_id]

    def get_slots(self, seq_ids: List[int]) -> torch.Tensor:
        """Vectorized lookup. Returns a fresh ``[len(seq_ids)]`` long
        tensor on ``cfg.device`` mapping each input seq to its slot.

        DO NOT cache the returned tensor across steps; cache only the int
        map (``self._seq_to_slot``). The tensor is per-step input to the
        cudagraph.
        """
        slots = [self._seq_to_slot[s] for s in seq_ids]
        return torch.tensor(slots, dtype=torch.long, device=self.cfg.device)

    def num_free_slots(self) -> int:
        return len(self._free)

    def num_active_seqs(self) -> int:
        return len(self._seq_to_slot)

    # ---- per-step indexing (called by ForwardBatch builder) ----

    def compute_out_cache_loc(
        self,
        positions: torch.Tensor,
        slot_indices: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        ring: RingName = "main",
    ) -> torch.Tensor:
        """Per-token absolute write slot in the chosen cache's flat ring.

        Formula (mirrors SGLang
        ``CompressStatePool.translate_from_swa_loc_to_state_loc``,
        ``compress_state.py:147``)::

            seg_id = bucketize(token_idx, cu_seqlens_q[1:])  # which seq each token belongs to
            slot   = slot_indices[seg_id]                    # [num_tokens]
            return slot * ring_size + (positions % ring_size)  # [num_tokens]

        Args
        ----
        positions: ``[num_tokens]`` long, per-token absolute position
            (``DSV4ForwardBatch.positions``).
        slot_indices: ``[num_seqs]`` long, per-seq slot id
            (``DSV4ForwardBatch.req_pool_indices``).
        cu_seqlens_q: ``[num_seqs+1]`` long, cumulative-tokens-per-seq
            prefix (``DSV4ForwardBatch.cu_seqlens_q``).
        ring: which cache's ring size to use.

        Returns
        -------
        ``[num_tokens]`` long tensor, on ``positions.device``.

        Prefill semantics
        -----------------
        For a fresh seq (slot=k, S new tokens at positions=[0..S-1]):
        ``out = [k*R, k*R+1, ..., k*R+S-1]``.

        Decode semantics
        ----------------
        Each seq emits 1 token at position p:
        ``out[i] = slot_indices[i] * R + (p % R)``.

        Mixed batch
        -----------
        ``cu_seqlens_q`` correctly demarcates each seq's token span, so
        the same kernel works for prefill+decode mixed batches.
        """
        ring_size = self._ring_size(ring)
        device = positions.device
        T = positions.numel()
        if T == 0:
            return torch.zeros(0, dtype=torch.long, device=device)

        # Token index in the packed batch.
        token_idx = torch.arange(T, dtype=torch.long, device=device)

        # ``bucketize(x, boundaries, right=True)`` returns the index of the
        # smallest boundary STRICTLY GREATER than x (``searchsorted`` side
        # 'right'). With boundaries = cu_seqlens_q[1:] (right edges of each
        # seq's span), token i lands in the seq whose right edge first
        # exceeds i. ``right=False`` would treat ``token == boundary`` as
        # belonging to the *previous* segment, off-by-one for boundary
        # tokens — the exact bug caught by the decode-lockstep UT.
        cu_right = cu_seqlens_q[1:].to(device=device, dtype=torch.long)
        seg_id = torch.bucketize(token_idx, cu_right, right=True)

        # Gather slot per token; align dtype/device with positions.
        slots = slot_indices.to(device=device, dtype=torch.long)
        slot_per_token = slots[seg_id]

        positions_long = positions.to(torch.long)
        return slot_per_token * ring_size + (positions_long % ring_size)

    def _ring_size(self, ring: RingName) -> int:
        if ring == "main":
            return self.cfg.ring_size_main
        if ring == "compressor":
            # Per-ratio ring sizes diverge after Sprint 2; return the
            # backward-compat union (= max of c4/c128). Production callers
            # should use ``ring_size_for_layer`` instead.
            return self.cfg.ring_size_compressor
        if ring == "indexer":
            return self.cfg.ring_size_indexer
        raise ValueError(f"unknown ring {ring!r}")

    def ring_size_for_layer(self, layer_id: int, ring: RingName) -> int:
        """Layer-aware ring size lookup (Sprint 2).

        For ``ring='main'`` and ``ring='indexer'`` returns the cfg-level value
        (these are not per-ratio). For ``ring='compressor'`` returns the
        per-ratio slab's ring (c4 → ``ring_size_compressor_c4``,
        c128 → ``ring_size_compressor_c128``).

        Raises ``ValueError`` if the layer doesn't carry a compressor (ratio==0
        and ring=='compressor').
        """
        if ring in ("main", "indexer"):
            return self._ring_size(ring)
        if ring == "compressor":
            ratio = self.cfg.compress_ratio_per_layer[layer_id]
            if ratio == 4:
                return self.cfg.ring_size_compressor_c4
            if ratio == 128:
                return self.cfg.ring_size_compressor_c128
            raise ValueError(
                f"layer_id={layer_id} has compress_ratio={ratio}; "
                "no compressor ring on this layer"
            )
        raise ValueError(f"unknown ring {ring!r}")

    # ---- model wiring (consumed in W4.3 / W4.4) ----

    def view_for_layer(self, layer_id: int) -> Dict[str, Optional[torch.Tensor]]:
        """Per-layer zero-copy views into the pool's tensors.

        Returned dict (NOT NamedTuple — extensibility in W4.5+)::

            {
              "kv_cache":    main_kv slice for this layer,
              "kv_state":    compressor kv_state slice (None for compress_ratio=0),
              "score_state": compressor score_state slice (None if no compressor),
              "indexer_kv":  indexer cache slice (None unless compress_ratio=4),
            }

        W4.3 will replace ``self.kv_cache[rows, ...]`` in
        ``deepseek_v4.py:Attention.forward`` (line 1215 register_buffer)
        with ``pool.view_for_layer(layer_id)["kv_cache"]``. W4.4 drops
        the ``register_buffer`` declarations entirely.

        Slicing is index-only (``[layer_id]``), guaranteed contiguous in
        the layout chosen in ``_build_buffers``. Consumers may further
        gather by ``slot_indices`` for per-seq rows.
        """
        if not 0 <= layer_id < self.cfg.num_layers:
            raise IndexError(
                f"layer_id={layer_id} out of range [0, {self.cfg.num_layers})"
            )

        ratio = self.cfg.compress_ratio_per_layer[layer_id]

        kv_view = self._main_kv[layer_id]

        kv_state_view: Optional[torch.Tensor] = None
        score_state_view: Optional[torch.Tensor] = None
        indexer_view: Optional[torch.Tensor] = None

        # Sprint 2: dispatch into the per-ratio slab (Evidence K Bug #1+#2 fix).
        if ratio == 4 and self._compressor_state_c4 is not None:
            c4_idx = self._c4_layer_idx[layer_id]
            kv_state_view = self._compressor_state_c4[c4_idx]
            assert self._compressor_score_c4 is not None
            score_state_view = self._compressor_score_c4[c4_idx]
        elif ratio == 128 and self._compressor_state_c128 is not None:
            c128_idx = self._c128_layer_idx[layer_id]
            kv_state_view = self._compressor_state_c128[c128_idx]
            assert self._compressor_score_c128 is not None
            score_state_view = self._compressor_score_c128[c128_idx]

        if ratio == 4 and self._indexer_kv is not None:
            idx_idx = self._indexer_layer_idx[layer_id]
            indexer_view = self._indexer_kv[idx_idx]

        return {
            "kv_cache": kv_view,
            "kv_state": kv_state_view,
            "score_state": score_state_view,
            "indexer_kv": indexer_view,
        }
