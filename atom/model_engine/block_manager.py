# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-pool block manager.

Per RFC v0.2.6 §6.2.1 / §6.2.4 (logical-vs-physical pool model):

- Every stateful KV-class layer in a model declares a :class:`KVCacheSpec`
  via ``get_kv_cache_spec()``. The block manager collects these specs and
  coalesces them into physical :class:`BlockPool` instances keyed by
  ``physical_pool_key(spec)``.
- :class:`Sequence` carries a dict ``block_tables`` keyed by **logical**
  cache name. The manager looks up the physical pool for each logical
  name internally.
- Prefix-cache hash is scoped to ``(logical_cache_name, prefix_hash)`` so
  unrelated logical caches that happen to share a coalesced physical pool
  never reuse each other's blocks (Codex final-pass fix).

Backwards compatibility: ``BlockManager(config)`` without
``kv_cache_specs`` keeps single-pool behavior. The single pool is
registered under the logical name ``"main"``; non-DSV4 models continue
to read ``seq.block_table`` (which aliases ``seq.block_tables["main"]``).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import xxhash

from atom.config import Config
from atom.model_engine.sequence import Sequence
from atom.v1.kv_cache_interface import KVCacheSpec, physical_pool_key


# ---------------------------------------------------------------------------
# Block — per-block metadata (NOT the underlying KV memory; that is owned
# by the model runner and indexed via slot_mapping).
# ---------------------------------------------------------------------------


class Block:
    """Per-block metadata used by the prefix-cache hash table.

    The underlying KV memory is allocated separately by the model runner
    and indexed via ``slot_mapping``. Here we only track the bookkeeping.
    """

    __slots__ = ("block_id", "ref_count", "hash", "token_ids", "logical_cache_name")

    def __init__(self, block_id: int, logical_cache_name: str = "main"):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []
        # Which logical cache this block currently serves. Used as the
        # prefix-cache hash namespace so two unrelated logical caches that
        # share a physical pool never collide.
        self.logical_cache_name = logical_cache_name

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# ---------------------------------------------------------------------------
# BlockPool — one physical pool, may serve multiple logical caches.
# ---------------------------------------------------------------------------


class BlockPool:
    """A physical pool of blocks all sharing the same
    ``physical_pool_key`` (page shape, dtype, spec subclass).

    Multiple logical caches may map to the same physical pool when their
    specs coalesce. Prefix-cache hash is keyed by
    ``(logical_cache_name, prefix_hash)`` so cross-cache reuse is
    impossible.
    """

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool):
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.free_block_ids_set: set[int] = set(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # Keyed by (logical_cache_name, prefix_hash). RFC §6.2.1 Codex pass.
        self.hash_to_block_id: dict[tuple[str, int], int] = {}

    @staticmethod
    def compute_hash(token_ids: list[int], prefix: int = -1) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _pop_free_block(self) -> int:
        while self.free_block_ids:
            block_id = self.free_block_ids.popleft()
            if block_id in self.free_block_ids_set:
                self.free_block_ids_set.discard(block_id)
                return block_id
        raise AssertionError("No free blocks available")

    def _allocate_block(self, block_id: int, logical_cache_name: str) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # Evict stale hash entry (scoped by logical name).
        if block.hash != -1:
            stale_key = (block.logical_cache_name, block.hash)
            if self.hash_to_block_id.get(stale_key) == block_id:
                del self.hash_to_block_id[stale_key]
        block.reset()
        block.logical_cache_name = logical_cache_name
        self.free_block_ids_set.discard(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        self.free_block_ids_set.add(block_id)

    def has_free(self, n: int) -> bool:
        return len(self.free_block_ids_set) >= n


# ---------------------------------------------------------------------------
# BlockManager — coordinator over (one or many) BlockPools.
# ---------------------------------------------------------------------------


class BlockManager:
    """Multi-pool block coordinator.

    Three call signatures:

    1. ``BlockManager(config)`` — legacy single-pool path. Creates one
       :class:`BlockPool` of ``config.num_kvcache_blocks`` blocks
       registered under the logical name ``"main"``. Existing call sites
       (and existing tests) see no behavioral change.
    2. ``BlockManager(config, kv_cache_specs=...)`` — multi-pool. Each
       logical name maps to a physical pool keyed by
       ``physical_pool_key(spec)``. ``blocks_per_pool`` (optional dict
       keyed by physical pool key) controls per-pool block count;
       defaults to splitting ``config.num_kvcache_blocks`` evenly.

    All public ops (``can_allocate``, ``allocate``, ``deallocate``,
    ``can_append``, ``may_append``) iterate over physical pools and treat
    the operation atomically: alloc that fails on any pool rolls back all
    partial allocations across earlier pools (no leak).
    """

    def __init__(
        self,
        config: Config,
        kv_cache_specs: Optional[dict[str, KVCacheSpec]] = None,
        blocks_per_pool: Optional[dict[tuple, int]] = None,
    ):
        self.block_size = config.kv_cache_block_size
        self.enable_prefix_caching = config.enable_prefix_caching

        if kv_cache_specs is None:
            # Legacy single-pool — preserve pre-reform behavior exactly.
            num_blocks = config.num_kvcache_blocks
            pool = BlockPool(
                num_blocks=num_blocks,
                block_size=self.block_size,
                enable_prefix_caching=self.enable_prefix_caching,
            )
            # Synthesize a sentinel pool key for the legacy single pool.
            self._legacy_pool_key = ("__legacy_main__",)
            self.pools: dict[tuple, BlockPool] = {self._legacy_pool_key: pool}
            self.logical_to_pool: dict[str, tuple] = {"main": self._legacy_pool_key}
        else:
            self._legacy_pool_key = None
            self.pools = {}
            self.logical_to_pool = {}
            blocks_per_pool = blocks_per_pool or {}
            # Default: split config.num_kvcache_blocks evenly across unique
            # physical pools.
            unique_pool_keys = {
                physical_pool_key(spec) for spec in kv_cache_specs.values()
            }
            default_per_pool = max(
                1, config.num_kvcache_blocks // max(1, len(unique_pool_keys))
            )
            for logical_name, spec in kv_cache_specs.items():
                pkey = physical_pool_key(spec)
                if pkey not in self.pools:
                    n = blocks_per_pool.get(pkey, default_per_pool)
                    self.pools[pkey] = BlockPool(
                        num_blocks=n,
                        block_size=self.block_size,
                        enable_prefix_caching=self.enable_prefix_caching,
                    )
                self.logical_to_pool[logical_name] = pkey

        # Mamba/GDN recurrent state — single-instance, not pool-scoped.
        # Lives at the manager level for backwards compatibility with the
        # existing per-request slot bookkeeping.
        self.mamba_equiv_per_req: int = getattr(config, "mamba_equiv_per_req", 0)
        num_mamba_groups: int = getattr(config, "num_mamba_groups", 0)
        self.free_mamba_slots: list[int] = list(range(num_mamba_groups))
        self.mamba_accounting: dict[int, list[int]] = {}

    # ── Backwards-compat properties: legacy code touches bm.blocks /
    # bm.free_block_ids / bm.hash_to_block_id directly. Forward those to
    # the legacy single pool when one exists.

    @property
    def _legacy_pool(self) -> BlockPool:
        if self._legacy_pool_key is None:
            raise RuntimeError(
                "Multi-pool BlockManager: legacy aliases (bm.blocks, "
                "bm.free_block_ids, bm.hash_to_block_id) are not "
                "available; iterate self.pools[...] instead."
            )
        return self.pools[self._legacy_pool_key]

    @property
    def blocks(self) -> list[Block]:
        return self._legacy_pool.blocks

    @property
    def free_block_ids(self) -> deque[int]:
        return self._legacy_pool.free_block_ids

    @property
    def free_block_ids_set(self) -> set[int]:
        return self._legacy_pool.free_block_ids_set

    @property
    def used_block_ids(self) -> set[int]:
        return self._legacy_pool.used_block_ids

    @property
    def hash_to_block_id(self) -> dict[tuple[str, int], int]:
        return self._legacy_pool.hash_to_block_id

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        return BlockPool.compute_hash(token_ids, prefix)

    # ── Internal helpers replicating legacy single-pool behavior on a
    # given BlockPool, so legacy code paths can dispatch through the same
    # pool object.

    def _pop_free_block(self) -> int:
        # Legacy single-pool only.
        return self._legacy_pool._pop_free_block()

    def _allocate_block(self, block_id: int) -> Block:
        # Legacy single-pool only — assumes "main".
        return self._legacy_pool._allocate_block(block_id, "main")

    def _deallocate_block(self, block_id: int):
        return self._legacy_pool._deallocate_block(block_id)

    # ── Allocate / deallocate — multi-pool aware ──────────────────────────

    def _per_pool_blocks_needed(
        self, seq: Sequence, logical_names: list[str]
    ) -> dict[tuple, int]:
        """How many blocks each physical pool needs for ``seq``.

        Today every logical cache uses the same number of blocks per
        sequence (= ``seq.num_blocks``). Future work: per-spec
        ``storage_block_size`` may scale this differently.
        """
        per_pool: dict[tuple, int] = {}
        for logical_name in logical_names:
            pkey = self.logical_to_pool[logical_name]
            per_pool[pkey] = per_pool.get(pkey, 0) + seq.num_blocks
        return per_pool

    def can_allocate(self, seq: Sequence) -> bool:
        # Mamba accounting (single-instance) preserved verbatim from legacy.
        mamba_cost = self.mamba_equiv_per_req if seq.mamba_enabled else 0
        mamba_slot_ok = (not seq.mamba_enabled) or len(self.free_mamba_slots) > 0
        if not mamba_slot_ok:
            return False

        # In legacy single-pool mode mamba blocks come out of the same pool
        # as the main KV. In multi-pool mode mamba is independent of all
        # logical caches (matches today's behavior — mamba was only ever
        # used by Mamba/GDN models which don't enter the multi-pool path).
        if self._legacy_pool_key is not None:
            pool = self._legacy_pool
            if not self.enable_prefix_caching:
                return pool.has_free(seq.num_blocks + mamba_cost)
            # Prefix-cache dry-run for legacy path.
            h = -1
            cache_miss = False
            needed_free = 0
            for i in range(seq.num_blocks):
                token_ids = seq.block(i)
                h = (
                    pool.compute_hash(token_ids, h)
                    if len(token_ids) == self.block_size
                    else -1
                )
                key = ("main", h)
                block_id = pool.hash_to_block_id.get(key, -1)
                if block_id == -1 or pool.blocks[block_id].token_ids != token_ids:
                    cache_miss = True
                if cache_miss:
                    needed_free += 1
            return pool.has_free(needed_free + mamba_cost)

        # Multi-pool: every logical cache must have room.
        per_pool_need = self._per_pool_blocks_needed(seq, list(self.logical_to_pool))
        for pkey, n_needed in per_pool_need.items():
            if not self.pools[pkey].has_free(n_needed):
                return False
        return True

    def allocate(self, seq: Sequence):
        if self._legacy_pool_key is not None:
            self._allocate_legacy(seq)
            return

        # Multi-pool: allocate per logical cache. On any partial failure,
        # roll back across all earlier logical caches (no leaked blocks).
        # Codex P1: also track the IN-PROGRESS logical cache's partial
        # block_ids so an AssertionError mid-inner-loop doesn't leak the
        # blocks already popped/allocated under that logical name.
        allocated: list[tuple[str, list[int]]] = []
        cur_logical: str | None = None
        cur_block_ids: list[int] = []
        try:
            for logical_name, pkey in self.logical_to_pool.items():
                pool = self.pools[pkey]
                seq.block_tables.setdefault(logical_name, [])
                cur_logical = logical_name
                cur_block_ids = []
                for _ in range(seq.num_blocks):
                    bid = pool._pop_free_block()
                    pool._allocate_block(bid, logical_name)
                    cur_block_ids.append(bid)
                seq.block_tables[logical_name] = list(cur_block_ids)
                allocated.append((logical_name, cur_block_ids))
                cur_logical = None
                cur_block_ids = []
        except AssertionError:
            # Rollback completed logicals.
            for logical_name, block_ids in allocated:
                pool = self.pools[self.logical_to_pool[logical_name]]
                for bid in reversed(block_ids):
                    pool.blocks[bid].ref_count = 0
                    pool._deallocate_block(bid)
                seq.block_tables[logical_name] = []
            # Rollback the IN-PROGRESS logical (if any) — these blocks
            # were popped + _allocate_block'd but not yet recorded into
            # `allocated`.
            if cur_logical is not None and cur_block_ids:
                pool = self.pools[self.logical_to_pool[cur_logical]]
                for bid in reversed(cur_block_ids):
                    pool.blocks[bid].ref_count = 0
                    pool._deallocate_block(bid)
                seq.block_tables[cur_logical] = []
            raise

    def _allocate_legacy(self, seq: Sequence):
        """Legacy single-pool allocate — preserves pre-reform behavior."""
        pool = self._legacy_pool
        assert not seq.block_table
        h = -1
        cache_miss = False

        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                pool.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            key = ("main", h)
            block_id = (
                pool.hash_to_block_id.get(key, -1)
                if self.enable_prefix_caching
                else -1
            )
            if block_id == -1 or pool.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = pool._pop_free_block()
                block = pool._allocate_block(block_id, "main")
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in pool.used_block_ids:
                    block = pool.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = pool._allocate_block(block_id, "main")
            if h != -1:
                block.update(h, token_ids)
                pool.hash_to_block_id[key] = block_id
            seq.block_table.append(block_id)

        # Mamba/GDN accounting.
        if seq.mamba_enabled:
            accounting_blocks = []
            for _ in range(self.mamba_equiv_per_req):
                block_id = pool._pop_free_block()
                pool._allocate_block(block_id, "main")
                accounting_blocks.append(block_id)
            self.mamba_accounting[seq.id] = accounting_blocks
            seq.mamba_state_slot = self.free_mamba_slots.pop()

    def deallocate(self, seq: Sequence):
        if self._legacy_pool_key is not None:
            self._deallocate_legacy(seq)
            return

        # Multi-pool: free across every logical cache.
        for logical_name, block_ids in list(seq.block_tables.items()):
            pkey = self.logical_to_pool.get(logical_name)
            if pkey is None:
                continue
            pool = self.pools[pkey]
            for bid in reversed(block_ids):
                block = pool.blocks[bid]
                block.ref_count -= 1
                if block.ref_count == 0:
                    pool._deallocate_block(bid)
            seq.block_tables[logical_name] = []
        seq.num_cached_tokens = 0

    def _deallocate_legacy(self, seq: Sequence):
        pool = self._legacy_pool
        for block_id in reversed(seq.block_table):
            block = pool.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                pool._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        if seq.mamba_enabled and seq.mamba_state_slot >= 0:
            for block_id in self.mamba_accounting.pop(seq.id, []):
                block = pool.blocks[block_id]
                block.ref_count = 0  # accounting blocks bypass ref-counting
                pool._deallocate_block(block_id)
            self.free_mamba_slots.append(seq.mamba_state_slot)
            seq.mamba_state_slot = -1

    def can_append(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        seq_len = len(seq)
        needed_blocks = (
            seq_len + num_new_tokens + self.block_size - 1
        ) // self.block_size
        # Across all pools: must have room for new_blocks_needed in each.
        if self._legacy_pool_key is not None:
            current_blocks = len(seq.block_table)
            new_blocks_needed = max(0, needed_blocks - current_blocks)
            return self._legacy_pool.has_free(new_blocks_needed)
        # Multi-pool: AGGREGATE per-physical-pool demand before checking
        # (Codex P1). Two logical caches that coalesce to the same physical
        # pool would each pass an independent has_free() check while their
        # SUM exceeds capacity, leading to admission of an seq that then
        # fails to extend.
        per_pool_need: dict[tuple, int] = {}
        for logical_name, pkey in self.logical_to_pool.items():
            current = len(seq.block_tables.get(logical_name, []))
            new_needed = max(0, needed_blocks - current)
            per_pool_need[pkey] = per_pool_need.get(pkey, 0) + new_needed
        for pkey, total_need in per_pool_need.items():
            if not self.pools[pkey].has_free(total_need):
                return False
        return True

    def may_append(self, seq: Sequence, num_new_tokens: int = 1):
        if self._legacy_pool_key is not None:
            self._may_append_legacy(seq, num_new_tokens)
            return
        seq_len = len(seq)
        if 0 < seq_len % self.block_size <= num_new_tokens or self.block_size == 1:
            needed_blocks = (seq_len + self.block_size - 1) // self.block_size
            for logical_name, pkey in self.logical_to_pool.items():
                pool = self.pools[pkey]
                table = seq.block_tables.setdefault(logical_name, [])
                while len(table) < needed_blocks:
                    bid = pool._pop_free_block()
                    pool._allocate_block(bid, logical_name)
                    table.append(bid)

    def _may_append_legacy(self, seq: Sequence, num_new_tokens: int):
        pool = self._legacy_pool
        block_table = seq.block_table
        seq_len = len(seq)
        if 0 < seq_len % self.block_size <= num_new_tokens or self.block_size == 1:
            needed_blocks = (seq_len + self.block_size - 1) // self.block_size
            while len(block_table) < needed_blocks:
                block_id = pool._pop_free_block()
                pool._allocate_block(block_id, "main")
                block_table.append(block_id)
