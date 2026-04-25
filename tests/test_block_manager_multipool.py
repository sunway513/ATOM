# SPDX-License-Identifier: MIT
# Layer 1 unit tests for atom/model_engine/block_manager.py multi-pool extension.
# Per RFC §6.2.1 / §9.5.2 — CPU only, mocked, ≤ 30s.

import pytest
import torch

from conftest import MockConfig

from atom.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    physical_pool_key,
)


def _spec_full(num_kv_heads=8):
    return FullAttentionSpec(
        block_size=4,
        page_size_bytes=2048,
        num_kv_heads=num_kv_heads,
        head_size=128,
        dtype=torch.bfloat16,
    )


def _spec_mla_c4():
    return MLAAttentionSpec(
        block_size=4,
        page_size_bytes=4096,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )


def _spec_compressor_c4():
    return SlidingWindowMLASpec(
        block_size=4,
        page_size_bytes=4096,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
        sliding_window=8,
    )


def _make_bm_legacy(num_blocks=8, block_size=4, prefix_cache=False):
    """Legacy single-pool path (no kv_cache_specs)."""
    from atom.model_engine.block_manager import BlockManager

    cfg = MockConfig(
        num_kvcache_blocks=num_blocks,
        kv_cache_block_size=block_size,
        enable_prefix_caching=prefix_cache,
    )
    return BlockManager(cfg)


def _make_bm_multipool(specs, blocks_per_pool=8, block_size=4):
    """Multi-pool path. ``specs`` maps logical name → KVCacheSpec."""
    from atom.model_engine.block_manager import BlockManager

    cfg = MockConfig(
        num_kvcache_blocks=blocks_per_pool * len(set(physical_pool_key(s) for s in specs.values())),
        kv_cache_block_size=block_size,
        enable_prefix_caching=False,
    )
    return BlockManager(
        cfg,
        kv_cache_specs=specs,
        blocks_per_pool={physical_pool_key(s): blocks_per_pool for s in specs.values()},
    )


def _make_seq(token_ids=None, block_size=4):
    from atom.model_engine.sequence import Sequence

    return Sequence(token_ids or [1, 2, 3, 4, 5], block_size=block_size)


# ── 1. Backwards-compat — legacy single-pool init still works ──────────────


class TestLegacyCompat:
    def test_legacy_init_creates_main_pool(self):
        bm = _make_bm_legacy()
        assert "main" in bm.logical_to_pool
        assert len(bm.pools) == 1

    def test_legacy_allocate_writes_main_block_table(self):
        bm = _make_bm_legacy()
        seq = _make_seq()
        bm.allocate(seq)
        assert seq.block_tables["main"] != []
        assert seq.block_table == seq.block_tables["main"]


# ── 2. Multi-pool registration ─────────────────────────────────────────────


class TestMultiPoolRegistration:
    def test_three_logical_caches_distinct_specs_three_physical_pools(self):
        bm = _make_bm_multipool(
            {
                "main": _spec_full(),
                "compress": _spec_compressor_c4(),
                "indexer": _spec_mla_c4(),
            }
        )
        # Each spec has a different physical_pool_key → 3 physical pools.
        assert len(bm.pools) == 3
        assert set(bm.logical_to_pool.keys()) == {"main", "compress", "indexer"}

    def test_two_logical_caches_same_spec_one_physical_pool(self):
        # Same spec used twice (e.g. two attention layers using the same
        # MLAAttentionSpec) must coalesce into ONE physical pool.
        spec = _spec_mla_c4()
        bm = _make_bm_multipool({"layer.0.attn.main_c4": spec, "layer.1.attn.main_c4": spec})
        assert len(bm.pools) == 1
        # Both logical names map to the same physical pool key.
        assert bm.logical_to_pool["layer.0.attn.main_c4"] == bm.logical_to_pool["layer.1.attn.main_c4"]

    def test_pool_keys_use_physical_pool_key_function(self):
        spec = _spec_mla_c4()
        bm = _make_bm_multipool({"a": spec})
        expected_key = physical_pool_key(spec)
        assert expected_key in bm.pools


# ── 3. Per-pool free_block_ids independence ────────────────────────────────


class TestPoolIndependence:
    def test_pools_have_independent_free_lists(self):
        bm = _make_bm_multipool(
            {"main": _spec_full(), "compress": _spec_compressor_c4()},
            blocks_per_pool=4,
        )
        main_key = bm.logical_to_pool["main"]
        compress_key = bm.logical_to_pool["compress"]
        assert main_key != compress_key
        # Both pools start with their own block IDs 0..3 — block-IDs are
        # local to each pool so identical block_ids in different pools
        # refer to different physical memory.
        assert len(bm.pools[main_key].free_block_ids_set) == 4
        assert len(bm.pools[compress_key].free_block_ids_set) == 4


# ── 4. Atomic allocate across multiple pools ───────────────────────────────


class TestAtomicAllocate:
    def test_can_allocate_returns_true_when_all_pools_satisfy(self):
        bm = _make_bm_multipool(
            {"main": _spec_full(), "compress": _spec_compressor_c4()},
            blocks_per_pool=4,
        )
        seq = _make_seq([1, 2, 3, 4])  # 1 block needed (block_size=4)
        assert bm.can_allocate(seq) is True

    def test_can_allocate_returns_false_when_any_pool_short(self):
        # Make one pool tiny so the seq can't fit there.
        bm = _make_bm_multipool(
            {"main": _spec_full(), "compress": _spec_compressor_c4()},
            blocks_per_pool=8,
        )
        # Drain one pool to 0 free.
        compress_key = bm.logical_to_pool["compress"]
        for _ in range(8):
            bm.pools[compress_key]._pop_free_block()
        seq = _make_seq([1, 2, 3, 4])
        assert bm.can_allocate(seq) is False

    def test_allocate_writes_block_table_for_every_logical(self):
        bm = _make_bm_multipool(
            {"main": _spec_full(), "compress": _spec_compressor_c4(), "indexer": _spec_mla_c4()}
        )
        seq = _make_seq([1, 2, 3, 4])
        bm.allocate(seq)
        assert "main" in seq.block_tables and seq.block_tables["main"]
        assert "compress" in seq.block_tables and seq.block_tables["compress"]
        assert "indexer" in seq.block_tables and seq.block_tables["indexer"]

    def test_free_releases_blocks_across_all_pools(self):
        bm = _make_bm_multipool(
            {"main": _spec_full(), "compress": _spec_compressor_c4()},
            blocks_per_pool=4,
        )
        seq = _make_seq([1, 2, 3, 4])
        bm.allocate(seq)
        # All pools have 1 block consumed.
        free_main_pre = len(bm.pools[bm.logical_to_pool["main"]].free_block_ids_set)
        free_compress_pre = len(bm.pools[bm.logical_to_pool["compress"]].free_block_ids_set)
        bm.deallocate(seq)
        free_main_post = len(bm.pools[bm.logical_to_pool["main"]].free_block_ids_set)
        free_compress_post = len(bm.pools[bm.logical_to_pool["compress"]].free_block_ids_set)
        assert free_main_post > free_main_pre
        assert free_compress_post > free_compress_pre
        # And block_tables for the seq are now empty.
        assert seq.block_tables["main"] == []
        assert seq.block_tables["compress"] == []


# ── 5. Per-logical-cache prefix-cache hash scope (Codex final pass fix) ────


class TestPrefixCacheLogicalScope:
    def test_block_carries_logical_cache_name(self):
        bm = _make_bm_multipool({"main": _spec_full(), "compress": _spec_compressor_c4()})
        seq = _make_seq([1, 2, 3, 4])
        bm.allocate(seq)
        main_key = bm.logical_to_pool["main"]
        compress_key = bm.logical_to_pool["compress"]
        # Block IDs come from each pool's local space, but each Block
        # records WHICH logical cache it belongs to so the prefix-cache
        # hash can be scoped per-logical (RFC §6.2.1 Codex final pass).
        for bid in seq.block_tables["main"]:
            assert bm.pools[main_key].blocks[bid].logical_cache_name == "main"
        for bid in seq.block_tables["compress"]:
            assert bm.pools[compress_key].blocks[bid].logical_cache_name == "compress"


# ── 6. Block.reset only clears metadata (Q8.1 audit) ───────────────────────


class TestBlockResetOnlyMetadata:
    def test_reset_clears_metadata_only_no_memory_zero(self):
        # The audit confirms ATOM Block.reset clears only ref_count / hash
        # / token_ids. Underlying KV memory is NOT zero-init on alloc; the
        # write-before-read invariant guarantees correctness (RFC §8.1).
        from atom.model_engine.block_manager import Block

        b = Block(0)
        b.ref_count = 5
        b.hash = 12345
        b.token_ids = [1, 2, 3]
        b.reset()
        assert b.ref_count == 1  # default for newly-allocated block
        assert b.hash == -1
        assert b.token_ids == []
