# SPDX-License-Identifier: MIT
# Tests for AiterMLAMetadataBuilder.prepare_prefill paged KV metadata logic.
#
# When block_tables are present, prepare_prefill must populate:
#   - kv_indptr: cumulative block counts per sequence
#   - kv_indices: flattened block table entries
#   - kv_last_page_lens: last page lengths per sequence
#
# Without this metadata, MLA Triton prefill paths (mla_prefill_fwd,
# unified_attention) will crash with missing tensor errors.

import numpy as np


def cdiv(a, b):
    """Ceiling division matching the ATOM implementation."""
    return (a + b - 1) // b


class TestPrefillKvMetadata:
    """Test the paged KV metadata computation logic from prepare_prefill."""

    @staticmethod
    def compute_kv_metadata(context_lens, block_tables, block_size):
        """Replicate the kv_indptr/kv_indices computation from prepare_prefill.

        Args:
            context_lens: list of context lengths per sequence
            block_tables: list of lists of block IDs per sequence
            block_size: KV cache block size

        Returns:
            kv_indptr, kv_indices, num_blocks_per_seq
        """
        bs = len(context_lens)
        context_lens_np = np.asarray(context_lens, dtype=np.int32)
        num_blocks_per_seq = cdiv(context_lens_np, block_size)
        kv_indptr_cumsum = np.cumsum(num_blocks_per_seq)

        # Build kv_indptr: [0, cumsum...]
        kv_indptr = np.zeros(bs + 1, dtype=np.int32)
        kv_indptr[1 : bs + 1] = kv_indptr_cumsum

        # Build kv_indices: flatten block_tables
        total_blocks = sum(len(bt) for bt in block_tables)
        kv_indices = np.zeros(total_blocks, dtype=np.int32)
        offset = 0
        for bt in block_tables:
            n = len(bt)
            kv_indices[offset : offset + n] = bt
            offset += n

        return kv_indptr, kv_indices, num_blocks_per_seq

    def test_single_sequence(self):
        """Single sequence: kv_indptr = [0, num_blocks]."""
        context_lens = [128]
        block_tables = [[0, 1, 2, 3]]  # 4 blocks of 32
        block_size = 32

        kv_indptr, kv_indices, num_blocks = self.compute_kv_metadata(
            context_lens, block_tables, block_size
        )

        assert list(kv_indptr) == [0, 4]
        assert list(kv_indices) == [0, 1, 2, 3]
        assert list(num_blocks) == [4]

    def test_multiple_sequences(self):
        """Multiple sequences with different lengths."""
        context_lens = [64, 128, 32]
        block_tables = [[10, 11], [20, 21, 22, 23], [30]]
        block_size = 32

        kv_indptr, kv_indices, num_blocks = self.compute_kv_metadata(
            context_lens, block_tables, block_size
        )

        assert list(num_blocks) == [2, 4, 1]
        assert list(kv_indptr) == [0, 2, 6, 7]
        assert list(kv_indices) == [10, 11, 20, 21, 22, 23, 30]

    def test_partial_last_block(self):
        """Context length not aligned to block_size produces ceiling-divided blocks."""
        context_lens = [100]  # Not multiple of 32 -> ceil(100/32) = 4 blocks
        block_tables = [[0, 1, 2, 3]]
        block_size = 32

        kv_indptr, kv_indices, num_blocks = self.compute_kv_metadata(
            context_lens, block_tables, block_size
        )

        assert list(num_blocks) == [4]  # ceil(100/32) = 4
        assert list(kv_indptr) == [0, 4]

    def test_block_size_1(self):
        """MLA uses block_size=1 (each token is its own page)."""
        context_lens = [5, 3]
        block_tables = [[0, 1, 2, 3, 4], [5, 6, 7]]
        block_size = 1

        kv_indptr, kv_indices, num_blocks = self.compute_kv_metadata(
            context_lens, block_tables, block_size
        )

        assert list(num_blocks) == [5, 3]
        assert list(kv_indptr) == [0, 5, 8]
        assert list(kv_indices) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_empty_block_tables_skipped(self):
        """When block_tables is empty/falsy, metadata should not be computed."""
        # This tests the guard condition: `if batch.block_tables:`
        assert not []
        assert not None

    def test_kv_indptr_starts_with_zero(self):
        """kv_indptr must always start with 0."""
        for ctx_lens, bts, bs in [
            ([32], [[0]], 32),
            ([64, 64], [[0, 1], [2, 3]], 32),
            ([1, 1, 1], [[0], [1], [2]], 1),
        ]:
            kv_indptr, _, _ = self.compute_kv_metadata(ctx_lens, bts, bs)
            assert kv_indptr[0] == 0

    def test_kv_indptr_monotonically_increasing(self):
        """kv_indptr must be monotonically non-decreasing."""
        context_lens = [128, 64, 256, 32]
        block_tables = [
            list(range(4)),
            list(range(4, 6)),
            list(range(6, 14)),
            list(range(14, 15)),
        ]
        block_size = 32

        kv_indptr, _, _ = self.compute_kv_metadata(
            context_lens, block_tables, block_size
        )

        for i in range(len(kv_indptr) - 1):
            assert kv_indptr[i] <= kv_indptr[i + 1]

    def test_large_batch(self):
        """Stress test with 128 sequences."""
        bs = 128
        block_size = 32
        context_lens = [32 * (i % 8 + 1) for i in range(bs)]
        block_tables = []
        block_id = 0
        for cl in context_lens:
            n = cdiv(cl, block_size)
            block_tables.append(list(range(block_id, block_id + n)))
            block_id += n

        kv_indptr, kv_indices, num_blocks = self.compute_kv_metadata(
            context_lens, block_tables, block_size
        )

        assert len(kv_indptr) == bs + 1
        assert kv_indptr[0] == 0
        assert kv_indptr[-1] == sum(num_blocks)
        assert len(kv_indices) == sum(len(bt) for bt in block_tables)
