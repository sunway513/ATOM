# SPDX-License-Identifier: MIT
"""W4.1 (issue #37 Path 3): DSV4ForwardBatch metadata structure tests.

Validates the dataclass invariants and the `from_attn_metadata()`
adapter without touching any model layer (W4.1 scope).
"""

from types import SimpleNamespace

import pytest
import torch

from atom.utils.dsv4_forward_batch import DSV4ForwardBatch, DSV4ForwardMode


def _make_basic_prefill():
    """4 seqs of length 12/13/15/12 packed (52 total tokens)."""
    extend = [12, 13, 15, 12]
    cu = [0, 12, 25, 40, 52]
    positions = []
    for n in extend:
        positions.extend(range(n))
    return DSV4ForwardBatch(
        forward_mode=DSV4ForwardMode.PREFILL,
        positions=torch.tensor(positions, dtype=torch.long),
        seq_lens=torch.tensor(extend, dtype=torch.long),
        extend_seq_lens=torch.tensor(extend, dtype=torch.long),
        cu_seqlens_q=torch.tensor(cu, dtype=torch.long),
        req_pool_indices=torch.tensor([10, 20, 30, 40], dtype=torch.long),
    )


def _make_basic_decode():
    """4 seqs in decode, each generating 1 new token at position 12/13/15/12."""
    return DSV4ForwardBatch(
        forward_mode=DSV4ForwardMode.DECODE,
        positions=torch.tensor([12, 13, 15, 12], dtype=torch.long),
        seq_lens=torch.tensor([13, 14, 16, 13], dtype=torch.long),
        extend_seq_lens=torch.tensor([1, 1, 1, 1], dtype=torch.long),
        cu_seqlens_q=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        req_pool_indices=torch.tensor([10, 20, 30, 40], dtype=torch.long),
    )


class TestConstruction:
    def test_basic_prefill_constructs(self):
        fb = _make_basic_prefill()
        assert fb.is_prefill
        assert not fb.is_decode
        assert fb.num_seqs == 4
        assert fb.num_tokens == 52

    def test_basic_decode_constructs(self):
        fb = _make_basic_decode()
        assert fb.is_decode
        assert not fb.is_prefill
        assert fb.num_seqs == 4
        assert fb.num_tokens == 4

    def test_idle_empty_constructs(self):
        fb = DSV4ForwardBatch(
            forward_mode=DSV4ForwardMode.IDLE,
            positions=torch.zeros(0, dtype=torch.long),
            seq_lens=torch.zeros(0, dtype=torch.long),
            extend_seq_lens=torch.zeros(0, dtype=torch.long),
            cu_seqlens_q=torch.zeros(1, dtype=torch.long),
            req_pool_indices=torch.zeros(0, dtype=torch.long),
        )
        assert fb.num_seqs == 0
        assert fb.num_tokens == 0


class TestInvariants:
    def test_positions_must_be_1d(self):
        with pytest.raises(AssertionError, match="positions must be 1-D"):
            DSV4ForwardBatch(
                forward_mode=DSV4ForwardMode.PREFILL,
                positions=torch.zeros(2, 3, dtype=torch.long),
                seq_lens=torch.tensor([3], dtype=torch.long),
                extend_seq_lens=torch.tensor([3], dtype=torch.long),
                cu_seqlens_q=torch.tensor([0, 3], dtype=torch.long),
                req_pool_indices=torch.tensor([0], dtype=torch.long),
            )

    def test_cu_seqlens_q_size_must_be_num_seqs_plus_1(self):
        with pytest.raises(AssertionError, match="cu_seqlens_q.numel"):
            DSV4ForwardBatch(
                forward_mode=DSV4ForwardMode.PREFILL,
                positions=torch.tensor([0, 1, 2], dtype=torch.long),
                seq_lens=torch.tensor([3], dtype=torch.long),
                extend_seq_lens=torch.tensor([3], dtype=torch.long),
                cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.long),  # wrong size
                req_pool_indices=torch.tensor([0], dtype=torch.long),
            )

    def test_cu_seqlens_q_tail_must_match_num_tokens(self):
        with pytest.raises(AssertionError, match="cu_seqlens_q\\[-1\\]"):
            DSV4ForwardBatch(
                forward_mode=DSV4ForwardMode.PREFILL,
                positions=torch.tensor([0, 1, 2], dtype=torch.long),
                seq_lens=torch.tensor([3], dtype=torch.long),
                extend_seq_lens=torch.tensor([3], dtype=torch.long),
                cu_seqlens_q=torch.tensor([0, 5], dtype=torch.long),  # tail != 3
                req_pool_indices=torch.tensor([0], dtype=torch.long),
            )

    def test_decode_mode_requires_all_extend_eq_1(self):
        with pytest.raises(AssertionError, match="DECODE mode requires"):
            DSV4ForwardBatch(
                forward_mode=DSV4ForwardMode.DECODE,
                positions=torch.tensor([12, 13, 14], dtype=torch.long),
                seq_lens=torch.tensor([13, 14], dtype=torch.long),
                extend_seq_lens=torch.tensor(
                    [1, 2], dtype=torch.long
                ),  # second seq has 2
                cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.long),
                req_pool_indices=torch.tensor([0, 1], dtype=torch.long),
            )

    def test_prefill_mode_requires_at_least_one_extend_gt_1(self):
        with pytest.raises(AssertionError, match="PREFILL mode requires"):
            DSV4ForwardBatch(
                forward_mode=DSV4ForwardMode.PREFILL,
                positions=torch.tensor([12, 13], dtype=torch.long),
                seq_lens=torch.tensor([13, 14], dtype=torch.long),
                extend_seq_lens=torch.tensor([1, 1], dtype=torch.long),
                cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.long),
                req_pool_indices=torch.tensor([0, 1], dtype=torch.long),
            )

    def test_dtype_must_be_int(self):
        with pytest.raises(AssertionError, match="positions must be int dtype"):
            DSV4ForwardBatch(
                forward_mode=DSV4ForwardMode.DECODE,
                positions=torch.tensor([12.0], dtype=torch.float32),
                seq_lens=torch.tensor([13], dtype=torch.long),
                extend_seq_lens=torch.tensor([1], dtype=torch.long),
                cu_seqlens_q=torch.tensor([0, 1], dtype=torch.long),
                req_pool_indices=torch.tensor([0], dtype=torch.long),
            )


class TestFromAttnMetadata:
    def test_prefill_4seqs_packed(self):
        """The exact W3.2 silicon test scenario."""
        cu = torch.tensor([0, 12, 25, 40, 52], dtype=torch.long)
        block_tables = torch.tensor(
            [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.long
        )
        context_lens = torch.zeros(4, dtype=torch.long)  # fresh request
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=block_tables, context_lens=context_lens
        )
        positions = torch.tensor(
            list(range(12)) + list(range(13)) + list(range(15)) + list(range(12)),
            dtype=torch.long,
        )
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        assert fb.is_prefill
        assert fb.num_seqs == 4
        assert fb.num_tokens == 52
        assert fb.extend_seq_lens.tolist() == [12, 13, 15, 12]
        assert fb.req_pool_indices.tolist() == [10, 20, 30, 40]

    def test_decode_4seqs_lockstep(self):
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        block_tables = torch.tensor(
            [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.long
        )
        context_lens = torch.tensor([12, 13, 15, 12], dtype=torch.long)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=block_tables, context_lens=context_lens
        )
        positions = torch.tensor([12, 13, 15, 12], dtype=torch.long)
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        assert fb.is_decode
        assert fb.seq_lens.tolist() == [13, 14, 16, 13]
        assert fb.extend_seq_lens.tolist() == [1, 1, 1, 1]
        assert fb.req_pool_indices.tolist() == [10, 20, 30, 40]

    def test_no_block_tables_falls_back_to_arange(self):
        cu = torch.tensor([0, 5], dtype=torch.long)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=None, context_lens=None
        )
        positions = torch.arange(5, dtype=torch.long)
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        assert fb.req_pool_indices.tolist() == [0]

    def test_empty_batch_yields_idle(self):
        cu = torch.tensor([0], dtype=torch.long)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=None, context_lens=None
        )
        positions = torch.zeros(0, dtype=torch.long)
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        assert fb.forward_mode == DSV4ForwardMode.IDLE

    def test_int32_cu_normalized_to_long(self):
        cu = torch.tensor([0, 3], dtype=torch.int32)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=None, context_lens=None
        )
        positions = torch.tensor([0, 1, 2], dtype=torch.int32)
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        assert fb.positions.dtype == torch.long
        assert fb.cu_seqlens_q.dtype == torch.long

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA for cross-device test"
    )
    def test_cpu_metadata_with_cuda_positions_normalizes_to_positions_device(self):
        """P1.2 regression: previously cu stayed on CPU while positions
        was on CUDA, causing __post_init__ device assert to fail. Adapter
        must move ALL metadata tensors to positions.device.
        """
        cu = torch.tensor([0, 12, 25], dtype=torch.long, device="cpu")
        block_tables = torch.tensor(
            [[10, 11], [20, 21]], dtype=torch.long, device="cpu"
        )
        context_lens = torch.zeros(2, dtype=torch.long, device="cpu")
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=block_tables, context_lens=context_lens
        )
        positions = torch.tensor(
            list(range(12)) + list(range(13)), dtype=torch.long, device="cuda"
        )
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        # All metadata tensors covered by __post_init__ device-invariant
        # must now share positions.device (= cuda).
        assert fb.positions.device.type == "cuda"
        assert fb.cu_seqlens_q.device.type == "cuda"
        assert fb.extend_seq_lens.device.type == "cuda"
        assert fb.seq_lens.device.type == "cuda"
        assert fb.req_pool_indices.device.type == "cuda"

    def test_cpu_only_metadata_all_devices_match_positions(self):
        """CPU-only sanity: when positions and metadata both start on
        CPU, __post_init__ device-uniformity is satisfied (regression-
        guards the case where the adapter accidentally diverged devices).
        """
        cu = torch.tensor([0, 12, 25], dtype=torch.long)
        block_tables = torch.tensor([[10, 11], [20, 21]], dtype=torch.long)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=block_tables, context_lens=None
        )
        positions = torch.tensor(list(range(12)) + list(range(13)), dtype=torch.long)
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        ref_dev = positions.device
        for fname in (
            "cu_seqlens_q",
            "extend_seq_lens",
            "seq_lens",
            "req_pool_indices",
        ):
            assert (
                getattr(fb, fname).device == ref_dev
            ), f"{fname} on {getattr(fb, fname).device}, expected {ref_dev}"


class TestPoolSlotUniqueness:
    """P2.1: req_pool_indices is a W4.1 placeholder filled with
    block_tables[:, 0]; under prefix caching two seqs may share a first
    block. `assert_pool_slot_unique()` is the explicit gate for callers
    that need uniqueness."""

    def test_unique_passes(self):
        fb = _make_basic_decode()
        # No exception
        fb.assert_pool_slot_unique()

    def test_duplicate_raises(self):
        # Simulate a prefix-cache collision: two seqs share first block.
        cu = torch.tensor([0, 1, 2], dtype=torch.long)
        block_tables = torch.tensor(
            [[10, 11], [10, 12]], dtype=torch.long  # same first block
        )
        context_lens = torch.tensor([5, 5], dtype=torch.long)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=block_tables, context_lens=context_lens
        )
        positions = torch.tensor([5, 5], dtype=torch.long)
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        # The placeholder copies the colliding first-block ids verbatim.
        assert fb.req_pool_indices.tolist() == [10, 10]
        with pytest.raises(ValueError, match="not unique"):
            fb.assert_pool_slot_unique()

    def test_empty_batch_skips_check(self):
        cu = torch.tensor([0], dtype=torch.long)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu, block_tables=None, context_lens=None
        )
        positions = torch.zeros(0, dtype=torch.long)
        fb = DSV4ForwardBatch.from_attn_metadata(attn_meta, positions)
        # No exception even though there are no seqs
        fb.assert_pool_slot_unique()
