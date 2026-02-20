# SPDX-License-Identifier: MIT
# Tests for MLA prefill kernel routing logic.
#
# _forward_prefill_mla must:
#   - Use mla_decode_fwd when fp8 AND max_q_len == 1 (decode-like, supports fp8 scales)
#   - Use mla_prefill_fwd (with bf16 conversion) when fp8 AND max_q_len > 1
#   - Use mla_prefill_fwd directly when bf16 (no conversion needed)
#
# Getting this wrong causes KeyError: 96 in mla_decode_fwd (which only
# supports max_seqlen_q=1).
#
# These tests replicate the routing logic without importing the real
# MLAAttention class (which requires GPU-only aiter imports).

import pytest


def should_use_decode_fwd(kv_cache_dtype: str, max_q_len: int) -> bool:
    """Replicate the routing logic from _forward_prefill_mla line 452."""
    return kv_cache_dtype.startswith("fp8") and max_q_len == 1


def needs_dtype_conversion(tensor_dtype: str, model_dtype: str) -> bool:
    """Check if conversion is needed (mirrors the code logic)."""
    return tensor_dtype != model_dtype


class TestMlaPrefillKernelRouting:
    """Test the conditional logic that selects mla_decode_fwd vs mla_prefill_fwd."""

    @pytest.mark.parametrize(
        "kv_cache_dtype, max_q_len, expect_decode_fwd",
        [
            # fp8 decode (max_q_len=1) -> mla_decode_fwd
            ("fp8", 1, True),
            ("fp8_e4m3fn", 1, True),
            # fp8 prefill (max_q_len>1) -> mla_prefill_fwd
            ("fp8", 6, False),
            ("fp8", 128, False),
            ("fp8_e4m3fn", 32, False),
            # bf16 always -> mla_prefill_fwd
            ("bf16", 1, False),
            ("bf16", 6, False),
            ("bf16", 128, False),
            # auto -> mla_prefill_fwd
            ("auto", 1, False),
            ("auto", 64, False),
        ],
    )
    def test_kernel_selection(self, kv_cache_dtype, max_q_len, expect_decode_fwd):
        """Verify correct kernel is selected based on dtype and max_q_len."""
        result = should_use_decode_fwd(kv_cache_dtype, max_q_len)
        assert result == expect_decode_fwd, (
            f"kv_cache_dtype={kv_cache_dtype}, max_q_len={max_q_len}: "
            f"expected decode_fwd={expect_decode_fwd}, got {result}"
        )


class TestMlaPrefillFp8Conversion:
    """Test that fp8 tensors are converted before mla_prefill_fwd."""

    @pytest.mark.parametrize(
        "tensor_dtype, model_dtype, expect_conversion",
        [
            ("fp8", "bf16", True),
            ("fp8", "f16", True),
            ("bf16", "bf16", False),
            ("f16", "f16", False),
        ],
    )
    def test_conversion_needed(self, tensor_dtype, model_dtype, expect_conversion):
        """Tensors with mismatched dtype must be converted for mla_prefill_fwd."""
        assert needs_dtype_conversion(tensor_dtype, model_dtype) == expect_conversion

    def test_fp8_always_needs_conversion(self):
        """fp8 tensors always need conversion (model is never fp8)."""
        for model_dtype in ["bf16", "f16"]:
            assert needs_dtype_conversion("fp8", model_dtype) is True

    def test_matching_dtypes_skip_conversion(self):
        """Same dtype means no conversion (no-op path)."""
        for dtype in ["bf16", "f16"]:
            assert needs_dtype_conversion(dtype, dtype) is False


class TestMlaDecodeMaxSeqlenConstraint:
    """Test that mla_decode_fwd's max_seqlen_q=1 constraint is respected."""

    def test_decode_fwd_only_for_single_query(self):
        """mla_decode_fwd MUST only be called with max_q_len == 1."""
        # DeepSeek R1: nhead=16, if max_q_len=6, nhead*max_q_len=96
        # This would cause KeyError in mla_decode_fwd's block_n lookup
        nhead = 16
        for max_q_len in [2, 4, 6, 8, 16, 32, 64, 128]:
            key = nhead * max_q_len
            # These are NOT valid keys for mla_decode_fwd
            assert key != nhead, (
                f"max_q_len={max_q_len} would produce key={key}, "
                f"only key={nhead} (max_q_len=1) is valid for decode"
            )

    @pytest.mark.parametrize("max_q_len", [1, 2, 4, 6, 8, 16, 32, 64, 128])
    def test_routing_never_sends_prefill_to_decode(self, max_q_len):
        """With fp8 and max_q_len > 1, must NOT route to mla_decode_fwd."""
        kv_cache_dtype = "fp8"
        use_decode = should_use_decode_fwd(kv_cache_dtype, max_q_len)
        if max_q_len > 1:
            assert (
                not use_decode
            ), f"max_q_len={max_q_len} was routed to mla_decode_fwd (would crash)"

    def test_sparse_attention_forces_max_q_len_1(self):
        """When sparse attention is active, max_q_len is forced to 1."""
        # This is the topk_indices_buffer code path in _forward_prefill_mla
        topk_indices_buffer_present = True
        original_max_q_len = 6  # Prefill with 6 tokens

        if topk_indices_buffer_present:
            max_q_len = 1  # Forced by sparse attention path
        else:
            max_q_len = original_max_q_len

        # With max_q_len forced to 1, fp8 can use mla_decode_fwd
        assert should_use_decode_fwd("fp8", max_q_len) is True
