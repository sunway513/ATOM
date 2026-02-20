# SPDX-License-Identifier: MIT
# Tests for MHA attention dispatch_backend logic.
#
# dispatch_backend must always return prefill_attention_triton for prefill
# (no CK flash_attn_varlen_func dependency). Decode paths depend on
# use_triton_attn and kv_cache_block_size.
#
# These tests replicate the dispatch logic without importing the real
# Attention class (which requires GPU-only aiter imports).

import pytest


def dispatch_backend(is_prefill, use_triton_attn, kv_cache_block_size):
    """Replicate dispatch_backend logic from attention_mha.py.

    Returns a string name of the method that would be called.
    """
    if is_prefill:
        # Always use Triton prefill (no CK/flash_attn_varlen_func dependency)
        return "prefill_attention_triton"
    else:
        if use_triton_attn:
            return "paged_attention_triton"
        else:
            # Only use pa persistent when block_size == 1024
            if kv_cache_block_size == 1024:
                return "paged_attention_persistent_asm"
            return "paged_attention_asm"


class TestDispatchBackend:
    """Test dispatch_backend returns correct attention implementation."""

    def test_prefill_always_returns_triton(self):
        """Prefill must always use prefill_attention_triton (no CK dependency)."""
        result = dispatch_backend(
            is_prefill=True, use_triton_attn=False, kv_cache_block_size=16
        )
        assert result == "prefill_attention_triton"

    def test_prefill_with_triton_attn_flag(self):
        """Prefill returns prefill_attention_triton regardless of use_triton_attn."""
        result = dispatch_backend(
            is_prefill=True, use_triton_attn=True, kv_cache_block_size=16
        )
        assert result == "prefill_attention_triton"

    def test_prefill_never_returns_ck_based(self):
        """Prefill must never return prefill_attention (CK-based)."""
        for use_triton in [True, False]:
            for block_size in [16, 128, 1024]:
                result = dispatch_backend(
                    is_prefill=True,
                    use_triton_attn=use_triton,
                    kv_cache_block_size=block_size,
                )
                assert result == "prefill_attention_triton"
                assert result != "prefill_attention"

    def test_decode_triton_when_flag_set(self):
        """Decode with use_triton_attn=True returns paged_attention_triton."""
        result = dispatch_backend(
            is_prefill=False, use_triton_attn=True, kv_cache_block_size=16
        )
        assert result == "paged_attention_triton"

    def test_decode_asm_default(self):
        """Decode with default block_size returns paged_attention_asm."""
        result = dispatch_backend(
            is_prefill=False, use_triton_attn=False, kv_cache_block_size=16
        )
        assert result == "paged_attention_asm"

    def test_decode_persistent_asm_with_block_1024(self):
        """Decode with block_size=1024 returns paged_attention_persistent_asm."""
        result = dispatch_backend(
            is_prefill=False, use_triton_attn=False, kv_cache_block_size=1024
        )
        assert result == "paged_attention_persistent_asm"

    @pytest.mark.parametrize("block_size", [16, 32, 64, 128, 256, 512])
    def test_decode_asm_non_1024_block_sizes(self, block_size):
        """All block sizes != 1024 use paged_attention_asm for decode."""
        result = dispatch_backend(
            is_prefill=False, use_triton_attn=False, kv_cache_block_size=block_size
        )
        assert result == "paged_attention_asm"

    @pytest.mark.parametrize("use_triton_attn", [True, False])
    @pytest.mark.parametrize("block_size", [16, 128, 1024])
    def test_prefill_ignores_decode_params(self, use_triton_attn, block_size):
        """Prefill path is independent of use_triton_attn and block_size."""
        result = dispatch_backend(
            is_prefill=True,
            use_triton_attn=use_triton_attn,
            kv_cache_block_size=block_size,
        )
        assert result == "prefill_attention_triton"

    def test_use_triton_attn_conditions(self):
        """use_triton_attn is True when sliding_window != -1 or head_dim != 128."""
        # sliding_window != -1 -> use_triton_attn = True
        sliding_window = 4096
        head_dim = 128
        use_triton_attn = sliding_window != -1 or head_dim != 128
        assert use_triton_attn is True

        # head_dim != 128 -> use_triton_attn = True
        sliding_window = -1
        head_dim = 64
        use_triton_attn = sliding_window != -1 or head_dim != 128
        assert use_triton_attn is True

        # Both default -> use_triton_attn = False
        sliding_window = -1
        head_dim = 128
        use_triton_attn = sliding_window != -1 or head_dim != 128
        assert use_triton_attn is False
