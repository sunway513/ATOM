# SPDX-License-Identifier: MIT
# Layer 1 unit tests for atom/v1/kv_cache_interface.py
# Per RFC §9.5.2 — CPU only, mocked, ≤ 30s.

import dataclasses

import pytest
import torch

from atom.v1.kv_cache_interface import (
    _VLLM_AUDIT_COMMIT,
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    physical_pool_key,
)


# ── KVCacheSpec base ───────────────────────────────────────────────────────


class TestKVCacheSpec:
    def test_basic_init(self):
        s = KVCacheSpec(block_size=256, page_size_bytes=4096)
        assert s.block_size == 256
        assert s.page_size_bytes == 4096

    def test_rejects_nonpositive_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            KVCacheSpec(block_size=0, page_size_bytes=4096)
        with pytest.raises(ValueError, match="block_size"):
            KVCacheSpec(block_size=-1, page_size_bytes=4096)

    def test_rejects_nonpositive_page_size_bytes(self):
        with pytest.raises(ValueError, match="page_size_bytes"):
            KVCacheSpec(block_size=256, page_size_bytes=0)

    def test_is_frozen_dataclass(self):
        s = KVCacheSpec(block_size=256, page_size_bytes=4096)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.block_size = 128  # type: ignore[misc]

    def test_equality_and_hash(self):
        a = KVCacheSpec(block_size=256, page_size_bytes=4096)
        b = KVCacheSpec(block_size=256, page_size_bytes=4096)
        c = KVCacheSpec(block_size=128, page_size_bytes=4096)
        assert a == b and hash(a) == hash(b)
        assert a != c and hash(a) != hash(c)


# ── AttentionSpec / FullAttentionSpec ──────────────────────────────────────


class TestAttentionSpec:
    def test_full_attention_spec_basic(self):
        s = FullAttentionSpec(
            block_size=16,
            page_size_bytes=2048,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )
        assert s.use_mla is False
        assert s.num_kv_heads == 8
        assert s.head_size == 128
        assert s.dtype == torch.bfloat16

    def test_attention_spec_rejects_nonpositive_heads_or_size(self):
        with pytest.raises(ValueError, match="num_kv_heads"):
            FullAttentionSpec(
                block_size=16,
                page_size_bytes=2048,
                num_kv_heads=0,
                head_size=128,
                dtype=torch.bfloat16,
            )
        with pytest.raises(ValueError, match="head_size"):
            FullAttentionSpec(
                block_size=16,
                page_size_bytes=2048,
                num_kv_heads=8,
                head_size=0,
                dtype=torch.bfloat16,
            )


# ── MLAAttentionSpec — DSV4 main / indexer KV ──────────────────────────────


class TestMLAAttentionSpec:
    def _make(self, *, block_size=256, compress_ratio=1):
        return MLAAttentionSpec(
            block_size=block_size,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=compress_ratio,
        )

    def test_default_use_mla_true(self):
        s = self._make()
        assert s.use_mla is True

    def test_default_compress_ratio_one_dense(self):
        s = self._make(compress_ratio=1)
        assert s.compress_ratio == 1
        assert s.storage_block_size == 256

    def test_storage_block_size_c4(self):
        s = self._make(block_size=256, compress_ratio=4)
        assert s.storage_block_size == 64  # 256 // 4

    def test_storage_block_size_c128(self):
        s = self._make(block_size=256, compress_ratio=128)
        assert s.storage_block_size == 2  # 256 // 128

    def test_rejects_compress_ratio_not_dividing_block_size(self):
        # block_size=256 with compress_ratio=7 would yield non-integer
        # storage_block_size; must reject at construction.
        with pytest.raises(ValueError, match="does not divide"):
            self._make(block_size=256, compress_ratio=7)

    def test_rejects_use_mla_false(self):
        # MLAAttentionSpec must keep use_mla=True; explicit override must fail.
        with pytest.raises(ValueError, match="use_mla=True"):
            MLAAttentionSpec(
                block_size=256,
                page_size_bytes=4096,
                num_kv_heads=1,
                head_size=512,
                dtype=torch.bfloat16,
                use_mla=False,
                compress_ratio=4,
            )


# ── SlidingWindowMLASpec — DSV4 Compressor state ───────────────────────────


class TestSlidingWindowMLASpec:
    def _make(self, *, compress_ratio=4, sliding_window=8):
        return SlidingWindowMLASpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=compress_ratio,
            sliding_window=sliding_window,
        )

    def test_compressor_c4_canonical_window(self):
        # vLLM blog: for C4 with overlap=True, sliding_window = coff * compress_ratio
        # = 2 * 4 = 8.
        s = self._make(compress_ratio=4, sliding_window=8)
        assert s.sliding_window == 8
        assert s.compress_ratio == 4
        assert s.storage_block_size == 64

    def test_compressor_c128_canonical_window(self):
        # For C128, sliding_window = 1 * 128 = 128.
        s = self._make(compress_ratio=128, sliding_window=128)
        assert s.sliding_window == 128

    def test_rejects_zero_sliding_window(self):
        with pytest.raises(ValueError, match="sliding_window"):
            self._make(sliding_window=0)

    def test_rejects_sliding_window_not_multiple_of_compress_ratio(self):
        # sliding_window=10 with compress_ratio=4 is invalid (10 % 4 != 0).
        with pytest.raises(ValueError, match="multiple of"):
            self._make(compress_ratio=4, sliding_window=10)


# ── physical_pool_key — coalescence semantics ──────────────────────────────


class TestPhysicalPoolKey:
    def test_same_subclass_same_fields_collide(self):
        # Two MLA layers with identical specs share a physical pool.
        a = MLAAttentionSpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
        )
        b = MLAAttentionSpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
        )
        assert physical_pool_key(a) == physical_pool_key(b)

    def test_different_compress_ratio_does_not_collide(self):
        c4 = MLAAttentionSpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
        )
        c128 = MLAAttentionSpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=128,
        )
        assert physical_pool_key(c4) != physical_pool_key(c128)

    def test_full_attention_does_not_collide_with_mla_dense(self):
        # FullAttentionSpec and MLAAttentionSpec(compress_ratio=1) might
        # *look* similar but they store different data layouts (K+V vs
        # latent only). The physical-pool key must distinguish them.
        full = FullAttentionSpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        )
        mla_dense = MLAAttentionSpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=1,
        )
        assert physical_pool_key(full) != physical_pool_key(mla_dense)

    def test_sliding_window_distinct_from_plain_mla(self):
        # SlidingWindowMLASpec must produce a distinct pool key from a
        # bare MLAAttentionSpec, because the underlying tensor shape /
        # ring-buffer layout differs.
        plain = MLAAttentionSpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
        )
        swa = SlidingWindowMLASpec(
            block_size=256,
            page_size_bytes=4096,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            compress_ratio=4,
            sliding_window=8,
        )
        assert physical_pool_key(plain) != physical_pool_key(swa)


# ── vLLM audit pin sanity ──────────────────────────────────────────────────


class TestVLLMAuditPin:
    def test_pin_is_a_full_sha(self):
        # Hard contract: _VLLM_AUDIT_COMMIT is a 40-char hex SHA pinning a
        # specific vLLM commit. Not a branch name, not a placeholder.
        assert isinstance(_VLLM_AUDIT_COMMIT, str)
        assert len(_VLLM_AUDIT_COMMIT) == 40, (
            f"_VLLM_AUDIT_COMMIT must be a 40-char SHA, got "
            f"{_VLLM_AUDIT_COMMIT!r}"
        )
        assert all(c in "0123456789abcdef" for c in _VLLM_AUDIT_COMMIT)
