# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sprint 6 B0a — `DSV4KVPoolConfig.indexer_dtype` (issue sunway513/atom#37).

Phase A4 audit identified that `DSV4KVPool` allocates the indexer KV slab
with `cfg.dtype` (typically bfloat16 or float8_e4m3fn from kv_cache_dtype),
but DeepSeek V4 paper §2.3.4 specifies the lightning indexer is performed
in FP4 precision. The model already calls `fp4_act_quant_inplace` on the
indexer KV before the cache write at `deepseek_v4.py:1119`, so the
quantization math is correct — but the cache slab silently re-casts to
the wider pool dtype on storage (`deepseek_v4.py:1125: kv_seq.to(self.kv_cache.dtype)`).

This is a silent-failure bug class — no shape mismatch, no warning, just
precision loss across every Indexer top-k selection.

B0a is a non-invasive opt-in fix: a new `indexer_dtype` field on the pool
config defaults to `None` (current behavior preserved), and when set
(typically to `torch.float8_e4m3fn` as the closest practical FP4 proxy)
the indexer slab is allocated with that dtype.

These tests verify:
- Backward compat: `indexer_dtype=None` keeps existing behavior.
- Opt-in: setting `indexer_dtype=torch.float8_e4m3fn` allocates the
  indexer slab in fp8_e4m3fn while leaving every other slab untouched.
- Defensive: setting `indexer_dtype` without c4 layers is a no-op
  (no indexer slab to allocate).
"""

from __future__ import annotations

import pytest
import torch

from atom.engine.kv_pool import DSV4KVPool, DSV4KVPoolConfig


def _make_cfg(indexer_dtype=None, num_c4_layers=2, num_c128_layers=1):
    """Minimal config covering both c4 (Indexer-bearing) and c128 layers."""
    # Build ratios from scratch to exactly match the requested c4/c128 counts.
    ratios = [4] * num_c4_layers + [128] * num_c128_layers
    return DSV4KVPoolConfig(
        max_active_seqs=2,
        num_layers=len(ratios),
        num_c4_layers=num_c4_layers,
        num_c128_layers=num_c128_layers,
        head_dim=512,
        rope_head_dim=64,
        window_size=128,
        max_seq_len=2048,
        ring_size_main=128,
        ring_size_compressor_c4=8,
        ring_size_compressor_c128=128,
        ring_size_indexer=128,
        index_head_dim=128,
        state_inner_dim_c4=256,
        state_inner_dim_c128=512,
        compress_ratio_per_layer=ratios,
        dtype=torch.bfloat16,
        indexer_dtype=indexer_dtype,
        device=torch.device("cpu"),
    )


def test_indexer_dtype_default_inherits_pool_dtype():
    """Default behavior (indexer_dtype=None) preserves Sprint-1/2 layout."""
    cfg = _make_cfg(indexer_dtype=None)
    pool = DSV4KVPool(cfg)
    assert pool._indexer_kv is not None
    assert pool._indexer_kv.dtype == torch.bfloat16
    # Main slab dtype unchanged.
    assert pool._main_kv.dtype == torch.bfloat16


def test_indexer_dtype_opt_in_uses_fp8_e4m3fn():
    """Setting indexer_dtype overrides the slab allocation dtype."""
    cfg = _make_cfg(indexer_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    assert pool._indexer_kv is not None
    assert pool._indexer_kv.dtype == torch.float8_e4m3fn
    # Crucially, every OTHER slab keeps its original dtype.
    assert pool._main_kv.dtype == torch.bfloat16
    assert pool._compressor_state_c4.dtype == torch.float32
    assert pool._compressor_state_c128.dtype == torch.float32
    assert pool._compressor_main_kv_c4.dtype == torch.bfloat16
    assert pool._compressor_main_kv_c128.dtype == torch.bfloat16


def test_indexer_dtype_no_c4_layers_is_safe():
    """If model has no c4 layers, indexer_dtype is a no-op (no slab created)."""
    cfg = _make_cfg(
        indexer_dtype=torch.float8_e4m3fn, num_c4_layers=0, num_c128_layers=2
    )
    pool = DSV4KVPool(cfg)
    # No indexer slab allocated.
    assert pool._indexer_kv is None


@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.bfloat16, torch.float16],
)
def test_indexer_dtype_accepts_any_torch_dtype(dtype):
    """Pool must not hard-code an allow-list — any torch.dtype goes through."""
    cfg = _make_cfg(indexer_dtype=dtype)
    pool = DSV4KVPool(cfg)
    assert pool._indexer_kv.dtype == dtype


def test_indexer_dtype_field_default_is_none():
    """Backward-compat: the field must default to None so callers that don't set it get the legacy slab dtype."""
    # Build cfg without passing indexer_dtype.
    cfg = DSV4KVPoolConfig(
        max_active_seqs=1,
        num_layers=2,
        num_c4_layers=1,
        num_c128_layers=1,
        head_dim=512,
        rope_head_dim=64,
        window_size=128,
        max_seq_len=2048,
        ring_size_main=128,
        ring_size_compressor_c4=8,
        ring_size_compressor_c128=128,
        ring_size_indexer=128,
        index_head_dim=128,
        state_inner_dim_c4=256,
        state_inner_dim_c128=512,
        compress_ratio_per_layer=[4, 128],
        dtype=torch.bfloat16,
    )
    assert cfg.indexer_dtype is None
