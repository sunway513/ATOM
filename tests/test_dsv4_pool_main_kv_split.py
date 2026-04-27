# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sprint 6 B0b — `DSV4KVPoolConfig.main_kv_nope_dtype` (issue sunway513/atom#37).

Phase A4 audit identified that ATOM allocates `_main_kv` as a single
`[L, N, ring_main, head_dim]` slab with uniform dtype. DeepSeek V4 paper
§2.3.4 mandates: "BF16 precision is used for the rotary positional embedding
(RoPE) dimensions, while FP8 precision is applied to the remaining dimensions".

Sprint 6 B0b.1 is the pool-side half: when ``main_kv_nope_dtype`` is set,
the pool allocates two physical slabs:
  _main_kv_nope: [L, N, ring_main, head_dim - rope_head_dim] @ requested dtype
  _main_kv_rope: [L, N, ring_main, rope_head_dim]            @ cfg.dtype

The legacy ``_main_kv`` becomes None in split-on mode. ``view_for_layer``
materializes a BF16-cat ``[N, ring_main, head_dim]`` tensor for the
``kv_cache`` key (concat-on-read) so existing readers (sparse_attn etc.)
see the same shape/dtype regardless of split mode. A new ``kv_cache_split``
key carries a 2-tuple ``(nope_view, rope_view)`` for split-aware writers.

Subsequent sub-commits B0b.2-5 add the pool write helper + model-side
migration. B0b.1 is purely additive — it doesn't change any model behavior
unless the new env var is set.
"""

from __future__ import annotations

import pytest
import torch

from atom.engine.kv_pool import DSV4KVPool, DSV4KVPoolConfig


def _make_cfg(main_kv_nope_dtype=None, num_c4_layers=2, num_c128_layers=1):
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
        main_kv_nope_dtype=main_kv_nope_dtype,
        device=torch.device("cpu"),
    )


def test_split_off_inherits_legacy_layout():
    """Default behavior: single ``_main_kv`` slab, dual-slab fields are None."""
    cfg = _make_cfg(main_kv_nope_dtype=None)
    pool = DSV4KVPool(cfg)
    assert pool._main_kv is not None
    assert pool._main_kv.shape == (3, 2, 128, 512)
    assert pool._main_kv.dtype == torch.bfloat16
    assert pool._main_kv_nope is None
    assert pool._main_kv_rope is None


def test_split_on_allocates_dual_slabs_with_correct_dtypes():
    """``main_kv_nope_dtype`` set -> two slabs, legacy slab is None."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    assert pool._main_kv is None
    # nope slab: head_dim - rope_head_dim = 512 - 64 = 448
    assert pool._main_kv_nope is not None
    assert pool._main_kv_nope.shape == (3, 2, 128, 448)
    assert pool._main_kv_nope.dtype == torch.float8_e4m3fn
    # rope slab: rope_head_dim = 64, dtype = cfg.dtype (bf16)
    assert pool._main_kv_rope is not None
    assert pool._main_kv_rope.shape == (3, 2, 128, 64)
    assert pool._main_kv_rope.dtype == torch.bfloat16


def test_split_off_view_for_layer_unchanged():
    """API contract: ``kv_cache`` key returns 3D BF16 ``[N, ring_main, head_dim]``."""
    cfg = _make_cfg(main_kv_nope_dtype=None)
    pool = DSV4KVPool(cfg)
    view = pool.view_for_layer(0)
    assert view["kv_cache"].shape == (2, 128, 512)
    assert view["kv_cache"].dtype == torch.bfloat16
    assert view["kv_cache_split"] is None


def test_split_on_view_for_layer_kv_cache_is_bf16_concat():
    """Split-on: ``kv_cache`` is materialized BF16 concat with same shape."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    view = pool.view_for_layer(0)
    assert view["kv_cache"].shape == (2, 128, 512)
    assert view["kv_cache"].dtype == torch.bfloat16


def test_split_on_view_for_layer_split_key_is_tuple():
    """Split-on: ``kv_cache_split`` is a 2-tuple ``(nope_view, rope_view)``."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    view = pool.view_for_layer(1)  # any layer
    split = view["kv_cache_split"]
    assert isinstance(split, tuple) and len(split) == 2
    nope_view, rope_view = split
    assert nope_view.shape == (2, 128, 448)
    assert nope_view.dtype == torch.float8_e4m3fn
    assert rope_view.shape == (2, 128, 64)
    assert rope_view.dtype == torch.bfloat16


def test_split_on_split_views_are_zero_copy():
    """Mutating the split views must round-trip into the underlying slabs."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    view = pool.view_for_layer(0)
    nope_view, rope_view = view["kv_cache_split"]
    # Cast a known fp8 value via bf16 -> fp8 round-trip
    nope_view[0, 0, 0] = torch.tensor(1.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rope_view[0, 0, 0] = torch.tensor(2.0, dtype=torch.bfloat16)
    assert pool._main_kv_nope[0, 0, 0, 0].to(torch.float32).item() == 1.0
    assert pool._main_kv_rope[0, 0, 0, 0].to(torch.float32).item() == 2.0


def test_split_on_concat_view_reflects_writes():
    """Writes to split views show up in the materialized concat read."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    view = pool.view_for_layer(0)
    nope, rope = view["kv_cache_split"]
    nope[0, 0, 0] = torch.tensor(1.0, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rope[0, 0, 0] = torch.tensor(2.0, dtype=torch.bfloat16)
    # Re-read materialized concat (note: the tensor returned earlier is a snapshot;
    # call view_for_layer again to materialize a fresh concat).
    fresh = pool.view_for_layer(0)["kv_cache"]
    # First 448 dims = nope (after fp8->bf16 cast), last 64 dims = rope (bf16 native)
    assert fresh[0, 0, 0].to(torch.float32).item() == 1.0  # nope dim 0
    assert fresh[0, 0, 448].to(torch.float32).item() == 2.0  # rope dim 0


def test_split_on_memory_savings():
    """nope slab element_size == 1 (fp8); rope stays at 2 (bf16)."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    assert pool._main_kv_nope.element_size() == 1
    assert pool._main_kv_rope.element_size() == 2


def test_split_on_no_c4_layers_safe():
    """Split-on still works if model has no compressor c4 layers."""
    cfg = _make_cfg(
        main_kv_nope_dtype=torch.float8_e4m3fn,
        num_c4_layers=0,
        num_c128_layers=2,
    )
    pool = DSV4KVPool(cfg)
    assert pool._main_kv is None
    assert pool._main_kv_nope is not None
    assert pool._main_kv_rope is not None


def test_main_kv_nope_dtype_field_default_is_none():
    """Backward-compat: field defaults to None so existing callers get legacy slab."""
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
    assert cfg.main_kv_nope_dtype is None


def test_write_main_kv_split_off_roundtrip():
    """B0b.2: split-off helper writes single slab; read-back matches input."""
    cfg = _make_cfg(main_kv_nope_dtype=None)
    # Pool has max_active_seqs=2 * ring_size_main=128 = 256 flat slots [0, 256).
    pool = DSV4KVPool(cfg)
    n_tokens = 4
    kv = torch.randn(n_tokens, 512, dtype=torch.bfloat16)
    # Write to slot 0 positions 0,1 (flat=0,1) + slot 1 positions 0,1 (flat=128,129).
    out_cache_loc = torch.tensor([0, 1, 128, 129], dtype=torch.long)
    pool.write_main_kv(layer_id=0, out_cache_loc=out_cache_loc, kv=kv)
    slab_flat = pool._main_kv[0].view(-1, 512)
    assert torch.allclose(slab_flat[0], kv[0], atol=0.0)
    assert torch.allclose(slab_flat[1], kv[1], atol=0.0)
    assert torch.allclose(slab_flat[128], kv[2], atol=0.0)
    assert torch.allclose(slab_flat[129], kv[3], atol=0.0)


def test_write_main_kv_split_on_rope_dims_bit_exact():
    """B0b.2: split-on helper preserves RoPE dims at BF16 (no FP8 coercion)."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    n_tokens = 2
    # Use values that BF16 can represent exactly.
    kv = torch.zeros(n_tokens, 512, dtype=torch.bfloat16)
    kv[0, -64:] = torch.arange(64, dtype=torch.bfloat16) / 8.0  # fractional
    kv[1, -64:] = torch.arange(64, dtype=torch.bfloat16) * 2.0
    out_cache_loc = torch.tensor([0, 1], dtype=torch.long)
    pool.write_main_kv(layer_id=0, out_cache_loc=out_cache_loc, kv=kv)
    rope_flat = pool._main_kv_rope[0].view(-1, 64)
    # RoPE dims stayed BF16 — bit-exact.
    assert torch.equal(rope_flat[0], kv[0, -64:])
    assert torch.equal(rope_flat[1], kv[1, -64:])


def test_write_main_kv_split_on_nope_dims_within_fp8_tolerance():
    """B0b.2: split-on helper writes nope dims at FP8 (within FP8 quant tolerance)."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    n_tokens = 1
    # FP8 e4m3 has limited precision; use a value that round-trips cleanly.
    kv = torch.zeros(n_tokens, 512, dtype=torch.bfloat16)
    kv[0, :448] = 1.5  # exactly representable in FP8 e4m3
    out_cache_loc = torch.tensor([0], dtype=torch.long)
    pool.write_main_kv(layer_id=0, out_cache_loc=out_cache_loc, kv=kv)
    nope_flat = pool._main_kv_nope[0].view(-1, 448)
    # Read back via cast to bf16 — should still be 1.5
    nope_bf16 = nope_flat[0].to(torch.bfloat16)
    assert torch.allclose(nope_bf16, torch.full((448,), 1.5, dtype=torch.bfloat16))


def test_write_main_kv_empty_batch_is_noop():
    """B0b.2: zero-token call must not crash and must not mutate."""
    cfg = _make_cfg(main_kv_nope_dtype=torch.float8_e4m3fn)
    pool = DSV4KVPool(cfg)
    pre_nope = pool._main_kv_nope.clone()
    pre_rope = pool._main_kv_rope.clone()
    pool.write_main_kv(
        layer_id=0,
        out_cache_loc=torch.zeros(0, dtype=torch.long),
        kv=torch.zeros(0, 512, dtype=torch.bfloat16),
    )
    assert torch.equal(pool._main_kv_nope, pre_nope)
    assert torch.equal(pool._main_kv_rope, pre_rope)


def test_write_main_kv_invalid_layer_id():
    """B0b.2: out-of-range layer_id raises IndexError."""
    cfg = _make_cfg(main_kv_nope_dtype=None)
    pool = DSV4KVPool(cfg)
    with pytest.raises(IndexError, match="layer_id"):
        pool.write_main_kv(
            layer_id=999,
            out_cache_loc=torch.tensor([0], dtype=torch.long),
            kv=torch.zeros(1, 512, dtype=torch.bfloat16),
        )


def test_split_on_assertion_when_nope_dim_invalid():
    """Misconfigured head_dim <= rope_head_dim must fail loudly, not silently."""
    with pytest.raises(AssertionError, match="nope_dim"):
        cfg = DSV4KVPoolConfig(
            max_active_seqs=1,
            num_layers=2,
            num_c4_layers=1,
            num_c128_layers=1,
            head_dim=64,
            rope_head_dim=64,  # equal → nope_dim = 0, must assert
            window_size=128,
            max_seq_len=2048,
            ring_size_main=128,
            ring_size_compressor_c4=8,
            ring_size_compressor_c128=128,
            ring_size_indexer=128,
            index_head_dim=64,
            state_inner_dim_c4=128,
            state_inner_dim_c128=64,
            compress_ratio_per_layer=[4, 128],
            dtype=torch.bfloat16,
            main_kv_nope_dtype=torch.float8_e4m3fn,
        )
        DSV4KVPool(cfg)
