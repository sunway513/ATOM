# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``DSV4KVPool`` (W4.2 — issue #37).

All tests run on CPU per ATOM CI constraint
(``CLAUDE.md``: "all tests, no GPU needed"). One optional GPU smoke test
is gated on ``torch.cuda.is_available``.

W4.2 SCOPE: pool only. Scheduler integration tests live in
``tests/test_scheduler.py`` once the admit/finish hooks land (W4.2b/W4.3).
"""

from __future__ import annotations

import pytest
import torch

from atom.engine.kv_pool import DSV4KVPool, DSV4KVPoolConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cfg(
    max_active_seqs: int = 4,
    num_layers: int = 8,
    head_dim: int = 16,
    ring_main: int = 64,
    ring_comp: int = 8,
    ring_idx: int = 64,
    state_inner_dim: int = 16,
    device: str = "cpu",
) -> DSV4KVPoolConfig:
    """Default fixture: 8 layers, [0, 4, 4, 4, 4, 128, 128, 0] ratios.

    => 4 c4 layers (1, 2, 3, 4), 2 c128 layers (5, 6), 2 plain layers (0, 7).
    """
    ratios = [0, 4, 4, 4, 4, 128, 128, 0][:num_layers]
    num_c4 = sum(1 for r in ratios if r == 4)
    num_c128 = sum(1 for r in ratios if r == 128)
    return DSV4KVPoolConfig(
        max_active_seqs=max_active_seqs,
        num_layers=num_layers,
        num_c4_layers=num_c4,
        num_c128_layers=num_c128,
        head_dim=head_dim,
        rope_head_dim=head_dim // 2,
        window_size=ring_main,
        max_seq_len=4096,
        ring_size_main=ring_main,
        ring_size_compressor=ring_comp,
        ring_size_indexer=ring_idx,
        compress_ratio_per_layer=ratios,
        state_inner_dim=state_inner_dim,
        device=torch.device(device),
        dtype=torch.bfloat16,
        state_dtype=torch.float32,
    )


@pytest.fixture
def pool() -> DSV4KVPool:
    return DSV4KVPool(_make_cfg(max_active_seqs=4))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_admit_returns_slot_in_range(pool: DSV4KVPool) -> None:
    """Slot is in [0, max_active_seqs)."""
    s = pool.admit_request(seq_id=42)
    assert 0 <= s < 4
    assert pool.num_active_seqs() == 1
    assert pool.num_free_slots() == 3


def test_admit_idempotent(pool: DSV4KVPool) -> None:
    """Re-admitting same seq_id returns same slot, does not consume new free slot."""
    s1 = pool.admit_request(7)
    free_after_first = pool.num_free_slots()
    s2 = pool.admit_request(7)
    assert s1 == s2
    # No additional slot consumed by the idempotent re-admit
    assert pool.num_free_slots() == free_after_first


def test_finish_recycles_slot(pool: DSV4KVPool) -> None:
    """Slot recycle: a finished seq's slot becomes available (LIFO)."""
    s_a = pool.admit_request(1)
    pool.admit_request(2)
    pool.admit_request(3)
    pool.admit_request(4)
    assert pool.num_free_slots() == 0
    pool.finish_request(1)
    assert pool.num_free_slots() == 1
    s_e = pool.admit_request(5)
    # LIFO free list: the just-freed slot is the one returned next.
    assert s_e == s_a, "slot was not recycled in LIFO order"


def test_finish_unknown_seq_is_noop(pool: DSV4KVPool) -> None:
    """Defensive: scheduler may double-deallocate; pool must tolerate."""
    pool.finish_request(seq_id=999)  # never admitted
    pool.admit_request(1)
    pool.finish_request(1)
    pool.finish_request(1)  # second finish, no error
    # Pool returned to empty, all slots free.
    assert pool.num_active_seqs() == 0
    assert pool.num_free_slots() == 4


def test_overadmit_raises(pool: DSV4KVPool) -> None:
    """max_active_seqs=4 — admitting a 5th distinct seq raises RuntimeError."""
    for i in range(4):
        pool.admit_request(i)
    with pytest.raises(RuntimeError, match="no free slot"):
        pool.admit_request(99)


def test_get_slot_unknown_raises(pool: DSV4KVPool) -> None:
    with pytest.raises(KeyError):
        pool.get_slot(seq_id=12345)


# ---------------------------------------------------------------------------
# THE W4.1 BUG FIX: prefix-cache first-block collision
# ---------------------------------------------------------------------------


def test_prefix_cache_first_block_collision(pool: DSV4KVPool) -> None:
    """Two seqs sharing ``block_tables[:, 0]`` (prefix cache hit) MUST get
    DIFFERENT pool slots.

    This is the exact failure mode of the W4.1 placeholder
    (``req_pool_indices = block_tables[:, 0]``). The pool is the fix.
    """
    # Simulate prefix caching: both seqs would have the same first
    # physical block id (e.g., 7) because they share a prefix. The pool
    # never consults block_tables — it allocates by seq_id only.
    seq_a, seq_b = 100, 101
    s_a = pool.admit_request(seq_a)
    s_b = pool.admit_request(seq_b)
    assert s_a != s_b, (
        "Pool slot must be stable-unique per request — even when "
        "block_tables[:, 0] would collide under prefix caching. This "
        "is the W4.1 → W4.2 contract."
    )


# ---------------------------------------------------------------------------
# Vectorized lookup
# ---------------------------------------------------------------------------


def test_get_slots_returns_device_tensor(pool: DSV4KVPool) -> None:
    pool.admit_request(10)
    pool.admit_request(20)
    pool.admit_request(30)
    t = pool.get_slots([10, 20, 30])
    assert t.dtype == torch.long
    assert t.device == pool.cfg.device
    assert t.shape == (3,)
    # Values must equal the per-seq slot:
    assert t[0].item() == pool.get_slot(10)
    assert t[1].item() == pool.get_slot(20)
    assert t[2].item() == pool.get_slot(30)


def test_get_slots_order_preserves_input(pool: DSV4KVPool) -> None:
    pool.admit_request(1)
    pool.admit_request(2)
    t1 = pool.get_slots([1, 2])
    t2 = pool.get_slots([2, 1])
    assert t1.tolist() == [pool.get_slot(1), pool.get_slot(2)]
    assert t2.tolist() == [pool.get_slot(2), pool.get_slot(1)]


# ---------------------------------------------------------------------------
# compute_out_cache_loc
# ---------------------------------------------------------------------------


def test_compute_out_cache_loc_prefill_single_seq(pool: DSV4KVPool) -> None:
    """One fresh seq, S=10 prompt tokens.

    positions=[0..9], slot=k => out_cache_loc=[k*R..k*R+9].
    """
    s = pool.admit_request(1)
    R = pool.cfg.ring_size_main
    positions = torch.arange(10, dtype=torch.long, device=pool.cfg.device)
    slot_indices = torch.tensor([s], dtype=torch.long, device=pool.cfg.device)
    cu = torch.tensor([0, 10], dtype=torch.long, device=pool.cfg.device)

    out = pool.compute_out_cache_loc(positions, slot_indices, cu, ring="main")
    expected = torch.arange(s * R, s * R + 10, dtype=torch.long)
    assert torch.equal(out.cpu(), expected)


def test_compute_out_cache_loc_decode_lockstep(pool: DSV4KVPool) -> None:
    """4 seqs, each emits 1 token at its own absolute position.

    Each ``out_cache_loc[i] = slot[i] * R + (positions[i] % R)``.
    """
    R = pool.cfg.ring_size_main
    seq_ids = [11, 12, 13, 14]
    for sid in seq_ids:
        pool.admit_request(sid)
    slot_indices = pool.get_slots(seq_ids)

    # Decode at varied positions (well past the ring wraparound point)
    positions = torch.tensor(
        [R + 5, 2 * R + 0, 3, R - 1],
        dtype=torch.long,
        device=pool.cfg.device,
    )
    cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=pool.cfg.device)

    out = pool.compute_out_cache_loc(positions, slot_indices, cu, ring="main")

    expected = slot_indices * R + (positions % R)
    assert torch.equal(out.cpu(), expected.cpu())


def test_compute_out_cache_loc_mixed_batch(pool: DSV4KVPool) -> None:
    """Mixed batch: seq A is prefill (S_a tokens), seq B is decode (1 token)."""
    s_a = pool.admit_request(101)
    s_b = pool.admit_request(102)
    R = pool.cfg.ring_size_main
    S_a = 6

    positions = torch.cat(
        [
            torch.arange(S_a, dtype=torch.long),  # A: 0..5
            torch.tensor([42], dtype=torch.long),  # B: pos=42
        ]
    ).to(pool.cfg.device)
    slot_indices = torch.tensor([s_a, s_b], dtype=torch.long, device=pool.cfg.device)
    cu = torch.tensor([0, S_a, S_a + 1], dtype=torch.long, device=pool.cfg.device)

    out = pool.compute_out_cache_loc(positions, slot_indices, cu, ring="main")

    expected = torch.cat(
        [
            torch.arange(s_a * R, s_a * R + S_a, dtype=torch.long),
            torch.tensor([s_b * R + (42 % R)], dtype=torch.long),
        ]
    )
    assert torch.equal(out.cpu(), expected)


def test_compute_out_cache_loc_compressor_ring(pool: DSV4KVPool) -> None:
    """Same formula, ``ring='compressor'`` uses the smaller ring size."""
    s = pool.admit_request(1)
    R_c = pool.cfg.ring_size_compressor
    positions = torch.tensor([0, 3, 7, 11], dtype=torch.long, device=pool.cfg.device)
    slot_indices = torch.tensor([s], dtype=torch.long, device=pool.cfg.device)
    cu = torch.tensor([0, 4], dtype=torch.long, device=pool.cfg.device)

    out = pool.compute_out_cache_loc(positions, slot_indices, cu, ring="compressor")
    expected = s * R_c + positions % R_c
    assert torch.equal(out.cpu(), expected.cpu())


def test_compute_out_cache_loc_indexer_ring(pool: DSV4KVPool) -> None:
    """``ring='indexer'`` uses the indexer-specific ring size."""
    s = pool.admit_request(1)
    R_i = pool.cfg.ring_size_indexer
    positions = torch.tensor(
        [0, R_i, R_i + 5], dtype=torch.long, device=pool.cfg.device
    )
    slot_indices = torch.tensor([s], dtype=torch.long, device=pool.cfg.device)
    cu = torch.tensor([0, 3], dtype=torch.long, device=pool.cfg.device)

    out = pool.compute_out_cache_loc(positions, slot_indices, cu, ring="indexer")
    expected = s * R_i + positions % R_i
    assert torch.equal(out.cpu(), expected.cpu())


def test_compute_out_cache_loc_unknown_ring(pool: DSV4KVPool) -> None:
    pool.admit_request(1)
    positions = torch.tensor([0], dtype=torch.long)
    slot_indices = torch.tensor([0], dtype=torch.long)
    cu = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="unknown ring"):
        pool.compute_out_cache_loc(positions, slot_indices, cu, ring="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# view_for_layer
# ---------------------------------------------------------------------------


def test_view_for_layer_main_shape(pool: DSV4KVPool) -> None:
    """layer_id=0 (compress_ratio=0 in fixture) returns main kv only."""
    view = pool.view_for_layer(layer_id=0)
    assert view["kv_cache"] is not None
    assert view["kv_cache"].shape == (
        pool.cfg.max_active_seqs,
        pool.cfg.ring_size_main,
        pool.cfg.head_dim,
    )
    assert view["kv_cache"].dtype == pool.cfg.dtype
    # ratio=0 ⇒ no compressor / indexer views
    assert view["kv_state"] is None
    assert view["score_state"] is None
    assert view["indexer_kv"] is None


def test_view_for_layer_c4_has_compressor_and_indexer(pool: DSV4KVPool) -> None:
    """layer_id=1 (compress_ratio=4 in fixture): all four views populated."""
    view = pool.view_for_layer(layer_id=1)
    assert view["kv_cache"] is not None
    assert view["kv_state"] is not None
    assert view["score_state"] is not None
    assert view["indexer_kv"] is not None
    assert view["kv_state"].shape == (
        pool.cfg.max_active_seqs,
        pool.cfg.ring_size_compressor,
        pool.cfg.state_inner_dim,
    )
    assert view["indexer_kv"].shape == (
        pool.cfg.max_active_seqs,
        pool.cfg.ring_size_indexer,
        pool.cfg.head_dim,
    )
    assert view["kv_state"].dtype == pool.cfg.state_dtype
    # score_state initialized to -inf per design
    assert torch.isinf(view["score_state"]).all().item()
    assert (view["score_state"] < 0).all().item()


def test_view_for_layer_c128_has_no_indexer(pool: DSV4KVPool) -> None:
    """layer_id=5 (compress_ratio=128 in fixture): indexer_kv is None."""
    view = pool.view_for_layer(layer_id=5)
    assert view["kv_cache"] is not None
    assert view["kv_state"] is not None  # c128 has compressor (no overlap)
    assert view["score_state"] is not None
    assert view["indexer_kv"] is None


def test_view_for_layer_zero_copy(pool: DSV4KVPool) -> None:
    """Mutating a view writes through to the underlying pool tensor."""
    view = pool.view_for_layer(layer_id=0)
    view["kv_cache"][0, 0, 0] = 1.5
    view2 = pool.view_for_layer(layer_id=0)
    assert view2["kv_cache"][0, 0, 0].item() == pytest.approx(1.5)

    view_c4 = pool.view_for_layer(layer_id=1)
    view_c4["kv_state"][0, 0, 0] = 7.25
    view_c4_again = pool.view_for_layer(layer_id=1)
    assert view_c4_again["kv_state"][0, 0, 0].item() == pytest.approx(7.25)


def test_view_for_layer_out_of_range(pool: DSV4KVPool) -> None:
    with pytest.raises(IndexError):
        pool.view_for_layer(layer_id=pool.cfg.num_layers)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_mismatched_ratio_count() -> None:
    """compress_ratio_per_layer length must equal num_layers."""
    with pytest.raises(ValueError, match="compress_ratio_per_layer length"):
        DSV4KVPoolConfig(
            max_active_seqs=2,
            num_layers=4,
            num_c4_layers=1,
            num_c128_layers=0,
            head_dim=8,
            rope_head_dim=4,
            window_size=16,
            max_seq_len=128,
            ring_size_main=16,
            ring_size_compressor=4,
            ring_size_indexer=16,
            # only 3 entries for num_layers=4
            compress_ratio_per_layer=[0, 4, 0],
            device=torch.device("cpu"),
        )


def test_config_rejects_inconsistent_c4_count() -> None:
    """num_c4_layers must match the actual number of ratio==4 entries."""
    with pytest.raises(ValueError, match="num_c4_layers"):
        DSV4KVPoolConfig(
            max_active_seqs=2,
            num_layers=4,
            num_c4_layers=2,  # claim 2, but list contains only 1
            num_c128_layers=0,
            head_dim=8,
            rope_head_dim=4,
            window_size=16,
            max_seq_len=128,
            ring_size_main=16,
            ring_size_compressor=4,
            ring_size_indexer=16,
            compress_ratio_per_layer=[0, 4, 0, 0],
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Optional GPU smoke (skipped on CI runners without GPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_gpu_smoke_compute_out_cache_loc() -> None:
    cfg = _make_cfg(max_active_seqs=4, device="cuda")
    p = DSV4KVPool(cfg)
    p.admit_request(1)
    p.admit_request(2)
    slots = p.get_slots([1, 2])
    pos = torch.arange(8, dtype=torch.long, device="cuda")
    cu = torch.tensor([0, 5, 8], dtype=torch.long, device="cuda")
    out = p.compute_out_cache_loc(pos, slots, cu, ring="main")
    assert out.device.type == "cuda"
    assert out.shape == (8,)
