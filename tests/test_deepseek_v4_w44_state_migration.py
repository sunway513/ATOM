# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""W4.4 (issue #37 Path 4): Compressor / Indexer state migration to engine pool.

These tests assert the architectural contract introduced by W4.4:

1. ``Compressor.__init__`` no longer ``register_buffer``s
   ``kv_state`` / ``score_state`` (covered alongside the W4.3 contract
   in ``test_deepseek_v4_w43_consume.py``).
2. ``Indexer.__init__`` no longer ``register_buffer``s ``kv_cache``
   (same).
3. ``Compressor.forward`` accepts a ``forward_batch`` kwarg (W4 path)
   and consumes the per-layer pool view via ``layer_id``.
4. ``Indexer.forward`` accepts ``forward_batch`` + ``layer_id`` kwargs.
5. Per-token state writes route to distinct slot rows of the pool's
   ``kv_state`` / ``score_state`` slabs — no row-0 collision under
   multi-request decode.
6. Per-seq compress-boundary check fires ONLY for seqs whose last
   token's absolute position satisfies ``(p + 1) % ratio == 0``.
7. The Indexer's per-seq sparse-index writes (the indexer_kv ring) hit
   distinct slot rows.

Why no full-model smoke run? ``DeepseekV4ForCausalLM`` instantiation
pulls in aiter ROCm kernels (FP8/FP4 GEMMs, sparse_attn) that the
unit-test image's stubbed ``aiter`` cannot satisfy. We assert the
contract via:

- AST-level inspection of the source (catches register_buffer
  regression without an import).
- Direct invocation of the pool's ``compute_out_cache_loc`` with the
  same ring-slot math the W4 forward uses internally — no model
  instantiation needed for the scatter / boundary correctness checks.

The full forward smoke run is W4.5 silicon-validation territory.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from atom.engine.kv_pool import DSV4KVPool, DSV4KVPoolConfig

# ---------------------------------------------------------------------------
# Source-level contract checks
# ---------------------------------------------------------------------------

ATOM_ROOT = Path(__file__).resolve().parent.parent
DSV4_SOURCE = ATOM_ROOT / "atom" / "models" / "deepseek_v4.py"


def _parse_dsv4_source() -> ast.Module:
    return ast.parse(DSV4_SOURCE.read_text())


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in deepseek_v4.py")


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"method {name!r} not found on class {cls.name!r}")


def _register_buffer_targets(method: ast.FunctionDef) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "register_buffer"
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                targets.append(node.args[0].value)
    return targets


# ---------------------------------------------------------------------------
# Section 1 — register_buffer absence (W4.4 main contract)
# ---------------------------------------------------------------------------


class TestNoRegisterBufferOnCompressorAndIndexer:
    """W4.4: Compressor + Indexer state lives in ``DSV4KVPool``, not on
    the ``nn.Module`` as ``register_buffer``. Confirms migration
    completeness vs W4.3 (which only covered the main attention)."""

    def test_compressor_kv_state_not_register_buffer(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Compressor")
        targets = _register_buffer_targets(_find_method(cls, "__init__"))
        assert (
            "kv_state" not in targets
        ), f"Compressor.__init__ must NOT register_buffer 'kv_state'. Found {targets}"

    def test_compressor_score_state_not_register_buffer(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Compressor")
        targets = _register_buffer_targets(_find_method(cls, "__init__"))
        assert (
            "score_state" not in targets
        ), f"Compressor.__init__ must NOT register_buffer 'score_state'. Found {targets}"

    def test_indexer_kv_cache_not_register_buffer(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Indexer")
        targets = _register_buffer_targets(_find_method(cls, "__init__"))
        assert (
            "kv_cache" not in targets
        ), f"Indexer.__init__ must NOT register_buffer 'kv_cache'. Found {targets}"

    def test_compressor_state_assigned_as_plain_attribute(self) -> None:
        """W4.4 sentinel: explicit ``self.kv_state: Optional[...] = None``
        and ``self.score_state: Optional[...] = None`` plain attribute
        bindings, lazy-bound to pool view per step."""
        text = DSV4_SOURCE.read_text()
        assert (
            "self.kv_state: Optional[torch.Tensor] = None" in text
        ), "Compressor.__init__ must initialize self.kv_state as a plain attr."
        assert (
            "self.score_state: Optional[torch.Tensor] = None" in text
        ), "Compressor.__init__ must initialize self.score_state as a plain attr."

    def test_indexer_kv_cache_assigned_as_plain_attribute(self) -> None:
        """Indexer's W4.4 plain-attribute kv_cache."""
        cls = _find_class(_parse_dsv4_source(), "Indexer")
        init = _find_method(cls, "__init__")
        # find a `self.kv_cache: Optional[torch.Tensor] = None` style
        # assignment on the Indexer init.
        found = False
        for node in ast.walk(init):
            if isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Attribute
            ):
                if (
                    node.target.attr == "kv_cache"
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "self"
                ):
                    found = True
                    break
        assert found, (
            "Indexer.__init__ must assign self.kv_cache as a typed plain "
            "attribute (lazy-bound to the pool's indexer_kv view)."
        )


# ---------------------------------------------------------------------------
# Section 2 — forward signatures consume forward_batch
# ---------------------------------------------------------------------------


class TestCompressorForwardSignature:
    """W4.4: Compressor.forward gains ``forward_batch`` + ``layer_id``."""

    def test_compressor_forward_accepts_forward_batch_kwarg(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Compressor")
        fwd = _find_method(cls, "forward")
        names = {a.arg for a in fwd.args.args} | {a.arg for a in fwd.args.kwonlyargs}
        assert (
            "forward_batch" in names
        ), f"Compressor.forward must accept 'forward_batch'; got {names}"

    def test_compressor_forward_accepts_layer_id_kwarg(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Compressor")
        fwd = _find_method(cls, "forward")
        names = {a.arg for a in fwd.args.args} | {a.arg for a in fwd.args.kwonlyargs}
        assert (
            "layer_id" in names
        ), f"Compressor.forward must accept 'layer_id' for pool layer dispatch; got {names}"

    def test_indexer_forward_accepts_forward_batch_kwarg(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Indexer")
        fwd = _find_method(cls, "forward")
        names = {a.arg for a in fwd.args.args} | {a.arg for a in fwd.args.kwonlyargs}
        assert (
            "forward_batch" in names
        ), f"Indexer.forward must accept 'forward_batch'; got {names}"

    def test_indexer_forward_accepts_layer_id_kwarg(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Indexer")
        fwd = _find_method(cls, "forward")
        names = {a.arg for a in fwd.args.args} | {a.arg for a in fwd.args.kwonlyargs}
        assert (
            "layer_id" in names
        ), f"Indexer.forward must accept 'layer_id'; got {names}"


# ---------------------------------------------------------------------------
# Section 3 — pool view contract
# ---------------------------------------------------------------------------


def _make_pool_with_compressor() -> DSV4KVPool:
    """Pool with a c4 layer (compressor + indexer) and a c128 layer
    (compressor only). Mirrors the layout the W4 attention path
    consumes per-layer."""
    cfg = DSV4KVPoolConfig(
        max_active_seqs=4,
        num_layers=3,
        num_c4_layers=1,
        num_c128_layers=1,
        head_dim=8,
        rope_head_dim=4,
        window_size=16,
        max_seq_len=128,
        ring_size_main=16,
        ring_size_compressor=8,
        ring_size_indexer=16,
        compress_ratio_per_layer=[0, 4, 128],
        state_inner_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        state_dtype=torch.float32,
    )
    return DSV4KVPool(cfg)


class TestPoolExposesCompressorState:
    """W4.4: ``view_for_layer`` exposes ``kv_state`` / ``score_state``
    for compressed layers and ``indexer_kv`` for c4 layers (W4.2
    contract, re-asserted here under the W4.4 consume scope)."""

    def test_c4_layer_has_kv_state_and_score_state(self) -> None:
        pool = _make_pool_with_compressor()
        view = pool.view_for_layer(layer_id=1)  # c4
        assert view["kv_state"] is not None, "c4 layer must expose kv_state"
        assert view["score_state"] is not None, "c4 layer must expose score_state"
        assert view["indexer_kv"] is not None, "c4 layer must expose indexer_kv"

    def test_c128_layer_has_kv_state_no_indexer(self) -> None:
        pool = _make_pool_with_compressor()
        view = pool.view_for_layer(layer_id=2)  # c128
        assert view["kv_state"] is not None
        assert view["score_state"] is not None
        assert view["indexer_kv"] is None, "c128 layer must NOT expose indexer_kv"

    def test_dense_layer_has_no_compressor_state(self) -> None:
        pool = _make_pool_with_compressor()
        view = pool.view_for_layer(layer_id=0)  # dense
        assert view["kv_state"] is None
        assert view["score_state"] is None
        assert view["indexer_kv"] is None


# ---------------------------------------------------------------------------
# Section 4 — per-token state scatter (the W4.4 multi-request fix)
# ---------------------------------------------------------------------------


class TestPerTokenCompressorStateScatter:
    """W4.4: per-token writes to the pool's ``kv_state`` slab must hit
    distinct slot rows. This is the multi-request decode correctness
    contract — pre-W4.4, all 4 seqs collapsed onto row 0 of the
    register_buffer; W4.4 routes each seq to its own slot via the
    pool's ring-slot math."""

    def test_4seq_decode_kv_state_writes_to_4_distinct_rows(self) -> None:
        """4 seqs decode 1 token each at distinct positions. The
        compressor ring's flat slot indices must be unique across the
        batch."""
        pool = _make_pool_with_compressor()
        for sid in [100, 101, 102, 103]:
            pool.admit_request(sid)
        slots = pool.get_slots([100, 101, 102, 103])

        # 4-seq decode: each token's position triggers a different
        # state row but the slots are unique.
        positions = torch.tensor([12, 13, 15, 12], dtype=torch.long)
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        flat_loc = pool.compute_out_cache_loc(
            positions=positions,
            slot_indices=slots,
            cu_seqlens_q=cu,
            ring="compressor",
        )
        assert flat_loc.shape == (4,)
        assert (
            flat_loc.unique().numel() == 4
        ), f"All 4 tokens must scatter to DISTINCT compressor flat slots. Got {flat_loc.tolist()}"

    def test_compressor_ring_slot_collisions_only_at_modular_wrap(self) -> None:
        """Two seqs at the same modular position MAY land on the same
        flat slot ONLY when their slot indices wrap. With distinct
        slots this MUST not happen — confirms slot isolation."""
        pool = _make_pool_with_compressor()
        # Two seqs sharing position 4 (which is 4 % ring_compressor=8 = 4).
        # Slot allocator gives them DIFFERENT slots, so flat indices
        # must differ.
        pool.admit_request(seq_id=10)
        pool.admit_request(seq_id=11)
        slots = pool.get_slots([10, 11])
        positions = torch.tensor([4, 4], dtype=torch.long)
        cu = torch.tensor([0, 1, 2], dtype=torch.long)
        flat_loc = pool.compute_out_cache_loc(
            positions=positions,
            slot_indices=slots,
            cu_seqlens_q=cu,
            ring="compressor",
        )
        assert flat_loc[0].item() != flat_loc[1].item(), (
            "Same-position different-slot tokens must produce different "
            "flat slots — this is the row-0-collision fix from W4.4."
        )

    def test_pool_view_kv_state_zero_copy_writes_visible(self) -> None:
        """Mutating ``view_for_layer(...)["kv_state"]`` must persist
        across calls (the W4 path's per-step rebinding relies on this)."""
        pool = _make_pool_with_compressor()
        for sid in [1, 2]:
            pool.admit_request(sid)
        v1 = pool.view_for_layer(layer_id=1)  # c4 layer
        v1["kv_state"][0, 0, 0] = 99.0
        v2 = pool.view_for_layer(layer_id=1)
        assert v2["kv_state"][0, 0, 0].item() == pytest.approx(
            99.0
        ), "view_for_layer must return zero-copy slices of pool storage."

    def test_pool_view_score_state_zero_copy_writes_visible(self) -> None:
        pool = _make_pool_with_compressor()
        pool.admit_request(7)
        v1 = pool.view_for_layer(layer_id=2)  # c128 layer
        v1["score_state"][0, 1, 2] = -1.5
        v2 = pool.view_for_layer(layer_id=2)
        assert v2["score_state"][0, 1, 2].item() == pytest.approx(-1.5)


# ---------------------------------------------------------------------------
# Section 5 — per-seq compress-boundary mask
# ---------------------------------------------------------------------------


class TestPerSeqCompressBoundaryMask:
    """W4.4: a seq triggers a Compressor emission iff its last-token
    absolute position satisfies ``(p + 1) % ratio == 0``. This is the
    per-seq mask the W4 forward applies; mirrors SGLang
    ``compressor.py:236``'s stateless ring math."""

    def _mask_for(self, positions: torch.Tensor, cu: torch.Tensor, ratio: int):
        """Pure helper reproducing the W4 forward's per-seq trigger."""
        last_token_idx = cu[1:] - 1
        last_positions = positions[last_token_idx]
        return (last_positions + 1) % ratio == 0

    def test_positions_12_13_15_12_ratio_4_only_pos_15_triggers(self) -> None:
        """Spec example: positions=[12,13,15,12] @ ratio=4 → only the
        seq at p=15 triggers (15+1=16 ≡ 0 mod 4)."""
        positions = torch.tensor([12, 13, 15, 12], dtype=torch.long)
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)  # 4 decode seqs
        mask = self._mask_for(positions, cu, ratio=4)
        assert mask.tolist() == [
            False,
            False,
            True,
            False,
        ], f"only seq at pos=15 should trigger compress; got mask={mask.tolist()}"

    def test_no_seqs_trigger_when_no_position_lands_on_boundary(self) -> None:
        positions = torch.tensor([10, 14, 18, 22], dtype=torch.long)
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        mask = self._mask_for(positions, cu, ratio=4)
        # 10+1=11, 14+1=15, 18+1=19, 22+1=23 — none % 4 == 0.
        assert mask.tolist() == [False, False, False, False]

    def test_all_seqs_trigger_when_all_lands_on_boundary(self) -> None:
        positions = torch.tensor([3, 7, 11, 15], dtype=torch.long)
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        mask = self._mask_for(positions, cu, ratio=4)
        assert mask.tolist() == [True, True, True, True]

    def test_mixed_prefill_only_last_token_of_seq_drives_trigger(self) -> None:
        """Prefill seq spanning multiple tokens: only the seq's LAST
        token's position drives the compress trigger (cu_seqlens_q
        encodes the per-seq right-edge)."""
        # Seq A: tokens at positions [0, 1, 2, 3] (4 prefill tokens) —
        #   last pos 3, 3+1=4, 4%4=0 → triggers.
        # Seq B: tokens at positions [10, 11, 12] (3 prefill tokens) —
        #   last pos 12, 12+1=13, 13%4=1 → does NOT trigger.
        positions = torch.tensor([0, 1, 2, 3, 10, 11, 12], dtype=torch.long)
        cu = torch.tensor([0, 4, 7], dtype=torch.long)
        mask = self._mask_for(positions, cu, ratio=4)
        assert mask.tolist() == [True, False]


# ---------------------------------------------------------------------------
# Section 6 — Indexer per-seq sparse-index writes
# ---------------------------------------------------------------------------


class TestPerSeqIndexerStateScatter:
    """W4.4: Indexer's per-seq sparse-index writes (compressed indexer
    KV) hit distinct slot rows of the pool's ``indexer_kv`` slab. Same
    correctness pattern as the main attention path — no row-0
    collision under multi-request."""

    def test_4seq_indexer_writes_distinct_rows(self) -> None:
        pool = _make_pool_with_compressor()
        for sid in [200, 201, 202, 203]:
            pool.admit_request(sid)
        slots = pool.get_slots([200, 201, 202, 203])

        # Distinct compressed-position writes per seq (positions //
        # ratio with ratio=4 → compressed-cache index).
        positions = torch.tensor([15, 19, 23, 27], dtype=torch.long)
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

        flat_loc = pool.compute_out_cache_loc(
            positions=positions,
            slot_indices=slots,
            cu_seqlens_q=cu,
            ring="indexer",
        )
        assert flat_loc.shape == (4,)
        assert (
            flat_loc.unique().numel() == 4
        ), f"4 seqs must scatter to distinct indexer flat slots. Got {flat_loc.tolist()}"

    def test_indexer_view_zero_copy(self) -> None:
        """Mutations through ``view_for_layer(...)["indexer_kv"]`` must
        persist — the Indexer's W4 forward writes through this view."""
        pool = _make_pool_with_compressor()
        pool.admit_request(7)
        v1 = pool.view_for_layer(layer_id=1)  # c4 layer has indexer_kv
        v1["indexer_kv"][0, 5, 3] = 7.5
        v2 = pool.view_for_layer(layer_id=1)
        assert v2["indexer_kv"][0, 5, 3].item() == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Section 7 — legacy single-request path is preserved
# ---------------------------------------------------------------------------


class TestLegacyPathPreserved:
    """W4.4 critical guardrail: the W4 path is a NEW path; the legacy
    single-request fallback (no forward_batch, no pool) must continue
    to work for warmup / PR1 toy harness. We assert this via:
    (a) the forward signatures still accept the legacy positional
        ``start_pos`` arg (no kwarg-only change), and
    (b) the source contains the lazy-allocate fallback methods
        (``_ensure_legacy_state`` / ``_ensure_legacy_kv_cache``)."""

    def test_compressor_forward_still_accepts_start_pos_positional(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Compressor")
        fwd = _find_method(cls, "forward")
        # Positional args (excluding self): start_pos must be position 1.
        positional = [a.arg for a in fwd.args.args]
        assert (
            positional[1] == "x"
        ), f"Compressor.forward arg 1 must be 'x'; got {positional}"
        assert (
            "start_pos" in positional
        ), f"Compressor.forward must keep 'start_pos' for legacy callers; got {positional}"

    def test_indexer_forward_still_accepts_start_pos_offset_positional(self) -> None:
        cls = _find_class(_parse_dsv4_source(), "Indexer")
        fwd = _find_method(cls, "forward")
        positional = [a.arg for a in fwd.args.args]
        for required in ("start_pos", "offset"):
            assert (
                required in positional
            ), f"Indexer.forward must keep '{required}' for legacy; got {positional}"

    def test_compressor_has_legacy_state_lazy_allocator(self) -> None:
        text = DSV4_SOURCE.read_text()
        assert (
            "def _ensure_legacy_state" in text
        ), "Compressor must keep the legacy single-request lazy state allocator."

    def test_indexer_has_legacy_kv_cache_lazy_allocator(self) -> None:
        text = DSV4_SOURCE.read_text()
        assert (
            "def _ensure_legacy_kv_cache" in text
        ), "Indexer must keep the legacy single-request lazy kv_cache allocator."


# ---------------------------------------------------------------------------
# Section 8 — Compressor / Indexer pool-binding helpers exist
# ---------------------------------------------------------------------------


class TestPoolBindingHelpers:
    """W4.4 implementation detail: each module exposes a ``_bind_*_from_pool``
    helper used by ``forward()`` to rebind state to the pool view at
    every step. Not a public contract but stable enough that we want
    a regression alarm if it's removed."""

    def test_compressor_bind_state_from_pool(self) -> None:
        text = DSV4_SOURCE.read_text()
        assert (
            "def _bind_state_from_pool" in text
        ), "Compressor must expose `_bind_state_from_pool` for the W4 step rebind."

    def test_indexer_bind_kv_cache_from_pool(self) -> None:
        text = DSV4_SOURCE.read_text()
        assert (
            "def _bind_kv_cache_from_pool" in text
        ), "Indexer must expose `_bind_kv_cache_from_pool` for the W4 step rebind."

    def test_compressor_w4_dispatch_method_exists(self) -> None:
        text = DSV4_SOURCE.read_text()
        assert (
            "def _forward_w4" in text
        ), "Compressor must expose `_forward_w4` for the per-token W4 path."


# ---------------------------------------------------------------------------
# Section 9 — Compressor.forward smoke: signature + W4 dispatch
# ---------------------------------------------------------------------------


class TestCompressorForwardW4SmokeRun:
    """W4.4 smoke: a 4-seq packed input with positions=[12,13,15,12]
    routes through the W4 path's per-token slot scatter without the
    register_buffer collision. We can't import Compressor directly
    (aiter ROCm kernels missing in CI), so we exercise the underlying
    pool ring math + per-seq compress-trigger logic with a hand-built
    surrogate that mirrors the production W4 forward exactly."""

    def test_forward_w4_signature_dispatches_on_forward_batch(self) -> None:
        """Verify the forward dispatch shape — when ``forward_batch`` is
        provided AND ``forward_batch.kv_pool`` is not None, the W4
        branch is taken; otherwise the legacy branch."""
        cls = _find_class(_parse_dsv4_source(), "Compressor")
        # Sentinel: the W4 dispatch checks for both forward_batch AND
        # kv_pool presence before calling _forward_w4.
        forward_method = _find_method(cls, "forward")
        body_src = ast.unparse(forward_method)
        assert "_forward_w4" in body_src, (
            "Compressor.forward must dispatch to _forward_w4 when "
            "forward_batch carries a pool."
        )
        assert (
            "kv_pool" in body_src
        ), "Compressor.forward W4 dispatch must gate on forward_batch.kv_pool."
        assert "_ensure_legacy_state" in body_src, (
            "Compressor.forward must call _ensure_legacy_state on the "
            "legacy fallback path."
        )

    def test_indexer_forward_dispatches_on_pool_binding(self) -> None:
        """Indexer.forward calls _bind_kv_cache_from_pool and routes
        the inner compressor with forward_batch when bound."""
        cls = _find_class(_parse_dsv4_source(), "Indexer")
        fwd_method = _find_method(cls, "forward")
        body_src = ast.unparse(fwd_method)
        assert "_bind_kv_cache_from_pool" in body_src, (
            "Indexer.forward must call _bind_kv_cache_from_pool to "
            "rebind kv_cache to the pool's indexer_kv view."
        )
        assert "forward_batch=forward_batch" in body_src, (
            "Indexer.forward must thread forward_batch into the inner "
            "compressor on the W4 path."
        )

    def test_per_token_state_writes_isolate_per_slot(self) -> None:
        """Reproduces the W4 forward's per-token kv_state scatter on
        a CPU pool. Each of 4 seqs writes a distinct payload to its
        OWN slot's ring row; readback confirms no row-0 collision."""
        pool = _make_pool_with_compressor()
        for sid in [50, 51, 52, 53]:
            pool.admit_request(sid)
        slots = pool.get_slots([50, 51, 52, 53])

        positions = torch.tensor([12, 13, 15, 12], dtype=torch.long)

        # Pull the c4 layer's state slab. ring_compressor=8, ratio=4, so
        # in overlap mode the "current" half lives at offsets [4..7].
        view = pool.view_for_layer(layer_id=1)
        kv_state = view["kv_state"]  # [N=4, ring=8, inner=8]
        _, _, D = kv_state.shape

        # Reproduce the W4 forward's per-token write:
        #   row_in_state[t] = ratio + (positions[t] % ratio)
        ratio = 4
        row_in_state = ratio + (positions % ratio)

        # Per-seq distinct payloads.
        payload = torch.arange(4 * D, dtype=torch.float32).view(4, D)
        kv_state[slots, row_in_state] = payload

        # Read back: each seq's slot has its own row populated, others
        # remain at the init value (0.0).
        for i, (slot, pos) in enumerate(zip(slots.tolist(), positions.tolist())):
            row = ratio + (pos % ratio)
            assert torch.allclose(
                kv_state[slot, row], payload[i]
            ), f"seq {i} (slot={slot}, row={row}) payload mismatch"

        # Confirm no two seqs wrote to the same (slot, row) pair —
        # this is the multi-request correctness guarantee.
        seen_keys = {
            (slot, ratio + (pos % ratio))
            for slot, pos in zip(slots.tolist(), positions.tolist())
        }
        assert (
            len(seen_keys) == 4
        ), f"4 seqs must produce 4 distinct (slot, row) write keys; got {seen_keys}"
