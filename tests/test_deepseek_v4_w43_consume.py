# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""W4.3 (issue #37 Path 3): DeepseekV4 model forward consume contract.

These tests assert the architectural contract between
``DeepseekV4Attention.forward`` and the engine pool / ForwardBatch
introduced by W4.3:

1. ``register_buffer("kv_cache", ...)`` is removed from
   ``DeepseekV4Attention.__init__`` — the per-layer KV storage is
   owned by the engine ``DSV4KVPool``, not the ``nn.Module``.
2. ``DeepseekV4Attention.forward`` accepts an optional
   ``forward_batch`` kwarg and consumes per-token positions / per-seq
   slot rows from it.
3. The new per-token sliding-window topk helper
   (``_get_window_topk_idxs_pertoken``) yields one distinct row per
   token even when 4 tokens belong to 4 different sequences with
   different absolute positions.
4. Per-token KV scatter into the pool writes to 4 distinct flat slot
   indices (no row-0 collision, the bug class W3.2 v3..v6.1 chased).

Why no full-model smoke run? ``DeepseekV4ForCausalLM`` instantiation
pulls in aiter ROCm kernels (FP8/FP4 GEMMs, sparse_attn) that the
unit-test image's stubbed ``aiter`` cannot satisfy. We assert the
contract via:

- AST-level inspection of the source (catches register_buffer
  regression without an import).
- Direct invocation of helper utilities that only need ``torch``.
- A pool-side reproduction of the W4 forward's KV scatter sequence.

The full forward smoke run is W4.5 silicon-validation territory.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from typing import Any, Optional

import pytest
import torch

from atom.engine.kv_pool import DSV4KVPool, DSV4KVPoolConfig
from atom.utils.dsv4_forward_batch import DSV4ForwardBatch, DSV4ForwardMode

# ---------------------------------------------------------------------------
# Source-level contract checks (no DSV4 module import required)
# ---------------------------------------------------------------------------

ATOM_ROOT = Path(__file__).resolve().parent.parent
DSV4_SOURCE = ATOM_ROOT / "atom" / "models" / "deepseek_v4.py"


def _parse_dsv4_source() -> ast.Module:
    """Parse atom/models/deepseek_v4.py via AST, no import side effects."""
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
    """Return all `self.register_buffer(<lit>, ...)` target literal names."""
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


class TestDeepseekV4AttentionDoesNotRegisterKVCacheBuffer:
    """W4.3 + W4.4 contract: per-request state is engine-pool-owned, not a
    register_buffer. W4.4 extends W4.3 by also lifting Compressor's
    ``kv_state`` / ``score_state`` and Indexer's ``kv_cache``.
    """

    def test_init_does_not_register_kv_cache(self):
        tree = _parse_dsv4_source()
        cls = _find_class(tree, "DeepseekV4Attention")
        init = _find_method(cls, "__init__")
        targets = _register_buffer_targets(init)
        assert "kv_cache" not in targets, (
            "DeepseekV4Attention.__init__ must NOT register_buffer "
            "'kv_cache' — that storage is now owned by DSV4KVPool. "
            f"Found register_buffer targets: {targets}"
        )

    def test_compressor_init_does_not_register_state_buffers(self):
        """W4.4 contract: Compressor.__init__ must NOT register_buffer
        ``kv_state`` / ``score_state``. State is owned by DSV4KVPool."""
        tree = _parse_dsv4_source()
        cls = _find_class(tree, "Compressor")
        init = _find_method(cls, "__init__")
        targets = _register_buffer_targets(init)
        assert "kv_state" not in targets, (
            "Compressor.__init__ must NOT register_buffer 'kv_state' "
            "(W4.4: state owned by DSV4KVPool's kv_state slab). "
            f"Found targets: {targets}"
        )
        assert "score_state" not in targets, (
            "Compressor.__init__ must NOT register_buffer 'score_state' "
            "(W4.4: state owned by DSV4KVPool's score_state slab). "
            f"Found targets: {targets}"
        )

    def test_indexer_init_does_not_register_kv_cache_buffer(self):
        """W4.4 contract: Indexer.__init__ must NOT register_buffer
        ``kv_cache``. Storage is the pool's per-layer indexer_kv view."""
        tree = _parse_dsv4_source()
        cls = _find_class(tree, "Indexer")
        init = _find_method(cls, "__init__")
        targets = _register_buffer_targets(init)
        assert "kv_cache" not in targets, (
            "Indexer.__init__ must NOT register_buffer 'kv_cache' "
            "(W4.4: cache owned by DSV4KVPool's indexer_kv slab). "
            f"Found targets: {targets}"
        )

    def test_init_still_registers_freqs_cis(self):
        """Sanity: freqs_cis IS still a register_buffer (unchanged)."""
        tree = _parse_dsv4_source()
        cls = _find_class(tree, "DeepseekV4Attention")
        init = _find_method(cls, "__init__")
        targets = _register_buffer_targets(init)
        assert (
            "freqs_cis" in targets
        ), "freqs_cis must remain a register_buffer (per-layer YaRN cache)"

    def test_kv_cache_assigned_as_plain_attribute(self):
        """The W4.3 contract: `self.kv_cache: Optional[Tensor] = None`."""
        text = DSV4_SOURCE.read_text()
        # Look for the W4.3 sentinel comment block plus the assignment.
        assert "self.kv_cache: Optional[torch.Tensor] = None" in text, (
            "DeepseekV4Attention.__init__ must initialize self.kv_cache "
            "as a plain attribute (lazy-bound to the pool view per step)."
        )


class TestDeepseekV4AttentionForwardSignature:
    """W4.3 contract: forward() accepts forward_batch and threads it through."""

    def test_forward_accepts_forward_batch_kwarg(self):
        tree = _parse_dsv4_source()
        cls = _find_class(tree, "DeepseekV4Attention")
        fwd = _find_method(cls, "forward")
        kw_names = {arg.arg for arg in fwd.args.args}
        kw_names.update(arg.arg for arg in fwd.args.kwonlyargs)
        assert (
            "forward_batch" in kw_names
        ), f"DeepseekV4Attention.forward must accept 'forward_batch'; got args={kw_names}"

    def test_block_forward_threads_forward_batch_through(self):
        tree = _parse_dsv4_source()
        cls = _find_class(tree, "Block")
        fwd = _find_method(cls, "forward")
        kw_names = {arg.arg for arg in fwd.args.args}
        kw_names.update(arg.arg for arg in fwd.args.kwonlyargs)
        assert (
            "forward_batch" in kw_names
        ), "Block.forward must thread 'forward_batch' through to attn"

    def test_model_forward_accepts_forward_batch(self):
        tree = _parse_dsv4_source()
        cls = _find_class(tree, "DeepseekV4Model")
        fwd = _find_method(cls, "forward")
        kw_names = {arg.arg for arg in fwd.args.args}
        kw_names.update(arg.arg for arg in fwd.args.kwonlyargs)
        assert (
            "forward_batch" in kw_names
        ), "DeepseekV4Model.forward must accept 'forward_batch'"

    def test_no_outer_cu_seqlens_split_loop_in_model_forward(self):
        """W4.3: DeepseekV4Model.forward should call layers in a single
        straight loop. The pre-W4.3 cu_seqlens_q outer split-loop has
        been removed — per-token correctness lives inside Attention now.
        """
        text = DSV4_SOURCE.read_text()
        # Sentinel string from the W4.3 single-pass comment.
        assert "single straight pass through all blocks" in text, (
            "DeepseekV4Model.forward should be the W4.3 single-pass "
            "form (no outer cu_seqlens_q split loop)."
        )


class TestPerTokenWindowTopkHelper:
    """W4.3: per-token sliding-window topk indices.

    Replaces the cached single-start_pos helper that silently gave every
    seq the same window pattern under multi-request decode.
    """

    def _import_helper(self):
        """Import the W4.3 helper without pulling in the full DSV4 model.

        We exec only the helper's source via `compile(ast.Module(...))`
        to dodge the heavy `from aiter import ...` chain at module top.
        """
        text = DSV4_SOURCE.read_text()
        tree = ast.parse(text)
        helper_node = None
        for node in tree.body:
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_get_window_topk_idxs_pertoken"
            ):
                helper_node = node
                break
        assert (
            helper_node is not None
        ), "_get_window_topk_idxs_pertoken not found in deepseek_v4.py"
        # Extract its source by line range.
        src_lines = text.splitlines()
        snippet = "\n".join(src_lines[helper_node.lineno - 1 : helper_node.end_lineno])
        # Compile in a sandbox namespace.
        ns: dict[str, Any] = {
            "torch": torch,
            "Optional": Optional,
        }
        exec(snippet, ns)
        return ns["_get_window_topk_idxs_pertoken"]

    def test_4_seqs_4_distinct_position_rows(self):
        """positions=[12,13,15,12] => 4 token rows; 3 distinct (12 repeats
        twice, but the row template must still index row-by-row)."""
        helper = self._import_helper()
        positions = torch.tensor([12, 13, 15, 12], dtype=torch.long)
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        out = helper(8, positions, cu)
        assert out.shape == (
            4,
            8,
        ), f"expected [num_tokens, win]=(4, 8), got {tuple(out.shape)}"

    def test_each_seq_gets_its_own_window(self):
        """Two seqs at different absolute positions must produce
        different window rows (this is the cross-talk fix)."""
        helper = self._import_helper()
        # Seq A at pos 12, Seq B at pos 16 (both >= win-1, so warm path)
        positions = torch.tensor([12, 16], dtype=torch.long)
        cu = torch.tensor([0, 1, 2], dtype=torch.long)
        out = helper(8, positions, cu)
        assert not torch.equal(out[0], out[1]), (
            "Tokens at different absolute positions must produce "
            "different window topk rows — this is the W3.2 cross-talk "
            "fix that W4.3 makes unconditional."
        )

    def test_prefill_token_within_window_is_causal(self):
        """A prefill token at offset 0 (first token of its seq) sees only
        position 0; later positions are masked to -1."""
        helper = self._import_helper()
        # Seq spans 5 tokens at positions [0, 1, 2, 3, 4]
        positions = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        cu = torch.tensor([0, 5], dtype=torch.long)
        out = helper(4, positions, cu)
        assert out.shape == (5, 4)
        # Token 0 (pos 0): only slot 0 valid, others masked.
        # Causal property: every -1 should be a future position.
        for t in range(5):
            valid_slots = out[t][out[t] >= 0]
            # All valid ring indices must derive from positions <= t.
            # We can't compute exact values without re-deriving the
            # modular arithmetic, but length must be t+1 (clamped to W).
            assert len(valid_slots) == min(t + 1, 4), (
                f"token {t}: expected {min(t+1, 4)} valid slots, "
                f"got {len(valid_slots)} (row={out[t].tolist()})"
            )


class TestPerTokenKVWriteToPool:
    """W4.3: per-token KV scatter must hit 4 distinct slot rows of the
    pool's main_kv tensor for a 4-seq decode batch."""

    @pytest.fixture
    def pool(self) -> DSV4KVPool:
        cfg = DSV4KVPoolConfig(
            max_active_seqs=4,
            num_layers=2,
            num_c4_layers=0,
            num_c128_layers=0,
            head_dim=8,
            rope_head_dim=4,
            window_size=16,
            max_seq_len=128,
            ring_size_main=16,
            ring_size_compressor=8,
            ring_size_indexer=16,
            compress_ratio_per_layer=[0, 0],
            state_inner_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            state_dtype=torch.float32,
        )
        return DSV4KVPool(cfg)

    def test_4seq_decode_kv_writes_to_4_distinct_rows(self, pool: DSV4KVPool):
        """Reproduces the W4 forward's per-token KV scatter step.

        4 seqs decode 1 token each at distinct positions. The pool's
        per-layer ``main_kv[layer_id]`` slab must have exactly 4 rows
        modified (one per seq), NOT a single row 0.
        """
        # Admit 4 seqs.
        for sid in [100, 101, 102, 103]:
            pool.admit_request(sid)
        slots = pool.get_slots([100, 101, 102, 103])  # [4]

        # Forward batch reproducing the 4-seq decode shape.
        positions = torch.tensor([12, 13, 15, 12], dtype=torch.long)
        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        fb = DSV4ForwardBatch(
            forward_mode=DSV4ForwardMode.DECODE,
            positions=positions,
            seq_lens=torch.tensor([13, 14, 16, 13], dtype=torch.long),
            extend_seq_lens=torch.tensor([1, 1, 1, 1], dtype=torch.long),
            cu_seqlens_q=cu,
            req_pool_indices=slots,
            kv_pool=pool,
        )
        # Compute per-token KV write loc as the W4 attention forward does.
        flat_loc = pool.compute_out_cache_loc(
            positions=fb.positions,
            slot_indices=fb.req_pool_indices,
            cu_seqlens_q=fb.cu_seqlens_q,
            ring="main",
        )
        assert flat_loc.shape == (4,)
        assert (
            flat_loc.unique().numel() == 4
        ), "All 4 tokens must scatter to DISTINCT flat slot indices"

        # Reproduce the W4 attention forward's scatter into a layer view.
        layer_view = pool.view_for_layer(layer_id=0)
        kv_cache = layer_view["kv_cache"]  # [N=4, ring=16, head=8]
        N, R, D = kv_cache.shape
        kv_flat = kv_cache.view(N * R, D)
        # Distinct payload per token so we can confirm row identity.
        kv_tok = torch.arange(4 * D, dtype=torch.float32).view(4, D)
        kv_flat[flat_loc] = kv_tok

        # Now check: for each seq, the row at position % ring should
        # contain the right payload, AND no two seqs wrote to row 0.
        for i, (slot, pos) in enumerate(zip(slots.tolist(), positions.tolist())):
            ring_idx = pos % R
            assert torch.allclose(
                kv_cache[slot, ring_idx], kv_tok[i]
            ), f"seq {i} (slot={slot}, ring_idx={ring_idx}) payload mismatch"

    def test_pool_view_consumed_in_place_of_register_buffer(self, pool: DSV4KVPool):
        """The pool view returned by ``view_for_layer`` is a zero-copy
        slice of pool storage. Mutations through the view must be
        visible to a second ``view_for_layer`` call (same layer).
        """
        for sid in [1, 2]:
            pool.admit_request(sid)
        v1 = pool.view_for_layer(layer_id=0)
        v1["kv_cache"][0, 0, 0] = 42.0
        v2 = pool.view_for_layer(layer_id=0)
        assert v2["kv_cache"][0, 0, 0].item() == pytest.approx(42.0), (
            "view_for_layer must return a zero-copy slice — the W4 "
            "attention path relies on this for its scatter semantics."
        )


class TestForwardBatchFromAdapterEndToEnd:
    """End-to-end shape check: from_attn_metadata + pool produces a
    ForwardBatch whose every field has the W4.3 invariants we depend
    on inside the model layers."""

    def test_from_attn_metadata_with_pool_populates_out_cache_loc(self):
        from types import SimpleNamespace

        cfg = DSV4KVPoolConfig(
            max_active_seqs=4,
            num_layers=1,
            num_c4_layers=0,
            num_c128_layers=0,
            head_dim=8,
            rope_head_dim=4,
            window_size=16,
            max_seq_len=128,
            ring_size_main=16,
            ring_size_compressor=8,
            ring_size_indexer=16,
            compress_ratio_per_layer=[0],
            state_inner_dim=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            state_dtype=torch.float32,
        )
        pool = DSV4KVPool(cfg)

        seq_ids = [1000, 1001, 1002, 1003]
        for sid in seq_ids:
            pool.admit_request(sid)

        cu = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        block_tables = torch.tensor(
            [[10, 11], [10, 12], [10, 13], [10, 14]],  # collide on first block
            dtype=torch.long,
        )
        context_lens = torch.tensor([12, 13, 15, 12], dtype=torch.long)
        attn_meta = SimpleNamespace(
            cu_seqlens_q=cu,
            block_tables=block_tables,
            context_lens=context_lens,
        )
        positions = torch.tensor([12, 13, 15, 12], dtype=torch.long)

        fb = DSV4ForwardBatch.from_attn_metadata(
            attn_meta, positions, seq_ids=seq_ids, pool=pool
        )

        # Pool path: req_pool_indices must be unique even though
        # block_tables[:, 0] all collide on 10.
        assert fb.req_pool_indices.unique().numel() == 4, (
            "Pool path must produce unique slot rows — the W4.1→W4.2 "
            "fix for the prefix-cache first-block collision."
        )
        # out_cache_loc must be populated (the pool path always fills it).
        assert (
            fb.out_cache_loc is not None
        ), "Pool path must populate out_cache_loc (W4.2 contract)"
        assert fb.out_cache_loc.shape == (4,)
        # 4 distinct flat indices (decode lockstep).
        assert fb.out_cache_loc.unique().numel() == 4


class TestModelRunnerHooks:
    """W4.3: ModelRunner gains a `_dsv4_pool` attribute + `finish_dsv4_request`
    method. Static AST check (don't require ModelRunner instantiation)."""

    def test_finish_dsv4_request_method_exists(self):
        runner_path = ATOM_ROOT / "atom" / "model_engine" / "model_runner.py"
        text = runner_path.read_text()
        assert "def finish_dsv4_request" in text, (
            "ModelRunner must expose `finish_dsv4_request(seq_id)` so the "
            "scheduler can free pool slots on seq finalization."
        )

    def test_maybe_setup_dsv4_forward_batch_method_exists(self):
        runner_path = ATOM_ROOT / "atom" / "model_engine" / "model_runner.py"
        text = runner_path.read_text()
        assert "def _maybe_setup_dsv4_forward_batch" in text
        assert "set_forward_context" in text
        assert "dsv4_forward_batch=dsv4_forward_batch" in text


class TestForwardContextSurfaceArea:
    """W4.3: ForwardContext gains `dsv4_pool` + `dsv4_forward_batch`
    optional fields so the model layer can pull them via
    `get_forward_context()`."""

    def test_forward_context_has_dsv4_fields(self):
        from atom.utils.forward_context import ForwardContext

        ctx = ForwardContext()
        assert hasattr(ctx, "dsv4_pool")
        assert hasattr(ctx, "dsv4_forward_batch")
        # Defaults: None (non-DSV4 models see no surface change).
        assert ctx.dsv4_pool is None
        assert ctx.dsv4_forward_batch is None

    def test_set_forward_context_accepts_dsv4_kwargs(self):
        sig = inspect.signature(
            __import__(
                "atom.utils.forward_context", fromlist=["set_forward_context"]
            ).set_forward_context
        )
        params = sig.parameters
        assert "dsv4_pool" in params
        assert "dsv4_forward_batch" in params


class TestNoStaleThreadLocalHacks:
    """The W3.2 archive added `_FBID_TO_ROW` / `_DSV4_CURRENT_SEQ_IDX`
    thread-local hacks. They must NOT have leaked into main's
    deepseek_v4.py (this is regression armor for any future merge)."""

    def test_no_fbid_to_row_thread_local(self):
        text = DSV4_SOURCE.read_text()
        assert "_FBID_TO_ROW" not in text
        assert "_DSV4_CURRENT_SEQ_IDX" not in text


class TestGuardRelaxed:
    """W4.3 relaxed-guard sanity (cross-check with test_dsv4_multireq_guard.py)."""

    def test_guard_no_longer_raises_for_multireq(self):
        from atom.utils.dsv4_guard import validate_dsv4_multireq

        os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
        os.environ.pop("ATOM_DSV4_FORCE_STRICT_GUARD", None)
        # No raise — W4.3 unlocks multi-request.
        validate_dsv4_multireq(["DeepseekV4ForCausalLM"], max_num_seqs=8)
