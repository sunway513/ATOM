# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""W4.3-redo: DeepseekV4Attention consumes DSV4ForwardBatch + KVPool (issue sunway513/atom#37).

Validates spec §3 'ATOM W4.3-redo (main attention)' component:

- ``DeepseekV4Attention.forward`` accepts an optional ``forward_batch``
  kwarg.
- ``ATOM_DSV4_USE_W4_PATH=1`` + non-None ``forward_batch`` enters the
  per-token RoPE / per-seq slot KV scatter path (``_forward_w4``).
- ``ATOM_DSV4_USE_W4_PATH=0`` (or ``forward_batch is None``) keeps the
  legacy single-request register_buffer path (``_forward_legacy``,
  bit-for-bit preserved post-#44).
- ``_forward_w4`` calls the AITER metadata validator
  (``dsv4_validate_sparse_attn_metadata``) immediately before
  ``sparse_attn`` when ``ATOM_AITER_VALIDATE=1``.
- ``_forward_w4`` uses per-token RoPE indexing
  (``freqs_cis[positions]``) rather than the legacy
  ``freqs_cis[start_pos:start_pos+seqlen]`` contiguous slice (the
  W3.2 v3..v6.1 bug class).
- ``register_buffer("kv_cache", ...)`` is preserved in ``__init__`` for
  the legacy fallback (deliberate, per Task 10 spec §10.6).

We use AST-level inspection so the test runs on CPU without HF
download or GPU. ``DeepseekV4ForCausalLM`` instantiation pulls in
aiter ROCm kernels that the unit-test image's stubbed ``aiter``
cannot satisfy. The full forward smoke run is silicon-validation
territory (Task 14).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# AST helpers — read the source without importing the module
# ---------------------------------------------------------------------------

ATOM_ROOT = Path(__file__).resolve().parent.parent
DSV4_SOURCE = ATOM_ROOT / "atom" / "models" / "deepseek_v4.py"


def _tree() -> ast.Module:
    return ast.parse(DSV4_SOURCE.read_text())


# ---------------------------------------------------------------------------
# Numerical helpers — extracted from source via AST exec (no aiter import needed)
# ---------------------------------------------------------------------------


def _load_topk_helpers():
    """Exec just the two topk helpers from deepseek_v4.py source.

    Avoids full-module import (which requires ROCm/aiter) by compiling
    only the two function definitions.  Works in both GPU and CPU-only
    test environments.
    """
    import torch
    import torch.nn.functional as F
    from functools import lru_cache
    from typing import Optional

    fn_names = {"_get_window_topk_idxs", "_get_window_topk_idxs_pertoken"}
    nodes = [
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name in fn_names
    ]
    if len(nodes) < 2:
        return None, None
    module = ast.Module(body=nodes, type_ignores=[])
    code = compile(module, str(DSV4_SOURCE), "exec")
    ns: dict = {"torch": torch, "F": F, "lru_cache": lru_cache, "Optional": Optional}
    exec(code, ns)  # noqa: S102
    return ns.get("_get_window_topk_idxs"), ns.get("_get_window_topk_idxs_pertoken")


try:
    _get_window_topk_idxs, _get_window_topk_idxs_pertoken = _load_topk_helpers()
    _HELPERS_AVAILABLE = (
        _get_window_topk_idxs is not None and _get_window_topk_idxs_pertoken is not None
    )
except Exception:
    _get_window_topk_idxs = None
    _get_window_topk_idxs_pertoken = None
    _HELPERS_AVAILABLE = False

_skip_if_no_helpers = pytest.mark.skipif(
    not _HELPERS_AVAILABLE,
    reason="topk helpers not extractable from deepseek_v4.py source",
)


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found in deepseek_v4.py")


def _find_method(cls: ast.ClassDef, name: str):
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def _method_src(cls: ast.ClassDef, name: str) -> str:
    m = _find_method(cls, name)
    if m is None:
        return ""
    return ast.unparse(m)


def _method_params(cls: ast.ClassDef, name: str) -> list[str]:
    m = _find_method(cls, name)
    if m is None:
        return []
    args = m.args
    out: list[str] = [a.arg for a in args.args]
    out += [a.arg for a in args.kwonlyargs]
    if args.vararg is not None:
        out.append("*" + args.vararg.arg)
    if args.kwarg is not None:
        out.append("**" + args.kwarg.arg)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestForwardSignature:
    def test_forward_accepts_forward_batch_kwarg(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        params = _method_params(cls, "forward")
        assert "forward_batch" in params, (
            "DeepseekV4Attention.forward must accept a forward_batch kwarg; "
            f"got params {params}"
        )

    def test_block_forward_accepts_forward_batch_kwarg(self):
        cls = _find_class(_tree(), "Block")
        params = _method_params(cls, "forward")
        assert (
            "forward_batch" in params
        ), f"Block.forward must accept a forward_batch kwarg; got params {params}"

    def test_model_forward_accepts_forward_batch_kwarg(self):
        cls = _find_class(_tree(), "DeepseekV4Model")
        params = _method_params(cls, "forward")
        # Either named arg or **kwargs that the body propagates.
        has_forward_batch = "forward_batch" in params
        # Inspect body for a forward_batch=... pull-through if not named.
        src = _method_src(cls, "forward")
        propagates_to_layer = "forward_batch=forward_batch" in src
        assert has_forward_batch or propagates_to_layer, (
            "DeepseekV4Model.forward must accept and propagate forward_batch; "
            f"params={params}"
        )


class TestDispatch:
    def test_attention_has_legacy_method(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        assert (
            _find_method(cls, "_forward_legacy") is not None
        ), "DeepseekV4Attention must expose _forward_legacy"

    def test_attention_has_w4_method(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        assert (
            _find_method(cls, "_forward_w4") is not None
        ), "DeepseekV4Attention must expose _forward_w4"

    def test_forward_dispatches_on_flag_and_forward_batch(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "forward")
        # Flag check
        assert (
            "ATOM_DSV4_USE_W4_PATH" in src
        ), "forward() must consult ATOM_DSV4_USE_W4_PATH to dispatch to W4 path"
        # Both branches dispatched
        assert "_forward_legacy" in src, "forward() must dispatch to _forward_legacy"
        assert "_forward_w4" in src, "forward() must dispatch to _forward_w4"


class TestPerTokenRoPE:
    def test_w4_path_uses_per_token_rope(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "_forward_w4")
        assert src, "_forward_w4 must exist"
        per_token_markers = (
            "freqs_cis[positions]",
            "freqs_cis[forward_batch.positions]",
        )
        assert any(m in src for m in per_token_markers), (
            "_forward_w4 must use per-token RoPE indexing "
            "(freqs_cis[positions] or freqs_cis[forward_batch.positions])"
        )

    def test_w4_path_does_not_use_legacy_contiguous_slice(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "_forward_w4")
        # any of these patterns indicates the legacy single-start_pos slice
        forbidden = (
            "start_pos : start_pos + seqlen",
            "start_pos:start_pos + seqlen",
            "start_pos:start_pos+seqlen",
        )
        for f in forbidden:
            assert f not in src, (
                "_forward_w4 must not use the legacy contiguous freqs_cis slice "
                f"(found {f!r})"
            )


class TestKVScatter:
    def test_w4_path_uses_compute_out_cache_loc(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "_forward_w4")
        assert (
            "compute_out_cache_loc" in src or "out_cache_loc" in src
        ), "_forward_w4 must compute or use the per-token out_cache_loc"

    def test_w4_path_uses_pool_view_for_layer(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "_forward_w4")
        assert "view_for_layer" in src or "kv_pool" in src, (
            "_forward_w4 must consume forward_batch.kv_pool / view_for_layer "
            "instead of the legacy register_buffer kv_cache"
        )


class TestValidatorIntegration:
    def test_w4_path_calls_validator(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "_forward_w4")
        assert (
            "dsv4_validate_sparse_attn_metadata" in src
        ), "_forward_w4 must call dsv4_validate_sparse_attn_metadata"
        assert (
            "ATOM_AITER_VALIDATE" in src
        ), "_forward_w4 must gate the validator on ATOM_AITER_VALIDATE"

    def test_validator_called_before_sparse_attn(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "_forward_w4")
        i_validator = src.find("dsv4_validate_sparse_attn_metadata")
        i_sparse_attn = src.find("sparse_attn(")
        assert i_validator != -1, "validator missing"
        assert i_sparse_attn != -1, "sparse_attn call missing"
        assert i_validator < i_sparse_attn, (
            "Validator must be invoked before sparse_attn so OOB indices "
            "fail with a host-side ValueError, not a GPU HSA crash"
        )


class TestPerTokenTopkHelper:
    def test_helper_method_or_function_exists(self):
        tree = _tree()
        # method on the class
        cls = _find_class(tree, "DeepseekV4Attention")
        has_method = _find_method(cls, "_build_topk_per_token") is not None
        # or module-level function
        has_module_helper = False
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in (
                "_build_topk_per_token",
                "_get_window_topk_idxs_pertoken",
            ):
                has_module_helper = True
                break
        assert (
            has_method or has_module_helper
        ), "Per-token topk helper (method or module-level fn) must exist"

    @_skip_if_no_helpers
    def test_pertoken_decode_matches_legacy_single_seq(self):
        """Warm decode rows must exactly match legacy; cold rows must cover the same valid-ring set."""
        import torch

        W = 64

        def valid_set(t):
            return set(t[t != -1].tolist())

        # Warm positions (p >= W-1): ring is full, ordering must be bit-identical.
        for p in [63, 64, 127, 200]:
            pos_t = torch.tensor([p], dtype=torch.long)
            cu = torch.tensor([0, 1], dtype=torch.long)
            pt = _get_window_topk_idxs_pertoken(W, pos_t, cu)  # [1, W]
            lg = _get_window_topk_idxs(W, 1, 1, p)  # [1, 1, W]
            assert torch.equal(
                pt[0], lg[0, 0]
            ), f"p={p}: pertoken {pt[0].tolist()} != legacy {lg[0,0].tolist()}"

        # Cold positions (0 < p < W-1): same valid-ring-index set but different column order.
        for p in [1, 32]:
            pos_t = torch.tensor([p], dtype=torch.long)
            cu = torch.tensor([0, 1], dtype=torch.long)
            pt = _get_window_topk_idxs_pertoken(W, pos_t, cu)  # [1, W]
            lg = _get_window_topk_idxs(W, 1, 1, p)  # [1, 1, W]
            assert pt.shape == (1, W)
            assert valid_set(pt[0]) == valid_set(lg[0, 0]), (
                f"p={p}: valid ring-index sets differ: "
                f"pertoken={sorted(valid_set(pt[0]))} legacy={sorted(valid_set(lg[0,0]))}"
            )
            assert (pt[0] == -1).sum() == (
                lg[0, 0] == -1
            ).sum(), f"p={p}: -1 sentinel count differs"

        # p=0: only ring-slot 0 is valid; must appear at the last column of the row.
        pos_t = torch.tensor([0], dtype=torch.long)
        cu = torch.tensor([0, 1], dtype=torch.long)
        pt = _get_window_topk_idxs_pertoken(W, pos_t, cu)
        assert pt.shape == (1, W)
        assert valid_set(pt[0]) == {
            0
        }, f"p=0: expected only ring-slot 0, got {valid_set(pt[0])}"
        assert pt[0, -1].item() == 0, "p=0: ring-slot 0 must appear at last column"

    @_skip_if_no_helpers
    def test_pertoken_prefill_matches_legacy_single_seq(self):
        """Single-seq prefill: per-row valid ring-index set must equal legacy's."""
        import torch

        W = 64

        def valid_set(t):
            return set(t[t != -1].tolist())

        for S in [8, 12, 64]:
            positions = torch.arange(S, dtype=torch.long)
            cu = torch.tensor([0, S], dtype=torch.long)
            pt = _get_window_topk_idxs_pertoken(W, positions, cu)  # [S, W]
            lg = _get_window_topk_idxs(W, 1, S, 0)[0]  # [S, K]
            assert pt.shape == (S, W), f"S={S}: expected ({S},{W}), got {pt.shape}"
            for row in range(S):
                pt_v = valid_set(pt[row])
                lg_v = valid_set(lg[row])
                assert pt_v == lg_v, (
                    f"S={S}, row={row} (pos={row}): "
                    f"pertoken valid={sorted(pt_v)} legacy valid={sorted(lg_v)}"
                )

    @_skip_if_no_helpers
    def test_pertoken_multi_seq_packed_no_crosstalk(self):
        """Multi-seq packed batch: each token row depends only on its own absolute position."""
        import torch

        W = 64
        # seq A: 1 decode token at pos=200; seq B: 8 prefill tokens at pos 0..7
        positions = torch.tensor([200, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
        cu = torch.tensor([0, 1, 9], dtype=torch.long)
        out = _get_window_topk_idxs_pertoken(W, positions, cu)  # [9, W]
        assert out.shape == (9, W)

        # Row 0 (seq A, pos=200): must be identical to a solo decode at pos=200.
        solo = _get_window_topk_idxs_pertoken(
            W,
            torch.tensor([200], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        )
        assert torch.equal(
            out[0], solo[0]
        ), "Row 0 (seq A decode pos=200) differs from solo call — cross-seq contamination"

        # Row 1 (seq B, pos=0): only ring-slot 0 valid, at last column.
        assert set(out[1][out[1] != -1].tolist()) == {
            0
        }, f"Row 1 (pos=0): expected only ring-slot 0, got {set(out[1][out[1]!=-1].tolist())}"
        assert out[1, -1].item() == 0, "Row 1: ring-slot 0 must be at last column"

        # Row 8 (seq B, pos=7): slots 0..7 valid, appearing in the last 8 columns.
        valid_r8 = set(out[8][out[8] != -1].tolist())
        assert valid_r8 == set(
            range(8)
        ), f"Row 8 (pos=7): expected {{0..7}}, got {valid_r8}"
        assert out[8, -8:].tolist() == list(
            range(8)
        ), f"Row 8: last 8 columns must be [0..7], got {out[8,-8:].tolist()}"

    @_skip_if_no_helpers
    def test_pertoken_empty_input(self):
        """Empty positions tensor returns shape [0, W] without error."""
        import torch

        W = 64
        positions = torch.empty(0, dtype=torch.long)
        cu = torch.tensor([0], dtype=torch.long)
        out = _get_window_topk_idxs_pertoken(W, positions, cu)
        assert out.shape == (0, W), f"expected (0, {W}), got {out.shape}"


class TestLegacyPreserved:
    def test_legacy_method_signature(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        params = _method_params(cls, "_forward_legacy")
        # _forward_legacy(self, x[, start_pos]) — must take x positionally.
        assert (
            "x" in params and len(params) >= 2
        ), f"_forward_legacy must take self,x[,start_pos]; got {params}"

    def test_legacy_path_body_preserves_register_buffer_kv_cache(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "_forward_legacy")
        assert (
            "self.kv_cache" in src
        ), "_forward_legacy must keep self.kv_cache (the register_buffer)"

    def test_attention_init_keeps_register_buffer(self):
        cls = _find_class(_tree(), "DeepseekV4Attention")
        src = _method_src(cls, "__init__")
        # ast.unparse normalizes string quotes to single quotes, so accept
        # either single- or double-quoted "kv_cache" literal.
        kv_cache_literal = ("'kv_cache'" in src) or ('"kv_cache"' in src)
        assert "register_buffer(" in src and kv_cache_literal, (
            "DeepseekV4Attention.__init__ must keep " 'register_buffer("kv_cache", ...)'
        )


class TestForCausalLMForwardBatchPlumbing:
    def test_for_causal_lm_pulls_forward_batch_from_context(self):
        cls = _find_class(_tree(), "DeepseekV4ForCausalLM")
        src = _method_src(cls, "forward")
        assert (
            "dsv4_forward_batch" in src or "forward_batch" in src
        ), "DeepseekV4ForCausalLM.forward must wire forward_batch from ForwardContext"


class TestBlockForwardThreadsForwardBatch:
    def test_block_forward_threads_forward_batch_into_attn(self):
        cls = _find_class(_tree(), "Block")
        src = _method_src(cls, "forward")
        # Body must pass forward_batch to self.attn(...) somewhere.
        assert (
            "forward_batch=forward_batch" in src
            or "forward_batch=forward_batch," in src
            or "forward_batch=forward_batch)" in src
        ), "Block.forward must pass forward_batch to self.attn(...)"


# ---------------------------------------------------------------------------
# Numerical helpers — Compressor extracted via AST exec (no aiter import)
# ---------------------------------------------------------------------------


def _load_compressor():
    """Exec Compressor + required deps from deepseek_v4.py source.

    Avoids the full module import (which requires ROCm/aiter).  Stubs out
    all external references so Compressor can be instantiated and run on CPU
    with identity / no-op quantization.  Works in both GPU and CPU-only
    test environments.
    """
    import dataclasses
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from functools import lru_cache
    from typing import Any, Iterable, List, Literal, Optional, Tuple

    needed = {
        "_FP4_BLOCK_SIZE",
        "_RMSNorm",
        "_precompute_freqs_cis",
        "_apply_rotary_emb",
        "DeepseekV4Args",
        "Compressor",
    }
    selected = []
    for node in _tree().body:
        name = None
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    name = t.id
                    break
        if name in needed:
            selected.append(node)

    if len(selected) < len(needed):
        return None, None, None

    mini = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(mini)
    code = compile(mini, str(DSV4_SOURCE), "exec")

    class _Stub:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    ns: dict = {
        "__builtins__": __builtins__,
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "Optional": Optional,
        "Any": Any,
        "List": List,
        "Tuple": Tuple,
        "Literal": Literal,
        "Iterable": Iterable,
        "dataclass": dataclasses.dataclass,
        "field": dataclasses.field,
        "lru_cache": lru_cache,
        # External stubs — no-ops so Compressor math is pure float32
        "get_tensor_model_parallel_world_size": lambda: 1,
        "SlidingWindowMLASpec": _Stub,
        "MLAAttentionSpec": _Stub,
        "act_quant_inplace": lambda x, b, fmt: None,
        "fp4_act_quant_inplace": lambda x, b: None,
        "rotate_activation": lambda x: x,
    }
    exec(code, ns)  # noqa: S102
    return (
        ns.get("Compressor"),
        ns.get("DeepseekV4Args"),
        ns.get("_precompute_freqs_cis"),
    )


try:
    _Compressor, _DSV4Args, _precompute_freqs_cis_fn = _load_compressor()
    _COMPRESSOR_AVAILABLE = _Compressor is not None
except Exception:
    _Compressor = _DSV4Args = _precompute_freqs_cis_fn = None
    _COMPRESSOR_AVAILABLE = False

_skip_if_no_compressor = pytest.mark.skipif(
    not _COMPRESSOR_AVAILABLE,
    reason="Compressor not extractable from deepseek_v4.py source",
)


class TestCompressorW4PrefillBlockEmit:
    """Verify _forward_w4 emits one compressed entry per ratio block.

    Bug 2 root cause: the prior code checked only each seq's LAST token,
    so a 12-token prefill (ratio=4) emitted 1 entry instead of 3.
    kv_cache[slot, 0] and [slot, 1] were left as stale zeros, causing
    attention degeneration during decode.
    """

    @staticmethod
    def _make_compressor(ratio=4, dim=16, head_dim=8, rope_head_dim=4, seed=42):
        """Instantiate a small Compressor with seeded random weights."""
        import torch

        args = _DSV4Args(
            dim=dim,
            rope_head_dim=rope_head_dim,
            norm_eps=1e-6,
            scale_fmt=None,
            max_batch_size=4,
        )
        torch.manual_seed(seed)
        c = _Compressor(args, compress_ratio=ratio, head_dim=head_dim)
        # ape uses torch.empty (uninitialized); seed it explicitly so every
        # make_compressor(seed=42) call produces identical APE values.
        torch.nn.init.normal_(c.ape, mean=0.0, std=0.02)
        c.freqs_cis = _precompute_freqs_cis_fn(
            dim=rope_head_dim,
            seqlen=256,
            original_seq_len=0,
            base=10000.0,
            factor=1.0,
            beta_fast=32,
            beta_slow=1,
        )
        return c

    @staticmethod
    def _init_state(c, max_slots=4, num_cache_cols=32, kv_cache_dtype=None):
        """Attach zeroed kv_state / score_state / kv_cache to c.

        kv_cache_dtype defaults to float32 so numerical comparisons avoid
        bf16-rounding variance that appears when autograd is active.
        """
        import torch

        if kv_cache_dtype is None:
            kv_cache_dtype = torch.float32
        coff = 1 + c.overlap
        ring = coff * c.compress_ratio
        inner = coff * c.head_dim
        c.kv_state = torch.zeros(max_slots, ring, inner)
        c.score_state = torch.full((max_slots, ring, inner), float("-inf"))
        c.kv_cache = torch.zeros(
            max_slots, num_cache_cols, c.head_dim, dtype=kv_cache_dtype
        )

    @staticmethod
    def _run_legacy(c, x_flat):
        """Run Compressor legacy prefill path under no_grad; returns kv_cache[0, :T//ratio]."""
        import torch

        coff = 1 + c.overlap
        ring = coff * c.compress_ratio
        inner = coff * c.head_dim
        T = x_flat.size(0)
        c.kv_state = torch.zeros(1, ring, inner)
        c.score_state = torch.full((1, ring, inner), float("-inf"))
        c.kv_cache = torch.zeros(1, T // c.compress_ratio + 1, c.head_dim)  # float32
        with torch.no_grad():
            c.forward(
                x_flat.unsqueeze(0), start_pos=0, forward_batch=None, layer_id=None
            )
        return c.kv_cache[0, : T // c.compress_ratio].detach().clone()

    @_skip_if_no_compressor
    def test_w4_prefill_emits_one_per_ratio_block_single_seq(self):
        """12-token prefill, ratio=4 → 3 entries written to kv_cache (not 1)."""
        import torch
        from types import SimpleNamespace

        ratio, dim, head_dim, rope_head_dim = 4, 16, 8, 4
        seqlen = 12  # 3 complete blocks

        torch.manual_seed(7)
        x_flat = torch.randn(seqlen, dim)

        # Legacy reference (float32 kv_cache, no_grad inside _run_legacy)
        c_leg = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        kv_cache_legacy = self._run_legacy(c_leg, x_flat)  # [3, head_dim]

        # W4 path (float32 kv_cache by default, no_grad context)
        c_w4 = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        self._init_state(c_w4, max_slots=4)
        fb = SimpleNamespace(
            positions=torch.arange(seqlen, dtype=torch.long),
            cu_seqlens_q=torch.tensor([0, seqlen], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.long),
        )
        with torch.no_grad():
            result = c_w4._forward_w4(x_flat, fb, layer_id=0)

        assert result is not None, "_forward_w4 returned None for 12-token prefill"
        assert (
            result.shape[0] == seqlen // ratio
        ), f"Expected {seqlen // ratio} emissions, got {result.shape[0]}"
        kv_cache_w4 = c_w4.kv_cache[0, : seqlen // ratio].detach()  # [3, head_dim]
        torch.testing.assert_close(
            kv_cache_w4,
            kv_cache_legacy,
            atol=1e-5,
            rtol=1e-5,
            msg="W4 kv_cache must match legacy within fp32 tolerance",
        )

    @_skip_if_no_compressor
    def test_w4_decode_single_step_at_boundary(self):
        """Decode token at pos=3, ratio=4 → emits 1 entry to kv_cache[slot, 0]."""
        import torch
        from types import SimpleNamespace

        ratio, dim, head_dim, rope_head_dim = 4, 16, 8, 4

        c = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        self._init_state(c, max_slots=4)

        # Pre-populate state as if positions 0..2 were already scattered.
        # In overlap mode, decode writes to row ratio + pos%ratio.
        torch.manual_seed(11)
        c.kv_state[0, ratio : ratio + 3] = torch.randn(3, (1 + c.overlap) * head_dim)
        c.score_state[0, ratio : ratio + 3] = torch.randn(3, (1 + c.overlap) * head_dim)

        torch.manual_seed(13)
        x_flat = torch.randn(1, dim)
        fb = SimpleNamespace(
            positions=torch.tensor([3], dtype=torch.long),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.long),
        )
        with torch.no_grad():
            result = c._forward_w4(x_flat, fb, layer_id=0)

        assert result is not None, "pos=3 (boundary) must emit 1 entry"
        assert result.shape[0] == 1, f"Expected 1 emission, got {result.shape[0]}"
        assert c.kv_cache[0, 0].abs().sum() > 0, "kv_cache[slot, 0] must be written"

    @_skip_if_no_compressor
    def test_w4_decode_single_step_off_boundary(self):
        """Decode token at pos=2, ratio=4 → returns None (no boundary crossed)."""
        import torch
        from types import SimpleNamespace

        ratio, dim, head_dim, rope_head_dim = 4, 16, 8, 4

        c = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        self._init_state(c, max_slots=4)

        torch.manual_seed(17)
        x_flat = torch.randn(1, dim)
        fb = SimpleNamespace(
            positions=torch.tensor([2], dtype=torch.long),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.long),
        )
        with torch.no_grad():
            result = c._forward_w4(x_flat, fb, layer_id=0)

        assert result is None, "pos=2 (non-boundary) must return None"

    @_skip_if_no_compressor
    def test_w4_multi_seq_independent_emits(self):
        """Packed 2-seq prefill (4 toks each, ratio=4) → 2 independent emissions."""
        import torch
        from types import SimpleNamespace

        ratio, dim, head_dim, rope_head_dim = 4, 16, 8, 4
        seqlen = 4  # 1 complete block per seq

        torch.manual_seed(19)
        x_flat = torch.randn(2 * seqlen, dim)

        # Both seqs at positions [0,1,2,3] each, mapped to distinct slots 1 and 2.
        positions = torch.cat(
            [
                torch.arange(seqlen, dtype=torch.long),
                torch.arange(seqlen, dtype=torch.long),
            ]
        )
        cu = torch.tensor([0, seqlen, 2 * seqlen], dtype=torch.long)
        req_pool_indices = torch.tensor([1, 2], dtype=torch.long)

        c = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        self._init_state(c, max_slots=4)
        fb = SimpleNamespace(
            positions=positions,
            cu_seqlens_q=cu,
            req_pool_indices=req_pool_indices,
        )
        with torch.no_grad():
            result = c._forward_w4(x_flat, fb, layer_id=0)

        assert result is not None, "Packed 2-seq prefill must emit outputs"
        assert (
            result.shape[0] == 2
        ), f"Expected 2 emissions (one per seq), got {result.shape[0]}"
        assert c.kv_cache[1, 0].abs().sum() > 0, "slot 1 kv_cache[0] must be written"
        assert c.kv_cache[2, 0].abs().sum() > 0, "slot 2 kv_cache[0] must be written"

        # Each slot's output must equal an independent solo run (no cross-seq contamination).
        c0 = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        self._init_state(c0, max_slots=4)
        fb0 = SimpleNamespace(
            positions=torch.arange(seqlen, dtype=torch.long),
            cu_seqlens_q=torch.tensor([0, seqlen], dtype=torch.long),
            req_pool_indices=torch.tensor([1], dtype=torch.long),
        )
        with torch.no_grad():
            c0._forward_w4(x_flat[:seqlen], fb0, layer_id=0)

        c1 = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        self._init_state(c1, max_slots=4)
        fb1 = SimpleNamespace(
            positions=torch.arange(seqlen, dtype=torch.long),
            cu_seqlens_q=torch.tensor([0, seqlen], dtype=torch.long),
            req_pool_indices=torch.tensor([2], dtype=torch.long),
        )
        with torch.no_grad():
            c1._forward_w4(x_flat[seqlen:], fb1, layer_id=0)

        torch.testing.assert_close(
            c.kv_cache[1, 0].detach(),
            c0.kv_cache[1, 0].detach(),
            atol=1e-5,
            rtol=1e-5,
            msg="Slot 1 must match independent solo run for seq 0",
        )
        torch.testing.assert_close(
            c.kv_cache[2, 0].detach(),
            c1.kv_cache[2, 0].detach(),
            atol=1e-5,
            rtol=1e-5,
            msg="Slot 2 must match independent solo run for seq 1",
        )

    @_skip_if_no_compressor
    def test_w4_prefill_then_decode_matches_legacy(self):
        """Prefill 8 toks then decode 1 tok at pos=8: W4 path must match legacy single-seq forward."""
        import torch
        from types import SimpleNamespace

        ratio, dim, head_dim, rope_head_dim = 4, 16, 8, 4
        prefill_len = 8  # 2 complete blocks
        torch.manual_seed(23)
        x_prefill = torch.randn(prefill_len, dim)
        torch.manual_seed(29)
        x_decode = torch.randn(1, dim)

        # Legacy: prefill (start_pos=0), then decode (start_pos=prefill_len).
        c_leg = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        coff = 1 + c_leg.overlap
        ring = coff * c_leg.compress_ratio
        inner = coff * c_leg.head_dim
        c_leg.kv_state = torch.zeros(1, ring, inner)
        c_leg.score_state = torch.full((1, ring, inner), float("-inf"))
        c_leg.kv_cache = torch.zeros(1, 8, c_leg.head_dim, dtype=torch.float32)
        with torch.no_grad():
            c_leg.forward(
                x_prefill.unsqueeze(0), start_pos=0, forward_batch=None, layer_id=None
            )
            # Decode at pos=8 (next ratio boundary): legacy expects shape [1,1,dim].
            c_leg.forward(
                x_decode.unsqueeze(0),
                start_pos=prefill_len,
                forward_batch=None,
                layer_id=None,
            )
        leg_kv = c_leg.kv_cache[0, :3].detach().clone()  # blocks 0,1,2

        # W4: same prefill+decode via _forward_w4, with shared kv_state across the two calls.
        c_w4 = self._make_compressor(ratio, dim, head_dim, rope_head_dim)
        self._init_state(c_w4, max_slots=4)
        fb_pre = SimpleNamespace(
            positions=torch.arange(prefill_len, dtype=torch.long),
            cu_seqlens_q=torch.tensor([0, prefill_len], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.long),
        )
        with torch.no_grad():
            c_w4._forward_w4(x_prefill, fb_pre, layer_id=0)
        fb_dec = SimpleNamespace(
            positions=torch.tensor([prefill_len], dtype=torch.long),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.long),
            req_pool_indices=torch.tensor([0], dtype=torch.long),
        )
        with torch.no_grad():
            c_w4._forward_w4(x_decode, fb_dec, layer_id=0)
        w4_kv = c_w4.kv_cache[0, :3].detach().clone()

        # Block 0 and block 1 are written during prefill; block 2 during decode at pos=8.
        torch.testing.assert_close(
            w4_kv,
            leg_kv,
            atol=1e-5,
            rtol=1e-5,
            msg="W4 prefill+decode handoff must match legacy",
        )
