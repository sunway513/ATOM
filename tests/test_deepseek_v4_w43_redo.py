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

# ---------------------------------------------------------------------------
# AST helpers — read the source without importing the module
# ---------------------------------------------------------------------------

ATOM_ROOT = Path(__file__).resolve().parent.parent
DSV4_SOURCE = ATOM_ROOT / "atom" / "models" / "deepseek_v4.py"


def _tree() -> ast.Module:
    return ast.parse(DSV4_SOURCE.read_text())


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
