# SPDX-License-Identifier: MIT
"""Vendored snapshot of vLLM KV cache spec signatures.

Pinned to the vLLM commit recorded in
:data:`atom.v1.kv_cache_interface._VLLM_AUDIT_COMMIT`.

This file is **data**, not code. It is what
``tests/audit/test_spec_alignment_with_vllm.py`` diffs ATOM specs against on
every PR. Refresh requires running ``tests/audit/refresh_vllm_snapshot.sh``
manually as a separate scheduled job — never as a per-PR network fetch (per
RFC v0.2.6 §9.5.5).

Source: vllm-project/vllm tree at zyongye/vllm:dsv4
        (vllm/v1/kv_cache_interface.py)

Captured on: 2026-04-25
"""

from __future__ import annotations

# Captured commit (must match atom.v1.kv_cache_interface._VLLM_AUDIT_COMMIT).
SNAPSHOT_COMMIT = "e8e38e1686c3ca0835b9556fc1f9b28b9e1a455f"

# Each entry records the vLLM class definition we contract on. Fields list
# the (name, type-as-string, has-default) tuples for every dataclass field
# declared at that level. Inherited fields are not repeated. Properties
# (e.g. vLLM's ``page_size_bytes`` on AttentionSpec) are listed under
# ``properties`` separately because they are not constructor parameters.
#
# Format kept deliberately flat / data-only so the audit logic stays
# trivial and the snapshot stays human-diffable on refresh PRs.

VLLM_SPEC_SIGNATURES = {
    "KVCacheSpec": {
        "bases": [],
        "kw_only": True,
        "frozen": True,
        "fields": [
            ("block_size", "int", False),
        ],
        "properties": [
            "page_size_bytes",  # abstract / overridden in subclasses
            "type_id",
        ],
    },
    "AttentionSpec": {
        "bases": ["KVCacheSpec"],
        "kw_only": True,
        "frozen": True,
        "fields": [
            ("num_kv_heads", "int", False),
            ("head_size", "int", False),
            ("dtype", "torch.dtype", False),
            ("kv_quant_mode", "KVQuantMode", True),  # default KVQuantMode.NONE
            ("page_size_padded", "int | None", True),  # default None
        ],
        "properties": [
            "page_size_bytes",  # computed from real_page_size_bytes + quant
            "real_page_size_bytes",
        ],
    },
    "FullAttentionSpec": {
        "bases": ["AttentionSpec"],
        "kw_only": True,
        "frozen": True,
        # FullAttentionSpec inherits all fields from AttentionSpec; v3.2/v4
        # only add a docstring + merge() classmethod here.
        "fields": [],
        "properties": [],
    },
    "MLAAttentionSpec": {
        "bases": ["FullAttentionSpec"],
        "kw_only": True,
        "frozen": True,
        "fields": [
            ("cache_dtype_str", "str | None", True),  # default None
            ("alignment", "int | None", True),  # default None
            ("compress_ratio", "int", True),  # default 1
            ("model_version", "str | None", True),  # default None
        ],
        "properties": [
            "storage_block_size",
            "real_page_size_bytes",
        ],
    },
    "SlidingWindowSpec": {
        "bases": ["AttentionSpec"],
        "kw_only": True,
        "frozen": True,
        "fields": [
            ("sliding_window", "int", False),
        ],
        "properties": [],
    },
    "SlidingWindowMLASpec": {
        "bases": ["SlidingWindowSpec"],
        "kw_only": True,
        "frozen": True,
        "fields": [
            ("cache_dtype_str", "str | None", True),
            ("alignment", "int | None", True),
            ("compress_ratio", "int", True),
            ("model_version", "str | None", True),
        ],
        "properties": [
            "storage_block_size",
            "real_page_size_bytes",
        ],
    },
}

# The "canonical fields" — names ATOM specs MUST carry verbatim for
# portability with vLLM. These are the load-bearing fields used by the
# block manager, slot mapping, and metadata builder. Adding to this set
# requires both ATOM and vLLM to agree.
CANONICAL_FIELDS_BY_SPEC = {
    "KVCacheSpec": {"block_size"},
    "AttentionSpec": {"num_kv_heads", "head_size", "dtype"},
    "FullAttentionSpec": set(),
    "MLAAttentionSpec": {"compress_ratio"},
    "SlidingWindowMLASpec": {"compress_ratio", "sliding_window"},
}

# Canonical type expectations for each canonical field. Audit fails if
# ATOM declares a field with a different type annotation.
CANONICAL_FIELD_TYPES = {
    "block_size": "int",
    "num_kv_heads": "int",
    "head_size": "int",
    "dtype": "torch.dtype",
    "compress_ratio": "int",
    "sliding_window": "int",
}
