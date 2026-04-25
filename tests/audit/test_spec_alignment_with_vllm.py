# SPDX-License-Identifier: MIT
"""Layer 4 audit: ATOM ↔ vLLM KV-cache spec alignment.

Diffs ATOM's specs (``atom/v1/kv_cache_interface.py``) against the vendored
vLLM snapshot (``tests/audit/_vllm_spec_snapshot.py``). Hard contract per
RFC v0.2.6 §6.2.2 / §9.5.5: every ATOM-declared canonical field MUST exist
in vLLM under the same name with a compatible type annotation. ATOM may
declare fewer fields than vLLM (e.g. omit V3.2-only quantization fields) —
adding NEW canonical fields requires vLLM to have them too.

The snapshot is vendored on disk; this audit is fully offline (no network
fetch on per-PR CI). Refresh via the separate scheduled snapshot job per
RFC §9.5.5.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing

import pytest

from atom.v1.kv_cache_interface import (
    _VLLM_AUDIT_COMMIT,
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)
from tests.audit._vllm_spec_snapshot import (
    CANONICAL_FIELD_TYPES,
    CANONICAL_FIELDS_BY_SPEC,
    SNAPSHOT_COMMIT,
    VLLM_SPEC_SIGNATURES,
)


_ATOM_SPECS = {
    "KVCacheSpec": KVCacheSpec,
    "AttentionSpec": AttentionSpec,
    "FullAttentionSpec": FullAttentionSpec,
    "MLAAttentionSpec": MLAAttentionSpec,
    "SlidingWindowMLASpec": SlidingWindowMLASpec,
}


def _atom_field_type(cls, name: str) -> str:
    """Return ATOM's annotation for ``cls.<name>`` as a normalized string,
    matching the format used in the snapshot ("int", "torch.dtype", etc.).
    """
    hints = typing.get_type_hints(cls)
    if name not in hints:
        raise KeyError(f"{cls.__name__}.{name} has no type annotation")
    t = hints[name]
    # Normalize torch.dtype to its dotted name.
    if getattr(t, "__module__", None) == "torch" and t.__name__ == "dtype":
        return "torch.dtype"
    return getattr(t, "__name__", str(t))


# ── Pin alignment ──────────────────────────────────────────────────────────


class TestPinAlignment:
    def test_atom_pin_matches_snapshot(self):
        # ATOM's _VLLM_AUDIT_COMMIT and the snapshot's SNAPSHOT_COMMIT must
        # always agree. If they drift, refresh is half-done.
        assert _VLLM_AUDIT_COMMIT == SNAPSHOT_COMMIT, (
            f"ATOM _VLLM_AUDIT_COMMIT={_VLLM_AUDIT_COMMIT!r} != "
            f"snapshot SNAPSHOT_COMMIT={SNAPSHOT_COMMIT!r}; refresh "
            "tests/audit/_vllm_spec_snapshot.py to match or revert "
            "atom/v1/kv_cache_interface.py."
        )


# ── Canonical-field presence ───────────────────────────────────────────────


class TestCanonicalFields:
    @pytest.mark.parametrize("spec_name", list(CANONICAL_FIELDS_BY_SPEC.keys()))
    def test_atom_declares_canonical_fields(self, spec_name):
        atom_cls = _ATOM_SPECS[spec_name]
        canon = CANONICAL_FIELDS_BY_SPEC[spec_name]
        # All canonical fields for this spec level must be visible (declared
        # at this level OR inherited) on the ATOM class.
        atom_field_names = {f.name for f in dataclasses.fields(atom_cls)}
        missing = canon - atom_field_names
        assert not missing, (
            f"{spec_name}: missing canonical fields {missing}; "
            f"ATOM has {sorted(atom_field_names)}"
        )


class TestCanonicalFieldTypes:
    @pytest.mark.parametrize(
        "spec_name,field_name",
        [
            (sname, fname)
            for sname, fields in CANONICAL_FIELDS_BY_SPEC.items()
            for fname in fields
        ],
    )
    def test_atom_canonical_field_type_matches(self, spec_name, field_name):
        atom_cls = _ATOM_SPECS[spec_name]
        expected = CANONICAL_FIELD_TYPES[field_name]
        actual = _atom_field_type(atom_cls, field_name)
        assert actual == expected, (
            f"{spec_name}.{field_name}: ATOM annotation {actual!r} != "
            f"canonical {expected!r}"
        )


# ── Subclass relationship match ────────────────────────────────────────────


class TestSubclassRelationships:
    def test_attention_extends_kv_cache_spec(self):
        assert issubclass(AttentionSpec, KVCacheSpec)

    def test_full_attention_extends_attention(self):
        assert issubclass(FullAttentionSpec, AttentionSpec)

    def test_mla_extends_attention_chain(self):
        # vLLM has MLA -> FullAttention -> AttentionSpec -> KVCacheSpec.
        # ATOM may take a different intermediate path (currently MLA
        # extends AttentionSpec directly), but it MUST eventually inherit
        # AttentionSpec + KVCacheSpec so the block manager treats it as an
        # attention-style cache.
        assert issubclass(MLAAttentionSpec, AttentionSpec)
        assert issubclass(MLAAttentionSpec, KVCacheSpec)

    def test_sliding_window_mla_eventually_under_kv_cache_spec(self):
        # vLLM places SlidingWindowMLASpec under SlidingWindowSpec ->
        # AttentionSpec. ATOM places it under MLAAttentionSpec for
        # convenience (so storage_block_size + compress_ratio are reused).
        # The contract is that it's still an MLA-flavored attention cache
        # exposing both compress_ratio and sliding_window canonical fields.
        assert issubclass(SlidingWindowMLASpec, MLAAttentionSpec)
        assert issubclass(SlidingWindowMLASpec, KVCacheSpec)


# ── Hard contract: no surprise ATOM-only canonical fields ──────────────────


class TestNoUndeclaredCanonicalFields:
    """Reverse direction: every field ATOM declares at a spec level must
    EITHER be a canonical field (cataloged in the snapshot) OR exist in
    vLLM's snapshot for that spec OR be a known ATOM-side denormalization
    (currently only ``page_size_bytes``, since vLLM models it as a
    property and ATOM stores it).
    """

    KNOWN_DENORMALIZATIONS = {
        # ATOM stores page_size_bytes as a stored field for simplicity.
        # vLLM models it as a property that depends on dtype, num_kv_heads,
        # block_size, and quant mode. Both encodings are equivalent at the
        # spec level — record the divergence here so the audit doesn't
        # complain about it.
        ("KVCacheSpec", "page_size_bytes"),
    }

    @pytest.mark.parametrize("spec_name", list(_ATOM_SPECS.keys()))
    def test_atom_fields_declared_in_snapshot_or_canonical(self, spec_name):
        atom_cls = _ATOM_SPECS[spec_name]
        # Walk only fields declared *at this class level*, not inherited.
        own_fields = {
            f.name
            for f in dataclasses.fields(atom_cls)
            if f.name in atom_cls.__annotations__
        }
        snap_fields = {
            n for n, _t, _d in VLLM_SPEC_SIGNATURES[spec_name]["fields"]
        }
        # Walk vLLM's class chain to allow inherited fields too — ATOM may
        # declare a field at a different level in the inheritance hierarchy.
        vllm_chain_fields = set(snap_fields)
        for ancestor in VLLM_SPEC_SIGNATURES[spec_name]["bases"]:
            walker = ancestor
            while walker:
                vllm_chain_fields |= {
                    n for n, _t, _d in VLLM_SPEC_SIGNATURES[walker]["fields"]
                }
                bases = VLLM_SPEC_SIGNATURES[walker].get("bases", [])
                walker = bases[0] if bases else None

        canon_for_spec = CANONICAL_FIELDS_BY_SPEC.get(spec_name, set())

        for fname in own_fields:
            if (spec_name, fname) in self.KNOWN_DENORMALIZATIONS:
                continue
            assert (
                fname in canon_for_spec
                or fname in vllm_chain_fields
            ), (
                f"{spec_name} declares ATOM-only field {fname!r}; either "
                f"add it to canonical fields (snapshot) or ensure vLLM has "
                f"the same field at some ancestor level."
            )


# ── Frozen dataclass contract ──────────────────────────────────────────────


class TestFrozenDataclassContract:
    @pytest.mark.parametrize("spec_name", list(_ATOM_SPECS.keys()))
    def test_atom_spec_is_frozen_dataclass(self, spec_name):
        atom_cls = _ATOM_SPECS[spec_name]
        assert dataclasses.is_dataclass(atom_cls), (
            f"{spec_name} must be a dataclass"
        )
        params = atom_cls.__dataclass_params__
        # vLLM specs are all frozen=True; ATOM matches.
        assert params.frozen is True, f"{spec_name} must be frozen=True"
