# SPDX-License-Identifier: MIT
# Layer 1 unit tests for atom/model_engine/sequence.py multi-pool block tables.
# Per RFC §9.5.2 — CPU only, mocked, ≤ 30s.


from conftest import MockConfig  # noqa: F401  (forces conftest stubs to run)


def _make_seq(token_ids=None):
    """Construct a Sequence using lazily-imported real class."""
    from atom.model_engine.sequence import Sequence

    return Sequence(token_ids or [1, 2, 3, 4], block_size=4)


# ── block_tables dict default + main alias ─────────────────────────────────


class TestBlockTablesDictDefault:
    def test_block_tables_is_dict(self):
        seq = _make_seq()
        assert isinstance(seq.block_tables, dict)

    def test_block_tables_starts_with_main_empty_list(self):
        # Backwards-compat: every Sequence has a "main" entry from day 1
        # so non-DSV4 single-pool consumers reading `seq.block_tables["main"]`
        # never KeyError.
        seq = _make_seq()
        assert "main" in seq.block_tables
        assert seq.block_tables["main"] == []

    def test_block_table_property_aliases_main(self):
        # Legacy seq.block_table reads must return the same list object as
        # seq.block_tables["main"] — append on one is visible on the other.
        seq = _make_seq()
        seq.block_tables["main"].append(7)
        assert seq.block_table == [7]
        seq.block_table.append(11)
        assert seq.block_tables["main"] == [7, 11]

    def test_block_table_property_setter_writes_main(self):
        # Existing call site model_runner.py:882 does `seq.block_table = [0]`.
        # The property setter must redirect to block_tables["main"].
        seq = _make_seq()
        seq.block_table = [42, 43]
        assert seq.block_tables["main"] == [42, 43]
        assert seq.block_table == [42, 43]

    def test_other_pools_independent_from_main(self):
        seq = _make_seq()
        seq.block_tables["compress"] = [1, 2]
        seq.block_tables["indexer"] = [10]
        seq.block_table.append(99)  # writes main
        assert seq.block_tables["main"] == [99]
        assert seq.block_tables["compress"] == [1, 2]
        assert seq.block_tables["indexer"] == [10]
        # mutating compress doesn't leak into main
        seq.block_tables["compress"].append(3)
        assert seq.block_tables["main"] == [99]


# ── mamba_state_slot precedent untouched ───────────────────────────────────


class TestMambaPrecedent:
    def test_mamba_state_slot_default_minus_one(self):
        # Pre-existing per-request cache-id field (mamba recurrent state)
        # must remain. RFC §6.2.1 cited it as the "non-standard cache pool"
        # precedent; verify we didn't accidentally change it while adding
        # block_tables.
        seq = _make_seq()
        assert seq.mamba_state_slot == -1


# ── id / status / type basics still work ──────────────────────────────────


class TestSequenceBasicsUnchanged:
    def test_id_is_unique(self):
        a = _make_seq()
        b = _make_seq()
        assert a.id != b.id

    def test_token_ids_preserved(self):
        seq = _make_seq([10, 20, 30])
        assert seq.token_ids == [10, 20, 30]
        assert seq.num_tokens == 3
        assert seq.num_prompt_tokens == 3
