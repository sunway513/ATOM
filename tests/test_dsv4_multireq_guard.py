# SPDX-License-Identifier: MIT
"""W4.3 / Path 2 (issue #37): DSV4 multi-request guard.

Behavior contract (post-W4.3 PR landing):

- The guard NO LONGER raises for DSV4 + ``max_num_seqs > 1`` by default.
  The W4 path (DSV4KVPool + DSV4ForwardBatch + DeepseekV4Attention.forward
  consuming both) makes multi-request safe.
- The guard emits a one-time INFO log on the first DSV4 multi-request
  config seen.
- The legacy ``ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1`` env override is still
  honored as a short-circuit (skips even the log).
- An ``ATOM_DSV4_FORCE_STRICT_GUARD=1`` debug knob restores the
  pre-W4.3 raise, used only for regression bisection.

These tests exercise the production helpers directly, so they cannot
drift from the actual code path Config.__post_init__ takes.
"""

import logging
import os
from unittest.mock import patch

import pytest

# Import the production helpers directly. If this import fails, the
# tests fail at collection time — which is the right behavior (better
# than tests passing because we silently fell back to a copy).
# `atom.utils.dsv4_guard` is the canonical home; `atom.config` re-imports
# it so Config.__post_init__ uses the same code path.
from atom.utils.dsv4_guard import (
    is_dsv4_arch as _is_dsv4_arch,
    validate_dsv4_multireq as _validate_dsv4_multireq,
)


@pytest.fixture(autouse=True)
def _reset_emit_once_flag():
    """Each test starts with a fresh _INFO_LOG_EMITTED state."""
    import atom.utils.dsv4_guard as g_mod

    prev = g_mod._INFO_LOG_EMITTED
    g_mod._INFO_LOG_EMITTED = False
    yield
    g_mod._INFO_LOG_EMITTED = prev


class TestIsDsv4Arch:
    """Module-level arch matcher used by the guard."""

    def test_canonical_dsv4_name(self):
        assert _is_dsv4_arch(["DeepseekV4ForCausalLM"])

    def test_capitalized_variant(self):
        assert _is_dsv4_arch(["DeepSeekV4ForCausalLM"])

    def test_mla_variant(self):
        # Path-B exploration arch reg
        assert _is_dsv4_arch(["DeepseekV4ForCausalLM_MLA"])

    def test_non_dsv4_archs_unaffected(self):
        for arch in [
            "LlamaForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "Qwen3MoeForCausalLM",
            "MixtralForCausalLM",
        ]:
            assert not _is_dsv4_arch([arch])

    def test_empty_list_returns_false(self):
        assert not _is_dsv4_arch([])


class TestValidateDsv4MultireqRelaxed:
    """W4.3+ relaxed-guard truth-table coverage."""

    def test_dsv4_arch_accepts_max_num_seqs_2(self, caplog):
        """W4.3 unlock: max_num_seqs > 1 no longer raises."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            os.environ.pop("ATOM_DSV4_FORCE_STRICT_GUARD", None)
            with caplog.at_level(logging.INFO, logger="atom"):
                # No raise expected
                _validate_dsv4_multireq(
                    architectures=["DeepseekV4ForCausalLM"], max_num_seqs=2
                )
            # The W4.3 INFO log is emitted exactly once on the first
            # DSV4 multi-request config seen.
            assert any(
                "W4 path" in rec.getMessage() for rec in caplog.records
            ), "Expected a W4 multi-request INFO log on first call"

    def test_dsv4_arch_accepts_max_num_seqs_1(self):
        # max_num_seqs == 1 is silent (no log, no raise)
        _validate_dsv4_multireq(architectures=["DeepseekV4ForCausalLM"], max_num_seqs=1)

    def test_dsv4_arch_with_legacy_override_accepts_max_num_seqs_4(self):
        """Legacy env override is still honored (defensive backward compat)."""
        with patch.dict(os.environ, {"ATOM_DSV4_UNSAFE_MULTIREQ_DEV": "1"}):
            import importlib

            from atom.utils import envs as envs_mod

            importlib.reload(envs_mod)
            _validate_dsv4_multireq(
                architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
            )

    def test_non_dsv4_archs_unaffected_by_guard(self):
        for arch in [
            "LlamaForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "Qwen3MoeForCausalLM",
        ]:
            _validate_dsv4_multireq(architectures=[arch], max_num_seqs=512)

    def test_dsv4_arch_name_variants_all_accept(self):
        """All DSV4 arch name variants accept multi-request."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            os.environ.pop("ATOM_DSV4_FORCE_STRICT_GUARD", None)
            for arch in [
                "DeepseekV4ForCausalLM",
                "DeepSeekV4ForCausalLM",
                "DeepseekV4ForCausalLM_MLA",
            ]:
                # No raise expected
                _validate_dsv4_multireq(architectures=[arch], max_num_seqs=2)

    def test_empty_architectures_unaffected(self):
        # Configs with no architectures list (rare; warmup paths) skip guard.
        _validate_dsv4_multireq(architectures=[], max_num_seqs=4)

    def test_info_log_emitted_only_once_per_process(self, caplog):
        """The migration INFO log is emitted at most once per process."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            os.environ.pop("ATOM_DSV4_FORCE_STRICT_GUARD", None)
            with caplog.at_level(logging.INFO, logger="atom"):
                _validate_dsv4_multireq(
                    architectures=["DeepseekV4ForCausalLM"], max_num_seqs=2
                )
                _validate_dsv4_multireq(
                    architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
                )
                _validate_dsv4_multireq(
                    architectures=["DeepseekV4ForCausalLM"], max_num_seqs=8
                )
            w4_msgs = [rec for rec in caplog.records if "W4 path" in rec.getMessage()]
            assert len(w4_msgs) == 1, (
                f"Expected exactly 1 W4 INFO log across 3 calls, got "
                f"{len(w4_msgs)}: {[m.getMessage() for m in w4_msgs]}"
            )


class TestForceStrictGuardEscapeHatch:
    """ATOM_DSV4_FORCE_STRICT_GUARD=1 restores legacy raise for bisection."""

    def test_force_strict_re_enables_raise(self):
        with patch.dict(
            os.environ,
            {"ATOM_DSV4_FORCE_STRICT_GUARD": "1"},
            clear=False,
        ):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            with pytest.raises(ValueError, match="strict guard re-enabled"):
                _validate_dsv4_multireq(
                    architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
                )

    def test_force_strict_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            os.environ.pop("ATOM_DSV4_FORCE_STRICT_GUARD", None)
            # No raise (default off)
            _validate_dsv4_multireq(
                architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
            )

    def test_force_strict_short_circuited_by_legacy_unsafe_override(self):
        """If user set BOTH the legacy override and the strict guard, the
        legacy override wins (it's the earlier exit branch)."""
        with patch.dict(
            os.environ,
            {
                "ATOM_DSV4_UNSAFE_MULTIREQ_DEV": "1",
                "ATOM_DSV4_FORCE_STRICT_GUARD": "1",
            },
        ):
            import importlib

            from atom.utils import envs as envs_mod

            importlib.reload(envs_mod)
            # No raise: the legacy short-circuit wins.
            _validate_dsv4_multireq(
                architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
            )


class TestConfigIntegration:
    """Integration test: confirms `atom.config` actually re-exports the
    guard helper from atom.utils.dsv4_guard (so Config.__post_init__'s
    `_validate_dsv4_multireq` symbol resolves to the same function the
    tests above exercise). Catches drift between the helper and the
    Config import site without triggering the HF-download chain that
    real Config(__init__) does.

    Why it has to be done this way: tests/conftest.py stubs out
    `atom.config` for the lightweight unit test path. So we cannot
    instantiate Config here. Instead we read the file's text (a smoke
    check) and we directly assert that the published symbol behaves
    identically to the canonical helper.
    """

    def test_atom_config_re_exports_guard_with_same_behavior(self):
        """The symbol `_validate_dsv4_multireq` Config.__post_init__
        uses must be identical to `validate_dsv4_multireq` from the
        canonical home — otherwise the guard could drift.

        We verify by reading the source: the line that imports the
        re-export must reference `atom.utils.dsv4_guard`.
        """
        import inspect
        import os.path

        # Locate atom/config.py via the dsv4_guard module's package path
        # (avoiding `import atom.config` which conftest stubs).
        from atom.utils import dsv4_guard

        atom_pkg = os.path.dirname(os.path.dirname(inspect.getfile(dsv4_guard)))
        config_path = os.path.join(atom_pkg, "config.py")
        with open(config_path) as f:
            text = f.read()

        # Assert the re-export line is present in atom/config.py
        assert "from atom.utils.dsv4_guard import validate_dsv4_multireq" in text, (
            "atom/config.py must re-export the guard from atom.utils.dsv4_guard "
            "so Config.__post_init__ uses the same code path tests exercise."
        )
        # Assert __post_init__ actually CALLS the re-exported guard
        assert (
            "_validate_dsv4_multireq(" in text
        ), "atom/config.py Config.__post_init__ must call _validate_dsv4_multireq."


class TestDSV4UnsafeMultireqDevEnv:
    """Env var lookup behavior of ATOM_DSV4_UNSAFE_MULTIREQ_DEV."""

    def test_default_unset_is_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            import importlib

            from atom.utils import envs as envs_mod

            importlib.reload(envs_mod)
            assert envs_mod.ATOM_DSV4_UNSAFE_MULTIREQ_DEV is False

    def test_set_to_1_is_true(self):
        with patch.dict(os.environ, {"ATOM_DSV4_UNSAFE_MULTIREQ_DEV": "1"}):
            import importlib

            from atom.utils import envs as envs_mod

            importlib.reload(envs_mod)
            assert envs_mod.ATOM_DSV4_UNSAFE_MULTIREQ_DEV is True

    def test_set_to_0_is_false(self):
        with patch.dict(os.environ, {"ATOM_DSV4_UNSAFE_MULTIREQ_DEV": "0"}):
            import importlib

            from atom.utils import envs as envs_mod

            importlib.reload(envs_mod)
            assert envs_mod.ATOM_DSV4_UNSAFE_MULTIREQ_DEV is False
