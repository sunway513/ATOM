#!/usr/bin/env python3
"""
Tests for ATOM_CK_FREE=1 routing logic.
No GPU or model weights required — tests env var detection and routing conditions.

Run: cd /home/pensun/ATOM && python3 -m pytest tests/test_ck_free_mode.py -v
"""

import pytest
import importlib

# All ATOM_* env vars that could affect tests
_ATOM_ENV_VARS = [
    "ATOM_CK_FREE",
    "ATOM_USE_TRITON_MLA_DECODE",
    "ATOM_USE_FLYDSL_MOE",
    "ATOM_USE_TRITON_GEMM",
    "ATOM_USE_TRITON_MXFP4_BMM",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure ATOM_* env vars are unset so defaults are tested reliably."""
    for var in _ATOM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _get_envs():
    """Return the envs module; lazy __getattr__ re-evaluates on each access."""
    import atom.utils.envs as envs

    return envs


class TestAtomCkFreeEnvVar:
    """Test ATOM_CK_FREE env var detection."""

    def test_default_is_false(self):
        assert _get_envs().ATOM_CK_FREE is False

    def test_set_to_1_is_true(self, monkeypatch):
        monkeypatch.setenv("ATOM_CK_FREE", "1")
        assert _get_envs().ATOM_CK_FREE is True

    def test_set_to_0_is_false(self, monkeypatch):
        monkeypatch.setenv("ATOM_CK_FREE", "0")
        assert _get_envs().ATOM_CK_FREE is False

    def test_set_to_empty_is_false(self, monkeypatch):
        monkeypatch.setenv("ATOM_CK_FREE", "")
        assert _get_envs().ATOM_CK_FREE is False


class TestMoeRouting:
    """Test the MOE CK-free condition logic (without importing heavy moe.py)."""

    def _has_ck_moe_sorting(self) -> bool:
        """Replicate the check from moe.py without importing it."""
        try:
            import importlib

            return importlib.util.find_spec("aiter.jit.module_moe_sorting") is not None
        except Exception:
            return False

    def test_ck_free_forces_non_ck_in_condition(self, monkeypatch):
        """Verify the 'or envs.ATOM_CK_FREE' condition works."""
        monkeypatch.setenv("ATOM_CK_FREE", "1")
        envs = _get_envs()

        # The condition in moe.py is:
        #   if not _has_ck_moe_sorting() or envs.ATOM_CK_FREE:
        # When ATOM_CK_FREE=1, this should be True regardless of _has_ck_moe_sorting()
        result = not self._has_ck_moe_sorting() or envs.ATOM_CK_FREE
        assert result is True

    def test_ck_free_off_respects_ck_availability(self, monkeypatch):
        """When ATOM_CK_FREE=0, the condition depends on _has_ck_moe_sorting()."""
        monkeypatch.setenv("ATOM_CK_FREE", "0")
        envs = _get_envs()

        has_ck = self._has_ck_moe_sorting()
        result = not has_ck or envs.ATOM_CK_FREE
        # If CK is available, result should be False (use CK path)
        # If CK is not available, result should be True (use fallback)
        assert result == (not has_ck)


class TestMhaRouting:
    """Test MHA attention routing with ATOM_CK_FREE."""

    def test_ck_free_forces_triton_attn(self, monkeypatch):
        """Verify use_triton_attn is forced True when ATOM_CK_FREE=1."""
        monkeypatch.setenv("ATOM_CK_FREE", "1")
        envs = _get_envs()

        # Simulate the routing logic from attention_mha.py:
        # use_triton_attn = sliding_window != -1 or head_dim != 128
        # if envs.ATOM_CK_FREE: use_triton_attn = True
        sliding_window = -1
        head_dim = 128
        use_triton_attn = sliding_window != -1 or head_dim != 128
        assert use_triton_attn is False  # Would normally be False
        if envs.ATOM_CK_FREE:
            use_triton_attn = True
        assert use_triton_attn is True  # But CK-free forces it True

    def test_ck_free_off_normal_routing(self, monkeypatch):
        """Without CK-free, routing follows normal sliding_window/head_dim logic."""
        monkeypatch.setenv("ATOM_CK_FREE", "0")
        envs = _get_envs()

        sliding_window = -1
        head_dim = 128
        use_triton_attn = sliding_window != -1 or head_dim != 128
        if envs.ATOM_CK_FREE:
            use_triton_attn = True
        assert use_triton_attn is False  # Normal routing, no override


class TestMlaRouting:
    """Test MLA decode routing with ATOM_CK_FREE."""

    def test_ck_free_forces_triton_mla_decode(self, monkeypatch):
        """Verify use_triton_mla_decode is True when ATOM_CK_FREE=1."""
        monkeypatch.setenv("ATOM_CK_FREE", "1")
        monkeypatch.setenv("ATOM_USE_TRITON_MLA_DECODE", "0")
        envs = _get_envs()

        use_triton_mla_decode = envs.ATOM_USE_TRITON_MLA_DECODE or envs.ATOM_CK_FREE
        assert use_triton_mla_decode is True

    def test_triton_mla_decode_standalone(self, monkeypatch):
        """ATOM_USE_TRITON_MLA_DECODE=1 still works independently."""
        monkeypatch.setenv("ATOM_CK_FREE", "0")
        monkeypatch.setenv("ATOM_USE_TRITON_MLA_DECODE", "1")
        envs = _get_envs()

        use_triton_mla_decode = envs.ATOM_USE_TRITON_MLA_DECODE or envs.ATOM_CK_FREE
        assert use_triton_mla_decode is True

    def test_both_off_no_triton(self, monkeypatch):
        """When both are off, use_triton_mla_decode is False."""
        monkeypatch.setenv("ATOM_CK_FREE", "0")
        monkeypatch.setenv("ATOM_USE_TRITON_MLA_DECODE", "0")
        envs = _get_envs()

        use_triton_mla_decode = envs.ATOM_USE_TRITON_MLA_DECODE or envs.ATOM_CK_FREE
        assert use_triton_mla_decode is False
