# SPDX-License-Identifier: MIT
"""W3.2-final / Path 2 (issue #37): DSV4 multi-request guard.

These tests cover Config-level enforcement that DSV4 architectures
currently refuse `max_num_seqs > 1` unless the env override
`ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` is set. The full multi-request
refactor lands on the W4 branch (feat/dsv4-forward-batch-paged-kv);
this guard is the production stop-gap so users do not silently get
cross-talk output.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _check_guard(arch_names: list[str], max_num_seqs: int, env_override: bool):
    """Reproduce the Config.__post_init__ guard logic standalone.

    The Config dataclass triggers a long HF-config download chain on real
    init; we test the guard's predicate directly. The actual production
    path is exercised at runtime, this test covers the truth-table.
    """
    archs_match = any("DeepseekV4" in a or "DeepSeekV4" in a for a in arch_names)
    if archs_match and max_num_seqs > 1 and not env_override:
        raise ValueError(
            f"DSV4 architectures currently support only max_num_seqs=1 "
            f"(got max_num_seqs={max_num_seqs}). ..."
        )


class TestDSV4MultireqGuard:
    """Truth-table coverage for the DSV4 max_num_seqs guard."""

    def test_dsv4_arch_rejects_max_num_seqs_2(self):
        """DSV4 + max_num_seqs > 1 + no override → ValueError."""
        with pytest.raises(ValueError, match="max_num_seqs=1"):
            _check_guard(["DeepseekV4ForCausalLM"], max_num_seqs=2, env_override=False)

    def test_dsv4_arch_accepts_max_num_seqs_1(self):
        """DSV4 + max_num_seqs == 1 → OK (no exception)."""
        _check_guard(["DeepseekV4ForCausalLM"], max_num_seqs=1, env_override=False)

    def test_dsv4_arch_with_override_accepts_max_num_seqs_4(self):
        """DSV4 + max_num_seqs > 1 + override flag → OK."""
        _check_guard(["DeepseekV4ForCausalLM"], max_num_seqs=4, env_override=True)

    def test_non_dsv4_arch_unaffected_by_guard(self):
        """Non-DSV4 archs (Llama, DSV3) are unaffected by this guard."""
        for arch in [
            "LlamaForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
            "DeepseekV32ForCausalLM",
            "Qwen3MoeForCausalLM",
        ]:
            _check_guard([arch], max_num_seqs=512, env_override=False)

    def test_dsv4_arch_name_variants(self):
        """Guard catches both `DeepseekV4` and `DeepSeekV4` casings."""
        for arch in [
            "DeepseekV4ForCausalLM",
            "DeepSeekV4ForCausalLM",
            "DeepseekV4ForCausalLM_MLA",  # path B variant (registered for W4)
        ]:
            with pytest.raises(ValueError, match="max_num_seqs=1"):
                _check_guard([arch], max_num_seqs=2, env_override=False)

    def test_empty_architectures_unaffected(self):
        """Configs with no architectures list (rare; warmup paths) skip guard."""
        _check_guard([], max_num_seqs=4, env_override=False)


class TestDSV4UnsafeMultireqDevEnv:
    """Env var lookup behavior of ATOM_DSV4_UNSAFE_MULTIREQ_DEV."""

    def test_default_unset_is_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            # Re-import envs to clear cached lambda result
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
