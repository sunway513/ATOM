# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 multi-request guard helpers (issue #37 Path 2).

Lives outside `atom.config` so unit tests can `import` and exercise the
real code path directly without being short-circuited by the conftest
stub of `atom.config` (which replaces the module entirely to skip the
HF download chain in light-weight tests).

`Config.__post_init__` calls `_validate_dsv4_multireq` from here.
"""

from __future__ import annotations


def is_dsv4_arch(architectures: list[str]) -> bool:
    """True iff any architecture name matches DeepSeek-V4.

    Catches both `DeepseekV4` and `DeepSeekV4` casings, plus path-B
    `DeepseekV4ForCausalLM_MLA`.
    """
    return any("DeepseekV4" in a or "DeepSeekV4" in a for a in architectures)


def validate_dsv4_multireq(
    architectures: list[str], max_num_seqs: int
) -> None:
    """DSV4 multi-request guard (issue #37).

    lingpeng's PR1 DSV4 reference uses a B=1-implicit `register_buffer`
    flat KV cache + scalar `start_pos`. The W3.2 iteration chain
    (v3→v6.1) confirmed at least 4 architectural walls preventing
    correct multi-request decode: row collision, cu_seqlens_q packed
    input, allocator eviction, scalar position in RoPE. Full fix
    requires the W4 SGLang-isomorphic refactor (positions:Tensor +
    engine-owned KV pools, branch `feat/dsv4-forward-batch-paged-kv`).

    Until W4 lands, refuse multi-request configs to prevent silently-
    broken output. The `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` env override
    exists for kernel-level perf experiments where output correctness
    is not being measured.

    Raises:
        ValueError: when DSV4 + max_num_seqs > 1 + override unset.
    """
    if not is_dsv4_arch(architectures):
        return
    if max_num_seqs <= 1:
        return
    from atom.utils import envs

    if envs.ATOM_DSV4_UNSAFE_MULTIREQ_DEV:
        return
    raise ValueError(
        "DSV4 architectures currently support only "
        f"max_num_seqs=1 (got max_num_seqs={max_num_seqs}). "
        "Multi-request support is in development on the W4 branch "
        "(feat/dsv4-forward-batch-paged-kv); see "
        "https://github.com/sunway513/ATOM/issues/37 for context. "
        "To bypass this guard for kernel-level perf experiments "
        "where output correctness is NOT being measured, set "
        "ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1."
    )
