# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 multi-request guard helpers (issue #37 Path 2).

Lives outside `atom.config` so unit tests can `import` and exercise the
real code path directly without being short-circuited by the conftest
stub of `atom.config` (which replaces the module entirely to skip the
HF download chain in light-weight tests).

`Config.__post_init__` calls `_validate_dsv4_multireq` from here.

W4.3 update (issue #37 Path 3 landing)
--------------------------------------
With the W4.1/W4.2/W4.3 stack landing — engine-owned `DSV4KVPool`,
per-token `DSV4ForwardBatch`, and `DeepseekV4Attention.forward` consuming
both — DSV4 now correctly handles batched multi-request decode without
the row-collision / scalar-position / register-buffer-share bugs that
the W3.2 v3..v6.1 archive chain documented. The hard `raise` is therefore
relaxed in this file: the guard now emits a one-time INFO log explaining
the migration and lets execution proceed.

The legacy `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` env override is preserved so
older scripts and CI tasks that explicitly set it continue to work as a
no-op (the check is now structurally identical regardless of the env).

If a regression in W4.3 ever forces us to re-enable the strict guard, set
`ATOM_DSV4_FORCE_STRICT_GUARD=1` in the environment to restore the
pre-W4.3 raise. This is a debug knob — production code paths should not
rely on it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("atom")

_INFO_LOG_EMITTED = False


def is_dsv4_arch(architectures: list[str]) -> bool:
    """True iff any architecture name matches DeepSeek-V4.

    Catches both `DeepseekV4` and `DeepSeekV4` casings, plus path-B
    `DeepseekV4ForCausalLM_MLA`.
    """
    return any("DeepseekV4" in a or "DeepSeekV4" in a for a in architectures)


def _emit_info_once() -> None:
    """Emit the W4.3 migration notice exactly once per process."""
    global _INFO_LOG_EMITTED
    if _INFO_LOG_EMITTED:
        return
    _INFO_LOG_EMITTED = True
    logger.info(
        "DSV4 multi-request supported via W4 path "
        "(DSV4KVPool + DSV4ForwardBatch). The legacy guard "
        "previously raised here for max_num_seqs > 1 — see issue #37 "
        "for the W4.1/W4.2/W4.3 PR chain."
    )


def validate_dsv4_multireq(architectures: list[str], max_num_seqs: int) -> None:
    """DSV4 multi-request guard (issue #37).

    W4.3+ behavior: emits a one-time INFO log on the first DSV4
    multi-request config seen and returns. Pre-W4.3 this raised
    `ValueError` for `max_num_seqs > 1`; that strict mode is preserved
    only behind `ATOM_DSV4_FORCE_STRICT_GUARD=1` for debug bisection.

    The legacy `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` override still
    short-circuits the function before any logging — kept for backward
    compat with operators who set it via runbooks.

    Raises:
        ValueError: only when ``ATOM_DSV4_FORCE_STRICT_GUARD=1`` AND
            DSV4 + ``max_num_seqs > 1`` AND the unsafe-dev override is
            unset. Production should never hit this path.
    """
    if not is_dsv4_arch(architectures):
        return
    if max_num_seqs <= 1:
        return

    from atom.utils import envs

    if envs.ATOM_DSV4_UNSAFE_MULTIREQ_DEV:
        return

    # W4.3+ default path: log once and proceed. The W4 implementation
    # (engine-owned KV pool + per-token ForwardBatch + multi-row
    # DeepseekV4Attention.forward) is responsible for correctness.
    _emit_info_once()

    # Optional debug-only escape hatch: a force-strict env var that
    # restores the legacy raise. Defaults off; production code paths
    # ignore it entirely.
    import os

    if os.environ.get("ATOM_DSV4_FORCE_STRICT_GUARD", "") in ("1", "true", "TRUE"):
        raise ValueError(
            "DSV4 strict guard re-enabled via ATOM_DSV4_FORCE_STRICT_GUARD=1: "
            f"max_num_seqs={max_num_seqs} > 1 rejected. Unset the env var "
            "to use the W4 multi-request path. See issue #37."
        )
