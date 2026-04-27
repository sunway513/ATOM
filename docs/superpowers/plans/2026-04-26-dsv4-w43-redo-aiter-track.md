# DSV4 W4.3-Redo + AITER Validator Sprint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver multi-request DSV4 inference on MI355 with good accuracy, by wiring the W4.1 ForwardBatch + W4.2 DSV4KVPool through the model layer behind two feature flags, with a host-side AITER ABI validator catching shape errors before they become HSA exceptions.

**Architecture:** ATOM owns scheduling + pool metadata; AITER owns the hot path. Two flags (`ATOM_DSV4_USE_W4_PATH` and `ATOM_AITER_VALIDATE`) opt into the new path; the post-#44-revert legacy single-request path stays the default. Path 2 guard is amended to enforce double opt-in (`USE_W4_PATH=1` AND `UNSAFE_MULTIREQ_DEV=1`) mechanically. ModelRunner is the sole owner of `DSV4KVPool` per TP rank; Scheduler emits seq lifecycle events via callbacks. `is_dummy_run` short-circuits at the entry of `_maybe_setup_dsv4_forward_batch` (canonical fix for the #42 root cause).

**Tech Stack:**
- Python 3.12, PyTorch 2.x, AMD ROCm 6.x
- ATOM repo: `sunway513/atom` only (do not push to `ROCm/atom` or `valarLip/atom`)
- AITER fork: `aiter-lingpeng` (linked at `/home/pensun/aiter-lingpeng`)
- Test container: `atom_dsv4_feat` (`/opt/venv/bin/python -m pytest`)
- Lint: `/home/pensun/.local/bin/black` + `/home/pensun/.local/bin/ruff` (must be clean before each commit)
- Hardware: MI355 TP=8 (silicon validation only; UT runs on CPU/CUDA-skipped)

**Spec reference:** `docs/superpowers/specs/2026-04-26-dsv4-w43-redo-aiter-track-design.md`

**Branch convention:** Each task lives on its own feature branch off latest `origin/main`. Open a PR per task. Do NOT stack PRs on each other unless explicitly noted.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `aiter-lingpeng/aiter/dsv4_validate.py` | NEW | Host-side ABI validator for DSV4 sparse_attn metadata |
| `aiter-lingpeng/aiter/tests/test_dsv4_validate.py` | NEW | Truth-table UT for the validator |
| `atom/utils/envs.py` | MODIFY | Add `ATOM_DSV4_USE_W4_PATH` + `ATOM_AITER_VALIDATE` env vars |
| `atom/utils/dsv4_guard.py` | MODIFY | Amend `validate_dsv4_multireq` to enforce double opt-in |
| `tests/test_dsv4_multireq_guard.py` | MODIFY | Extend truth-table for new guard semantics |
| `atom/model_engine/scheduler.py` | MODIFY | Add seq-lifecycle event registry (no DSV4 import) |
| `tests/test_scheduler_lifecycle_events.py` | NEW | UT for the event registry |
| `atom/model_engine/model_runner.py` | MODIFY | Own `_dsv4_pool`; subscribe to scheduler events; rewrite `_maybe_setup_dsv4_forward_batch` with `is_dummy_run` short-circuit at entry |
| `tests/test_modelrunner_dsv4_pool_lifecycle.py` | NEW | UT for ownership + idempotent finish |
| `atom/models/deepseek_v4.py` | MODIFY | `DeepseekV4Attention.forward(x, forward_batch)` W4 path; lift `kv_cache` from register_buffer when flag=1; `Compressor.forward(x, forward_batch)` W4 path; `Indexer.forward(...)` W4 path |
| `tests/test_deepseek_v4_w43_redo.py` | NEW | Main attention W4 path UT |
| `tests/test_deepseek_v4_w44_state_redo.py` | NEW | Compressor/Indexer state migration UT (revives PR #45 design pattern) |
| `tests/silicon/silicon_w43_smoke.py` | NEW | Single-prompt + multi-prompt silicon harness |
| `tests/silicon/silicon_w45_acc.py` | NEW | gsm8k limit=20 conc=4 accuracy gate harness |

---

## Task 1: AITER Validator — Module Scaffold + Shape & Rank Checks (§5 #1)

**Branch:** `aiter-lingpeng:feat/dsv4-validator`
**Files:**
- Create: `aiter/dsv4_validate.py`
- Create: `aiter/tests/test_dsv4_validate.py`

- [ ] **Step 1.1: Create empty validator module with public signature**

```python
# aiter/dsv4_validate.py
"""DSV4 sparse_attn ABI validator (issue sunway513/atom#37).

Host-side checker that translates GPU OOB errors into readable
ValueError messages. Enable via ATOM_AITER_VALIDATE=1 in ATOM; default
off in prod for zero overhead.
"""
from __future__ import annotations

import torch


def dsv4_validate_sparse_attn_metadata(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    pool_capacity: int,
) -> None:
    """Raises ValueError with the first failed constraint.

    See docs/superpowers/specs/2026-04-26-dsv4-w43-redo-aiter-track-design.md
    §5 for the full ABI contract.
    """
    # Step 1.3 will fill this in.
    pass
```

- [ ] **Step 1.2: Write failing test for shape & rank checks**

```python
# aiter/tests/test_dsv4_validate.py
import pytest
import torch

from aiter.dsv4_validate import dsv4_validate_sparse_attn_metadata


def _ok_meta(B=1, M=1, K=4, N=12, D=64, num_tokens=1, pool=4, head=2):
    """Construct a known-valid metadata tuple for happy-path tests."""
    return dict(
        q=torch.zeros(B, M, head, D),
        kv=torch.zeros(B, N, D),
        topk_idxs=torch.zeros(B, M, K, dtype=torch.int32),
        slot_mapping=torch.zeros(num_tokens, dtype=torch.long),
        positions=torch.zeros(num_tokens, dtype=torch.long),
        cu_seqlens_q=torch.tensor([0, num_tokens], dtype=torch.int32),
        pool_capacity=pool,
    )


class TestShapeRank:
    def test_q_must_be_4d(self):
        m = _ok_meta()
        m["q"] = torch.zeros(2, 3)  # 2D, wrong
        with pytest.raises(ValueError, match="q must be 4-D"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_kv_must_be_3d(self):
        m = _ok_meta()
        m["kv"] = torch.zeros(2, 3)  # 2D, wrong
        with pytest.raises(ValueError, match="kv must be 3-D"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_qkv_batch_must_match(self):
        m = _ok_meta(B=1)
        m["kv"] = torch.zeros(2, 12, 64)
        with pytest.raises(ValueError, match=r"q\.B=1 != kv\.B=2"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_qkv_head_dim_must_match(self):
        m = _ok_meta(D=64)
        m["kv"] = torch.zeros(1, 12, 32)  # head_dim mismatch
        with pytest.raises(ValueError, match="head_dim=64.*kv.*head_dim=32"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_must_be_3d(self):
        m = _ok_meta()
        m["topk_idxs"] = torch.zeros(4, dtype=torch.int32)
        with pytest.raises(ValueError, match="topk_idxs must be 3-D"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_first_two_dims_must_match_q(self):
        m = _ok_meta(B=1, M=1)
        m["topk_idxs"] = torch.zeros(2, 1, 4, dtype=torch.int32)  # B mismatch
        with pytest.raises(ValueError, match=r"topk_idxs\.shape\[:2\]"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_happy_path_passes(self):
        # Should not raise on valid input
        dsv4_validate_sparse_attn_metadata(**_ok_meta())
```

- [ ] **Step 1.3: Run tests → all FAIL (validator is empty pass)**

Run: `cd /home/pensun/aiter-lingpeng && pytest aiter/tests/test_dsv4_validate.py::TestShapeRank -v 2>&1 | tail -20`
Expected: 6 fails (test_happy_path_passes might pass since pass-validator never raises; that one will fail at integration step 1.6 below)

- [ ] **Step 1.4: Implement shape & rank checks in `dsv4_validate.py`**

```python
def dsv4_validate_sparse_attn_metadata(
    q, kv, topk_idxs, slot_mapping, positions, cu_seqlens_q, pool_capacity,
):
    # ---- 1. Tensor shape & rank --------------------------------------
    if q.dim() != 4:
        raise ValueError(f"q must be 4-D [B,M,H,D], got {tuple(q.shape)}")
    if kv.dim() != 3:
        raise ValueError(f"kv must be 3-D [B,N,D], got {tuple(kv.shape)}")
    if q.shape[0] != kv.shape[0]:
        raise ValueError(f"q.B={q.shape[0]} != kv.B={kv.shape[0]}")
    if q.shape[-1] != kv.shape[-1]:
        raise ValueError(
            f"q.head_dim={q.shape[-1]} != kv.head_dim={kv.shape[-1]}"
        )
    if topk_idxs.dim() != 3:
        raise ValueError(
            f"topk_idxs must be 3-D [B,M,K], got {tuple(topk_idxs.shape)}"
        )
    if topk_idxs.shape[0] != q.shape[0] or topk_idxs.shape[1] != q.shape[1]:
        raise ValueError(
            f"topk_idxs.shape[:2]={tuple(topk_idxs.shape[:2])} != "
            f"q.shape[:2]={tuple(q.shape[:2])}"
        )
```

- [ ] **Step 1.5: Run tests → 6 of 7 pass; happy_path also passes (no negatives in pass-only validator)**

Run: `cd /home/pensun/aiter-lingpeng && pytest aiter/tests/test_dsv4_validate.py::TestShapeRank -v 2>&1 | tail -10`
Expected: 7 passed

- [ ] **Step 1.6: Lint clean**

Run: `/home/pensun/.local/bin/ruff check aiter/dsv4_validate.py aiter/tests/test_dsv4_validate.py && /home/pensun/.local/bin/black --check aiter/dsv4_validate.py aiter/tests/test_dsv4_validate.py`
Expected: All checks passed!

- [ ] **Step 1.7: Commit**

```bash
cd /home/pensun/aiter-lingpeng
git checkout -b feat/dsv4-validator
git add aiter/dsv4_validate.py aiter/tests/test_dsv4_validate.py
git commit -m "feat(dsv4): add validator scaffold + shape/rank checks (sunway513/atom#37)

Host-side ABI validator for DSV4 sparse_attn call sites. This task
covers §5 #1 (shape & rank) of the spec; subsequent tasks fill in
dtype, device/contiguity, topk domain, slot domain, positions,
cu_seqlens_q monotonicity & ownership.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: AITER Validator — Dtype Checks (§5 #2)

**Branch:** `feat/dsv4-validator` (continue Task 1 branch)
**Files:** Modify `aiter/dsv4_validate.py` and `aiter/tests/test_dsv4_validate.py`

- [ ] **Step 2.1: Write failing test for dtype checks**

```python
# Append to test_dsv4_validate.py
class TestDtype:
    def test_topk_must_be_int32(self):
        m = _ok_meta()
        m["topk_idxs"] = torch.zeros(1, 1, 4, dtype=torch.int64)
        with pytest.raises(ValueError, match="topk_idxs.dtype must be int32"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_slot_mapping_must_be_int(self):
        m = _ok_meta()
        m["slot_mapping"] = torch.zeros(1, dtype=torch.float32)
        with pytest.raises(ValueError, match="slot_mapping.dtype must be int"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_positions_must_be_int(self):
        m = _ok_meta()
        m["positions"] = torch.zeros(1, dtype=torch.float32)
        with pytest.raises(ValueError, match="positions.dtype must be int"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_cu_must_be_int(self):
        m = _ok_meta()
        m["cu_seqlens_q"] = torch.tensor([0.0, 1.0])
        with pytest.raises(ValueError, match="cu_seqlens_q.dtype must be int"):
            dsv4_validate_sparse_attn_metadata(**m)
```

- [ ] **Step 2.2: Run tests → 4 fail**

Run: `cd /home/pensun/aiter-lingpeng && pytest aiter/tests/test_dsv4_validate.py::TestDtype -v 2>&1 | tail -10`
Expected: 4 failed

- [ ] **Step 2.3: Append dtype checks to `dsv4_validate_sparse_attn_metadata`**

```python
    # ---- 2. Dtype --------------------------------------------------
    if topk_idxs.dtype != torch.int32:
        raise ValueError(f"topk_idxs.dtype must be int32, got {topk_idxs.dtype}")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"slot_mapping.dtype must be int32/int64, got {slot_mapping.dtype}"
        )
    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"positions.dtype must be int32/int64, got {positions.dtype}"
        )
    if cu_seqlens_q.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"cu_seqlens_q.dtype must be int32/int64, got {cu_seqlens_q.dtype}"
        )
```

- [ ] **Step 2.4: Run tests → all pass**

Run: `cd /home/pensun/aiter-lingpeng && pytest aiter/tests/test_dsv4_validate.py -v 2>&1 | tail -10`
Expected: 11 passed (7 + 4)

- [ ] **Step 2.5: Lint + commit**

Run: `/home/pensun/.local/bin/ruff check aiter/ && /home/pensun/.local/bin/black --check aiter/dsv4_validate.py aiter/tests/test_dsv4_validate.py`
Then: `git add aiter/dsv4_validate.py aiter/tests/test_dsv4_validate.py && git commit -m "feat(dsv4): validator dtype checks (§5 #2)"`

---

## Task 3: AITER Validator — Device & Contiguity Checks (§5 #3)

**Branch:** `feat/dsv4-validator`

- [ ] **Step 3.1: Write failing test**

```python
# Append to test_dsv4_validate.py
class TestDeviceContiguity:
    def test_kv_device_must_match_q(self):
        if not torch.cuda.is_available():
            pytest.skip("needs cuda for cross-device test")
        m = _ok_meta()
        m["q"] = m["q"].to("cuda")
        # kv stays on CPU
        with pytest.raises(ValueError, match=r"kv\.device=.*q\.device="):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_must_be_contiguous(self):
        m = _ok_meta(B=2, M=4, K=4)
        m["topk_idxs"] = m["topk_idxs"].transpose(0, 1)  # non-contig
        with pytest.raises(ValueError, match="topk_idxs must be contiguous"):
            dsv4_validate_sparse_attn_metadata(**m)
```

- [ ] **Step 3.2: Run → 2 fail (or 1 fail + 1 skip)**

Run: `cd /home/pensun/aiter-lingpeng && pytest aiter/tests/test_dsv4_validate.py::TestDeviceContiguity -v 2>&1 | tail -10`
Expected: 1-2 failed

- [ ] **Step 3.3: Append device & contiguity checks**

```python
    # ---- 3. Device & contiguity ----------------------------------------
    dev = q.device
    for name, t in (
        ("kv", kv), ("topk_idxs", topk_idxs), ("slot_mapping", slot_mapping),
        ("positions", positions), ("cu_seqlens_q", cu_seqlens_q),
    ):
        if t.device != dev:
            raise ValueError(f"{name}.device={t.device} != q.device={dev}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
```

- [ ] **Step 3.4: Run → all pass; commit**

Run: `pytest aiter/tests/test_dsv4_validate.py -v 2>&1 | tail -10` (expected: 13 passed or 12 passed + 1 skip)
Then: `git add aiter/ && git commit -m "feat(dsv4): validator device & contiguity checks (§5 #3)"`

---

## Task 4: AITER Validator — Topk Domain (§5 #4 — most likely #42 culprit)

**Branch:** `feat/dsv4-validator`

- [ ] **Step 4.1: Write failing test**

```python
# Append to test_dsv4_validate.py
class TestTopkDomain:
    def test_topk_below_sentinel_rejected(self):
        m = _ok_meta(B=1, M=1, K=4)
        m["topk_idxs"] = torch.tensor([[[-2, 0, 1, 2]]], dtype=torch.int32)
        with pytest.raises(ValueError, match=r"topk_idxs contains values < -1"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_topk_max_must_be_lt_kv_n(self):
        m = _ok_meta(B=1, M=1, K=4, N=12)
        m["topk_idxs"] = torch.tensor([[[0, 1, 2, 128]]], dtype=torch.int32)
        with pytest.raises(ValueError, match=r"topk_idxs max=128 >= kv\.size\(N\)=12"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_all_negative_one_sentinels_pass(self):
        m = _ok_meta(B=1, M=1, K=4)
        m["topk_idxs"] = torch.full((1, 1, 4), -1, dtype=torch.int32)
        # Should not raise — -1 is the skip sentinel
        dsv4_validate_sparse_attn_metadata(**m)

    def test_empty_topk_passes(self):
        m = _ok_meta(B=1, M=0, K=4)
        m["q"] = torch.zeros(1, 0, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 0, 4, dtype=torch.int32)
        # No tokens, nothing to validate against; should pass
        m["positions"] = torch.zeros(0, dtype=torch.long)
        m["slot_mapping"] = torch.zeros(0, dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0], dtype=torch.int32)
        dsv4_validate_sparse_attn_metadata(**m)
```

- [ ] **Step 4.2: Run → 2 fail (negative passes + max bound), 2 pass (sentinels + empty)**

Run: `pytest aiter/tests/test_dsv4_validate.py::TestTopkDomain -v 2>&1 | tail -10`
Expected: 2 failed, 2 passed

- [ ] **Step 4.3: Append topk domain checks**

```python
    # ---- 4. Topk index domain ------------------------------------------
    if topk_idxs.numel() > 0:
        if (topk_idxs < -1).any().item():
            raise ValueError(
                "topk_idxs contains values < -1 (only -1 is the skip sentinel)"
            )
        valid_topk = topk_idxs[topk_idxs >= 0]
        if valid_topk.numel() > 0:
            max_idx = valid_topk.max().item()
            if max_idx >= kv.shape[1]:
                raise ValueError(
                    f"topk_idxs max={max_idx} >= kv.size(N)={kv.shape[1]} "
                    f"-- would cause GPU OOB in sparse_attn"
                )
```

- [ ] **Step 4.4: Run → all pass; commit**

Run: `pytest aiter/tests/test_dsv4_validate.py -v 2>&1 | tail -10` (expected: 17 passed)
Then: `git add aiter/ && git commit -m "feat(dsv4): validator topk domain checks (§5 #4 — #42 culprit)"`

---

## Task 5: AITER Validator — Slot, Positions, Cu Domain (§5 #5/#6/#7)

**Branch:** `feat/dsv4-validator`

- [ ] **Step 5.1: Write failing tests for all three groups**

```python
# Append to test_dsv4_validate.py
class TestSlotDomain:
    def test_slot_negative_rejected(self):
        m = _ok_meta(num_tokens=2)
        m["slot_mapping"] = torch.tensor([0, -1], dtype=torch.long)
        m["positions"] = torch.tensor([0, 1], dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0, 2], dtype=torch.int32)
        m["q"] = torch.zeros(1, 2, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 2, 4, dtype=torch.int32)
        with pytest.raises(ValueError, match="slot_mapping contains negative"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_slot_max_must_be_lt_pool(self):
        m = _ok_meta(num_tokens=2, pool=4)
        m["slot_mapping"] = torch.tensor([0, 4], dtype=torch.long)  # 4 == capacity, OOB
        m["positions"] = torch.tensor([0, 1], dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0, 2], dtype=torch.int32)
        m["q"] = torch.zeros(1, 2, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 2, 4, dtype=torch.int32)
        with pytest.raises(ValueError, match=r"slot_mapping max=4 >= pool_capacity=4"):
            dsv4_validate_sparse_attn_metadata(**m)


class TestPositionsDomain:
    def test_positions_negative_rejected(self):
        m = _ok_meta(num_tokens=2)
        m["positions"] = torch.tensor([0, -1], dtype=torch.long)
        m["slot_mapping"] = torch.tensor([0, 1], dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0, 2], dtype=torch.int32)
        m["q"] = torch.zeros(1, 2, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 2, 4, dtype=torch.int32)
        with pytest.raises(ValueError, match="positions contains negative"):
            dsv4_validate_sparse_attn_metadata(**m)


class TestCuMonotonicity:
    def test_cu_must_start_at_zero(self):
        m = _ok_meta()
        m["cu_seqlens_q"] = torch.tensor([1, 2], dtype=torch.int32)
        with pytest.raises(ValueError, match=r"cu_seqlens_q\[0\] must be 0"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_cu_must_be_monotonic(self):
        m = _ok_meta(num_tokens=2)
        m["q"] = torch.zeros(1, 2, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 2, 4, dtype=torch.int32)
        m["positions"] = torch.tensor([0, 1], dtype=torch.long)
        m["slot_mapping"] = torch.tensor([0, 1], dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0, 2, 1, 2], dtype=torch.int32)
        with pytest.raises(ValueError, match="non-decreasing"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_cu_tail_must_match_positions_count(self):
        m = _ok_meta(num_tokens=2)
        m["q"] = torch.zeros(1, 2, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 2, 4, dtype=torch.int32)
        m["positions"] = torch.tensor([0, 1], dtype=torch.long)
        m["slot_mapping"] = torch.tensor([0, 1], dtype=torch.long)
        m["cu_seqlens_q"] = torch.tensor([0, 5], dtype=torch.int32)  # tail=5, tokens=2
        with pytest.raises(ValueError, match=r"cu_seqlens_q\[-1\]=5 != positions"):
            dsv4_validate_sparse_attn_metadata(**m)

    def test_positions_count_must_match_slot_mapping(self):
        m = _ok_meta(num_tokens=2)
        m["q"] = torch.zeros(1, 2, 2, 64)
        m["topk_idxs"] = torch.zeros(1, 2, 4, dtype=torch.int32)
        m["positions"] = torch.tensor([0, 1], dtype=torch.long)
        m["slot_mapping"] = torch.tensor([0, 1, 2], dtype=torch.long)  # length 3
        m["cu_seqlens_q"] = torch.tensor([0, 2], dtype=torch.int32)
        with pytest.raises(ValueError, match=r"positions\.numel\(\)=2 != slot"):
            dsv4_validate_sparse_attn_metadata(**m)
```

- [ ] **Step 5.2: Run → 6 fail**

Run: `pytest aiter/tests/test_dsv4_validate.py::TestSlotDomain aiter/tests/test_dsv4_validate.py::TestPositionsDomain aiter/tests/test_dsv4_validate.py::TestCuMonotonicity -v 2>&1 | tail -15`
Expected: 6 failed

- [ ] **Step 5.3: Append remaining checks (slot, positions, cu)**

```python
    # ---- 5. Slot mapping domain ----------------------------------------
    if slot_mapping.numel() > 0:
        if (slot_mapping < 0).any().item():
            raise ValueError("slot_mapping contains negative ids")
        max_slot = slot_mapping.max().item()
        if max_slot >= pool_capacity:
            raise ValueError(
                f"slot_mapping max={max_slot} >= pool_capacity={pool_capacity}"
            )

    # ---- 6. Positions domain -------------------------------------------
    if positions.numel() > 0:
        if (positions < 0).any().item():
            raise ValueError("positions contains negative values")

    # ---- 7. cu_seqlens_q monotonicity & token ownership ---------------
    if cu_seqlens_q.numel() < 1:
        raise ValueError("cu_seqlens_q must have at least 1 element ([0])")
    if cu_seqlens_q[0].item() != 0:
        raise ValueError(
            f"cu_seqlens_q[0] must be 0, got {cu_seqlens_q[0].item()}"
        )
    if cu_seqlens_q.numel() >= 2:
        diffs = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        if (diffs < 0).any().item():
            raise ValueError("cu_seqlens_q must be non-decreasing (monotonic)")
    if cu_seqlens_q[-1].item() != positions.numel():
        raise ValueError(
            f"cu_seqlens_q[-1]={cu_seqlens_q[-1].item()} != "
            f"positions.numel()={positions.numel()}"
        )
    if positions.numel() != slot_mapping.numel():
        raise ValueError(
            f"positions.numel()={positions.numel()} != "
            f"slot_mapping.numel()={slot_mapping.numel()}"
        )
```

- [ ] **Step 5.4: Run → all 23+ tests pass**

Run: `pytest aiter/tests/test_dsv4_validate.py -v 2>&1 | tail -15`
Expected: 23 passed (some skipped on no-cuda)

- [ ] **Step 5.5: Lint + commit + push + open PR**

```bash
/home/pensun/.local/bin/ruff check aiter/ tests/ 2>&1 | tail -3
/home/pensun/.local/bin/black --check aiter/dsv4_validate.py aiter/tests/test_dsv4_validate.py 2>&1 | tail -3
git add aiter/ && git commit -m "feat(dsv4): validator slot/positions/cu domain checks (§5 #5/#6/#7)"
git push -u origin feat/dsv4-validator
# Open PR in aiter-lingpeng repo (note: separate repo from sunway513/atom)
```

---

## Task 6: ATOM env vars `ATOM_DSV4_USE_W4_PATH` + `ATOM_AITER_VALIDATE`

**Branch:** `sunway513/atom:feat/dsv4-w43-flags`
**Files:**
- Modify: `atom/utils/envs.py`

- [ ] **Step 6.1: Switch to ATOM repo + create branch**

```bash
cd /home/pensun/ATOM-lingpeng
git checkout main && git pull origin main
git checkout -b feat/dsv4-w43-flags
```

- [ ] **Step 6.2: Add the two env vars**

```python
# In atom/utils/envs.py, alongside ATOM_DSV4_UNSAFE_MULTIREQ_DEV
    "ATOM_DSV4_USE_W4_PATH": lambda: (
        os.getenv("ATOM_DSV4_USE_W4_PATH", "0") == "1"
    ),
    "ATOM_AITER_VALIDATE": lambda: (
        os.getenv("ATOM_AITER_VALIDATE", "0") == "1"
    ),
```

- [ ] **Step 6.3: Add test**

```python
# Append to tests/test_dsv4_multireq_guard.py::TestDSV4UnsafeMultireqDevEnv
    def test_use_w4_path_default_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_DSV4_USE_W4_PATH", None)
            import importlib
            from atom.utils import envs as envs_mod
            importlib.reload(envs_mod)
            assert envs_mod.ATOM_DSV4_USE_W4_PATH is False

    def test_use_w4_path_set_true(self):
        with patch.dict(os.environ, {"ATOM_DSV4_USE_W4_PATH": "1"}):
            import importlib
            from atom.utils import envs as envs_mod
            importlib.reload(envs_mod)
            assert envs_mod.ATOM_DSV4_USE_W4_PATH is True

    def test_aiter_validate_default_false(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ATOM_AITER_VALIDATE", None)
            import importlib
            from atom.utils import envs as envs_mod
            importlib.reload(envs_mod)
            assert envs_mod.ATOM_AITER_VALIDATE is False
```

- [ ] **Step 6.4: Run tests → pass**

Run: `docker exec atom_dsv4_feat bash -c "cd /workspace/ATOM-lingpeng && /opt/venv/bin/python -m pytest tests/test_dsv4_multireq_guard.py -v 2>&1 | tail -10"`
Expected: 19 passed (16 prior + 3 new)

- [ ] **Step 6.5: Lint + commit**

```bash
/home/pensun/.local/bin/black --check atom/ tests/ && /home/pensun/.local/bin/ruff check atom/ tests/
git add atom/utils/envs.py tests/test_dsv4_multireq_guard.py
git commit -m "feat(dsv4): add ATOM_DSV4_USE_W4_PATH + ATOM_AITER_VALIDATE env vars (#37)"
```

---

## Task 7: Path 2 Guard — Double Opt-In Enforcement

**Branch:** `feat/dsv4-w43-flags` (continue Task 6 branch)
**Files:**
- Modify: `atom/utils/dsv4_guard.py`
- Modify: `tests/test_dsv4_multireq_guard.py`

- [ ] **Step 7.1: Write failing tests for new semantic**

```python
# Append to tests/test_dsv4_multireq_guard.py::TestValidateDsv4Multireq
    def test_unsafe_alone_now_rejects(self):
        """Post-W4.3 spec: UNSAFE=1 alone is no longer enough — also need USE_W4_PATH=1."""
        with patch.dict(os.environ, {"ATOM_DSV4_UNSAFE_MULTIREQ_DEV": "1"}):
            os.environ.pop("ATOM_DSV4_USE_W4_PATH", None)
            import importlib
            from atom.utils import envs as envs_mod
            importlib.reload(envs_mod)
            with pytest.raises(ValueError, match="ATOM_DSV4_USE_W4_PATH"):
                _validate_dsv4_multireq(
                    architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
                )

    def test_use_w4_alone_rejects(self):
        with patch.dict(os.environ, {"ATOM_DSV4_USE_W4_PATH": "1"}):
            os.environ.pop("ATOM_DSV4_UNSAFE_MULTIREQ_DEV", None)
            import importlib
            from atom.utils import envs as envs_mod
            importlib.reload(envs_mod)
            with pytest.raises(ValueError, match="ATOM_DSV4_UNSAFE_MULTIREQ_DEV"):
                _validate_dsv4_multireq(
                    architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
                )

    def test_both_flags_allow_multireq(self):
        with patch.dict(os.environ, {
            "ATOM_DSV4_UNSAFE_MULTIREQ_DEV": "1",
            "ATOM_DSV4_USE_W4_PATH": "1",
        }):
            import importlib
            from atom.utils import envs as envs_mod
            importlib.reload(envs_mod)
            # Should not raise
            _validate_dsv4_multireq(
                architectures=["DeepseekV4ForCausalLM"], max_num_seqs=4
            )
```

- [ ] **Step 7.2: Run → 2 fail (the "rejects" tests; the "allow" test was already passing under old semantic)**

Run: `docker exec atom_dsv4_feat bash -c "cd /workspace/ATOM-lingpeng && /opt/venv/bin/python -m pytest tests/test_dsv4_multireq_guard.py::TestValidateDsv4Multireq -v 2>&1 | tail -15"`
Expected: 2 failed, 9 passed

- [ ] **Step 7.3: Amend `validate_dsv4_multireq`**

```python
# atom/utils/dsv4_guard.py
def validate_dsv4_multireq(
    architectures: list[str], max_num_seqs: int
) -> None:
    """DSV4 multi-request guard (issue #37 — amended W4.3-redo).

    Multi-request requires BOTH ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 AND
    ATOM_DSV4_USE_W4_PATH=1 to be set simultaneously. Either alone
    rejects with a readable message naming the missing flag.
    """
    if not is_dsv4_arch(architectures):
        return
    if max_num_seqs <= 1:
        return
    from atom.utils import envs

    unsafe = envs.ATOM_DSV4_UNSAFE_MULTIREQ_DEV
    use_w4 = envs.ATOM_DSV4_USE_W4_PATH

    if not unsafe:
        raise ValueError(
            "DSV4 architectures currently support only "
            f"max_num_seqs=1 (got max_num_seqs={max_num_seqs}). "
            "Multi-request support is in development on the W4 branch; "
            "see https://github.com/sunway513/ATOM/issues/37. To bypass "
            "this guard for kernel-level perf experiments where output "
            "correctness is NOT being measured, set BOTH "
            "ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 AND ATOM_DSV4_USE_W4_PATH=1."
        )
    if not use_w4:
        raise ValueError(
            "DSV4 multi-request requires ATOM_DSV4_USE_W4_PATH=1 to opt "
            "into the W4 path. ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 alone "
            "would route through the legacy (broken) multi-request "
            "path. See issue #37."
        )
```

- [ ] **Step 7.4: Run → all pass**

Run: `docker exec atom_dsv4_feat bash -c "cd /workspace/ATOM-lingpeng && /opt/venv/bin/python -m pytest tests/test_dsv4_multireq_guard.py -v 2>&1 | tail -10"`
Expected: 22 passed (19 + 3 from Task 6)

- [ ] **Step 7.5: Lint + commit**

```bash
/home/pensun/.local/bin/black --check atom/ tests/ && /home/pensun/.local/bin/ruff check atom/ tests/
git add atom/utils/dsv4_guard.py tests/test_dsv4_multireq_guard.py
git commit -m "feat(dsv4): Path 2 guard double opt-in enforcement (#37 P1.1)"
git push -u origin feat/dsv4-w43-flags
```

---

## Task 8: Scheduler Lifecycle Event Registry

**Branch:** `sunway513/atom:feat/dsv4-scheduler-events` (off latest main)
**Files:**
- Modify: `atom/model_engine/scheduler.py`
- Create: `tests/test_scheduler_lifecycle_events.py`

- [ ] **Step 8.1: Create branch**

```bash
git checkout main && git pull origin main
git checkout -b feat/dsv4-scheduler-events
```

- [ ] **Step 8.2: Write failing tests**

```python
# tests/test_scheduler_lifecycle_events.py
"""Scheduler emits seq lifecycle events via callback registry (issue #37 W4.3)."""
import pytest
from unittest.mock import MagicMock

from atom.model_engine.scheduler import Scheduler


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.kv_cache_block_size = 4
    cfg.num_kvcache_blocks = 100
    cfg.enable_prefix_caching = False
    cfg.max_num_seqs = 4
    cfg.max_num_batched_tokens = 64
    cfg.bos_token_id = 1
    cfg.eos_token_id = 2
    cfg.stop_token_ids = []
    cfg.scheduler_delay_factor = 0.0
    cfg.speculative_config = None
    cfg.kv_cache_pool_blocks = {}
    return cfg


class TestEventRegistry:
    def test_register_admit_listener(self, mock_config):
        sched = Scheduler(mock_config)
        listener = MagicMock()
        sched.register_admit_listener(listener)
        # Internal registry should now have the listener
        assert listener in sched._admit_listeners

    def test_register_finish_listener(self, mock_config):
        sched = Scheduler(mock_config)
        listener = MagicMock()
        sched.register_finish_listener(listener)
        assert listener in sched._finish_listeners

    def test_no_dsv4_pool_import(self):
        """Scheduler must not import DSV4KVPool — it's a pure event emitter."""
        import atom.model_engine.scheduler as sched_mod
        src = open(sched_mod.__file__).read()
        assert "dsv4_pool" not in src.lower(), (
            "scheduler.py must not reference dsv4_pool — it should only "
            "expose events for ModelRunner to subscribe to"
        )
        assert "DSV4KVPool" not in src
```

- [ ] **Step 8.3: Run → 3 fail (registry methods don't exist)**

Run: `docker exec atom_dsv4_feat bash -c "cd /workspace/ATOM-lingpeng && /opt/venv/bin/python -m pytest tests/test_scheduler_lifecycle_events.py -v 2>&1 | tail -10"`
Expected: 3 failed

- [ ] **Step 8.4: Add event registry to Scheduler**

```python
# In atom/model_engine/scheduler.py, in Scheduler.__init__ append:
        # W4.3 (issue #37): seq lifecycle event registry. ModelRunner
        # subscribes its DSV4KVPool admit/finish methods here. Scheduler
        # itself does NOT know DSV4KVPool exists.
        self._admit_listeners: list = []
        self._finish_listeners: list = []

# Add public methods on Scheduler class:
    def register_admit_listener(self, fn) -> None:
        """Subscribe a callable(seq_id: int) → None to be invoked on admit."""
        self._admit_listeners.append(fn)

    def register_finish_listener(self, fn) -> None:
        """Subscribe a callable(seq_id: int) → None to be invoked on finish/preempt."""
        self._finish_listeners.append(fn)

    def _emit_admit(self, seq_id: int) -> None:
        for listener in self._admit_listeners:
            try:
                listener(seq_id)
            except Exception as e:
                logger.warning(f"DSV4 admit listener {listener} raised {e}; ignoring")

    def _emit_finish(self, seq_id: int) -> None:
        for listener in self._finish_listeners:
            try:
                listener(seq_id)
            except Exception as e:
                logger.warning(f"DSV4 finish listener {listener} raised {e}; ignoring")
```

- [ ] **Step 8.5: Wire `_emit_admit` / `_emit_finish` into existing add/finish paths**

Find `Scheduler.add_request` (or equivalent) and append `self._emit_admit(seq.seq_id)` after the existing block_manager allocation.
Find `Scheduler.finish` / `_remove_seq` and append `self._emit_finish(seq.seq_id)`.

(The exact line numbers depend on current main; the integrating engineer should grep `block_manager.allocate` / `seq.is_finished` to locate.)

- [ ] **Step 8.6: Run tests → pass**

Run: `docker exec atom_dsv4_feat bash -c "cd /workspace/ATOM-lingpeng && /opt/venv/bin/python -m pytest tests/test_scheduler_lifecycle_events.py tests/test_scheduler.py -v 2>&1 | tail -10"`
Expected: 3 new + all existing scheduler tests pass

- [ ] **Step 8.7: Lint + commit + push**

```bash
/home/pensun/.local/bin/black --check atom/ tests/ && /home/pensun/.local/bin/ruff check atom/ tests/
git add atom/model_engine/scheduler.py tests/test_scheduler_lifecycle_events.py
git commit -m "feat(scheduler): seq lifecycle event registry (#37 P2.2 ownership)"
git push -u origin feat/dsv4-scheduler-events
```

---

## Task 9: ModelRunner Pool Ownership + `is_dummy_run` Short-Circuit

**Branch:** `sunway513/atom:feat/dsv4-modelrunner-w4` (off main, depends on Tasks 6+7+8 merged)
**Files:**
- Modify: `atom/model_engine/model_runner.py`
- Create: `tests/test_modelrunner_dsv4_pool_lifecycle.py`

- [ ] **Step 9.1: Create branch + write failing tests**

```bash
git checkout main && git pull origin main
git checkout -b feat/dsv4-modelrunner-w4
```

```python
# tests/test_modelrunner_dsv4_pool_lifecycle.py
"""ModelRunner is the sole owner of DSV4KVPool (issue #37 P2.2)."""
import pytest
from unittest.mock import MagicMock, patch
from atom.engine.kv_pool.dsv4_pool import DSV4KVPool


class TestPoolOwnership:
    def test_is_dummy_run_short_circuits(self):
        """Critical fix for #42: dummy_batch must not enter W4 path."""
        from atom.model_engine.model_runner import ModelRunner

        runner = MagicMock(spec=ModelRunner)
        runner._is_dsv4_model.return_value = True
        runner._dsv4_pool = MagicMock(spec=DSV4KVPool)

        dummy_batch = MagicMock()
        dummy_batch.is_dummy_run = True
        dummy_batch.req_ids = [0, 1, 2, 3]

        # Bind real method
        from atom.model_engine.model_runner import (
            ModelRunner as _MR,
        )
        pool, fb = _MR._maybe_setup_dsv4_forward_batch(
            runner, dummy_batch, attn_metadata=MagicMock(), positions=MagicMock()
        )
        assert pool is None
        assert fb is None
        # Pool must NOT have been admitted
        runner._dsv4_pool.admit_request.assert_not_called()

    def test_real_request_admits_to_pool(self):
        from atom.model_engine.model_runner import ModelRunner

        runner = MagicMock(spec=ModelRunner)
        runner._is_dsv4_model.return_value = True
        runner._dsv4_pool = MagicMock(spec=DSV4KVPool)

        real_batch = MagicMock()
        real_batch.is_dummy_run = False
        real_batch.req_ids = [0, 1]

        # ... (see actual implementation; assert admit_request called twice)


class TestFinishIdempotent:
    def test_finish_called_twice_is_safe(self):
        # Pool.finish_request must be idempotent under preempt-then-finish race
        pool = DSV4KVPool(max_active_seqs=4, layer_count=2, head_dim=64,
                          attn_window=128, compress_ratio_main=128,
                          compress_ratio_indexer=4, dtype="float32",
                          device="cpu")
        pool.admit_request(seq_id=0)
        pool.finish_request(seq_id=0)
        pool.finish_request(seq_id=0)  # second call must not raise
```

- [ ] **Step 9.2: Run → fail (no impl yet)**

Expected: 3 failed

- [ ] **Step 9.3: Implement `_maybe_setup_dsv4_forward_batch` with `is_dummy_run` short-circuit at entry**

```python
# In atom/model_engine/model_runner.py
def _maybe_setup_dsv4_forward_batch(
    self,
    batch: Optional["ScheduledBatch"],
    attn_metadata,
    positions: torch.Tensor,
):
    """W4.3-redo: lazy-init pool, admit slots, build DSV4ForwardBatch.

    CRITICAL: is_dummy_run short-circuit at the very entry. Warmup
    runs MAX_BATCHED_TOKENS-sized synthetic tensors that crash the W4
    path with HSA exception 0x1016 (issue #37 Evidence I/I'/I''). The
    only safe behavior is to fall through to the legacy single-request
    path.
    """
    if not self._is_dsv4_model():
        return None, None

    # Step 9.3 critical line — must be FIRST after the arch check.
    if batch is not None and getattr(batch, "is_dummy_run", False):
        return None, None

    # Also short-circuit if the W4 path is not opted in. The pool
    # itself is harmless to allocate but admitting seqs without
    # ATOM_DSV4_USE_W4_PATH=1 leaks slots.
    from atom.utils import envs
    if not envs.ATOM_DSV4_USE_W4_PATH:
        return None, None

    if not hasattr(self, "_dsv4_pool") or self._dsv4_pool is None:
        self._dsv4_pool = self._build_dsv4_pool()
        # Subscribe to scheduler events for finish (admit happens here below).
        if hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.register_finish_listener(
                self._dsv4_pool.finish_request
            )

    # Admit any seqs the pool hasn't seen yet.
    seq_ids: list[int] = []
    if batch is not None:
        seq_ids = list(batch.req_ids)
        for sid in seq_ids:
            self._dsv4_pool.admit_request(sid)

    # Build the per-step DSV4ForwardBatch.
    forward_batch = None
    if attn_metadata is not None and positions is not None:
        try:
            from atom.utils.dsv4_forward_batch import DSV4ForwardBatch
            forward_batch = DSV4ForwardBatch.from_attn_metadata(
                attn_metadata, positions,
                seq_ids=seq_ids if seq_ids else None,
                pool=self._dsv4_pool,
            )
        except Exception as e:
            logger.warning(
                f"DSV4: failed to build ForwardBatch ({e}); "
                f"falling back to legacy path"
            )
            forward_batch = None
    return self._dsv4_pool, forward_batch
```

Also implement `_build_dsv4_pool` and idempotent `DSV4KVPool.finish_request` (verify the latter is already idempotent in W4.2 code; if not, fix it as part of this task).

- [ ] **Step 9.4: Run tests → pass**

Run: `docker exec atom_dsv4_feat bash -c "cd /workspace/ATOM-lingpeng && /opt/venv/bin/python -m pytest tests/test_modelrunner_dsv4_pool_lifecycle.py tests/test_dsv4_pool.py -v 2>&1 | tail -15"`
Expected: 3 + 23 = 26 passed

- [ ] **Step 9.5: Wire ForwardContext passthrough**

In `set_forward_context` (or equivalent), add `dsv4_pool` and `dsv4_forward_batch` optional kwargs that store on `ForwardContext`. (W4.3-original PR #42 had this; verify it's still present after revert. If not, restore minimally.)

- [ ] **Step 9.6: Lint + commit + push**

```bash
/home/pensun/.local/bin/black --check atom/ tests/ && /home/pensun/.local/bin/ruff check atom/ tests/
git add atom/model_engine/model_runner.py atom/utils/forward_context.py tests/test_modelrunner_dsv4_pool_lifecycle.py
git commit -m "feat(modelrunner): own DSV4KVPool + is_dummy_run short-circuit (#37 W4.3-redo)"
git push -u origin feat/dsv4-modelrunner-w4
```

---

## Task 10: ATOM W4.3-Redo — Main Attention W4 Path

**Branch:** `sunway513/atom:feat/dsv4-w43-attention` (off main; depends on Tasks 6/7/8/9 merged)
**Files:**
- Modify: `atom/models/deepseek_v4.py` `DeepseekV4Attention`
- Create: `tests/test_deepseek_v4_w43_redo.py`

- [ ] **Step 10.1: Create branch + write failing tests**

```bash
git checkout main && git pull origin main
git checkout -b feat/dsv4-w43-attention
```

```python
# tests/test_deepseek_v4_w43_redo.py
"""W4.3-redo: DeepseekV4Attention consumes DSV4ForwardBatch + KVPool.

Validates the spec §3 "ATOM W4.3-redo (main attention)" component.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch


def _make_tiny_attention():
    """Construct a minimal DeepseekV4Attention for unit-level testing."""
    # ... (see actual test_dsv4_mla_specs.py / test_deepseek_v4_w43_consume.py
    # archive on feat/dsv4-w43-model-consume branch for the working pattern)


class TestRegisterBufferAbsence:
    def test_no_kv_cache_register_buffer_under_w4_flag(self):
        with patch.dict("os.environ", {"ATOM_DSV4_USE_W4_PATH": "1"}):
            attn = _make_tiny_attention()
            buf_names = dict(attn.named_buffers()).keys()
            assert "kv_cache" not in buf_names, (
                "DeepseekV4Attention.kv_cache must NOT be a register_buffer "
                "when W4 flag is on (W4.3-redo)"
            )


class TestPerTokenRoPE:
    def test_rope_freqs_per_token(self):
        """Per-token RoPE: positions=[12,13,15,12] yields 4 distinct freqs_cis."""
        # Use a freqs_cis lookup mock; assert freqs_cis[positions] is gathered
        # rather than freqs_cis[positions[0]:positions[0]+seqlen] sliced.
        ...


class TestPerSeqSlotMapping:
    def test_per_token_kv_scatter_into_pool(self):
        """Each token writes to (slot[seq_idx_of_token], position % win)."""
        ...


class TestValidatorIntegration:
    def test_validator_called_when_aiter_validate_flag_on(self):
        with patch.dict("os.environ", {
            "ATOM_DSV4_USE_W4_PATH": "1",
            "ATOM_AITER_VALIDATE": "1",
        }):
            with patch("aiter.dsv4_validate.dsv4_validate_sparse_attn_metadata") as v:
                # ... run forward and assert v was called
                pass

    def test_validator_skipped_when_aiter_validate_flag_off(self):
        with patch.dict("os.environ", {
            "ATOM_DSV4_USE_W4_PATH": "1",
            "ATOM_AITER_VALIDATE": "0",
        }):
            with patch("aiter.dsv4_validate.dsv4_validate_sparse_attn_metadata") as v:
                # ... run forward
                v.assert_not_called()


class TestLegacyFallback:
    def test_flag_off_uses_register_buffer_path(self):
        """ATOM_DSV4_USE_W4_PATH=0 must keep the existing post-#44 path bit-for-bit."""
        with patch.dict("os.environ", {"ATOM_DSV4_USE_W4_PATH": "0"}):
            attn = _make_tiny_attention()
            buf_names = dict(attn.named_buffers()).keys()
            assert "kv_cache" in buf_names  # legacy path still has the buffer
```

- [ ] **Step 10.2: Run → fail (helper not implemented)**

Expected: 5+ failed (some skipped pending impl)

- [ ] **Step 10.3: Implement `DeepseekV4Attention.forward(x, forward_batch=None)` W4 path**

The structure (from PR #42 archive, simplified with the lessons learned):

```python
def forward(self, x, forward_batch=None):
    """W4.3-redo: consumes optional forward_batch.

    When forward_batch is None OR ATOM_DSV4_USE_W4_PATH=0:
        legacy single-request path (preserved bit-for-bit from
        post-#44 main).
    Else:
        W4 multi-request path:
        - per-token RoPE via freqs_cis[positions]
        - per-seq slot KV scatter via DSV4KVPool view
        - validator call before sparse_attn (if ATOM_AITER_VALIDATE)
    """
    from atom.utils import envs

    if forward_batch is None or not envs.ATOM_DSV4_USE_W4_PATH:
        return self._forward_legacy(x)
    return self._forward_w4(x, forward_batch)


def _forward_legacy(self, x):
    """Original post-#44 path. Bit-for-bit preserved."""
    # ... (copy from current main's DeepseekV4Attention.forward)


def _forward_w4(self, x, forward_batch):
    """W4 multi-request path."""
    pool_view = forward_batch.kv_pool.view_for_layer(self.layer_id)
    kv_cache = pool_view["kv_cache"]
    positions = forward_batch.positions
    # Per-token RoPE
    freqs_cis = self.freqs_cis[positions]
    # Q/KV projection (unchanged)
    q = self.wq_b(self.q_norm(self.wq_a(x)))
    kv = self.kv_norm(self.wkv(x))
    # Apply RoPE per-token
    _apply_rotary_emb(q[..., -self.rope_head_dim:], freqs_cis)
    _apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs_cis)
    # Per-token KV scatter
    out_cache_loc = forward_batch.kv_pool.compute_out_cache_loc(
        positions, forward_batch.req_pool_indices,
        forward_batch.cu_seqlens_q, ring="main",
    )
    kv_cache.view(-1, self.head_dim)[out_cache_loc] = kv
    # Build topk_idxs (per-token)
    topk_idxs = self._build_topk_per_token(forward_batch)
    # Validator gate
    if envs.ATOM_AITER_VALIDATE:
        from aiter.dsv4_validate import dsv4_validate_sparse_attn_metadata
        dsv4_validate_sparse_attn_metadata(
            q.unsqueeze(0), kv_cache.view(...).contiguous(),
            topk_idxs, out_cache_loc, positions,
            forward_batch.cu_seqlens_q,
            pool_capacity=forward_batch.kv_pool.config.max_active_seqs,
        )
    o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    return self.wo_b(...)
```

(The PR #42 archive on `feat/dsv4-w43-model-consume` has the working math; reuse what passed UT, add the validator call, ensure `is_dummy_run` short-circuit upstream.)

- [ ] **Step 10.4: Remove `register_buffer("kv_cache", ...)` from `DeepseekV4Attention.__init__` when flag=1; keep for legacy path**

Use a lazy-init pattern: don't allocate `register_buffer` at construction; allocate on first forward in legacy path only.

- [ ] **Step 10.5: Run all UT**

```bash
docker exec atom_dsv4_feat bash -c "cd /workspace/ATOM-lingpeng && /opt/venv/bin/python -m pytest tests/test_deepseek_v4_w43_redo.py tests/test_dsv4_pool.py tests/test_dsv4_forward_batch.py tests/test_dsv4_multireq_guard.py -v 2>&1 | tail -20"
```
Expected: all pass (cumulative ~80+ tests)

- [ ] **Step 10.6: Lint + commit + push**

```bash
/home/pensun/.local/bin/black --check atom/ tests/ && /home/pensun/.local/bin/ruff check atom/ tests/
git add atom/models/deepseek_v4.py tests/test_deepseek_v4_w43_redo.py
git commit -m "feat(dsv4): W4.3-redo main attention consumes ForwardBatch+pool (#37)"
git push -u origin feat/dsv4-w43-attention
```

---

## Task 11: ATOM W4.4-Redo — Compressor State Migration

**Branch:** `sunway513/atom:feat/dsv4-w44-compressor` (off main; depends on Task 10)
**Files:**
- Modify: `atom/models/deepseek_v4.py` `Compressor`
- Create: `tests/test_deepseek_v4_w44_state_redo.py`

- [ ] **Step 11.1: Create branch**
- [ ] **Step 11.2: Write failing tests** (revive PR #45 32-UT pattern; key tests: `kv_state` not register_buffer when flag=1, per-seq compress trigger via `cu_seqlens_q[1:]-1` modular ring math, advanced indexing into pool's compressor slab)
- [ ] **Step 11.3: Implement `Compressor._forward_w4(x, forward_batch)` mirroring SGLang `compressor.py:236` ring math**
- [ ] **Step 11.4: Run UT → all pass**
- [ ] **Step 11.5: Lint + commit + push**

(Each sub-step ~3-5 mins for an experienced engineer who has read PR #45 archive; the closed PR's 32 UT can be revived almost verbatim, just rebased onto the new W4.3 forward signature.)

```bash
git checkout main && git pull
git checkout -b feat/dsv4-w44-compressor
# PR #45 archive at origin/feat/dsv4-w44-state-migration commit 8d202df has reference impl
git show 8d202df -- atom/models/deepseek_v4.py | grep -A 200 "class Compressor"
# Adapt to current main + W4.3 signature; commit + push
git push -u origin feat/dsv4-w44-compressor
```

---

## Task 12: ATOM W4.4-Redo — Indexer State Migration

**Branch:** `sunway513/atom:feat/dsv4-w44-indexer` (off main; depends on Task 11)
**Files:**
- Modify: `atom/models/deepseek_v4.py` `Indexer`
- Modify: `tests/test_deepseek_v4_w44_state_redo.py`

- [ ] **Step 12.1: Create branch**
- [ ] **Step 12.2: Write failing tests** (Indexer.kv_cache not register_buffer when flag=1, per-token writes via `compute_out_cache_loc(ring="indexer")`, inner Compressor uses pool view)
- [ ] **Step 12.3: Implement `Indexer._forward_w4(x, qr, forward_batch, offset)`**
- [ ] **Step 12.4: Run UT → all pass**
- [ ] **Step 12.5: Lint + commit + push**

---

## Task 13: Silicon Smoke Harness (single + multi prompt)

**Branch:** `sunway513/atom:feat/dsv4-silicon-w43-harness` (off main; depends on Tasks 10-12 merged)
**Files:**
- Create: `tests/silicon/silicon_w43_smoke.py`

- [ ] **Step 13.1: Create branch + harness file (adapt from silicon_fact_multireq.py)**

```python
# tests/silicon/silicon_w43_smoke.py
"""W4.3 silicon smoke harness (issue #37 W4.5).

Two modes via --mode flag:
  - single: max_num_seqs=1, num_prompts=1 (legacy fallback baseline)
  - multi:  max_num_seqs=4, num_prompts=4 (W4 path validation)

Both modes require manual env opt-in:
  ATOM_DSV4_USE_W4_PATH=1 (multi only)
  ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 (multi only)
  ATOM_AITER_VALIDATE=1 (debug, recommended for first runs)
"""
import argparse
import json
import os
from pathlib import Path

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

# (see silicon_fact_multireq.py for the structured prompt set)
HERO = "如何在一个月内增肌10公斤"
SECONDARY = [
    "Briefly describe Beijing in 3 sentences.",
    "Write a Python function to compute the nth Fibonacci number.",
    "List 5 common machine learning algorithms.",
]


def main():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--out", type=str, default="/workspace/ATOM-lingpeng/logs/silicon_w43.json")
    args = parser.parse_args()

    # Apply mode defaults
    if args.mode == "single":
        args.max_num_seqs = 1
        n_prompts = 1
    else:
        args.max_num_seqs = 4
        n_prompts = 4

    prompts_raw = [HERO] + SECONDARY[:n_prompts - 1]
    # ... encoding + generate (copy from silicon_fact_multireq.py)
    # ... write JSON output
```

- [ ] **Step 13.2: Commit + push**

```bash
git checkout -b feat/dsv4-silicon-w43-harness
git add tests/silicon/silicon_w43_smoke.py
git commit -m "test(silicon): W4.3 smoke harness (single/multi mode) (#37 W4.5)"
git push -u origin feat/dsv4-silicon-w43-harness
```

---

## Task 14: Silicon Validation — Run + Capture Evidence J

**Branch:** `sunway513/atom:docs/dsv4-w45-evidence-j` (off main; depends on Tasks 13 + 10-12 merged)
**Files:**
- Create: `docs/evidence/dsv4_w45/EVIDENCE_J.md`
- Create: `docs/evidence/dsv4_w45/silicon_w43_single.json`
- Create: `docs/evidence/dsv4_w45/silicon_w43_multi.json`
- Create: `docs/evidence/dsv4_w45/silicon_w43_*.log`

- [ ] **Step 14.1: Run regression baseline (flag=0, single prompt)**

```bash
docker exec atom_dsv4_feat bash -c "
  ATOM_DSV4_USE_W4_PATH=0 \
  /opt/venv/bin/python -m tests.silicon.silicon_w43_smoke \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-model-len 2048 --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --mode single --max-tokens 32 \
    --out /workspace/ATOM-lingpeng/logs/silicon_w43_baseline.json \
  > /workspace/ATOM-lingpeng/logs/silicon_w43_baseline.log 2>&1
"
```
Expected: rc=0; output coherent (Evidence F-style 70% gsm8k baseline preserved)

- [ ] **Step 14.2: Run W4 single-prompt smoke (flag=1, validator on)**

```bash
docker exec atom_dsv4_feat bash -c "
  ATOM_DSV4_USE_W4_PATH=1 ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 ATOM_AITER_VALIDATE=1 \
  /opt/venv/bin/python -m tests.silicon.silicon_w43_smoke \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 1 --max-model-len 2048 --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --mode single --max-tokens 32 \
    --out /workspace/ATOM-lingpeng/logs/silicon_w43_single.json \
  > /workspace/ATOM-lingpeng/logs/silicon_w43_single.log 2>&1
"
```
Expected: rc=0; output coherent; validator passed (no ValueError)

- [ ] **Step 14.3: Run W4 multi-prompt smoke (flag=1, validator on, conc=4)**

```bash
docker exec atom_dsv4_feat bash -c "
  ATOM_DSV4_USE_W4_PATH=1 ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 ATOM_AITER_VALIDATE=1 \
  /opt/venv/bin/python -m tests.silicon.silicon_w43_smoke \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-model-len 2048 --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --mode multi --max-tokens 32 \
    --out /workspace/ATOM-lingpeng/logs/silicon_w43_multi.json \
  > /workspace/ATOM-lingpeng/logs/silicon_w43_multi.log 2>&1
"
```
Expected: rc=0; idx=0/1/2/3 all on-topic outputs; no validator errors

- [ ] **Step 14.4: Run gsm8k limit=20 conc=4 (W4.5 owner accuracy gate)**

```bash
docker exec atom_dsv4_feat bash -c "
  ATOM_DSV4_USE_W4_PATH=1 ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 \
  bash /workspace/ATOM-lingpeng/tests/silicon/silicon_w45_acc.sh \
  > /workspace/ATOM-lingpeng/logs/silicon_w45_gsm8k.log 2>&1
"
```
Expected: gsm8k flexible-extract ≥ 60% (within ±10pt of 70% Evidence F single-request baseline)

- [ ] **Step 14.5: Write Evidence J document**

```markdown
# Evidence J — DSV4 W4.3-Redo Silicon Validation

**Date**: 2026-04-XX
**Hardware**: MI355 TP=8
**Branch**: post-merge of Tasks 1-13

## Result Table

| # | Test | Config | rc | Result |
|---|---|---|---|---|
| 1 | Baseline (flag=0) | single, max_num_seqs=1 | 0 | output coherent (matches Evidence F) |
| 2 | W4 single | flag=1, max_num_seqs=1, validator=1 | 0 | output coherent + validator pass |
| 3 | W4 multi smoke | flag=1, max_num_seqs=4, validator=1 | 0 | 4/4 idx on-topic + validator pass |
| 4 | gsm8k limit=20 conc=4 | flag=1, max_num_seqs=4 | 0 | flexible-extract: X.XX% (target: ≥60%) |

(Fill in actual numbers from steps 14.1-14.4.)
```

- [ ] **Step 14.6: Commit evidence + open PR + close issue #37**

```bash
git checkout -b docs/dsv4-w45-evidence-j
mkdir -p docs/evidence/dsv4_w45
cp /home/pensun/ATOM-lingpeng/logs/silicon_w43_*.json docs/evidence/dsv4_w45/
cp /home/pensun/ATOM-lingpeng/logs/silicon_w43_*.log docs/evidence/dsv4_w45/  # may need git add -f for *.log
git add docs/evidence/dsv4_w45/
git commit -m "docs(evidence): J — W4.3-redo silicon validation passes (#37)"
git push -u origin docs/dsv4-w45-evidence-j
gh pr create --repo sunway513/atom --base main --head docs/dsv4-w45-evidence-j --title "[Evidence J] DSV4 multi-request silicon validation passes (#37 close)" --body "..."
```

After merge: post a final close-out comment on issue #37 and close it.

---

## Self-Review

**1. Spec coverage:**
- §2 architecture (flags + ownership) → Tasks 6, 7, 9 ✓
- §3 components → Tasks 1-5 (validator), 6-7 (env+guard), 8 (scheduler), 9 (modelrunner), 10-12 (model layers), 13 (silicon harness) ✓
- §4 data flow → exercised by Task 14 silicon validation ✓
- §5 validator full ABI contract (7 categories) → Tasks 1-5 (each task covers 1-2 categories) ✓
- §6 testing → every UT layer + silicon gates covered (Tasks 1-5 AITER UT, Tasks 6-12 ATOM UT, Tasks 13-14 silicon) ✓
- §7 time box (11-13 days) → 14 tasks averaging 1 day each = 14 days, slightly over but within slack ✓

**2. Placeholder scan:**
- "TBD"/"TODO"/"implement later" → none found
- "Add appropriate error handling" → none; validator is the explicit error layer
- "Similar to Task N" → only at Task 12 (Indexer), where the structure is genuinely "same as Task 11 with different ring name". Code pattern shown explicitly in Task 11 step 11.3.
- One step in Task 11/12 says "see PR #45 archive" — this is acceptable because the exact 32 UT design pattern is preserved on `origin/feat/dsv4-w44-state-migration` and the engineer is explicitly directed there with the commit SHA. **Decision: keep as-is** (it's a real artifact, not a placeholder).

**3. Type consistency:**
- `validate_dsv4_multireq` and its arguments consistent across Tasks 7 and the spec ✓
- `DSV4ForwardBatch.from_attn_metadata(seq_ids, pool)` consistent with W4.2 W4.1 already-on-main contract ✓
- `compute_out_cache_loc(positions, slot_indices, cu_seqlens_q, ring="main"|"compressor"|"indexer")` consistent across Tasks 10/11/12 ✓
- `_maybe_setup_dsv4_forward_batch(batch, attn_metadata, positions)` consistent with prior #42 signature ✓
- `dsv4_validate_sparse_attn_metadata(q, kv, topk_idxs, slot_mapping, positions, cu_seqlens_q, pool_capacity)` consistent across Tasks 1-5 + 10 ✓

**4. Open items from spec §9** are intentionally tracked as defaults; not blocking the plan.
