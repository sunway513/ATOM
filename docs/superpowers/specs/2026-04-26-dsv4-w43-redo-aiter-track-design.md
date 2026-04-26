# DSV4 W4.3-Redo + AITER Validator Sprint — Design Spec

**Status**: Approved (2026-04-26 brainstorm)
**Tracks**: ATOM + AITER coordinated sprint
**Targets**: Issue #37 — multi-request DSV4 inference on MI355 with good accuracy
**Replaces**: PR #42 (W4.3 model consume) which was reverted by PR #44 due to silicon HSA exception 0x1016
**Repos**: `sunway513/atom` (ATOM-side changes) and `aiter-lingpeng` (AITER-side validator under `/home/pensun/aiter-lingpeng`, the fork ATOM currently links against). Do **not** touch ROCm/atom upstream or valarLip/atom.

---

## 1 Background

The W4 stack landed structure-only PRs (#39 W4.1 ForwardBatch, #41 W4.2 DSV4KVPool) on top of a stable foundation (#40 lingpeng skeleton, #36 W3 reform, #38 Path 2 hard-assert guard). The first attempt at the model-side wiring (#42) crashed silicon during warmup with an HSA 0x1016 (out-of-bounds memory access in AITER kernels), and the warmup-only hotfix (#43) revealed the same crash on real prefill — even on `max_num_seqs=1`. Both were reverted (#44).

Root cause hypothesis: the W4 path's `out_cache_loc` / `topk_idxs` shapes did not match the AITER kernel's expected layout, but the failure mode was an unreadable GPU exception rather than a host-side error. 80 unit tests pass with mocked deps; AITER kernel runtime path was never exercised.

Owner direction (this sprint): **ATOM owns scheduling + pool metadata; AITER owns the hot path.** No more pure-Python heroics for performance. Add a debug-only AITER metadata validator that hard-fails with a readable host-side error before the kernel is invoked, so the iteration cycle goes from hours to seconds.

---

## 2 Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  ATOM (scheduling + pool metadata)         │  AITER (validator + kernels)    │
│ ─────────────────────────────────────────  │  ─────────────────────────────  │
│   Scheduler                                │  dsv4_validate_sparse_attn_     │
│      │ admit_request / finish_request      │   metadata(...)                 │
│      ▼                                     │      │                          │
│   DSV4KVPool (W4.2, on main)               │      ▼ host-side ABI checks     │
│      │ slot allocator                      │      │ — readable error msg     │
│      ▼ compute_out_cache_loc()             │      ▼                          │
│   ModelRunner._maybe_setup_dsv4_           │   sparse_attn (Python, Phase 1) │
│      forward_batch (with is_dummy_run      │      │ — kernel rewrite is S2+  │
│      gate, hotfix #43 lessons baked in)    │      ▼                          │
│      │                                     │   AITER MQA / fused MoE / ...   │
│      ▼                                     │                                 │
│   DeepseekV4Attention.forward(             │                                 │
│      x, forward_batch)                     │                                 │
│      │ ATOM_DSV4_USE_W4_PATH=1 enters W4   │                                 │
│      │ ATOM_AITER_VALIDATE=1 calls valid.  │                                 │
│      ▼                                     │                                 │
│   sparse_attn(...)                         │                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Two feature flags** are the owner-controlled levers:

| Flag | Default | Purpose |
|---|---|---|
| `ATOM_DSV4_USE_W4_PATH` | `0` | Enable W4.3-redo multi-request path. When `0`, main behaves exactly as the post-#44-revert single-request path (validated by silicon). |
| `ATOM_AITER_VALIDATE` | `0` | Debug-only host-side ABI validator before each `sparse_attn` call. Zero overhead in prod when off. |

**Path 2 hard-assert guard stays strict**. To run multi-request, the user must opt in twice: `ATOM_DSV4_USE_W4_PATH=1` AND `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1`. This double-factor protection means a typo in a config does not silently expose users to the broken path.

---

## 3 Components

| Component | File | Change |
|---|---|---|
| **AITER validator** | `aiter/dsv4_validate.py` (new in AITER repo) | `dsv4_validate_sparse_attn_metadata(q, kv, topk, slot_mapping, positions, cu_seqlens_q, pool_capacity)` — host-side checks; raise `ValueError` with a crisp message naming the constraint that failed. |
| **AITER UT** | `aiter/tests/test_dsv4_validate.py` (new) | Truth-table coverage: in-bounds / OOB topk / OOB slot / cu_seqlens_q-token mismatch / empty batch. CPU-only. |
| **ATOM W4.3-redo** | `atom/models/deepseek_v4.py` | `DeepseekV4Attention.forward(x, forward_batch)` path, gated on `ATOM_DSV4_USE_W4_PATH`. **Default falls through to single-request legacy path** (preserved bit-for-bit from post-#44-revert state). When flag=1: per-token RoPE, per-seq slot KV scatter via `DSV4KVPool`, validator call before `sparse_attn`. |
| **ATOM ModelRunner** | `atom/model_engine/model_runner.py` | `_maybe_setup_dsv4_forward_batch` rewritten with **`is_dummy_run` short-circuit at the very entry** (the gap that hotfix #43 missed). Real requests build `DSV4ForwardBatch` + admit slot. |
| **ATOM Scheduler hooks** | `atom/model_engine/scheduler.py` | Finish / preempt path notifies `pool.finish_request`. Idempotent. |
| **silicon smoke harness** | `tests/silicon/silicon_w43_smoke.py` (new) | 4-prompt conc=4 + gsm8k limit=20 conc=4. Required to be invoked with both flags on; PR description must paste rc=0 + output evidence. |

**Design principles**:

- **Backwards compatibility is not optional.** When `ATOM_DSV4_USE_W4_PATH=0` the legacy path runs unchanged. The post-revert main is the bit-for-bit baseline that any wiring change must respect.
- **Validator and `sparse_attn` share the call site.** Validator runs in the same function, one line before the kernel call. No metadata can drift between validator and kernel.
- **No state machine in the model layer.** Compressor / Indexer state lives in `DSV4KVPool` (W4.2 deliverable). The model layer fetches views and writes per-token; no cached state.
- **`is_dummy_run` short-circuit at entry.** This is the canonical fix for the warmup-vs-real-request divergence that broke #42.

---

## 4 Data Flow

```
1. Server starts:
   max_num_seqs=4
   + ATOM_DSV4_USE_W4_PATH=1
   + ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1
   + ATOM_AITER_VALIDATE=1 (debug, optional)
   → Path 2 guard: passes (unsafe override)
   → ModelRunner lazy-init DSV4KVPool

2. Warmup phase:
   → ModelRunner.warmup_model runs dummy_batch (is_dummy_run=True)
   → _maybe_setup_dsv4_forward_batch returns (None, None) — short-circuit at entry
   → DeepseekV4Attention sees forward_batch=None → legacy single-request path
   → No admit_request, no validator, no pool interaction
   → Warmup OK (canonical fix for #42 root cause)

3. Real requests arrive (4 prompts concurrent):
   → Scheduler.add_request × 4 → pool.admit_request returns slot 0/1/2/3
   → prepare_inputs builds cu_seqlens_q + positions
   → _maybe_setup_dsv4_forward_batch sees is_dummy_run=False → builds DSV4ForwardBatch
   → run_model invokes model.forward(input_ids, positions, forward_batch=fb)
   → DeepseekV4Attention sees ATOM_DSV4_USE_W4_PATH=1 → W4 path
   → ATOM_AITER_VALIDATE=1 → validator runs immediately before sparse_attn
       ├─ pass → sparse_attn (Python kernel, Phase 1) → logits
       └─ fail → ValueError with readable message (e.g.,
                  "topk_idxs[t=2,k=15]=128 ≥ kv.size(N)=12 — would
                   cause GPU OOB in sparse_attn")

4. Decode N steps: same as 3, per-token positions, slot stable across steps.

5. Completion: Scheduler.finish → pool.finish_request → slot recycled.
```

**Critical invariant**: step 2 (warmup short-circuit) must be the very first check in `_maybe_setup_dsv4_forward_batch`. The #42 / #43 / #44 chain established this: any code path that admits the dummy seqs into the pool, or builds a ForwardBatch from warmup metadata, will land synthetic shapes in the kernel and trigger HSA 0x1016.

---

## 5 Error Handling — Validator (the load-bearing piece)

```python
def dsv4_validate_sparse_attn_metadata(
    q: Tensor,            # [B, M, H, D]
    kv: Tensor,           # [B, N, D]
    topk_idxs: Tensor,    # [B, M, K] int32, -1 = skip sentinel
    slot_mapping: Tensor, # [num_tokens] long
    positions: Tensor,    # [num_tokens] long
    cu_seqlens_q: Tensor, # [num_seqs+1] int32
    pool_capacity: int,
) -> None:
    """Host-side ABI checker for DSV4 sparse_attn call site.

    Raises ValueError with crisp message on first violation.
    Set ATOM_AITER_VALIDATE=1 to enable; default off in prod.
    """
    # 1. Shape consistency (broadest catch-all)
    if q.shape[0] != kv.shape[0]:
        raise ValueError(f"q.B={q.shape[0]} != kv.B={kv.shape[0]}")

    # 2. Topk bounds — most likely #42 culprit
    valid_topk = topk_idxs[topk_idxs >= 0]  # exclude -1 sentinel
    if valid_topk.numel() > 0:
        max_idx = valid_topk.max().item()
        if max_idx >= kv.shape[1]:
            raise ValueError(
                f"topk_idxs max={max_idx} >= kv.size(N)={kv.shape[1]} "
                f"-- would cause GPU OOB in sparse_attn"
            )

    # 3. Slot mapping bounds
    if slot_mapping.numel() > 0:
        max_slot = slot_mapping.max().item()
        if max_slot >= pool_capacity:
            raise ValueError(
                f"slot_mapping max={max_slot} >= pool_capacity={pool_capacity}"
            )

    # 4. cu_seqlens_q ↔ positions consistency
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

**Design rationale**:

- **Host-side only.** The validator runs in Python on CPU. A `.item()` call is ~50 µs; that is negligible compared to the 1–2 hours saved per HSA-crash debugging cycle.
- **Zero prod cost.** When `ATOM_AITER_VALIDATE=0` (default), the validator is not called at all. No GPU sync, no `.item()`, no allocation.
- **Errors live in Python stack traces, not GPU coredumps.** The whole point of this layer is to translate "queue 0x... aborting with HSA_STATUS_ERROR_EXCEPTION code: 0x1016" into "topk_idxs max=128 >= kv.size(N)=12 in DeepseekV4Attention layer 7 step decode#3".
- **First-failure semantics.** Validator stops at the first violation rather than collecting all of them. When kernel calls happen many times per step, fast-fail is cheaper.

---

## 6 Testing

| Layer | Content | Trigger |
|---|---|---|
| **AITER UT** | `aiter/tests/test_dsv4_validate.py` — truth table for the validator (in-bounds / OOB topk / OOB slot / cu mismatch / empty batch / first-failure ordering). CPU-only. | CI on AITER PR |
| **ATOM UT** | `tests/test_deepseek_v4_w43_redo.py` (new) — register_buffer absent on the W4 path; per-token RoPE math on positions=[12,13,15,12]; per-seq slot mapping correctness; validator-fail propagation. Mocks the pool. | CI on ATOM PR |
| **silicon smoke (mandatory PR gate)** | `tests/silicon/silicon_w43_smoke.py` — single-prompt smoke (`max_num_seqs=1` + `ATOM_DSV4_USE_W4_PATH=1` + `ATOM_AITER_VALIDATE=1`). PR description must paste rc=0 + output coherence + validator pass. | Any model_runner / model.forward change PR |
| **silicon accuracy gate** | `tests/silicon/silicon_w45_acc.py` — gsm8k limit=20 conc=4 with all flags on. ≥60 % pass-rate (within ±10pt of conc=1 70 % Evidence F baseline). | W4.5 gate PR (separate, blocks DSV4 multi-request opt-in default) |
| **silicon regression baseline** | `silicon_w43_smoke.py` with `ATOM_DSV4_USE_W4_PATH=0` (default). Confirms main behavior unchanged from post-#44-revert. | Any wiring PR |

**Mandatory: the regression baseline must run before the new path on silicon, in every wiring PR.** This catches any accidental break of the legacy single-request path before flagging it as "the new path is broken".

---

## 7 Time Box

| Sprint sub-step | Content | Estimate | Blocker |
|---|---|---|---|
| **S1.1** | AITER validator + UT | 2–3 days | — |
| **S1.2** | ATOM W4.3-redo + 2 flags + ModelRunner `is_dummy_run` short-circuit | 3–4 days | S1.1 ABI definition |
| **S1.3** | silicon smoke gate (post-S1.2 commit) | 1 day | S1.2 |
| **S1.4** | conc=4 4-prompt + gsm8k limit=20 silicon validation | 1–2 days | S1.3 green |
| **Total Sprint 1** | "good accuracy + multi-request working" first wave | **7–10 days** | — |

Subsequent sprints (decoupled from S1):

- **S2** — DSV4 SWA sparse attention AITER kernel (replaces Python sparse_attn for the main ring)
- **S3** — DSV4 hybrid sparse attention (SWA + compressed in one kernel)
- **S4** — Flash Compressor
- **S5** — Lightning TopK / radix-select

S2–S5 are perf-tier; correctness is delivered by S1.

---

## 8 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Validator misses the actual #42 root cause | medium | sprint slips; need silicon iteration | Validator runs **at every** `sparse_attn` call site, not only in W4 path; can also be enabled on legacy path for diagnostics |
| AITER validator and Python kernel diverge in shape contract over time | medium | future kernel rewrite breaks validator | Validator + kernel must live in the same AITER PR going forward; document the contract in `aiter/dsv4_validate.py` docstring |
| Sub-agent claims "fallback preserved" without silicon evidence (the #42 lesson) | high (recurring) | regressions slip in | **Mandatory silicon smoke gate** in every PR description; manual paste of rc=0 + output + validator pass required |
| `is_dummy_run` flag does not actually exist on every batch path | low | warmup short-circuit ineffective | Verify by reading `ScheduledBatch` definition; if the flag is missing, add it as part of S1.2 |
| AITER kernel ABI ambiguity (e.g., does `topk_idxs` use -1 or INT_MIN as the skip sentinel?) | low | validator's check is wrong | Resolve in S1.1 by reading current `aiter/sparse_attn` source; document chosen sentinel in validator docstring |

---

## 9 Open Items (Owner Decisions)

| | Item | Default Assumption | Owner Override Path |
|---|---|---|---|
| Q2a | AITER ownership | ATOM team + lingpeng coordinate; not a hard handoff dependency | If lingpeng owns AITER side end-to-end, S1.1 timeline may compress |
| Q2b | Accuracy gate scope | gsm8k limit=20 conc=4 ≥ 60 % is "good accuracy" delivered | Owner can tighten to gsm8k 250 + humaneval if InferenceMax submission is the goal |

These are intentionally left as default assumptions in this spec. They can be tightened in a follow-up PR-comment without re-running the brainstorm.

---

## 10 References

- Issue #37 — decision record + Evidence A through I''
- PR #36 — W3 reform foundation (W1/W2/W3.1 + Codex P1)
- PR #38 — Path 2 hard-assert guard
- PR #39 — W4.1 ForwardBatch metadata scaffold
- PR #41 — W4.2 DSV4KVPool engine pool
- PR #42 — W4.3 model consume (reverted by #44)
- PR #43 — Warmup hotfix (reverted by #44)
- PR #44 — Revert PR #42 + #43 to restore foundation
- PR #45 — W4.4 state migration (closed; basis reverted, awaits W4.3-redo)
- SGLang DSV4 blog: <https://www.lmsys.org/blog/2026-04-25-deepseek-v4/>
- SGLang DSV4 PR: <https://github.com/sgl-project/sglang/pull/23600>
- vLLM DSV4 PR: <https://github.com/vllm-project/vllm/pull/40860>
- vLLM ROCm DSV4 PR: <https://github.com/vllm-project/vllm/pull/40871>
