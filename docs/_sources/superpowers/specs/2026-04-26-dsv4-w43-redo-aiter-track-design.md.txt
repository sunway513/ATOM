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

**Path 2 guard is amended to enforce the double opt-in mechanically**. The current `validate_dsv4_multireq` allows `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` alone to bypass — that lets users hit the broken legacy multi-request path without going through W4. S1 changes the guard so multi-request requires **BOTH** flags simultaneously:

- `max_num_seqs > 1` AND `UNSAFE_MULTIREQ_DEV=0`        → hard ValueError (current behavior preserved)
- `max_num_seqs > 1` AND `UNSAFE_MULTIREQ_DEV=1` AND `USE_W4_PATH=0` → **also hard ValueError** (new check; refuses the broken legacy multi-request)
- `max_num_seqs > 1` AND `UNSAFE_MULTIREQ_DEV=1` AND `USE_W4_PATH=1` → allowed
- `max_num_seqs == 1` → always allowed regardless of flags

This makes the double-opt-in enforced by the engine, not by spec convention. A typo or partial flag set raises a readable error.

---

## 3 Components

| Component | File | Change |
|---|---|---|
| **AITER validator** | `aiter/dsv4_validate.py` (new in `aiter-lingpeng`) | `dsv4_validate_sparse_attn_metadata(q, kv, topk, slot_mapping, positions, cu_seqlens_q, pool_capacity)` — full host-side ABI checker; raise `ValueError` with the failed constraint. See §5 for the full check list. |
| **AITER UT** | `aiter/tests/test_dsv4_validate.py` (new) | Truth-table for every check in §5. CPU-only. |
| **ATOM Path 2 guard amendment** | `atom/utils/dsv4_guard.py` | Modify `validate_dsv4_multireq` so it requires `ATOM_DSV4_USE_W4_PATH=1` AND `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` together to allow `max_num_seqs > 1`. Either flag alone still rejects. |
| **ATOM W4.3-redo (main attention)** | `atom/models/deepseek_v4.py` `DeepseekV4Attention` | `forward(x, forward_batch)` path gated on `ATOM_DSV4_USE_W4_PATH`. **Default falls through to single-request legacy path** (preserved bit-for-bit from post-#44-revert state). When flag=1: per-token RoPE, per-seq slot KV scatter into `DSV4KVPool` main ring, validator call before `sparse_attn`. |
| **ATOM W4.4-redo (Compressor/Indexer state)** | same file, `Compressor` + `Indexer` | Same flag (`ATOM_DSV4_USE_W4_PATH`). When flag=1: lift `kv_state` / `score_state` / `Indexer.kv_cache` out of `register_buffer`; per-token writes via `compute_out_cache_loc(..., ring="compressor"/"indexer")`; per-seq compress-boundary detection via `cu_seqlens_q[1:]-1` modular ring math (mirror SGLang `compressor.py:236`). When flag=0: legacy `register_buffer` path unchanged. **This is critical for correctness** — without it the C4/C128 compressed-state collisions that contributed to the W3.2 v3→v6.1 chain are still present, and the validator alone would not catch them. |
| **ATOM ModelRunner** | `atom/model_engine/model_runner.py` | `_maybe_setup_dsv4_forward_batch` rewritten with **`is_dummy_run` short-circuit at the very entry** (the canonical fix for the #42 root cause). Real requests build `DSV4ForwardBatch` + invoke the pool admit hook. **ModelRunner is the sole owner of pool lifetime per TP rank** (see ownership boundary below). |
| **ATOM Scheduler events** | `atom/model_engine/scheduler.py` | Emits seq lifecycle events (`on_admit(req_id)`, `on_finish(req_id)`, `on_preempt(req_id)`) via a callback registry. Does **not** call into `DSV4KVPool` directly — it only exposes the events. ModelRunner registers the pool's admit/finish methods as listeners. |
| **silicon smoke harness** | `tests/silicon/silicon_w43_smoke.py` (new) | Single-prompt smoke + 4-prompt conc=4 + gsm8k limit=20 conc=4. Both flags + validator on. PR description must paste rc=0 + output evidence. |

**Ownership boundary (P2.2 fix)**:

```
┌────────────────────────────────────────────────────────────────┐
│ Layer        │ Owns                          │ Talks to        │
├──────────────┼───────────────────────────────┼─────────────────┤
│ Scheduler    │ seq lifecycle events only     │ ModelRunner     │
│              │ (add_request, finish, preempt)│ (via callbacks) │
│ ModelRunner  │ DSV4KVPool instance per       │ Scheduler (recv │
│              │ TP rank (lazy-init);          │ events), Pool   │
│              │ ForwardBatch construction;    │ (lifecycle ops) │
│              │ feature flag dispatch         │                 │
│ DSV4KVPool   │ slot allocation, cache        │ ModelRunner     │
│              │ tensors, ring math            │ only            │
│ Model layers │ stateless compute, consume    │ ForwardBatch +  │
│              │ ForwardBatch                  │ pool views      │
└────────────────────────────────────────────────────────────────┘
```

Scheduler does not know `DSV4KVPool` exists. ModelRunner is the only layer that has both pool and scheduler references. This eliminates the leaked-slot / partial-lifecycle ambiguity that comes from each layer holding its own pool reference.

**Design principles**:

- **Backwards compatibility is not optional.** When `ATOM_DSV4_USE_W4_PATH=0` the legacy path runs unchanged. The post-revert main is the bit-for-bit baseline that any wiring change must respect.
- **Validator and `sparse_attn` share the call site.** Validator runs in the same function, one line before the kernel call. No metadata can drift between validator and kernel.
- **State out of `register_buffer` for both main attention AND compressor/indexer.** S1 covers all three; deferring compressor/indexer to S2 would mean the `flag=1` path is still partly broken and S1 cannot honestly claim "multi-request working".
- **`is_dummy_run` short-circuit at entry.** This is the canonical fix for the warmup-vs-real-request divergence that broke #42.
- **One-owner ownership.** Pool lifetime: ModelRunner. Slot lifecycle events: Scheduler. No layer owns more than one of these.

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
    topk_idxs: Tensor,    # [B, M, K] int32, -1 = skip sentinel (the only
                          #                                   allowed negative)
    slot_mapping: Tensor, # [num_tokens] long, must be >= 0
    positions: Tensor,    # [num_tokens] long, must be >= 0
    cu_seqlens_q: Tensor, # [num_seqs+1] int32, monotonic, [0]==0
    pool_capacity: int,
) -> None:
    """Host-side ABI checker for DSV4 sparse_attn call site.

    Raises ValueError with the first failed constraint. Set
    ATOM_AITER_VALIDATE=1 to enable; default off in prod.

    Full ABI contract (any failure → readable host-side error, never
    HSA exception). The kernel side may assume all of these are true
    once the validator returns successfully.
    """
    # ---- 1. Tensor shape & dtype contract ---------------------------------
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

    # ---- 2. Dtype contract -------------------------------------------------
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

    # ---- 3. Device & contiguity --------------------------------------------
    dev = q.device
    for name, t in (
        ("kv", kv), ("topk_idxs", topk_idxs), ("slot_mapping", slot_mapping),
        ("positions", positions), ("cu_seqlens_q", cu_seqlens_q),
    ):
        if t.device != dev:
            raise ValueError(f"{name}.device={t.device} != q.device={dev}")
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    # ---- 4. Topk index domain (most likely #42 culprit) -------------------
    if topk_idxs.numel() > 0:
        # 4a. Only -1 is the allowed negative sentinel
        below_sentinel = (topk_idxs < -1).any().item()
        if below_sentinel:
            raise ValueError(
                "topk_idxs contains values < -1 (only -1 is the skip sentinel)"
            )
        # 4b. Upper bound: any non-sentinel index must be in [0, kv.N)
        valid_topk = topk_idxs[topk_idxs >= 0]
        if valid_topk.numel() > 0:
            max_idx = valid_topk.max().item()
            if max_idx >= kv.shape[1]:
                raise ValueError(
                    f"topk_idxs max={max_idx} >= kv.size(N)={kv.shape[1]} "
                    f"-- would cause GPU OOB in sparse_attn"
                )

    # ---- 5. Slot mapping domain --------------------------------------------
    if slot_mapping.numel() > 0:
        if (slot_mapping < 0).any().item():
            raise ValueError("slot_mapping contains negative ids")
        max_slot = slot_mapping.max().item()
        if max_slot >= pool_capacity:
            raise ValueError(
                f"slot_mapping max={max_slot} >= pool_capacity={pool_capacity}"
            )

    # ---- 6. Positions domain ----------------------------------------------
    if positions.numel() > 0:
        if (positions < 0).any().item():
            raise ValueError("positions contains negative values")

    # ---- 7. cu_seqlens_q monotonicity & token ownership -------------------
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

**The validator is a contract, not just a fence.** Once it returns successfully, the AITER kernel may assume:

1. `q` is a 4-D contiguous tensor on the same device as all metadata
2. `kv` is a 3-D contiguous tensor with matching `B` and `head_dim`
3. `topk_idxs` is `[B, M, K]` int32, contiguous, contains only `-1` sentinels or valid in-range indices into `kv.size(N)`
4. `slot_mapping` is non-negative, contiguous, and within `[0, pool_capacity)`
5. `positions` is non-negative, contiguous
6. `cu_seqlens_q` starts at 0, is monotonically non-decreasing, and its last element equals `positions.numel()`
7. `slot_mapping.numel() == positions.numel()`

Any AITER kernel deviating from these assumptions is a kernel bug and must be fixed there, not papered over in the validator.

**Design rationale**:

- **Host-side only.** The validator runs in Python on CPU. A `.item()` call is ~50 µs; that is negligible compared to the 1–2 hours saved per HSA-crash debugging cycle.
- **Zero prod cost.** When `ATOM_AITER_VALIDATE=0` (default), the validator is not called at all. No GPU sync, no `.item()`, no allocation.
- **Errors live in Python stack traces, not GPU coredumps.** The whole point of this layer is to translate "queue 0x... aborting with HSA_STATUS_ERROR_EXCEPTION code: 0x1016" into "topk_idxs max=128 >= kv.size(N)=12 in DeepseekV4Attention layer 7 step decode#3".
- **First-failure semantics.** Validator stops at the first violation rather than collecting all of them. When kernel calls happen many times per step, fast-fail is cheaper.

---

## 6 Testing

| Layer | Content | Trigger |
|---|---|---|
| **AITER UT** | `aiter/tests/test_dsv4_validate.py` — truth table for **every** check in the §5 contract: shape (1), dtype (2), device/contiguity (3), topk domain (4a-b), slot domain (5), positions (6), cu monotonicity & token-ownership (7). CPU-only. | CI on AITER PR |
| **ATOM UT — main attention W4 path** | `tests/test_deepseek_v4_w43_redo.py` (new) — `register_buffer` absent for `kv_cache` on W4 path; per-token RoPE math on positions=[12,13,15,12]; per-seq slot mapping; validator-fail propagation. Mocks the pool. | CI on ATOM PR |
| **ATOM UT — Compressor / Indexer W4 path (W4.4 slice)** | `tests/test_deepseek_v4_w44_state_redo.py` (new; revives 32-UT pattern from closed PR #45) — `register_buffer` absent for `kv_state`/`score_state`/`Indexer.kv_cache` on W4 path; per-seq compress-trigger via `cu_seqlens_q[1:]-1` (positions [12,13,15,12], ratio=4, only token at pos=15 fires); state scatter into pool's compressor + indexer rings. | CI on ATOM PR |
| **ATOM UT — Path 2 guard amendment** | `tests/test_dsv4_multireq_guard.py` (extend) — new truth-table rows: `UNSAFE=1, USE_W4=0` rejects, `UNSAFE=1, USE_W4=1` allowed, `UNSAFE=0, USE_W4=1` rejects. Drift-catcher integration test still asserts the helper module is the source of truth. | CI on ATOM PR |
| **ATOM UT — Scheduler events + ModelRunner ownership** | `tests/test_modelrunner_dsv4_pool_lifecycle.py` (new) — Scheduler emits events without `DSV4KVPool` import; ModelRunner subscribes pool's admit/finish; finish path is idempotent under preempt-then-finish race. Mocks. | CI on ATOM PR |
| **silicon regression baseline (BLOCKING)** | `silicon_w43_smoke.py` with `ATOM_DSV4_USE_W4_PATH=0` (default). Confirms main behavior unchanged from post-#44-revert. **Must run before the new-path test in every wiring PR.** | Any wiring PR |
| **silicon smoke (mandatory PR gate)** | `tests/silicon/silicon_w43_smoke.py` — single-prompt smoke (`max_num_seqs=1` + `ATOM_DSV4_USE_W4_PATH=1` + `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` + `ATOM_AITER_VALIDATE=1`). PR description must paste rc=0 + output coherence + validator pass. | Any model_runner / model.forward / Compressor / Indexer change PR |
| **silicon multi-request smoke** | same harness with `--max-num-seqs 4 --num-prompts 4`. Same flags. PR description must paste rc=0 + 4 idx outputs all on-topic. | S1.4 acceptance |
| **silicon accuracy gate (W4.5 owner gate)** | `tests/silicon/silicon_w45_acc.py` — gsm8k limit=20 conc=4 with all flags on. ≥60 % pass-rate (within ±10pt of conc=1 70 % Evidence F baseline). | W4.5 gate PR (separate, blocks DSV4 multi-request opt-in default) |

**The regression baseline (legacy path with flag=0) is the load-bearing test.** Without it we cannot tell whether a silicon failure is "new path broken" or "we broke the legacy path too" — exactly the ambiguity that made #42 / #43 / #44 expensive to diagnose.

---

## 7 Time Box

| Sprint sub-step | Content | Estimate | Blocker |
|---|---|---|---|
| **S1.1** | AITER validator + full UT (every §5 check) | 3 days | — |
| **S1.2** | ATOM Path 2 guard amendment + UT (drift-catcher refresh) | 0.5 day | — |
| **S1.3** | ATOM W4.3-redo (main attention) + UT + ModelRunner `is_dummy_run` short-circuit + Scheduler event registry | 3 days | S1.1 ABI |
| **S1.4** | ATOM W4.4-redo (Compressor/Indexer state migration) + UT — revives PR #45 design pattern | 2 days | S1.3 |
| **S1.5** | silicon regression baseline + single-prompt smoke (`max_num_seqs=1` + flags) | 1 day | S1.3+S1.4 commits |
| **S1.6** | silicon multi-request smoke (conc=4 4-prompt) — flags on | 1 day | S1.5 green |
| **S1.7** | silicon accuracy gate (gsm8k limit=20 conc=4 ≥60%) — Evidence J | 1 day | S1.6 green |
| **Total Sprint 1** | "good accuracy + multi-request working" first wave | **11–13 days** | — |

(Time box widened from 7-10 to 11-13 days to absorb the W4.4 slice that P1.2 review surfaced. Honest accounting.)

Subsequent sprints (decoupled from S1):

- **S2** — DSV4 SWA sparse attention AITER kernel (replaces Python `sparse_attn` for the main ring)
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
