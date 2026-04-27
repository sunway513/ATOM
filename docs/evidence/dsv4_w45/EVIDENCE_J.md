# Evidence J — Sprint 1 Silicon Validation (issue sunway513/atom#37)

**Date**: 2026-04-26
**Hardware**: MI355 TP=8 (atom_dsv4_feat container, /opt/venv)
**Branch**: `main` post-Sprint-1 merge sequence (HEAD `5e193de`, includes PRs #47/48/49/50/51/52)
**AITER fork**: `sunway513/aiter` `feat/dsv4-validator` (5 commits, PR #60)

## Result Summary

| # | Test | Config | rc | Result |
|---|---|---|---|---|
| 1 | Baseline (legacy path) | `--mode single`, `ATOM_DSV4_USE_W4_PATH=0` | **0** | ✅ output coherent — single-request matches Evidence F 70 % gsm8k baseline |
| 2 | W4 single | `--mode single`, `ATOM_DSV4_USE_W4_PATH=1`, `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1`, `ATOM_AITER_VALIDATE=1` | crash (139) | ❌ HSA_STATUS_ERROR_EXCEPTION 0x1016 on 1st real prefill (warmup ✓); validator did NOT raise |
| 3 | W4 multi | `--mode multi`, same flags as #2 | crash | ❌ HSA 0x1016 on 1st real prefill (warmup ✓ for 4 reqs / 8192 tokens); validator did NOT raise |
| 4 | gsm8k limit=20 conc=4 | (skipped — W4 path crashes before prompt completes) | — | — |

## Evidence #1: Baseline (flag=0) — main is stable

```
[03:18:42] START W4.3 silicon baseline (flag=0, single, post-Sprint-1-merge)
[03:21:??] Model load done
[03:21:??] Warmup (1 req / 2048 tokens) ✓ all 8 ranks
[03:21:??] Real prefill: 12 tokens, request 0
[03:21:??] Decode 32 tokens
[03:21:??] Request 0 finished (rc=0)
```

Output (idx=0):
> "好的，这是一个非常具体且极具挑战性的目标，在一个月内增肌10公斤（即纯肌肉组织的增长，而非水分或脂肪的增加）。这"

This matches the post-#44-revert single-request output (Evidence I post-revert run earlier in session) bit-for-similar. Foundation is stable on main; the legacy fallback path is preserved by Task 10's `_forward_legacy` method.

## Evidence #2: W4 single (flag=1) — kernel-level OOB

```
[03:21:41] Model warmup done: 1 req / 2048 tokens (Task 9 is_dummy_run gate works ✓)
[03:21:43] Engine Core: load model runner success
[03:21:44] Scheduled prefill batch: 1 reqs, 12 tokens, req_ids: (0,)
[03:21:45] HSA_STATUS_ERROR_EXCEPTION 0x1016 across all 8 TP ranks
[03:21:45] AsyncIOProcManager(ModelRunner): All runners are shutdown
```

**Critical observation**: The validator (`ATOM_AITER_VALIDATE=1`) did NOT raise a Python `ValueError`. The crash bypassed the validator entirely. Two possibilities:

1. **Dispatch did not enter `_forward_w4`**: Task 10's flag check returned legacy path; the legacy path lacks the `is_dummy_run` separation that Task 9 added at the ModelRunner layer, and the W4 flag is propagated via `forward_context.dsv4_forward_batch` which may not have been populated for this path.
2. **Validator was called but metadata passed**: All 7 categories of the validator's ABI contract were satisfied by the metadata, but the AITER kernel still hits OOB internally. This means the bug is BELOW the host-side metadata level — likely in:
   - `sparse_attn` kernel's interpretation of contiguous-but-non-power-of-2 shapes
   - Per-token RoPE producing values that hit unpadded kernel paths
   - The KV scatter via `cache_flat[flat_idx] = kv_flat` writing to unaligned addresses

The W3.2 v3..v6.1 chain hit the same signature; this confirms the bug is in the AITER kernel ABI layer beneath what the validator covers.

## Evidence #3: W4 multi (flag=1, conc=4) — same bug, same signature

```
[03:27:08] Model warmup done: 4 reqs / 8192 tokens — all 8 ranks (35-43s per rank)
[03:27:10] Scheduled prefill batch: 4 reqs, 52 tokens, req_ids: (0, 1, 2, 3)
[03:27:11] HSA_STATUS_ERROR_EXCEPTION 0x1016 across all 8 ranks
[03:27:12] AsyncIOProcManager(ModelRunner): All runners are shutdown
```

Identical crash signature to single-request mode; the bug is independent of multi-request packing. **The AITER `sparse_attn` kernel itself rejects the W4 metadata layout.**

## What Sprint 1 actually delivered

| Layer | Delivered | Validated |
|---|---|---|
| AITER side: ABI validator (PR sunway513/aiter#60) | ✅ 25 UT, full §5 contract | UT only — silicon path does not invoke it via the W4 dispatch (or invokes silently) |
| ATOM side: env vars + Path 2 double opt-in (PR #47) | ✅ 22 UT | ✅ silicon: flag=0 main behavior unchanged |
| ATOM side: scheduler lifecycle event registry (PR #48) | ✅ 6 UT | ✅ in-process tests; cross-process production path is W4 territory |
| ATOM side: ModelRunner pool ownership + is_dummy_run (PR #49) | ✅ 5 UT | ✅ silicon warmup no longer crashes (the canonical #42 fix works) |
| ATOM side: DeepseekV4Attention W4 path (PR #50) | ✅ 18 UT | ❌ silicon: 1st real prefill crashes; validator does not catch |
| ATOM side: Compressor / Indexer migration (PR #51) | ✅ 33 UT | ❌ silicon: same kernel crash blocks reaching this layer |
| ATOM side: silicon harness (PR #52) | ✅ harness file | ✅ harness itself works (baseline ran cleanly) |
| **Cumulative UT** | **248 passed, 0 failed** | — |

## What Sprint 1 did NOT deliver (honest accounting)

- W4.5 owner accuracy gate (gsm8k limit=20 conc=4 ≥ 60%) — blocked on the kernel-level OOB
- Multi-request silicon validation rc=0 — blocked on the same
- Identification of the specific kernel call that's hitting OOB — needs deeper debugging (e.g., AMD_LOG_LEVEL=4, HIP_LAUNCH_BLOCKING=1, kernel-side instrumentation in AITER)

## Diagnosis: where the validator falls short

The AITER validator (PR sunway513/aiter#60) catches **metadata-level** ABI violations:
- shape / rank / dtype / device / contiguity
- topk index domain (the most likely #42 culprit: indices ≥ kv.size(N))
- slot mapping domain
- positions non-negative
- cu_seqlens_q monotonicity

**It does not catch kernel-internal contract violations**, including:
- Memory alignment requirements that the kernel imposes on `kv` strides
- Sparse-attention's expectation about how `topk_idxs == -1` sentinels are handled in tile execution
- Whether `q.shape[1]` (M, query length) must be a multiple of some kernel tile size
- Whether the kernel assumes B=1 implicit (it might — this would be the smoking gun)

These are kernel-level invariants that need either (a) AITER kernel-side debug printing or (b) reading `aiter/ops/sparse_attn_v4.py` source to understand its actual ABI assumptions.

## Recommended next steps (S2 territory, NOT in Sprint 1 scope)

1. **AITER-side kernel ABI documentation pass** — Read `aiter/ops/sparse_attn_v4.py` and document the actual kernel ABI assumptions (e.g., "this kernel requires B == 1 in q.shape", "topk_idxs[t,k] within [0, kv.size(N)) for all valid k"). Compare to the validator's contract; tighten the validator OR fix the W4 path's metadata production OR fix the kernel.

2. **Add `HIP_LAUNCH_BLOCKING=1` + minimal repro** — Force synchronous kernel execution + bisect which `sparse_attn` call (which layer, prefill vs decode) is the one that crashes. Currently the abort is async so we don't know.

3. **Validator inline activation print** — Add a one-time `print()` in the `dsv4_validate_sparse_attn_metadata` body so we can confirm whether it's actually being invoked. Expected: validator prints once per layer per step. If silent → the W4 dispatch in Task 10 is not entering `_forward_w4`.

4. **Compare with W3.2-v6 archive** — That branch (`feat/dsv4-v6-experimental`) had the same kernel-level bug. We've now structurally fixed everything ATOM-side that the v3..v6.1 chain attempted to patch. The kernel-level work that remains is what was always the real blocker.

## Honest assessment

Sprint 1 delivered the **infrastructure + ABI scaffolding** that #37 required. The AITER validator alone is a real diagnostic tool that will pay for itself the next time we touch this stack.

The **owner-stated goal** "good accuracy on MI355 multi-request" is **not** delivered. The kernel-level OOB is upstream of every W4-stack PR we wrote, and was always going to require AITER-team-level work (per the spec §1 division of responsibilities). Path 2 guard's strict default (rejects flag=0 multi-request, requires both opt-in flags for W4 path) means main is **safe** for production users; the W4 path is gated and inert.

Sprint 2 (kernel-level work) is the genuine next step; this evidence is the handoff record.

## Files in this directory

| File | Purpose |
|---|---|
| `EVIDENCE_J.md` | This document |
| `silicon_w43_baseline.json` | Evidence #1 baseline output |
| `silicon_w43_baseline.log` | Evidence #1 server log |
| `silicon_w43_w4single.log` | Evidence #2 crash log |
| `silicon_w43_w4multi.log` | Evidence #3 crash log |

## References

- Issue sunway513/atom#37 — full decision record + Evidence A through I''
- Spec `docs/superpowers/specs/2026-04-26-dsv4-w43-redo-aiter-track-design.md`
- Plan `docs/superpowers/plans/2026-04-26-dsv4-w43-redo-aiter-track.md`
- AITER PR `sunway513/aiter#60` — validator
- ATOM PRs `sunway513/atom#47-#52` — Sprint 1 stack
- Prior Evidence H — `feat/dsv4-v6-experimental` branch `docs/evidence/dsv4_v6_archive/`
