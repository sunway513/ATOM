# Plan: ATOM DSV4 Full Functionality Closure (Sprint 6)

**Date:** 2026-04-27
**Issue:** sunway513/atom#37 (W4.5+ — full DSV4 functional support)
**Owner:** code agent + user review gate

## Goal

> **"Get to the point where we feel comfortable claiming ATOM has full functionality support for DeepSeek V4."** — user, 2026-04-27

This is a **functional-correctness sprint**, not a perf sprint. Acceptance is binary: ATOM either matches the upstream reference behavior end-to-end (all 4 axes below) or it does not. No partial-credit "0.60 vs 0.96 declared production" claims like Sprint 5e.

## Pre-sprint state of the world (post-Sprint 5e)

Confirmed via 3 silicon experiments + 4 sub-agent comparison reports (DSV4 paper, SGLang code, vLLM code, ATOM W4 audit):

| Axis | Current ATOM state | Source-of-truth (paper / SGLang / vLLM) |
|---|---|---|
| KV cache architecture | Per-ratio slabs (`_compressor_state_c4`, `_compressor_state_c128`, `_compressor_main_kv_c4`, `_compressor_main_kv_c128`, `_indexer_cache`) — **matches paper §3.6.1 two-pool design** | Paper: classical paged KV + state cache slab. SGLang/vLLM use unified buffers as *shim* for their MLA framework. |
| Sparse Indexer (CSA `m=4`) | Implemented in `deepseek_v4.py` Indexer (`index_topk` reads from config, defaults 1024 = V4-Pro) | Paper §2.3.1 lightning indexer. SGLang has equivalent NSA Indexer. vLLM: not yet. |
| Compressor (HCA `m'=128`) | Per-token block-boundary loop (post Sprint 4 commit `8fa0129`) | Paper §2.3.2. SGLang: equivalent. vLLM: no analog. |
| MTP layer | `compress_ratios[-1] == 0` → SWA-only handling | Paper §2.1: 1 MTP block, no compressor. ✅ ATOM matches. |
| Multi-request KV pool | LIFO slot allocator, max=512, gated behind `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` | Paper: per-request slabs ✅. SGLang/vLLM: paged + slot_mapping (unconditionally enabled). |
| Scheduler↔Pool finish-pipeline | Cross-process via `ScheduledBatch.finished_seq_ids` (Sprint 4 fix) | Equivalent to vLLM/SGLang's batch lifecycle ✅ |
| MoE FP4 backend | `Mxfp4MoEMethod` with CK/FlyDSL/Triton dispatch — **CK/FlyDSL produces 0.45/0.00, Triton produces 0.60/0.60 at conc=1 5-shot** | SGLang: single `QuantType.per_1x32` flashinfer/cutlass. vLLM: FlashInfer CUTLASS / Marlin. ATOM is the outlier. |

## Diff candidates (3-way vs paper) — pre-validated

**P0 (functional blockers)**:

1. **0-shot prompts produce garbled output even with Triton MoE** (silicon-verified 2026-04-27 0/4 raw 0-shot prompts → loop / off-topic / nonsense). 5-shot context masks it (3/4 sensible). Likely root causes (top 3, not yet bisected):
   - Sparse Indexer top-k window calculation may have an off-by-one when context is too short (Sprint 4 only fixed warm path)
   - FP8 KV cache scale handling in CSA layers — paper §2.3.4 requires "BF16 for RoPE dims, FP8 for the rest, FP4 for indexer"; ATOM may quantize uniformly
   - Triton MoE precision — 36pp gap to SGLang's 0.96 even at 5-shot suggests Triton kernel is lossy
2. **Multi-request batched decode (conc≥4) untested with Triton** — Sprint 4's 4-distinct-output evidence was on CK backend (1/4 fluent). Sprint 5e Triton lm_eval was conc=1 only. Need silicon validation at conc∈{4, 8}.
3. **Multi-request gated behind `ATOM_DSV4_UNSAFE_MULTIREQ_DEV` flag** (`atom/utils/dsv4_guard.py:59`). Until conc>1 is silicon-validated, this is correct, but the goal is to remove the flag for production.

**P1 (correctness with low blast radius)**:

4. **CK/FlyDSL fused MoE layout dispatch bug** (Sprint 5b/5c RCA). Triton bypasses it but at higher latency. Long-term should fix in `aiter_lingpeng/aiter/ops/flydsl/moe_kernels.py` so production has a fast path.
5. **MLA-class reuse risk** — V4 uses MQA-with-shared-KV, not MLA. Audit `atom/plugin/attention_mla.py` and `attention_mla_sparse.py` for any MLA-style K/V split assumption that might silently bias V4 outputs.

**P2 (cosmetic / external-readability)**:

6. **Naming**: rename `c4`/`c128` → `csa`/`hca` to match paper §2.3 vocabulary. Rename `ATOM_DSV4_USE_W4_PATH` → `ATOM_DSV4_PAGED_PATH` (or similar — `W4` is internal sprint name, confuses external readers). Defer to a separate housekeeping PR.

## Acceptance gates

To call DSV4 "full functionality support" we must close every row below. Each row has a specific silicon command and pass criterion.

| # | Test | Pass criterion | Silicon cmd |
|---|---|---|---|
| 1 | gsm8k limit=20, conc=1, 5-shot, Triton | flexible-extract ≥ 0.65 (exclude noise: ≥ SGLang_ref - 0.30) | `lm_eval --tasks gsm8k --num_fewshot 5 --limit 20 ...` |
| 2 | gsm8k full (n=1319), conc=1, 5-shot, Triton | flexible-extract ≥ 0.85 (tight stderr) | same with `--limit ""` |
| 3 | conc=4 sensible smoke, Triton + UNSAFE_MULTIREQ | 4/4 distinct prompts produce sensible answers | 4 parallel curls — script ready at `/home/pensun/ATOM-lingpeng/conc4_triton.sh` |
| 4 | conc=8 sensible smoke, Triton + UNSAFE_MULTIREQ | 8/8 distinct prompts produce sensible answers | extend (3) to 8 prompts |
| 5 | gsm8k limit=20, conc=4 batched | flexible-extract ≥ 0.55 (allow 10pp regression vs conc=1) | `lm_eval --num_concurrent 4` against `--max-num-seqs 4` server |
| 6 | 0-shot 5-question pop quiz (math + factual + code + summary + list) | ≥ 4/5 sensible answers (not loop, not off-topic, model-recognized format) | manual curl battery |
| 7 | Side-by-side vs SGLang docker | Same 5-shot prompt → both ATOM & SGLang produce a "correct" answer per gsm8k strict-match for ≥ 18/20 prompts | `docker run lmsysorg/sglang:deepseek-v4-b300-dev` (B300 — code-readable but won't run on MI355X; use SGLang's published API endpoint or a hosted instance instead for runtime parity) |
| 8 | Per-layer numerics: ATOM vs paper-defined torch reference (`ATOM_V4_TORCH_MOE=1`) | Per-layer L2 norm ratio ≥ 0.95, per-token logit cosine ≥ 0.99 | new diagnostic harness |
| 9 | Continuous batching stability (1 hour, conc=4, gsm8k stream) | No KV pool slot exhaustion, no rank crash, no OOM | endurance silicon run |

## Plan-of-attack ordering

Must respect blast radius — start with smallest diagnostics and only commit code changes after the bug is localized.

### Phase A — Diagnose (no code commits yet)

A1. **Run gate #6 immediately**. Already partially done (0/4 raw 0-shot today). Expand to 5-question battery on the *currently running* Triton server. Establish baseline 0-shot quality number.

A2. **Run gate #8 — torch-ref comparison**. Single curl with `ATOM_V4_TORCH_MOE=1`, dump per-layer KV state via instrumentation hook, compare layer-by-layer to fused-kernel server. This isolates which layer first diverges.

A3. **Audit `attention_mla*.py` for V4 misuse**. Read `atom/plugin/attention_mla.py:*`, `attention_mla_sparse.py:*` and `atom/model_ops/attention_mla.py` (if exists), check whether V4 path goes through any MLA-style decompressed K/V split. Output: file:line diff list.

A4. **Audit FP8 KV scale handling per paper §2.3.4**. Check `atom/quantize/quant_v4.py` and `dsv4_pool.py` for whether RoPE dims stay BF16 or get quantized along with the rest. Paper requirement is non-uniform.

### Phase B — Fix (one bug per commit)

Each commit must have a silicon validation showing the gate it closes. No multi-bug commits like Sprint 4.

B1. Apply A2's localized fix → silicon validate gate #8 ≥ 0.95.
B2. Apply A3's fix (if MLA misuse confirmed) → silicon validate gate #6 ≥ 4/5.
B3. Apply A4's fix (if FP8 mis-quant confirmed) → silicon validate gate #2 ≥ 0.85.

### Phase C — Multi-request validation

C1. Silicon gate #3 (conc=4 smoke) with all phase-B fixes applied. If 4/4 → gate #4 (conc=8). If <4/4 → bisect with binary-search prompt sets.
C2. Silicon gate #5 (lm_eval conc=4). Pass → drop `ATOM_DSV4_UNSAFE_MULTIREQ_DEV` flag from `dsv4_guard.py`.

### Phase D — Endurance + parity

D1. Gate #9 (1-hour stream).
D2. Gate #7 (vs SGLang). May require pulling SGLang DSV4 instance or using their hosted endpoint.

### Phase E — Cosmetic + PR

E1. Rename pass (CSA/HCA, `ATOM_DSV4_PAGED_PATH`).
E2. Update `recipes/DeepSeek-V4-Pro.md` to reflect actual production config (drop or keep `ATOM_USE_TRITON_MOE=1` based on whether B-phase fixes the CK/FlyDSL path).
E3. Open PR #59 successor or merge into existing #59.

## Out of scope for Sprint 6 (defer)

- Closing the remaining 36pp gap to SGLang B300 (gate #2 = 0.85 vs SGLang 0.96 still leaves 11pp). Acceptable for "full functional support" claim — perf parity is Sprint 7.
- Speculative decoding / MTP integration. Paper says MTP block exists; ATOM's `n_mtp_layers=1` reads it but `--method mtp` integration with the W4 path is untested. Keep behind a separate flag.
- Long-context (>4K) silicon validation. All Sprint 5/6 testing was max-model-len=4096. Paper claims million-token; that's a separate validation campaign.

## What I'm NOT going to do without explicit user approval

- Rename anything in user-visible config / env vars (P2 cosmetic only at the end).
- Drop `ATOM_DSV4_UNSAFE_MULTIREQ_DEV` flag until gate #5 passes.
- Merge to main / mark PR #59 ready-for-review until all P0 gates close.
- Touch `aiter_lingpeng/` (CK/FlyDSL kernels) — that's AITER team scope and a separate PR.

## Estimated cost

- Phase A: 4-6 silicon hours (3 diagnostics × ~1.5h each including server boot/JIT)
- Phase B: variable — depends on what A finds. If MLA misuse: 4h. If FP8 quant: 2h. If layer-specific kernel bug: 8-16h.
- Phase C: 3-4 silicon hours
- Phase D: 6 hours (1h endurance + 5h SGLang parity setup)
- Phase E: 1 hour
- **Total**: 18-32 silicon hours, ~2-4 calendar days with parallel agent dispatch.

## Plan-revision log

### v1 — 2026-04-27 (this draft)

Initial plan based on:
- 4 sub-agent reports completed 2026-04-27 (paper, SGLang, vLLM, ATOM audit)
- Silicon evidence: Sprint 4 (multi-conc CK 1/4 fluent), Sprint 5e (Triton conc=1 5-shot 0.60), Sprint 5f-on-the-fly (Triton conc=1 0-shot 0/4)
- Confirmed: ATOM's per-ratio slab design IS paper-faithful (not a divergence to fix). The bugs are in the implementation details (MoE backend, possibly MLA reuse, possibly FP8 quant non-uniformity).

Submitted to user for review. Will not start phase A diagnostics until user signs off on this plan or asks for revisions.

## Plan-revision log

### v2 — 2026-04-27 (post Phase A audit)

Phase A audit results landed:
- A0 paper truth: ATOM's per-ratio slab design is paper-faithful (paper §3.6.1 two-pool). NO architecture change needed. Naming (CSA/HCA vs c4/c128) is cosmetic-only.
- A3 MLA reuse audit: NEGATIVE. V4 uses own `DeepseekV4Attention`, never enters MLAAttention. Optional guard in MLAAttention.__init__ recommended for future-proofing only.
- **A4 KV quantization audit: TWO CONFIRMED BUGS** in `dsv4_pool.py` matching paper §2.3.4 violations.

Plan v2 changes:
- Add **Phase B0** (highest priority code change) to fix A4.1 (main KV uniform dtype) + A4.2 (Indexer non-FP4).
- B0a (Indexer FP8 storage): **LANDED commit `a8e3a02`** — 16 LOC, opt-in via `ATOM_DSV4_INDEXER_FP8=1`. 8 new + 23 legacy unit tests pass. No model code change.
- B0b (Main KV nope/rope split): **DESIGNED, DEFERRED**. ~200 LOC across 6 sub-commits (pool dual-slab + write helper + W4 path migration + legacy migration + wiring + tests). Per plan rule "each commit silicon-validated", we wait for B0a silicon validation before launching B0b.
- A1 silicon battery + A2 torch-ref deferred until user's v9a completes (silicon currently busy).

### B0b deferred-design summary (full in Plan agent report)

| Sub-commit | Files | LOC | Tests after |
|---|---|---|---|
| B0b.1 Pool config + dual-slab alloc | `dsv4_pool.py:127, 263-322, 680, 710` | ~70 | 23 legacy + 1 new pass |
| B0b.2 Pool write helper `write_main_kv` | `dsv4_pool.py` (new method) | ~35 | + roundtrip test |
| B0b.3 Model W4 write site migration | `deepseek_v4.py:1932-1966, 2027` | ~25 | + W4 path tests |
| B0b.4 Legacy path migration | `deepseek_v4.py:1760-1778` | ~30 | + W43_redo tests |
| B0b.5 Env var + wiring | `envs.py` + `model_runner.py` | ~12 | + envs test |
| B0b.6 Test suite | `tests/test_dsv4_pool_main_kv_split.py` (new) | ~180 | 41 total pool tests |

**B0b GO/NO-GO criterion** (silicon-driven):
- gsm8k 5-shot limit=20 with B0a only ≥ 0.85 → REJECT (B0a sufficient)
- gsm8k 5-shot limit=20 with B0a only 0.65-0.84 → GO (need both halves of A4)
- gsm8k 5-shot limit=20 with B0a only ~0.49 (no improvement) → DEFER (root cause elsewhere; Phase C / D)

### Pre-B0b silicon validation gate (B0d) — pending

When user's v9a server frees silicon, run:
```bash
ATOM_USE_TRITON_MOE=1 ATOM_DSV4_USE_W4_PATH=1 USE_W4_PATH=1 \
ATOM_DSV4_INDEXER_FP8=1 \
  python -m atom.entrypoints.openai_server ...
# then lm_eval gsm8k limit=20 num_fewshot=5 max_gen_toks=1024
# also: 5-question 0-shot battery (test_dsv4_pool_battery.sh — pending)
```

Compare to v8 baseline (Triton without indexer FP8): flexible 0.60 / strict 0.60.
