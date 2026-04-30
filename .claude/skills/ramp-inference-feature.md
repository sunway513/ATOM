---
name: ramp-inference-feature
description: Bringing a NEW fundamental feature into ATOM (or any inference engine) — new model architecture, new KV cache scheme, new attention path, new quantization. Use when the task is "build feature X end-to-end so it actually works at the production claim", NOT when the task is "fix a known bug". For bug-fixing on an existing path, use `dump-bisect-debug` instead.
version: 1.0.0
scope: ATOM (generalizable to vLLM / SGLang / any serving stack with custom kernels)
last_updated: 2026-04-30
companion_skills:
  - dump-bisect-debug   # forward numerical bisect, used inside Phase 4 below
  - atom-patterns       # ATOM-specific conventions (block_table, FusedMoE, ParallelLinear)
---

## Why this skill exists

A new feature in an inference engine is NOT a debugging exercise. It's an architecture exercise that ends at "production claim is silicon-validated". The dominant failure mode is treating it as "patch the existing skeleton until tests pass" — that maximises silicon iterations, locks you into prior wrong assumptions, and produces a fragile path that the next feature will inherit.

The 2026-04-27/30 DSV4-Pro multi-request KV cache work made this concrete:
- **Patcher (5+ silicon hours, 0/4 conc=2 smoke at end)**: 4 candidate root causes (slot lookup, per-seq Indexer loop, GEMM m-pad at two paths, model-level pad), all reverted. Bisect harness added at hour 5+, contaminated by TP-rank race.
- **Rampers (one push, 4/4 conc=4 smoke + gsm8k 0.90)**: rewrote the V4 attention as a dedicated module per paper §3.6.1, used the standard `block_table` primitive, landed multi-seq dispatch as a per-seq Python loop, and shipped the bisect harness (`atom/utils/debug_helper/`) BEFORE the model code.

Same problem, ~50× output ratio. This skill captures what made the difference.

## The five rules

1. **Reference impl is Phase 0.** Stand up a known-good single-prompt reference (CPU torch, paper inference/model.py, or a different engine with the same model) BEFORE writing production code. Without a reference, every diff is suggestive; with one, every diff is conclusive.
2. **Bisect harness ships before model code.** Build the env-gated dump + cos-similarity-compare + ref-patch trio FIRST. If you find yourself adding `torch.save()` calls to debug a wrong output, you're already 4 hours late.
3. **A documented-broken path is a SIGNAL to rewrite, not a CHALLENGE to fix.** When the in-code comment says "stop-gap with known bugs, will be replaced in Sprint X", patching it stacks bugs on top of admitted bugs. Read the comment, do the rewrite, save the patches.
4. **Use STANDARD engine primitives until you have proof you cannot.** `block_table`, `slot_mapping`, `cu_seqlens_q`, `FusedMoE`, `ColumnParallelLinear` exist for a reason. Custom slab allocators and bespoke pool layouts are 10× the surface area for the same capability. Re-use first; invent only when re-use can't fit.
5. **Land in phases. Each phase ships a demonstrable capability.** PR1 = skeleton runs at conc=1 with paper output. PR2 = abstraction (per-req cache / block_table integration). PR3 = multi-seq dispatch. PR4 = perf. Each PR independently testable + revertable.

## Phase-at-a-glance

| # | Phase | Goal | Time | Move on when |
|---|-------|------|------|--------------|
| 0 | Reference | Stand up the known-good impl | ≤ 4h | Reference produces user-expected output end-to-end on a toy shape |
| 1 | Infra | Bisect harness + atom.__file__ verify in every script | ≤ 4h | `dump.py` + `compare.py` produce cos-table for one layer between ref and ATOM |
| 2 | Skeleton | New module(s) — model class, attention class, custom op shims; conc=1 only | ≤ 1d | Conc=1 single prompt runs end-to-end on real ckpt with paper-matching output |
| 3 | Per-req abstraction | Per-request state cache via standard `block_table` | ≤ 1d | Two prompts back-to-back share the engine, second prompt's state isolated from first |
| 4 | Multi-seq dispatch | Per-seq Python loop dispatch (not packed batch) | ≤ 1d | Conc=2 smoke 4/4 + lm_eval ≥ conc=1 baseline − 5pp |
| 5 | Backend gates | Triton / aiter kernel scaffolds for perf path; opt-in env flags | ≤ 2d | Default path remains correct; opt-in flag matches default within tolerance |
| 6 | Perf | Kernel fusion, batched dispatch, CUDAGraph | open-ended | Production target met; correctness uncompromised |

Total per feature (correctness-only, Phases 0–5): ~4–6 days. Phase 6 is its own multi-week effort. **You do not start Phase 6 until Phase 5 silicon-validates.**

## Phase 0 — Reference impl (mandatory, no shortcuts)

The reference is your oracle for every numerical claim downstream. It must:

- Produce paper-matching output on at least one toy shape (e.g. tiny config you constructed yourself, or the model's official `inference/model.py` if released).
- Run on CPU or single-GPU eager with `--enforce-eager`. No CUDAGraph. No torch.compile. No kernel fusion. No quantization unless that's what you're testing.
- Be invocable from a Python REPL in <1s for a tiny shape so you can iterate locally.

**Common dodge to avoid**: "the existing ATOM legacy path works at conc=1, I'll use it as the reference for the new conc=N path." Wrong. The legacy path may fork to entirely different code in conc=N (DSV4 did: `_forward_legacy` vs `_forward_w4`), so any "diff" between them at conc=2 is meaningless.

If no reference exists you can run, your first PR is a Phase 0 PR. Do not skip.

## Phase 1 — Infra before model code

Steal Lingpeng's pattern:

```
atom/utils/debug_helper/
├── dump.py          # env-gated forward / weight / sampler dump; multi-class hooks
├── compare.py       # cos_max (fp64 to avoid the cos>1.0 trap), slot_split, CLI
└── ref_patch.py     # monkey-patch context for instrumenting read-only refs
```

Wire 9 env vars in `atom/utils/envs.py` (`ATOM_FWD_DUMP_*`, `ATOM_WEIGHT_DUMP_*`, `ATOM_DEBUG_TOPK*`), all no-op by default.

**Test the harness before testing the feature**: dump the SAME run twice with the same seed. Cos must equal 1.0. If it doesn't, your harness has a bug (e.g. TP rank race writing same filename — add `_R{rank:02d}` suffix).

Add to every silicon script:

```bash
/opt/venv/bin/python -c "import atom; print('[sanity]', atom.__file__)"
```

If it doesn't print the editable install path you expect, fix the install before launching the server. (The DSV4 work lost 4 hours to this exact gotcha.)

## Phase 2 — Skeleton (conc=1 only)

Write the new module(s) cleanly. Prefer NEW files over EDITING existing files. Per the DSV4 PR #650 example: dedicated `atom/model_ops/attentions/<model>_attn.py` + `sparse_attn_<model>.py` + `quant_<model>.py` rather than wedging into the shared base file.

Use standard ATOM primitives:

- `ColumnParallelLinear`, `RowParallelLinear`, `ReplicatedLinear`, `FusedMoE`, `MergedColumnParallelLinear`
- `_apply_rotary_emb`, `act_quant_inplace`
- `block_table` for KV (NOT a custom slot allocator)

Goal at end of Phase 2: `python -m atom.examples.simple_inference --max-num-seqs 1` produces paper-matching output on one prompt. Use the Phase 0 reference to assert via cos > 0.9999 at the final hidden state.

## Phase 3 — Per-request cache abstraction

Multi-seq is downstream of per-req. If your model needs custom state (compressor state, sliding window, etc.), wire it through the standard per-req cache abstraction (e.g. `per_req_cache` in ATOM) — NOT through a model-level singleton.

Test: send two prompts back-to-back at `--max-num-seqs 1`. The second prompt's state must be unaffected by the first's residue. If it isn't, you have a state leak — fix here, not in Phase 4.

## Phase 4 — Multi-seq dispatch

**Per-seq Python loop is the correct first implementation.** Not "premature pessimization" — the only way to know your per-token math is right is to first prove it works one seq at a time:

```python
def forward(self, hidden_states, forward_batch):
    outputs = []
    for i in range(forward_batch.batch_size):
        beg = forward_batch.cu_seqlens_q[i]
        end = forward_batch.cu_seqlens_q[i+1]
        slot = forward_batch.req_pool_indices[i]
        x_i = hidden_states[beg:end]
        out_i = self._forward_single_seq(x_i, slot, ...)
        outputs.append(out_i)
    return torch.cat(outputs, dim=0)
```

Phase 4 acceptance gate (mandatory, no exceptions):

1. **Conc=2 smoke**: 4 parallel curls with distinct math prompts; ≥3/4 must answer correctly.
2. **Conc=4 smoke**: same, ≥3/4.
3. **Lm_eval gsm8k limit=20 5-shot at conc=4**: within −5pp of the conc=1 baseline.

If any of these fails, Phase 4 is not done. Do NOT advance to Phase 5 (perf) on a broken Phase 4.

If Phase 4 fails repeatedly, run `dump-bisect-debug` on a single layer between conc=1 and conc=2 (with rank-suffixed dumps) to localize. Per-seq loop should bit-equal single-seq for seq 0 — if it doesn't, you have a TP rank race or a shared-state leak.

## Phase 5 — Backend gates

Now you can scaffold the perf-oriented kernels (Triton, aiter custom). Each backend lives behind an opt-in env flag (`ATOM_USE_<MODEL>_TRITON_ATTN=1`). Default is the Phase 4 per-seq loop.

Acceptance gate per backend: cos > 0.9999 vs the default path at conc=1 AND conc=4.

## Phase 6 — Perf

Begins only after Phase 5 silicon-validates. Use `/benchmark-guide` for measurement. Lock determinism via `--temperature 0.0 --seed 42`.

## Anti-patterns (what NOT to do)

| Anti-pattern | Why it fails |
|---|---|
| Patching a "stop-gap with known bugs" comment | Stacks new bugs on admitted bugs. Rewrite. |
| Adding `torch.save()` debug hooks AFTER an output bug appears | You're 5 hours late. Build dump.py before the bug exists. |
| Comparing legacy-path conc=1 to new-path conc=N as if they're the same code | They forked at the dispatch. Use a Phase 0 reference, not a sibling code path. |
| Skipping the per-seq loop and writing packed-batch dispatch first | Packed batch hides per-token bugs in batch-axis broadcasts. Loop first, optimize later. |
| Inventing a custom KV pool when `block_table` would fit | 10× surface area for the same capability. The custom pool will have its own bugs. |
| TP-shared trace files without rank suffix | Workers race; conc=1 captures rank-X, conc=N captures rank-Y, "diff" is meaningless. |
| 10-min silicon iteration as your debug loop | Forces you into "just one more try" thinking. Build local CPU/tiny-GPU loop. |
| "Audit → patch → silicon" with no falsifiable prediction per patch | A fix without a falsifiable prediction is a guess in a lab coat. State what trace data you expect to change BEFORE the silicon run. |
| Treating each Sprint as "the fix" instead of one step in a phased plan | No milestone discipline → no incremental confidence → false rollbacks. |

## Decision tree

```
Task: "Make feature X work in ATOM."
│
├─ Is there code labelled "X stop-gap" in the repo?
│   ├─ YES → REWRITE per spec. The labelled code is technical debt that
│   │       was never meant to ship.
│   └─ NO  → Greenfield. Skip to Phase 0.
│
├─ Phase 0: Have a reference I can diff against per layer?
│   ├─ NO  → Build one (paper inference/model.py / vLLM / SGLang / CPU torch).
│   │       This is your oracle. Cannot proceed without.
│   └─ YES → Phase 1.
│
├─ Phase 1: Has bisect harness been used to compare ref vs new on at least
│            one layer with cos table?
│   ├─ NO  → Build atom/utils/debug_helper/ + envs + run a 1-layer cos test.
│   └─ YES → Phase 2.
│
├─ Phase 2: Conc=1 produces paper-matching output, cos > 0.9999 at logits?
│   ├─ NO  → Bisect with debug_helper. Find first layer where cos < 0.99.
│   └─ YES → Phase 3.
│
├─ Phase 3: Two back-to-back prompts at --max-num-seqs 1, no state leak?
│   ├─ NO  → Fix per-req cache; do NOT proceed. State leak at conc=1 will
│   │       compound at conc=N.
│   └─ YES → Phase 4.
│
├─ Phase 4: Conc=2/4 smoke 4/4 AND lm_eval conc=4 within −5pp of conc=1?
│   ├─ NO  → Bisect SAME prompt at conc=1 vs conc=N for seq 0 row, layer
│   │       by layer. (Use rank-suffixed dumps; see dump-bisect-debug.)
│   └─ YES → Phase 5.
│
└─ Phase 5/6: Backend gates → Perf. Each opt-in flag silicon-validated
              against default path before flag becomes default.
```

## When to invoke this skill

- Adding a new model class (DSV4, GLM-5, qwen3-next, etc.)
- Replacing a KV cache scheme (slot-based → block-based, etc.)
- Adding a new attention path (sparse, sliding window, MoBA, NSA, etc.)
- Adding a new quantization (MXFP4 → A4W8, etc.)

If the task is instead "fix a numerical bug in an EXISTING working path", invoke `dump-bisect-debug` directly — that one is for surgical bisect within a working framework.

## Companion files

- `dump-bisect-debug.md` — Phase 4 bisect mechanics in detail
- `atom-patterns.md` — ATOM-specific conventions (block_table, FusedMoE, TP linears)
- `recipes/<model>.md` — production deployment recipe per model (cherry-pick from this skill's outputs)

## Concrete reference: DSV4-Pro PR #650

A worked example of this methodology. Read https://github.com/ROCm/ATOM/pull/650 alongside this skill — its commit chain (PR1 → pre2a → pre2c-A → pre2c-B → PR3-main → PR-A) maps 1:1 to Phases 2 → 3 → 4 → 5 above.
