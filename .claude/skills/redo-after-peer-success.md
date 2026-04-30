---
name: redo-after-peer-success
description: A peer (or parallel agent) shipped the working solution to the same problem you were stuck on. Use this skill to (1) extract maximum learning from their solution, (2) decide what of YOUR work survives, (3) plan the re-do as a day-by-day checklist, (4) preserve the lesson so the failure mode never repeats. NOT for "I'm stuck and want to ask for help" — for "they're done and I need to catch up + internalize".
version: 1.0.0
scope: Inference engines / large refactors / multi-week features
last_updated: 2026-04-30
companion_skills:
  - ramp-inference-feature   # the methodology you should have followed
  - dump-bisect-debug         # the bisect mechanics you should have used
---

## When this skill applies

You spent N hours/days on a feature. You hit walls. You reverted patches. You opened a hand-off. Then a peer's PR lands the working solution while you were stuck.

The wrong reactions:
- Ignore their PR ("not invented here").
- Rebuild from scratch ignoring their work ("I want to learn it myself").
- Cherry-pick their commits into your branch without reading them ("merge and move on").

The right reaction is this skill — a structured 1-day catch-up that turns their win into your durable capability.

## The four moves (in order)

### Move 1 — Validate their solution silicon yourself (≤ 1h)

Before reading a single line of their code, verify their claim. The DSV4 case:

```bash
# Check out their branch in the working dir (host)
cd /home/pensun/ATOM
git remote add upstream https://github.com/ROCm/ATOM.git  # if not present
git fetch upstream pull/650/head:pr650 && git checkout pr650

# Make the docker container actually use it
sudo docker exec atom_dsv4_feat /opt/venv/bin/pip install -e \
    /workspace/ATOM-feat --no-deps --no-build-isolation
sudo docker exec atom_dsv4_feat /opt/venv/bin/python -c "import atom; print(atom.__file__)"
# MUST print /workspace/ATOM-feat/atom/__init__.py

# Run THEIR recommended invocation. Their PR body has it; use it as-is.
# Don't add your patches. Don't change their flags. Verify the unmodified version.

# Run YOUR previously-failing test against it (the smoke test that previously failed).
# In DSV4: 4-prompt math smoke at conc=4. Result: 4/4. Take 5 min.
# Run a real eval (gsm8k limit=20 5-shot conc=4). Take 5 min. Result: 0.90.
```

If their solution doesn't actually work in your environment, stop here and resolve infra. Don't read their code yet.

If it works, you have ground truth: their architecture is correct. Move to Move 2.

### Move 2 — Diff their solution against your dead-end (≤ 2h)

Map their commit chain against your sprint chain. For DSV4, what comes out:

| Their PR phase | Their commit | Your equivalent attempt | Verdict |
|---|---|---|---|
| PR1 skeleton | `09db456` `880c410` `5838ee1` | (none — I patched existing) | **Architectural rewrite vs patch** |
| Per-req cache | `84e0a05` (`pre2a`) `047ee11` (`pre2c-B`) | DSV4KVPool slot allocator | **Standard `block_table` vs custom slabs** |
| Multi-seq dispatch | `aebe2ff` (`PR3-main`) per-seq Python loop | `_forward_w4` packed batch | **Per-seq loop vs packed batch** |
| Bisect harness | `3870756` (`atom/utils/debug_helper/`) | trace hooks added at hour 5+ | **Tool first vs tool last** |
| Methodology doc | `.claude/skills/dump-bisect-debug.md` (32KB) | (none) | **Skill written before code vs no skill** |
| Backend gate scaffold | PR-A in progress | (premature; tried to fix existing kernel) | **Default = correct path; opt-in perf** |

This table is the lesson. Save it. The DELTAS (right column) are what you must internalize. They are NOT specific to DSV4 — they are the general pattern.

### Move 3 — Salvage what survives, kill what doesn't (≤ 2h)

Inventory your branch. For each commit, decide:

| Disposition | Criteria | Action |
|---|---|---|
| **Keep** | Silicon-proven AND independent of the new architecture (env flags, env vars, infra fixes, doc commits) | Preserve as-is; consider cherry-picking into the peer's PR if useful |
| **Replace** | Same intent as the peer's commit but different architecture | Drop yours; accept theirs |
| **Reject** | Patches to the now-superseded path | Revert if not yet reverted; document in EVIDENCE.md why |

For DSV4, the inventory:

- KEEP: Sprint 7b cast fix (`2d0098c`), 7c admit-gate (`96176c4`), 6 B0a Indexer FP8 (`a8e3a02`) — env-flag scoped, value preserved. Hand-off issue #60 already calls these out as Tier 1.
- REPLACE: my `DSV4KVPool` work (Sprint 6 B0b.1-5) by their `block_table` design.
- REJECT: Sprint 7d slot lookup, 7e per-seq Indexer loop, 7g pad-fix — already reverted.

The "REJECT" pile is your evidence of the failure pattern. Keep it; it's how you internalize "don't do this again".

### Move 4 — Write the lesson into a skill BEFORE next session (≤ 2h)

This skill (or the companion `ramp-inference-feature`) is that artifact. Your next attempt at this kind of feature must START by reading the skill, not by writing code. The DSV4 redo-day-by-day is the worked example below.

## Worked example: DSV4-Pro multi-req redo plan

If I were starting DSV4-Pro multi-req tomorrow with what I now know:

### Day 1 — Reference + infra (zero model code)

| Hour | Task | Output |
|---|---|---|
| 0 | Read paper §3.6.1 + §3.3 batch invariance | Notes on ring buffer, per-paper KV layout |
| 1 | `git clone` official `inference/model.py` if available; or build a CPU torch reference for one V4 layer at toy shape | Reference runs in Python REPL in <2s |
| 2 | Stand up `atom/utils/debug_helper/` (steal Lingpeng's structure verbatim if PR #650 is merged) | `dump.py` + `compare.py` + 9 env vars |
| 3 | Wire 1-layer cos test: ATOM toy V4 vs reference, expect cos > 0.9999 | First green light |
| 4 | Commit. PR0: "infra and reference for DSV4-Pro" | Reviewable in <30 min |
| 5–8 | Read the paper inference code top-to-bottom, mapping to ATOM primitives. Specifically: which existing `atom/model_ops/attentions/*.py` is the closest precedent? | Implementation notes |

Hard rule: **no model code written today.** If I'm tempted to write code on day 1, I haven't done day 1.

### Day 2 — Skeleton (Phase 2 of `ramp-inference-feature`)

| Hour | Task | Output |
|---|---|---|
| 0–4 | Write `atom/models/deepseek_v4.py` (model class, top-level forward) reusing standard primitives: `ColumnParallelLinear`, `FusedMoE`, `_apply_rotary_emb` | Compiles, loads weights |
| 4–6 | Write `atom/model_ops/attentions/deepseek_v4_attn.py` — dedicated attention class, NOT a fork of `attention_mha.py` | Forward returns shape-correct tensor |
| 6–8 | `atom/examples/simple_inference.py --max-num-seqs 1` produces output. Cos vs reference at final hidden state. Target > 0.9999. | Conc=1 working |

End of day 2 silicon checkpoint: ONE prompt at conc=1, output cos > 0.9999 vs reference. If not green, debug WITHIN day 2; do not start day 3.

### Day 3 — Per-req abstraction (Phase 3)

| Hour | Task | Output |
|---|---|---|
| 0–4 | Wire `block_table` for V4 KV. Use ATOM's standard `block_manager.py`, do not invent a custom pool. | Two prompts back-to-back at `--max-num-seqs 1` produce isolated state |
| 4–6 | Add per-req state cache for compressor (per paper §3.6.1 ring buffer) — use `per_req_cache` abstraction if available | Compressor state isolated per req |
| 6–8 | Test: prefill prompt A, finish, prefill prompt B. B must not see A's residue. | Conc=1 sequential isolation green |

### Day 4 — Multi-seq dispatch (Phase 4)

| Hour | Task | Output |
|---|---|---|
| 0–4 | Write the per-seq Python loop in attention forward. Slow is FINE. Just correct. | Compiles |
| 4–6 | Conc=2 smoke (2 prompts) at `--max-num-seqs 2`. Target ≥ 1/2 (then 4/4). If wrong, bisect with `debug_helper` rank-suffixed dumps. | Smoke result |
| 6–8 | Conc=4 smoke. Target 4/4. Then `lm_eval gsm8k limit=20 5-shot conc=4`. Target within −5pp of conc=1. | Multi-req silicon-validated |

End of day 4 silicon checkpoint: conc=4 smoke 4/4 + lm_eval ≥ baseline−5pp. THIS is where Lingpeng's PR #650 sits today.

### Day 5 — Backend gate scaffold (Phase 5, optional this week)

Custom Triton kernels (`swa_write`, `update_compressor_states`) behind opt-in env flag. Default path (per-seq Python loop) stays correct. Cos test per backend.

### Days 6+ — Perf (Phase 6)

Only after silicon-validated correctness. Use `/benchmark-guide`.

## Process meta-rules I now follow

1. **No silicon iteration as primary debug loop.** Local CPU/tiny-GPU torch ref is sub-second. Silicon is for end-to-end gates only.
2. **No code on day 1.** Day 1 is reading + infra + reference. If I code on day 1, I haven't read enough.
3. **Comment-driven decisions.** When code says "stop-gap with known bugs", I rewrite. When code says "battle-tested for X", I extend. Read the comments.
4. **Per-seq loop before packed batch, always.** Performance is a Phase 6 problem. Correctness is a Phase 4 gate.
5. **Standard primitives until proven they don't fit.** `block_table`, `FusedMoE`, `ColumnParallelLinear`. The custom path adds 10× the bugs for the same capability.
6. **Skills committed before code.** If the methodology isn't written down, the next person (including future-me) re-learns it the expensive way.

## Anti-patterns when reading a peer's solution

| Anti-pattern | Why it's worse than not reading at all |
|---|---|
| "I'll cherry-pick their commits without reading" | You inherit their architecture without understanding it; first divergence point breaks you completely. |
| "I'll rewrite their solution my way to learn" | You re-derive their architecture and re-discover their bugs. Wasted week. |
| "I'll ignore their PR and finish my approach" | Sunk cost. Their architecture is silicon-proven; yours is silicon-failed. Ignoring it is irrational. |
| "I'll just merge and not internalize" | The next feature like this, you'll fail again the same way. The skill is the moat. |
| "I'll defer reading until I'm 'unblocked'" | You're already maximally blocked. The unblocking is the reading. |

## When NOT to invoke this skill

- The peer's solution doesn't actually work (validate Move 1 first).
- The peer's solution solves a DIFFERENT problem and yours is still open (rare; usually you're rationalizing).
- Your work is in a fundamentally different stack (e.g. their solution is for vLLM, yours is for ATOM, and they don't share kernels — even then, the methodology transfers).

## Output of this skill

After running the four moves, you should have produced:
1. A silicon-validated test of the peer's solution in your environment.
2. A diff/inventory mapping their commits to your dead-ends.
3. An updated hand-off note marking your "REJECT" pile as superseded.
4. An updated or new skill capturing the lesson.
5. A day-by-day re-do plan for "if I had to do it again".

Items 4 and 5 are the durable artifacts. Items 1–3 are housekeeping.
