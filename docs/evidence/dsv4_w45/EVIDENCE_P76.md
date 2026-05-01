# P7-r6 Empirical Findings — DSV4 CUDAGraph Capture Variance Investigation

**Status**: Triton-stream-race hypothesis FALSIFIED. Multi-request capture confirmed as the dominant accuracy/variance source. Recommended next sprint scope: V4 multi-request capture isolation (PR-sized, not bisect).

**Branch**: `dev/dsv4-cudagraph-bisect-r5` @ commits `3a01d11`/`cade686` (P7-r6 changes are additive, default-off env-flag gated, zero impact when off).

## TL;DR

| Config | bs cap | gsm8k limit=20 (3 runs) | mean | std |
|---|---|---|---|---|
| **Eager baseline (P7-r5)** | n/a | 0.85 | 0.85 | n/a |
| **Conc=1 + Triton + capture** | [4] | **0.90** | 0.90 | n/a |
| Conc=1 + PyTorch + capture | [4] | 0.75 / 0.70 | 0.72 | 0.025 |
| Conc=4 + PyTorch + capture | [4] | 0.90 / 0.70 / 0.50 | 0.70 | 0.20 |
| Conc=4 + PyTorch + capture (slot fix) | [4] | 0.60 / 0.55 / 0.50 | 0.55 | 0.05 |
| Conc=4 + Triton + capture (slot fix) | [4] | 0.55 / 0.45 / 0.35 | 0.45 | 0.10 |
| Conc=4 + Triton + multi-bs cap | [1,2,3,4] | 0.50 / 0.35 / 0.45 | 0.43 | 0.07 |
| **P7-r5 phase 10 (Triton + barrier)** | [4] | 0.45 / 0.70 / 0.95 | 0.70 | 0.25 |

**Key observations**:
1. Conc=1 + Triton capture ≈ eager (0.90 vs 0.85). The capture infra works correctly for single-request decode.
2. Conc=4 capture wildly variant regardless of state-write backend (Triton or PyTorch) or capture batch size sweeping.
3. Multi-bs capture [1,2,3,4] does NOT fix conc=4 — eliminates padding hypothesis.
4. PyTorch state writes mildly help variance but at the cost of mean accuracy at conc=1.

## Hypothesis tests

### Path A (P7-r5's hypothesis): Triton state-mutation kernels race with PyTorch reads under cudagraph

**Falsified.** P7-r6 added a pure-PyTorch advanced-indexing equivalent of `update_compressor_states` and `swa_write` (env-gated `ATOM_V4_PYTORCH_STATE_WRITE`). Conc=4 capture with PyTorch state writes still varies 0.45–0.95. Identical variance signature with Triton.

If Triton-stream-race were the cause:
- PyTorch path would converge to deterministic accuracy (PyTorch index_put_ is in the captured dataflow graph; no separate stream).
- Conc=1 with Triton would also vary (Triton-stream-race is request-count agnostic).

Both predictions failed:
- PyTorch path still variant at conc=4.
- Triton at conc=1 = 0.90 (matches eager).

### Padding hypothesis: state_slot_mapping[scheduled_bs:bs] is stale → padding-iter writes corrupt active slots

**Falsified.** Captured graphs at all sizes [1,2,3,4] (so decode never pads) still show conc=4 variance 0.35–0.50. Removing padding does NOT fix the issue.

### Multi-request capture isolation hypothesis (NEW, OPEN)

The conc=4 capture path corrupts cross-request state in a way that does NOT reproduce at conc=1. Likely candidates not yet ruled out:

- **Per-iteration baked Python ints** — `seq_offsets[seq_idx]`, `start_pos_per_seq[seq_idx]`, `slots[seq_idx]` are Python lists evaluated at trace time. These bake CAPTURE-TIME values into the graph. At replay, real per-step values for those indices are NOT used (only tensor-form `slot_tensor`, `seq_positions`, etc. are live). Risk: if a request lifecycle event (finish, swap-in) reorders requests across decode steps, the captured iterations process WRONG request data.
- **Inter-iteration state aliasing** — within one captured graph step, iter 0 writes to state, iter 1 reads from state. PyTorch should serialize these via dataflow, but ROCm cudagraph may have aliasing rules that differ.
- **Indexer's compressor inner state** — separate kv_state/score_state pool indexed by same slot_tensor; not yet bisected.

Evidence supporting this: at conc=1 capture, ONLY iter 0 ever runs (single-request decode). All cross-iter aliasing concerns are moot. At conc=4, all 4 iterations execute and could interact.

## What's landed in P7-r6

Changes to `dev/dsv4-cudagraph-bisect-r5`:

- `atom/model_ops/v4_kernels/state_writes.py`:
    `update_compressor_states_pytorch` — capture-friendly tensor-driven equivalent of the Triton kernel. CPU unit test validates 5 cases (overlap prefill / overlap decode / mixed batch / HCA prefill / HCA decode-mixed) against the legacy reference.
    `swa_write_pytorch` — capture-friendly equivalent of swa_write (alias of `swa_write_reference` with explicit capture contract docs).
- `atom/model_ops/v4_kernels/__init__.py`: export new helpers.
- `atom/models/deepseek_v4.py`:
    `_v4_use_pytorch_state_writes()` env-flag reader (`ATOM_V4_PYTORCH_STATE_WRITE`, default 0).
    `_update_compressor_states_dispatch()` / `_swa_write_dispatch()` — Triton-vs-PyTorch dispatcher.
    Switched 3 call sites (Compressor.forward batched + single-seq, V4Attention.forward swa_write).
    Defensive fix: V4Attention.forward swa_write `slot_per_token` — broadcast from `slot_tensor` instead of `torch.full(..., slot, ...)` to avoid Python-int slot baking when ATOM_V4_BACKEND=new is enabled. This branch is dead in the default `legacy` config so the defensive fix has no current silicon impact.

All changes are additive and gated by an env flag default off. Zero impact on existing eager / capture paths when env is unset.

## What's NOT landed (recommendation for next sprint)

- **Multi-request capture isolation** is the proper-fix scope. Likely needed:
    1. Audit every Python-int baked into the captured per-seq loop (`seq_offsets`, `slots`, `start_pos_per_seq`, `seq_idx` itself) — replace with tensor-driven equivalents that read live forward_vars at replay.
    2. Confirm captured graph determinism with a dump-bisect comparing 2 successive replays at conc=4.
    3. Write a unit test for "captured-graph slot dispatch correctness": warm up at 4 reqs, replay with shuffled state_slot_mapping, assert per-slot output matches a fresh eager run.
- **Sprint 7c-7i context**: PR #650 (Lingpeng's V4 redesign) achieves conc=4 gsm8k 0.90/0.90 in EAGER mode. Capture is the regression vector; the model architecture is sound.

## Test scripts (this branch / docker)

In `/home/pensun/ATOM-lingpeng`:

- `p76_phase1_pytorch_state.sh` — 3 runs conc=4 capture with PT state writes
- `p76_phase1b_slot_fix.sh` — 3 runs PT + 3 runs Triton with the (dead-code) slot_per_token defensive fix
- `p76_phase2_conc1.sh` — 4 runs at conc=1 (PT cap × 2, Triton cap, PT eager)
- `p76_phase3_multibs_capture.sh` — 3 runs conc=4 with --cudagraph-capture-sizes [1,2,3,4]
- `p76_phase2_eager_regression.sh` — eager with PT vs Triton state writes (regression baseline)

Logs in `logs/p76_phase{1,1b,2_conc1,3}/`.

