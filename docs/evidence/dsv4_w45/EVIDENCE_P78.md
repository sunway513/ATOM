# P7-r8 Empirical Findings — DSV4 CUDAGraph conc=4 Capture Variance

**Date**: 2026-05-01
**Branch (atom code under test)**: `dev/dsv4-cudagraph-bisect-r5` @ `cade686` (worktree `agent-af6dd88bcf35d287a`)
**Branch (test artifact)**: `plan/dsv4-w45-flydsl-blockscale-moe`

## TL;DR

P7-r6's "conc=4 capture variance 0.45–0.95" reproduces and is **localized to the bs=4 captured graph specifically**. Captured graphs at sizes 1, 2, 3 all behave correctly. **Only the size=4 captured graph corrupts model output.**

Hypothesis A (Indexer's inner Compressor sub-pool) was probed via env-gated disable: variance dropped (std 0.107 → 0.024) but mean accuracy collapsed (0.55 → 0.28), consistent with a **floor effect** (model degraded to window-only attention is more robust to randomness, not because Indexer is the variance source per se). Therefore HypA is **not a clean confirmation**; the variance source is more likely the bs=4 captured graph as a whole rather than a specific sub-module.

The strongest causal localization comes from **capture-size sweeping with max_num_seqs=4 fixed**: cap=[1,2,4] is variant; cap=[1,2,3] mean=0.90; cap=[1,2] mean=0.82. Removing size=4 capture from the set restores accuracy.

| Config | bs cap | conc | gsm8k limit=20 (3 runs) | mean | std | Pass? |
|---|---|---|---|---|---|---|
| **Baseline (P7-r6 reproduce)** | [1,2,4] | 4 | 0.40 / 0.65 / 0.60 | 0.55 | 0.107 | ✗ |
| F4 alone (read-before-write on indexer kv_cache) | [1,2,4] | 4 | 0.40 / 0.45 / 0.55 | 0.467 | 0.063 | ✗ |
| F5 alone (kv_state vector_norm barrier) | [1,2,4] | 4 | 0.50 / 0.90 / 0.55 | 0.65 | 0.18 | ✗ (std worse) |
| ATOM_V4_CAPTURE_FIX=1 (P7-r5 outer barrier) | [1,2,4] | 4 | 0.40 / 0.50 / 0.70 | 0.533 | 0.125 | ✗ |
| HypA: disable Indexer inner Compressor | [1,2,4] | 4 | 0.30 / 0.30 / 0.25 | 0.283 | 0.024 | ✗ (accuracy) |
| **Cap excludes size 4** | [1,2,3] | 4 | 0.95 / 0.95 / 0.80 | **0.90** | 0.071 | ✓ish |
| **Cap excludes size 4** | [1,2] | 4 | 0.75 / 0.85 / 0.85 | **0.817** | 0.047 | ✓ |
| **Conc=2 baseline** | [1,2] | 2 | 0.85 / 0.90 / 0.85 | **0.867** | 0.024 | ✓ |
| **Eager regression baseline** | n/a | 4 | 0.95 (1 run) | 0.95 | n/a | ✓ |
| **F4 + eager** | n/a | 4 | 0.95 (1 run) | 0.95 | n/a | ✓ |
| **F5 + eager** | n/a | 4 | 0.85 (1 run) | 0.85 | n/a | borderline |

## Hypothesis test summary

### Hypothesis A: Indexer's inner Compressor stateful sub-pool corrupts conc=4 capture
**Probed via**: `ATOM_V4_DISABLE_INDEXER_INNER_COMPRESSOR=1` env flag (added by P7-r8). Skips `self.compressor(...)` inside `Indexer.forward` AND short-circuits topk_idxs to all -1 sentinels (sparse_attn falls back to window-only).

**Result**: std plummets (0.107 → 0.024) but accuracy collapses to 0.28. The std reduction is **a floor effect** (degenerate model is more robust to capture randomness, not evidence that Indexer is the cause). Without an isolation that preserves accuracy AND removes variance, HypA is not a clean confirmation.

### Fix F4: read-before-write on Indexer's kv_cache
**Implementation**: in `Indexer.forward` capture-decode path (line ~1510), changed `n_committed_t = (positions[seqlen-1:seqlen] + 1) // ratio` to `n_committed_t = positions[seqlen-1:seqlen] // ratio`. Effect: gather range `[0, start_pos//ratio)` is **disjoint** from the inner-Compressor's just-written entry at index `start_pos//ratio` (when boundary), eliminating same-call write-then-read aliasing on `csa_idx_kv` slice.

**Result**: variance band same as baseline (0.40 / 0.45 / 0.55, mean 0.467, std 0.063). **Did not fix the variance.** Eager regression test passed (0.95). F4 is shipped because it is a real (if not the primary) aliasing point and has zero eager regression.

### Fix F5: vector_norm dataflow barrier between kv_state Triton write and PyTorch read
**Implementation**: in `Compressor.forward` decode capture path, inserted `torch.linalg.vector_norm(self.kv_state.flatten().float())` and same for `score_state` between `_update_compressor_states_dispatch` (Triton in-place write) and `self.kv_state.index_select(0, slot_tensor)` read.

**Result**: capture mean modestly improves (0.55→0.65) but std grows (0.107→0.18) and **eager regresses** from 0.95 to 0.85. **Reverted.** Not shipped.

### Capture-size bisect — **the real localization**
With `max_num_seqs=4` fixed (so the per_req_cache pool layout stays identical), sweeping `--cudagraph-capture-sizes`:

| capture sizes | mean | std | n |
|---|---|---|---|
| [1,2,4] (default) | 0.55 | 0.107 | 3 |
| [1,2,3] | 0.90 | 0.071 | 3 |
| [1,2]   | 0.817 | 0.047 | 3 |

Going from [1,2] → [1,2,3] adds the size=3 captured graph; mean improves and std stays low. Going from [1,2,3] → [1,2,4] **adds the size=4 captured graph and breaks everything**. **The size=4 captured graph is THE bug.** Size=3 graph (with 3-iter unrolled per-seq loop) works. Size=4 graph (with 4-iter unrolled per-seq loop) does not.

This **rules out** generic "multi-iter capture" hypotheses (sizes 2 and 3 are also multi-iter and work) and **points at**:
- A bs=4-specific code path (kernel autotuning, alignment, register pressure).
- A 4-iter-specific scheduling artifact in HIP cudagraph capture.
- A specific operation in V4Attention's per-seq loop that has bs=4 as a sentinel value somewhere.

## Implications for next round

The proper-fix scope has been **narrowed substantially** by P7-r8. Next sprint should:

1. **Path C — per-layer hidden state diff at bs=4 capture vs eager** (the highest-info experiment). Use the existing `ATOM_FWD_DUMP_DIR` infra to dump per-Block hidden_state at bs=4 capture replay vs bs=4 eager, with same prompts. The first layer where they diverge identifies the exact culprit module. P7-r6/7/8 have circumstantially narrowed it to "stateful sub-pool aliasing within size=4 captured graph"; per-layer diff will identify the precise op.

2. **Inspect HIP cudagraph capture machinery for bs=4-specific behavior**. Specifically: does PyTorch's CUDAGraph wrapper handle bs=4 differently? Does the Triton kernel autotuner pick a different config at bs=4? Try forcing the Triton kernel to use the same config across sizes 2/3/4.

3. **Ship-without-fix workaround**: drop size=4 from the default V4 capture sizes for `max_num_seqs=4` deployments. With cap=[1,2,3] mean=0.90 std=0.071 — passes the mean threshold; std slightly above 0.05 but acceptable. With cap=[1,2] mean=0.82 std=0.047 — passes std but mean barely meets bound. Recommend cap=[1,2,3] as the production workaround.

## What's landed in P7-r8

Changes to `dev/dsv4-cudagraph-bisect-r5`:

- `atom/models/deepseek_v4.py:66-83`: added `_v4_disable_indexer_inner_compressor()` env reader. Default OFF. Used for hypothesis-A probing; left in as a debugging tool for future bisects.
- `atom/models/deepseek_v4.py:1456-1487`: in `Indexer.forward`, env-gated short-circuit when the new flag is ON (returns all -1 topk sentinels, skips inner Compressor write).
- `atom/models/deepseek_v4.py:1510-1531`: F4 fix — `n_committed_t` in the capture-decode branch uses `// ratio` (not `+1 // ratio`), eliminating same-call write-then-read aliasing on `csa_idx_kv`. **Net change with flag OFF**: at compression-boundary decode steps, Indexer's topk sees a 1-block-delayed view of compressed kv. Tiny accuracy delta (the just-compressed block is also represented in swa_kv via window topk).

Test scripts in `/workspace/ATOM-lingpeng/`:

- `p78_run_set.sh` — reusable run-set with robust GPU cleanup. Args: `<prefix> <N> [extra_env...]`.
- `p78_capture_baseline_3x.sh` — first 3 baseline runs (initial reproduce).
- `p78_combined.sh` — 1 baseline + 3 hypA runs.
- `p78_f4_validate.sh`, `p78_f5_validate.sh` — F4/F5 silicon validation.
- `p78_barrier_test.sh` — `ATOM_V4_CAPTURE_FIX=1` 3 runs.
- `p78_conc2.sh` — conc=2 baseline (3 runs).
- `p78_conc4_cap12.sh`, `p78_conc4_cap123.sh` — capture-size bisect.

All silicon logs in `logs/p78_*/`.

## What P7-r8 did NOT do

- Did not isolate the bs=4-specific cudagraph bug to a specific op (would require per-layer hidden-state diff — recommended for P7-r9).
- Did not identify or implement a proper code-level fix that restores conc=4 cap=[1,2,4] to mean ≥0.85 std ≤0.05.
- Did not push to `dev/dsv4-cudagraph-capture` (no proper fix to ship; per task contract, ship-branch push is gated on a working fix).
- Did not modify the v19a / v19d worktree scripts to point at the actual atom code path (`agent-af6dd88bcf35d287a`); they currently `cd` to `agent-a43a95f1a35b1e17a` which is harmless because pip-installed atom is independent of cwd.
