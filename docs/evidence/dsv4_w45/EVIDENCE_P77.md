# P7-r7 Empirical Findings — DSV4 CUDAGraph Capture Isolation Test

**Date:** 2026-04-30 (P7 round 7, 1M-context Opus 4.7)
**Branch (test artifact):** `plan/dsv4-w45-flydsl-blockscale-moe`
**Branch (atom code under test):** `dev/dsv4-cudagraph-bisect-r5` @ `cade686`
**Status:** Multi-request capture isolation **mechanism PROVEN sound** by 5 hard-correctness tests; gsm8k variance root cause is therefore **NOT** in the per-seq slot dispatch primitive. P7-r7 falsifies the P7-r6 candidate hypothesis (per-iteration baked Python ints).

## TL;DR

P7-r6 proposed that the conc=4 capture variance was caused by Python ints (`seq_offsets[seq_idx]`, `slots[seq_idx]`, `start_pos_per_seq[seq_idx]`) being baked into the captured graph during the per-seq Python loop in `V4Attention.forward` (`atom/models/deepseek_v4.py:1948-2272`).

P7-r7 wrote a **micro-fixture** that mirrors the exact dispatch primitive (per-seq tensor view of `state_slot_mapping`, in-place index_put_ on shared `swa_kv`, advanced index_select read, constant-shape gather + cat with `n_committed_t` mask) **without** the real Triton kernels / FP8 quant / sparse_attn — so any failure would be exclusively attributable to the cudagraph capture mechanism itself.

**All 5 tests pass on real ROCm GPU** (MI355X, torch 2.10.0+rocm7.2.2):

| Test | What it proves |
|---|---|
| `test_swa_dispatch_unshuffled_replay` | Capture@bs=4 + replay with same `state_slot_mapping=[0,1,2,3]` produces per-slot bit-identical output to fresh eager. Capture infra trivial-case sound. |
| `test_swa_dispatch_shuffled_replay` | Capture@bs=4 with `[0,1,2,3]` then replay with shuffled `[3,2,1,0]` — per-slot output bit-identical to fresh eager at the **shuffled** slot indices. **Per-tensor slot dispatch IS resolved correctly at replay.** |
| `test_swa_dispatch_state_persists_across_replays` | Replay with disjoint slot set `[4,5,6,7]` after capture@`[0,1,2,3]` does NOT mutate slots `[0..3]`. Captured graph mutates only the slots referenced by the live `state_slot_mapping` view. |
| `test_multi_layer_shuffled_replay` | 8-layer stack capture+shuffled replay also bit-exact. **No cross-layer pool aliasing breaks isolation.** |
| `test_consecutive_replays_deterministic` | Two replays with identical state and inputs produce bit-identical output. **No non-deterministic kernels or uninitialized-memory reads in the captured intermediates.** |

## Implications

The remaining hypothesis space (after P7-r6 falsified Triton-stream-race + padding):

1. ~~Per-iteration baked Python ints~~ — **FALSIFIED by P7-r7 unit tests above.** The micro-fixture has the exact same Python-int loop shape and pattern, captures, and replays correctly under shuffle.
2. ~~Inter-iteration state aliasing on shared swa_kv across iters~~ — **FALSIFIED by `test_swa_dispatch_state_persists_across_replays`.** Iter `i+1` writes do not clobber iter `i` writes when slots differ; same-slot write→read within an iter is correctly serialized.
3. ~~Cross-layer pool aliasing~~ — **FALSIFIED by `test_multi_layer_shuffled_replay`.** Stacking 8 layers with their own swa/comp buffers and shared `state_slot_mapping` view still produces bit-exact output.
4. ~~Non-determinism in captured intermediates~~ — **FALSIFIED by `test_consecutive_replays_deterministic`.**

What remains:

- **A. Indexer's compressor inner state pool semantics.** Real V4 has TWO separate slot-indexed pools: `Compressor.kv_state`/`score_state` (per-layer state-cache) AND `Indexer.compressor.kv_state`/`score_state` (per-CSA-layer Indexer's own compressor). The P7-r6 hypothesis listed this as "not yet bisected." P7-r7's micro-fixture simulates ONE swa+comp pair per layer, not the dual Indexer+main split. If the Indexer compressor's state writes interact with the main compressor's state reads via the captured graph in a way the micro-fixture does not, that bug remains live.
- **B. Real Triton kernel determinism.** The micro-fixture uses pure torch arithmetic (no Triton kernels). `update_compressor_states` (Triton, in-place ring-buffer write) and the v4 sparse_attn kernel (in real V4) may have non-determinism under multi-request decode that the micro-fixture cannot expose. The PT-state-write A/B in P7-r6 partially probed this — PT path was **also** variant at conc=4 — so it is not pure Triton race, but a kernel-level numerical issue in the captured graph cannot be ruled out without Triton-equivalent torch ops being run inside the captured pool.
- **C. Cross-request slot recycling lifecycle.** When a request finishes mid-decode and its slot is freed, the next prefill (eager) reset zeros the slot via the `start_pos == 0` branch in `V4Attention.forward:1979-1986`. This branch is dead in capture (capture has start_pos=1). If a slot is recycled into the SAME captured-decode batch without an intervening eager prefill that touches the recycled slot, stale state from the prior tenant leaks. (Routes through eager when scheduled with prefill, but verifying this requires scheduler tracing.)
- **D. The `_v4_get_seq_metadata()` num_seqs override.** `num_seqs` is set to `md.block_tables.size(0)` which at replay equals **scheduled_bs**, not graph_bs. Captured graph unrolls the loop to `range(num_seqs_at_capture) = range(graph_bs_at_capture)` because the loop is Python-evaluated during trace. At replay, captured loop body runs `graph_bs` iters regardless of replay-time `num_seqs` value. If at replay `scheduled_bs < graph_bs`, the tail iters process stale forward_vars data — but the **multi-bs capture sweep [1,2,3,4]** in P7-r6 falsified the padding hypothesis.

The strongest remaining hypothesis is **A (Indexer compressor)** because:
- It is the only sub-system the micro-fixture explicitly does not model.
- Indexer is enabled only for CSA layers (compress_ratio==4); HCA layers (ratio>=8) skip Indexer. Conc=1 with one decode token traverses Indexer for CSA layers exactly as conc=4 does (Indexer.forward is called per-seq from V4Attention's loop). So Indexer involvement does not by itself differ between conc=1 and conc=4 — but its inner Compressor's `update_compressor_states_dispatch` is called per-seq, and the inner Compressor's own `kv_state`/`score_state`/`kv_cache` pools are SEPARATE from the outer Compressor's pools yet share the same slot indexing.

A high-leverage P7-r8 experiment: **disable the Indexer entirely** (force HCA-style code path on CSA layers, accepting accuracy loss) and run conc=4 capture gsm8k. If variance disappears, Indexer's slot-state pool is the contamination source.

## Test artifact

New file: `/home/pensun/ATOM-lingpeng/tests/test_dsv4_capture_isolation.py` (572 lines, 5 tests).

Run command (in atom_dsv4_feat container):
```bash
sudo docker exec -e HIP_VISIBLE_DEVICES=0 atom_dsv4_feat \
    /opt/venv/bin/python -m pytest \
    /workspace/ATOM-lingpeng/tests/test_dsv4_capture_isolation.py -v -s
```

Expected output:
```
collected 5 items
test_swa_dispatch_unshuffled_replay PASSED
test_swa_dispatch_shuffled_replay PASSED
test_swa_dispatch_state_persists_across_replays PASSED
test_multi_layer_shuffled_replay PASSED
test_consecutive_replays_deterministic PASSED
==== 5 passed in 1.35s ====
```

## P7-r7 audit of V4Attention.forward inner loop

Detailed file walk of `atom/models/deepseek_v4.py:1948-2272` (V4Attention per-seq dispatch loop) confirms:

- Loop counter `for seq_idx in range(num_seqs)` unrolls to `num_seqs_at_capture` Python iterations baked into the graph.
- `slot_tensor = state_slot_mapping_long[seq_idx:seq_idx+1]` is a tensor view (Python-int slice on tensor produces aten::slice op recorded into graph; at replay the slice resolves into the live `state_slot_mapping_long` allocated at the captured pool slot, which itself is the result of `state_slot_mapping.long()` — this `.long()` op is captured and re-runs at replay producing a fresh int64 tensor with REPLAY-TIME values from the pinned `state_slot_mapping` forward_var). **This indirection is the exact pattern the unit test validates.**
- `start_pos_per_seq[seq_idx]` evaluated at trace time becomes a Python int CONSTANT in the captured graph. **However:** `build_for_cudagraph_capture` sets `start_pos_per_seq_cpu = ones(bs)` so capture's `start_pos == 1` for all seqs. The `start_pos == 0` branch (line 1979) — which contains the `swa_kv[slot].zero_()` reset — is therefore **NOT** baked. ✅
- All write-to-state operations under `slot_tensor is not None and start_pos > 0` (the always-true capture branch) use tensor-driven indices. Verified by code walk; `_v4_gather_compressed_batched_single`, `_v4_scatter_compressed_conditional`, `swa_kv[slot_tensor, pos_mod_t] = ...`, and Compressor's batched `_update_compressor_states_dispatch(state_slot_mapping=slot_tensor)` all have pure-tensor operands.
- Python-int `slot` is passed only to `update_compressor_states_dispatch(slot=slot)` as a SINGLE-SEQ FALLBACK (line 1116), but the batched path (line 1075) is taken because `slot_tensor.numel() > 0` is always true at capture for V4 dispatch.
- `_v4_is_dummy_run()` returns False during capture (`Context.is_dummy_run` defaults to False; `build_for_cudagraph_capture` does not set it). So all `if not _v4_is_dummy_run()` block_table writes are baked into the graph. ✅

No remaining `[seq_idx]` or `int(...)` calls inside the active capture branch were found that bake stale runtime state.

## Recommendation for P7-r8

Given P7-r7 narrows the bug from "the V4Attention per-seq loop" to "**not** the V4Attention per-seq loop primitive", the next sprint should:

1. **A. Bisect Indexer involvement.** Add an env flag `ATOM_V4_DISABLE_INDEXER_TOPK=1` that replaces Indexer's `topk_idxs` output with the HCA decode lookup table path (just window topk, no learned indexing). Capture@conc=4 gsm8k limit=20. If accuracy stabilizes, Indexer's stateful Compressor sub-pool is the culprit and needs the same `slot_tensor=` capture-friendly refactor that the outer Compressor already received in P7-r5.
2. **B. Bisect cross-layer state.** Capture+replay with only first 16 layers active (envvar to short-circuit later layers' V4Attention.forward to a no-op identity). If conc=4 variance disappears at ≤16 layers but reappears at ≥32, accumulating state-write error is the issue.
3. **C. Tracer-based diff.** Use `torch.profiler` capture+replay traces to compare iter-by-iter output norms between (i) eager bs=4 and (ii) captured bs=4 replay with same inputs. The first layer where outputs diverge identifies the culprit module exactly.

P7-r7 cannot complete A/B/C in remaining budget but lands the test artifact and audit so P7-r8 has a starting point.

## What P7-r7 did NOT do

- Did not run silicon validation (`gsm8k limit=20` × 3+ runs at conc=4). Wall-clock budget was consumed by:
  - GPU preflight + container atom-package verification
  - Worktree + branch topology resolution
  - Reading prior evidence + 2272-line V4Attention.forward audit
  - 5-test fixture authorship + iterative debugging on real ROCm
- Did not modify `atom/models/deepseek_v4.py` (no fix candidate identified after the unit-test mechanism was proven sound — making a speculative change without evidence would risk regressing v19a eager).
- Did not push the test artifact to `dev/dsv4-cudagraph-capture` — landed on `plan/dsv4-w45-flydsl-blockscale-moe` (the docs branch) so capture-branch surface stays clean.

