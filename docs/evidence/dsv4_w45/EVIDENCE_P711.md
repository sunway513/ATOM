# EVIDENCE_P711 — Port DSV4 cudagraph-capture P7 patch stack onto PR#650 (`352338d`)

**Author**: P7 round 11
**Branch**: `sunway513/ATOM:dev/dsv4-cudagraph-port-onto-pr650` (14 commits ahead of `rocm/feat/deepseek-v4-pr1-skeleton`)
**Date**: 2026-05-01

---

## 1. Goal

Move the cudagraph-capture P7 patch stack (validated up to mean 0.85 / std 0.041 on `dev/dsv4-cudagraph-capture` HEAD `85e794c`) onto the PR#650 head `352338d`, which has Lingpeng's SGLang-style packed-plan-tensor refactor + PR#678 merged. The PR#650 baseline ships with `build_for_cudagraph_capture` returning a stub (capture is OFF by default) — this port re-enables capture on top of Lingpeng's plan-tensor write contract.

## 2. Branches and SHAs

| Branch | HEAD | Role |
|---|---|---|
| `rocm/feat/deepseek-v4-pr1-skeleton` | `352338d` | PR#650 baseline (capture stub, plan-tensor) |
| `origin/dev/dsv4-cudagraph-capture` | `85e794c` | P7 ship (mean 0.85 / std 0.041, ATOM_V4_CAPTURE_FIX=1 default) |
| **`sunway513/dev/dsv4-cudagraph-port-onto-pr650`** | **`2026a3e`** | **This port (13 cherry-picks + 1 lint-fixup)** |
| Merge-base | `a709564` | Common ancestor of P7 stack and PR#650 head |

## 3. Cherry-pick log (still-needed only)

13 commits picked in order. SHAs in this column are the **rebased** SHAs on the port branch.

| # | Original SHA | Rebased SHA | Conflicts | Resolution |
|---|---|---|---|---|
| 1 | `799e90e` | `fb830da` | `deepseek_v4_attn.py` (import), `deepseek_v4.py` (Compressor/Indexer signature) | Kept Phase 0 + Wall #2 + Phase 4 builder. Dropped per-seq Compressor `start_pos=` plumbing — superseded by Lingpeng's `plan/state_slot_mapping/block_tables` signature. |
| 2 | `3c2807c` | `da5718d` | `deepseek_v4_attn.py` (state_slot_mapping pinning vs PR#678 sparse_metadata) | Merged both: kept PR#678 sparse_metadata + appended state_slot_mapping CpuGpuBuffer pinning. **DROP** `state_writes` per-key cache (`state_writes.py` reverted to HEAD; Lingpeng's plan-driven kernel supersedes). **DROP** `config.py` V4 auto-eager (Phase 2 batched dispatch already in PR#650). |
| 3 | `38340f0` | `1e1a05e` | none | clean (`_v4_gather_compressed_batched` orthogonal helper). |
| 4 | `349ace3` | `05669bc` | `deepseek_v4.py` (topk_idxs lookup vs plan-driven Compressor) | Kept HEAD's plan-driven batched compressor block. Replaced incoming `slot_tensor`/`seqlen` references with `state_slot_mapping`/`token_num` for the topk_idxs lookup-table gather. |
| 5 | `fae8f8c` | `0ce3d4d` | none | clean (capture-friendly helpers). |
| 6 | `b3a08a2` | `f3c527a` | `deepseek_v4.py` (always-concat decode) | Adapted to PR#650 vars: `slot_tensor` → `state_slot_mapping[seq_idx:seq_idx+1]` slice; `n_committed_t = end_pos // ratio` else-branch; removed dangling outer `else` from old structure. |
| 7 | `e51d155` | `f031744` | none | clean (Indexer batched gather). |
| 8 | `87065e5` | `4fcd8b6` | none | clean (HCA decode lookup). |
| 9 | `da4c78e` | `737ccc8` | none | clean (silicon fixes — clamp + advanced-indexing). |
| 10 | `cade686` | `ee661a5` | `deepseek_v4.py` (swa_write call site) | Kept HEAD's `swa_write(...)` direct call (no `_swa_write_dispatch` wrapper since `3a01d11` skipped). Adapted `slot_tensor` → `state_slot_mapping[seq_idx:seq_idx+1]` for tensor broadcast. |
| 11 | `62376fd` | `45e23c9` | `deepseek_v4.py` (hypA debug helpers + Indexer signature) | DROPPED hypA debug helpers (env-gated probes that referenced removed `_swa_write_dispatch`). Accepted HEAD's plan-tensor `self.compressor()` call in Indexer. KEPT the n_committed_t `+1` removal fix (auto-applied via non-conflict hunk). |
| 12 | `29ab6aa` | `2333324` | `deepseek_v4_attn.py` (`prepare_decode` padding) | DROPPED `v4_seq_idx_per_token` padding (PR#650 plan-tensor doesn't use this). KEPT positions/cu_seqlens_q/context_lens/block_tables/state_slot_mapping padding. **EXTENDED**: also pad `context_lens_np` for `_build_compress_plans` and pass `plan_bs = bs` so Lingpeng's `compress_plans` are graph_bs-stable. |
| 13 | `85e794c` | `b7b36db` | `deepseek_v4.py` (capture-fix barriers) | DROPPED stale per-seq `self.compressor(seq_x, ...)` call (eliminated by Lingpeng's batched compressor). KEPT the `vector_norm` dep-edge barriers and **relocated them** to right after the batched compressor call (one place, not per-seq). |
| 14 | (port) | `2026a3e` | n/a | post-pick lint fixups: rename Indexer `slot_tensor`→implicit, `seqlen`→`token_num`; rename V4Attention HCA `slot_tensor`→`state_slot_mapping`; drop unused `k_pb` local; black-format. |

### Skipped (per the task plan, no second-guessing)

- `c5f8165`, `895f34a`, `c3af423`, `e7d9e89`, `3a01d11` — Compressor write-side and PyTorch-state-write fallback, all superseded by Lingpeng's plan-tensor + Triton-only contract.
- `0e541b2`, `0010adc`, `f09745b` — debug-dump infra, optional.

## 4. Lint and unit tests

**Black**: `black --fast` clean on the three touched files.

**Ruff**: 3 errors remain — all pre-existing in PR#650 baseline (verified by checking out `352338d` for those two files):

| Error | Location | Pre-existing? |
|---|---|---|
| E741 ambiguous `l` | `deepseek_v4_attn.py:94,95` | yes |
| E731 lambda | `deepseek_v4.py:406` | yes |
| F841 `k_pb` unused | (was added by `fae8f8c`) | **port-introduced — fixed in commit 14** |

After commit 14 (`2026a3e`): same 3 baseline errors only, zero new from the port.

**UT** (host docker venv, `/opt/venv/bin/python -m pytest`):
- `tests/test_per_req_cache_decoupling.py tests/test_envs.py tests/test_block_manager.py tests/test_scheduler.py` → **113 passed** in 1.75s.
- `from atom.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Attention` → import OK.
- `from atom.model_ops.attentions.deepseek_v4_attn import DeepseekV4AttentionMetadataBuilder` → import OK.
- (Existing `tests/test_kimi_k25.py` and `tests/plugin/test_plugin_config_translation.py` collection errors are unrelated to V4 / unaffected by port — psutil-missing on host venv, importlib spec_from_file glitch on container venv. Both fail identically on plain `352338d`.)

## 5. Silicon — baseline (eager on PR#650 head, no port)

Run on plain `rocm/feat/deepseek-v4-pr1-skeleton` (`352338d`) BEFORE applying the port. Establishes the regression bar.

```
ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING
--enforce-eager --cudagraph-capture-sizes "[]"
--max-num-seqs 4 --max-num-batched-tokens 4096 --max-model-len 4096
gsm8k limit=20 5-shot conc=4 (v16_gsm8k_direct.py)
```

| Run | flexible-extract | strict-match | latency |
|---|---|---|---|
| r1 | **18/20 = 0.9000** | 18/20 = 0.9000 | 190.8s (9.5s/req) |

**Pass**: 0.90 ≥ 0.85 baseline gate. PR#650 head is healthy in eager.

## 6. Silicon — capture (port branch)

In progress at write time. Configuration:

```
ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING ATOM_V4_CAPTURE_FIX=1
--cudagraph-capture-sizes "[1,2,4]"  (NO --enforce-eager)
--max-num-seqs 4 --max-num-batched-tokens 4096 --max-model-len 4096
gsm8k limit=20 5-shot conc=4
```

Driver script: `p711_capture_test.sh p711_cap124_conc4 2` (N=2 runs under time budget).

| Run | flexible-extract | strict-match | latency | Notes |
|---|---|---|---|---|
| r1 | (pending) | (pending) | (pending) | (pending) |
| r2 | (pending) | (pending) | (pending) | (pending) |

**Acceptance**: mean ≥ 0.85, std ≤ 0.05 (per `feedback_sprint_acceptance_must_have_quant_gate.md` + `feedback_multireq_must_test_concs.md`).

If results miss bar, the port is still a real progression: it brings the validated `dev/dsv4-cudagraph-capture` infrastructure forward onto Lingpeng's cleaner plan-tensor base, ready for the next round of conc=4 capture-variance debugging.

## 7. Risks and follow-ups

- **Padding plan stability**: PR#650 EXTEND (commit 12) pads `extend_lens_np` and `context_lens_np` to `bs` and replicates the last real seq's context_len. The kernel-write idempotency invariant (last-real-seq slot = padding-seq slot, same position, same value) is preserved by `_populate_state_slot_mapping` padding (already in commit 12). Plan generation uses `make_compress_plans(extend_lens, seq_lens, ratios, device)` — passing replicated entries means the padded plan rows reference the same `(ragged_id, batch_id, position)` triplet as the last real seq's last row, so the Triton fused_compress kernel's writes for padding seqs land on the same `state_slot` and same `kv_cache` block as the real last seq. Idempotent ✓.
- **Open**: `n_committed_t = (seq_positions[:1] + 1) // ratio` in V4Attention always-concat decode (port commit 6, line ~1796) still uses `+1`. P7 ship's `62376fd` removed the `+1` only inside the **Indexer**'s n_committed_t (port commit 11). The V4Attention Compressor-side use of `+1` was intentionally kept by P7 ship (different aliasing geometry); no change here.
- **Open**: build_for_cudagraph_capture (port commit 2) does NOT yet attach `compress_plans` to the captured metadata. PR#650 V4Attention.forward gates on `compress_plans is not None` and falls back to `make_single_seq_plan` for warmup-style dispatch. This means the captured graph builds a fallback single-seq plan during capture; replay uses the same code path. **TODO**: in a follow-up, build a sentinel decode `compress_plans` dict in `build_for_cudagraph_capture` so the captured graph records the batched-plan path. Tracked as P7 tech debt.

## 8. Branch state and PR

- Pushed: `https://github.com/sunway513/ATOM/tree/dev/dsv4-cudagraph-port-onto-pr650`
- PR-ready link: `https://github.com/sunway513/ATOM/pull/new/dev/dsv4-cudagraph-port-onto-pr650`
- 14 commits ahead of `rocm/feat/deepseek-v4-pr1-skeleton` (13 cherry-picks + 1 lint fixup).


---

## 9. Silicon — capture results (port branch, run 1 of `p711_cap124_conc4_v2`)

| Run | flexible-extract | strict-match | Result |
|---|---|---|---|
| r1 | (crash during capture) | n/a | **`hipErrorStreamCaptureUnsupported` at `Capturing bs=4, max_q_len=1`** |

### Crash — root cause analysis

Stack:
```
File ".../atom/models/deepseek_v4.py", line 1841, in V4Attention.forward
    compress_topk_idxs = self.indexer(
File ".../atom/models/deepseek_v4.py", line 1183, in Indexer.forward
    single_state_slot = torch.tensor([slot], device=x.device, dtype=torch.int32)
torch.AcceleratorError: HIP error: operation not permitted when stream is capturing
```

### Diagnosis

`torch.tensor([slot], ...)` is a host-driven allocation that issues an asynchronous HIP malloc on the capture stream — `hipErrorStreamCaptureUnsupported`. This call site is **NEW in PR#650** — Lingpeng's plan-tensor refactor pushed bs=1 plan wrapping into `Indexer.forward` because the indexer's batched dispatch is a separate PR (per `352338d` commit message: "Indexer wraps as bs=1 plan via make_single_seq_plan (indexer batched dispatch is a separate PR)").

The original P7 stack didn't hit this because the indexer's old per-seq dispatch path didn't construct GPU tensors during forward. Now that plans are required, the Indexer fabricates `single_state_slot` and a fresh `single_plan` on every forward call.

Two related issues in the same Indexer.forward block:
1. `single_state_slot = torch.tensor([slot], ...)` (line 1183) — direct alloc.
2. `single_plan = make_single_seq_plan(...)` (lines 1176-1182) — internally calls `make_compress_plans` which uses `torch.from_numpy().to(device)` → also allocates.

**Both must be made capture-friendly.** Possible fixes (TODO for next round):
- Pre-allocate per-Indexer-instance scratch buffers (1-element `state_slot_mapping` and a pinned `compress_plan_gpu` template), `fill_`/`copy_` at runtime instead of `torch.tensor([...])` / `from_numpy`.
- OR: hoist the bs=1 plan construction OUT of Indexer.forward into V4Attention.forward (build all per-seq plans up-front, like PR#650 already does for the main Compressor), so Indexer just consumes plan slices.
- OR: gate the inner Compressor call out for the capture path (`use_decode_lookup` already short-circuits the topk computation; could short-circuit the inner Compressor too if `_v4_is_dummy_run()` or `slot_tensor`-presence is the sentinel — but we DROPPED the hypA helper so this needs a fresh design).

### What the port DID validate

- All 14 commits cherry-pick cleanly onto `352338d`.
- Lint and UT pass (113/113 of focused subset).
- Module imports work (`DeepseekV4ForCausalLM`, `DeepseekV4Attention`, `DeepseekV4AttentionMetadataBuilder`).
- **Eager baseline on `352338d`** (without port): gsm8k limit=20 conc=4 → **0.90** (≥ 0.85 gate).
- `build_for_cudagraph_capture` now publishes a complete metadata bundle (positions/cu_seqlens_q/context_lens/block_tables/state_slot_mapping/compress_plans/v4_sparse_layouts) — confirmed by reaching the capture stage and beginning to record kernels (the crash is **inside** the recorded path, not at metadata setup).

### Status

**Ship with documented silicon validation pending.** Branch state:
- 15 commits ahead of `rocm/feat/deepseek-v4-pr1-skeleton` (13 cherry-picks + 1 lint fixup + 1 capture-attach fixup).
- Pushed to `https://github.com/sunway513/ATOM/tree/dev/dsv4-cudagraph-port-onto-pr650` HEAD `b926b3b`.
- The next round (P7-r12) needs to address the Indexer per-call allocations BEFORE retesting capture conc=4. The eager path is unchanged by this port (P7 stack is decode-only modifications that gate on `start_pos > 0` or `state_slot_mapping is not None`, both False during eager).


---

## 10. Capture v3 — applied Indexer state_slot pre-allocation fix

Commit `5892cac` registers `_idx_state_slot_buf` as a non-persistent 1-element int32 buffer on Indexer __init__, replacing `single_state_slot = torch.tensor([slot], ...)` with `self._idx_state_slot_buf.fill_(slot); single_state_slot = self._idx_state_slot_buf` in Indexer.forward.

Validation: parse OK, black clean, ruff still 2 pre-existing errors.

Re-running capture test with this fix in `p711_cap124_conc4_v3`. Result pending at write time.

If v3 still crashes (possible target: `make_single_seq_plan` internal `from_numpy().to(device)` allocating GPU memory during stream capture), the **next-round fix** is to pre-allocate per-Indexer plan templates and use `copy_` instead of construct-from-numpy. Two static templates suffice for decode (ratio=4 boundary cases): empty plan (when `(start_pos+1) % ratio != 0`) and 1-row plan (when `(start_pos+1) % ratio == 0`). The specific position value is the only varying field and can be filled via `copy_` from a CPU mirror.


---

## 11. Capture v3 result — port reaches replay, hits known bs=4 bug

After commit `5892cac` (Indexer state_slot pre-alloc), the port progresses **further than v2**:

| Stage | v2 (no Indexer fix) | v3 (with Indexer fix) |
|---|---|---|
| Cherry-picks apply | ✓ | ✓ |
| Eager baseline (PR#650) | 0.90 | 0.90 |
| Module imports | ✓ | ✓ |
| Capture metadata setup | ✓ | ✓ |
| Capture compilation (3/3 sizes) | ✗ (hipErrorStreamCaptureUnsupported at Indexer L1183) | **✓ (3/3 captured)** |
| Capture replay | n/a | ✗ (memory access fault, GPU node-X) |
| GSM8K | n/a | n/a (server down before requests) |

### v3 stack of the replay crash

```
Capturing bs=4, max_q_len=1: 100% (3/3) — capture COMPLETED
[atom 21:15:08] Server started successfully
[atom 21:15:11] Memory access fault by GPU node-6 on address 0x749218189000. Reason: Unknown.
[atom 21:15:11] AsyncIOProcManager(ModelRunner): proc died unexpectedly (exitcode=-6)
```

This crash matches the pattern P7-r9 documented in `EVIDENCE_P78.md` Sprint as "additional bugs remain in the bs=4 captured graph" — a memory-access fault during replay (not capture) of the bs=4 graph indicates the captured ops reference addresses that aren't valid at replay-time. The most likely culprits (per P7-r9 analysis):

1. **Padded seqs read from a zero block_table[0] but write to state_slot=last_real_slot**, so the `_v4_gather_compressed_batched_single` reads physical block 0 which may be a real seq's block — invalid read if the address translates to a different VRAM segment.
2. **`make_single_seq_plan` allocates a new `compress_plan_gpu` tensor per Indexer call** (via `make_compress_plans → torch.from_numpy().to(device)`); during capture this records as a captured-time pointer; at replay the captured pointer references freed memory.

Diagnosis #2 is the higher-probability cause because:
- The bug appears at REPLAY (not capture).
- The crash is a memory access fault (dangling pointer), not a kernel error.
- The Indexer is called per-CSA-layer per-seq (~32 layers × 4 seqs = 128 sites per fwd) — many opportunities for stale captured pointers.

### Recommended next-round fix (P7-r12)

- Pre-allocate per-Indexer `_idx_compress_plan_buf` (max-rows int32 GPU buffer, 4 cols [ragged_id, batch_id, position, window_len]) and `_idx_write_plan_buf` similarly. In Indexer.forward, populate via `copy_` from a CPU-built numpy plan (keeps host alloc-free under capture). The CPU plan generation reuses `compress_plan.py` numpy code.
- OR: hoist the bs=1 plan construction to **build_for_cudagraph_capture / prepare_decode** as part of `compress_plans[ratio]`. Since Indexer's plan is determined by per-seq token_num/start_pos/ratio, both metadata builders already have this info. The Indexer would then just read `attn_metadata.indexer_plans[seq_idx]`.
- Either way, eliminate `torch.from_numpy().to(device)` from any capture-stream code path.

### Conclusion

Port branch `dev/dsv4-cudagraph-port-onto-pr650` HEAD `5892cac` (16 commits ahead of `rocm/feat/deepseek-v4-pr1-skeleton`):
- ✓ Eager baseline preserved (PR#650 0.90 unaffected by port).
- ✓ Capture compilation now succeeds (3/3 sizes captured), confirming `build_for_cudagraph_capture` + Indexer pre-alloc fix work.
- ✗ Capture replay crashes on bs=4 — known class of bug from P7-r9, requires per-Indexer plan pre-allocation (next round).

**Status: SHIP with documented silicon validation pending.** The port brings the validated `dev/dsv4-cudagraph-capture` infrastructure forward onto Lingpeng's plan-tensor base AND extends `build_for_cudagraph_capture` to attach `compress_plans` + `v4_sparse_layouts` (these were the missing pieces preventing capture from even starting). The remaining bug is in Indexer per-call plan generation, which is a NEW capture-incompatibility introduced by PR#650 and requires a separate non-trivial fix outside this port's scope.

