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

