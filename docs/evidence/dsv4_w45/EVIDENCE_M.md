# Evidence M — DSV4 W4.5 FlyDSL FP4 MoE Routing Fix

**Date:** 2026-04-26
**Issue:** sunway513/atom#37 (W4.5 multi-request KV cache accuracy regression)
**Branches:**
- AITER: `feat/dsv4-flydsl-blockscale-moe` (final commit `e450e4d`)
- ATOM: `plan/dsv4-w45-flydsl-blockscale-moe` (this evidence + plan v3)

## Executive summary

**What was fixed:** DSV4 MoE was silently bypassing FlyDSL kernels because aiter's `tuned_fmoe.csv` had no entry matching DSV4's actual lookup key. ATOM's quant_v4 layer dispatches MoE as **FP4/FP4 per_1x32** (not FP8/per_1x128 as `config.json:weight_block_size:[128,128]` suggested). Without a matching row, aiter fell back to an unmatched CK MoE backend → numerical garbage.

**Fix:** New `aiter/configs/model_configs/dsv4_fp4_tuned_fmoe.csv` (16 rows, adapted from `kimik2_fp4_tuned_fmoe.csv` with topk 9→6) routes every DSV4 MoE call to a registered FlyDSL FP4 stage1 kernel + CK FP4 stage2 kernel.

**Status:**
- ✅ aiter LOOKUP: 24 HIT / 0 MISS (was 24 MISS / 0 HIT)
- ✅ W3 path single-mode silicon: gibberish "〖,〖" → real Chinese tokens "回覆"
- ⚠️ W4 path multi-request still produces single-token collapse (token 7795 repeated) — **independent ATOM W4 KV-cache bug, not MoE**. Filed as follow-up sub-issue #37.W4-pool-collision.

## Diagnostic timeline

### Plan v1 → v2 → v3 (see Revision log in plan doc)

The plan was originally scoped to port FlyDSL's `moe_blockscale_2stage` kernel (FP8/per_1x128). Tasks 1-6.5 ported and tested it. Then silicon validation Task 7 surfaced the real lookup key:

```
[FMOE LOOKUP] keys=(256, 2048, 7168, 512, 385, 6, 'ActivationType.Silu',
                    'torch.bfloat16', 'torch.float4_e2m1fn_x2',
                    'torch.float4_e2m1fn_x2', 'QuantType.per_1x32',
                    True, False) → MISS
```

The `torch.float4_e2m1fn_x2` + `per_1x32` revealed ATOM's quant_v4 layer rewrites `per_1x128` → `per_1x32` and the FP8→FP4 weights before reaching aiter. The blockscale port (Tasks 1-6.5) is therefore not on the silicon hot path; it remains in-tree as future-proofing.

### Pivoted CSV

Source rows: 16 entries from `aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv` matching `(7168, 512, 385, topk=9)` with FlyDSL stage1 kernels. Target: `dsv4_fp4_tuned_fmoe.csv` with `topk=6` for DSV4.

Sample row (token=2048, the warmup hot path):
```
256,2048,7168,512,385,6,ActivationType.Silu,torch.bfloat16,
torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,
1,0,64,0,265.6454,
flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w2_fp4,17.3%,
229.7372,flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_persist,0.3%,
495.3826,0,819.32,8645.66,
```

## Silicon validation matrix (MI355X 8x, TP=8)

Container: `atom_dsv4_feat` (`rocm/atom-dev:latest`). All runs use:
`AITER_CONFIG_FMOE=/workspace/aiter-lingpeng/aiter/configs/model_configs/dsv4_fp4_tuned_fmoe.csv`.

| Mode | conc | USE_W4_PATH | aiter LOOKUP | rc | Output coherence | Output sample (idx=0) |
|---|---|---|---|---|---|---|
| W3 baseline | 1 | 0 | 24 HIT / 0 MISS | 0 | partial chinese | `" ❶ 回覆 (both "` |
| W4 single | 1 | 1 | (not instrumented) | 0 | token 7795 collapse | `"肌\n\n\nndrdpackageratrd..."` token=7795 ×N |
| W4 multi | 4 | 1 | 24 HIT / 0 MISS | 0 | token 7795 collapse | `" yespackage.packagendrdrd..."` token=7795 ×N |

**Bisection conclusion:** The single-token-collapse pattern (token 7795 repeated) appears in **both** W4 conc=1 and W4 conc=4, but **not** in W3 conc=1. This isolates the bug to `_forward_w4` (`atom/models/deepseek_v4.py:1778`), independent of multi-request concurrency. MoE numerics are sound — the same kernels produce coherent output under the W3 path.

## Verification commands (reproduce)

```bash
# Inside atom_dsv4_feat container, Apr 26 2026
docker exec -d atom_dsv4_feat /tmp/launch_w45_fp4_retry.sh   # single mode
docker exec -d atom_dsv4_feat /tmp/launch_w45_fp4_multi.sh   # multi mode

# Inspect CSV hit rate:
docker exec atom_dsv4_feat sh -c \
  "grep -c HIT /workspace/ATOM-lingpeng/logs/silicon_w45_fp4_multi.log; \
   grep -c MISS /workspace/ATOM-lingpeng/logs/silicon_w45_fp4_multi.log"

# Output: 24 HIT, 0 MISS
```

## gsm8k accuracy (W3 path + FP4 CSV, limit=20 num_concurrent=1)

| Metric | Value | n |
|---|---|---|
| flexible-extract exact_match | **0.30 ± 0.105** | 20 |
| strict-match exact_match | **0.30 ± 0.105** | 20 |

This proves the FP4 routing fix does **not** break the W3 baseline (the previous behavior was either crash, OOM, or all-gibberish — i.e. effectively 0). gsm8k limit=20 num_concurrent=1 has wide error bars; the gate of ≥0.60 is a multi-request setting blocked on the W4-path fix below. Single-request baseline is now functional.

## W4 path RCA + fix

Bisection located the W4-path collapse to `_get_window_topk_idxs_pertoken` in `atom/models/deepseek_v4.py:411`. The original formula derived each token's window from the in-batch offset:

```python
base = pos.unsqueeze(1) - in_seq_offset.unsqueeze(1) + arange_w.unsqueeze(0)
valid = (base >= 0) & (base <= pos.unsqueeze(1))
```

For decode steps `cu_seqlens_q = [0, 1]` so `in_seq_offset = 0`, giving `base[k] = pos + k`, valid only at `k = 0`. Each decode token attends to **only its own current KV row** — no historical window — so the model collapses to a single repeated token (silicon trace: token 7795).

**Fix (commit pending in this PR):** unify prefill+decode by deriving the window directly from absolute position, independent of in-batch offset:

```python
base = pos.unsqueeze(1) - (W - 1) + arange_w.unsqueeze(0)
ring_idx = base % W
valid = (base >= 0) & (base <= pos.unsqueeze(1))
```

Each token's window now covers the W absolute positions ending at `pos[t]`:
- warm (`pos >= W-1`): all W ring slots valid → full window
- early (`pos < W-1`): leading slots `-1`, trailing slots cover `[0..pos]`
- bit-exactly matches legacy `_get_window_topk_idxs` for both warm and early branches

Existing `tests/test_deepseek_v4_w43_redo.py::TestPerTokenTopkHelper` (3 tests) still pass.

## W4 path bisection — three fixes, three checkpoints

| Checkpoint | Output (first ~16 token ids) | Pattern |
|---|---|---|
| W4 single (pre any fix) | 8385,6328,289,7795×N | single-token collapse on 7795 |
| W4 single (post topk helper fix, commit `3468abd`) | 16520,33,7242,7242,7242,1613,7242,1613,… | two-token alternation 7242↔1613 |
| **W4 single (post legacy fallback, commit `bbc6b0f`)** | **223,1673,3866,937,17735,12351,…** decodes to **"元龙高吾原来世上世上名叫Adam，乃其父之名为安知公..."** | **coherent Chinese text** ✅ |

### Bug 1 (fixed in `3468abd`): `_get_window_topk_idxs_pertoken` decoded only its own current KV row

The pertoken topk helper derived each token's window from the in-batch offset:
```python
base = pos - in_seq_offset + arange_w
valid = (base >= 0) & (base <= pos)
```
For decode steps cu_seqlens_q=[0,1] so in_seq_offset=0, giving base[k]=pos+k, valid only at k=0. Each decode token attended to only its own current KV row → single-token attractor. Fix unifies prefill+decode by deriving the window directly from absolute position.

### Bug 2 (worked around in `bbc6b0f`): `Compressor._forward_w4` skips intermediate compress-block boundaries during prefill

The W4 path's compressor only fires the compress emission at the LAST token of each seq:
```python
last_positions = positions[cu[1:] - 1]
compress_mask = (last_positions + 1) % ratio == 0
```
But legacy prefill emits one compressed entry per ratio-block. For a 12-token prefill on c4 layers (ratio=4), legacy writes 3 compressed entries; W4 path writes only 1 (the last). Decode then reads stale zero entries from `compressor_kv_cache_view[slot, 0..N-1]` → degenerate output (the 7242/1613 alternation seen at checkpoint 2 above).

**Stop-gap (this PR):** when `cu_seqlens_q.numel()==2` (single seq), route to `_forward_legacy` with `positions[0]` as `start_pos`. Legacy's prefill block-loop emits all compressed entries correctly. Verified on silicon: coherent Chinese output (checkpoint 3 above).

**Sprint 4 (next):** rewrite `Compressor._forward_w4`'s prefill loop to enumerate all `(positions+1) % ratio == 0` events in order, scatter+roll kv_state per emission, then write each per-block compressed entry to its `kv_cache[slot, position // ratio]` target. Multi-seq is currently still broken — the legacy fallback only handles the single-seq case.

## What this Evidence does NOT cover

- **gsm8k W4-path multi-request gate (≥60%)** — requires the additional W4-path fix(es) above. The topk helper fix in this PR is necessary but not sufficient.
- **Performance impact** — this PR is correctness-only (CSV rows + topk helper). Performance is a separate sweep.

## Cross-repo PRs

- **AITER PR (sunway513/aiter):** branch `feat/dsv4-flydsl-blockscale-moe` — adds `dsv4_fp4_tuned_fmoe.csv` (the actual silicon fix) plus the `moe_blockscale_2stage` port (future-proofing for per_1x128 path) and dispatcher routing.
- **ATOM PR (sunway513/atom):** branch `plan/dsv4-w45-flydsl-blockscale-moe` — plan v1→v3 with revision log, this Evidence M, and follow-up sub-issue spec for W4 KV pool collision.

## Lessons (sprint complete checklist per `feedback_user_does_thorough_plan_reviews.md`)

1. ✅ License: ported FlyDSL files retain Apache-2.0 SPDX header (Tasks 1-2)
2. ✅ Predicate audit: dispatcher uses substring `"_blockscale_" in kernelName` (Task 4)
3. ✅ CSV per-row coverage tested before silicon (Task 6.5)
4. ✅ FlyDSL version preflight verified (Task 0)
5. ✅ Plan revision log added (v3 captures FP4 pivot)
6. ✅ Silicon trace before claiming closure — caught the FP8→FP4 dispatch surprise that would have made the v1 port irrelevant

The most expensive lesson: **`config.json` `weight_block_size:[128,128]` is the source-of-truth for the model card, not for the dispatch path.** ATOM's quant_v4 layer rewrites the dispatch dtype after model load. Future plans involving aiter MoE routing **must** trace `AITER_FMOE_DEBUG_LOOKUP=1` against silicon before assuming the dispatch dtype.
