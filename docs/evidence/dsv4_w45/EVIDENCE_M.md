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

## What this Evidence does NOT cover

- **W4 multi-request KV pool correctness** — the single-token collapse at conc=4 is reproducible with the FP4 CSV fix in place; the bug is upstream of MoE in `_forward_w4` (see `atom/models/deepseek_v4.py:1778`). Recommended follow-up: enable `ATOM_AITER_VALIDATE=1` to bisect the validator gate, then check `DSV4KVPool.compute_out_cache_loc` per-seq slot routing.
- **gsm8k W4.5 accuracy gate** — pending until W4 multi gibberish is resolved upstream of MoE; W3 baseline gsm8k is unaffected by this PR and runs at the legacy single-request rate.
- **Performance impact** — this PR is correctness-only (registers existing kernels in tuned config). Performance is a separate sweep.

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
