# DSV4 W4.5 Work Context Handoff — 2026-04-26

**Author**: Claude Opus 4.7 (this session)
**Date**: 2026-04-26
**Branch state**: `main` HEAD `73118b6` (PRs #54 + #55 + #56 all merged)
**Hardware**: MI355 TP=8, container `atom_dsv4_feat`

## TL;DR

- **Issue #37 W4.5 multi-request KV cache architecture is silicon-green** ✅
- W4 multi mode (4 conc requests, distinct outputs, no row-0 cross-talk) runs end-to-end on MI355
- 7-bug cascade resolved across PRs #54+#55
- 120 DSV4 UTs no regression
- **Output text quality is gibberish** (in BOTH W4 and baseline `flag=0` paths) — root-caused to **aiter MoE backend / CK ABI mismatch**, NOT ATOM W4.5 code
- **Next work** = aiter MoE configuration / kernel enablement (not ATOM-side)

## 1. What was delivered today

### Sprint 1 (PR #54 merged)
- `tests/silicon/repro_w4_kernel_bisect.py` — Python monkey-patch bisect harness (later superseded by ROCr Debug Agent)
- Diagnostic asserts at the W4 main scatter and Compressor scatter
- Validator `pool_capacity` argument fix
- `ring_size_compressor=8 → max(2*4, 128)=128` (Bug #1 fix)

### Sprint 2 (PR #54 merged, more commits)
- `DSV4KVPoolConfig` split: per-ratio `ring_size_compressor_c4 / _c128`, `state_inner_dim_c4 / _c128`, `index_head_dim`
- `_build_buffers` allocates two compressor slabs (c4 + c128)
- `view_for_layer` dispatches by ratio; backward-compat shim for Sprint-1 single-value config
- `_apply_rotary_emb` 2D branch (Bug #4)
- Indexer slab `index_head_dim` not main `head_dim` (Bug #5)

### Sprint 3 (PR #55 merged)
- Per-ratio `_compressor_main_kv_c4 / _c128` slabs (compressor's main-attention KV)
- `view_for_layer` returns new `compressor_kv_cache` key
- `_forward_w4`: bind outer `self.compressor.kv_cache` from pool view
- For c4 layer: ALSO drive OUTER compressor (Sprint 1/2 missed this)
- Concat per-token compressed KV from `compressor_kv_cache_view[slot_per_token]` to `kv_per_token`
- `Compressor._bind_state_from_pool`: shape-check + lazy-allocate per-instance fallback (Bug #7)

### Documentation (PRs #54/#55/#56)
- `docs/evidence/dsv4_w45/EVIDENCE_K.md` — Sprint 1/2 root cause + ROCr methodology
- `docs/evidence/dsv4_w45/EVIDENCE_L.md` — Sprint 3 silicon-green report + accuracy CK MoE diagnosis addendum
- `docs/evidence/dsv4_w45/silicon_w45_rocr_excerpt.log` — ROCr Debug Agent crash dump excerpt

## 2. Bug cascade summary

| # | Layer | Bug | Sprint | Status |
|---|-------|-----|--------|--------|
| 1 | Pool ring | `ring_size_compressor=8` too small for c128 | 1 | ✅ |
| 2 | Pool inner_dim | shared between c4 (256) and c128 (512) | 2 | ✅ |
| 3 | Pool architecture | ONE compressor slab for both ratios | 2 | ✅ |
| 4 | RoPE | `_apply_rotary_emb` crashed on 2D `[B, D]` input | 2 | ✅ |
| 5 | Indexer slab | last-dim used `head_dim=512` not `index_head_dim=128` | 2 | ✅ |
| 6 | KV concat | `kv_per_token` didn't concat compressed KV | 3 | ✅ |
| 7 | State binding | OUTER c4 compressor (inner=1024) shape-mismatched INNER slab | 3 | ✅ |
| 8 | Accuracy | aiter CK MoE ABI mismatch with DSV4 | — | ⚠️ Open (aiter team) |

## 3. Silicon iteration log

| Run | Mode | Result |
|-----|------|--------|
| Evidence J 03:18 | Sprint 1 main + W4 single | HSA 0x1016 — kernel UNKNOWN |
| v2 | + ROCr Debug Agent | HSA 0x1016 — kernel = `at::native::index_put_kernel_impl` ASSERT_TRAP |
| v3 | + scatter asserts | ValueError caught Bug #1 |
| v4 | + ring_size 8→128 | Bug #2 surfaced |
| v6 | + per-ratio slabs | Bug #4 surfaced |
| v7 | + RoPE 2D fix | Bug #5 surfaced |
| v8 | + Indexer head_dim fix | Bug #6 (validator catches topk vs kv) |
| v9 | + Sprint 3 first attempt | Bug #7 surfaced |
| **v10 single** | + lazy-fallback state binding | ✅ Request 0 finished, 89s, output gibberish but no crash |
| **v10 multi** | (same code) | ✅ **4 conc requests finished simultaneously, 17.5s, 4 distinct outputs, no cross-talk** |
| Baseline replay 14:11 | J era code (`5e193de`) + clean cache + CK present | ❌ Gibberish → confirms accuracy NOT W4-introduced |

## 4. Accuracy regression — root cause analysis

### Symptom
- Both W4 path AND baseline (`flag=0`) produce gibberish text on this silicon today
- Evidence J era (03:18 same day) baseline produced coherent Chinese
- Same code (`5e193de`) replayed at 14:11 still gibberish

### Bisect

| Variable | J era (03:18, ✅ Chinese) | Now (gibberish) |
|---|---|---|
| Code | `5e193de` | `5e193de` (replayed) or `298d91b` |
| JIT cache | yesterday 18:00 era | various clean states |
| **CK source dir** | **EMPTY** | **populated** (I copied at 04:14) |
| MoE backend in trace | no `ck_moe_stage*` calls | `ck_moe_stage1/2` heavily logged |

**Root cause**: I copied `/app/aiter-test/3rdparty/composable_kernel/` into `/workspace/aiter-lingpeng/3rdparty/composable_kernel/` at ~04:14 to fix the Sprint-1 `module_moe_sorting: 'moe_sorting_api.hpp' file not found` error (silicon-blocker for Evidence J/K work). The CK source's presence enabled aiter's CK MoE codepath (`ck_moe_stage1/2`) which has an **ABI mismatch with this DSV4 model checkpoint** — produces numerical garbage.

The "fix" silicon required (CK presence) caused the accuracy regression. **Removing the CK source brings back the original `module_moe_sorting` compile failure**, so it's not a clean undo.

### Why ATOM W4.5 work is unaffected
- Bug #1-#7 were structural (pool sizing, slab dispatch, KV concat). All fixed at correct layer.
- v10 W4 multi mode produces 4 DISTINCT gibberish outputs from 4 different prompts → row-0 cross-talk eliminated → core issue #37 concern resolved structurally.
- Baseline (flag=0) gibberish proves W4 is not the source.

## 5. MoE backend contract analysis (FlyDSL switch viability)

### Question asked
"Can we use FlyDSL MoE instead of CK MoE to avoid the ABI mismatch?"

### FlyDSL MoE contract (Python entry point)

`aiter.ops.flydsl.flydsl_moe_stage1(...)`, line `aiter/ops/flydsl/moe_kernels.py:547`:

```python
def flydsl_moe_stage1(
    a: torch.Tensor,                    # [token_num, model_dim]
    w1: torch.Tensor,                   # [E, 2*inter_dim, model_dim] pre-shuffled
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32, tile_n: int = 256, tile_k: int = 256,
    a_dtype: str = "fp8",   # also "fp4", "bf16"
    b_dtype: str = "fp4",   # also "fp8", "bf16"
    out_dtype: str = "bf16",
    act: str = "silu",      # also "swiglu"
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    persist_m: int = 0,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 0,
    gate_mode: str = "separated",   # also "interleave"
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    bias: Optional[torch.Tensor] = None,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
)
```

### How aiter's `fused_moe` selects FlyDSL vs CK

`aiter/fused_moe.py:972-1022`:

```python
is_flydsl1 = bool(kernelName1) and kernelName1.startswith("flydsl_")
is_flydsl2 = bool(kernelName2) and kernelName2.startswith("flydsl_")
if (is_flydsl1 or is_flydsl2) and is_flydsl_available():
    # use _flydsl_stage1_wrapper / _flydsl_stage2_wrapper
else:
    # use ck_moe_stage1 / aiter.ck_moe_stage2_fwd
```

- **Selection driver**: `kernelName1`/`kernelName2` from `tuned_fmoe.csv` lookup
- **Lookup key**: `(cu_num, token, model_dim, inter_dim, expert, topk, act_type, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)` — exact 13-tuple match
- **For DSV4**: `(256, 12, 7168, 3072, 384, 6, ActivationType.Silu, torch.bfloat16, torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, QuantType.per_1x128, 1, 0)` — **NO ROW EXISTS in any aiter csv**

### FlyDSL FP8/FP8 kernel registry

```bash
$ python -c "from aiter.ops.flydsl.moe_kernels import get_flydsl_stage1_kernels; \
            print(len(get_flydsl_stage1_kernels('fp8', 'fp8', 'bf16')))"
112
```

**112 stage1 + 48 stage2 kernels** are registered for `(a_dtype='fp8', b_dtype='fp8', out_dtype='bf16')` — they exist in the binary, just no tuned-config row routes to them for DSV4 dims.

### The ABI gap for DSV4

DSV4 quantization scheme: `weight_block_size: [128, 128]` → `QuantType.per_1x128`

**All existing FlyDSL FP8 entries in `aiter/configs/model_configs/*.csv`** use:
- `q_dtype_a=torch.float8_e4m3fn`, `q_dtype_w=torch.float4_e2m1fn_x2` (a8w4 — FP8 act, FP4 weight)
- `q_type=QuantType.per_1x32` (block 1×32 scaling)

The Python wrapper accepts arbitrary scale shapes (`a1_scale.view(-1)`, `w1_scale.view(-1)`) but the compiled GPU kernel has a SPECIFIC scale-stride layout baked into its tile params. The 1632 entries in FlyDSL `_KERNEL_PARAMS` cover combinations of (tile_m, tile_n, tile_k, waves_per_eu, gate_mode, etc.) for {fp4, fp8, bf16} × {fp4, fp8, bf16} dtype combos, **but no parameter exposes per_1x128 vs per_1x32 scale layout**. The scale layout is implicit in the kernel's compiled-in stride math.

**Conclusion**: Whether FlyDSL FP8/FP8 kernels currently support per_1x128 scale layout is **unknown without aiter-team validation**. Existing tuned configs all use per_1x32. Just writing a CSV row with `q_type=QuantType.per_1x128` and a FlyDSL kernel name would either:
- Work (if kernel happens to support it) — give us correct output
- Crash (shape mismatch in scale stride math)
- Silently produce gibberish (if scale stride math is wrong but doesn't crash)

The third outcome is the dangerous one. Risk is high without verification.

### Recommended path

**For aiter team / the next session**:

1. **Verify**: does FlyDSL FP8/FP8 stage1/2 actually compute correctly with per_1x128 scales? (Run a unit test with known-good reference output, e.g., torch reference.)
2. **If yes**: write a tuned `tuned_fmoe.csv` row (or new `dsv4_fp8_tuned_fmoe.csv` config) with `flydsl_moe1_afp8_wfp8_bf16_t64x128x256_gui` + matching stage2 name, reroute aiter to use FlyDSL on DSV4.
3. **If no**: either adapt FlyDSL FP8/FP8 kernel to support per_1x128 (kernel-level work), OR fix the CK MoE path (which is what's currently broken).

User memory `project_aiterforge_fp8.md` recommends `tile_m=64, tile_n=128, tile_k=256` for tokens > 128 (validated). For tokens ≤ 128 (DSV4 prefill of 12 tokens), `tile_m=32, tile_n=128, tile_k=256` is safer (avoiding the documented `tile_m=16` NaN bug).

## 6. Repository state summary

**Branches**:
- `main` HEAD `73118b6` — Sprint 1+2+3 + accuracy diagnosis docs all merged
- `feat/dsv4-w4-kernel-bisect` (PR #54, merged) — Sprint 1+2 work
- `feat/dsv4-w4-kv-concat` (PR #55, merged) — Sprint 3 work
- `docs/dsv4-w45-ck-moe-diagnosis` (PR #56, merged) — accuracy diagnosis

**Key files modified**:
- `atom/engine/kv_pool/dsv4_pool.py` — per-ratio compressor + main_kv slabs, view_for_layer dispatch
- `atom/model_engine/model_runner.py` — pool config wiring
- `atom/models/deepseek_v4.py` — `_apply_rotary_emb` 2D branch, scatter asserts, `_bind_state_from_pool` shape-check, KV concat in `_forward_w4`

**Tests**:
- 120 DSV4 UTs all passing
- `tests/silicon/repro_w4_kernel_bisect.py` (kept for record)
- `tests/silicon/silicon_w43_smoke.py` (W4 smoke test, used for v10 silicon green)

## 7. What the next session should focus on

| Priority | Task | Owner |
|---|---|---|
| P0 | Validate FlyDSL FP8/FP8 with per_1x128 quant against torch reference (1 op test) | aiter team or follow-up |
| P0 | If valid → write DSV4 tuned_fmoe.csv rows, switch aiter to FlyDSL MoE for DSV4 | aiter team |
| P1 | Re-run silicon W4 single + multi → expect coherent output | this team |
| P1 | gsm8k limit=20 conc=4 ≥ 60% accuracy gate | this team |
| P2 | Performance tune (drop HIP_LAUNCH_BLOCKING, async kernels) | this team |
| P3 | Per-slab UT coverage for the new pool architecture | this team |

The W4.5 multi-request architectural work (#37) is **closed at the architecture level**. Accuracy is a separate workstream that lives in aiter MoE configuration, not in ATOM.
