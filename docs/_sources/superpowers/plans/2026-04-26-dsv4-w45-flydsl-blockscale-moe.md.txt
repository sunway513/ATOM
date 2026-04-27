# DSV4 W4.5 Accuracy Fix — FlyDSL Blockscale MoE Enablement

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) for tracking.

## Revision log

- **v1 (initial)** — assumed DSV4 dispatches FP8/FP8 per_1x128 (from `config.json` `weight_block_size:[128,128]`); plan to port FlyDSL `moe_blockscale_2stage` kernel into aiter.
- **v2 (review fixes)** — addressed 4 review findings: (P1) Apache-2.0 SPDX header on ported file, (P1) substring `_blockscale_` (not `startswith`) in dispatcher, (P1) per-row CSV resolution test, (P2) FlyDSL version preflight.
- **v3 (FP4 pivot, 2026-04-26 W4.5)** — silicon trace with `AITER_FMOE_DEBUG_LOOKUP=1` revealed actual lookup key is **FP4/FP4 per_1x32**, NOT FP8/per_1x128. ATOM's quant_v4 layer rewrites the dispatch dtype before reaching aiter. **Fix path collapses to a 16-row CSV** (`dsv4_fp4_tuned_fmoe.csv`) reusing existing FlyDSL FP4 kernels (`flydsl_moe1_afp4_wfp4_bf16_*`) — the blockscale port (Tasks 1-6.5) remains in-tree as future-proofing if DSV4 ever switches to per_1x128. Tasks 7-10 retargeted to FP4 path.

**Goal:** Restore W4.5 silicon output coherence (currently gibberish in both W4 and baseline `flag=0`) by routing DSV4's MoE through FlyDSL kernels (FP4/per_1x32 in v3, FP8/per_1x128 blockscale in v1-v2) instead of the unmatched CK MoE fallback. Closes the accuracy half of issue sunway513/atom#37.

**Architecture:** Single workstream spanning three repos owned by the same team:
1. **FlyDSL upstream** (`ROCm/FlyDSL`) — already has `kernels/moe_blockscale_2stage.py` (ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128, FP8-only, g1u1) + tested via `tests/kernels/test_moe_blockscale.py`. Source of truth.
2. **AITER** (`sunway513/aiter`) — port the kernel, register kernel names, dispatcher, tune config. Where the integration lives.
3. **ATOM** (`sunway513/atom`) — re-run silicon W4 single+multi + accuracy gate. Validation.

**Tech Stack:** FlyDSL (Python DSL → MLIR → ROCm), aiter (Python wrapper + JIT cache), ATOM (engine), MI355X (gfx950 / FP8 e4m3fn).

**Upstream pins (verified before plan execution — see Task 0):**
- FlyDSL `ROCm/FlyDSL` commit `8bee73f` (verified to contain `kernels/moe_blockscale_2stage.py`, ~136KB, latest as of 2026-04-26).
- AITER `sunway513/aiter` `_MIN_FLYDSL_VERSION = Version("0.1.3")` (`aiter/ops/flydsl/__init__.py`); container has `flydsl 0.1.4.2` installed which satisfies the pin. If reviewing against a different aiter checkout that pins `0.1.2`, bump the pin first.
- License: FlyDSL upstream is **Apache-2.0** with `Copyright (c) 2025 FlyDSL Project Contributors`. AITER and ATOM are MIT. **All ported source MUST preserve the Apache-2.0 SPDX header + Apache copyright notice. The new aiter file is a derivative work licensed Apache-2.0, NOT MIT** (mixing licenses in one file requires explicit dual-license header — we don't do that).

---

## Context (must-read background)

- `docs/evidence/dsv4_w45/CONTEXT_HANDOFF_W45.md` — full session context
- `docs/evidence/dsv4_w45/EVIDENCE_L.md` — Sprint 3 silicon-green report + accuracy CK MoE diagnosis addendum
- User memory `project_aiterforge_fp8.md` — validated tile configs (tile_m=64, tile_n=128, tile_k=256 for >128 tokens; tile_m=16 has NaN bug, use tile_m=32 for ≤128 tokens)
- `feedback_aiter_atom_one_team.md` — ATOM/AITER/FlyDSL is one workstream

## Why the bug exists (one-paragraph recap)

DSV4's `config.json` declares `weight_block_size: [128, 128]` → `QuantType.per_1x128`. ATOM passes this to `aiter.fused_moe`. aiter's tuned-config lookup misses (no DSV4 row exists). The fallback path picks CK MoE (`ck_moe_stage1/2`) which was compiled assuming a different scale-stride layout. Numerical garbage → gibberish output. FlyDSL's upstream `moe_blockscale_2stage.py` is built **explicitly** for ScaleBlockN/K=128 FP8/FP8 — exactly the right kernel for DSV4. We just haven't ported it.

## What's already there (so we don't double-build)

| Artifact | Location | Status |
|---|---|---|
| `compile_moe_blockscale_gemm1` (kernel impl) | `FlyDSL/kernels/moe_blockscale_2stage.py:73` | ✅ exists, tested |
| `compile_moe_blockscale_gemm2` / `_ex` (kernel impl) | `FlyDSL/kernels/moe_blockscale_2stage.py` | ✅ exists, tested |
| Numerical reference test | `FlyDSL/tests/kernels/test_moe_blockscale.py` | ✅ exists, runs against aiter fused blockscale ref |
| `mfma_preshuffle_pipeline.py` (shared) | `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py` | ✅ already in aiter |
| `mfma_epilogues.py` (shared) | `aiter/ops/flydsl/kernels/mfma_epilogues.py` | ✅ already in aiter |
| FlyDSL FP4 MoE `flydsl_moe_stage1/2` | `aiter/ops/flydsl/moe_kernels.py:547,825` | ✅ used as the **template** for our port |
| AITER FlyDSL dispatcher | `aiter/fused_moe.py:972-1022` | ✅ existing routing logic; we extend it |
| AITER tuned config CSV | `aiter/configs/tuned_fmoe.csv` + `model_configs/*.csv` | ✅ existing format; we add DSV4 rows |
| FlyDSL FP8/FP8/bf16 kernel registry | 112 stage1 + 48 stage2 already registered (per_1x32) | ⚠️ exists for FP8/FP8 dtypes but NOT for blockscale variant |

## File Structure

### AITER repo (`sunway513/aiter`)

**New files:**
- `aiter/ops/flydsl/moe_blockscale_kernels.py` — Python wrapper around FlyDSL's `compile_moe_blockscale_gemm1/2`, mirroring `aiter/ops/flydsl/moe_kernels.py`'s API surface.
- `aiter/configs/model_configs/dsv4_fp8_blockscale_tuned_fmoe.csv` — DSV4-specific tuned MoE config rows pointing to `flydsl_moe1/2_blockscale_*` kernel names.
- `op_tests/test_flydsl_moe_blockscale.py` — port of `FlyDSL/tests/kernels/test_moe_blockscale.py`, asserts numerical correctness vs torch reference.

**Modified files:**
- `aiter/ops/flydsl/__init__.py` — export `flydsl_moe_blockscale_stage1/2`.
- `aiter/ops/flydsl/moe_kernels.py` — add `get_flydsl_blockscale_stage1/2_kernels` registries (kernel names: `flydsl_moe1_afp8_wfp8_bf16_blockscale_t{M}x{N}x{K}[_{mode}]`).
- `aiter/fused_moe.py` — extend `is_flydsl1`/`is_flydsl2` dispatcher: when **`"_blockscale_" in kernelName1`** (substring match, NOT prefix — kernel names are `flydsl_moe1_afp8_wfp8_bf16_blockscale_t...`) route to a new `_flydsl_blockscale_stage1_wrapper`. Same for stage2.
- `aiter/jit/optCompilerConfig.json` — register the new module if compilation needs codegen flags.

### FlyDSL upstream (`ROCm/FlyDSL`)

**No code changes** — we consume `kernels/moe_blockscale_2stage.py` as-is. If during port we hit a bug or missing tile combo, fix upstream and bump our pin (treated as one workstream, but keep the patch surfaces minimal).

### ATOM repo (`sunway513/atom`)

**No code changes** — once aiter's `fused_moe` routes DSV4 to FlyDSL blockscale, ATOM picks it up automatically through the existing `aiter.fused_moe.fused_moe` import in `atom/model_ops/fused_moe/modular_kernel.py:347`.

**Validation only:**
- Re-run `tests/silicon/silicon_w43_smoke.py --mode single` (flag=0 baseline + flag=1 W4) → expect coherent Chinese
- Re-run `--mode multi` → 4-conc with coherent outputs
- gsm8k limit=20 conc=4 ≥ 60% W4.5 owner gate

---

## Bite-Sized Tasks

### Task 0: Preflight — pin & verify FlyDSL upstream + version compatibility

**Files:** read-only inspection, no commits in this task.

- [ ] **Step 0.1: Verify FlyDSL upstream commit + file presence**

```bash
git -C /home/pensun/FlyDSL log --oneline -1
ls -la /home/pensun/FlyDSL/kernels/moe_blockscale_2stage.py
ls -la /home/pensun/FlyDSL/kernels/mfma_preshuffle_pipeline.py
ls -la /home/pensun/FlyDSL/kernels/mfma_epilogues.py
git -C /home/pensun/FlyDSL log --oneline -- kernels/moe_blockscale_2stage.py | head -3
```

Expected: HEAD `8bee73f` (or newer). All three files present. If `moe_blockscale_2stage.py` is missing, **STOP** — pull upstream first (`cd /home/pensun/FlyDSL && git fetch origin && git checkout origin/main`).

- [ ] **Step 0.2: Verify installed `flydsl` Python package satisfies aiter's `_MIN_FLYDSL_VERSION`**

```bash
docker exec atom_dsv4_feat /opt/venv/bin/python -c "
import flydsl
print(f'installed: {flydsl.__version__}')
from aiter.ops.flydsl.utils import is_flydsl_available
print(f'aiter_satisfied: {is_flydsl_available()}')
"
```

Expected: installed >= `0.1.3` AND `aiter_satisfied=True`. If the local aiter checkout pins `0.1.2` or older, bump the pin in `aiter/ops/flydsl/__init__.py` first as a separate PR.

- [ ] **Step 0.3: Verify FlyDSL APIs used by `moe_blockscale_2stage.py` are present in installed `flydsl`**

```bash
docker exec atom_dsv4_feat /opt/venv/bin/python -c "
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, buffer_ops, vector, rocdl, range_constexpr
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects import math as math_dialect
from flydsl.expr.typing import T
from flydsl.expr.arith import ArithValue
print('all FlyDSL imports used by moe_blockscale_2stage.py: OK')
"
```

Expected: `OK`, no `ImportError`. If any import fails, escalate (FlyDSL upstream + version drift; bump or backport).

- [ ] **Step 0.4: Diff aiter's `mfma_preshuffle_pipeline.py` against FlyDSL upstream**

```bash
diff -u /home/pensun/FlyDSL/kernels/mfma_preshuffle_pipeline.py \
        /home/pensun/aiter-lingpeng/aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py | head -80
diff -u /home/pensun/FlyDSL/kernels/mfma_epilogues.py \
        /home/pensun/aiter-lingpeng/aiter/ops/flydsl/kernels/mfma_epilogues.py | head -80
```

Decide: if diffs are nontrivial AND `moe_blockscale_2stage.py` uses APIs only in upstream version → port the upstream version into aiter as part of Task 2 (extends scope but keeps the kernel set consistent). If diffs are noise (whitespace / docstrings), proceed unmodified.

- [ ] **Step 0.5: Document the verified upstream pin** in a comment at the top of the plan + add to commit messages of subsequent tasks. No PR yet.

---

### Task 1: Port `moe_blockscale_2stage.py` skeleton into aiter

**Files:**
- Create: `aiter/ops/flydsl/moe_blockscale_kernels.py`
- Reference: `/home/pensun/FlyDSL/kernels/moe_blockscale_2stage.py:1-2828`
- Reference: `aiter/ops/flydsl/moe_kernels.py` (existing FP4 port — same structure to mirror)

- [ ] **Step 1.1: Create `moe_blockscale_kernels.py` with the same module shape as `moe_kernels.py`**

**License preservation (P1)**: this file is a derivative of `ROCm/FlyDSL/kernels/moe_blockscale_2stage.py` (Apache-2.0). The new file MUST carry an Apache-2.0 SPDX header and preserve the upstream copyright. Do NOT use AITER's default MIT header — mixing licenses in one file requires an explicit dual-license declaration that we are not doing.

```python
# aiter/ops/flydsl/moe_blockscale_kernels.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Derived from ROCm/FlyDSL kernels/moe_blockscale_2stage.py (Apache-2.0).
# Upstream ref: ROCm/FlyDSL commit 8bee73f (or newer) — see Task 0.1.
"""FlyDSL Blockscale MOE kernel management (per_1x128 FP8/FP8, g1u1).

Mirrors aiter/ops/flydsl/moe_kernels.py for the BLOCKSCALE variant.
Kernel name pattern: ``flydsl_moe{stage}_afp8_wfp8_{out}_blockscale_t{M}x{N}x{K}[_{mode}]``.

Source of truth: ROCm/FlyDSL `kernels/moe_blockscale_2stage.py`.
"""
import functools
import re
from typing import Dict, Optional
import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}
_SUFFIX_RE = re.compile(r"(?:_w(?P<wpe>\d+))?(?:_bnt(?P<bnt>\d+))?(?:_xcd(?P<xcd>\d+))?$")


def flydsl_blockscale_kernel_name(stage, tile_m, tile_n, tile_k,
                                   waves_per_eu=None, b_nt=None, xcd=None):
    name = f"flydsl_moe{stage}_afp8_wfp8_bf16_blockscale_t{tile_m}x{tile_n}x{tile_k}"
    if waves_per_eu and waves_per_eu != 1:
        name += f"_w{waves_per_eu}"
    if b_nt is not None and b_nt != 0:
        name += f"_bnt{b_nt}"
    if xcd is not None and xcd > 0:
        name += f"_xcd{xcd}"
    return name


def get_flydsl_blockscale_stage1_kernels() -> Dict[str, Dict]:
    """Generate FlyDSL blockscale stage1 kernel registry."""
    kernels = {}
    for tm in [32, 64, 128]:
        for tn in [128, 256]:
            for tk in [128, 256]:  # blockscale uses ScaleBlockK=128, kernel tile_k can be 128 or 256
                for wpe in [1, 2, 3, 4]:
                    for bnt in [0, 2]:
                        name = flydsl_blockscale_kernel_name(1, tm, tn, tk, wpe, bnt)
                        kernels[name] = {
                            "stage": 1, "tile_m": tm, "tile_n": tn, "tile_k": tk,
                            "waves_per_eu": wpe, "b_nt": bnt, "scale_block_k": 128,
                            "a_dtype": "fp8", "b_dtype": "fp8", "out_dtype": "bf16",
                        }
    _KERNEL_PARAMS.update(kernels)
    return kernels


def get_flydsl_blockscale_stage2_kernels() -> Dict[str, Dict]:
    """Same pattern for stage2."""
    kernels = {}
    # ... mirror stage1 with stage=2, optionally include `_atomic`/`_persist` modes
    return kernels


def get_flydsl_blockscale_kernel_params(name: str) -> Optional[Dict]:
    """Lookup with suffix parsing."""
    return _KERNEL_PARAMS.get(name)
```

- [ ] **Step 1.2: Run `python -c "from aiter.ops.flydsl.moe_blockscale_kernels import get_flydsl_blockscale_stage1_kernels; print(len(get_flydsl_blockscale_stage1_kernels()))"`. Expect non-zero count.**

- [ ] **Step 1.3: Commit**

```bash
git add aiter/ops/flydsl/moe_blockscale_kernels.py
git commit -m "feat(flydsl): scaffold blockscale MoE kernel registry (#37 W4.5 accuracy)"
```

---

### Task 2: Port `compile_moe_blockscale_gemm1` and `compile_moe_blockscale_gemm2`

**Files:**
- Modify: `aiter/ops/flydsl/moe_blockscale_kernels.py`
- Reference: `/home/pensun/FlyDSL/kernels/moe_blockscale_2stage.py:73-1500` (gemm1) and `:1500-2700` (gemm2/gemm2_ex)

- [ ] **Step 2.1: Copy `compile_moe_blockscale_gemm1` body verbatim into `moe_blockscale_kernels.py`. Adjust imports** (`from kernels.mfma_preshuffle_pipeline import ...` → `from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import ...`; same for `mfma_epilogues`). **Per Task 0.4 finding**: if upstream `mfma_preshuffle_pipeline.py` differs from aiter's, port the upstream version too in this commit. Keep the Apache-2.0 SPDX block from Step 1.1 at the top.

- [ ] **Step 2.2: Copy `compile_moe_blockscale_gemm2` and `compile_moe_blockscale_gemm2_ex`. Same import adjustments.**

- [ ] **Step 2.3: Add Python-level wrappers `flydsl_moe_blockscale_stage1` and `_stage2`** mirroring the API of `flydsl_moe_stage1/2` in `aiter/ops/flydsl/moe_kernels.py:547`. Wrappers handle: scale flatten, output buffer allocation, `compile_*` invocation, `_run_compiled` dispatch.

- [ ] **Step 2.4: Smoke test the import chain**

```bash
docker exec atom_dsv4_feat /opt/venv/bin/python -c "
from aiter.ops.flydsl.moe_blockscale_kernels import (
    flydsl_moe_blockscale_stage1, flydsl_moe_blockscale_stage2,
    compile_moe_blockscale_gemm1, compile_moe_blockscale_gemm2)
print('imports OK')
"
```

Expected: `imports OK` with no exceptions.

- [ ] **Step 2.5: Commit**

```bash
git commit -m "feat(flydsl): port compile_moe_blockscale_gemm1/2 from FlyDSL upstream (#37 W4.5)"
```

---

### Task 3: Wire AITER FlyDSL `__init__.py` exports

**Files:**
- Modify: `aiter/ops/flydsl/__init__.py`

- [ ] **Step 3.1: Add the new wrappers to imports + `__all__`**

```python
# in the `if is_flydsl_available():` block, after existing imports:
from .moe_blockscale_kernels import (
    flydsl_moe_blockscale_stage1,
    flydsl_moe_blockscale_stage2,
)
__all__ += [
    "flydsl_moe_blockscale_stage1",
    "flydsl_moe_blockscale_stage2",
]
```

- [ ] **Step 3.2: Verify**

```bash
docker exec atom_dsv4_feat /opt/venv/bin/python -c "
import aiter.ops.flydsl as f
print('flydsl_moe_blockscale_stage1' in f.__all__)
print('flydsl_moe_blockscale_stage2' in f.__all__)
"
```

Expected: `True / True`.

- [ ] **Step 3.3: Commit.**

---

### Task 4: Extend `aiter/fused_moe.py` dispatcher to route blockscale kernels

**Files:**
- Modify: `aiter/fused_moe.py:972-1022` (`is_flydsl1` block)

- [ ] **Step 4.1: Add substring-match predicate (NOT prefix) to keep names stable.** Kernel names are `flydsl_moe1_afp8_wfp8_bf16_blockscale_t...` — they start with the same `flydsl_moe1_` prefix as the FP4 path; the distinguishing token is the literal substring `_blockscale_`. Use `"_blockscale_" in kernelName1` to pick the blockscale wrapper. Inside the existing `if (is_flydsl1 or is_flydsl2) and is_flydsl_available():` block, add a sub-branch:

```python
if is_flydsl1:
    if "_blockscale_" in kernelName1:   # P1 fix: substring, NOT startswith("flydsl_moe1_blockscale_")
        stage1_func = functools.partial(
            _flydsl_blockscale_stage1_wrapper,
            kernelName=kernelName1,
            activation=activation,
        )
    else:
        stage1_func = functools.partial(
            _flydsl_stage1_wrapper,
            kernelName=kernelName1,
            activation=activation,
        )

if is_flydsl2:
    if "_blockscale_" in kernelName2:   # same substring rule for stage2
        stage2_func = functools.partial(
            _flydsl_blockscale_stage2_wrapper,
            kernelName=kernelName2,
        )
    else:
        stage2_func = functools.partial(
            _flydsl_stage2_wrapper,
            kernelName=kernelName2,
        )
```

- [ ] **Step 4.2: Add `_flydsl_blockscale_stage1_wrapper` and `_stage2_wrapper`** beside the existing FP4 wrappers (line ~654, ~688). Use `aiter.ops.flydsl.flydsl_moe_blockscale_stage1` instead of FP4 entry point.

- [ ] **Step 4.3: Verify dispatcher coverage**

```bash
docker exec atom_dsv4_feat /opt/venv/bin/python -c "
from aiter.fused_moe import _flydsl_blockscale_stage1_wrapper, _flydsl_blockscale_stage2_wrapper
print('dispatchers OK')
"
```

- [ ] **Step 4.4: Commit.**

---

### Task 5: Numerical correctness test (port FlyDSL's blockscale test)

**Files:**
- Create: `op_tests/test_flydsl_moe_blockscale.py`
- Reference: `/home/pensun/FlyDSL/tests/kernels/test_moe_blockscale.py`

- [ ] **Step 5.1: Port the test, adjust imports** to use `aiter.ops.flydsl.moe_blockscale_kernels.flydsl_moe_blockscale_stage1/2`. Keep the torch reference (`torch_stage1_blockscale_ref`) unchanged.

- [ ] **Step 5.2: Add a DSV4-shape test case** alongside the existing parameter sweep:

```python
@pytest.mark.parametrize(
    "B,model_dim,inter_dim,E,topk",
    [(12, 7168, 3072, 384, 6)],  # DSV4 prefill 12-token, single seq
)
def test_moe_blockscale_dsv4_shape(B, model_dim, inter_dim, E, topk):
    # ... build inputs, run flydsl_moe_blockscale_stage1, compare against torch ref
```

- [ ] **Step 5.3: Run on silicon**

```bash
docker exec atom_dsv4_feat /opt/venv/bin/python -m pytest op_tests/test_flydsl_moe_blockscale.py -v 2>&1 | tail -20
```

Expected: `PASSED` for at least the DSV4-shape case (allclose against torch reference, atol=1e-2 rtol=1e-2 typical for FP8 tolerance).

- [ ] **Step 5.4: Commit.**

---

### Task 6: Add DSV4 tuned config rows

**Files:**
- Create: `aiter/configs/model_configs/dsv4_fp8_blockscale_tuned_fmoe.csv`
- Reference column order: `aiter/configs/tuned_fmoe.csv:1`

- [ ] **Step 6.1: Build the CSV with header + DSV4 rows for token counts {1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024}.** Each row uses the same key format aiter expects:

```csv
cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw,_tag
256,1,7168,3072,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,32,0,0.0,flydsl_moe1_afp8_wfp8_bf16_blockscale_t32x128x128_w2,0.0%,0.0,flydsl_moe2_afp8_wfp8_bf16_blockscale_t32x128x128_atomic,0.0%,0.0,0,0.0,0.0,
256,12,7168,3072,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,32,0,0.0,flydsl_moe1_afp8_wfp8_bf16_blockscale_t32x128x128_w2,0.0%,0.0,flydsl_moe2_afp8_wfp8_bf16_blockscale_t32x128x128_atomic,0.0%,0.0,0,0.0,0.0,
256,128,7168,3072,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,32,0,0.0,flydsl_moe1_afp8_wfp8_bf16_blockscale_t32x128x256,0.0%,0.0,flydsl_moe2_afp8_wfp8_bf16_blockscale_t32x256x256_atomic,0.0%,0.0,0,0.0,0.0,
256,512,7168,3072,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,64,0,0.0,flydsl_moe1_afp8_wfp8_bf16_blockscale_t64x128x256,0.0%,0.0,flydsl_moe2_afp8_wfp8_bf16_blockscale_t64x256x256_atomic,0.0%,0.0,0,0.0,0.0,
256,1024,7168,3072,384,6,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e4m3fn,QuantType.per_1x128,1,0,64,0,0.0,flydsl_moe1_afp8_wfp8_bf16_blockscale_t64x128x256,0.0%,0.0,flydsl_moe2_afp8_wfp8_bf16_blockscale_t64x256x256_atomic,0.0%,0.0,0,0.0,0.0,
```

(Tile choices follow user memory `project_aiterforge_fp8.md`: `tile_m=32` for ≤128 tokens, `tile_m=64` for >128. Avoiding `tile_m=16` per its NaN bug. Times left as 0 — they're ignored at lookup time, only kernel name matters.)

- [ ] **Step 6.2: Verify aiter loads the new config**

```bash
docker exec atom_dsv4_feat env AITER_CONFIG_FMOE=/workspace/aiter-lingpeng/aiter/configs/model_configs/dsv4_fp8_blockscale_tuned_fmoe.csv \
    /opt/venv/bin/python -c "
import aiter.fused_moe
# trigger config load by doing a no-op moe call OR direct load
import pandas as pd
df = pd.read_csv('/workspace/aiter-lingpeng/aiter/configs/model_configs/dsv4_fp8_blockscale_tuned_fmoe.csv')
print(f'rows={len(df)} kernels={df[\"kernelName1\"].unique()[:3]}')
"
```

- [ ] **Step 6.3: Commit.**

---

### Task 6.5: Full-CSV coverage test — every kernelName1/kernelName2 in the DSV4 config must compute correctly (P1 fix)

**Why this exists**: Task 5's numerical test only proved one shape (12 tokens). Task 6 installs CSV rows that span 6+ token counts and 2+ stage2 variants (`_atomic`, plus possibly `_persist`/`_reduce`). A 12-token-only test will silently miss: (a) an unregistered kernel name in another row, (b) a stage2 mode that's wrong for decode/prefill shapes, (c) a tile-specific bug in a single combo. We have to prove **every kernel name actually used by the CSV** computes correctly before silicon.

**Files:**
- Modify/extend: `op_tests/test_flydsl_moe_blockscale.py`

- [ ] **Step 6.5.1: Iterate the CSV and run aiter's lookup for each row** to confirm every `kernelName1`/`kernelName2` value resolves through `_flydsl_blockscale_stage1_wrapper` / `_stage2_wrapper` (NOT falling back to the FP4 wrapper or to CK).

```python
def test_dsv4_csv_kernel_names_resolve():
    import pandas as pd
    df = pd.read_csv("aiter/configs/model_configs/dsv4_fp8_blockscale_tuned_fmoe.csv")
    from aiter.ops.flydsl.moe_blockscale_kernels import get_flydsl_blockscale_kernel_params
    for _, row in df.iterrows():
        for col in ("kernelName1", "kernelName2"):
            name = str(row[col])
            assert "_blockscale_" in name, f"row {row['token']}: {col}={name} missing _blockscale_"
            params = get_flydsl_blockscale_kernel_params(name)
            assert params is not None, f"unregistered kernel: {name}"
```

- [ ] **Step 6.5.2: Numerically validate each distinct (kernelName1, kernelName2) pair against torch reference** for representative DSV4-shape inputs. Cover at least the unique kernels the CSV uses (typically 3-5 stage1 + 3-5 stage2 distinct entries):

```python
@pytest.mark.parametrize("token_count", [1, 12, 128, 512, 1024])
def test_dsv4_csv_pair_correctness(token_count):
    import pandas as pd
    df = pd.read_csv("aiter/configs/model_configs/dsv4_fp8_blockscale_tuned_fmoe.csv")
    row = df[df["token"] == token_count].iloc[0]
    kn1, kn2 = str(row["kernelName1"]), str(row["kernelName2"])
    # Build DSV4-shape inputs (token_count, 7168, 3072, 384, topk=6)
    # Run flydsl_moe_blockscale_stage1 with kn1's tile params
    # Run flydsl_moe_blockscale_stage2 with kn2's tile params
    # Compare with torch_stage1_blockscale_ref / torch_stage2_blockscale_ref
    # Assert allclose (atol=1e-2 rtol=1e-2 — FP8 tolerance)
```

- [ ] **Step 6.5.3: Run all CSV-coverage tests on silicon**

```bash
docker exec atom_dsv4_feat /opt/venv/bin/python -m pytest \
    op_tests/test_flydsl_moe_blockscale.py::test_dsv4_csv_kernel_names_resolve \
    op_tests/test_flydsl_moe_blockscale.py::test_dsv4_csv_pair_correctness \
    -v 2>&1 | tail -30
```

Expected: **every parametrized case PASSED** (no skipped, no fall-back, no allclose mismatch). Any failure here MUST be fixed before Task 7 silicon validation — fix the CSV row's tile choice OR the upstream kernel.

- [ ] **Step 6.5.4: Commit + ensure this test is in the AITER PR (not skipped in CI).**

---

### Task 7: ATOM-side silicon validation — single mode

**Files:** No ATOM code changes. Use existing `tests/silicon/silicon_w43_smoke.py`.

- [ ] **Step 7.1: Pre-flight**

```bash
rocm-smi --showpids 2>&1 | grep -E "KFD|No KFD"
# kill orphans if any (per feedback_gpu_preflight.md)
```

- [ ] **Step 7.2: Run baseline (flag=0) with FlyDSL config**

```bash
docker exec -d atom_dsv4_feat bash -c '
cd /workspace/ATOM-lingpeng &&
HIP_LAUNCH_BLOCKING=1 \
ATOM_DSV4_USE_W4_PATH=0 \
AITER_CONFIG_FMOE=/workspace/aiter-lingpeng/aiter/configs/model_configs/dsv4_fp8_blockscale_tuned_fmoe.csv \
AITER_LOG_LEVEL=WARNING \
/opt/venv/bin/python -m tests.silicon.silicon_w43_smoke \
    --mode single \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-model-len 2048 --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --max-tokens 32 \
    --out /workspace/ATOM-lingpeng/logs/silicon_baseline_flydsl.json \
    > /workspace/ATOM-lingpeng/logs/silicon_baseline_flydsl.log 2>&1
'
```

- [ ] **Step 7.3: Verify the output is coherent Chinese (matches Evidence J era expected output)**

Expected: completion starts with `好的，这是一个非常具体...` (or similar coherent Chinese), NOT gibberish.

- [ ] **Step 7.4: Run W4 mode (flag=1) with FlyDSL config**

```bash
# same command, change ATOM_DSV4_USE_W4_PATH=1 ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 ATOM_AITER_VALIDATE=1
```

Expected: same coherent output, no HSA crashes (Sprint 1+2+3 architecture intact).

---

### Task 8: ATOM-side silicon validation — multi mode (4 conc)

- [ ] **Step 8.1: Run silicon_w43_smoke `--mode multi` with FlyDSL config + flag=1**

Expected: 4 distinct coherent completions (each matches what their respective prompts should produce).

---

### Task 9: gsm8k W4.5 owner accuracy gate

- [ ] **Step 9.1: Run gsm8k limit=20 conc=4 with W4 path enabled + FlyDSL MoE config**

```bash
# Use existing benchmark harness (per CLAUDE.md /benchmark-guide)
```

Expected: ≥ 60% accuracy (W4.5 owner threshold).

- [ ] **Step 9.2: Capture results into Evidence M** (new file).

---

### Task 10: Documentation closeout

**Files:**
- Create: `docs/evidence/dsv4_w45/EVIDENCE_M.md` — Sprint 4 accuracy fix report
- Modify: existing `EVIDENCE_L.md` — add a closing pointer to M

- [ ] **Step 10.1: Write Evidence M** with:
  - Full silicon iteration table (baseline + W4 single + W4 multi all coherent)
  - gsm8k score
  - Kernel routing trace showing FlyDSL blockscale calls (no more `ck_moe_stage1/2` calls)
  - Final architecture summary diagram

- [ ] **Step 10.2: Open ATOM PR + AITER PR + (if needed) FlyDSL PR.** All three repos same commit chain.

- [ ] **Step 10.3: Issue #37 closeout comment** referencing all 4 Evidence docs (J → M).

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| FlyDSL blockscale kernel has FP8 e4m3fn vs e4m3fnuz dtype mismatch on MI355 | Medium | gfx950 uses `torch.float8_e4m3fn` per the FlyDSL test (line 50). Confirm in Step 5 numerical test. |
| Tile params from user memory (m=64, n=128, k=256) need re-tuning for blockscale variant | Medium | Step 5 numerical test passes regardless of tile (correctness ≠ perf). If perf matters for gsm8k timing, sweep tiles in a follow-up |
| Some FlyDSL kernel imports break because the upstream version on aiter container is older | Low-medium | aiter pins `_MIN_FLYDSL_VERSION = 0.1.3`. Verify locally in Step 2.4. If mismatch, bump aiter pin or pull FlyDSL upstream. |
| stage2 has multiple variants (`_atomic`, `_persist`, `_reduce`); picking the wrong one for DSV4 dims | Medium | Numerical test in Step 5 covers all variants; pick the passing one for the CSV. |
| The mfma_preshuffle_pipeline.py in aiter is OLDER than what moe_blockscale_2stage.py expects | Medium | Diff aiter's vs FlyDSL upstream's. If aiter's is older, port the upstream version (small file, 562 lines). |

---

## Review Checklist (for the human reviewer)

- [ ] Plan correctly identifies the bug location (aiter MoE backend, NOT ATOM)
- [ ] Cross-repo split is sane: kernel impl in FlyDSL, integration in aiter, validation in ATOM
- [ ] Tile choices match user memory's validated values
- [ ] Numerical test covers DSV4-shape (model_dim=7168, inter_dim=3072, E=384, topk=6)
- [ ] No new shape-handling code in ATOM — purely consumed via aiter
- [ ] All risks have mitigations OR explicit "verify in Step X" anchors
- [ ] Estimated effort feels right (~1-2 days for one engineer; mostly Tasks 2 + 5)
- [ ] Sprint 1+2+3 W4.5 architecture is preserved (we're only fixing the MoE backend wiring)

---

## Approval gate

After review:

```
[ ] Approved as-is — proceed with subagent-driven-development
[ ] Approved with revisions:
    -
    -
[ ] Reject — re-plan
```

When approved, invoke `superpowers:subagent-driven-development` to execute Tasks 0-10 with two-stage review per task.

---

## Revision log

### v2 — 2026-04-26 (review feedback addressed)

- **P1 fix (license)**: Plan now mandates Apache-2.0 SPDX header + upstream copyright preservation in `aiter/ops/flydsl/moe_blockscale_kernels.py`. The new file is a derivative work licensed Apache-2.0, NOT MIT (mixing requires explicit dual-license declaration we do not do). Updated File Structure section + Step 1.1 scaffold + Step 2.1 reminder.
- **P1 fix (dispatcher prefix)**: Predicate corrected from `kernelName1.startswith("flydsl_moe1_blockscale_")` (which would never match) to **`"_blockscale_" in kernelName1`** (substring match). Kernel names are `flydsl_moe1_afp8_wfp8_bf16_blockscale_t...` so substring is the correct distinguishing rule. Fixed both the File Structure summary AND Task 4's code block. Same rule for stage2.
- **P1 fix (CSV-coverage test)**: Added Task 6.5 — iterates every CSV row, asserts each `kernelName1/2` resolves through the blockscale wrapper (not falling through to FP4 wrapper or CK), and runs numerical correctness vs torch reference for each distinct (kn1, kn2) pair across token counts 1/12/128/512/1024. Closes the gap where Task 5's 12-token-only test would silently miss decode-shape failures.
- **P2 fix (FlyDSL pin)**: Added Task 0 (preflight) — explicitly verifies (a) FlyDSL upstream commit `8bee73f` (or newer) and the presence of `kernels/moe_blockscale_2stage.py`; (b) installed `flydsl` Python package satisfies aiter's `_MIN_FLYDSL_VERSION = 0.1.3` (current sunway513/aiter pin); (c) every FlyDSL API used by `moe_blockscale_2stage.py` imports successfully; (d) diff aiter's `mfma_preshuffle_pipeline.py` against upstream and decide port-or-keep. Reviewer noted some checkouts pin `0.1.2` — that requires a separate pin-bump PR before this plan starts.

### v1 — 2026-04-26 (initial plan)

10 tasks, no preflight, MIT SPDX in scaffold, prefix-match predicate, single-shape numerical test, no FlyDSL version verification.
