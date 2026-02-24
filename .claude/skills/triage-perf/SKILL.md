# Skill: Triage Performance Issues in LLM Inference

## Description
Systematic methodology for diagnosing and fixing performance regressions in LLM inference — slow token generation, high latency, or throughput far below expectations. Distilled from a debugging campaign where a clean AITER/ATOM Docker image was 570x slower than the public image, and 5 root causes were identified and fixed to reach 80% of reference performance.

## When to Use
- Inference throughput far below expectations or a known-good reference
- Decode or prefill latency regression after a build/config/environment change
- New Docker image or new hardware (e.g., gfx942 -> gfx950) shows unexplained slowness
- Suspected GEMM dispatch, kernel fallback, or architecture detection issues

---

## Core Principle

**Slow inference is almost always a dispatch problem, not a kernel problem.** The kernels themselves are fast — the question is whether the right kernel is being called. Silent fallbacks to unoptimized paths (torch.mm, Python loops, wrong tile sizes) cause 10-100x slowdowns that compound across layers.

---

## Phase 1: Baseline and Compare

### 1.1 Establish a reference
- Public/production Docker image on same hardware
- Published benchmark numbers for same model/hardware
- Previous build that was fast

### 1.2 Run identical benchmarks
```bash
curl -s http://localhost:PORT/v1/completions \
  -d '{"model":"MODEL","prompt":"The capital of France is","max_tokens":100,"temperature":0}'
```

### 1.3 Measure decode and prefill separately
| Phase | Bottleneck | Common Root Cause |
|-------|-----------|-------------------|
| Decode (M=1) | Memory-bandwidth, GEMM dispatch | Wrong default (torch.mm vs hipBLASLt vs ASM) |
| Prefill (M=batch) | Compute, GEMM throughput | Missing tuned configs, wrong tile size |
| Both | MOE dispatch | Architecture detection miss, LDS constraint |

---

## Phase 2: Check for Silent Fallbacks

### 2.1 JIT module build failures
```bash
grep -i "failed jit build" server.log
grep -i "fallback" server.log | head -20
```

### 2.2 Architecture detection
```python
# BAD (misses gfx950):
target.arch == 'gfx942'
get_gfx().startswith("gfx94")

# GOOD:
target.arch in ('gfx942', 'gfx950')
```
Search: `grep -rn "gfx94\|gfx942\|get_gfx" atom/model_ops/ aiter/ops/`

### 2.3 GEMM dispatch default
```python
# BAD: default to torch.mm (50-100x slower)
default_config["libtype"] = "torch"
# GOOD: default to hipBLASLt
default_config["libtype"] = "hipblaslt"
```
Check `aiter/tuned_gemm.py` for default fallback logic.

### 2.4 Tuned GEMM config coverage
Common gap: tuned configs cover M<=256 (decode) but not M>256 (prefill).

### 2.5 MOE kernel dispatch
- Architecture gating on MOE Triton kernels
- LDS size constraints (gfx950 != gfx942)
- Sorting/routing fallback paths

---

## Phase 3: Component-Level Profiling

```python
import torch, time

def bench(name, fn, warmup=5, runs=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs): fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / runs * 1000
    print(f"{name}: {elapsed:.2f} ms")

bench("torch.mm M=1",     lambda: torch.mm(a1, b))
bench("hipblaslt M=1",    lambda: hipblaslt_gemm(a1, b))
bench("tuned_gemm M=1",   lambda: tuned_gemm(a1, b))
bench("fused_moe",        lambda: fused_moe(hidden, gate, experts))
bench("paged_attn",       lambda: paged_attention(q, k_cache, v_cache))
```

**What to look for**:
- torch.mm vs hipBLASLt >10x gap -> GEMM default is wrong
- hipBLASLt vs tuned >2x gap -> tuned configs missing
- MOE Triton vs Python >5x gap -> dispatch failed

---

## Phase 4: Docker-Specific Debugging

### 4.1 Patch inside running containers
```bash
docker run -d --name debug ... IMAGE sleep infinity
docker exec debug sed -i 's/old/new/' /path/to/file.py
docker commit debug image:patched-v1
```

### 4.2 Incremental validation (one fix at a time)
```
Baseline:      0.17 tok/s  (570x slower)
+ Fix gfx950:  12 tok/s    (70x — dispatch was worst)
+ Fix LDS:     45 tok/s    (3.8x — MOE crashes eliminated)
+ Fix GEMM:    82 tok/s    (1.8x — hipBLASLt vs torch.mm)
+ Fix layout:  95 tok/s    (1.15x — correct scale strides)
Reference:     118 tok/s   (80% achieved)
```

### 4.3 Compare Docker images
```bash
# Diff packages
docker run --rm image1 pip list > /tmp/pkgs1.txt
docker run --rm image2 pip list > /tmp/pkgs2.txt
diff /tmp/pkgs1.txt /tmp/pkgs2.txt

# Diff AITER configs
docker cp container1:/app/aiter/configs/ /tmp/configs1/
docker cp container2:/app/aiter/configs/ /tmp/configs2/
diff -r /tmp/configs1 /tmp/configs2

# Check ASM kernels
docker run --rm image1 find /app/aiter/hsa/ -name "*.co" | sort > /tmp/asm1.txt
docker run --rm image2 find /app/aiter/hsa/ -name "*.co" | sort > /tmp/asm2.txt
diff /tmp/asm1.txt /tmp/asm2.txt
```

---

## Common Root Causes (Ranked by Impact)

1. **GEMM dispatch default (50-100x)** — untuned shapes -> torch.mm instead of hipBLASLt
2. **Architecture detection miss (10-100x)** — new GPU arch not in dispatch conditions
3. **LDS/resource constraint violation (crash -> fallback)** — tile sizes don't fit on new arch
4. **Missing ASM backend (1.5-3x on decode)** — M=1 GEMM falls to CK-Tile or hipBLASLt
5. **Tuned CSV coverage gap (1.5-2x on prefill)** — configs only cover decode shapes
6. **JIT build failure (2-5x per kernel)** — C++/ASM -> Triton/Python fallback

---

## Known ATOM Fixes for MI355X (gfx950)

These patches are commonly needed when running ATOM Docker images built for gfx942 on gfx950:

### Fix 0: gfx950 Triton MOE detection
```python
# atom/model_ops/moe.py
# Change: get_gfx().startswith("gfx94") -> get_gfx() in ("gfx942", "gfx950")
```

### Fix 2: CDNA4MXScaleLayout rename
```python
# atom/model_ops/fused_moe_triton.py
# Change: GFX950MXScaleLayout -> CDNA4MXScaleLayout
```

### Fix 3: gfx950 LDS constraint
```python
# atom/model_ops/fused_moe_triton.py
# Add: update_opt_flags_constraints({"block_m": 128})
```

### Fix 4: GEMM default fallback (AITER)
```python
# aiter/tuned_gemm.py
# Change: default_config["libtype"] = "torch" -> "hipblaslt"
```

### Fix 5: JIT non-fatal errors (AITER)
```python
# aiter/jit/core.py
# Change: raise SystemExit(...) -> raise RuntimeError(...)
```

### Fix 6: MOE sorting Triton fallback
```python
# aiter/fused_moe.py
# Wrap moe_sorting_fwd in try/except with Triton fallback
```

---

## Decision Tree

```
Inference is slower than expected
|
+- How much slower?
|  +- >10x -> dispatch/fallback bug (Phase 2)
|  +- 2-10x -> missing tuned configs or wrong default (Phase 2.3-2.4)
|  +- <2x -> missing ASM backend or partial tuned coverage
|
+- Decode or prefill slow?
|  +- DECODE -> GEMM default for M=1, ASM backend presence
|  +- PREFILL -> tuned CSV coverage for large M, MOE dispatch
|  +- BOTH -> architecture detection miss or JIT failures
|
+- JIT build failures in stderr?
|  +- YES -> identify which modules, check fallbacks
|  +- NO -> continue
|
+- GPU arch in all dispatch conditions?
|  +- NO -> add arch to dispatch checks
|  +- YES -> continue
|
+- GEMM default fallback?
|  +- "torch" -> change to "hipblaslt"
|  +- "hipblaslt" -> check tuned config coverage
|
+- All dispatch correct?
   +- Component-level profiling (Phase 3)
```
