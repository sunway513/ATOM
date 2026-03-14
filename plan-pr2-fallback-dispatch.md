# PR 2: Fallback Dispatch in JIT — Comprehensive Implementation Plan

## Context

RFC #2073 PR 2 is the **critical infrastructure** that enables graceful degradation when
CK/ASM kernels fail to compile. Without it, AITER hard-crashes on new architectures (MI400)
where `compile_ops` modules can't build. This is the single biggest blocker for MI400 bring-up.

## Problem Statement

Today, when a `compile_ops`-decorated function fails to build:
1. `build_module()` raises `RuntimeError`
2. The error propagates up to the first call site
3. **The entire process crashes** — no fallback, no degradation

There are **29 HIP JIT kernel functions** used across **14 ATOM files** with **50+ call sites**.
Only **3 functions** currently have try/except protection.

---

## Design Goals

1. **Zero-crash guarantee**: Any `compile_ops` build failure → automatic fallback to Triton/PyTorch
2. **Zero perf regression on supported archs**: Fallback logic adds no overhead when HIP JIT succeeds
3. **Transparent**: Log warnings when falling back so users know performance may differ
4. **Incremental**: Can be implemented module-by-module; each module is independently useful
5. **MI400 bring-up unblocked**: Llama-3.1-8B runs day-1 with just this + arch whitelist fix

---

## Architecture

### Two-Layer Approach

**Layer 1 — AITER side (upstream):** Modify `compile_ops` decorator to accept a `fallback` parameter.
When JIT build fails, invoke the fallback instead of crashing.

**Layer 2 — ATOM side (this repo):** Add arch-aware dispatch logic that proactively selects
Triton paths on unsupported architectures, avoiding JIT failures entirely.

Both layers are needed: Layer 1 is the safety net; Layer 2 is the performance-optimal path.

---

## Layer 1: AITER `compile_ops` Fallback Mechanism

### 1.1 Core Changes to `aiter/jit/core.py`

#### `_failed_modules` Global Cache

```python
# Global dict caching modules that failed to build.
# Key: module name, Value: exception that caused the failure.
_failed_modules: dict[str, Exception] = {}
```

Purpose: Avoid retrying builds that already failed. A single JIT failure per module is
cached for the lifetime of the process.

#### Modified `compile_ops` Signature

```python
def compile_ops(
    _md_name: str,
    fc_name: Optional[str] = None,
    gen_func: Optional[Callable[..., dict[str, Any]]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
    fallback: Optional[Callable] = None,  # NEW
):
```

#### Modified Wrapper Logic

```python
def decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        md_name = _md_name
        fn_name = fc_name or func.__name__

        # Fast path: module already known to have failed
        if md_name in _failed_modules:
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise _failed_modules[md_name]

        try:
            module = get_module(md_name)
            op = getattr(module, fn_name)
            return op(*args, **kwargs)
        except (ModuleNotFoundError, RuntimeError) as e:
            # Build failed or module unavailable
            _failed_modules[md_name] = e
            if fallback is not None:
                logger.warning(
                    f"[aiter] {md_name}.{fn_name} build failed, "
                    f"using fallback: {e}"
                )
                return fallback(*args, **kwargs)
            raise
    return wrapper
return decorator
```

Key behaviors:
- First call attempts JIT build as usual
- On failure, caches the error and invokes fallback
- Subsequent calls skip JIT attempt entirely (via `_failed_modules` cache)
- If no fallback provided, behavior is unchanged (raises RuntimeError)
- `functools.lru_cache` on `get_module()` needs to NOT cache failures (clear on error)

#### `get_module()` Cache Invalidation

Current `get_module()` uses `@functools.lru_cache()`. A failed import would cache `None`
or raise. Need to ensure failed modules don't pollute the LRU cache:

```python
@functools.lru_cache(maxsize=1024)
def get_module(md_name):
    if md_name in _failed_modules:
        raise _failed_modules[md_name]
    try:
        # ... existing import logic ...
        return __mds[md_name]
    except Exception as e:
        _failed_modules[md_name] = e
        raise
```

### 1.2 Fallback Implementations Per Module

Each `compile_ops` module needs fallback functions registered. These live in a new file:

**New file: `aiter/ops/fallbacks.py`**

#### RMSNorm Fallbacks (`module_norm`)

```python
def rmsnorm2d_fwd_fallback(out, input, weight, epsilon):
    """PyTorch fallback for RMSNorm."""
    variance = input.float().pow(2).mean(-1, keepdim=True)
    hidden = input * torch.rsqrt(variance + epsilon)
    out.copy_(weight * hidden.to(input.dtype))

def rmsnorm2d_fwd_with_add_fallback(out, input, residual_in, residual_out, weight, epsilon):
    """PyTorch fallback for fused RMSNorm + residual add."""
    residual_out.copy_(input + residual_in)
    rmsnorm2d_fwd_fallback(out, residual_out, weight, epsilon)
```

Alternative: Route to existing Triton `fused_add_rmsnorm_pad` when available.

#### Activation Fallbacks (`module_activation`)

```python
def silu_and_mul_fallback(out, input):
    """PyTorch fallback for SiLU * Mul."""
    d = input.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(input[..., :d]) * input[..., d:])

def gelu_and_mul_fallback(out, input):
    d = input.shape[-1] // 2
    out.copy_(torch.nn.functional.gelu(input[..., :d]) * input[..., d:])
```

#### Cache Fallbacks (`module_cache`)

```python
def reshape_and_cache_fallback(key, value, key_cache, value_cache, slot_mapping,
                                kv_cache_dtype, k_scale=None, v_scale=None, asm_layout=False):
    """PyTorch fallback for KV cache reshape."""
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        block_idx = slot // key_cache.shape[1]
        block_offset = slot % key_cache.shape[1]
        key_cache[block_idx, block_offset] = key[i]
        value_cache[block_idx, block_offset] = value[i]

def concat_and_cache_mla_fallback(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale):
    """PyTorch fallback for MLA KV cache concat."""
    kv = torch.cat([kv_c, k_pe], dim=-1)
    for i, slot in enumerate(slot_mapping):
        if slot < 0:
            continue
        kv_cache[slot] = kv[i]
```

Alternatively: Route to Triton `fused_kv_cache.py` functions.

#### TopK Fallbacks (`module_moe_asm`, `module_topk`)

```python
def topk_softmax_fallback(topk_weights, topk_ids, token_expert_indicies, gating_output):
    """PyTorch fallback for TopK softmax."""
    scores = torch.softmax(gating_output, dim=-1)
    topk_weights_out, topk_ids_out = torch.topk(scores, k=topk_weights.shape[-1], dim=-1)
    topk_weights.copy_(topk_weights_out)
    topk_ids.copy_(topk_ids_out)

def grouped_topk_fallback(hidden_states, gating_output, topk, renormalize,
                           num_expert_group, topk_group):
    """PyTorch fallback for grouped TopK."""
    scores = torch.softmax(gating_output, dim=-1)
    # Group-level selection then per-group topk
    num_experts = gating_output.shape[-1]
    group_size = num_experts // num_expert_group
    grouped = scores.view(*scores.shape[:-1], num_expert_group, group_size)
    group_scores = grouped.max(dim=-1).values
    top_groups = torch.topk(group_scores, k=topk_group, dim=-1)
    # ... expand to per-expert selection within top groups
```

Route to existing Triton `topk.py` when available.

#### Attention Metadata Fallbacks

`get_pa_metadata_v1` and `get_mla_metadata_v1` are metadata-preparation C++ functions.
These are **NOT compute kernels** — they prepare index buffers for attention dispatch.

Fallback options:
1. **PyTorch reimplementation** — these are index/offset computations, feasible in Python
2. **Triton kernel** — overkill for metadata prep
3. **Keep as HIP JIT** — these are arch-agnostic C++ (no ASM/CK), should compile on MI400

Recommendation: These should compile on MI400 once the arch whitelist is updated (Phase 0).
Add fallback as low priority.

### 1.3 Registration Pattern

Two approaches for wiring fallbacks to `compile_ops`:

**Option A: Inline at definition site (preferred)**

```python
# aiter/ops/norm.py
from aiter.ops.fallbacks import rmsnorm2d_fwd_fallback

@compile_ops("module_norm", fc_name="rmsnorm2d_fwd", fallback=rmsnorm2d_fwd_fallback)
def rmsnorm2d_fwd(out, input, weight, epsilon): ...
```

**Option B: Post-registration**

```python
# aiter/ops/fallbacks.py
from aiter.jit.core import register_fallback

register_fallback("module_norm", "rmsnorm2d_fwd", rmsnorm2d_fwd_fallback)
register_fallback("module_activation", "silu_and_mul", silu_and_mul_fallback)
```

Option A is simpler and keeps fallback registration close to the function definition.
Option B is better for downstream projects (ATOM) that want to register custom fallbacks.

**Recommendation: Implement both.** `fallback` parameter on `compile_ops` for AITER-internal
fallbacks, plus a `register_fallback()` API for downstream overrides.

---

## Layer 2: ATOM-Side Dispatch Changes

### 2.1 New: `ATOM_FORCE_TRITON` Environment Variable

```python
# atom/utils/envs.py
ATOM_FORCE_TRITON: bool = os.environ.get("ATOM_FORCE_TRITON", "0") == "1"
```

When `True`, ATOM proactively selects Triton paths without attempting HIP JIT.
This is the "MI400 big switch" — set it and everything routes to Triton.

### 2.2 New: Architecture Detection Helper

```python
# atom/utils/backends.py
from aiter.jit.utils.chip_info import get_gfx

@functools.lru_cache()
def is_known_arch() -> bool:
    """Returns True if current GPU has known ASM/CK kernel support."""
    gfx = get_gfx()
    return gfx in ("gfx942", "gfx950")

@functools.lru_cache()
def prefer_triton() -> bool:
    """Returns True if Triton should be preferred over ASM/HIP JIT."""
    return envs.ATOM_FORCE_TRITON or not is_known_arch()
```

### 2.3 Modified Dispatch Per Operation

#### attention_mha.py — Paged Attention Decode

```python
# Current (line 124-125):
use_triton_attn = self.sliding_window != -1 or self.head_dim != 128

# New:
from atom.utils.backends import prefer_triton
use_triton_attn = (
    self.sliding_window != -1
    or self.head_dim != 128
    or prefer_triton()
)
```

This routes ALL decode attention to Triton `pa_decode_gluon` on MI400.

#### attention_mha.py — Prefill Attention

```python
# Current dispatch_backend() always returns:
#   prefill: flash_attn_varlen_func (CK)
#   decode: pa_fwd_asm (ASM)

# New: Add Triton prefill path
def dispatch_backend(self, is_prefill, ...):
    if is_prefill:
        if prefer_triton():
            return self.prefill_attention_triton  # unified_attention
        return self.prefill_attention  # flash_attn_varlen_func
    ...
```

#### attention_mla.py — MLA Decode/Prefill

This is the hardest gap. Options:

**Option A (short-term): Decomposed MLA via existing Triton BMM kernels**
```python
def _forward_decode_triton(self, ...):
    # Step 1: Q * K^T attention scores via Triton BMM
    # Step 2: Softmax (Triton softmax.py)
    # Step 3: Scores * V via Triton BMM
    # Step 4: V up-projection via existing _aiter_triton_fp8_bmm
    # This is slower than fused MLA but functionally correct
```

**Option B (medium-term): Wire AITER Triton MLA kernel**
```python
# aiter/ops/triton/attention/mla_decode_rope.py exists
# Need to verify completeness and wire into ATOM dispatch
from aiter.ops.triton.attention.mla_decode_rope import mla_decode_triton
```

**Option C (for bring-up): Fall back to standard MHA path**
```python
def _forward_decode(self, ...):
    if prefer_triton():
        # Decompose MLA to standard multi-head attention
        # q_nope + q_pe -> full Q
        # Reconstruct K from kv_cache
        # Use unified_attention for decode
        return self._forward_decode_as_mha(...)
    return self._forward_decode_mla(...)  # Original ASM path
```

Recommendation: Start with Option C for bring-up, then Option B for performance.

#### moe.py — MoE Backend Selection

```python
# Current (line 643):
gfx = get_gfx()
self.use_triton = gfx.startswith("gfx94") or (
    gfx.startswith("gfx95") and envs.ATOM_USE_TRITON_GEMM
)

# New:
self.use_triton = (
    gfx.startswith("gfx94")
    or (gfx.startswith("gfx95") and envs.ATOM_USE_TRITON_GEMM)
    or prefer_triton()
)
```

#### linear.py — GEMM Backend Selection

```python
# Current: Triton GEMM only used when ATOM_USE_TRITON_GEMM=1

# New: Auto-enable Triton GEMM on unknown architectures
def use_triton_gemm() -> bool:
    return envs.ATOM_USE_TRITON_GEMM or prefer_triton()
```

#### layernorm.py — RMSNorm

```python
# Current: Always uses aiter HIP JIT rmsnorm2d_fwd

# New: When prefer_triton(), use fused Triton path or PyTorch fallback
def forward_native(self, input, residual=None):
    if prefer_triton():
        return self._forward_triton(input, residual)
    return self._forward_hip(input, residual)

def _forward_triton(self, input, residual=None):
    # Use aiter.ops.triton.normalization.rmsnorm or PyTorch
    variance = input.float().pow(2).mean(-1, keepdim=True)
    normed = input * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * normed.to(input.dtype)
```

#### activation.py — SiLU+Mul

```python
# Current: Always uses aiter silu_and_mul for non-quantized path

# New:
def forward(self, x, x_scale=None):
    if x_scale is not None and self.fused_quant:
        return self._forward_fused_fp8(x, x_scale)  # Already Triton
    elif self.fused_quant and self.quant_type == QuantType.per_1x32:
        return self._forward_fused_mxfp4(x)  # Already Triton
    elif prefer_triton():
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    else:
        out = torch.empty(...)
        silu_and_mul(out, x)  # HIP JIT
        return out
```

#### rotary_embedding.py — RoPE

```python
# Current: Always uses aiter.rope_cached_positions_2c_fwd_inplace

# New:
def forward(self, positions, query, key):
    if prefer_triton():
        # Use the existing apply_rotary_emb (already defined in file!)
        # Or route to aiter.ops.triton.rope.rope
        return self._forward_triton(positions, query, key)
    aiter.rope_cached_positions_2c_fwd_inplace(...)
```

---

## Implementation Phases

### Phase A: AITER Core Infrastructure (PR to ROCm/aiter)

| Task | Files | Effort | Dependency |
|------|-------|--------|------------|
| A1. Add `_failed_modules` dict + modify `compile_ops` wrapper | `aiter/jit/core.py` | S | None |
| A2. Add `register_fallback()` API | `aiter/jit/core.py` | S | A1 |
| A3. Fix `get_module()` LRU cache to not cache failures | `aiter/jit/core.py` | S | A1 |
| A4. Add MI400 gfx to arch whitelist | `aiter/jit/core.py` | XS | None |
| A5. Write PyTorch fallbacks for `module_norm` | `aiter/ops/fallbacks.py` (new) | M | A1 |
| A6. Write PyTorch fallbacks for `module_activation` | `aiter/ops/fallbacks.py` | S | A1 |
| A7. Write PyTorch fallbacks for `module_cache` | `aiter/ops/fallbacks.py` | M | A1 |
| A8. Write Triton-routing fallbacks for `module_moe_asm` TopK | `aiter/ops/fallbacks.py` | M | A1 |
| A9. Wire fallbacks to compile_ops decorators in ops/*.py | `aiter/ops/norm.py`, `activation.py`, `cache.py` | M | A5-A8 |
| A10. Tests for fallback dispatch | `tests/test_ck_free_fallbacks.py` | M | A9 |

### Phase B: ATOM Dispatch Changes (PR to sunway513/ATOM)

| Task | Files | Effort | Dependency |
|------|-------|--------|------------|
| B1. Add `ATOM_FORCE_TRITON` env var | `atom/utils/envs.py` | XS | None |
| B2. Add `prefer_triton()` / `is_known_arch()` helpers | `atom/utils/backends.py` | S | B1 |
| B3. Modify PA decode dispatch | `atom/model_ops/attention_mha.py` | S | B2 |
| B4. Modify prefill attention dispatch | `atom/model_ops/attention_mha.py` | S | B2 |
| B5. Modify MoE gfx detection | `atom/model_ops/moe.py` | S | B2 |
| B6. Modify GEMM dispatch (auto-enable Triton) | `atom/model_ops/linear.py` | S | B2 |
| B7. Add RMSNorm Triton/PyTorch fallback | `atom/model_ops/layernorm.py` | M | B2 |
| B8. Add SiLU PyTorch fallback | `atom/model_ops/activation.py` | S | B2 |
| B9. Add RoPE Triton fallback | `atom/model_ops/rotary_embedding.py` | S | B2 |
| B10. MLA decode decomposed fallback | `atom/model_ops/attention_mla.py` | L | B2 |
| B11. Integration test: Llama on simulated MI400 | tests/ | M | B3-B9 |
| B12. Integration test: DeepSeek on simulated MI400 | tests/ | M | B10 |

### Phase C: Validation

| Task | Effort | Dependency |
|------|--------|------------|
| C1. Correctness: Compare Triton vs HIP JIT outputs on gfx942 | M | B11 |
| C2. Performance: Benchmark Triton-only vs mixed on gfx942 | M | B11 |
| C3. CI: Add `ATOM_FORCE_TRITON=1` test job | S | B11 |

---

## Effort Estimates

| Phase | Total Effort |
|-------|-------------|
| Phase A (AITER) | ~1 week |
| Phase B (ATOM) — excluding MLA | ~1 week |
| Phase B10 (MLA fallback) | ~1 week |
| Phase C (Validation) | ~3-5 days |
| **Total** | **~3-4 weeks** |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Triton kernel numerical mismatch vs HIP JIT | Model output divergence | Phase C1 validation; accept small epsilon diffs |
| `get_module()` LRU cache interaction with `_failed_modules` | Stale cache entries | Clear LRU cache entry on failure before adding to `_failed_modules` |
| MLA decomposed decode is too slow for production | DeepSeek-R1 unusable perf | Acceptable for bring-up; fast-follow with AITER Triton MLA kernel |
| AITER upstream review cycle delays | MI400 timeline risk | ATOM-side Layer 2 can ship independently of AITER PR 2 |
| `compile_ops` decorator is used at import time, not call time | Fallback must handle import-time failures too | Ensure `_failed_modules` check happens in wrapper, not in decorator body |

---

## Dependency Graph

```
A4 (arch whitelist) ─────────────────────────────────┐
                                                      │
A1 (compile_ops fallback) → A5-A8 (fallbacks) → A9 → A10
                                                      │
B1 (ATOM_FORCE_TRITON) → B2 (prefer_triton) ─────────┤
                              │                       │
                   ┌──────────┼──────────┐            │
                   ▼          ▼          ▼            │
                  B3-B4      B5-B6     B7-B9          │
                (attention)  (moe/gemm) (norm/act)    │
                   │          │          │            │
                   └──────────┼──────────┘            │
                              ▼                       │
                          B10 (MLA)                   │
                              │                       │
                              ▼                       │
                         B11-B12 (tests)              │
                              │                       │
                              ▼                       │
                         C1-C3 (validation) ◄─────────┘
```

**Critical Path:** A1 → A9 → B2 → B3 → B11 (Llama bring-up)
**DeepSeek Critical Path:** A1 → A9 → B2 → B10 → B12

---

## Testing Strategy

### Unit Tests (per fallback)

For each fallback function, compare output against HIP JIT version on supported GPU:

```python
def test_rmsnorm_fallback_matches_hip():
    x = torch.randn(4, 128, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")

    out_hip = torch.empty_like(x)
    rmsnorm2d_fwd(out_hip, x, w, 1e-5)

    out_fb = torch.empty_like(x)
    rmsnorm2d_fwd_fallback(out_fb, x, w, 1e-5)

    torch.testing.assert_close(out_hip, out_fb, atol=1e-3, rtol=1e-3)
```

### Integration Tests

```python
def test_llama_triton_only():
    """Run Llama-3.1-8B inference with ATOM_FORCE_TRITON=1."""
    os.environ["ATOM_FORCE_TRITON"] = "1"
    # Run single prefill + decode step
    # Compare logits against reference (from HIP JIT run)

def test_deepseek_triton_only():
    """Run DeepSeek-R1 inference with ATOM_FORCE_TRITON=1."""
    os.environ["ATOM_FORCE_TRITON"] = "1"
    # Run single prefill + decode step with MoE + MLA
```

### Simulated MI400 Test

On gfx942, test with `ATOM_FORCE_TRITON=1` to validate the Triton-only path
without needing actual MI400 hardware:

```bash
ATOM_FORCE_TRITON=1 python -m atom.entrypoints.openai_server \
    --model meta-llama/Llama-3.1-8B \
    --quantization fp8_blockscale \
    --tensor-parallel-size 1
```

---

## Open Questions

1. **Should `ATOM_FORCE_TRITON` be auto-detected?** If `get_gfx()` returns an unknown arch,
   should ATOM automatically enable Triton-only mode without explicit env var?
   → Recommendation: Yes, via `prefer_triton() = not is_known_arch()`. Env var is override.

2. **Should fallbacks be registered at AITER level or ATOM level?**
   → Both. AITER provides basic PyTorch fallbacks. ATOM can override with Triton-routing
   fallbacks that leverage ATOM's dispatch knowledge.

3. **`register_fallback()` priority order?**
   → ATOM-registered > AITER `compile_ops` fallback > crash. Last registered wins.

4. **Should MLA decode fallback decompose to MHA or use Triton MLA?**
   → For bring-up: decompose to MHA (known to work). Then validate AITER's
   `mla_decode_rope.py` Triton kernel and switch to it when confirmed correct.

5. **FP8 per-token GEMM (`gemm_a8w8_bpreshuffle`) has NO Triton path — what to do?**
   → Fall back to FP8 blockscale Triton GEMM (different quantization scheme) or
   BF16 `torch.F.linear`. For MI400 bring-up, recommend blockscale or MXFP4 quantization.
