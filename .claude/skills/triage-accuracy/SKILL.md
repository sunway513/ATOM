# Skill: Triage Accuracy Issues in LLM Inference

## Description
Systematic methodology for diagnosing and fixing correctness/accuracy issues in LLM inference — garbage output, wrong answers, numerical instability, or degraded quality. Works across any backend (CK, Triton, ASM, hipBLASLt), any quantization format (FP8, FP4, INT8, BF16), and any model architecture.

## When to Use
- Model produces garbage, spaces, random tokens, or incoherent text
- Model gives wrong answers but no errors/crashes/NaN
- Output quality degraded after a code change, build change, or environment change
- Numerical instability (NaN/Inf) during inference
- Suspected kernel, quantization, or routing bugs

---

## Core Principle

**Check DIRECTIONS, not magnitudes.** Norms, means, min/max can all look reasonable while the hidden state points in a completely wrong direction. **Cosine similarity against a known-good reference** is the gold standard metric.

| Cosine Similarity | Verdict |
|-------------------|---------|
| > 0.999 | Correct |
| 0.99 – 0.999 | Suspicious — investigate |
| 0.9 – 0.99 | Broken — will degrade over layers |
| < 0.9 | Catastrophically wrong |

A per-layer cos_sim of 0.85 looks tolerable but compounds across N layers. For a 61-layer model, the final hidden state can be completely uncorrelated with the reference (cos_sim ~ 0.05).

---

## Phase 1: Reproduce and Characterize

### 1.1 Use completions API, not chat
Chat APIs apply templates, `<think>` handling, and streaming that can mask issues:
```bash
curl -s http://localhost:PORT/v1/completions \
  -d '{"model":"MODEL","prompt":"The capital of France is","max_tokens":30,"temperature":0}'
```

### 1.2 Use deterministic prompts with known answers
| Prompt | Expected | Tests |
|--------|----------|-------|
| "The capital of France is" | " Paris" | Basic knowledge |
| "1+1=" | "2" | Arithmetic |
| "John went to the store. John" | "bought" or similar | Coherence |
| "The" | Any reasonable continuation | Not garbage |

### 1.3 Compare with a known-good reference
Run identical prompts on a reference build/image. If both are wrong, the issue is upstream (model weights, config). If only one is wrong, the issue is in the code delta between them.

### 1.4 Check for build/JIT failures
Look for silent JIT failures in stderr:
```bash
# AITER JIT
grep -i "failed jit build" server.log
# General
grep -i "fallback\|warning\|error" server.log | head -50
```
Silent fallback to a less-tested code path is a top root cause.

---

## Phase 2: Binary Search Through the Model Pipeline

Start from both ends (input and output) and narrow inward.

### Step 1: Verify Embeddings
```python
from safetensors import safe_open
with safe_open(path, framework='pt', device='cpu') as sf:
    embed_w = sf.get_tensor('model.embed_tokens.weight')

ref = embed_w[token_ids]  # should match exactly
cos = F.cosine_similarity(ref.flatten(), model_hidden.flatten(), dim=0)
# Expected: 1.000000 (exact match)
```

### Step 2: Verify lm_head (output projection)
```python
logits_manual = hidden_states @ lm_head_weight.T
cos = F.cosine_similarity(logits_manual.flatten(), model_logits.flatten(), dim=0)
# If this matches but final output is wrong -> bug is in the transformer layers
```

### Step 3: Narrow to layer granularity
```python
for layer_idx in [0, 15, 30, 45, 60]:
    cos = F.cosine_similarity(
        test_hidden[layer_idx].flatten(),
        ref_hidden[layer_idx].flatten(), dim=0
    )
    print(f"Layer {layer_idx}: cos_sim={cos:.6f}")
```
Find where cos_sim drops — the bug is in that layer's subcomponents.

### Step 4: Narrow within a layer
1. **RMSNorm / LayerNorm** — recompute manually
2. **Attention** (Q/K/V projections, RoPE, softmax, output projection)
3. **MoE routing** (expert selection, gating scores)
4. **FP8/INT8 GEMM** (quantize -> matmul -> dequantize)
5. **Residual connection** (addition, not a common failure point)

---

## Phase 3: Build Standalone Verification

```python
import torch
from safetensors import safe_open

# 1. Load actual model weights (bypass model loading)
with safe_open(safetensors_path, framework='pt', device='cpu') as sf:
    weight = sf.get_tensor('model.layers.0.mlp.gate_proj.weight')
    scale = sf.get_tensor('model.layers.0.mlp.gate_proj.weight_scale_inv')

# 2. Compute FP32 reference on CPU
weight_f32 = dequantize(weight, scale)  # manual block dequant
ref_output = input_f32 @ weight_f32.T

# 3. Call actual kernel with same inputs
kernel_output = suspect_kernel(input_quant, weight, input_scale, weight_scale)

# 4. Compare
cos = F.cosine_similarity(ref_output.flatten(), kernel_output.float().flatten(), dim=0)
print(f"cos_sim = {cos:.6f}")  # < 0.999 = bug confirmed
```

### Key: Test the FULL chain, not components in isolation
The most dangerous bugs live at **interfaces between components**. A GEMM kernel can be correct, and a quantization kernel can be correct, but if the scale layout from quant doesn't match what GEMM expects, the result is wrong. Always test quant->GEMM as one unit.

---

## Phase 4: Trace the Data Flow

### For quantized GEMM paths (ATOM):
```
model.forward()
  -> LinearBase.forward()         # atom/model_ops/linear.py
    -> quant_function(input, ...)  # produces (input_fp8, input_scale)
      -> [primary kernel OR fallback]
    -> gemm_function(input_fp8, weight, input_scale, weight_scale)
      -> [primary kernel OR fallback]
```

### What to check at each hop:
1. **Tensor shapes** — are they what the next function expects?
2. **Memory layout** — row-major vs column-major, contiguous vs strided
3. **Scale tensor layout** — this is the #1 source of silent correctness bugs
4. **Dtype** — especially FP8 variants (e4m3fn vs e4m3fnuz on AMD)
5. **Fallback path** — does the fallback handle ALL parameters the primary does?

### The "Silent Parameter" Bug Pattern
```python
# DANGEROUS: function accepts parameter but ignores it
def fallback_quant(out, input, scales, shuffle_scale=False):
    triton_quant(out, input, scales)
    # shuffle_scale is SILENTLY IGNORED
```
Every fallback function should either implement the parameter fully, or `raise NotImplementedError`.

---

## Phase 5: Fix and Verify

### 5.1 Apply minimal fix
### 5.2 Verify with standalone test (cos_sim should jump from 0.7-0.9 to 0.9999+)
### 5.3 Verify end-to-end on deterministic prompts from Phase 1
### 5.4 Docker patch workflow
```bash
docker cp fix.py container:/tmp/
docker exec container python3 /tmp/fix.py
docker commit container image:fixed
curl -s http://localhost:PORT/v1/completions -d '{"prompt":"The capital of France is",...}'
```

---

## Common Root Causes (Ranked by Frequency)

1. **Scale/metadata layout mismatch** — quant writes one layout, GEMM expects another. Produces cos_sim 0.7-0.9 per layer, compounds to garbage.
2. **Silent fallback to untested code path** — JIT build fails, fallback ignores parameters.
3. **Dtype mismatch** — FP8 variants (e4m3fn vs e4m3fnuz). gfx942=FNUZ, gfx950=FN, safetensors=FN.
4. **Transpose/permutation errors** — correct shape, wrong values.
5. **Precision loss in accumulation** — FP8->FP16 instead of FP8->FP32.
6. **RoPE / positional encoding bugs** — wrong frequency, dimension ordering, or position indices.

---

## Anti-Patterns

1. **Don't check only magnitudes** — check directions (cosine similarity).
2. **Don't test components in isolation only** — test the full quant->kernel chain.
3. **Don't assume fallbacks are complete** — verify every parameter is handled.
4. **Don't use chat API for initial triage** — use `/v1/completions`.
5. **Don't assume the first wrong layer IS the root cause** — systematic bugs affect ALL layers equally.
6. **Don't rebuild containers for each test** — patch in-place, commit, test.
7. **Don't trust "no NaN/Inf" as proof of correctness** — the most dangerous bugs produce finite, wrong-direction outputs.

---

## Decision Tree

```
Model output is wrong/garbage
|
+- NaN/Inf in output?
|  +- YES -> dtype mismatch, overflow, missing eps
|  +- NO -> continue
|
+- First token wrong? (completions API)
|  +- YES -> Prefill broken (GEMM/quant/attention)
|  +- NO -> Decode broken (KV cache, decode kernels)
|
+- Embeddings correct? (compare with safetensors)
|  +- NO -> Weight loading or dtype conversion bug
|  +- YES -> continue
|
+- lm_head correct? (manual matmul matches logits?)
|  +- NO -> lm_head kernel or weight bug
|  +- YES -> Bug in transformer layers
|
+- Binary search layers: where does cos_sim drop?
|  +- EVERY LAYER -> Systematic bug (quant, scale layout, dispatch)
|  +- SPECIFIC LAYER -> Layer-specific bug (attention, RoPE, routing)
|
+- Found the buggy kernel?
   +- Using a fallback? Does fallback handle ALL params?
   +- Scale layout match between producer and consumer?
   +- Correct dtype for this GPU architecture?
```
