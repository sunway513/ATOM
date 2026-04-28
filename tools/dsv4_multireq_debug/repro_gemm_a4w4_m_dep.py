"""Minimal reproducer for AITER `gemm_a4w4` batch-dependence bug.

Symptom: same input row x[0] produces DIFFERENT output y[0] when called
with batch m=1 vs m=2 (both <MXFP4_QUANT_BLOCK_SIZE=32). Discovered while
debugging DeepSeek-V4-Pro multi-request decode in ATOM
(sunway513/ATOM#37).

Per-stage trace at L00 of the W4 attention path:
- input x[row_for_seqA] is bit-identical between conc=1 and conc=2 (max
  diff = 0.000)
- after wq_b (a4w4 GEMM), y[row_for_seqA] differs by 0.569 (8% of norm)
- divergence then amplifies through RMSNorm → 9.94 → garbled tokens

Expected: y_m1[0] == y_m2[0] (within FP rounding) since x[0] is identical
and the GEMM is per-row. Observed: ~0.5 max abs diff (well above noise).

Run: python repro_gemm_a4w4_m_dep.py
"""
import torch
from aiter import QuantType, gemm_a4w4, get_hip_quant
from aiter.utility import fp4_utils

MXFP4_QUANT_BLOCK_SIZE = 32

torch.manual_seed(0)
device = "cuda:0"
N = 8192  # output dim (mirrors DSV4 wq_b output)
K = 1536  # input dim  (mirrors DSV4 wq_b input = q_lora_rank)

quant_func = get_hip_quant(QuantType.per_1x32)

# Build a random MXFP4 weight (per-1x32 quant, no shuffle).
w_bf16 = (torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.05).contiguous()
w_q, w_s = quant_func(w_bf16, quant_dtype=torch.float4_e2m1fn_x2, shuffle=False)
w_s_shuffled = fp4_utils.e8m0_shuffle(w_s)

# Fix one input row (this is the row whose output we compare).
x_row0 = (torch.randn(1, K, dtype=torch.bfloat16, device=device) * 0.5).contiguous()


def run(x: torch.Tensor) -> torch.Tensor:
    """Mirror atom/model_ops/linear.py:gemm_a4w4_quant Path-1 call."""
    m = x.shape[0]
    x_q, x_s = quant_func(x, quant_dtype=torch.float4_e2m1fn_x2, shuffle=True)
    y = torch.empty(
        ((m + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE * MXFP4_QUANT_BLOCK_SIZE, N),
        dtype=torch.bfloat16,
        device=x.device,
    )
    y = gemm_a4w4(x_q, w_q, x_s, w_s_shuffled, y)
    return y[:m]


# m=1
y_m1 = run(x_row0)
# m=2 with x_row0 as row 0 + a random row 1
x_extra = (torch.randn(1, K, dtype=torch.bfloat16, device=device) * 0.5).contiguous()
x_m2 = torch.cat([x_row0, x_extra], dim=0).contiguous()
y_m2 = run(x_m2)

# Compare row 0 (driven by SAME x_row0 in both calls).
diff = (y_m1[0].float() - y_m2[0].float()).abs()
print("=" * 70)
print(f"AITER gemm_a4w4 batch-dependence repro (sunway513/aiter@e450e4d)")
print("=" * 70)
print(f"input dim K={K}, output dim N={N}, MXFP4 block={MXFP4_QUANT_BLOCK_SIZE}")
print(f"y_m1[0]  norm: {y_m1[0].float().norm().item():.4f}")
print(f"y_m2[0]  norm: {y_m2[0].float().norm().item():.4f}  (should match y_m1[0])")
print(f"max abs diff (y_m1[0] vs y_m2[0]): {diff.max().item():.6f}")
print(f"mean abs diff: {diff.mean().item():.6f}")
print()
threshold = 0.05  # bf16 + per-1x32 noise floor
status = "PASS" if diff.max().item() < threshold else "FAIL"
print(f"Verdict: {status} (threshold={threshold} for bf16+mxfp4 noise)")
