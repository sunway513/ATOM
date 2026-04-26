"""W4 path kernel bisect repro (issue sunway513/atom#37).

Locates the AITER kernel call that triggers HSA exception 0x1016 in the
W4 multi-request path on MI355. Run with HIP_LAUNCH_BLOCKING=1 so each
kernel completes synchronously and the abort happens AT the offending
call (not deferred via async queue).

Usage:
    HIP_LAUNCH_BLOCKING=1 \
    AMD_LOG_LEVEL=3 \
    ATOM_DSV4_USE_W4_PATH=1 \
    ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 \
    ATOM_AITER_VALIDATE=1 \
    /opt/venv/bin/python -m tests.silicon.repro_w4_kernel_bisect \
        --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
        --kv_cache_dtype fp8 -tp 8 \
        --max-num-seqs 1 --num-prompts 1 \
        --max-tokens 32 --max-model-len 2048 --enforce-eager \
        --gpu-memory-utilization 0.85 \
        2>&1 | tee /workspace/ATOM-lingpeng/logs/repro_w4_bisect.log

The output will show "[bisect step N] entering <name>" /
"[bisect step N] exiting <name> dt=<ms>" pairs. The last "entering"
without a matching "exiting" is the kernel that crashed — that's the
bisect result for the AITER kernel team.

Once isolated, write a standalone 50-line reproducer that calls just
that kernel with the captured input shapes/dtypes/strides. The AITER
team can then bisect within the kernel without ATOM dependencies.
"""

import argparse
import json
import os
import time

# Force synchronous kernel execution so HSA abort happens AT the call.
os.environ["HIP_LAUNCH_BLOCKING"] = "1"

import torch  # noqa: E402

from atom import SamplingParams  # noqa: E402
from atom.model_engine.arg_utils import EngineArgs  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# ----- Instrument deepseek_v4 with kernel-call markers ----------------------
# Monkey-patch a small set of suspect AITER kernel call sites in
# DeepseekV4Attention._forward_w4 to print before/after each. The patches
# are minimally invasive — no logic change, just stderr prints with
# torch.cuda.synchronize() between each so the abort is precisely
# attributable.

_BISECT_STEP = [0]


def _mark(label: str) -> float:
    """Print a bisect marker. The print is unbuffered + sync'd so the
    last marker before HSA abort is the kernel that crashed."""
    _BISECT_STEP[0] += 1
    n = _BISECT_STEP[0]
    msg = f"[bisect step {n}] entering {label}"
    print(msg, flush=True)
    # Force a sync so any pending kernel surfaces its error here, not
    # at the next CPU-GPU boundary.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _unmark(label: str, t0: float) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000
    print(
        f"[bisect step {_BISECT_STEP[0]}] exiting {label} dt={dt_ms:.2f}ms", flush=True
    )


def _install_patches() -> None:
    """Wrap a curated set of W4-path kernel call sites with bisect markers."""
    from atom.model_ops import quant_v4
    from atom.model_ops import linear
    from atom.model_ops import sparse_attn_v4

    # 1. Quantization fallbacks (pure torch but use float8 — ROCm path)
    _orig_act_quant = quant_v4.act_quant_inplace
    _orig_fp4_quant = quant_v4.fp4_act_quant_inplace
    _orig_rotate = quant_v4.rotate_activation

    def _patched_act_quant(x, *a, **kw):
        t = _mark(f"act_quant_inplace shape={tuple(x.shape)} dtype={x.dtype}")
        try:
            return _orig_act_quant(x, *a, **kw)
        finally:
            _unmark("act_quant_inplace", t)

    def _patched_fp4_quant(x, *a, **kw):
        t = _mark(f"fp4_act_quant_inplace shape={tuple(x.shape)} dtype={x.dtype}")
        try:
            return _orig_fp4_quant(x, *a, **kw)
        finally:
            _unmark("fp4_act_quant_inplace", t)

    def _patched_rotate(x, *a, **kw):
        t = _mark(f"rotate_activation shape={tuple(x.shape)} dtype={x.dtype}")
        try:
            return _orig_rotate(x, *a, **kw)
        finally:
            _unmark("rotate_activation", t)

    quant_v4.act_quant_inplace = _patched_act_quant
    quant_v4.fp4_act_quant_inplace = _patched_fp4_quant
    quant_v4.rotate_activation = _patched_rotate

    # The deepseek_v4 module already imported these symbols; rebind there too.
    from atom.models import deepseek_v4 as _dv4

    _dv4.act_quant_inplace = _patched_act_quant
    _dv4.fp4_act_quant_inplace = _patched_fp4_quant
    _dv4.rotate_activation = _patched_rotate

    # 2. sparse_attn (pure torch fallback in this build, but log inputs)
    _orig_sparse = sparse_attn_v4.sparse_attn

    def _patched_sparse(q, kv, attn_sink, topk_idxs, scale):
        label = (
            f"sparse_attn q={tuple(q.shape)} kv={tuple(kv.shape)} "
            f"topk={tuple(topk_idxs.shape)} dtype_q={q.dtype} dtype_kv={kv.dtype}"
        )
        t = _mark(label)
        try:
            return _orig_sparse(q, kv, attn_sink, topk_idxs, scale)
        finally:
            _unmark("sparse_attn", t)

    sparse_attn_v4.sparse_attn = _patched_sparse
    _dv4.sparse_attn = _patched_sparse

    # 3. AITER GEMM call sites in linear.py — wrap the dispatch table
    # entries. Each Linear.forward picks one of these via use_triton_gemm()
    # / FP8/FP4 dispatch.
    for name in (
        "gemm_a4w4",
        "gemm_a8w8",
        "gemm_a8w8_blockscale_bpreshuffle",
        "gemm_a8w8_bpreshuffle",
    ):
        if hasattr(linear, name):
            orig = getattr(linear, name)
            if callable(orig):

                def _make_patch(_orig, _name):
                    def _p(*a, **kw):
                        # First positional arg is usually the activation tensor
                        shapes = []
                        for arg in a[:3]:
                            if torch.is_tensor(arg):
                                shapes.append(f"{tuple(arg.shape)}")
                        t = _mark(f"{_name} args[0:3].shapes={shapes}")
                        try:
                            return _orig(*a, **kw)
                        finally:
                            _unmark(_name, t)

                    return _p

                setattr(linear, name, _make_patch(orig, name))

    # 4. tuned_gemm.tgemm
    try:
        from aiter import tuned_gemm

        _orig_tgemm = tuned_gemm.tgemm

        def _patched_tgemm(*a, **kw):
            shapes = []
            for arg in a[:3]:
                if torch.is_tensor(arg):
                    shapes.append(f"{tuple(arg.shape)}")
            t = _mark(f"tgemm args[0:3].shapes={shapes}")
            try:
                return _orig_tgemm(*a, **kw)
            finally:
                _unmark("tgemm", t)

        tuned_gemm.tgemm = _patched_tgemm
    except Exception as e:
        print(f"[bisect] could not patch tgemm: {e}", flush=True)

    print("[bisect] kernel call-site patches installed", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--out", type=str, default="/workspace/ATOM-lingpeng/logs/repro_w4_bisect.json"
    )
    args = parser.parse_args()

    _install_patches()

    # Single short prompt to keep the kernel sequence small + bisect-friendly.
    prompts_raw = ["如何在一个月内增肌10公斤"]
    if args.num_prompts > 1:
        prompts_raw += [
            "Briefly describe Beijing in 3 sentences.",
            "Write a Python function to compute the nth Fibonacci number.",
            "List 5 common machine learning algorithms.",
        ][: args.num_prompts - 1]

    sizes, p = [], 1
    while p <= args.num_prompts:
        sizes.append(p)
        p *= 2
    args.cudagraph_capture_sizes = str(sizes)

    print("=== W4 kernel bisect repro ===")
    print(f"  HIP_LAUNCH_BLOCKING={os.getenv('HIP_LAUNCH_BLOCKING')}")
    print(f"  AMD_LOG_LEVEL={os.getenv('AMD_LOG_LEVEL', 'unset')}")
    print(f"  ATOM_DSV4_USE_W4_PATH={os.getenv('ATOM_DSV4_USE_W4_PATH', '0')}")
    print(f"  num_prompts={args.num_prompts}")
    print(f"  max_num_seqs={args.max_num_seqs}")

    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    AutoTokenizer.from_pretrained(args.model)  # warm tokenizer cache

    # Apply V4 chat template
    enc_path = os.path.join(args.model, "encoding", "encoding_dsv4.py")
    prompts = list(prompts_raw)
    if os.path.exists(enc_path):
        import importlib.util

        spec = importlib.util.spec_from_file_location("encoding_dsv4", enc_path)
        enc_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enc_mod)
        prompts = [
            enc_mod.encode_messages(
                [{"role": "user", "content": p}], thinking_mode="chat"
            )
            for p in prompts_raw
        ]

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )

    print(
        "=== generating (HSA abort expected at last [bisect step N] entering line) ==="
    )
    outputs = llm.generate(prompts, sampling_params)

    # If we get here, no abort.
    results = []
    for i, (raw, output) in enumerate(zip(prompts_raw, outputs)):
        text = output["text"]
        results.append({"idx": i, "prompt": raw, "completion": text})

    with open(args.out, "w") as f:
        json.dump(
            {"final_bisect_step": _BISECT_STEP[0], "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"=== ALL DONE final_bisect_step={_BISECT_STEP[0]} (no crash) ===")


if __name__ == "__main__":
    main()
