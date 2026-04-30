"""Evidence script: capture multi-request failure signature on lingpeng/dsv4-pr1-skeleton.

Runs N=4 prompts together (lockstep batch). Saves each output. Includes the
same hero prompt (idx 0) as the conc=1 baseline so we can do token-ID equality
check against silicon_fact_short.log offline.
"""
import argparse
import json
import os
import sys

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer


HERO = "如何在一个月内增肌10公斤"  # MUST match simple_inference.py:42 baseline
SECONDARY = [
    "Briefly describe Beijing in 3 sentences.",
    "Write a Python function to compute the nth Fibonacci number.",
    "List 5 common machine learning algorithms.",
]


def main():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-prompts", type=int, default=4, choices=[1, 2, 4])
    parser.add_argument("--out", type=str, default="/workspace/ATOM-lingpeng/logs/silicon_fact_multireq.json")
    args = parser.parse_args()
    n = args.num_prompts
    prompts_raw = [HERO] + SECONDARY[: n - 1] if n > 1 else [HERO]

    # power-of-2 cudagraph sizes  
    sizes, p = [], 1
    while p <= n:
        sizes.append(p); p *= 2
    args.cudagraph_capture_sizes = str(sizes)

    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply V4 encoding (mirrors simple_inference.py path)
    enc_path = os.path.join(args.model, "encoding", "encoding_dsv4.py")
    prompts = list(prompts_raw)
    if os.path.exists(enc_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("encoding_dsv4", enc_path)
        enc_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(enc_mod)
        prompts = [enc_mod.encode_messages([{"role": "user", "content": p}], thinking_mode="chat") for p in prompts_raw]
        print(f"  V4 encoding applied, tokens per prompt: {[len(tokenizer.encode(p)) for p in prompts]}")

    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    print(f"=== conc={n} | running batch ===")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, (raw, output) in enumerate(zip(prompts_raw, outputs)):
        text = output["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False) if text else []
        results.append({"idx": i, "prompt": raw, "completion": text, "token_ids": token_ids[:64]})
        print(f"\n[idx={i}] prompt: {raw!r}")
        print(f"[idx={i}] completion ({len(text)} chars): {text!r}")
        print(f"[idx={i}] first 16 token_ids: {token_ids[:16]}")

    with open(args.out, "w") as f:
        json.dump({"conc": n, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
