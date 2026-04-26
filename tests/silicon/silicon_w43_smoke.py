"""W4.3 silicon smoke harness (issue sunway513/atom#37 W4.5).

Two modes via --mode flag:
  - single: max_num_seqs=1, num_prompts=1 (legacy fallback baseline,
            or W4 single-prompt if flags set)
  - multi:  max_num_seqs=4, num_prompts=4 (W4 multi-request validation)

Required env vars for the W4 path:
  ATOM_DSV4_USE_W4_PATH=1
  ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1
  ATOM_AITER_VALIDATE=1   # debug, recommended for first runs

Without those flags, the harness will hit the Path 2 hard-assert when
--max-num-seqs > 1.
"""

import argparse
import json
import os

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer

HERO = "如何在一个月内增肌10公斤"
SECONDARY = [
    "Briefly describe Beijing in 3 sentences.",
    "Write a Python function to compute the nth Fibonacci number.",
    "List 5 common machine learning algorithms.",
]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        required=True,
        help="single: 1 prompt @ max_num_seqs=1; multi: 4 prompts @ max_num_seqs=4",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--out",
        type=str,
        default="/workspace/ATOM-lingpeng/logs/silicon_w43.json",
    )
    args = parser.parse_args()

    # Apply mode defaults
    if args.mode == "single":
        args.max_num_seqs = 1
        n_prompts = 1
    else:  # multi
        args.max_num_seqs = 4
        n_prompts = 4

    prompts_raw = [HERO] + SECONDARY[: n_prompts - 1] if n_prompts > 1 else [HERO]

    # power-of-2 cudagraph sizes
    sizes, p = [], 1
    while p <= n_prompts:
        sizes.append(p)
        p *= 2
    args.cudagraph_capture_sizes = str(sizes)

    print("=== W4.3 silicon smoke ===")
    print(f"  mode={args.mode}")
    print(f"  max_num_seqs={args.max_num_seqs}")
    print(f"  num_prompts={n_prompts}")
    print(f"  ATOM_DSV4_USE_W4_PATH={os.getenv('ATOM_DSV4_USE_W4_PATH', '0')}")
    print(
        f"  ATOM_DSV4_UNSAFE_MULTIREQ_DEV={os.getenv('ATOM_DSV4_UNSAFE_MULTIREQ_DEV', '0')}"
    )
    print(f"  ATOM_AITER_VALIDATE={os.getenv('ATOM_AITER_VALIDATE', '0')}")

    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Apply V4 chat template / encoding (mirrors silicon_fact_multireq.py)
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
        print(
            f"  V4 encoding applied, tokens per prompt: {[len(tokenizer.encode(p)) for p in prompts]}"
        )

    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens
    )
    print("=== running batch ===")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, (raw, output) in enumerate(zip(prompts_raw, outputs)):
        text = output["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False) if text else []
        results.append(
            {
                "idx": i,
                "prompt": raw,
                "completion": text,
                "token_ids": token_ids[:64],
            }
        )
        print(f"\n[idx={i}] prompt: {raw!r}")
        print(f"[idx={i}] completion ({len(text)} chars): {text!r}")
        print(f"[idx={i}] first 16 token_ids: {token_ids[:16]}")

    payload = {
        "mode": args.mode,
        "max_num_seqs": args.max_num_seqs,
        "num_prompts": n_prompts,
        "max_tokens": args.max_tokens,
        "env": {
            "ATOM_DSV4_USE_W4_PATH": os.getenv("ATOM_DSV4_USE_W4_PATH", "0"),
            "ATOM_DSV4_UNSAFE_MULTIREQ_DEV": os.getenv(
                "ATOM_DSV4_UNSAFE_MULTIREQ_DEV", "0"
            ),
            "ATOM_AITER_VALIDATE": os.getenv("ATOM_AITER_VALIDATE", "0"),
        },
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n=== wrote {args.out} ===")


if __name__ == "__main__":
    main()
