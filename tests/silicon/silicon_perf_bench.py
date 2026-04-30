"""W3.2 silicon perf benchmark — DSV4-Pro on MI355X TP=8.

Measures TPOT / TTFT / throughput at variable concurrency + ISL/OSL,
matching InferenceX (formerly InferenceMax) DSV4-Pro axes per
RFC v0.2.6 §1.1 (1k/1k, 2k/2k, 8k/1k, 32k/2k).

Output correctness is NOT verified here — W3.2-v1 still has READ-side
cross-talk pollution (RFC §3.1). This script measures **compute
throughput** independent of output correctness, which is the right
slicing for perf comparison vs the InferenceX leaderboard.
"""

import argparse
import json
import os
import time

from atom import SamplingParams
from atom.model_engine.arg_utils import EngineArgs
from transformers import AutoTokenizer


# Synthesize a prompt of approximately `isl_tokens` length using a
# concatenation of varied technical prompts so the tokenizer doesn't
# compress to a tiny token count.
_FILLER = (
    "请详细对比并解释以下技术概念，给出三种实际应用案例和潜在风险。"
    "讨论需要覆盖底层原理、性能权衡、工程实现细节，以及与近期"
    "研究的关联。 "
)
_TOPICS = [
    "深度学习中的注意力机制",
    "强化学习的策略梯度方法",
    "大语言模型的稀疏专家",
    "计算机视觉中的Transformer",
    "自然语言处理的对齐技术",
    "图神经网络的消息传递",
    "多模态学习的对比目标",
    "联邦学习的隐私保护",
    "时间序列预测的状态空间模型",
    "自监督学习的表示崩溃",
    "知识蒸馏的中间层匹配",
    "扩散模型的逆过程",
    "神经架构搜索的强化学习方法",
    "推荐系统的因果推断",
    "生成对抗网络的训练稳定性",
    "持续学习的灾难性遗忘",
]


def make_prompts(n_prompts: int, isl_tokens: int, tokenizer) -> list[str]:
    """Generate n_prompts each padded to ~isl_tokens tokens (after V4 chat
    template wrapping)."""
    prompts: list[str] = []
    for i in range(n_prompts):
        topic = _TOPICS[i % len(_TOPICS)]
        body = topic + "。" + _FILLER
        # Repeat filler until token count >= target.
        while len(tokenizer.encode(body)) < isl_tokens:
            body += _FILLER
        # Truncate to target token length.
        ids = tokenizer.encode(body)[:isl_tokens]
        prompts.append(tokenizer.decode(ids))
    return prompts


def main():
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--isl", type=int, default=1024, help="input length tokens")
    parser.add_argument("--osl", type=int, default=1024, help="output length tokens")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument(
        "--out",
        type=str,
        default="/workspace/ATOM-feat/logs/silicon_perf_bench.json",
    )
    args = parser.parse_args()

    n = args.num_prompts
    sizes, p = [], 1
    while p <= n:
        sizes.append(p)
        p *= 2
    args.cudagraph_capture_sizes = str(sizes)

    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    prompts_raw = make_prompts(n, args.isl, tokenizer)

    # Apply V4 chat template / encoding.
    enc_path = os.path.join(args.model, "encoding", "encoding_dsv4.py")
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
    else:
        prompts = prompts_raw

    actual_in_tokens = [len(tokenizer.encode(p)) for p in prompts]
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.osl
    )

    print(
        f"=== conc={n} ISL={args.isl} OSL={args.osl} "
        f"actual_in={actual_in_tokens[:4]}{'...' if len(actual_in_tokens) > 4 else ''} ==="
    )
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    t1 = time.time()
    elapsed = t1 - t0

    total_out_tokens = 0
    for o in outputs:
        # Output text → tokens (rough).
        try:
            ot = len(tokenizer.encode(o["text"], add_special_tokens=False))
        except Exception:
            ot = 0
        total_out_tokens += ot

    total_in = sum(actual_in_tokens)
    # Throughput definitions (decode-only is the InferenceX-style
    # metric since prefill is amortized).
    decode_tok_per_s = (n * args.osl) / elapsed if elapsed > 0 else 0.0
    decode_tok_per_s_per_seq = args.osl / elapsed if elapsed > 0 else 0.0

    result = {
        "conc": n,
        "isl_target": args.isl,
        "osl_target": args.osl,
        "actual_in_tokens": actual_in_tokens,
        "total_in_tokens": total_in,
        "total_out_tokens_approx": total_out_tokens,
        "wall_time_s": elapsed,
        "throughput_decode_tok_per_s_aggregate": decode_tok_per_s,
        "throughput_decode_tok_per_s_per_seq": decode_tok_per_s_per_seq,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
