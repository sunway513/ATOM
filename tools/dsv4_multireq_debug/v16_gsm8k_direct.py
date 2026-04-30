"""Direct gsm8k limit=20 5-shot at conc=4 against the v16 PR#650 server.

Bypasses lm_eval (which crashes with UnboundLocalError on retry path
when server returns any 4xx). Uses datasets to load gsm8k, builds the
standard 5-shot prompt, fires conc=4 parallel curls, grades by regex
extraction matching lm_eval's flexible-extract.
"""
import asyncio
import json
import re
import sys
import time

import aiohttp
from datasets import load_dataset

MODEL = "/data/hf_models/deepseek-ai/DeepSeek-V4-Pro"
URL = "http://localhost:8000/v1/completions"
LIMIT = 20
NSHOT = 5
MAX_TOKENS = 512
CONC = 4

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def gold(text: str) -> str:
    m = ANS_RE.search(text)
    return m.group(1).replace(",", "").strip() if m else None


# Flexible extract: any number after the model's answer
FLEX_RE = re.compile(r"(-?[\d,]+\.?\d*)")


def extract_pred(text: str) -> str:
    nums = FLEX_RE.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").strip()


async def call(session, prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stop": ["Question:"],
    }
    async with session.post(URL, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as r:
        d = await r.json()
        return d["choices"][0]["text"]


async def main():
    ds_train = load_dataset("gsm8k", "main", split="train")
    ds_test = load_dataset("gsm8k", "main", split="test")
    fewshot = ds_train.shuffle(seed=42).select(range(NSHOT))
    prefix = ""
    for ex in fewshot:
        prefix += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"

    items = ds_test.select(range(LIMIT))
    prompts = [prefix + f"Question: {ex['question']}\nAnswer:" for ex in items]
    golds = [gold(ex["answer"]) for ex in items]

    sem = asyncio.Semaphore(CONC)

    async def run_one(idx, prompt):
        async with sem:
            try:
                resp = await call(session, prompt)
                return idx, resp, None
            except Exception as e:
                return idx, None, repr(e)

    async with aiohttp.ClientSession() as session:
        t0 = time.time()
        tasks = [run_one(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        dt = time.time() - t0

    results.sort()
    flex = strict = total = 0
    for idx, resp, err in results:
        if err:
            print(f"  [{idx:02d}] ERROR: {err}")
            total += 1
            continue
        pred = extract_pred(resp or "")
        g = golds[idx]
        ok_flex = pred == g
        ok_strict = bool(re.search(rf"####\s*{re.escape(g)}", resp or ""))
        flex += int(ok_flex)
        strict += int(ok_strict)
        total += 1
        marker = "✓" if ok_flex else "✗"
        print(f"  [{idx:02d}] gold={g} pred={pred} {marker} | {(resp or '').strip()[:120]}")

    print()
    print(f"=== gsm8k limit={LIMIT} 5-shot conc={CONC} ===")
    print(f"flexible-extract: {flex}/{total} = {flex/total:.4f}")
    print(f"strict-match    : {strict}/{total} = {strict/total:.4f}")
    print(f"latency         : {dt:.1f}s ({dt/total:.1f}s/req avg)")


if __name__ == "__main__":
    asyncio.run(main())
