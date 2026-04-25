# Evidence H — W3.2-v6.1 silicon retry (closed Path 1)

**Status**: Path 1 closed per [issue #37](https://github.com/sunway513/ATOM/issues/37) decision. This evidence is the final time-boxed retry confirming the patch-spiral exhausted itself; v6.1 code is archived on this branch (`feat/dsv4-v6-experimental`, commit `d5a7d79`) and is **not** for merge.

## Run config

- Branch / commit: `feat/dsv4-v6-experimental` @ `d5a7d79` (working tree)
- Container: `atom_dsv4_feat`
- Date: 2026-04-25
- Hardware: MI355X TP=8
- Test harness: `tests/silicon/silicon_fact_multireq.py`
- Args: `--num-prompts 4 --max-tokens 32 --max-num-seqs 4 --max-model-len 2048 --enforce-eager --gpu-memory-utilization 0.85`
- Wall time: 180s (rc=0)

## What was being tested

v6.1 = v6 (positions:Tensor threading + per-token RoPE + per-token KV scatter) + topk K-dim split (prefill uses scalar helper sized to seqlen, decode uses per-token helper sized to win=128). The split was meant to fix v6's regression where prefill's K-dim mismatch caused sparse_attn to read garbage.

## Result

| idx | prompt | output | status |
|---|---|---|---|
| 0 | 如何在一个月内增肌10公斤 | "好的，作为一个在健身领域深耕多年的爱好者..." | ✓ correct (idx=0 restored after v6 regression) |
| 1 | Briefly describe Beijing in 3 sentences. | `北京#  \|  \|  \|  \| ...` | ✗ degenerate (same as v3) |
| 2 | Write a Python function to compute the nth Fibonacci number. | "```\n    - 1. 1. 1. 1. 1. 1. 1." | ✗ degenerate (same as v3) |
| 3 | List 5 common machine learning algorithms. | (same pattern as v3) | ✗ degenerate |

Full token IDs and model logs live in `silicon_w32_v61.json` and `silicon_w32_v61.log` next to this file.

## Interpretation

- The topk K-dim split fix RESTORED idx=0 (which v6 had broken).
- idx=1/2/3 remain degenerate with the same signature as v3 — the topk fix only addressed one of the predicted "next walls"; the others (Compressor scalar `start_pos` for ring slot, Indexer compress-boundary scalar, Compressor decode RoPE scalar) remain unfixed.
- Path 1 (incremental patching of lingpeng's PR1 single-request skeleton) is empirically exhausted: each fix unmasks the next wall rather than getting closer to closure.

## Decision (recorded in #37)

- **Path 1 closed.** No more iterations on this branch.
- **Path 2** (PR #38): hard `max_num_seqs=1` guard with `ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1` opt-in override.
- **Path 3** (PR #39 W4.1, branch `feat/dsv4-forward-batch-paged-kv`): SGLang/vLLM-isomorphic refactor (positions:Tensor + engine-owned KV pools); proceeds as W4.1-W4.6 sub-PRs.

## Files in this directory

| File | Purpose |
|---|---|
| `EVIDENCE_H_v61.md` | This document |
| `silicon_w32_v61.json` | 4-prompt structured output: prompt + completion + first-32 token_ids per idx |
| `silicon_w32_v61.log` | Full server log (model load, prefill/decode timings, etc.) |
