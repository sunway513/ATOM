# DeepSeek-V4-Pro Usage Guide

[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) is the FP4-native MXFP4 mixture-of-experts model from DeepSeek (61 layers, hidden=7168, 384 routed experts + 1 shared, topk=6). Weights are stored as MXFP4 e2m1 with per-block ue8m0 scales (block size 32). Compared to DeepSeek-R1, V4-Pro adds:

- A multi-request KV cache (W4 path) backed by a per-ratio compressor pool, enabling batch decode at `--max-num-seqs > 1` once unblocked.
- A new sparse attention indexer for long-context efficiency.
- FP4 expert weights as the native checkpoint (no separate quantized variant).

ATOM provides built-in support on AMD MI355X (gfx950) silicon. The recommended MoE backend is the **Triton path** (see Sprint 5e of `docs/evidence/dsv4_w45/EVIDENCE_M.md` for the rationale).

## Preparing environment

Pull the latest docker from https://hub.docker.com/r/rocm/atom/ :
```bash
docker pull rocm/atom:latest
```
All operations below run inside the container.

The model weights can be pulled with:
```bash
bash recipes/pull_dsv4_pro_weights.sh
```

## Launching server (recommended: Triton MoE backend)

### Single-request mode (max_num_seqs=1) — full closure

```bash
ATOM_USE_TRITON_MOE=1 \
ATOM_DSV4_USE_W4_PATH=1 USE_W4_PATH=1 \
AITER_LOG_LEVEL=WARNING \
  python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 1 --max-model-len 4096 --enforce-eager
```

### Multi-request mode (development — UNSAFE flag required)

Multi-sequence batching of DSV4 is currently gated behind a development guard while the W4 KV pool finish-pipeline matures. To bypass for kernel-level perf experiments where output correctness is being measured separately:

```bash
ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 \
ATOM_DSV4_USE_W4_PATH=1 USE_W4_PATH=1 \
ATOM_USE_TRITON_MOE=1 \
  python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-model-len 4096 --enforce-eager
```

### Why Triton MoE?

The default CK + FlyDSL fused MoE backend has known layout-dispatch issues with DSV4-Pro on gfx950 — see `EVIDENCE_M.md` Sprint 5b/5c for the silicon evidence. Without `ATOM_USE_TRITON_MOE=1`, gsm8k flexible-extract caps at ~0.45 (strict-match 0.00) instead of 0.60+. Tracking issue: [sunway513/atom#37](https://github.com/sunway513/atom/issues/37).

## Accuracy baseline (gsm8k limit=20, num_fewshot=5, max_gen_toks=1024)

Verified on 8×MI355X TP=8, `--max-num-seqs 1`, `--enforce-eager`:

| Backend | flexible-extract | strict-match | latency/req |
|---|---|---|---|
| CK/FlyDSL fused (default) | 0.45 ± 0.114 | 0.00 ± 0.000 | 28s |
| **Triton MoE (`ATOM_USE_TRITON_MOE=1`)** | **0.60 ± 0.112** | **0.60 ± 0.112** | **36.82s** |
| SGLang on B300 (external reference) | 0.96 ± 0.020 | 0.96 ± 0.020 | (larger n) |

The remaining ~36pp gap to the SGLang reference is being investigated under follow-up Sprint 6 (Triton precision tuning + AITER-side FlyDSL layout audit). The Triton backend is the production-safe choice today.

## Tips

- **Always set `ATOM_USE_TRITON_MOE=1` for DSV4-Pro on MI355X.** Without it, accuracy regresses significantly (see table above).
- `ATOM_DSV4_USE_W4_PATH=1 USE_W4_PATH=1` enables the multi-request KV cache architecture. Without these, the model falls back to the legacy W3 path with no batching.
- Set `AITER_LOG_LEVEL=WARNING` before starting to suppress aiter kernel log noise.
- Clear compile cache before restarting after code changes: `rm -rf /root/.cache/atom/*`
- KV pool slot exhaustion (`DSV4KVPool: no free slot`) typically indicates a stale request from a prior crash — restart the server cleanly.

## Reference

- Tracking issue: [sunway513/atom#37](https://github.com/sunway513/atom/issues/37)
- Full Sprint 4/5 evidence and silicon traces: `docs/evidence/dsv4_w45/EVIDENCE_M.md`
- W4 path architecture (compressor pool, scheduler finish-pipeline): `atom/engine/kv_pool/dsv4_pool.py`, `atom/model_engine/scheduler.py`, `atom/model_runner.py`
- MoE backend dispatch: `atom/model_ops/moe.py:676` (`Mxfp4MoEMethod`), `:689` (Triton trigger via `ATOM_USE_TRITON_MOE`)
