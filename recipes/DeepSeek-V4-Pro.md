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

## Launching server (recommended: Triton MoE backend + Indexer FP8 storage)

### Single-request mode (max_num_seqs=1) — recommended production config

```bash
ATOM_USE_TRITON_MOE=1 \
ATOM_DSV4_INDEXER_FP8=1 \
ATOM_DSV4_USE_W4_PATH=1 USE_W4_PATH=1 \
AITER_LOG_LEVEL=WARNING \
  python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 1 --max-model-len 4096 --enforce-eager
```

The two opt-in env vars (`ATOM_USE_TRITON_MOE`, `ATOM_DSV4_INDEXER_FP8`) together close 30pp of the gsm8k gap vs the all-defaults config (see accuracy table below).

### Multi-request mode (development — UNSAFE flag required)

Multi-sequence batching of DSV4 is currently gated behind a development guard while the W4 KV pool finish-pipeline matures. To bypass for kernel-level perf experiments where output correctness is being measured separately:

```bash
ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 \
ATOM_DSV4_USE_W4_PATH=1 USE_W4_PATH=1 \
ATOM_USE_TRITON_MOE=1 \
ATOM_DSV4_INDEXER_FP8=1 \
  python -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-model-len 4096 --enforce-eager
```

### Why these env vars?

- **`ATOM_USE_TRITON_MOE=1`** — the default CK + FlyDSL fused MoE backend has known layout-dispatch issues with DSV4-Pro on gfx950 (see `EVIDENCE_M.md` Sprint 5b/5c). Without it, gsm8k flexible-extract caps at ~0.45 (strict-match 0.00).
- **`ATOM_DSV4_INDEXER_FP8=1`** — DeepSeek V4 paper §2.3.4 specifies the lightning indexer is performed in FP4 precision. ATOM's model already calls `fp4_act_quant_inplace` on the indexer KV before cache write, but the pool slab was allocated with the broader `kv_cache_dtype`, silently re-casting FP4 values wider on storage. This flag allocates the indexer slab in `float8_e4m3fn` (the closest practical FP4 proxy — torch lacks float4 cache writes) preserving FP4 magnitude granularity. Sprint 6 silicon-validated +15pp gsm8k 5-shot delta from this single flag.

Tracking issue: [sunway513/atom#37](https://github.com/sunway513/atom/issues/37).

### Available but NOT recommended: `ATOM_DSV4_KV_SPLIT_DTYPES=1`

Split main KV into nope (FP8) + rope (BF16) per paper §2.3.4. **Sprint 6 silicon validation showed no measurable accuracy benefit** (0.75 with B0a alone == 0.75 with B0a+B0b). Implementation is in `atom/engine/kv_pool/dsv4_pool.py:write_main_kv` for future revisits but is not part of the production recommendation. See Evidence M Sprint 6 B0b for the silicon trace.

## Accuracy baseline (gsm8k limit=20, num_fewshot=5, max_gen_toks=1024)

Verified on 8×MI355X TP=8, `--max-num-seqs 1`, `--enforce-eager`:

| Configuration | flexible-extract | strict-match | latency/req | gap to SGLang ref |
|---|---|---|---|---|
| CK/FlyDSL fused (all defaults) | 0.45 ± 0.114 | 0.00 ± 0.000 | 28s | 51pp |
| `ATOM_USE_TRITON_MOE=1` only | 0.60 ± 0.112 | 0.60 ± 0.112 | 36.82s | 36pp |
| **`ATOM_USE_TRITON_MOE=1` + `ATOM_DSV4_INDEXER_FP8=1`** | **0.75 ± 0.099** | **0.75 ± 0.099** | **34.23s** | **21pp** ✅ |
| SGLang on B300 (external reference) | 0.96 ± 0.020 | 0.96 ± 0.020 | (larger n) | — |

Sprint 6 net win: +15pp on both filters via the indexer flag alone. Cumulative since Sprint 4 closure: +30pp flexible / +75pp strict / -30pp gap to SGLang reference.

The remaining ~21pp gap is being investigated under Sprint 7 (Triton MoE kernel precision audit, larger-n eval to test small-sample noise, eval-config alignment with SGLang).

## Tips

- **Always set both `ATOM_USE_TRITON_MOE=1` and `ATOM_DSV4_INDEXER_FP8=1` for DSV4-Pro on MI355X.** Either alone leaves accuracy on the table.
- `ATOM_DSV4_USE_W4_PATH=1 USE_W4_PATH=1` enables the multi-request KV cache architecture. Without these, the model falls back to the legacy W3 path with no batching.
- Set `AITER_LOG_LEVEL=WARNING` before starting to suppress aiter kernel log noise.
- Clear compile cache before restarting after code changes: `rm -rf /root/.cache/atom/*`
- KV pool slot exhaustion (`DSV4KVPool: no free slot`) typically indicates a stale request from a prior crash — restart the server cleanly.

## Reference

- Tracking issue: [sunway513/atom#37](https://github.com/sunway513/atom/issues/37)
- Full Sprint 4/5 evidence and silicon traces: `docs/evidence/dsv4_w45/EVIDENCE_M.md`
- W4 path architecture (compressor pool, scheduler finish-pipeline): `atom/engine/kv_pool/dsv4_pool.py`, `atom/model_engine/scheduler.py`, `atom/model_runner.py`
- MoE backend dispatch: `atom/model_ops/moe.py:676` (`Mxfp4MoEMethod`), `:689` (Triton trigger via `ATOM_USE_TRITON_MOE`)
