# Silicon Evidence — DSV4 KV Cache Reform Bug Verification

| Field | Value |
|---|---|
| Date | 2026-04-25 |
| Hardware | mi355-gpu-15 · 8× AMD Instinct MI355X |
| Container | `rocm/atom-dev:latest` (43.9 GB) |
| ATOM branch | `lingpeng/dsv4-pr1-skeleton` (= ROCm/ATOM PR [#650](https://github.com/ROCm/ATOM/pull/650)) @ `cdbff359` |
| aiter branch | `lingpeng/fix-mhc-device` (= ROCm/aiter PR [#2916](https://github.com/ROCm/aiter/pull/2916)) @ `76ea1ed5` |
| Model | `deepseek-ai/DeepSeek-V4-Pro` (1.6T total / 49B active, FP4 native MXFP4 expert weights) — locally at `/data/hf_models/deepseek-ai/DeepSeek-V4-Pro` |
| triton_kernels | `triton-lang/triton@v3.5.1` `python/triton_kernels` (installed editable) |
| Command | PR#650 canonical (TP=8, FP8 KV, `ATOM_USE_TRITON_MOE=1`, `--enforce-eager`, `--temperature 0.0`, `--max-tokens 32`) |

## Purpose

Empirically verify two facts before correctness implementation begins:

1. **PASS @ conc=1**: lingpeng's stack produces a coherent, meaningful single-query response (validates PR#650 single-seq claim on real silicon).
2. **FAIL @ conc>1**: the multi-request bug surface documented in [RFC v0.2.6 §3.1](2026-04-25-dsv4-kvcache-reform.md) is observable with the exact line-level error site predicted.

Both facts establish the BEFORE-state for the correctness PR. Post-reform, conc>1 must succeed on the same prompts with token-ID equality vs conc=1 (per RFC §9.5.4 `test_dsv4_greedy_seq_vs_batch`).

---

## Evidence A — conc=1 PASS

### Command
```bash
ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING \
  python -m atom.examples.simple_inference \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-model-len 1024 --enforce-eager \
    --temperature 0.0 --max-tokens 32
```

### Result (from `silicon_fact_short.log`)
```
Prompt:     如何在一个月内增肌10公斤
Completion: 好的，这是一个非常具体且极具挑战性的目标，
            在一个月内增肌10公斤（即纯肌肉组织的增长，
            而非水分或脂肪的增加）。从
```

**Translation**: "Okay, this is a very specific and extremely challenging goal — gaining 10kg of muscle in a month (i.e., pure muscle tissue growth, not water or fat increase). From..."

**Verdict**: ✅ Coherent, on-topic, grammatically correct Chinese. Single-query path works.

---

## Evidence B — conc=4 HARD CRASH

### Command
4 distinct prompts batched in a single `llm.generate()` call (lockstep):

```python
HERO     = "如何在一个月内增肌10公斤"          # idx 0 — same as conc=1 baseline
SECONDARY = [
    "Briefly describe Beijing in 3 sentences.",
    "Write a Python function to compute the nth Fibonacci number.",
    "List 5 common machine learning algorithms.",
]
prompts = [HERO] + SECONDARY                  # n_prompts=4
```

Same engine args as conc=1 (TP=8, FP8 KV, `--temperature 0.0`, `--max-tokens 32`).

### Result (from `silicon_fact_conc4.log`)

**Hard runtime error** (not garble — the run terminates):

```
RuntimeError: The expanded size of the tensor (1) must match the existing size (4)
              at non-singleton dimension 0.
              Target sizes: [1, 512].  Tensor sizes: [4, 512]

File "/workspace/ATOM-lingpeng/atom/models/deepseek_v4.py", line 1054, in forward
    self.kv_cache[:1, start_pos % win] = kv.squeeze(1)
```

### Diagnosis — direct hit on RFC §3.1 bug source #6

RFC v0.2.6 §3.1 row "bug source #6" lists exactly five `[:1]` hardcode sites:
`deepseek_v4.py:1040, 1044, 1045, 1054, 1059`.

The crash hits **`:1054`** — one of those five sites. The error message *literally* describes the bug:
- **Target sizes `[1, 512]`** = the hardcoded `[:1, ...]` slice expects a singleton batch axis
- **Tensor sizes `[4, 512]`** = the actual `kv` arriving from the attention KV-projection has B=4 (one row per concurrent request)
- The `[1, ...] = [4, ...]` assignment crashes the broadcast.

### Verdict: ✅ Bug surface confirmed in physical silicon

This is **stronger** than PR#650's description ("doesn't crash, just garbles") — under our test conditions it produces a deterministic crash, making the bug detectable by any CI gate, not just by output-quality eyeballing.

---

## Implications for the implementation PR

1. **Removing the 5 `[:1]` hardcodes (1040/1044/1045/1054/1059) and bug source #5's reset orchestration (994-1002) is necessary AND sufficient to clear the immediate crash.** Per RFC §6.2.1 the slot-mapping rewire replaces these with paged writes.
2. **`test_dsv4_no_module_state.py` (RFC §9.5.5)** must AST-grep for `[:1, ...]` hardcodes; this evidence shows the AST audit is non-negotiable — a single missed site re-introduces the crash.
3. **`test_dsv4_greedy_seq_vs_batch` (RFC §9.5.4)** acceptance: same hero prompt at conc=4 must produce **token-ID equality** with the conc=1 baseline transcript captured in Evidence A above.
4. **The crash signature itself becomes a regression sentinel**: any future PR that reintroduces a `[:1, ...]` write to module state will immediately reproduce this exact error.

---

## Reproduction — minimal commands

```bash
# 1. weights (one-time, ~600 GB)
bash recipes/pull_dsv4_pro_weights.sh

# 2. persistent dev container
docker run -d --name atom_dsv4 \
  --device=/dev/kfd --device=/dev/dri --group-add video --network host \
  --ipc=host --shm-size 16G \
  -v $(realpath ../ATOM-lingpeng):/workspace/ATOM-lingpeng \
  -v $(realpath ../aiter-lingpeng):/workspace/aiter-lingpeng \
  -v /data/hf_models:/data/hf_models:ro \
  rocm/atom-dev:latest sleep infinity

# 3. install lingpeng stack + triton_kernels (one-time)
docker exec atom_dsv4 bash -c "
  cd /tmp && git clone --depth 1 -b v3.5.1 https://github.com/triton-lang/triton.git triton-src
  cd triton-src/python/triton_kernels && /opt/venv/bin/pip install -e . --no-deps
  cd /workspace/aiter-lingpeng && /opt/venv/bin/pip install -e . --no-deps --no-build-isolation
  cd /workspace/ATOM-lingpeng && /opt/venv/bin/pip install -e . --no-deps
"

# 4. conc=1 PASS reproduction (~3 min cold, ~1 min warm)
docker exec atom_dsv4 bash -c "
  ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.examples.simple_inference \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-model-len 1024 --enforce-eager \
    --temperature 0.0 --max-tokens 32
"

# 5. conc=4 CRASH reproduction (using silicon_fact_multireq.py — committed alongside this evidence)
docker exec atom_dsv4 bash -c "
  ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python /workspace/silicon_fact_multireq.py \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-model-len 1024 --enforce-eager \
    --temperature 0.0 --max-tokens 32 --num-prompts 4
"
```

---

## Cross-references

- RFC: [`docs/rfcs/2026-04-25-dsv4-kvcache-reform.md`](2026-04-25-dsv4-kvcache-reform.md) — see §3.1 bug surface table, §9.5.4 acceptance tests
- Tracking issue: [sunway513/ATOM#35](https://github.com/sunway513/ATOM/issues/35)
- Recipe: [`recipes/pull_dsv4_pro_weights.sh`](../../recipes/pull_dsv4_pro_weights.sh)
- Multireq evidence script: [`tests/silicon/silicon_fact_multireq.py`](../../tests/silicon/silicon_fact_multireq.py) (added with this commit)

---

## Evidence C — W3.1 silicon retry signature progression (multi-iteration)

> v0.2: appended after Week 3.1 (clear all shape-related model-side hardcodes
> + B=1-implicit squeeze patterns + topk helper M dim). All evidence captured
> on the same `mi355-gpu-15` + `atom_dsv4_feat` container.

### Iteration table

| Iter | Patch site count | Crash signature | RFC §3.1 bug source matched |
|---|---|---|---|
| BEFORE (lingpeng raw) | — | `RuntimeError @ deepseek_v4.py:1054`<br>`Target [1, 512] vs Tensor [4, 512]` | #6 (`DeepseekV4Attention.kv_cache` `[:1]` 5 sites) |
| W3.1-v3 | 6 `[:1]` cleared in Attention + Indexer | `RuntimeError @ deepseek_v4.py:597`<br>same `Target [1, 512] vs Tensor [4, 512]` | #1 (`Compressor.kv_state` / `score_state` write under B=1-implicit) |
| W3.1-v5 | 4 `kv.squeeze(1)` → `kv.squeeze(0)` + dim-1 batch on Compressor (3 decode write sites + post-write) | `AssertionError @ sparse_attn_v4.py:67`<br>`assert topk_idxs.shape == (B, M, K)` (M=1 vs M=4 mismatch) | #7 (scalar-position helpers `_get_window_topk_idxs`/`_get_compress_topk_idxs` `bsz=1` + 1D-matrix expand pattern) |
| **W3.1-v6** | 2 topk helpers tile 1D matrix to `[seqlen, K]` so M expands per-token | **`IndexError @ scheduler.py:609`** `prev_token_ids[idx]` out of range — **forward pass succeeded, crash moved to scheduler postprocess** | (scheduler / forward_output mapping — W3.2 territory; outside W3.1 model-shape scope) |

### Verdict

**W3.1 model-side shape bug surface is CLEAR.** The forward pass on conc=4
no longer crashes from any shape mismatch in `atom/models/deepseek_v4.py`,
`atom/model_ops/sparse_attn_v4.py`, or the topk index helpers. The remaining
`IndexError` lives in the scheduler's per-request output mapping
(`atom/model_engine/scheduler.py` + `model_runner.py`'s `fwd_output.get_idx`
contract), which is the next reform target per RFC §6.2.1 / §14 Week 3.2
(slot-mapping integration end-to-end).

This is exactly the layered signature progression RFC §3.1 predicted: each
fix unmasks the next bug source until the entire chain is corrected. The
chain matched the RFC's stated bug sources #6 → #1 → #7 → (W3.2 territory)
in order.

### What v6 changed in code (cumulative since lingpeng)

`atom/models/deepseek_v4.py`:
- Indexer einsum read (was `[:1]` line 740): `kv_cache[:q.shape[0], :end_pos//ratio]`
- DeepseekV4Attention prefill writes (lines 1040, 1044, 1045): `kv_cache[:bsz_kv]`
- DeepseekV4Attention decode write + read (lines 1054, 1059): `kv_cache[:kv_decode.shape[0]]` / `kv_cache[:q.shape[0]]`
- DeepseekV4Attention decode kv shape (was `kv.squeeze(1)` assuming S=1): `kv.squeeze(0)` + `kv.shape[1]`
- Compressor decode write — overlap branch (was `kv.squeeze(1)` × 2 + bsz indexing × 4): `kv.squeeze(0)` + `batch_decode = kv.shape[1]`
- Compressor decode write — non-overlap branch (was `kv.squeeze(1)` × 2 + bsz indexing × 2): same fix
- Compressor post-write to externally-set kv_cache (line 638): `kv_cache[:batch_decode]`
- `_get_window_topk_idxs` (start_pos > 0 branches): tile 1D matrix to `[seqlen, K]` before unsqueeze+expand → M dim now matches query token count
- `_get_compress_topk_idxs` (start_pos > 0 branch): same tile-to-seqlen fix

Total 14 patch sites in `deepseek_v4.py`, all annotated with `# W3.1 (RFC ...)` comments for traceability.

### Reproduction
- conc=1 baseline (Evidence A): unchanged, still produces coherent Chinese output.
- conc=4 W3.1-v6 retry: forward succeeds; scheduler crash captured in
  `/home/pensun/ATOM/logs/silicon_w31_v6.log`.
- Recipe: `bash /workspace/ATOM-feat/tests/silicon/silicon_fact_multireq.py --num-prompts 4` after `pip install -e` (or `PYTHONPATH=/workspace/ATOM-feat`) inside `atom_dsv4_feat`.
