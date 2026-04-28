# DSV4-Pro Multi-Request Debug Toolkit (issue #37 Sprint 7h)

Tools for bisecting why DeepSeek-V4-Pro produces degenerate output at
`--max-num-seqs >= 2` while conc=1 works.

## Files

| File | Purpose |
|---|---|
| `repro_gemm_a4w4_m_dep.py` | Standalone repro for AITER `gemm_a4w4` batch-dependence. **PASSES on random data** — useful as a NEGATIVE control to rule out kernel m-padding bug. |
| `diff_qstage.py` | Compare per-stage Q dumps from two trace runs (rank-matched, same-position-matched). |

## How to use

### 1. Verify atom package origin (avoid the 4-hour rathole)

```bash
sudo docker exec atom_dsv4_feat /opt/venv/bin/python -c \
    "import atom; print(atom.__file__)"
```

Must print `/workspace/ATOM-lingpeng/atom/__init__.py`. If not:

```bash
sudo docker exec atom_dsv4_feat /opt/venv/bin/pip install -e \
    /workspace/ATOM-lingpeng --no-deps --no-build-isolation
```

### 2. Capture per-stage Q dumps at conc=1 (FORCE_W4) and conc=2

The hook in `atom/models/deepseek_v4.py:DeepseekV4Attention._forward_w4`
fires when `ATOM_DSV4_TRACE_QSTAGE_DIR` is set. It also reads:

- `ATOM_DSV4_FORCE_W4=1` — bypass the single-seq fast-path so conc=1
  exercises the same `_forward_w4` code as conc=2.
- `ATOM_DSV4_TRACE_MAX_STEPS=1` — number of decode steps to dump per layer.

Files are named `L{layer:02d}_S{step:02d}_R{rank:02d}.pt` with a TP rank
suffix to avoid the cross-rank file race that contaminated my earlier
diffs.

### 3. Diff with rank- and position-matching

```bash
python tools/dsv4_multireq_debug/diff_qstage.py \
    traces_conc1 traces_conc2 \
    --layer 0 --target-pos 59
```

The `--target-pos` flag selects the row whose absolute position you want
to compare across runs (so seq-A in conc=1 is matched to seq-A in
conc=2 even when the multi-seq batch admits the prompts in a different
order).

### 4. Run the negative control

```bash
sudo docker exec -w /workspace/ATOM-lingpeng atom_dsv4_feat \
    /opt/venv/bin/python /workspace/ATOM-lingpeng/tools/dsv4_multireq_debug/repro_gemm_a4w4_m_dep.py
```

If this PASSES (max diff = 0.000) but the model trace shows wq_b
divergence, the bug is NOT in the kernel — look at the wrapper, quant
scale handling, or upstream.

## What I (the prior agent) tried and reverted

- Sprint 7d (`2b0627c` reverted by `8234a78`): per-seq slot lookup in
  Indexer. Did not fix multi-req.
- Sprint 7e (`b8020d5` reverted by `00d141f`): per-seq Indexer loop
  mirroring SGLang. Did not fix multi-req.
- Sprint 7g (uncommitted): zero-pad x to MXFP4 block size before
  `gemm_a4w4`. Verbose probe confirmed kernel called but output unchanged.

## What landed and is silicon-proven

| Commit | Sprint | Effect at conc=1 |
|---|---|---|
| `2d0098c` | 7b cast fix | +10pp |
| `96176c4` | 7c admit-gate | unblocks no-free-slot race |
| `a8e3a02` | 6 B0a Indexer FP8 | +15pp |

## Honest open question

I burned ~5 hours on multi-req without finding the root cause. Inputs
to `wq_b` for the same logical row are bit-identical between conc=1
and conc=2 (verified, max diff 0.000). After `wq_b` they appear to
diverge — but my prior diff was contaminated by the TP rank race.
The rank-suffixed trace + `diff_qstage.py` here gives a clean way
to re-test that hypothesis. If wq_b is in fact rank-clean, the bug
is downstream (Compressor state, KV pool slot, MoE batch, ...).
