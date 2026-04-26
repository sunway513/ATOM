# Evidence K — W4.5 Silicon Root Cause (issue sunway513/atom#37)

**Date**: 2026-04-26
**Hardware**: MI355 TP=8 (atom_dsv4_feat container, /opt/venv)
**Branch**: `feat/dsv4-w4-kernel-bisect` (HEAD `d528147`, off main `5e193de`)

## TL;DR — Evidence J's "AITER kernel bug" attribution was wrong

Sprint 1 closed (Evidence J) on the assumption that `HSA_STATUS_ERROR_EXCEPTION 0x1016` was an AITER-internal kernel ABI violation requiring AITER-team work. **It is not.** The faulting kernel is PyTorch's stock `at::native::index_put_kernel_impl<OpaqueType<4>>` — a GPU-side `s_trap 2` (`ASSERT_TRAP`) bounds-check on per-token scatter into the W4 Compressor pool. The bug is entirely ATOM-side.

This evidence captures:
1. The exact ROCr Debug Agent dump that pinpoints the kernel.
2. Two layered ATOM bugs the dump revealed.
3. Bug #1 fixed in this PR; Bug #2 + #3 deferred to Sprint 2 with explicit context.

## Evidence #1: ROCr Debug Agent dump — kernel identified

```
HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 \
HIP_LAUNCH_BLOCKING=1 \
HSA_ENABLE_QUEUE_FAULT_MESSAGE=1 HSA_ENABLE_VM_FAULT_MESSAGE=1 \
HSA_COREDUMP_PATTERN=... \
ATOM_DSV4_USE_W4_PATH=1 ATOM_DSV4_UNSAFE_MULTIREQ_DEV=1 \
ATOM_AITER_VALIDATE=1 \
python -m tests.silicon.silicon_w43_smoke --mode single ...
```

ROCr's stderr at `[atom 05:04:25] Scheduled prefill batch: 1 reqs, 12 tokens`:

```
wave_8: pc=0x731fd50599c8 (kernel_code_entry=0x731fd5058700
  <void at::native::index_elementwise_kernel<128, 4,
    at::native::gpu_index_kernel<
      at::native::index_put_kernel_impl<at::native::OpaqueType<4>>(
        at::TensorIterator&,
        c10::ArrayRef<long>,
        c10::ArrayRef<long>
      )::{lambda(char*, char const*, long)#1}
    >(...)
  >>>>) (stopped, reason: ASSERT_TRAP)

=> 0x731fd50599c8 <+4808>: s_or_b64 exec, exec, s[0:1]

HSA_STATUS_ERROR_EXCEPTION code: 0x1016
```

Interpretation: PyTorch compiles `index_put` with a `s_trap 2` device-side check that fires when an index value exceeds the indexed dim's size. The 0x1016 HSA exception is just the runtime translation of that `s_trap`. This is NOT an AITER kernel — `at::native::index_put_kernel_impl` is a stock PyTorch elementwise kernel.

## Evidence #2: Diagnostic asserts pinpoint the offending scatter

Added Python-boundary asserts before the two W4-path scatter sites
(`atom/models/deepseek_v4.py`):

```python
# DSV4Attention._forward_w4 (line ~1804)
kv_flat[out_cache_loc] = kv_tok.to(kv_flat.dtype)        # main KV scatter

# DSV4Compressor._forward_w4 (line ~942/943)
self.kv_state[slot_per_token, row_in_state] = kv.to(...)
self.score_state[slot_per_token, row_in_state] = score_with_ape.to(...)
```

Re-running silicon with the asserts produced:

```
ValueError: W4 Compressor kv_state scatter OOB:
  slot_per_token.max=0 (cap=1),
  row_in_state range=[0, 11] (cap=8),
  kv_state.shape=(1, 8, 512), ratio=128, overlap=False,
  positions.range=[0, 11], slot_indices=[0], cu_seqlens_q=[0, 12]
```

So the offending index is `row_in_state` against `kv_state.shape[1]=8`. For `ratio=128, overlap=False`, the Compressor formula `row_in_state = positions % ratio` produces values in `[0, 127]`, but the slab only has 8 rows.

## Bug #1 (FIXED in this PR): pool slab ring too small for c128 layers

`atom/model_engine/model_runner.py:1864`:

```python
# Before
ring_size_compressor = 2 * 4  # = 8

# After
ring_size_compressor = max(2 * 4, 128)  # = 128
```

The comment claimed "mirror SGLang's `get_compress_state_ring_size` (4→8, 128→128)" but the implementation collapsed SGLang's PER-LAYER ring sizes to a single value taking the c4 size. A c128 layer's Compressor (overlap=False, ratio=128) needs `ring=128`; getting 8 traps on the very first prefill row.

After Bug #1 fix, c128 scatter works (silicon clears the original trap). 16x memory waste on c4 layers' compressor slab, accepted tradeoff for getting silicon green; proper per-slab split is Sprint 2.

## Bug #2 (NEW, Sprint 2): pool slab inner_dim mismatch between c4 and c128

After Bug #1 fix, silicon v5 produced:

```
RuntimeError: shape mismatch: value tensor of shape [12, 256]
  cannot be broadcast to indexing result of shape [12, 512]
```

Compressor's `_state_inner_dim = coff * head_dim`:
- c4 (Indexer's inner Compressor, `head_dim=index_head_dim=128`, overlap=True, coff=2): inner = `2 * 128 = 256`
- c128 (main Compressor, `head_dim=512`, overlap=False, coff=1): inner = `1 * 512 = 512`

Pool's `state_inner_dim` is a single value (= main attention's `head_dim` = 512). When the Indexer's c4 Compressor writes into the slab, it produces `[T, 256]` but the slab expects `[T, 512]` — shape mismatch RuntimeError.

**Why Bug #1 fix doesn't address this**: extending ring_size doesn't help if the per-row inner dim is wrong. PyTorch index_put requires exact shape match between value tensor and indexing result.

**Proper fix (Sprint 2)**: split compressor pool into two slabs — `_compressor_state_c4` of shape `[num_c4_layers, N, 8, 256]` and `_compressor_state_c128` of shape `[num_c128_layers, N, 128, 512]`. SGLang's design has per-layer slabs; ATOM's pool needs to mirror that.

## Bug #3 (NEW, Sprint 2): per-layer ring sizing architectural debt

Bug #1 + Bug #2 are both manifestations of the same architectural decision: ATOM's `DSV4KVPool` collapses per-layer ring/inner sizes into a single value. SGLang's `get_compress_state_ring_size(compress_ratio)` returns different sizes per ratio. Until ATOM's pool replicates per-layer slabbing, every per-ratio mismatch will surface as a silicon trap or shape error.

## Diagnosis: why Evidence J's validator gap

Evidence J flagged that the AITER `dsv4_validate_sparse_attn_metadata` validator (sunway513/aiter#60) "did NOT raise" before the crash. Two reasons (now understood):

1. **Wrong call site**: validator is invoked AT `_forward_w4`'s `sparse_attn` boundary (line ~1865), but the crash happens EARLIER at the Compressor's `_forward_w4` scatter inside `self.indexer(...)` / `self.compressor(...)` (called at lines ~1822/1840). Python flow never reaches the sparse_attn validator.

2. **Wrong `pool_capacity` argument**: even if the flow reached the validator, the call passed `pool_capacity=n_slots` where `out_cache_loc` ranges in `[0, n_slots * ring_main)`. Now fixed to `n_slots * ring_main` in the same commit.

The `dsv4_validate` ABI contract (sunway513/aiter#60) is correct as a `sparse_attn` ABI gate; it just doesn't (and shouldn't) cover the Compressor pool scatter, which is an internal ATOM concern.

## Files in this directory

| File | Purpose |
|---|---|
| `EVIDENCE_K.md` | This document |
| `silicon_w45_rocr_excerpt.log` | ROCr Debug Agent crash dump excerpt (~95 KB, filtered from 1.3 MB raw to remove `hipGetLastError` noise; kernel name + ASSERT_TRAP preserved) |

## What this PR delivers (`feat/dsv4-w4-kernel-bisect` HEAD `d528147`)

| Layer | Change |
|---|---|
| `atom/models/deepseek_v4.py` | 2 scatter asserts (DSV4Attention main + DSV4Compressor); validator `pool_capacity` fix |
| `atom/model_engine/model_runner.py` | `ring_size_compressor`: 8 → 128 with comment explaining SGLang per-layer mismatch |
| `tests/silicon/repro_w4_kernel_bisect.py` | (already in PR #54 commits `d5ea0ca` + `9209f21`) Python monkey-patch bisect harness; superseded by ROCr Debug Agent but kept for record |
| `docs/evidence/dsv4_w45/EVIDENCE_K.md` | This document, replacing the AITER-kernel-bug attribution from Evidence J |

## What this PR does NOT deliver

- W4.5 owner accuracy gate (gsm8k limit=20 conc=4 ≥ 60%) — blocked on Bug #2/#3 (pool slab split + per-layer ring/inner sizing).
- AITER-side reproducer — **not needed**: the bug is not in AITER. Sprint 1's plan to write an AITER standalone reproducer is canceled.

## Sprint 2 progress (commit `d53c103`)

Sprint 2 split the unified Sprint-1 compressor pool into per-ratio slabs and fixed two more bugs that surfaced during silicon iteration:

| # | Bug | Status |
|---|-----|--------|
| 1 | `ring_size_compressor` collapsed PER-LAYER SGLang sizes into a single value (c128 row OOB) | ✅ Fixed in `d528147` |
| 2 | `state_inner_dim` shared between c4 (256) and c128 (512) — shape mismatch | ✅ Fixed in `d53c103` (per-ratio slabs) |
| 3 | Pool allocator architecture: ONE slab for both ratios — fundamentally wrong | ✅ Fixed in `d53c103` (split into `_compressor_state_c4` + `_compressor_state_c128` with separate index maps) |
| 4 | `_apply_rotary_emb` crashed on 2D `[B, D]` input (compressor compress-boundary emit) | ✅ Fixed in `d53c103` |
| 5 | `_indexer_kv` slab last-dim was `head_dim=512` but Indexer needs `index_head_dim=128` | ✅ Fixed in `d53c103` (added `cfg.index_head_dim`) |
| 6 | `kv_per_token` doesn't concat compressed KV but `topk_idxs` includes compressed positions → AITER validator catches `topk_idxs.max=130 >= kv.size(N)=128` | ⏸️ Deferred — W4 attention logic, not pool sizing. Separate PR. |

All 61 pre-existing UTs pass unchanged on `d53c103`. Backward-compat shim keeps Sprint-1 single-value config fields (`ring_size_compressor`, `state_inner_dim`) working.

## Silicon iteration log (single mode, 12-token prefill, max-tokens=32)

| Run | Action | Result |
|-----|--------|--------|
| v1 (Evidence J) | Sprint 1 main | HSA 0x1016, kernel UNKNOWN |
| v2 | + ROCr Debug Agent | HSA 0x1016, kernel = `at::native::index_put_kernel_impl` ASSERT_TRAP |
| v3 | + Python scatter asserts | `ValueError: W4 Compressor kv_state scatter OOB row=11/cap=8` (Bug #1) |
| v4 | + ring_size_compressor 8→128 | `RuntimeError: shape [12,256] vs [12,512]` (Bug #2 surfaced) |
| v5 | (same as v4 — confirmed Bug #1 fixed) | (same Bug #2) |
| v6 | + per-ratio pool slabs (Sprint 2) | `RuntimeError: shape '[1,32,1,32]' invalid for size 32` — RoPE 2D (Bug #4) |
| v7 | + RoPE 2D fix | `RuntimeError: expanded size (512) must match (128)` — Indexer head_dim (Bug #5) |
| v8 | + Indexer head_dim fix | AITER validator: `topk_idxs.max=130 >= kv.size(N)=128` (Bug #6 — DEFERRED) |

Each iteration **proves the prior fix works** (no regression of earlier symptom) before exposing the next layer.

## Bug #6 — what's left for the next PR

`atom/models/deepseek_v4.py:1885` does:

```python
topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)  # [1, T, win + compress_K]
...
kv_per_token = kv_cache_view[slot_per_token]  # [T, ring_main=128, D]   ← only main, missing compressed
...
o = sparse_attn(q_per_token, kv_per_token, ..., topk_per_token, ...)
```

The `compress_topk_idxs` references compressed-cache positions `[128, 128+max_K)`, but `kv_per_token` is sized for the main ring only. The fix requires:

1. Gather per-seq compressed KV from the indexer/compressor pool slab (already fed by Sprint 2).
2. Concatenate to `kv_per_token` along dim=1 so it becomes `[T, ring_main + max_compress, D]`.
3. Adjust `compress_topk_idxs` to be relative to the concatenated boundary (i.e., `compress_topk_idxs += ring_main`).

This is W4 attention logic, not pool sizing. Tracked for the next PR (`atom/models/deepseek_v4.py` change only — pool architecture is settled).

## Remaining Sprint 2+ work

- [ ] Bug #6: concatenate per-seq compressed KV into `kv_per_token` (W4 attention logic).
- [ ] silicon `--mode single` reaches decode + ALL DONE.
- [ ] silicon `--mode multi` reaches decode + ALL DONE.
- [ ] gsm8k accuracy gate post-silicon-green (W4.5 owner sign-off).
- [ ] Per-ratio UT coverage (Sprint-2 slab split is implicitly tested via existing UTs but explicit tests for divergent c4/c128 sizing would harden the contract).

## References

- Issue sunway513/atom#37 — full decision record + Evidence A through J
- Evidence J `EVIDENCE_J.md` — Sprint 1 silicon attribution (corrected by this Evidence K)
- AITER PR `sunway513/aiter#60` — validator (correct as `sparse_attn` ABI gate; out of scope for Compressor)
- ATOM PR `sunway513/atom#54` — this PR (originally a Python bisect harness; pivoted to root-cause + Bug #1 fix after ROCr Debug Agent provided the kernel identification)
- ROCr Debug Agent docs — https://rocm.docs.amd.com/projects/rocr_debug_agent/en/latest/
