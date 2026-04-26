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

## Sprint 2 next steps

1. Split `DSV4KVPool._compressor_state` / `_compressor_score` into per-ratio slabs (c4 / c128) with correct ring_size + state_inner_dim.
2. Update `compute_out_cache_loc` and `view_for_layer` to dispatch into the right slab by layer's compress_ratio.
3. UTs covering both ratios (currently only c4 path was implicitly exercised by Indexer; c128 was uncovered).
4. Re-run silicon `--mode single` then `--mode multi` after Bug #2 fix; both should reach decode + ALL DONE.
5. Run gsm8k accuracy gate post-silicon-green for the W4.5 owner sign-off.

## References

- Issue sunway513/atom#37 — full decision record + Evidence A through J
- Evidence J `EVIDENCE_J.md` — Sprint 1 silicon attribution (corrected by this Evidence K)
- AITER PR `sunway513/aiter#60` — validator (correct as `sparse_attn` ABI gate; out of scope for Compressor)
- ATOM PR `sunway513/atom#54` — this PR (originally a Python bisect harness; pivoted to root-cause + Bug #1 fix after ROCr Debug Agent provided the kernel identification)
- ROCr Debug Agent docs — https://rocm.docs.amd.com/projects/rocr_debug_agent/en/latest/
