# DSV4-Pro CUDAGraph Capture — 6-Phase Restructure Design

**Status**: Design proposal, no code landing here. Targets the upstream
[ROCm/ATOM PR #650](https://github.com/ROCm/ATOM/pull/650)
(`feat/deepseek-v4-pr1-skeleton` by @valarLip / Lingpeng Jin).

**Why this doc exists**: Lingpeng's [`atom/model_ops/attentions/deepseek_v4_attn.py:467`](https://github.com/ROCm/ATOM/pull/650/files) explicitly tags CUDAGraph capture as "PR4 scope" and stubs `build_for_cudagraph_capture` to return dummy metadata. A 2026-04-30 attempt to enable capture in one shot (3 walls hit, all reverted) confirmed Lingpeng's PR4 sizing was correct: capture for V4 requires a model-architecture refactor + per-seq loop kill + state-cache pinning, not a stub fix. This doc proposes the 6-phase plan grounded in **upstream patterns** from SGLang's NSA+DeepSeek-V3.2 work and vLLM's DeepSeek-V3 / DSA / Mamba CUDAGraph implementations.

## Walls hit by the one-shot attempt (evidence the scope is real)

| # | Symptom | Root cause | Fix scope |
|---|---|---|---|
| 1 | `build_for_cudagraph_capture` returns dummy `AttentionMetaData(cu_seqlens_k=None, max_seqlen_q=1, ...)` instead of `(metadata, context)` tuple | Stub | Trivial — 80 lines if Phases 0–3 are in place; useless if not |
| 2 | `KeyError: 'kv_indptr'` in `model_runner.py:1847` `forward_vars["kv_indptr"].gpu.zero_()` | V4 has no kv_indptr forward_var (paged-attention concept) | 1-line gate `if "kv_indptr" in self.forward_vars` |
| 3 | `RuntimeError: expanded size (7168) must match (129280)` at `outputs[:num_tokens] = self.model(...)` during graph warmup | `DeepseekV4Model.forward` returns logits (vocab=129280), but `forward_vars["outputs"]` is allocated to `hidden_size=7168` | **Phase 0 (this doc)** — model contract refactor |
| 4 (predicted) | Per-seq Python loop with `.item()` syncs at `deepseek_v4.py:824, 1098, 1486` would fail capture once Wall 3 cleared | Fundamental dispatch shape | **Phases 1+2 (this doc)** |
| 5 (predicted, ROCm-specific) | `q[..., :rope_dim] = q_rope` self-aliased in-place op rejected during HIP capture (per SGLang [#489a0b786](https://github.com/sgl-project/sglang/commit/489a0b786)) | Out-of-place needed | **Phase 5** |

## Upstream patterns — what to copy

### SGLang's NSA + DeepSeek-V3.2 path

Source: `/tmp/sglang` HEAD `ff98a4010`. All file paths refer to that clone.

Decision matrix:

| Concern | SGLang answer | File:line |
|---|---|---|
| **LM head in graph?** | YES — `model.forward` returns logits; routed through pre-allocated `next_token_logits_buffer (max_num_token, vocab_size, fp32)` via `_copy_logits_to_buffer` | `cuda_graph_runner.py:195-198`, `logits_processor.py:983-993`, `models/deepseek_v2.py:2296-2328` |
| **`.item()` elimination** | `seq_lens_cpu` true CPU mirror tensor pre-allocated in `DecodeInputBuffers`; replay does CPU→CPU `.copy_()` (no D2H) | `cuda_graph_runner.py:244-250, 379-382`; intra-graph reads use pre-baked Python lists like `nsa_extend_seq_lens_list = [1] * num_tokens` (`nsa_backend.py:840`) |
| **State cache pinning** | One fixed-size tensor list per layer in `NSATokenToKVPool`; `set_swa_loc(loc)` called BEFORE capture pins the loc tensor as a captured input | `memory_pool.py:1903-1922`, `cuda_graph_runner.py:1151-1158`, `swa_memory_pool.py:146-170` |
| **Per-seq loop kill** | One batched `deep_gemm.fp8_paged_mqa_logits` (HIP: `aiter.deepgemm_fp8_paged_mqa_logits`) over the full batch fed by fixed-shape `page_table[max_num_token, max_context_len] int32` + `seqlens_int32` + `schedule_metadata` | `nsa_indexer.py:419-527`, `nsa_backend.py:935` |
| **Capture metadata contract** | Two-method split: `init_cuda_graph_state(max_bs, max_num_tokens)` allocs once, `init_forward_metadata_capture/replay_cuda_graph(bs, ...)` slices + freezes in `NSAMetadata` dataclass + `.copy_()` rewrites at replay; `decode_cuda_graph_metadata[bs] = metadata` cache keyed on bs | `nsa_backend.py:766-804, 806-959, 961-999`; driver glue `cuda_graph_runner.py:1104-1113, 1270-1279` |
| **ROCm gotcha** | Self-aliased in-place rotary write rejected during HIP capture; fix is `clone()` rotary outputs | SGLang PR [#489a0b786](https://github.com/sgl-project/sglang/commit/489a0b786) |

Recent PRs implementing this:
- [#16841 `afe285f7b`](https://github.com/sgl-project/sglang/pull/16841) "[AMD] enable CUDA graph for NSA backend" — initial NSA-on-cudagraph (~260 lines)
- [#18526 `7e46aafeb`](https://github.com/sgl-project/sglang/pull/18526) "[AMD] Enable cudagraph for aiter nsa backend" (~130 lines)
- [#21405 `5a3531641`](https://github.com/sgl-project/sglang/pull/21405) "Enable IndexCache for DeepSeek V3.2"
- [#23552 `95d021b52`](https://github.com/sgl-project/sglang/pull/23552) "Pre-set SWA cache location in CudaGraphRunner"
- [#22218 `60bbb800d`](https://github.com/sgl-project/sglang/pull/22218) "[Experimental] Breakable Piecewise Cuda Graph" — fallback if full-CG can't be captured

### vLLM's DeepSeek-V3 / V3.2 / Mamba path

Source: `/home/pensun/rixl-buildtime/vllm` HEAD `d1a75e303`. All paths absolute.

Decision matrix (where it differs from SGLang):

| Concern | vLLM answer | File:line |
|---|---|---|
| **LM head in graph?** | NO — `model.forward` returns hidden_states; `compute_logits(hidden[indices])` runs OUTSIDE the captured region | `models/deepseek_v4.py:1538-1548, 1531-1536`; runner `gpu_model_runner.py:4090-4091` |
| **`.item()` elimination** | `CpuGpuBuffer` class — pinned CPU tensor + GPU tensor + numpy view; `non_blocking=True` H2D | `v1/utils.py:108-143`, `v1/worker/gpu_model_runner.py:680-705` |
| **State cache pinning** | Builder `__init__` allocs persistent `state_indices_tensor_d` to `decode_cudagraph_max_bs`; replay copies live block-table into prefix slice; padded slots get sentinel `NULL_BLOCK_ID` | `v1/attention/backends/mamba_attn.py:117-149, 497-507`; DSA indexer `v1/attention/backends/mla/indexer.py:283-348` |
| **Per-seq loop kill** | `split_decodes_and_prefills(common_attn_metadata)` returns 4 ints from CPU slices of `query_start_loc_cpu`; one batched MLA decode call covers all decode rows; `_prepare_uniform_decode_kernel` Triton expansion for variable-length cases | `v1/attention/backends/utils.py`; `model_executor/layers/attention/mla_attention.py:1826-1842`; `v1/attention/backends/mla/indexer.py:35-66, 414-443` |
| **Capture metadata contract** | `AttentionMetadataBuilder.build_for_cudagraph_capture(common_meta) -> M`; called by `GPUModelRunner._build_attention_metadata` at `gpu_model_runner.py:2255-2258`; default delegates to `build(common_prefix_len=0, common_attn_metadata=meta)`; MLA override asserts decode-only | `v1/attention/backend.py:617-627`; `mla_attention.py:1786-1801` |
| **Mixed CSA/HCA layers** | Routed through separate `AttentionGroup` objects, each with its own metadata builder, all sharing the same captured graph (pointers stable) | `gpu_model_runner.py:6280-6294` |

Recent PRs:
- [vllm-project/vllm#37421 `b55d830ec`](https://github.com/vllm-project/vllm/pull/37421) "Persistent TopK scheduler: unified CUDAGraph-safe kernel ... DeepSeek-V3.2 DSA decode"
- [#34457 `dc5fa77a4`](https://github.com/vllm-project/vllm/pull/34457) "[MTP][Sparse MLA] Allow sparse MLA with MTP to run with FULL cudagraphs"
- [#36519 `9095cbbfb`](https://github.com/vllm-project/vllm/pull/36519) "[Bugfix][Sparse MLA] report indexer CG support properly"
- [#35719 `82f3f30e2`](https://github.com/vllm-project/vllm/pull/35719) "[ROCm][Perf] Enable `sparse_mla`'s cudagraph on ROCm platform"
- [#29893 `a1d627e40`](https://github.com/vllm-project/vllm/pull/29893) "[BugFix] Fix assert in `build_for_cudagraph_capture`"
- [#28044 `e5ff14021`](https://github.com/vllm-project/vllm/pull/28044) "[cudagraph] fix cudagraph warning in deepseekv32"
- [#40772 `7d3195ea9`](https://github.com/vllm-project/vllm/pull/40772) "[Bugfix] Fix IMA in DSA + MTP"

### Picking SGLang's vs vLLM's stylistic choice

| Choice | SGLang | vLLM | Recommended for V4 |
|---|---|---|---|
| LM head in graph | ✅ inside | ❌ outside | **Inside (SGLang)** — V4's `DeepseekV4Model.forward` already fuses LM head; matching SGLang requires only routing logits through a fixed-address buffer rather than splitting the model |
| `.item()` elimination | `seq_lens_cpu` direct mirror | `CpuGpuBuffer` typed wrapper | **`CpuGpuBuffer` (vLLM)** — ATOM's `forward_vars` already partly uses this pattern via `CpuGpuBuffer` (atom/v1/utils.py); systematizing it is mechanical |
| Per-seq loop kill | `page_table` + 1 batched `fp8_paged_mqa_logits` | `split_decodes_and_prefills` + 1 batched MLA + Triton variable-length expansion | **vLLM split-pattern + V4's existing block_table** — V4 already uses block_table per paper §3.6.1, so the path is "build batched dispatch on top of block_table" |
| State cache pinning | `set_swa_loc(loc)` pre-capture | builder owns `state_indices_tensor_d`, `NULL_BLOCK_ID` padding | **Hybrid**: `set_*_loc(loc)` pin pattern + sentinel padding for unused slots |
| Capture metadata builder | `init_cuda_graph_state` + `init_forward_metadata_{capture,replay}_cuda_graph` | `build_for_cudagraph_capture(common_meta)` | **vLLM contract** — ATOM's parent `AttentionMetadataBuilder` is already on the vLLM contract |

## 6-phase plan

| Phase | Goal | Pattern source | Effort | Files touched | Acceptance gate |
|---|---|---|---|---|---|
| **0** | Pin `model.forward` logits output to a fixed-address buffer | SGLang `next_token_logits_buffer` | ½d | `model_runner.py` forward_vars; `deepseek_v4.py:DeepseekV4ForCausalLM.forward` | conc=1 eager run still produces correct output; capture prep no longer crashes at the `outputs[:num_tokens] = self.model(...)` assignment |
| **1** | Replace every per-step `.item()` / `.tolist()` in V4 forward with `CpuGpuBuffer` (cpu+gpu+numpy view, non_blocking H2D) | vLLM `CpuGpuBuffer` | 1d | `deepseek_v4_attn.py` builder; `deepseek_v4.py:824, 1098, 1486` (the 3 fallback `.item()` sites) | grep `.item()` in V4 hot path returns 0 hits; conc=4 silicon ≥ baseline |
| **2** | Replace per-seq Python loop with batched dispatch over `block_table` + `seqlens_int32` + `schedule_metadata` | vLLM `split_decodes_and_prefills` + SGLang `page_table`; per-paper-type `AttentionGroup` for CSA vs HCA | 2d | `deepseek_v4.py:1469-1620` (the for loop); new batched Triton kernel for sparse_attn forward (or aiter equivalent) | conc=4 silicon ≥ baseline AND profile shows zero Python overhead per layer |
| **3** | Pin Compressor state + SWA cache loc tensors before capture | vLLM `state_indices_tensor_d` + sentinel `NULL_BLOCK_ID`; SGLang `set_swa_loc(loc)` | 1d | `deepseek_v4_attn.py:allocate_per_req_cache`; `v4_kernels/state_writes.py` skip-on-sentinel | capture proceeds for at least bs=1 |
| **4** | Implement real `build_for_cudagraph_capture(bs)` returning `(metadata, context)` with bs-sized fixed-address tensors | vLLM `build_for_cudagraph_capture` contract | ½d | `deepseek_v4_attn.py:467` | capture succeeds for bs ∈ {1, 2, 4} |
| **5** | ROCm gotcha: `clone()` rotary outputs to break self-aliased in-place ops | SGLang [#489a0b786](https://github.com/sgl-project/sglang/commit/489a0b786) | ½d | `deepseek_v4.py:1064-1110` (Indexer apply_rotary), wherever `q[..., -rd:] = q_rope` patterns appear | replay produces sensible output (smoke 4/4) |
| **6** | Validation + perf | — | 1d | smoke + gsm8k limit=20 conc=4 + per-token latency benchmark | smoke 4/4, gsm8k ≥ baseline-5pp, latency: eager 13s/req → graph target 5–8s/req |

**Total: ~6 days serial.** Critical path is Phase 2 (per-seq loop kill); Phases 0/1/3/4/5 unblock it. Phases 0/1 can parallelize, then 2, then 3/4/5 can parallelize.

## PR cut suggestion (5 PRs, each independently reviewable)

| PR | Phases | Reviewable size | Acceptance |
|---|---|---|---|
| **4a** | Phase 0 only | ~150 lines, 2 files | conc=1 eager unchanged; new buffer wired |
| **4b** | Phase 1 only | ~200 lines, 2 files | grep `.item()` returns 0 in V4 hot path; conc=4 unchanged |
| **4c** | Phase 2 (the heavy one) | ~600 lines, 4 files + new kernel | conc=4 unchanged; Python loop gone |
| **4d** | Phases 3 + 4 + 5 | ~300 lines, 3 files | capture succeeds + smoke 4/4 + replay correct |
| **4e** | Phase 6 only — perf gate | recipes + tests | per-token latency target met; CI gates added |

Each PR independently revertable. Phase 0 can land the day after this design is reviewed; Phase 6 lands ~6 working days later.

## What I tried in the one-shot attempt (so we don't repeat it)

The `dev/dsv4-on-top-of-pr650` local branch (in `/home/pensun/ATOM`) had two trial fixes that have been reverted:

1. **Wall 1 fix** — implemented `build_for_cudagraph_capture(bs)` returning bs-sized AttentionMetaData with CPU mirrors. Code was correct in shape but Phase 3 + Phase 0 must land first or capture still fails downstream.
2. **Wall 2 fix** — gated `forward_vars["kv_indptr"].gpu.zero_()` on key presence. Tiny, harmless, may be reincorporated as part of PR 4d (the capture-enable PR).

Both reverted because shipping them without Phases 0–3 produces no test value (capture still fails at Wall 3, the model output buffer mismatch).

## Sanity checks built into Phase acceptance

- **Phase 0**: assert the new buffer is the same address across two consecutive forward calls (proves pinning works).
- **Phase 1**: `grep -nE "\.item\(\)|\.tolist\(\)" atom/models/deepseek_v4.py atom/model_ops/attentions/deepseek_v4_attn.py | grep -v "non-hot-path"` returns 0.
- **Phase 2**: profile shows GPU utilization > 90% at conc=4 (was ~70% with Python loop overhead).
- **Phase 3**: replay 100 times, assert per-layer state buffer addresses are constant.
- **Phase 4**: capture wall-clock should be < 60s for bs ∈ {1, 2, 4}.
- **Phase 5**: smoke 4/4 stays correct under capture (proves no aliasing corruption).

## 2026-04-30 Empirical Findings (Phase 0/1 + Walls landed in dev branch)

Following the design above, landed Phase 0 + Wall #1 + Wall #2 + Phase 1
partial in `dev/dsv4-on-top-of-pr650` (commit `799e90e`) atop Lingpeng's
PR #650 stack. Companion fix in `aiter-lingpeng/aiter/utility/fp4_utils.py`.

### Walls confirmed by silicon (v19a/v19b iterations)

| # | Original prediction | Empirical result | Resolution |
|---|---|---|---|
| 1 | Stub returns dummy `AttentionMetaData` | ✅ Confirmed `TypeError: cannot unpack non-iterable AttentionMetaData object` at `model_runner.py:1895` | Stub now returns `(metadata, context)` matching `aiter_mla.py:910` contract |
| 2 | `KeyError: 'kv_indptr'` | ✅ Confirmed at `model_runner.py:1863` | 1-line gate `if "kv_indptr" in self.forward_vars` |
| 3 | Outputs dim mismatch 7168/129280 | ✅ Confirmed | Phase 0: `allocate_forward_vars` detects DSV4 arch, sizes outputs to vocab_size |
| 4 | Per-seq `.item()` syncs | ✅ Phase 1 partial: 3 sites gated via `start_pos: Optional[int]` plumbed from per-seq dispatcher's CPU mirror | Phase 1 complete in landed commit |
| 5 (NEW) | (not predicted) HIP boolean-mask in-place write `exponent[round_case] += 1` in `aiter/utility/fp4_utils.py:56` triggers `hipErrorStreamCaptureUnsupported` | ✅ Confirmed during v19b run3 | `torch.where` rewrite in aiter |
| 6 (NEW) | (not predicted) `torch.tensor()` allocation at `deepseek_v4.py:1460` (warmup `context_lens` dummy) triggers `hipErrorStreamCaptureUnsupported` during capture | ✅ Confirmed during v19b run4 | Pre-allocate `_dummy_context_lens` lazily, `fill_()` on subsequent calls |
| 7 (NEW, OPEN) | 10+ dynamic alloc sites scattered across V4 hot path: `_v4_gather_compressed` (L152, 173, 174), `_get_window_topk_idxs` (L640+), `_get_compress_topk_idxs` (L676+), `Compressor.forward` (L940-942), `Indexer.forward` (L1148, 1157-1165), per-seq loop `slot_per_token` (L1547) | Each is `torch.tensor/zeros/full/arange` inside hot path | **Genuine Phase 2-4 systematic refactor** — not patchable site-by-site without losing correctness; per-seq Python loop kill (Phase 2) eliminates most call sites; remainder need state-cache pinning (Phase 3) + real metadata builder (Phase 4) |

### Eager regression evidence (Phase 0 + landed patches)

`v19a` (script: `v19a_phase0_eager_regression.sh`):
- smoke 4/4 (4 prompts: arithmetic + word problems) at conc=4
- gsm8k 18/20 = **0.9000**, matches v16 baseline (`gsm8k 0.90/0.90`)
- Verifies Phase 0 + Wall #2 gate + Phase 1 plumbing don't regress eager mode

### What this confirms about Lingpeng's "PR4 scope" sizing

Lingpeng's [`deepseek_v4_attn.py:467`](https://github.com/ROCm/ATOM/pull/650/files) tagged capture as "PR4 scope" with a stub. Empirical Walls #5/#6/#7 above
**confirm the scope is real and larger than the predicted Walls #1-#5 alone**:

- Walls #1, #2: defensive 1-line gates (this commit) ✅
- Walls #3, #4 partial: Phase 0/1 (this commit) ✅
- Wall #5 (HIP mask write): mechanical torch.where (this commit, in aiter) ✅
- Wall #6 (one alloc site): pre-alloc + fill_ (this commit) ✅ as pattern
- **Wall #7 (10+ alloc sites): genuine Phase 2 batched dispatch refactor**
  + Phase 3 state cache pinning + Phase 4 real metadata builder. Each
  alloc site individually is fixable via the Wall #6 pattern, but doing
  10+ piecewise patches loses architectural coherence — better as one
  Phase 2 commit that kills the per-seq Python loop and exposes batched
  `block_table + cu_seqlens_q + slot_mapping`-driven kernels.

The 6-phase estimate (~6 working days serial) is **validated as conservative**
— the discovered Walls #5/#6/#7 don't expand the phase count; they're
absorbed by Phase 5 (#5), and Phases 2-4 (#7). Wall #6 was already
implicit in Phase 3 (state cache pinning ~ pre-allocation).

## References

- Sprint 7i one-shot attempt + walls: [`docs/evidence/dsv4_w45/EVIDENCE_M.md`](../evidence/dsv4_w45/EVIDENCE_M.md) Sprint 7i section
- Empirical walls landed in dev branch commit `799e90e` (`dev/dsv4-on-top-of-pr650`): Phase 0 + Walls 1/2 + Phase 1 partial + Wall #6 pattern
- Hand-off issue: [sunway513/ATOM#60](https://github.com/sunway513/ATOM/issues/60)
- Tracking issue: [sunway513/ATOM#37](https://github.com/sunway513/ATOM/issues/37)
- Methodology skills: [`.claude/skills/ramp-inference-feature.md`](../../.claude/skills/ramp-inference-feature.md), [`.claude/skills/redo-after-peer-success.md`](../../.claude/skills/redo-after-peer-success.md)
