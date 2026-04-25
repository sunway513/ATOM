# RFC: DeepSeek-V4 KV Cache Reform

| Field | Value |
|---|---|
| RFC ID | 2026-04-25-dsv4-kvcache-reform |
| Status | **v0.2.6 — CLOSED for correctness implementation (Codex final-pass approved); release-track gates §1.1 separately** |
| Author | sunway513 |
| Reviewers | Codex (v0.1 + v0.2.3 second-pass applied) |
| Base branch (ATOM) | `lingpeng/dsv4-pr1-skeleton` ← ROCm/ATOM `feat/deepseek-v4-pr1-skeleton` (PR [#650](https://github.com/ROCm/ATOM/pull/650)). **Fork mirror**: [`sunway513/ATOM:lingpeng/dsv4-pr1-skeleton`](https://github.com/sunway513/ATOM/tree/lingpeng/dsv4-pr1-skeleton) @ `cdbff359` |
| Companion (aiter) | `lingpeng/fix-mhc-device` ← ROCm/aiter `fix_mhc_device` (PR [#2916](https://github.com/ROCm/aiter/pull/2916)). **Fork mirror**: [`sunway513/aiter:lingpeng/fix-mhc-device`](https://github.com/sunway513/aiter/tree/lingpeng/fix-mhc-device) @ `76ea1ed5` |
| Scope | KV-class state management for DSV4 multi-request inference + InferenceX (formerly InferenceMax) submission foundation |
| Tracking issue | [sunway513/ATOM#35](https://github.com/sunway513/ATOM/issues/35) |

### Quickstart for collaborators

Both reform-relevant branches are mirrored on `sunway513/{ATOM,aiter}` so anyone can fetch without being a ROCm-org member.

```bash
# ATOM — get the DSV4 PR1 skeleton + all 13 commits valarLip pushed
git clone https://github.com/sunway513/ATOM.git && cd ATOM
git checkout lingpeng/dsv4-pr1-skeleton
# To track upstream once you have ROCm/ATOM access:
#   git remote add upstream https://github.com/ROCm/ATOM.git && git fetch upstream

# aiter — companion mhc device fix (must be pinned for non-CUDA-default-device launch paths)
git clone https://github.com/sunway513/aiter.git && cd aiter
git checkout lingpeng/fix-mhc-device

# This RFC lives on rfc/dsv4-kvcache-reform branch (when published) or in tracking issue #35
```

---

## ⭐ Recommended path (set in v0.2)

| Decision | Choice | Reason (one line) |
|---|---|---|
| **Reform option (§6)** | **Option B — Spec contract** | Owns ATOM runtime, mirrors vLLM contract for portability, foundation for fast backends |
| **Performance backend (§7) for the correctness PR** | **P1 — vLLM-style flat `SlidingWindowMLASpec`** | Fastest path to correct multi-request + prefix cache + CUDAGraph parity; identical semantics to upstream |
| **Option A — Tactical slot mapper** | **DEMO-ONLY, do not merge** | Useful for a 1-day "batch doesn't garble" proof; creates debt; not on the long-term path |
| **Option C — Full vLLM adoption** | **Non-viable for this submission window** | Loses 13-commit ROCm/MoE fix set; inherits open vLLM ROCm blockers; strategic shift orthogonal to RFC |
| **P2 — SGLang `KVAndScore` + ring buffer** | **Perf follow-up RFC, gated on benchmark** | Not orthogonal to B (changes ownership/recovery semantics); needs FP8/FP4 quant audit |
| **P3 — Hybrid backend selector** | **Later — only if P2 lands and shows wins** | Premature now |

> **The closed implementation plan is the responsibility of the follow-up impl doc** referenced in §10. This RFC sets defaults and surfaces the still-open items in §1.1, §6.2.1, §7.4, §8 — those need explicit answers before impl starts but do not block consensus on the recommended path.

---

## 0. Decision Record (closed)

> v0.2.5 closes the RFC. The decisions below are locked; further changes require a new RFC version.

| Decision | Locked value | Owner | RFC §ref |
|---|---|---|---|
| **Reform path** | **B — spec-based ATOM contract** (vLLM-isomorphic field names; logical→physical pool coalescence) | this RFC | §6.2 |
| **Performance backend (correctness PR)** | **P1 — vLLM-style flat `SlidingWindowMLASpec`** | this RFC | §7.1 |
| **Block table shape** | **`Sequence.block_tables: dict[str, list[int]]`** keyed by logical cache name; `block_table` retained as backward-compat property | this RFC | §6.2.2 / §8.5 |
| **Default page (block) size** | **256 native tokens** matching vLLM; override gated by validation | this RFC | §6.2.2 |
| **Multi-pool config field** | **`Config.kv_cache_pool_blocks: dict[str, int]`** (new); `num_kvcache_blocks: int` deprecated alias | this RFC | §6.2.1 |
| **Reset semantics** | **Write-before-read invariant** at slot level; remove all model-level `start_pos==0` clears | this RFC | §8.1 |
| **Position semantics** | **Vector positions** via existing `cu_seqlens_q` / `cu_seqlens_k` / `token_to_seq_idxs` from `attn_metadata`; no scalar `start_pos` collapse | this RFC | §3.1 (bug #7) / §6.2.1 |
| **vLLM contract enforcement** | **Vendored snapshot** at `tests/audit/_vllm_spec_snapshot.py` + `_VLLM_AUDIT_COMMIT` constant; per-PR audit is **offline**; refresh via separate weekly cron | this RFC | §9.5.5 |
| **CUDAGraph stance (correctness PR)** | Decode capture path uses **fixed-shape metadata** + no request-dependent Python branching; prefill may stay eager | this RFC | §8.2 |
| **Test gate split** | **§10.1** correctness PR (L1/L2/L4 + L3 pre-merge subset) ↔ **§10.2** release/InferenceX (L3 nightly + L5 + §1.1 cells signed off) | this RFC | §10 |
| **Default starting parallelism (Phase 1)** | **TP=8 × DP=1 × EP=1** on a single MI355X 8-GPU node — easier to fit, fastest to bring up | this RFC | §1.1 |
| **InferenceX submission parallelism (Phase 2)** | **1P2D × EP=8** disaggregated, multi-node | this RFC | §1.1 |
| **Hardware target** | **MI355X** | user (sunway513) | §1.1 |
| **Baseline to beat (primary)** | **NVIDIA Blackwell B200 InferenceX submission** at matching matrix cell | this RFC | §1.1 |

**Status changes**:
- Option A (tactical slot mapper) → **demo-only, do not merge**
- Option C (full vLLM adoption) → **non-viable for this submission window**
- P2 (SGLang `KVAndScore` + ring buffer) → **separate perf RFC**, gated on P1 measured baseline
- P3 (hybrid backend selector) → **deferred until P2 ships**

**Items intentionally still open** (do NOT block correctness PR):
- §1.1 release-track items Q1.1.A / Q1.1.B / Q1.1.C — owned by InferenceX submission lead, gates §10.2 only
- Q8.8 — aiter version pin strategy (release coordination)
- Q6.3.1 — ATOM↔vLLM strategic merger (separate strategy doc)

---

## 1. Goal

> "Establish a robust KV cache management reform to adapt DeepSeek-V4, while maintaining our foundation to achieve very fast performance options."

Concretely:

1. **Robust**: multi-request inference must be correct (no cross-request state pollution, no token garbling, no shared-state races) for DSV4's three KV-class state-bearing subsystems — main MLA KV, Compressor state (`kv_state` + `score_state`), Indexer KV.
2. **Fast**: the design must not foreclose future kernel-level performance wins. The reform should be a **foundation** — additive, modular, swappable backends — not a tax on the hot path.
3. **Aligned**: by default, the on-disk interfaces (specs, slot mapping, block tables) should be **isomorphic to vLLM's** so that ATOM↔vLLM mutual portability stays cheap. Performance backends underneath that interface are free to go faster than vLLM (SGLang has shown how — §4).
4. **Submission-ready**: ship a path that lands on InferenceX (formerly InferenceMax)-class workloads with the perf primitives §1.1 lists.

This RFC does **not** scope: MTP integration (deferred per PR#650 roadmap PR5), the AITER native sparse_attn kernel (PR4), CUDAGraph removal of `--enforce-eager` (PR6), OAI server wiring.

---

## 1.1 Submission target — InferenceX (was InferenceMax) DSV4 path

> v0.2.5: TBDs filled against the InferenceX (formerly InferenceMax) live benchmark framework. Specific daily tok/s numbers are **not** pinned here because the leaderboard is updated continuously; the **configuration matrix and baseline source** are pinned. Live numbers track from [`inferencex.semianalysis.com`](https://inferencex.semianalysis.com) and the [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) repo.

| Axis | Value | Notes |
|---|---|---|
| Model | **DeepSeek-V4-Pro** (1.6T total / 49B active) — listed as "New" on InferenceX | Confirmed |
| Hardware | **MI355X** (1 node = 8 GPUs) | Confirmed (user input) |
| TP × DP × EP | **Phase 1 (default starting config, single-node, easier to fit)**: **TP=8 × DP=1 × EP=1** on one MI355X 8-GPU node. **Phase 2 (InferenceX submission target, multi-node)**: **1P2D × EP=8** disaggregated (1 prefill node + 2 decode nodes; expert parallel = 8 across nodes) — matches the AMD MI355X DSV4 profile on InferenceX | per [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) workload axes |
| ISL / OSL matrix | InferenceX standard set: **1k/1k** (latency-sensitive interactive), **2k/2k** (balanced), **32k/2k** (long-context) — plus **8k/1k** (used in current InferenceX DSV3.2/R1 reference path) | per InferenceX matrix |
| Concurrency / batch | Two-axis: **interactive** (single-stream low-latency) + **throughput** (bulk concurrent). InferenceX Pareto curve is the deliverable, not a single point. | per InferenceX |
| KV dtype | FP8 attention KV; FP4 indexer KV (per vLLM DSV4 blog) | Confirmed direction |
| Block size | 256 native tokens (matches vLLM) | Confirmed |
| Latency target (TPOT) | **Match or beat NVIDIA B200 InferenceX submission** at the same matrix cell. Reference quote (InferenceX MI355X article): MI355X "never more than ~15s slower than B200 for a given tok/s/gpu throughput" — this gap is the budget to close. | tracked daily on InferenceX dashboard |
| Throughput target (tokens/s/GPU) | **Per-GPU tok/s parity-or-ahead** vs B200 1P2D EP8 at each matrix cell | tracked daily on InferenceX dashboard |
| Memory budget per seq @ 1M ctx | ≤ 9.62 GiB bf16 baseline; ~50% via FP8/FP4 | per vLLM blog |
| Required perf primitives | block_size=256; FP8 KV; FP4 indexer cache; CUDAGraph (decode capture); AITER sparse attention; MoE EP / all-to-all (MORI); MTP if InferenceX rules permit | confirmed |
| Baseline to beat (primary) | **NVIDIA Blackwell B200** InferenceX DSV4 submission at the same matrix cell (1P2D EP8, MXFP4 native) — currently the leading published number on the InferenceX leaderboard for DSV4-Pro | live: `inferencex.semianalysis.com` |
| Baseline to beat (secondary) | **vLLM ROCm** PR [#40871](https://github.com/vllm-project/vllm/pull/40871) on MI355X (when blockers cleared) — provides the in-family upper bound for "what vLLM can do on this HW" | open PR |
| Reference framework versions | Pinned in `tests/audit/_vllm_spec_snapshot.py` per §6.2.1 / §9.5.5 — separate from runtime tracking | tooling |

**Open release-track items** (do NOT block correctness PR per §10.1; gate §10.2):
- Q1.1.A — confirm primary InferenceX matrix cells we commit to (1k/1k + 8k/1k + 32k/2k? all four?). *Owner: InferenceX (formerly InferenceMax) submission lead · Due: when impl correctness PR enters review*
- Q1.1.B — confirm framework path: ATOM-native vs SGLang-on-MI355 fallback for cells we can't yet hit. *Owner: same · Due: same*
- Q1.1.C — MTP inclusion gate per InferenceX rules. *Owner: same · Due: same*

> **Closure note**: §1.1 no longer blocks the correctness PR. The matrix and baseline source are concrete (InferenceX); the daily numbers update on the leaderboard; the three items above belong to a parallel release-planning doc owned by the InferenceX (formerly InferenceMax) submission lead.

---

## 2. Current state — what works, what doesn't

### 2.1 Verified working (`lingpeng/dsv4-pr1-skeleton`)
- Single-sequence inference at TP=8 on `/data/DeepSeek-V4-Pro` with `ATOM_USE_TRITON_MOE=1`
- 512-token coherent Chinese / English output, `--temperature 0.0`, `--enforce-eager`
- Bit-exact reference parity (259 tensors, max_abs_diff = 0.0) on toy 4-layer dummy weights
- 13-commit set of MoE / loader / hash-routing / dequant fixes (must not regress under reform)

### 2.2 Broken
**Multi-request inference does not crash but produces incoherent output** ("吐字不对，挂倒是不挂"). Per PR#650 description:
> Batching: `kv_cache[:1,...]` hardcoded; multi-request KV isolation pending scheduler integration

### 2.3 Surrounding infrastructure (already in `atom/`)
ATOM already ships a paged-KV / continuous-batching stack that PR#650's model file does not consume:

| Class | File | Currently used by DSV4? |
|---|---|---|
| `Scheduler` | `atom/model_engine/scheduler.py` | No — `simple_inference.py` is the only verified entry |
| `BlockManager` | `atom/model_engine/block_manager.py` | No |
| `Sequence` | `atom/model_engine/sequence.py` (incl. `block_table`, **`mamba_block_table`** precedent) | No |
| `ForwardContext` / `set_kv_cache_data` | `atom/utils/forward_context.py` | No |
| `attn_metadata_builder` | `atom/model_engine/model_runner.py:592` | No |
| `ModelRunner.get_num_blocks() → dict[str, int]` | `atom/model_engine/model_runner.py:1216` | **Already dict-typed** — multi-pool plumbing partially staged |
| `ModelRunner.allocate_kv_cache(num_kvcache_blocks)` | `atom/model_engine/model_runner.py:1315` | Single-int param — need multi-pool extension |
| `BlockManager.__init__(config)` | `atom/model_engine/block_manager.py:34` | Single `num_blocks = config.num_kvcache_blocks` — need multi-pool extension |
| `ScheduledBatch` | `atom/model_engine/scheduler.py:197` | Carries single `block_table`-shaped fields — needs per-pool extension |

> **The disconnect is real:** `get_num_blocks` already returns `dict[str, int]`, but `BlockManager.__init__` and `allocate_kv_cache` both consume a single integer. v0.2 names the exact files that need extension (§6.2.1).

---

## 3. Bug surface — grounded in `lingpeng/dsv4-pr1-skeleton:atom/models/deepseek_v4.py`

### 3.1 Mass survey

| Component | Cross-step state? | Bug source? | Evidence |
|---|---|---|---|
| `mHC` (`Block.hc_pre/hc_post`, `ParallelHead.hc_head`) | **No** — pure intra-forward residual transform | No | Confirmed against `vllm/model_executor/layers/mhc.py` (tilelang fused kernel, zero persistent state); ATOM PR#650 mHC blocks are stateless wrappers |
| MoE routing / hash routing (`tid2eid[input_ids]`) | No — per-token table lookup | No | Stateless per construction |
| `Compressor.kv_state` / `Compressor.score_state` | **Yes** — accumulating compressed-window state | **Yes — bug source #1** | `deepseek_v4.py:470-492`: `register_buffer` shape `[max_batch_size, ...]` but **no slot allocator**; `bsz` axis written `[:bsz]` from current call (`deepseek_v4.py:549,557,574,598,601,625,627`) — ANY arrival writes from row 0 |
| `Compressor.kv_cache` (the externally-provided main compressed-KV tensor) | Yes — long-lived per-layer KV | **Yes — bug source #2** | `deepseek_v4.py:464,522` — set by owner; `[:bsz]` writes share rows across requests |
| `Indexer.kv_cache` | Yes — indexer's own compressed KV | **Yes — bug source #3** | `deepseek_v4.py:682-687`: shape `[max_batch_size, max_seq_len // compress_ratio, head_dim]` — but `forward` consumes `self.kv_cache[:1, : end_pos // ratio]` at `deepseek_v4.py:740` (**hardcoded `:1`** — only row 0 ever read) |
| `Compressor` `start_pos == 0` reset (in-module) | — | **Yes — bug source #4** | `deepseek_v4.py:538,610,624`: in-module `score_state` / `kv_state` clears |
| **Model-level reset orchestration** | — | **Yes — bug source #5 (cross-cutting)** ⚠ NEW (audit) | `deepseek_v4.py:994-1002` — `DeepseekV4Attention.forward` orchestrates the actual cross-module reset: `self.kv_cache.zero_()` + `self.compressor.kv_state.zero_()` + `self.compressor.score_state.fill_(-inf)` + `self.indexer.kv_cache.zero_()` + `self.indexer.compressor.kv_state.zero_()` + `self.indexer.compressor.score_state.fill_(-inf)`. This is the **central reset point** that wipes everything on any `start_pos==0`. The accompanying comment at `:990-993` explicitly admits "ATOM warmup forward fills these buffers with garbage" — i.e., the reset is a workaround, not a correctness primitive. |
| `DeepseekV4Attention.kv_cache` (main MQA KV) | Yes | **Yes — bug source #6** ⚠ EXPANDED (audit) | `[:1]` hardcoded at **5 sites**: `deepseek_v4.py:1040, 1044, 1045, 1054, 1059`. Comment at `:1037` admits this is the bug: *"kv_cache slot 0 = our implicit B=1"*. Each site is a multi-request correctness blocker on its own. |
| **Scalar-position bug** at the model wrapper | — | **Yes — bug source #7 (cross-cutting)** ⚠ NEW (audit pass 2) | `deepseek_v4.py:1929` collapses the entire batch's positions to a single int: `start_pos = int(positions[0].item()) if positions is not None else 0`. Multi-request batches contain sequences at *different* positions; this scalar passes ONE value for ALL — every layer's `start_pos == 0` check, RoPE phase, KV-write offset, and topk mask construction is wrong for non-leading sequences. **Even after KV slots are isolated, this bug alone produces token garble.** Fix: DSV4 must consume vector `positions`, `cu_seqlens_q`, `cu_seqlens_k`, `token_to_seq_idxs` from `attn_metadata` (already populated by `attention_mla.py:441-555` consumers). |

### 3.2 Why the existing `[:bsz]` indexing isn't enough
PR#650's buffers are sized for `max_batch_size`, so it _looks_ batched. The actual breakage:

1. **Identity**: `bsz` in any given call is "tokens currently in flight in this forward", not "request id". Two requests in the same batch occupy rows `[0]` and `[1]`; if a third request arrives next step replacing the second, it inherits row `[1]`'s stale state.
2. **Reset**: `start_pos == 0` is checked as an int from the call site, not a per-row flag. It resets **all rows**, not "this request's row".
3. **Hardcodes**: 6 confirmed `[:1]` hardcode sites — `Indexer.forward:740`, `DeepseekV4Attention:1040,1044,1045,1054,1059`. Slot 0 is implicit-B=1 throughout the attention forward.
4. **Scalar position** (Codex pass 2 finding): `:1929` reduces the entire batch's positions to `int(positions[0].item())`, so every layer downstream sees a single `start_pos` regardless of how many sequences are co-batched. Even with slot identity correct, the RoPE phase / mask construction / start-of-prefill detection are all wrong for sequences 1..N-1.

Fix is **slot identity** + **per-row reset** + **eliminate `[:1]` hardcodes** + **vectorize positions** (consume existing `cu_seqlens_q` / `token_to_seq_idxs` already populated by ATOM's `attention_mla.py:441-555` path), not just buffer sizing.

### 3.4 Sub-agent audit findings (added v0.2.2)

Independent source audit of `lingpeng/dsv4-pr1-skeleton` (no GPU, code-only) confirmed all 5 main PR#650 claims with **MEDIUM-HIGH overall confidence**:

| Claim | Verified | Caveat |
|---|---|---|
| Single-seq 512-token output @ TP=8 | ✅ Partially | Hardcoded `[:1]` at 5 attention sites; batch>1 silently clobbers slot 0 |
| Bit-exact 259 tensors max_abs_diff=0.0 | ✅ (torch path) | FusedMoE+triton path validated only via commit messages, not source-auditable |
| 6 specific MoE/loader fixes | ✅ All 6 verified in source | One **stale comment** at `:1208-1211` claims hash routing "NOT yet wired" — contradicts actual implementation at `:1304-1329` (commit `3c37b76`); doc cleanup item, code is correct |
| 43/43 layer-0 params loaded | ✅ (commit msg) | Full-model count requires runtime |
| tid2eid 773k/775k entries | ✅ (claimed) | Buffer shape verified at `:1246-1251` (`vocab=129280 × n_act=6 = 775680`); ratio is runtime |

**Reproducibility risk for third party**:
- Requires `/data/DeepSeek-V4-Pro` weights (not bundled)
- Requires GPU + ROCm + AITER + triton stack
- Default `temperature=0.6` in `simple_inference.py:21` — user **must** explicitly pass `--temperature 0.0` to match the "coherent output" claim
- No in-tree pytest validating the 259-tensor parity (validation is in external team CI)

**Implications for this RFC**: confirms our `[:1]` hardcode + cross-cutting reset diagnosis. Adds 5 specific sites (1040/1044/1045/1054/1059) and the orchestration site (994-1002) that v0.2.1 missed. Highlights one doc-cleanup item (stale comment at 1208-1211) for the implementation PR.

### 3.3 What `lingpeng/fix-mhc-device` (aiter PR#2916) covers
Orthogonal to the above. PR#2916 fixes `mhc_pre`'s intermediate tensor allocations to honor `device=residual.device` instead of the global default. **Required as an aiter version pin** for any non-CUDA-default-device launch path, but does **not** affect multi-request correctness.

---

## 4. Prior art

### 4.1 vLLM (`vllm-project/vllm` PR [#40760](https://github.com/vllm-project/vllm/pull/40760), branch `zyongye/vllm:dsv4`)
**Pattern: every stateful KV-class layer self-declares a `KVCacheSpec`. A unified `KVCacheManager` runs paged allocation. Cross-request isolation = `block_table` per request.**

| Spec class | Used for | Key field |
|---|---|---|
| `MLAAttentionSpec` (`vllm/v1/kv_cache_interface.py:264`) | Main MLA KV; Indexer KV | `compress_ratio` (1 = standard MLA, 4 = c4a, 128 = c128a); `storage_block_size = block_size // compress_ratio` |
| `SlidingWindowMLASpec` (`vllm/v1/kv_cache_interface.py:376`) | Compressor state | `sliding_window = coff * compress_ratio` (8 for C4, 128 for C128) |
| `FullAttentionSpec` | Standard attention | — |

**Compressor self-registration** — `vllm/model_executor/layers/deepseek_compressor.py:147,162`:
```python
self.sliding_window = coff * compress_ratio
def get_kv_cache_spec(self, vllm_config):
    return SlidingWindowMLASpec(..., sliding_window=self.sliding_window, ...)
```

**Indexer slot mapping contract** — `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py:21-91`:
```python
kv_cache_spec = MLAAttentionSpec(block_size=256, compress_ratio=4, ...)
# storage_block_size = 256 // 4 = 64
# block_table_tensor[batch, num_blocks]   — per-request page table
# slot_mapping[num_actual_tokens]          — per-token slot id; -1 for compressed-out positions
```

**Three KV cache buckets** (per the [vLLM blog](https://vllm.ai/blog/deepseek-v4)):
- Largest bucket: c4a main KV + SWA KV + c4a/c128a compressor states
- Middle bucket: c4 indexer KV + c4 indexer compressor state
- Smallest bucket: c128a main KV

Logical block size = **256 native tokens** for compressed layers.

> **Critical observation (Codex):** vLLM models compressor state as a `SlidingWindowMLASpec` **specifically because** that's how it preserves prefix-cache, disaggregated-prefill, and CUDAGraph semantics. The compressor block boundary aligns with the main KV block boundary at the 256-token grid; prefix-cache hits land on that boundary; the compressor state at that boundary is already the correct handoff point. **Any backend that breaks this alignment (e.g., SGLang's ring buffer) breaks one or more of {prefix cache, disagg, CUDAGraph} unless explicitly designed around them.**

### 4.2 SGLang (`sgl-project/sglang` PR [#23600](https://github.com/sgl-project/sglang/pull/23600), branch `sgl-project/sglang:deepseek_v4`)
Same logical layer structure (mHC + Compressor + Indexer + MLA) but **bespoke memory pools instead of generic specs**.

#### 4.2.1 `KVAndScore` co-location (`compress_state.py:13-72`)
KV and score share a single allocation (last dim = `2 * item_size`). One allocation, atomic `clear()`, cache locality. Directly addresses ATOM bug source #1 (split `score_state` / `kv_state` reset race).

#### 4.2.2 Ring-buffer compress state pool (`compress_state.py:147-153`)
```python
state_loc = swa_pages * ring_size + (swa_loc % ring_size)
```
Compressor state needs only `ring_size` slots per page, not the full `sliding_window` flat allocation. Smaller VRAM footprint.

> **Codex caveat applied:** P2 (the SGLang backend) is **not orthogonal** to Option B. Ring-buffer ownership/recovery semantics differ from sliding-window-KV semantics; `state_loc` is computed from a translation that does not align with the prefix-cache block boundary by default. To use P2 with B without regressing prefix cache / disagg / CUDAGraph, the ring-buffer page size must be aligned to the main KV block boundary AND the recovery path must understand which `ring_size` slots map to which prefix-cache hash. This is a non-trivial perf RFC; not a "swap the backend" change.

#### 4.2.3 Specialized `deepseekv4_memory_pool.py` (810 lines)
Per-model memory pool, not a generic `KVCacheManager`. Tighter coupling buys speed; loses portability.

#### 4.2.4 `hisparse_coordinator.py` (449 lines)
Hierarchical sparse coordinator multiplexing c4a / c128a / SWA pools with different page sizes. Out of scope.

### 4.3 Bottom line on prior art
- **vLLM gives us the interface** (specs + slot mapping + block_table) — portable, well-documented, upstream-aligned. **Use this for B/P1.**
- **SGLang gives us potential speed primitives** (`KVAndScore`, ring buffer, hisparse coordinator) — bespoke, faster on paper, but not free-swap with vLLM contract. **Defer to perf follow-up.**

---

## 5. Design space

Two largely-orthogonal axes (with the §4.2 caveat — P2 is **not** truly orthogonal to B):

```
                Performance backend (axis 2)
                ─────────────────────────────────────────────────
                 P1: vLLM-style flat        P2: SGLang KVAndScore   P3: hybrid
                 SlidingWindowMLA           + ring-buffer pool       (configurable)
   ─────────┬─────────────────────────────────────────────────────────────────
   A: bolt  │
   slot     │   A·P1: minimum tax,       A·P2: weird — bolt doesn't  A·P3: not a real
   mapper   │   tactical fix             carry pool semantics          combo
   onto     │   ⚠ DEMO-ONLY              (skip)                        (skip)
   PR#650   │
   ─────────┼─────────────────────────────────────────────────────────────────
   B: spec- │   ★ B·P1: DEFAULT          B·P2: needs design        B·P3: env-flag toggle
   based    │   correctness-first        for prefix-cache /         between P1/P2 backends
   ATOM     │   submission path          CUDAGraph parity           (after P2 ships)
   contract │
   ─────────┼─────────────────────────────────────────────────────────────────
   C: full  │
   vLLM     │   C·P1: fork vLLM model     C·P2: same as B·P2,          C·P3: ATOM is a
   adoption │   files into ATOM           model files lifted           thin shell over vLLM
   ⚠ NON-   │
   VIABLE   │
   ─────────┴─────────────────────────────────────────────────────────────────
```

---

## 6. Reform options

### 6.1 Option A — Tactical slot mapper · ⚠ DEMO-ONLY (do not merge)

**Idea**: don't restructure. Add a `slot_id` argument to `Compressor.forward()` / `Indexer.forward()` / `Attention.forward()` indexing into the existing `register_buffer` `[max_batch_size, ...]` allocations. `Sequence` carries its `slot_id` via a Scheduler-managed free list.

**When you'd use it**: you need a 1-day "batch=4 doesn't garble" demo for a status report. You then **delete the branch** before the long-term path lands.

**Why it's not the merge path**:
- Bypasses `BlockManager` entirely → no prefix caching, no disagg, no fixed memory budget per-pool
- Field names not vLLM-compatible → upstream merge requires rewrite later
- Subtle slot-reuse bugs possible if a slot is freed and re-allocated before all in-flight reads complete (vLLM avoids this through ref-counting on blocks)
- `start_pos == 0` reset semantics still per-call

**Decision**: **demo only**. If you take this path, label the branch `demo/dsv4-batch-quick` and do not propose for merge.

---

### 6.2 Option B — Spec-based ATOM contract · ★ DEFAULT

**Idea**: introduce ATOM-side `KVCacheSpec`-style contract that mirrors vLLM's field-by-field. Each DSV4 layer exposes `get_kv_cache_spec()`. `BlockManager` learns to manage multiple pools (one per spec type). `Sequence` carries multiple block tables — one per pool, generalized via a dict.

#### 6.2.1 Required ATOM machinery changes (concrete file list)

> **Codex pass 1 flagged that v0.1 overclaimed "free" prefix caching / CUDAGraph compatibility.** Below is the actual list of ATOM internals that must change. None of them are "free"; all are tractable. **Codex pass 2 added** the `engine_core.py` / `config.py` scalar-pipe rows and reframed the pool model as logical-vs-physical.

#### Logical-vs-physical pool model (Codex pass 2)

DSV4 has **multiple logical caches** — main MQA KV, c4 main, c128 main, SWA, c4 indexer KV, c4 indexer compressor state, c4 compressor state, c128 compressor state — and **a smaller number of physical bucket sizes** (per the vLLM blog: 3 page-size buckets coalesce them all). The reform must therefore:

1. Each layer registers a logical `KVCacheSpec` (8+ logical caches across the model)
2. The block manager **coalesces** these into physical `BlockPool` instances keyed by `(page_shape, dtype, spec_kind)` — typically 3 physical pools
3. `Sequence.block_tables` is keyed by **logical name**; `BlockManager` does the logical→physical lookup internally

Three-pool hardcode (`main_kv` / `compress` / `indexer`) in DoD is rejected. DoD speaks in terms of logical-cache coverage, not physical-pool count. See updated §10.

#### File-by-file change list

| File / Class | Current shape | Required change | Why |
|---|---|---|---|
| `atom/v1/kv_cache_interface.py` | (not present) | **NEW**: `KVCacheSpec`, `AttentionSpec`, `MLAAttentionSpec`, `SlidingWindowMLASpec`, `FullAttentionSpec`. Field names byte-for-byte vLLM-isomorphic. | Spec contract foundation |
| `atom/model_engine/block_manager.py:34 BlockManager.__init__` | `num_blocks = config.num_kvcache_blocks` (single int) | Accept `kv_cache_specs: dict[str, KVCacheSpec]`. Coalesce into physical `BlockPool` instances keyed by `(page_shape, dtype, spec_kind)`. Maintain `pools: dict[pool_key, BlockPool]` and `logical_to_pool: dict[str, str]`. | Logical→physical coalescence |
| `atom/model_engine/block_manager.py BlockManager.can_allocate` / `allocate` / `free` / `can_append` | Single-pool semantics | Iterate over physical pools; per-pool sufficiency check; return ALL or NONE allocation (atomic across pools) | Avoid partial alloc → leak |
| `atom/model_engine/block_manager.py Block.reset` + prefix-cache hash | resets `ref_count`, `hash`, `token_ids` (metadata only); single global `hash_to_block_id` map | Unchanged for `reset` itself. **Scope the prefix-cache hash by `(logical_cache_name, prefix_hash)`, NOT by physical pool key.** Add `Block.logical_cache_name: str` field. `BlockManager.hash_to_block_id` becomes `dict[tuple[str, int], int]` keyed on `(logical_cache_name, prefix_hash)`. **Reason (Codex final pass)**: physical pools coalesce multiple logical caches with identical `(page_shape, dtype, spec_kind)`; if hash were scoped by physical-pool only, two unrelated logical caches (e.g., `layer.3.attn.indexer_c4` and `layer.5.attn.compressor_c4` if they happen to share a physical bucket) could collide on `prefix_hash` and reuse each other's blocks — silent corruption. Keying on `(logical_cache_name, prefix_hash)` keeps per-logical-cache prefix-sharing while still letting physical pools amortize allocation. | Prevents prefix-cache cross-cache poisoning |
| `atom/model_engine/sequence.py Sequence.block_table` | `list[int]` | Replace with `block_tables: dict[str, list[int]]` (keyed by **logical** cache name). `Sequence.block_table` becomes a property aliasing `block_tables["main"]`. | Per-logical-cache tracking; non-DSV4 unaffected |
| `atom/model_engine/scheduler.py ScheduledBatch:197` | Single-pool fields | Add per-logical `block_tables: dict[str, Tensor]` + per-logical `slot_mapping: dict[str, Tensor]`. Legacy fields aliased to `["main"]`. | Per-pool metadata to model_runner |
| `atom/model_engine/model_runner.py get_num_blocks:1216` | Returns `{"num_kvcache_blocks": int, ...}` — **scalar wrapped in dict**, not per-pool | **Rename to `get_kv_cache_pool_blocks() -> dict[str, int]`** keyed by physical-pool key. Wire actual values from per-pool `KVCacheSpec.page_size_bytes` and per-pool `gpu_memory_utilization` slice. Keep `get_num_blocks()` as deprecated shim returning sum across pools. | Real multi-pool dict (Codex pass 2) |
| `atom/model_engine/model_runner.py allocate_kv_cache:1315` | `num_kvcache_blocks: int` | `kv_cache_pool_blocks: dict[str, int]`; allocate per-pool tensors with per-pool dtype/shape | Multi-pool allocation |
| `atom/model_engine/model_runner.py attn_metadata_builder` | Single-spec builder | Extend with DSV4-aware sub-builder. **Reuse existing primitives**: `cu_seqlens_q`, `cu_seqlens_k`, `token_to_seq_idxs`, `sparse_cu_seqlens_q` from `attention_mla.py:441-555` — already populated, just plumb through to DSV4 layers. Build per-logical-cache slot mappings. | Per-pool metadata; vectorize positions (bug #7) |
| `atom/model_engine/engine_core.py:90,95,99` ⚠ NEW (Codex pass 2) | `block_info["num_kvcache_blocks"]` scalar; `allocate_kv_cache(num_blocks)` scalar; `config.num_kvcache_blocks = num_blocks` scalar | Consume `kv_cache_pool_blocks: dict[str, int]`; pass dict to `allocate_kv_cache`; write to `config.kv_cache_pool_blocks` | Wire scalar-pipe replacement end-to-end |
| `atom/config.py:818 num_kvcache_blocks: int = -1` ⚠ NEW (Codex pass 2) | Scalar field | **Add** `kv_cache_pool_blocks: dict[str, int] = field(default_factory=dict)`. Keep `num_kvcache_blocks: int = -1` as deprecated alias resolving to `sum(kv_cache_pool_blocks.values())`. | Multi-pool config field |
| `atom/utils/forward_context.py AttentionMetadata` | Single `slot_mapping`, `block_table` | Add **logically named** per-pool fields (NOT just three): `main_slot_mapping`/`main_block_table`, `compress_slot_mapping`/`compress_block_table`, `indexer_slot_mapping`/`indexer_block_table` — one set per logical cache the model declares. **Plus** the existing `cu_seqlens_q`/`cu_seqlens_k`/`token_to_seq_idxs` already there are reused for vector positions. | Plumb through; no scalar `start_pos` |
| `atom/models/deepseek_v4.py Compressor` | `register_buffer` for `kv_state`/`score_state`; `start_pos==0` resets | Remove buffers. `forward()` reads `compress_slot_mapping` / `compress_block_table`, indexes into pool tensor. Remove `start_pos==0` reset. **Replace scalar `start_pos` with vector `cu_seqlens_q + token_to_seq_idxs`.** | The fix (bugs #1, #4, #7) |
| `atom/models/deepseek_v4.py Indexer` | `register_buffer kv_cache`; `[:1]` hardcode at `:740` | Remove buffer. Read `indexer_slot_mapping` / `indexer_block_table`. Add `get_kv_cache_spec()`. **Vector positions.** | The fix (bugs #3, #7) |
| `atom/models/deepseek_v4.py DeepseekV4Attention` | External `kv_cache`; 5 `[:1]` sites at `:1040,1044,1045,1054,1059`; reset at `:994-1002` | Add `get_kv_cache_spec()`. forward consumes `slot_mapping` / `block_table`. Remove all 5 `[:1]` hardcodes. Remove `:994-1002` reset block. **Vector positions throughout.** | The fix (bugs #5, #6, #7) |
| `atom/models/deepseek_v4.py DeepseekV4ForCausalLM:1929` | `start_pos = int(positions[0].item())` | Pass full `positions` + `attn_metadata` (with `cu_seqlens_q`, `token_to_seq_idxs`) down. No scalar collapse. | The fix (bug #7 entry point) |
| `atom/models/deepseek_v4.py:1208-1211` (stale comment) | Comment claims hash routing "NOT yet wired" | Update comment to match actual implementation at `:1304-1329`. | Doc cleanup (audit pass 1) |
| `atom/model_ops/sparse_attn_v4.py` | Sparse mask uses externally-set tensors | Per-request masks via `indexer_block_table` | The fix |
| `tests/audit/spec_alignment_with_vllm.py` | (not present) | **NEW**: vendor a pinned vLLM-spec snapshot (`tests/audit/_vllm_spec_snapshot.py`) committed in tree; audit diffs against snapshot, NOT live network fetch. Snapshot refresh is a separate scheduled job. | Hard contract enforcement without network-dependent CI (Codex pass 2) |
| `tests/test_deepseek_v4_multireq.py` | (not present) | **NEW**: surgical test list (§9.5) | Acceptance gate |

**Estimated total**: ~2,000 LOC change (~600 new spec/pool code, ~700 model file rewiring with vector positions, ~500 engine_core / config / BlockManager / Sequence / ForwardContext / ScheduledBatch extensions, ~200 tests).

#### 6.2.4 Canonical logical cache name scheme (Codex final pass)

`Sequence.block_tables` is a `dict[str, list[int]]` keyed by **logical cache name**. The implementation must use a single canonical naming scheme so audit tests, prefix-cache hash, debugging, and tracing are consistent.

**Format**: `layer.{layer_id}.{module}.{cache_kind}` (dot-delimited, lowercase, no spaces).

| Component | Allowed values | Notes |
|---|---|---|
| `layer.{layer_id}` | `layer.0` … `layer.{N-1}` for N transformer layers; `head` for `ParallelHead`; `mtp.{i}` for MTP blocks (out of scope here, reserved) | one prefix per stateful block |
| `{module}` | `attn` (DeepseekV4Attention) · `compressor` (Compressor used by Attention or Indexer) · `indexer` (Indexer) | mirrors module attribute name |
| `{cache_kind}` | `main_c4` · `main_c128` · `main_dense` · `compressor_c4` · `compressor_c128` · `indexer_c4` | encodes `compress_ratio` (0/4/128); `dense` = uncompressed standard MLA |

**Examples** (4-layer DSV4 toy, layers 0/1/2 use c4 with hash routing, layer 3 uses dense):

```text
layer.0.attn.main_c4              ← MQA KV with compress_ratio=4
layer.0.attn.compressor_c4        ← attention's own compressor state
layer.0.attn.indexer_c4           ← indexer's KV (sparse-attn topk source)
layer.0.attn.indexer.compressor_c4 ← indexer's own compressor (rotated FP4 sim)
layer.1.attn.main_c4
layer.1.attn.compressor_c4
layer.1.attn.indexer_c4
layer.1.attn.indexer.compressor_c4
layer.2.attn.main_c4
layer.2.attn.compressor_c4
layer.2.attn.indexer_c4
layer.2.attn.indexer.compressor_c4
layer.3.attn.main_dense           ← uncompressed
```

**Rules**:
- Physical pool coalescence is invisible to consumers — they always address logical names. `BlockManager.allocate(seq)` walks the model to gather logical names, looks up physical pool per `KVCacheSpec`, allocates blocks, and writes them back to `seq.block_tables[logical_name]`.
- Prefix-cache hash key is `(logical_name, prefix_hash)` per §6.2.1.
- Audit test `tests/audit/test_dsv4_logical_cache_naming.py` (new) enforces: every layer's `get_kv_cache_spec()` returns a name matching this regex `^(layer\.\d+|head|mtp\.\d+)\.(attn|compressor|indexer)(\.compressor)?\.(main_(c4|c128|dense)|compressor_(c4|c128)|indexer_c4)$`.

This scheme survives renames, supports `mtp.*` extensions later, and makes debugging trivially greppable.

#### 6.2.2 Open questions
- **Q6.2.1 — Block table shape (RESOLVED → dict)**: Codex's recommendation applied: `Sequence.block_tables: dict[str, list[int]]`, with `Sequence.block_table` kept as a backward-compatible property alias for `block_tables["main"]`. Reasoning: DSV4 already has 3+ pools; sibling fields will multiply linearly with each new hybrid model.
- **Q6.2.2 — vLLM naming strictness (RESOLVED → hard contract)**: spec/metadata field names are a hard contract enforced by `tests/audit/spec_alignment_with_vllm.py` (new file). Failures block CI.
- **Q6.2.3 — Default block size (RESOLVED → 256, override gated)**: default 256 native tokens. Config override allowed only with explicit validation that `compress_ratio` divides `block_size` evenly. Not a tuning knob for first landing.

---

### 6.3 Option C — Full vLLM upstream adoption · ⚠ NON-VIABLE for this submission

**Idea**: stop maintaining a separate `atom/models/deepseek_v4.py`. Lift vLLM model + attention + indexer files wholesale.

**Why non-viable here**:
- Loses 13-commit set of ROCm/MoE-specific PR#650 fixes (weights mapper auto-read, wo_a FP8 dequant, hash route_scale, swiglu_limit, shared expert reduce, KV pollution reset)
- Inherits open vLLM ROCm DSV4 blocker chain (PR#40871 has `mul_cuda not impl for Float8_e8m0fnu` on MI325X)
- Strategic shift orthogonal to this RFC ("ATOM becomes vLLM with AITER kernels?")

**Open question Q6.3.1** (deferred to a separate strategy doc): does ATOM stop being an inference engine and become a "vLLM with ROCm/AITER kernels" distribution? **Out of scope for this RFC.**

---

### 6.4 Option summary

| | A · Tactical slot mapper | **B · Spec contract (default)** | C · Full vLLM adoption |
|---|---|---|---|
| **Status** | DEMO ONLY | **★ chosen** | NON-VIABLE this submission |
| **Engineering size** | ~300 LOC | **~1,800 LOC** | ~3,000+ LOC + strategy shift |
| **vLLM portability** | Low | **High (hard contract)** | N/A — already vLLM |
| **Foundation for fast perf options** | Limited | **High (P1/P2/P3 swappable)** | Medium (vLLM stack only) |
| **Carries PR#650's ROCm-specific MoE fixes** | Yes | **Yes** | Risky |
| **Multi-request correctness** | Yes (private slot path) | **Yes (paged)** | Yes |
| **Prefix caching** | No | **Yes** | Yes |
| **CUDAGraph compatibility** | Possible | **Designed for** | Inherited |
| **Time to first multi-request demo** | 1–2 days | **5–7 days** | 2–4 weeks |
| **Time to InferenceX-ready submission** | N/A (demo only) | **2–3 weeks** | 4–6 weeks |

---

## 7. Performance backend

### 7.1 P1 — vLLM-style flat `SlidingWindowMLASpec` · ★ DEFAULT for correctness PR

- Compressor pool allocates `sliding_window = coff * compress_ratio` slots per page, flat.
- **Pros**: dead simple; matches vLLM exactly; preserves prefix cache, disagg, CUDAGraph by construction; first-time-correct paged allocator.
- **Cons**: VRAM footprint = full `coff * compress_ratio * num_blocks`. SGLang has shown we can do less, but not "for free."
- **Decision**: ship P1 with the correctness PR. Numbers from P1 become the perf baseline against which P2 is justified.

### 7.2 P2 — SGLang `KVAndScore` + ring-buffer pool · perf follow-up RFC, gated on benchmark
- `KVAndScore`: KV and score in a single tensor, last-dim packed, atomic `clear()`.
- Ring buffer: `state_loc = swa_pages * ring_size + (swa_loc % ring_size)` — only `ring_size` slots per page.
- **Pros**: smaller VRAM, faster `clear()`.
- **Cons (Codex applied)**: NOT a free backend swap with B. Changes ownership/recovery semantics. To preserve prefix cache + disagg + CUDAGraph parity, the ring-buffer translation must align with main KV block boundaries, and the prefix-cache recovery path must understand the ring-slot ↔ hash mapping. **Treat P2 as a separate perf RFC.**
- **Gating criteria**: P2 only opens after P1 lands and the benchmark identifies compressor pool VRAM as a bottleneck.

### 7.3 P3 — Hybrid env-flag selector · later
- Add backend interface in P1 if cheap (likely free, since spec → backend mapping is one indirection); ship only `vllm` backend first.
- Not a goal for the correctness PR.

### 7.4 Open questions (axis 2)
- **Q7.1 — ring_size derivation**: SGLang's `CompressStatePool.__init__` derives ring_size from `swa_page_size + ring_size` runtime config. We'd need to read DSV4's HF config to verify. *Resolved: defer to P2 RFC; not blocking P1.*
- **Q7.2 — VRAM delta P1 vs P2**: needs measurement. *Resolved: measure P1 first, then justify P2 against P1's numbers.*
- **Q7.3 — `KVAndScore` × FP8/FP4 quant**: KV-half FP8/FP4 vs score-half FP32 in a single tensor needs ATOM quant wrapper audit. *Resolved: blocks any P2 attempt; explicit prereq.*
- **Q7.4 — `hisparse_coordinator` adoption**: open, defer.

---

## 8. Cross-cutting open questions — resolved to defaults

> v0.1 left these open. v0.2 sets default answers (Codex review applied). Items marked OPEN remain for impl doc.

- **Q8.1 — `start_pos==0` reset semantics (RESOLVED → write-before-read invariant)**: remove model-level `start_pos==0` global clears from the real path. Slot/request freshness is ensured by the **write-before-read invariant** at the slot level: `BlockManager.allocate(seq)` returns block ids whose underlying memory may be stale; the model layer is responsible for writing every slot it later reads (standard vLLM-style paged invariant). `Block.reset()` only clears metadata (`ref_count`, `hash`, `token_ids`); the KV memory itself is **not** zero-init on alloc, by design (mirrors vLLM `KVCacheManager`). The RFC must therefore also guarantee no read-before-write paths in the model layer — **§9.5 test_warmup_pollution gates this**.
- **Q8.2 — CUDAGraph compatibility (RESOLVED → decode capture path designed for it; prefill path not gated)**: full CUDAGraph capture is not the first correctness PR's goal. **The decode capture path** uses fixed-shape metadata buffers and avoids request-dependent Python branching; prefill paths may retain Python control flow. Design constraints preserved from day 1: pre-allocated metadata tensors sized to `max_num_seqs`; per-pool slot mapping/block table tensors zero-padded to fixed shape; no Python int comparisons in decode forward. Prefill correctness is gated independently and may use eager-only paths.
- **Q8.3 — DSV4 metadata builder (RESOLVED → extend, not fork)**: extend `attn_metadata_builder` with a DSV4-aware sub-builder, but keep common builder primitives reusable. No DSV4-specific giant builder.
- **Q8.4 — Prefix caching for compressor pool (RESOLVED → align to vLLM 256-token boundary)**: compressor state is treated as SWA-style KV registered under the compressor pool's KVCacheSpec; prefix-cache hash boundary is the same 256 native-token grid as main KV. Codex's recommendation: "treat compressor state like SWA state, not a separate ad hoc side buffer."
- **Q8.5 — Pool ownership (RESOLVED → single BlockManager coordinator, multi-pool registration)**: `BlockManager` extends to manage multiple `BlockPool` instances, one per `KVCacheSpec` group. No per-pool managers unless perf data demands. Mirrors vLLM `KVCacheManager`.
- **Q8.6 — vLLM upstream sync cadence (RESOLVED → pinned commit)**: pin to a specific vLLM commit at impl-doc time; manual re-audit before each ATOM minor release. No floating "track upstream" promise. The pinned commit is the reference for `tests/audit/spec_alignment_with_vllm.py`.
- **Q8.7 — Test gate (RESOLVED → see §9.5 surgical list)**: greedy sequential-vs-batched token equality + boundary tests + slot reuse + warmup pollution + prefix hit. FP8/FP4 paths use tensor tolerance, not bit-exact logits.
- **Q8.8 — aiter version pin (OPEN)**: PR#2916 needs to release. Strategy: pin to a tag if available, otherwise commit hash with comment justifying. Defer to release coordination.

---

## 9. Out of scope

- MTP head integration (PR5; 43 unloaded params)
- AITER native sparse_attn kernel (PR4)
- Full CUDAGraph capture removal of `--enforce-eager` (PR6) — though §8.2 design constraints honored from day 1
- OAI server wiring (PR6)
- Disaggregated prefill / mooncake KV transfer integration (separate RFC)
- vLLM upstream contribution-back of ATOM's six ROCm-specific MoE fixes
- ATOM ↔ vLLM strategic merger (§6.3 Q6.3.1)

---

## 9.5 Test coverage plan — explicit

> v0.1 had no test plan. v0.2 added 11 surgical system tests. **v0.2.1 expands this into a 5-layer pyramid** with explicit file-level coverage targets, mock strategy, and CI integration. ATOM's existing test convention (`tests/conftest.py` mocks AITER + `torch.cuda` for CPU-only fast tests; class-grouped pytest cases per `tests/test_block_manager.py`) is preserved.

### 9.5.1 Test pyramid

```
                 ┌────────────────────────────────────────────┐
   Layer 5       │ Perf regression (nightly, GPU, ~2h)        │   ~3 tests
                 └────────────────────────────────────────────┘
              ┌─────────────────────────────────────────────────┐
   Layer 4    │ Audit / contract (every PR, CPU, ~10s)         │   ~5 tests
              └─────────────────────────────────────────────────┘
           ┌────────────────────────────────────────────────────┐
   Layer 3 │ System (pre-merge + nightly, GPU + real ckpt, ~30m)│   ~12 tests
           └────────────────────────────────────────────────────┘
        ┌──────────────────────────────────────────────────────┐
   Layer 2│ Integration (every PR, GPU optional w/ mock, ~5m)  │   ~15 tests
        └──────────────────────────────────────────────────────┘
     ┌────────────────────────────────────────────────────────┐
   L1│ Unit (every PR, CPU only, mocked, ~30s)               │   ~60 tests
     └────────────────────────────────────────────────────────┘
```

### 9.5.2 Layer 1 — Unit tests (CPU only, mocked, every PR, runtime budget ~30s)

Pure logic tests against new/changed Python modules. **Must run without GPU, without real model weights, with AITER stubbed via `tests/conftest.py`.**

| Test file | Target SUT | Coverage targets | Approx tests |
|---|---|---|---|
| `tests/test_kv_cache_interface.py` (new) | `atom/v1/kv_cache_interface.py` | `KVCacheSpec` / `AttentionSpec` / `MLAAttentionSpec` / `SlidingWindowMLASpec` / `FullAttentionSpec` dataclass field validation; `MLAAttentionSpec.storage_block_size` derivation (block_size=256 ratio=4 → 64; ratio=128 → 2; ratio=1 → 256); equality + hashability for pool grouping; merge-spec semantics for same-pool layers; `__post_init__` rejection of invalid combinations (compress_ratio not dividing block_size; sliding_window not multiple of compress_ratio) | ~12 |
| `tests/test_block_manager_multipool.py` (new) | `atom/model_engine/block_manager.py` (extended) | Multi-spec registration `BlockManager({"main": MLAAttentionSpec(...), "compress": SlidingWindowMLASpec(...), "indexer": MLAAttentionSpec(...)})`; per-pool `free_block_ids` deque independence; `can_allocate(seq)` returns True only if ALL pools satisfy; `allocate(seq)` is atomic — partial-pool failure → rollback no leak; `free(seq)` releases blocks across all pools; per-pool prefix-cache scope (block hash collision across pools must NOT share); `Block.reset()` only clears metadata, KV memory persists (audit test using torch.zeros pattern); pool-key derivation from spec hash | ~14 |
| `tests/test_block_manager.py` (extend existing) | `atom/model_engine/block_manager.py` | Backwards-compat for non-DSV4 single-pool: existing test cases pass with `BlockManager({"main": MLAAttentionSpec(compress_ratio=1)})`; default `Sequence.block_table` property still works | ~3 added |
| `tests/test_sequence_block_tables.py` (new) | `atom/model_engine/sequence.py` | `Sequence.block_tables: dict[str, list[int]]` defaults; `Sequence.block_table` property aliasing `block_tables["main"]` (read + write); `mamba_block_table` precedent unchanged for non-DSV4; type stability under serialization | ~6 |
| `tests/test_scheduler_multipool.py` (new) | `atom/model_engine/scheduler.py` `ScheduledBatch` | `ScheduledBatch` carries per-pool `block_tables: dict[str, Tensor]` + per-pool `slot_mapping: dict[str, Tensor]`; legacy `block_table` / `slot_mapping` fields still readable via dict["main"] alias; `Scheduler.schedule()` admits when ALL pools allocate; preemption returns blocks to free lists across all pools; mixed prefill/decode batch metadata correctness | ~8 |
| `tests/test_forward_context_dsv4.py` (new) | `atom/utils/forward_context.py`, `model_runner.py:attn_metadata_builder` | Per-pool fields populated; DSV4 metadata builder produces correct slot mapping per logical cache. **Three distinct mapping semantics** (Codex pass 2 — NOT one shared invariant): (a) **compressed-KV mapping** (main `c4`/`c128` + indexer KV): every `compress_ratio`-th token gets a valid slot, rest = -1, `storage_block_size = block_size // compress_ratio`; (b) **compressor-state SWA mapping** (sliding-window state): every token contributes to the rolling window; slot derives from `pos % sliding_window` projected through the compressor's pool key; (c) **main-KV mapping** (uncompressed paths): one slot per token, no `-1`. Each tested separately. `block_table_tensor[B, max_blocks]` shape/dtype; `storage_block_size` boundary crossing (mirror of vLLM's slot mapping test, cited §4.1). Vector-position consumption: `cu_seqlens_q` / `token_to_seq_idxs` plumbed correctly. | ~12 |
| `tests/test_dsv4_specs.py` (new) | `atom/models/deepseek_v4.py` | `Compressor.get_kv_cache_spec(config)` returns `SlidingWindowMLASpec(sliding_window=coff*compress_ratio)` with correct fields; `Indexer.get_kv_cache_spec()` returns `MLAAttentionSpec(compress_ratio=4)`; `DeepseekV4Attention.get_kv_cache_spec()` returns layer-correct spec (compress_ratio = `args.compress_ratios[layer_id]`); spec is invariant across multiple instantiations of the same layer | ~8 |

**Layer 1 entry criterion**: every PR. Failure blocks merge. Runtime ≤ 30s on a developer laptop without GPU.

### 9.5.3 Layer 2 — Integration tests (GPU optional with mock fallback, every PR, ~5min)

Module boundary correctness with stand-in tensors but real torch ops. Use `pytest.mark.gpu` for paths needing real CUDA; CI runs both mocked + GPU variants. **Toy 4-layer DSV4 fixture in `tests/fixtures/dsv4_toy.py`** (synthesized weights, BF16 only, no quant — bit-exact reference parity preserved).

| Test file | Target SUT | Coverage targets | Approx tests |
|---|---|---|---|
| `tests/test_dsv4_compressor_paged.py` (new) | `Compressor.forward` rewired | Read/write to externally-allocated pool tensor via `compress_slot_mapping`; correct slot for given (block_id, pos_in_block); per-row `start_pos` independence (no global reset); two concurrent slot-ids produce independent state evolution; output unchanged from PR#650 baseline on 4-layer toy single-seq | ~7 |
| `tests/test_dsv4_indexer_paged.py` (new) | `Indexer.forward` rewired | `[:1]` hardcode at `:740` removed and replaced with `[:bsz]` indexed via `indexer_block_table`; topk indices correctness vs PR#650 baseline; sparse mask per-request (concurrent requests get distinct masks) | ~5 |
| `tests/test_dsv4_attention_paged.py` (new) | `DeepseekV4Attention.forward` rewired | Main MQA KV write via `slot_mapping`; `block_table_tensor` consumed; FP8 quant path preserved; output matches PR#650 single-seq for slot=0 | ~5 |
| `tests/test_dsv4_block_lifecycle.py` (new) | End-to-end alloc/free across all pools | Sequence enter → all pools allocate → forward writes → finish → all pools free → next sequence inherits clean blocks (no read-before-write); ref-count correctness for shared prefix blocks across pools | ~4 |
| `tests/test_dsv4_metadata_builder.py` (new) | `attn_metadata_builder` for DSV4 | Common builder primitives reused; DSV4-aware sub-builder produces all 3 pool slot_mappings; mixed-pool consistency (same query token gets same logical pos across all pools' slot_mappings) | ~5 |

**Layer 2 entry criterion**: every PR. CPU-mock variant must pass without GPU; GPU variant runs on CI agents with MI hardware. Runtime ≤ 5min.

### 9.5.4 Layer 3 — System tests (pre-merge + nightly, GPU + real ckpt, ~30min)

The 11 surgical tests from v0.2 §9.5, made explicit here. Run against `/data/DeepSeek-V4-Pro` at TP=8 with `ATOM_USE_TRITON_MOE=1`. **Token-ID equality for greedy `temperature=0.0`**; tensor tolerance for FP8/FP4 paths.

| Test | File | Equality criterion | Pre-merge / Nightly |
|---|---|---|---|
| `test_dsv4_greedy_seq_vs_batch` | `tests/test_dsv4_multireq.py` | Token-ID equality across N=1,2,4,8 batched runs vs sequential single-seq | **Pre-merge** |
| `test_dsv4_mixed_prefill_decode` | same | Token-ID equality on mid-flight request when new prefill arrives in same step | **Pre-merge** |
| `test_dsv4_slot_reuse` | same | Finish then reuse — Token-ID equality vs fresh-slot run | **Pre-merge** |
| `test_dsv4_preemption` | `tests/test_dsv4_scheduler.py` | Resume after preemption matches uninterrupted run, Token-ID equality | Nightly |
| `test_dsv4_prefix_cache_hit` | `tests/test_dsv4_prefix.py` | Cache hit rate ≥ expected; outputs match cold run | Nightly |
| `test_dsv4_c4_boundary` | `tests/test_dsv4_compression.py` | Hidden state `allclose(rtol=1e-2, atol=1e-2)` at exact 64-token c4 boundary crossing | **Pre-merge** |
| `test_dsv4_c128_boundary` | same | Same at exact 2-token c128 boundary | **Pre-merge** |
| `test_dsv4_start_pos_zero_during_decode` | `tests/test_dsv4_multireq.py` | Mid-flight request A's tokens survive when B enters prefill | **Pre-merge** |
| `test_dsv4_warmup_pollution` | `tests/test_dsv4_warmup.py` | First real request after warmup matches cold-start single-seq | **Pre-merge** |
| `test_dsv4_long_context_1m` | `tests/test_dsv4_long_ctx.py` | 1M token context, 8 concurrent — no OOM, coherent output | Nightly |
| `test_dsv4_24h_soak` | `recipes/dsv4_soak.sh` (CI nightly) | 24h conc=8 — no OOM / no garbled output / no token leak | Nightly |
| `test_dsv4_lm_eval_acc` | `tests/test_dsv4_eval.py` (gsm8k subset) | Accuracy within 0.5pt of PR#650 single-seq baseline | Nightly |

**Layer 3 entry criterion**: pre-merge tests block merge. Nightly tests block release.

### 9.5.5 Layer 4 — Audit / contract tests (every PR, CPU, ~10s)

Catch ATOM↔vLLM contract drift early.

| Test file | Target | Coverage targets | Approx tests |
|---|---|---|---|
| `tests/audit/test_spec_alignment_with_vllm.py` (new) | ATOM specs vs **vendored** pinned vLLM-spec snapshot | Field-name set equality on `MLAAttentionSpec`, `SlidingWindowMLASpec`, `FullAttentionSpec`; field type equality (annotations); default value equality (block_size=256 etc.); subclass relationship; fails if ATOM specs drift from the snapshot | ~5 |
| `tests/audit/_vllm_spec_snapshot.py` (new) | Vendored snapshot of pinned vLLM `kv_cache_interface.py` spec definitions | One-file copy of relevant vLLM dataclasses at `_VLLM_AUDIT_COMMIT`; **no network fetch on PR runs**. Snapshot refresh runs as a separate scheduled job (e.g., weekly cron). | data file |
| `tests/audit/test_dsv4_no_module_state.py` (new) | `atom/models/deepseek_v4.py` | Static AST audit: zero `register_buffer` for `score_state` / `kv_state` / `kv_cache` in `Compressor` / `Indexer` / `DeepseekV4Attention`; zero `start_pos == 0` reset patterns; zero `[:1, ...]` hardcoded indexing; **zero `int(positions[0].item())` scalar collapse** at the model wrapper | ~4 |
| `tests/audit/test_dsv4_vector_positions.py` (new) ⚠ NEW (Codex pass 2) | `atom/models/deepseek_v4.py:1929` and downstream | AST + runtime audit: model wrapper passes vector `positions` + `attn_metadata.cu_seqlens_q` + `attn_metadata.token_to_seq_idxs` down; no scalar `start_pos: int` parameter on Compressor / Indexer / Attention forward signatures | ~3 |

> **vLLM commit pinning** (Q8.6 resolution, Codex pass 2 refined): `atom/v1/kv_cache_interface.py` carries a `_VLLM_AUDIT_COMMIT = "<sha>"` constant. The audit diffs against a **vendored snapshot** at `tests/audit/_vllm_spec_snapshot.py` — committed in tree, no network fetch on PR runs. A separate scheduled job (weekly cron, separate from per-PR CI) refreshes the snapshot via a PR for human review. **Per-PR CI is fully offline.**

### 9.5.6 Layer 5 — Performance regression (nightly, GPU, ~2h)

Guardrails — must not silently regress single-seq perf while enabling multi-request.

| Test file | Target | Pass criterion |
|---|---|---|
| `tests/perf/test_dsv4_single_seq_no_regress.py` (new) | Single-seq decode TPOT | Within 5% of PR#650 baseline at TP=8 on `/data/DeepSeek-V4-Pro` |
| `tests/perf/test_dsv4_multireq_scaling.py` (new) | N-seq throughput scaling | ≥3.5× single-seq throughput at conc=4 (lockstep) |
| `tests/perf/test_dsv4_compressor_pool_footprint.py` (new) | Compressor pool VRAM | Within 1.2× theoretical `coff * compress_ratio * num_blocks` (validates flat P1 not blowing budget; baseline for justifying P2) |

### 9.5.7 Mock strategy

Per ATOM convention (`tests/conftest.py`):
- Layer 1 / Layer 4: **always mocked**. AITER stubbed via `_atom_pkg` redirection in conftest. `torch.cuda.*` stubs via `unittest.mock.MagicMock`. No real GPU touched.
- Layer 2: **dual-mode**. Default mocked path runs in CI on CPU agents. `pytest.mark.gpu` variants run on MI hardware nightly. Toy DSV4 fixture (`tests/fixtures/dsv4_toy.py`) provides 4-layer synthesized weights.
- Layer 3 / Layer 5: **real ckpt + real GPU**. `/data/DeepSeek-V4-Pro` mounted on CI agents. `pytest.mark.real_ckpt` marker.

### 9.5.8 CI integration & runtime budget

| Layer | When | Runtime | Blocks |
|---|---|---|---|
| L1 — Unit | Every PR | ≤ 30s | merge |
| L2 — Integration (mocked) | Every PR | ≤ 2min | merge |
| L2 — Integration (GPU) | Every PR (post-mocked-pass) | ≤ 5min | merge |
| L3 — System (pre-merge subset) | Pre-merge | ≤ 15min | merge |
| L3 — System (full) | Nightly | ≤ 30min | release |
| L4 — Audit | Every PR | ≤ 10s | merge |
| L5 — Perf regression | Nightly | ≤ 2h | release |

**Total per-PR runtime**: ~25 min wall clock with parallelism. Acceptable for 5–10 PR/day cadence.

### 9.5.9 Equality policy (consolidated)

- **Greedy paths** (`temperature=0.0`): token-ID equality is the strict gate. Any deviation = bug.
- **FP8/FP4 quant paths**: bit-exact logits across batched-vs-sequential are unrealistic due to non-commutative reductions in tile-grouped GEMMs. Use tensor tolerance (`rtol=1e-2`, `atol=1e-2`) on hidden states at well-defined checkpoints (post-RMSNorm, pre-sampler), **plus** token-ID equality at sampler output.
- **Stochastic paths** (`temperature>0`): use accept-rate / KL-divergence over a soak window of ≥10k tokens; not first-PR gate.

### 9.5.10 Coverage targets

| Metric | Target | Tooling |
|---|---|---|
| Line coverage on new files (§6.2.1) | ≥ 90% | `pytest-cov` |
| Branch coverage on `BlockManager` extensions | ≥ 95% | `pytest-cov` |
| Mutation testing on `KVCacheSpec` field validation | ≥ 80% kill rate | `mutmut` (post-Phase A) |

---

## 10. Definition-of-done — split gates (Codex pass 2)

> v0.2.1 had a single DoD that required "all §9.5 tests pass" — but §9.5 includes nightly perf and a 24h soak. Codex pass 2 split the gates by audience.

### 10.1 Correctness PR — merge gate

Required to merge the correctness PR (B/P1):

1. **Tests**: Layer 1 (Unit, every PR), Layer 2 (Integration, every PR), Layer 4 (Audit, every PR) **all pass**, plus the Layer 3 **pre-merge subset** marked in §9.5.4 (`test_dsv4_greedy_seq_vs_batch`, `test_dsv4_mixed_prefill_decode`, `test_dsv4_slot_reuse`, `test_dsv4_c4_boundary`, `test_dsv4_c128_boundary`, `test_dsv4_start_pos_zero_during_decode`, `test_dsv4_warmup_pollution`).
2. **Code hygiene** in `atom/models/deepseek_v4.py`:
   - Zero `kv_cache[:1, ...]` hardcodes (all 6 sites removed: Indexer:740 + Attention:1040,1044,1045,1054,1059)
   - Zero `start_pos == 0` global reset blocks (Compressor:538,610,624 + central reset at :994-1002 removed)
   - Zero module-level `score_state` / `kv_state` / `kv_cache` `register_buffer` declarations
   - Zero scalar `start_pos = int(positions[0].item())` collapse at `:1929` — vector positions throughout
   - Stale comment at `:1208-1211` updated to reflect actual hash routing implementation
3. **API contract**: each stateful layer in `deepseek_v4.py` exposes `get_kv_cache_spec()`.
4. **Block manager**: `BlockManager` accepts `kv_cache_specs: dict[str, KVCacheSpec]` and coalesces into physical pools by `(page_shape, dtype, spec_kind)`. `Sequence.block_tables` is dict-typed. `Config.kv_cache_pool_blocks: dict[str, int]` is populated; `engine_core.py:90,95,99` consumes the dict; `model_runner.get_kv_cache_pool_blocks()` returns it. `num_kvcache_blocks` deprecated alias still resolves correctly for non-DSV4 models.
5. **Audit**: `tests/audit/test_spec_alignment_with_vllm.py` passes against the **vendored** vLLM-spec snapshot at the pinned commit.
6. **Smoke**: `simple_inference.py --max-num-seqs 4 --temperature 0.0` produces correct distinct outputs for 4 prompts running concurrently.
7. **Companion docs updated**:
   - `docs/architecture_guide.md` — DSV4 logical-vs-physical pool diagram
   - `docs/scheduling_kv_cache_guide.md` — `BlockManager` multi-pool API + new `kv_cache_pool_blocks` config field
   - `docs/model_support_guide.md` — DSV4 status flipped from "single-seq only" to "multi-request"

### 10.2 Release / InferenceX submission gate — phased

Required to ship the InferenceX submission (post-correctness PR), in two phases.

#### Phase 2a — TP=8 single-node validation (default starting config)
1. Layer 3 nightly full passes on **MI355X 1 node, TP=8, DP=1, EP=1**
2. Layer 5 perf regression passes; single-seq TPOT ≤ 1.05× PR#650 baseline; conc=4 throughput ≥ 3.5× single-seq
3. CUDAGraph decode capture working (`--enforce-eager` removed for decode path)
4. ISL/OSL coverage on the easier matrix cells (1k/1k, 8k/1k confirmed runnable)

#### Phase 2b — 1P2D × EP=8 disaggregated (InferenceX submission target)
1. Q1.1.A / Q1.1.B / Q1.1.C closed by InferenceX submission lead
2. Multi-node disaggregated prefill working; 1 prefill node + 2 decode nodes; expert-parallel=8 across nodes
3. Submission perf primitives all wired: block_size=256, FP8 KV, FP4 indexer cache, CUDAGraph, AITER sparse attention, MoE EP/all-to-all (MORI), MTP if rules permit
4. **Baseline beat documented** vs **NVIDIA Blackwell B200** at matching matrix cell — daily numbers from [`inferencex.semianalysis.com`](https://inferencex.semianalysis.com)
5. 24h soak at conc=8 (per node) with no OOM, no garbled output, no token leak
6. Live leaderboard submission accepted on InferenceX

### 10.3 What this RFC does NOT close

- **§1.1 release-track items Q1.1.A/B/C** — matrix-cell commitments, framework path (ATOM-native vs SGLang fallback), MTP gate. The §1.1 target table itself is filled (MI355X, TP=8 phase 1, 1P2D EP8 phase 2, B200 baseline). These three sub-items belong to InferenceX submission planning.
- **Q8.8** — aiter version pin strategy (open, deferred to release coordination)
- **Q6.3.1** — ATOM↔vLLM strategic merger (out of scope, separate strategy doc)
- P2/P3 perf backend follow-up (separate perf RFC, gated on P1 baseline)

---

## 11. References

### ATOM (this repo, this branch)
- `lingpeng/dsv4-pr1-skeleton` (= ROCm/ATOM PR [#650](https://github.com/ROCm/ATOM/pull/650), branch `feat/deepseek-v4-pr1-skeleton`)
  - `atom/models/deepseek_v4.py` (2117 LOC) — bug sites at lines 464, 470, 481, 538, 549, 557, 569-601, 610, 624, 627, 682-687, 740
  - `atom/model_ops/{sparse_attn_v4,quant_v4,moe,fused_moe_triton}.py`
  - `atom/model_loader/loader.py`
  - `atom/examples/simple_inference.py`
- ATOM existing infra (preserved):
  - `atom/model_engine/{scheduler,block_manager,sequence,model_runner}.py`
  - `atom/utils/forward_context.py`
  - `docs/{architecture_guide,scheduling_kv_cache_guide,model_support_guide}.md`

### aiter
- `lingpeng/fix-mhc-device` (= ROCm/aiter PR [#2916](https://github.com/ROCm/aiter/pull/2916), branch `fix_mhc_device`) — single commit `fix mhc device`. Required version pin.

### vLLM
- [Blog: DeepSeek V4 in vLLM](https://vllm.ai/blog/deepseek-v4)
- [PR #40760 — Support DeepseekV4](https://github.com/vllm-project/vllm/pull/40760) (zyongye)
- [PR #40871 — ROCm DSV4](https://github.com/vllm-project/vllm/pull/40871) (whx-sjtu / hexwang)
- [PR #40860 — DeepSeek V4 Rebased](https://github.com/vllm-project/vllm/pull/40860) (ivanium)
- [PR #25869 — Support Deepseek V3.2](https://github.com/vllm-project/vllm/pull/25869) (heheda12345)
- Source files reviewed (8): `vllm/model_executor/models/deepseek_v4.py`, `vllm/model_executor/layers/{mhc,deepseek_compressor,deepseek_v4_attention,sparse_attn_indexer}.py`, `vllm/v1/attention/backends/mla/{indexer,sparse_swa,compressor_utils}.py`, `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`, `vllm/v1/kv_cache_interface.py`, `tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py`

### SGLang
- [PR #23600 — DeepSeek V4](https://github.com/sgl-project/sglang/pull/23600) (sgl-project)
- [Issue #23602 — DeepSeek V4 Roadmap](https://github.com/sgl-project/sglang/issues/23602)
- [Issue #23639 — UnifiedRadix HiCache for DSV4](https://github.com/sgl-project/sglang/issues/23639)
- Source files reviewed (8): `python/sglang/srt/models/{deepseek_v4,deepseek_v4_nextn}.py`, `python/sglang/srt/mem_cache/{deepseekv4_memory_pool,compress_state}.py`, `python/sglang/srt/layers/attention/compressed/{compressor,indexer,metadata,paged_prefill}.py`, `python/sglang/srt/managers/hisparse_coordinator.py`

---

## 13. Approvals

| Role | Name | Status | Date |
|---|---|---|---|
| Author | sunway513 | ✅ Author sign-off | 2026-04-25 |
| Reviewer (independent) | Codex | ✅ Pass 1 + Pass 2 review applied | 2026-04-25 |
| Sub-agent audit | Explore (PR#650 source audit) | ✅ MEDIUM-HIGH confidence on Lingpeng's single-seq claim | 2026-04-25 |
| Tech lead | _pending_ | _awaiting_ | — |
| Team approval | _pending_ | _awaiting_ | — |

**Sign-off semantics**: an "✅" against the author and Codex columns means the document is **closed for correctness implementation** (§10.1). Tech-lead and team approval gate handoff to the implementation doc + InferenceX submission planning (§10.2).

---

## 14. Implementation kickoff checklist

When tech-lead approval lands, the implementation engineer can use this as a literal day-1 to-do list.

### Day 1 — environment

- [ ] **Weights**: `bash recipes/pull_dsv4_pro_weights.sh` (run from the ATOM repo root) on a host with ≥1 TiB free at `/data/hf_models/` (default target `/data/hf_models/deepseek-ai/DeepSeek-V4-Pro`). Public HF repo, no token required. (Override targets via env: `DSV4_TARGET=/scratch/... DSV4_REPO=... bash recipes/pull_dsv4_pro_weights.sh`.)
- [ ] **Branches local**:
   - `git clone https://github.com/sunway513/ATOM.git && cd ATOM && git checkout lingpeng/dsv4-pr1-skeleton`
   - `git clone https://github.com/sunway513/aiter.git && cd aiter && git checkout lingpeng/fix-mhc-device`
- [ ] **Smoke**: reproduce single-seq inference on the cloned branches (`ATOM_USE_TRITON_MOE=1 python -m atom.examples.simple_inference --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro --kv_cache_dtype fp8 -tp 8 --max-num-seqs 4 --max-model-len 1024 --enforce-eager --temperature 0.0 --max-tokens 512`) — establishes the baseline before any reform changes.

### Week 1 — foundations
- [ ] Create implementation branch from `lingpeng/dsv4-pr1-skeleton`: e.g. `feat/dsv4-multireq-kvcache-reform`
- [ ] Add `atom/v1/kv_cache_interface.py` (specs) per §6.2.1, with `_VLLM_AUDIT_COMMIT` constant
- [ ] Vendor `tests/audit/_vllm_spec_snapshot.py` from the pinned vLLM commit
- [ ] Land Layer 1 unit tests (§9.5.2): start with `test_kv_cache_interface.py` + `test_dsv4_specs.py` — these are pure-Python and unblock the rest

### Week 2 — block manager + sequence
- [ ] Extend `atom/model_engine/block_manager.py` for multi-pool; write `test_block_manager_multipool.py`
- [ ] `Sequence.block_tables: dict[str, list[int]]` + property alias; write `test_sequence_block_tables.py`
- [ ] `ScheduledBatch` per-pool fields; write `test_scheduler_multipool.py`
- [ ] `Config.kv_cache_pool_blocks: dict[str, int]`; deprecate `num_kvcache_blocks` alias
- [ ] `engine_core.py:90,95,99` consume dict; verify non-DSV4 models still pass via Layer 1+2 mocked tests

### Week 3 — model file rewiring
- [ ] Add `get_kv_cache_spec()` to `Compressor`, `Indexer`, `DeepseekV4Attention` in `atom/models/deepseek_v4.py`
- [ ] Replace 6 `[:1]` hardcodes (lines 740, 1040, 1044, 1045, 1054, 1059) with slot-mapping reads
- [ ] Remove module-level `register_buffer` for `score_state` / `kv_state` / `kv_cache`
- [ ] Remove central reset block at `:994-1002`; write-before-read invariant honored
- [ ] Replace scalar `start_pos` (`:1929` `int(positions[0].item())`) with vector positions; consume `cu_seqlens_q`/`token_to_seq_idxs`
- [ ] **Rewrite topk helpers for vector positions** (Codex final pass): `_get_window_topk_idxs` and `_get_compress_topk_idxs` (in `atom/models/deepseek_v4.py` / `atom/model_ops/sparse_attn_v4.py`) currently assume `B=1` + scalar `start_pos`. Refactor to consume per-token absolute positions (built from `cu_seqlens_q + token_to_seq_idxs`) and produce per-request mask shapes. This is on the critical path of bug #7 — fix before re-running multi-request smoke.
- [ ] Update stale comment at `:1208-1211` to match actual hash routing
- [ ] Land Layer 2 integration tests on toy 4-layer fixture

### Week 4 — system + audit + correctness PR review
- [ ] Land Layer 3 pre-merge subset tests (§9.5.4) on `/data/hf_models/deepseek-ai/DeepSeek-V4-Pro`
- [ ] Land Layer 4 audit tests (§9.5.5) — including `test_dsv4_no_module_state.py`, `test_dsv4_vector_positions.py`
- [ ] Companion docs: `architecture_guide.md`, `scheduling_kv_cache_guide.md`, `model_support_guide.md` per §10.1.7
- [ ] **Correctness PR opens for review** — should hit §10.1 acceptance gate

### Phase 2a — TP=8 single-node validation (post-correctness merge)
- [ ] Layer 3 nightly full + Layer 5 perf regression on MI355X 1 node TP=8
- [ ] CUDAGraph decode capture (drop `--enforce-eager` for decode)
- [ ] InferenceX matrix cells 1k/1k + 8k/1k validated runnable

### Phase 2b — 1P2D × EP=8 disaggregated InferenceX submission
- [ ] InferenceX submission lead closes Q1.1.A / Q1.1.B / Q1.1.C
- [ ] Multi-node disagg prefill working
- [ ] Perf primitives all wired (FP8 KV / FP4 indexer / AITER sparse / MORI / MTP-if-allowed)
- [ ] Baseline beat documented vs B200 InferenceX submission
- [ ] 24h soak passes
- [ ] InferenceX leaderboard submission accepted

---

## 12. Changelog

- **2026-04-25 v0.1**: Initial draft. Surveyed PR#650 / vLLM #40760 / SGLang #23600. Documented bug surface against `lingpeng/dsv4-pr1-skeleton` source. Three reform options (A / B / C) and three performance backend options (P1 / P2 / P3) presented. All major decisions left open.
- **2026-04-25 v0.2**: Codex review applied. Recommended path set to **B/P1** (correctness-first, vLLM-isomorphic). A relabeled DEMO-ONLY; C non-viable for submission window. P2 explicitly noted as not orthogonal to B (compressor-as-SWA-KV semantics preservation argument). New §1.1 InferenceMax target matrix (TBD rows for team). New §6.2.1 concrete file-list with required ATOM machinery changes. New §9.5 surgical test list with 11 named tests. §8.1 Q8.1 reset semantics resolved to "write-before-read invariant" (Codex's "trust block alloc/zero-init" with the precise nuance that ATOM `Block.reset` only clears metadata — KV memory is not re-zeroed, so write-before-read is the actual guarantee, mirroring vLLM). All §6, §7, §8 open questions resolved to defaults except: §1.1 TBD rows, Q8.8 aiter version pin.
- **2026-04-25 v0.2.1**: §9.5 expanded into a 5-layer test pyramid (Unit / Integration / System / Audit / Perf) with explicit file-level coverage, mock strategy, CI integration, runtime budget, equality policy, and coverage targets. ~95 tests total across all layers. Layer 1 (CPU only, mocked, ≤30s) gates every PR; Layer 5 (perf regression, nightly, ≤2h) gates release. Audit layer (§9.5.5) introduces `_VLLM_AUDIT_COMMIT` constant pinning + AST audit for absence of module-level state.
- **2026-04-25 v0.2.2**: Independent source audit of PR#650 applied. §3.1 bug surface table extended: bug source #5 (model-level reset orchestration at `:994-1002`, the central reset point that the comment at `:990-993` admits is a workaround for ATOM warmup pollution); bug source #6 (`DeepseekV4Attention.kv_cache` 5 hardcoded `[:1]` sites at `:1040,1044,1045,1054,1059`). §3.2 hardcode count corrected from "1+" to "6". New §3.4 documents audit verdict (MEDIUM-HIGH confidence on Lingpeng's single-seq claim, with full caveat list including the stale `:1208-1211` comment requiring cleanup in the implementation PR).
- **2026-04-25 v0.2.3**: Lingpeng's branches mirrored to user's forks for collaborator access — `sunway513/ATOM:lingpeng/dsv4-pr1-skeleton` (@ `cdbff359`) and `sunway513/aiter:lingpeng/fix-mhc-device` (@ `76ea1ed5`). Frontmatter and a new "Quickstart for collaborators" subsection wire this in. No content changes.
- **2026-04-25 v0.2.4**: Codex second-pass review applied. (1) §3.1 bug source #7 added: scalar-position collapse at `deepseek_v4.py:1929` — `start_pos = int(positions[0].item())` is wrong for any multi-seq batch even after KV slots are isolated. Fix: consume existing `cu_seqlens_q` / `cu_seqlens_k` / `token_to_seq_idxs` from `attn_metadata` (already populated by `attention_mla.py:441-555`). (2) §6.2.1 reframed: logical-vs-physical pool model — layers register many logical caches (8+), `BlockManager` coalesces into physical `BlockPool`s by `(page_shape, dtype, spec_kind)`. DoD speaks logical, not "main_kv/compress/indexer." (3) §6.2.1 file list extended with `engine_core.py:90,95,99` and `config.py:818` (the scalar-pipe now exposed end-to-end); new `kv_cache_pool_blocks: dict[str, int]` API; legacy `num_kvcache_blocks` kept as deprecated alias. (4) §9.5.5 audit no longer depends on network: vendored snapshot `tests/audit/_vllm_spec_snapshot.py` committed in tree; weekly cron refresh job is separate from per-PR CI. (5) §10 split into two gates: §10.1 correctness PR (L1/L2/L4 + L3 pre-merge subset) vs §10.2 InferenceMax release (L3 nightly full + L5 perf + §1.1 TBDs filled with owners). (6) §9.5.3 `test_forward_context_dsv4` invariant split into 3 distinct semantics: compressed-KV mapping, compressor-state SWA mapping, main-KV mapping — not one shared rule. (7) §8.2 CUDAGraph wording softened: "decode capture path uses fixed-shape metadata"; prefill paths may stay eager. (8) §1.1 TBDs annotated with owner+date placeholders. (9) Status downgraded from "bug surface complete" to "bug surface audit-expanded (2 passes), implementation-ready when §1.1 TBDs filled." Verified: all Codex findings grep-confirmed against `lingpeng/dsv4-pr1-skeleton` source (`engine_core.py:90,95,99`, `config.py:818 num_kvcache_blocks: int = -1`, `deepseek_v4.py:1929`, `attention_mla.py:441-555 cu_seqlens / token_to_seq_idxs`).
- **2026-04-25 v0.2.5 (CLOSED for correctness implementation)**: §1.1 TBDs filled against InferenceX (formerly InferenceMax) submission framework. Hardware locked to **MI355X** (user input). Phase 1 default = **TP=8 × DP=1 × EP=1** single-node (per user); Phase 2 InferenceX submission target = **1P2D × EP=8** disaggregated. Baseline-to-beat = **NVIDIA Blackwell B200** InferenceX submission at matching matrix cell. Live tracking via `inferencex.semianalysis.com` and `SemiAnalysisAI/InferenceX`. New §0 Decision Record at the top with all locked decisions in one table. New §13 Approvals (author + Codex passes signed; tech-lead/team pending). New §14 Implementation Kickoff Checklist (Day-1 → Week-4 → Phase 2a/2b literal to-do list). §10.2 split into Phase 2a (TP=8 single-node) and Phase 2b (1P2D EP8 disagg). New `recipes/pull_dsv4_pro_weights.sh` (idempotent + resume-safe HF download recipe; verified DSV4-Pro is publicly accessible without HF token, started download to `/data/hf_models/deepseek-ai/DeepSeek-V4-Pro` on `mi355-gpu-15`). Items intentionally still open: §1.1 release-track Q1.1.A/B/C (owned by InferenceX submission lead, gates §10.2 only), Q8.8 aiter version pin (release coordination), Q6.3.1 ATOM↔vLLM strategic merger (separate strategy doc).
- **2026-04-25 v0.2.6 (Codex final-pass approved)**: 3 must-fix + 5 cleanups applied. **Must-fix**: (1) §6.2.1 prefix-cache hash namespace scoped to `(logical_cache_name, prefix_hash)` tuple, NOT physical pool key — prevents silent block reuse across unrelated logical caches that happen to share a coalesced physical pool. `Block.logical_cache_name` field added; `BlockManager.hash_to_block_id` retyped to `dict[tuple[str, int], int]`. (2) New §6.2.4 — canonical logical cache name scheme: `layer.{i}.{module}.{cache_kind}` regex enforced by new `tests/audit/test_dsv4_logical_cache_naming.py`. Naming covers main MQA KV, compressor state, indexer KV, indexer compressor, and dense layers — 13 logical caches per attention block surveyed in the example. (3) §14 Week 3 — explicit topk helper rewrite (`_get_window_topk_idxs` / `_get_compress_topk_idxs`) added to checklist; these are B=1/scalar-start_pos shaped today and are on the critical path of bug #7. **Cleanups**: (4) §10.3 reworded — §1.1 target table itself is filled; only Q1.1.A/B/C release-track sub-items remain open. (5) "InferenceMax" replaced with "InferenceX (formerly InferenceMax)" everywhere in body text; older changelog entries left intact for trace. (6) §14 Day 1 recipe path now repo-relative `bash recipes/pull_dsv4_pro_weights.sh` instead of absolute. (7) Branch `rfc/dsv4-kvcache-reform` committed and pushed to `sunway513/ATOM` so collaborators can fetch the RFC + recipe from GH. (8) Issue title flipped from "open for discussion" to "closed for correctness implementation."
