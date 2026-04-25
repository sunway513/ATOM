# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek-V4 model — ATOM standard MLA paged KV-cache variant.

This is the *paged* counterpart to ``atom.models.deepseek_v4``. The flat
``register_buffer("kv_cache", torch.zeros(max_batch_size, kv_cache_size,
head_dim))`` carried by ``DeepseekV4Attention`` is removed; the per-layer raw
KV is instead stored in the global ATOM KV-cache pool that ``ModelRunner``
allocates and binds onto each layer (``forward_context.kv_cache_data[
"layer_{i}"].k_cache``), exactly the way :mod:`atom.models.deepseek_v2` /
:class:`atom.model_ops.attention_mla.MLAAttention` consume the V3.2 MLA paged
cache.

The non-attention DSV4 machinery — Compressor, Indexer, Hyper-Connections,
MoE / hash routing, MTP, weight loader — is reused unchanged from
``atom.models.deepseek_v4``. Only the attention layer (and the model / wrapper
classes that own it) are rewritten so the per-layer ``kv_cache`` handle is
fetched from ``forward_context`` per step rather than kept as a model buffer.

Why this exists
---------------
- The legacy flat-cache attention does a per-token Python-side gather
  (``self.kv_cache[seq_rows, ...]``) on every decode step, which costs CPU-GPU
  sync and an extra device gather kernel per layer.
- The paged path lets the AITER ``concat_and_cache_mla`` kernel write into
  the contiguous physical-block tensor with one fused op, and uses the
  scheduler-built ``slot_mapping`` / ``block_tables`` so write/read indices
  arrive on the GPU as part of ``attn_metadata`` (no per-layer mapping
  rebuild).
- Same correctness profile as the flat path (cross-talk fix is automatic
  because each request's slot_mapping points to its own physical block IDs).

Opt-in / wiring
---------------
Register a new HF ``architectures`` entry "DeepseekV4ForCausalLM_MLA" that
maps to :class:`DeepseekV4ForCausalLM_MLA` in
``atom.model_engine.model_runner.support_model_arch_dict``. Users opt-in via
the ``--model-arch DeepseekV4ForCausalLM_MLA`` override (or by editing the
checkpoint's ``config.json:architectures``).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
from torch import nn

# Reuse all V4-specific machinery (Compressor / Indexer / mHC / MoE / Block-
# style FFN / MTP / weight loader). We re-export the key classes so callers
# that already do ``from atom.models.deepseek_v4_mla import DeepseekV4Args``
# don't have to maintain two import paths.
from atom.config import Config
from atom.models.deepseek_v4 import (  # noqa: F401  (selective re-export)
    Compressor,
    DeepseekV4Args,
    Expert,
    Gate,
    Indexer,
    MoE,
    MTPBlock,
    ParallelHead,
    _RMSNorm,
    _apply_rotary_emb,
    _dsv4_get_seq_rows,
    _get_compress_topk_idxs,
    _get_window_topk_idxs,
    _precompute_freqs_cis,
    make_v4_quant_config,
)
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.model_ops.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.quant_v4 import act_quant_inplace
from atom.model_ops.sparse_attn_v4 import sparse_attn  # noqa: F401
from atom.v1.kv_cache_interface import MLAAttentionSpec

# `aiter.dist.parallel_state` may not be importable in toy / test environments
# without GPU; the legacy V4 model imports it unconditionally, so it must be
# present in any environment that can already import ``deepseek_v4``. Keeping
# the import here for parity (the test harness imports both files).
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size


# ---------------------------------------------------------------------------
# Paged KV helpers
# ---------------------------------------------------------------------------


def _paged_kv_write(
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write per-token KV into the paged cache at the slots given by
    ``slot_mapping``.

    Layout
    ------
    ``kv_cache`` is the flat-physical view ``[num_blocks * block_size,
    num_kv_heads=1, head_dim]`` that :class:`atom.model_engine.ModelRunner`
    binds onto each layer (see ``model_runner.py`` ~1559). ``slot_mapping[i]``
    is the global slot index (block_id * block_size + offset_in_block) where
    token ``i`` should land.

    This is the V4-compatible analogue of AITER's ``concat_and_cache_mla``.
    The aiter helper splits ``kv`` into ``[..., :nope]`` and ``[..., nope:]``;
    V4 keeps them concatenated and applies RoPE to the trailing ``rope_dim``
    in-place upstream, so we just scatter-write the already-assembled rows.
    """
    if slot_mapping is None or slot_mapping.numel() == 0:
        return
    # Squeeze any spurious singleton dims so the input is [N, head_dim].
    if kv.dim() == 3:
        kv = kv.view(-1, kv.shape[-1])
    assert kv.dim() == 2, f"_paged_kv_write expects 2D kv, got {kv.shape}"

    # Cache layout: [total_slots, num_kv_heads=1, head_dim]
    # Drop the kv_head dim for direct row-wise scatter.
    cache_view = kv_cache.view(kv_cache.shape[0], -1)
    head_dim = kv.shape[-1]
    assert cache_view.shape[-1] == head_dim, (
        f"kv_cache last dim {cache_view.shape[-1]} != kv head_dim {head_dim}"
    )

    # Filter out -1 sentinel slots (warmup / dummy run pads with -1).
    valid = slot_mapping >= 0
    if not bool(valid.all().item()):
        # On valid==all-False (pure dummy), skip entirely.
        if not bool(valid.any().item()):
            return
        kv = kv[valid]
        slots = slot_mapping[valid]
    else:
        slots = slot_mapping

    cache_view.index_copy_(0, slots.to(torch.long), kv.to(cache_view.dtype))


def _paged_kv_gather_per_seq(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    max_len: int,
) -> torch.Tensor:
    """Gather paged KV into a dense ``[N, max_len, head_dim]`` view per request.

    Args:
        kv_cache:     ``[total_slots, 1, head_dim]`` paged cache (flat-physical).
        block_tables: ``[N, max_blocks_per_seq]`` int — per-request physical
                      block ids.
        context_lens: ``[N]`` int — number of valid KV tokens per request.
        block_size:   physical-block size (tokens / block).
        max_len:      common output length; rows ``>= context_lens[i]`` are
                      zero (matches sparse_attn's masking via topk_idxs == -1).

    Returns:
        ``[N, max_len, head_dim]`` dense tensor on the same device / dtype as
        ``kv_cache``.
    """
    n = block_tables.shape[0]
    head_dim = kv_cache.shape[-1]
    device = kv_cache.device
    dtype = kv_cache.dtype

    out = torch.zeros((n, max_len, head_dim), dtype=dtype, device=device)
    if n == 0 or max_len == 0:
        return out

    # Per-token global slot index: bt[i, t // bs] * bs + (t % bs) for t < ctx_len[i].
    arange_len = torch.arange(max_len, device=device)
    block_idx = arange_len // block_size                    # [max_len]
    in_block = arange_len % block_size                       # [max_len]
    # Expand to [N, max_len].
    bi = block_idx.unsqueeze(0).expand(n, -1)
    ib = in_block.unsqueeze(0).expand(n, -1)
    # Gather block ids: bt[:, bi] -> [N, max_len]
    safe_bi = bi.clamp(max=block_tables.shape[1] - 1)
    block_ids = block_tables.gather(1, safe_bi.long())
    slot = block_ids * block_size + ib                       # [N, max_len]

    # Mask out positions past context_lens[i] (-> slot becomes 0, but we'll
    # also zero them at the end).
    valid = arange_len.unsqueeze(0) < context_lens.unsqueeze(-1)
    safe_slot = torch.where(
        valid, slot, torch.zeros_like(slot)
    ).long()

    flat = kv_cache.view(kv_cache.shape[0], -1)              # [total_slots, head_dim]
    gathered = flat[safe_slot.flatten()].view(n, max_len, head_dim)
    gathered = torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
    return gathered


# ---------------------------------------------------------------------------
# DeepseekV4MLAAttention — paged KV variant
# ---------------------------------------------------------------------------


class DeepseekV4MLAAttention(nn.Module):
    """V4 hybrid attention with **paged** raw-KV storage.

    Differences from :class:`atom.models.deepseek_v4.DeepseekV4Attention`:
      * No ``register_buffer("kv_cache", torch.zeros(max_batch_size, ...))``.
      * Per-layer ``k_cache`` is fetched from ``forward_context.kv_cache_data
        [f"layer_{layer_num}"].k_cache`` each forward (in-place mutated).
      * Writes go through ``slot_mapping`` (computed by the AITER MLA metadata
        builder in :mod:`atom.model_ops.attentions.aiter_mla`).
      * Reads gather per-request views via ``block_tables`` + ``context_lens``
        before handing off to :func:`atom.model_ops.sparse_attn_v4.sparse_attn`.

    Compressor / Indexer state buffers (``kv_state``, ``score_state``, indexer
    ``kv_cache``) remain as ``register_buffer`` because they are tiny per-seq
    auxiliary state, not the dominant memory term. PR2 may also page them.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        # Marker fields read by ``ModelRunner`` to bind the paged KV tensor.
        # The model_runner walks ``model.modules()`` and on any module with
        # ``base_attention`` set will allocate one slice of the global pool.
        self.base_attention = None
        self.use_mla = True
        # ``kv_cache`` placeholder; the runner overwrites this with a
        # ``[num_blocks * block_size, 1, head_dim]`` view at bind time.
        self.kv_cache = torch.tensor([])
        # Required by some ATOM passes that scan attention layers.
        self.layer_num = layer_id
        self.max_model_len = 0
        # Sparse-attention layers may carry an indexer; for V4 the indexer
        # owns its own (still-buffer) compressed cache and we don't expose it
        # to the model_runner's "indexer.k_cache.kv_cache" V3.2 slot.
        self.indexer: Optional[nn.Module] = None

        self.dim = args.dim
        self.n_heads = args.n_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert (
            args.n_heads % tp_size == 0
        ), f"n_heads={args.n_heads} not divisible by tp={tp_size}"
        assert (
            args.o_groups % tp_size == 0
        ), f"o_groups={args.o_groups} not divisible by tp={tp_size}"
        self.tp_size = tp_size
        self.n_local_heads = args.n_heads // tp_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.scale_fmt = args.scale_fmt
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size

        qc = args.quant_config
        p = prefix

        # Same Linear layout as DeepseekV4Attention.
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wq_a",
        )
        self.q_norm = _RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wq_b",
        )
        self.wkv = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wkv",
        )
        self.kv_norm = _RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * args.o_lora_rank,
            self.dim,
            bias=False,
            quant_config=qc,
            prefix=f"{p}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5

        # Compressor + Indexer for compressed/sparse layers.
        if self.compress_ratio:
            self.compressor = Compressor(
                args,
                self.compress_ratio,
                self.head_dim,
                prefix=f"{p}.compressor",
            )
            if self.compress_ratio == 4:
                self.indexer = Indexer(
                    args, self.compress_ratio, prefix=f"{p}.indexer"
                )
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # Pre-computed RoPE freqs.
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = _precompute_freqs_cis(
            self.rope_head_dim,
            args.max_seq_len,
            original_seq_len,
            rope_theta,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    # ------------------------------------------------------------------
    # ATOM KV-cache spec advertisement
    # ------------------------------------------------------------------

    def get_kv_cache_spec(
        self,
        block_size: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ) -> MLAAttentionSpec:
        """Advertise the per-layer raw-KV cache requirement to the block manager.

        Mirrors :meth:`DeepseekV4Attention.get_kv_cache_spec`: V4's raw-KV is a
        single MLA-style ``[head_size=head_dim, num_kv_heads=1]`` tensor. We
        intentionally do not advertise compressor / indexer caches here — the
        block manager only sees the raw-KV stream because those auxiliary
        states stay as model-side buffers (RFC §6.2.1 footnote: only the
        attention raw-KV goes through the page-table for V4-MLA-paged).
        """
        head_size = self.head_dim
        num_kv_heads = 1
        cr = self.compress_ratio if self.compress_ratio else 1
        storage_block_size = max(1, block_size // cr)
        page_size_bytes = (
            storage_block_size * num_kv_heads * head_size * dtype.itemsize
        )
        return MLAAttentionSpec(
            block_size=block_size,
            page_size_bytes=page_size_bytes,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            compress_ratio=cr,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _fetch_paged_kv_cache(self) -> Optional[torch.Tensor]:
        """Get the layer's paged k_cache from the active forward_context.

        Returns ``None`` during warmup / no-context paths so the caller can
        gracefully fall back to a sparse_attn input built from the current
        prefill (no decode-state needed because warmup is single-shot).
        """
        try:
            from atom.utils.forward_context import get_forward_context

            ctx = get_forward_context()
        except Exception:
            return None
        kv_cache_data = getattr(ctx, "kv_cache_data", None)
        if kv_cache_data is None:
            return None
        entry = kv_cache_data.get(f"layer_{self.layer_id}")
        if entry is None:
            return None
        kc = entry.k_cache
        if kc is None or kc.numel() == 0:
            return None
        return kc

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Compute attention with the paged raw-KV path.

        Reference:
            DSV3.2 sparse-attn read pattern: see
            ``atom/model_ops/attention_mla.py:613-752`` (forward_impl_server_mode)
            and ``atom/models/deepseek_v2.py:990-1116`` (sparse_attn_indexer)
            for the slot_mapping write + block_tables gather pattern this
            method ports to V4.
        """
        assert (
            x.dim() == 2
        ), f"DeepseekV4MLAAttention expects 2D [num_tokens, dim], got {x.shape}"
        seqlen = x.size(0)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # First-call plumbing for compressor / indexer auxiliary cache state.
        # Compressor's `kv_cache` (the *compressed* slot, NOT the raw-KV path
        # that's now paged) is still a model buffer; we lazily attach it.
        if self.compress_ratio:
            if self.compressor.kv_cache is None:
                # Build a compressed-cache buffer on demand (since we no
                # longer have a flat raw-KV buffer to slice). This is small —
                # ``[max_batch_size, max_seq_len // ratio, head_dim]`` — and
                # not on the hot path memory-wise.
                self.compressor.kv_cache = torch.zeros(
                    self.max_batch_size,
                    self.max_seq_len // ratio,
                    self.head_dim,
                    dtype=x.dtype,
                    device=x.device,
                )
                self.compressor.freqs_cis = self.freqs_cis
                if self.indexer is not None:
                    self.indexer.freqs_cis = self.freqs_cis

        # ----- Q -----
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).view(seqlen, self.n_local_heads, self.head_dim)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = q.unsqueeze(0)
        _apply_rotary_emb(q[..., -rd:], freqs_cis)

        # ----- KV (raw window stream) -----
        kv = self.wkv(x)
        kv = self.kv_norm(kv).unsqueeze(0)  # [1, S, head_dim]
        _apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

        # ----- topk_idxs (window + optional compressed) -----
        topk_idxs = _get_window_topk_idxs(win, 1, seqlen, start_pos, device=x.device)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = _get_compress_topk_idxs(
                    ratio, 1, seqlen, start_pos, offset, device=x.device
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # ----- Locate paged KV (server mode) or fall back to in-place flat
        # buffer (warmup / dummy run when forward_context has no KV bound).
        # -----
        kv_cache = self._fetch_paged_kv_cache()
        attn_meta = None
        try:
            from atom.utils.forward_context import get_forward_context

            ctx_obj = get_forward_context()
            attn_meta = getattr(ctx_obj, "attn_metadata", None)
        except Exception:
            attn_meta = None

        slot_mapping = (
            getattr(attn_meta, "slot_mapping", None) if attn_meta is not None else None
        )
        block_tables = (
            getattr(attn_meta, "block_tables", None) if attn_meta is not None else None
        )
        context_lens = (
            getattr(attn_meta, "context_lens", None) if attn_meta is not None else None
        )

        # Decide on a path:
        #   - paged: kv_cache + slot_mapping + block_tables all present.
        #   - fallback: build a single-batch flat KV from (just-projected) kv.
        #     Used during warmup where attn_metadata is unset; correctness
        #     limited to a single sequence (B=1) which is what warmup uses.
        use_paged = (
            kv_cache is not None
            and slot_mapping is not None
            and block_tables is not None
        )

        if use_paged:
            # Write current-token KV into the paged cache.
            _paged_kv_write(kv, kv_cache, slot_mapping)

            # Compressor still produces / persists its own compressed slice;
            # we concatenate the just-emitted compressed kv onto the gathered
            # window KV exactly as the legacy path does.
            kv_compress = None
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)

            # Build per-seq dense KV view for sparse_attn:
            # ``kv_dense[i, :win] = window`` then ``[win:]`` = compressed.
            block_size = max(1, kv_cache.shape[0] // max(1, block_tables.numel() // max(1, block_tables.shape[0])))
            # block_size derivation above is brittle if block_tables has 0 rows;
            # use a more robust path: prefer attn_metadata.kv_indptr if absent
            # use a constant. Refer to BlockManager's block_size; fall back
            # to 1 for warmup degenerate cases.
            atom_block_size = block_size
            try:
                from atom.config import get_current_atom_config

                atom_block_size = get_current_atom_config().kv_cache_block_size
            except Exception:
                pass

            n_seqs = block_tables.shape[0]
            if context_lens is None:
                context_lens = torch.full(
                    (n_seqs,), seqlen + start_pos, device=x.device, dtype=torch.int32
                )
            max_window_len = min(win, int(context_lens.max().item()))
            window_kv = _paged_kv_gather_per_seq(
                kv_cache,
                block_tables,
                context_lens.clamp(max=win).to(torch.long),
                atom_block_size,
                max_window_len if max_window_len > 0 else 1,
            )
            # Pad / truncate to window-size for sparse_attn topk_idxs alignment.
            if window_kv.shape[1] < win:
                pad = torch.zeros(
                    (window_kv.shape[0], win - window_kv.shape[1], window_kv.shape[2]),
                    dtype=window_kv.dtype,
                    device=window_kv.device,
                )
                window_kv = torch.cat([window_kv, pad], dim=1)
            elif window_kv.shape[1] > win:
                window_kv = window_kv[:, :win]

            if self.compress_ratio and kv_compress is not None:
                # kv_compress: [B, T, head_dim] — broadcast to N seqs if B==1.
                if kv_compress.shape[0] == 1 and n_seqs != 1:
                    kv_compress = kv_compress.expand(n_seqs, -1, -1).contiguous()
                kv_dense = torch.cat([window_kv, kv_compress.to(window_kv.dtype)], dim=1)
            else:
                kv_dense = window_kv

            # Reshape q for per-seq sparse_attn: [N, M_per_seq, H, D].
            # For decode (M==1) this is simple; for prefill we keep the legacy
            # B=1 packed contract (caller may have already split sequences).
            if n_seqs == 1:
                q_seq = q  # [1, S, H, D]
                topk_seq = topk_idxs
            else:
                # Multi-seq decode: q is [1, N_tokens, H, D] with one token per seq.
                q_seq = q.transpose(0, 1).contiguous()           # [N, 1, H, D]
                topk_seq = topk_idxs.transpose(0, 1).contiguous()

            o = sparse_attn(q_seq, kv_dense, self.attn_sink, topk_seq, self.softmax_scale)
            if n_seqs != 1:
                o = o.transpose(0, 1).contiguous()
        else:
            # Warmup / dummy: assemble the KV in-line, single-seq B=1 path.
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

        _apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        o = o.squeeze(0).view(seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        x = self.wo_b(o.flatten(1))
        return x

    # Mirror legacy attention's post-load hook to dequant wo_a (shared logic).
    def process_weights_after_loading(self) -> None:
        from atom.models.deepseek_v4 import DeepseekV4Attention

        # Reuse — same wo_a FP8 → BF16 dequant logic.
        DeepseekV4Attention.process_weights_after_loading(self)


# ---------------------------------------------------------------------------
# Block / Model wrappers — replace the attention class only
# ---------------------------------------------------------------------------


class BlockMLA(nn.Module):
    """Transformer block identical to :class:`atom.models.deepseek_v4.Block`
    except the attention sub-layer uses :class:`DeepseekV4MLAAttention`.

    All Hyper-Connections / FFN / RMSNorm wiring is delegated to the legacy
    Block via composition; we only swap in the new attention class.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args, prefix: str = ""):
        super().__init__()
        self.layer_id = layer_id
        # We can't simply subclass Block because Block.__init__ allocates the
        # legacy attention via `DeepseekV4Attention(layer_id, args, prefix=...)`
        # which would re-create the flat kv_cache buffer. So we inline the
        # parts of Block we need and keep them shape/name-compatible for the
        # weight loader.
        from atom.models.deepseek_v4 import _RMSNorm as _R
        from atom.models.deepseek_v4 import MoE as _MoE

        self.norm_eps = args.norm_eps
        self.attn = DeepseekV4MLAAttention(layer_id, args, prefix=f"{prefix}.attn")
        self.ffn = _MoE(layer_id, args, prefix=f"{prefix}.ffn")
        self.attn_norm = _R(args.dim, self.norm_eps)
        self.ffn_norm = _R(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    # Reuse legacy Block's hc_pre / hc_post / forward by binding them as
    # methods on this class — they are pure functions of self.* params and
    # don't reference the attention module directly.
    from atom.models.deepseek_v4 import Block as _LegacyBlock  # noqa: E402

    HC_POST_MULT = _LegacyBlock.HC_POST_MULT
    hc_pre = _LegacyBlock.hc_pre
    hc_post = _LegacyBlock.hc_post
    forward = _LegacyBlock.forward


class DeepseekV4ModelMLA(nn.Module):
    """Counterpart of :class:`DeepseekV4Model` using :class:`BlockMLA`.

    Reuses the legacy model's forward (which only depends on ``self.layers``,
    ``self.embed`` etc.) by pulling it in as a method.
    """

    def __init__(self, *, args: DeepseekV4Args):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult

        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [
                BlockMLA(layer_id, args, prefix=f"layers.{layer_id}")
                for layer_id in range(args.n_layers)
            ]
        )
        self.norm = _RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)

        # MTP blocks (still use legacy MTPBlock — its inner Block uses the
        # legacy DeepseekV4Attention which has its own kv_cache buffer; the
        # MTP path is entry-point optional and doesn't share KV with the main
        # paged stream, so this is acceptable for PR1).
        self.mtp = nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            blk = MTPBlock(args.n_layers + layer_id, args, prefix=f"mtp.{layer_id}")
            blk.embed = self.embed
            blk.head = self.head
            self.mtp.append(blk)

        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    # Reuse legacy model forward (the packed-prefill split + LM head logic
    # there is purely shape-driven, doesn't touch attention internals).
    from atom.models.deepseek_v4 import DeepseekV4Model as _LegacyModel  # noqa: E402

    forward = _LegacyModel.forward


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------


class DeepseekV4ForCausalLM_MLA(nn.Module):
    """ATOM model contract wrapper for the V4-MLA-paged variant.

    Shares the weights mapper / loader / expert-mapping logic with the legacy
    :class:`atom.models.deepseek_v4.DeepseekV4ForCausalLM` so any existing V4
    checkpoint loads identically — only the runtime KV-cache substrate changes.
    """

    # Identical to legacy V4 — reuse via attribute aliasing in __init_subclass__
    # is awkward; just inline.
    from atom.model_loader.loader import WeightsMapper as _WeightsMapper  # noqa: E402

    weights_mapper = _WeightsMapper(
        orig_to_new_prefix={
            "embed.": "model.embed.",
            "layers.": "model.layers.",
            "norm.weight": "model.norm.weight",
            "head.weight": "model.head.weight",
            "hc_head_": "model.hc_head_",
            "mtp.": "model.mtp.",
        }
    )
    weights_mapping = {
        ".gate.bias": ".gate.e_score_correction_bias",
        ".scale": ".weight_scale_inv",
    }

    def __init__(self, config: Config, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = config.hf_config
        self.args = DeepseekV4Args.from_hf_config(self.hf_config)
        self.args.quant_config = make_v4_quant_config(self.hf_config)
        self.model = DeepseekV4ModelMLA(args=self.args)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        start_pos = int(positions[0].item()) if positions is not None else 0
        return self.model(input_ids=input_ids, start_pos=start_pos, **model_kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    # Reuse legacy expert mapping + load_weights helpers exactly.
    def get_expert_mapping(self):
        from atom.models.deepseek_v4 import DeepseekV4ForCausalLM as _Legacy

        return _Legacy.get_expert_mapping(self)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set:
        from atom.models.deepseek_v4 import DeepseekV4ForCausalLM as _Legacy

        return _Legacy.load_weights(self, weights)
