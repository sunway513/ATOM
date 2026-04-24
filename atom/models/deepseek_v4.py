# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
DeepSeek-V4 model for ATOM (PR1: skeleton + tiny-config eager forward).

Architecture reference: /data/DeepSeek-V4-Pro/inference/model.py
Tech report: /app/logs_claude/deepseek_v4/DeepSeek_V4.pdf

This file is the PR1 skeleton. It mirrors the reference implementation's class
structure so dummy state_dicts produced by the reference can be loaded directly
into ATOM modules for numerical parity validation. Production paths (FP8/FP4
weight loading, tensor parallelism, AITER kernels, KV cache integration, MTP
spec decode, torch.compile, server) land in PR2-PR6.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from atom.config import Config
from atom.model_ops.quant_v4 import (
    act_quant_inplace,
    fp4_act_quant_inplace,
    rotate_activation,
)
from atom.model_ops.sparse_attn_v4 import hc_split_sinkhorn, sparse_attn  # noqa: F401

# ---------------------------------------------------------------------------
# Config wrapper
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4Args:
    """Mirrors `inference/model.py:ModelArgs`. Constructed from `hf_config`.

    Field names match the V4 HuggingFace `config.json` keys where possible;
    aliases are documented inline.
    """

    # Core
    vocab_size: int = 129280
    dim: int = 7168  # hidden_size
    n_layers: int = 61  # num_hidden_layers
    n_mtp_layers: int = 1  # num_nextn_predict_layers
    n_hash_layers: int = 3  # num_hash_layers
    norm_eps: float = 1e-6  # rms_norm_eps
    max_seq_len: int = 1048576  # max_position_embeddings
    max_batch_size: int = 4  # PR1 toy default; PR3 driven by ATOM scheduler

    # Attention (MQA, single shared KV head)
    n_heads: int = 128  # num_attention_heads
    head_dim: int = 512
    rope_head_dim: int = 64  # qk_rope_head_dim
    q_lora_rank: int = 1536
    o_lora_rank: int = 1024
    o_groups: int = 16
    window_size: int = 128  # sliding_window

    # Per-layer attention type: 0=Dense, 4=CSA, 128 (or other large m')=HCA
    compress_ratios: Tuple[int, ...] = field(default_factory=tuple)

    # Indexer (CSA layers only)
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 1024

    # MoE
    moe_inter_dim: int = 3072  # moe_intermediate_size
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6  # num_experts_per_tok
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 2.5  # routed_scaling_factor
    swiglu_limit: float = 10.0

    # Hyper-Connections (mHC)
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # YaRN RoPE
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0  # rope_scaling.factor
    original_seq_len: int = 65536  # rope_scaling.original_max_position_embeddings
    beta_fast: int = 32
    beta_slow: int = 1

    # Quantization (PR1 ignores; PR2+ uses)
    dtype: Literal["bf16", "fp8"] = "bf16"
    expert_dtype: Optional[Literal["fp4", "fp8"]] = None
    scale_fmt: Optional[Literal["ue8m0"]] = None

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "DeepseekV4Args":
        rope_scaling = getattr(hf_config, "rope_scaling", {}) or {}
        return cls(
            vocab_size=hf_config.vocab_size,
            dim=hf_config.hidden_size,
            n_layers=hf_config.num_hidden_layers,
            n_mtp_layers=getattr(hf_config, "num_nextn_predict_layers", 1),
            n_hash_layers=getattr(hf_config, "num_hash_layers", 0),
            norm_eps=hf_config.rms_norm_eps,
            max_seq_len=hf_config.max_position_embeddings,
            n_heads=hf_config.num_attention_heads,
            head_dim=hf_config.head_dim,
            rope_head_dim=hf_config.qk_rope_head_dim,
            q_lora_rank=hf_config.q_lora_rank,
            o_lora_rank=hf_config.o_lora_rank,
            o_groups=hf_config.o_groups,
            window_size=hf_config.sliding_window,
            compress_ratios=tuple(hf_config.compress_ratios),
            index_n_heads=hf_config.index_n_heads,
            index_head_dim=hf_config.index_head_dim,
            index_topk=hf_config.index_topk,
            moe_inter_dim=hf_config.moe_intermediate_size,
            n_routed_experts=hf_config.n_routed_experts,
            n_shared_experts=hf_config.n_shared_experts,
            n_activated_experts=hf_config.num_experts_per_tok,
            score_func=hf_config.scoring_func,
            route_scale=hf_config.routed_scaling_factor,
            swiglu_limit=hf_config.swiglu_limit,
            hc_mult=hf_config.hc_mult,
            hc_sinkhorn_iters=hf_config.hc_sinkhorn_iters,
            hc_eps=hf_config.hc_eps,
            rope_theta=hf_config.rope_theta,
            compress_rope_theta=hf_config.compress_rope_theta,
            rope_factor=rope_scaling.get("factor", 1.0),
            original_seq_len=rope_scaling.get("original_max_position_embeddings", 0),
            beta_fast=rope_scaling.get("beta_fast", 32),
            beta_slow=rope_scaling.get("beta_slow", 1),
        )


# ---------------------------------------------------------------------------
# Module-level constants matching reference inference/model.py module globals
# ---------------------------------------------------------------------------

# PR1 always runs single-rank; TP comes in PR3.
_FP4_BLOCK_SIZE = 32  # matches reference's fp4_block_size


# ---------------------------------------------------------------------------
# Small utilities — port of inference/model.py:183-276
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Reference RMSNorm: weight stored in fp32, computation in fp32, output in input dtype.

    Port of inference/model.py:183-196. Distinct from atom.model_ops.layernorm.RMSNorm
    which is wired for TP/quantization; PR1 keeps a self-contained class so dummy
    state_dicts load directly. PR3 may swap to ATOM's RMSNorm with dtype shim.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def _precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> torch.Tensor:
    """Precompute complex exponentials for rotary embeddings with YaRN scaling.

    Port of inference/model.py:199-229. When `original_seq_len > 0`, applies YaRN
    frequency interpolation with a smooth linear ramp between beta_fast and
    beta_slow correction ranges.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_, max_, dim):
        if min_ == max_:
            max_ += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary positional embeddings IN-PLACE.

    Port of inference/model.py:232-244. The input tensor `x` is overwritten with
    the rotated values; the same tensor is also returned for chaining.
    `inverse=True` uses the conjugate (un-rotation) — used on the attention
    output to remove absolute-position embedding from the value contribution.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@lru_cache(1)
def _get_window_topk_idxs(
    window_size: int, bsz: int, seqlen: int, start_pos: int
) -> torch.Tensor:
    """Per-query topk-style indices into the sliding window KV cache.

    Port of inference/model.py:255-265. Returns [bsz, seqlen, window_size] of
    int positions into the KV buffer. -1 marks "skip" (causal mask, no fill).
    """
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [
                torch.arange(start_pos + 1, window_size),
                torch.arange(0, start_pos + 1),
            ],
            dim=0,
        )
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1
        )
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size)
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def _get_compress_topk_idxs(
    ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int
) -> torch.Tensor:
    """Per-query indices into the compressed KV cache (for HCA — no Indexer).

    Port of inference/model.py:269-276. -1 marks compressed blocks that are
    still in the future at this query position.
    """
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


# ---------------------------------------------------------------------------
# Compressor + Indexer — port of inference/model.py:279-433
# ---------------------------------------------------------------------------


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.

    Port of inference/model.py:279-377. When `overlap=True` (always set when
    ratio==4, used by CSA), the compression uses overlapping windows to smooth
    block boundaries. When `rotate=True` (only the Indexer's compressor),
    output is Hadamard-rotated and FP4-simulated; otherwise non-RoPE dims are
    FP8-simulated.
    """

    def __init__(
        self,
        args: DeepseekV4Args,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.scale_fmt = args.scale_fmt
        coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # wkv/wgate stored as fp32 (matches reference's Linear(dtype=fp32) BF16 path).
        self.wkv = nn.Linear(
            self.dim, coff * self.head_dim, bias=False, dtype=torch.float32
        )
        self.wgate = nn.Linear(
            self.dim, coff * self.head_dim, bias=False, dtype=torch.float32
        )
        self.norm = _RMSNorm(self.head_dim, args.norm_eps)

        # External tensors — assigned by the owning Attention / Indexer at first forward.
        self.kv_cache: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

        # Decode-phase state buffers. With overlap: state[:, :ratio] holds the
        # overlapping window from the previous compression block; state[:, ratio:]
        # holds the current in-progress window.
        self.register_buffer(
            "kv_state",
            torch.zeros(
                args.max_batch_size,
                coff * compress_ratio,
                coff * self.head_dim,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "score_state",
            torch.full(
                (
                    args.max_batch_size,
                    coff * compress_ratio,
                    coff * self.head_dim,
                ),
                float("-inf"),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def overlap_transform(
        self, tensor: torch.Tensor, value: float = 0.0
    ) -> torch.Tensor:
        """Reshape the [b, s, ratio, 2*d] overlap-projected tensor into [b, s, 2*ratio, d]
        such that the second half (size ratio) holds the current window's
        normal-projection slice and the first half (size ratio) holds the
        previous query position's overlap-projection slice (-inf padded for s=0).
        """
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int) -> Optional[torch.Tensor]:
        """Compress KV for the input tokens. Writes into self.kv_cache when a
        compression block boundary is hit; otherwise just buffers state and returns None.

        Args:
            x: [B, S, dim] input hidden states (BF16 or FP32).
            start_pos: starting position in the absolute sequence (0 = prefill).
        Returns:
            Compressed KV slice that was just written ([B, S/ratio, head_dim] in
            prefill, or [B, 1, head_dim] in decode), or None if no compression
            boundary was hit on this call.
        """
        assert self.kv_cache is not None, "compressor.kv_cache must be set by owner"
        assert self.freqs_cis is not None, "compressor.freqs_cis must be set by owner"
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        overlap = self.overlap
        d = self.head_dim
        rd = self.rope_head_dim
        dtype = x.dtype

        # Compression always done in fp32 for stability.
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        if start_pos == 0:
            # ===== Prefill path =====
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0

            # Save the last `ratio` overlap-slice tokens into kv_state for use
            # by the next decode call's overlap window.
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[:bsz, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            # Save the trailing partial block (remainder tokens) into kv_state.
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[:bsz, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            # Reshape to [B, num_blocks, ratio, coff*d] and softmax-pool.
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0.0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            # ===== Decode path (start_pos > 0, seqlen == 1) =====
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[:bsz, :ratio, :d],
                            self.kv_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[:bsz, :ratio, :d],
                            self.score_state[:bsz, ratio:, d:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    # Roll: the just-completed window becomes the next overlap window.
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)

        if not should_compress:
            return None

        kv = self.norm(kv.to(dtype))

        # Apply RoPE to the rope-head-dim tail of compressed entries.
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        _apply_rotary_emb(kv[..., -rd:], freqs_cis)

        # QAT round-trip: rotated branch (Indexer) uses Hadamard + FP4 sim;
        # plain branch uses FP8 sim on non-rope dims only.
        if self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant_inplace(kv, _FP4_BLOCK_SIZE)
        else:
            act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

        if start_pos == 0:
            self.kv_cache[:bsz, : seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.

    Port of inference/model.py:380-433. Has its own Compressor (with Hadamard
    rotation + FP4 simulation) to build a separate compressed KV cache used
    only for index scoring; query is also FP4-simulated.
    """

    def __init__(self, args: DeepseekV4Args, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        # PR1 single-rank: n_local_heads == n_heads (TP comes in PR3).
        self.n_local_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.compress_ratio = compress_ratio

        # Reference uses ColumnParallelLinear; PR1 uses plain Linear.
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.weights_proj = nn.Linear(
            self.dim, self.n_heads, bias=False, dtype=torch.bfloat16
        )
        self.softmax_scale = self.head_dim**-0.5

        self.compressor = Compressor(args, compress_ratio, self.head_dim, rotate=True)
        self.register_buffer(
            "kv_cache",
            torch.zeros(
                args.max_batch_size,
                args.max_seq_len // compress_ratio,
                self.head_dim,
            ),
            persistent=False,
        )
        self.freqs_cis: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
    ) -> torch.Tensor:
        """Compute sparse top-k indices over the indexer's compressed KV cache.

        Args:
            x: [B, S, dim] input hidden states (for compressor + weights_proj).
            qr: [B, S, q_lora_rank] latent query (shared with main attention's q_a).
            start_pos: absolute sequence start position.
            offset: offset added to returned indices to land them in the
                concatenated (window || compressed) KV layout consumed by sparse_attn.
        Returns:
            topk_idxs: [B, S, K] int — selected compressed-KV positions (with offset),
                       -1 = invalid (future-masked).
        """
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        # Lazy plumb the indexer's kv_cache + freqs_cis into its compressor.
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        # ----- Indexer Q -----
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        _apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # Indexer Q uses FP4 simulation (matches reference QAT).
        fp4_act_quant_inplace(q, _FP4_BLOCK_SIZE)

        # ----- Indexer KV -----
        # Compressor mutates self.kv_cache (writes the new compressed entries).
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

        # ----- Index score -----
        # q: [B, S, H, D]; kv_cache slice: [B, T, D] -> einsum to [B, S, H, T]
        index_score = torch.einsum(
            "bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio]
        )
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        # PR1 single-rank: skip all_reduce.

        # ----- Top-k selection over compressed positions -----
        if start_pos == 0:
            mask = (
                torch.arange(seqlen // ratio).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), 0.0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


# ---------------------------------------------------------------------------
# Stubs — implementations land in tasks #5-#8
# ---------------------------------------------------------------------------


class DeepseekV4Attention(nn.Module):
    """Hybrid attention: MQA + grouped output LoRA + sliding window + attn_sink.

    Port of inference/model.py:436-543. Per-layer behavior driven by
    `compress_ratio` (read from args.compress_ratios[layer_id]):

      - `compress_ratio == 0`: Dense (sliding-window only; no compressor/indexer)
      - `compress_ratio == 4`: CSA (compressor with overlap + indexer for top-k)
      - `compress_ratio >= 8`: HCA (compressor only; topk_idxs pre-computed)

    Layout:
      - Single shared MQA head for KV (head_dim=512). Each query head attends
        to the same compressed/window KV via per-query top-k gather.
      - q_lora_rank low-rank Q projection: wq_a -> q_norm -> wq_b -> RMSNorm-per-head -> RoPE
      - Grouped output LoRA: o_groups groups, each with rank o_lora_rank
      - Sliding window of `args.window_size=128` raw KV entries (BF16, FP8-simulated nope dims)
      - Compressed KV up to `max_seq_len // compress_ratio` entries (when ratio > 0)
      - attn_sink: per-head learnable logit added only to softmax denominator
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        # PR1 single-rank: n_local_heads == n_heads (TP comes in PR3).
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups  # single-rank
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.scale_fmt = args.scale_fmt

        # ----- Parameters (names mirror reference for state_dict load) -----
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = _RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = _RMSNorm(self.head_dim, self.eps)
        # wo_a is BF16 in reference (special-cased dtype). Layout: weight is
        # [n_groups * o_lora_rank, n_heads * head_dim / n_groups]. We reshape
        # to [n_groups, o_lora_rank, n_heads * head_dim / n_groups] in forward.
        self.wo_a = nn.Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.wo_b = nn.Linear(self.n_groups * args.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        # ----- Compressor (and Indexer for CSA) -----
        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
            else:
                self.indexer = None
        else:
            self.compressor = None
            self.indexer = None

        # ----- KV cache: [B, window_size + max_seq_len/ratio, head_dim] -----
        kv_cache_size = args.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.register_buffer(
            "kv_cache",
            torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim),
            persistent=False,
        )

        # ----- RoPE freqs (own freqs, not shared): YaRN for compressed
        # attention layers (long context), plain rope for dense (window-only) -----
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

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Compute attention for `x` at absolute position `start_pos`.

        Args:
            x: [B, S, dim] input hidden states (BF16).
            start_pos: 0 = prefill (whole sequence); >0 = decode (S typically 1).
        Returns:
            [B, S, dim] attention output (BF16).
        """
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        # First-call plumbing: hand the (compressed-half) KV cache + freqs_cis
        # to the compressor / indexer.
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis

        # ----- Q: low-rank projection + per-head RMSNorm + partial RoPE -----
        qr = q = self.q_norm(self.wq_a(x))  # qr is the latent shared with Indexer
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        # Per-head RMSNorm (no learnable weight; matches reference)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        _apply_rotary_emb(q[..., -rd:], freqs_cis)

        # ----- Window KV: project, RMSNorm, RoPE on rope dims, FP8-sim on nope -----
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        _apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant_inplace(kv[..., :-rd], 64, self.scale_fmt)

        # ----- Build topk_idxs: window indices first, then compressed indices -----
        topk_idxs = _get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            # `offset` is the position in the concatenated [window || compressed]
            # tensor where compressed KV entries start.
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = _get_compress_topk_idxs(
                    ratio, bsz, seqlen, start_pos, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # ----- Attention: prefill writes window KV linearly + concats fresh
        # compressed entries; decode writes one slot in window ring buffer
        # then reads from the persistent kv_cache. -----
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                # Wrap the last `win` tokens into a ring buffer rooted at `cutoff`.
                cutoff = seqlen % win
                (
                    self.kv_cache[:bsz, cutoff:win],
                    self.kv_cache[:bsz, :cutoff],
                ) = kv[
                    :, -win:
                ].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(
                q,
                self.kv_cache[:bsz],
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            )

        # Inverse RoPE on output's rope dims to remove absolute-position contribution
        # carried in by the value-side RoPE of the KV entries.
        _apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        # ----- Grouped output LoRA: einsum per group, then RowParallel-equivalent project -----
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE gate with sqrtsoftplus/sigmoid/softmax scoring + hash routing.

    Port of inference/model.py:546-584. For `layer_id < args.n_hash_layers`,
    routing is by precomputed token-id-to-expert table (`tid2eid`); no scoring,
    no bias. Otherwise routing is by `score_func(W @ x) + bias` topk.

    Bias affects expert SELECTION (added before topk) but NOT routing weights —
    weights come from the original (pre-bias) score gathered at the topk indices.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    args.vocab_size, args.n_activated_experts, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.bias = None
        else:
            self.bias = nn.Parameter(
                torch.empty(args.n_routed_experts, dtype=torch.float32)
            )

    def forward(
        self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus (V4 default)
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            assert input_ids is not None, "hash routing requires input_ids"
            indices = self.tid2eid[input_ids].long()
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights * self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability.

    Port of inference/model.py:587-606. With `swiglu_limit > 0`, clamps both gate
    and up projections (gate clipped above only, up clipped both sides) before
    the SiLU * up product — matches reference behavior exactly.
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(
        self, x: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Mixture-of-Experts: top-k routed experts + 1 shared expert.

    Port of inference/model.py:609-645. PR1 single-rank only — every rank holds
    every expert. PR3 will swap to FusedMoE for TP/EP sharding.

    `input_ids` is plumbed through (used by hash-routing layers' Gate).
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts  # PR1 single-rank
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = 0  # PR1 single-rank
        self.experts_end_idx = self.n_local_experts

        self.gate = Gate(layer_id, args)
        # PR1 BF16 experts (no FP4/FP8). PR2 will switch dtype to expert_dtype.
        self.experts = nn.ModuleList(
            [
                Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
                for _ in range(self.n_routed_experts)
            ]
        )
        assert args.n_shared_experts == 1, "V4 reference assumes 1 shared expert"
        # Shared expert: no swiglu_limit (matches reference)
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=0.0)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(
            indices.flatten(), minlength=self.n_routed_experts
        ).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] = y[idx] + expert(x[idx], weights[idx, top, None])
        # PR1 single-rank: no all_reduce
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Manifold-Constrained Hyper-Connections (mHC).

    Port of inference/model.py:648-701. The residual stream is widened to
    `[B, S, hc_mult, D]`. Each sub-layer (attn / ffn):
      1. `hc_pre`: project `[B, S, hc_mult, D]` -> `[B, S, D]` via Sinkhorn-projected
         pre-weights (also producing post-weights and combination matrix for hc_post).
      2. `attn_norm` + `attn` (or `ffn_norm` + `ffn`): standard sub-layer in `[B, S, D]`.
      3. `hc_post`: expand `[B, S, D]` back to `[B, S, hc_mult, D]` using the
         post-weights (gate on the new contribution) + the combination matrix
         applied to the previous residual.
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = DeepseekV4Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = _RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = _RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        # All HC params stored in fp32 (matches reference's `set_dtype(torch.float32)`).
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce [B, S, hc, D] residual to [B, S, D] sub-layer input.

        Also returns post-weights and combination matrix to be applied in hc_post.
        """
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()  # [B, S, hc*D]
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt  # [B, S, mix_hc]
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        # pre: [B, S, hc]; weighted-sum hc copies into single [B, S, D]
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand [B, S, D] sub-layer output back to [B, S, hc, D] residual.

        post[..., hc] gates the new contribution (broadcast across hc dim);
        comb[..., hc, hc] linearly combines the previous residual's hc copies.
        """
        # x: [B, S, D]; residual: [B, S, hc, D]
        # post.unsqueeze(-1) * x.unsqueeze(-2): [B, S, hc, D] gating
        # comb.unsqueeze(-1) * residual.unsqueeze(-2): [B, S, hc, hc, D]; sum over dim=2
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        input_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # ----- Attention sub-layer with mHC mixing -----
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        # ----- FFN sub-layer with mHC mixing -----
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):
    """LM head with mHC reduction.

    Port of inference/model.py:704-736. Unlike `Block.hc_pre` (which uses
    Sinkhorn projection on the combination matrix), `hc_head` uses simple
    `Sigmoid(mix*scale + base) + eps` weights to reduce the [B, S, hc, D]
    residual to [B, S, D] before applying the LM head linear projection.

    `get_logits` projects only the last token (decode mode); for prefill the
    caller should slice the desired positions before passing through.
    """

    def __init__(
        self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        # PR1 single-rank: full vocab on this rank.
        self.weight = nn.Parameter(
            torch.empty(self.vocab_size, self.dim, dtype=torch.float32)
        )

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        # Reference projects only the last token: x[:, -1].
        return F.linear(x[:, -1].float(), self.weight)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: nn.Module,
    ) -> torch.Tensor:
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x))
        # PR1 single-rank: skip all_gather
        return logits


class MTPBlock(Block):
    """MTP block: V4 dense block + e_proj/h_proj/enorm/hnorm + own hc_head params + LM head.

    Port of inference/model.py:739-767. Subclass of Block reusing all HC + Attention + FFN
    machinery; adds a token-embed projection (`e_proj`), a hidden-state projection
    (`h_proj`), per-input RMSNorms, and its own `hc_head_fn/base/scale` parameters
    for the final LM head reduction.

    `embed` and `head` are assigned externally by `DeepseekV4Model` (shared with
    the main model's embedding and LM head).
    """

    def __init__(self, layer_id: int, args: DeepseekV4Args):
        super().__init__(layer_id, args)
        self.e_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.h_proj = nn.Linear(args.dim, args.dim, bias=False)
        self.enorm = _RMSNorm(args.dim, args.norm_eps)
        self.hnorm = _RMSNorm(args.dim, args.norm_eps)
        self.norm = _RMSNorm(args.dim, args.norm_eps)
        # Per-MTP hc_head params (distinct from Block's hc_attn/hc_ffn params).
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        # Externally-assigned by DeepseekV4Model (shared with main model).
        self.embed: Optional[nn.Module] = None
        self.head: Optional[ParallelHead] = None

    def forward(
        self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor
    ) -> torch.Tensor:
        # x: [B, S, hc_mult, D] residual stream from main model
        assert (
            self.embed is not None and self.head is not None
        ), "MTPBlock requires .embed and .head to be assigned by the parent model"
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        # Mix embedding + hidden into a fresh residual stream
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(
            x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
        )
        return logits


class DeepseekV4Model(nn.Module):
    """Full model: embed -> expand to hc_mult copies -> N blocks -> hc_head -> logits.

    Port of inference/model.py:Transformer (770-810). MTP blocks are constructed
    and have their `.embed` and `.head` linked to the main model's, but they are
    NOT called from the main forward path — PR5 will integrate them into ATOM's
    EagleProposer via `self.mtp[k].forward(...)` from outside.

    PR1 single-rank: uses plain `nn.Embedding` for `self.embed` (state_dict-compatible
    with reference's `ParallelEmbedding` since both store a single `weight` parameter).
    """

    def __init__(self, *, args: DeepseekV4Args):
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.hc_mult = args.hc_mult

        # Reference's ParallelEmbedding stores `weight` as the partition's slice;
        # at single-rank that's the full vocab. nn.Embedding has the same `weight`
        # name, so dummy state_dicts load directly.
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(
            [Block(layer_id, args) for layer_id in range(args.n_layers)]
        )
        self.norm = _RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)

        # MTP blocks: constructed and linked, but only invoked externally (PR5).
        self.mtp = nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            blk = MTPBlock(args.n_layers + layer_id, args)
            blk.embed = self.embed
            blk.head = self.head
            self.mtp.append(blk)

        # Top-level hc_head params used to reduce the final hc_mult residual stack
        # before the LM head linear projection.
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections.
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        logits = self.head(
            h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm
        )
        return logits


class DeepseekV4ForCausalLM(nn.Module):
    """ATOM model contract wrapper.

    PR1: only `__init__` (from ATOM Config) + `forward` (toy validation only).
    PR3 will add `load_weights` for the real HF checkpoint.
    """

    def __init__(self, config: Config, prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = config.hf_config
        self.args = DeepseekV4Args.from_hf_config(self.hf_config)
        self.model = DeepseekV4Model(args=self.args)

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
        # In V4, the LM head is fused into DeepseekV4Model.forward (it consumes
        # the hc_mult-expanded residual). So `hidden_states` already IS logits.
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        raise NotImplementedError("PR3: real checkpoint loader")
