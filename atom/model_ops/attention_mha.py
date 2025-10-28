# from flash_attn import flash_attn_with_kvcache
from dataclasses import dataclass

import aiter
import torch
import triton
import triton.language as tl
from aiter.paged_attn import PagedAttention
from torch import nn

from atom.utils.context import get_context
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.forward_context import (
    AttentionMetadata,
    ForwardContext,
    get_forward_context,
    set_forward_context,
)
from atom.utils import mark_spliting_op
from .attention_mla import MLAModules


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        kv_cache_dtype="bf16",
        layer_num=0,
        mla_modules: MLAModules=None,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache_dtype = kv_cache_dtype
        self.max_model_len = 0
        self.k_scale = self.v_scale = None
        self.layer_num = layer_num

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, position: torch.Tensor=None):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        

        # o = torch.ops.aiter.unified_attention_with_output(q, k, v, 
        #             self.scale, self.kv_cache_dtype, self.layer_num)
        context = get_context()
        if context.slot_mapping.numel():
            # not dummy run
            forward_context: ForwardContext = get_forward_context()
            attn_metadata_ = forward_context.no_compile_layers[self.layer_num]
            k_cache = attn_metadata_.k_cache
            v_cache = attn_metadata_.v_cache
            k_scale = attn_metadata_.k_scale
            v_scale = attn_metadata_.v_scale
        else:
            # dummy run before allocate kv_cache, thus we create manually
            k_cache = v_cache = torch.tensor([])
            k_scale = v_scale = None

        if k_cache.numel() and v_cache.numel():
            if self.kv_cache_dtype == "fp8":
                aiter.reshape_and_cache_with_pertoken_quant(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    k_scale,
                    v_scale,
                    context.slot_mapping,
                    asm_layout=True,
                )
            else:
                aiter.reshape_and_cache(
                    k,
                    v,
                    k_cache,
                    v_cache,
                    context.slot_mapping,
                    kv_cache_dtype="auto",
                    k_scale=None,
                    v_scale=None,
                    asm_layout=True,
                )


        if context.is_prefill:
            # if context.block_tables is not None:  # prefix cache
            #     k, v = k_cache, v_cache
            o = aiter.flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                min_seqlen_q=context.min_seqlen_q,
                dropout_p=context.dropout_p,
                softmax_scale=self.scale,
                causal=True,
            )
        else:  # decode
            o = aiter.pa_fwd_asm(
                q,
                k_cache,
                v_cache,
                context.block_tables,
                context.context_lens,
                context.block_tables.stride(0),
                K_QScale=k_scale,
                V_QScale=v_scale,
                out_=None,
                high_precision=0,
            )


        o = o.view(-1, self.num_heads * self.head_dim)
        return o
