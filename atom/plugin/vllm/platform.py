"""ATOM vLLM platform integration.

This module contains the vLLM `Platform` subclass used in ATOM's vLLM plugin
mode. Keep platform behavior here so `register.py` can focus on registration
and wiring only.
"""

import logging

from atom.utils import envs

logger = logging.getLogger("atom")
# This flag is used to enable the vLLM plugin mode.
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN
disable_vllm_plugin_attention = envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION

def _vllm_at_least(major: int, minor: int) -> bool:
    """Best-effort vllm version gate. Returns True on parse failure to be conservative
    (the caller's compat path is a no-op on older vllm anyway)."""
    try:
        import vllm

        v = vllm.__version__.split("+")[0]
        parts = v.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return True


if not disable_vllm_plugin:
    from vllm.platforms.rocm import RocmPlatform

    class ATOMPlatform(RocmPlatform):
        # For multi-modality models, to make AiterBackend supported by ViT,
        # get_supported_vit_attn_backends may need to be overridden here
        @classmethod
        def get_attn_backend_cls(
            cls, selected_backend, attn_selector_config, num_heads
        ) -> str:
            if disable_vllm_plugin_attention:
                logger.info("Fallback to original vLLM attention backend")
                return super().get_attn_backend_cls(
                    selected_backend, attn_selector_config, num_heads
                )

            logger.info("Use atom attention backend")
            if attn_selector_config.use_mla:
                if getattr(attn_selector_config, "use_sparse", False):
                    return "atom.plugin.vllm.attention_backend.mla_sparse.AiterMLASparseBackend"
                return "atom.model_ops.attentions.aiter_mla.AiterMLABackend"
            return "atom.model_ops.attentions.aiter_attention.AiterBackend"

        @classmethod
        def check_and_update_config(cls, vllm_config) -> None:
            """ATOM-specific compat tweaks on top of RocmPlatform's defaults.

            vLLM 0.20+ ships a piecewise cudagraph + AOT-autograd pipeline that
            records `view([-1, num_kv_heads, head_dim])` on the attention key
            tensor with a symbolic shape annotation derived from the KV-cache
            layout (rather than as a constant from config). When the runtime
            cudagraph capture-size bucket lands on a value whose tensor numel is
            not divisible by `num_kv_heads * head_dim`, AOT autograd's
            `apply_view_meta_sequence` replays the recorded view and crashes
            with `RuntimeError: shape '[-1, 8, 64]' is invalid for input of
            size N`. See https://github.com/sunway513/ATOM/issues/61.

            Workaround: filter the materialized cudagraph_capture_sizes to
            multiples of 512 (= num_kv_heads * head_dim for GPT-OSS GQA, and a
            divisibility-safe lower bound for every ATOM-supported GQA model
            today). This is a no-op on vLLM 0.19, whose default capture-size
            heuristic was already divisibility-safe in this code path.

            MLA models exercise a different attention backend (AiterMLABackend)
            with their own view shapes; this filter does not regress them but
            the long-term fix lives upstream in vllm-project/vllm.
            """
            super().check_and_update_config(vllm_config)

            if not _vllm_at_least(0, 20):
                return

            cc = getattr(vllm_config, "compilation_config", None)
            if cc is None:
                return

            sizes = getattr(cc, "cudagraph_capture_sizes", None)
            if not sizes:
                return

            DIVISOR = 512
            kept = sorted({s for s in sizes if s % DIVISOR == 0})
            if not kept:
                kept = [DIVISOR]

            if kept != list(sizes):
                cc.cudagraph_capture_sizes = kept
                logger.warning(
                    "ATOM[vllm-0.20-compat]: filtered cudagraph_capture_sizes "
                    "from %d -> %d entries (kept multiples of %d). "
                    "Workaround for upstream view-meta divisibility regression. "
                    "See https://github.com/sunway513/ATOM/issues/61.",
                    len(sizes),
                    len(kept),
                    DIVISOR,
                )

else:
    ATOMPlatform = None
