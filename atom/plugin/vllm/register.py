from typing import Optional
import logging

import torch
from atom.plugin.prepare import _set_framework_backbone
from atom.utils import envs
from atom.plugin.vllm.mla_patch import patch_vllm_mla_attention

logger = logging.getLogger("atom")

# this flag is used to enable the vllm plugin mode
disable_vllm_plugin = envs.ATOM_DISABLE_VLLM_PLUGIN
disable_vllm_plugin_attention = envs.ATOM_DISABLE_VLLM_PLUGIN_ATTENTION

# those 2 models are covering most of dense and moe models
ATOM_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMForCausalLM"
ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER = "atom.plugin.vllm.model_wrapper:ATOMMoEForCausalLM"

# when register new model to vllm, add here
# Keys is from hf config arch name
_VLLM_MODEL_REGISTRY_OVERRIDES: dict[str, str] = {
    "LlamaForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3ForCausalLM": ATOM_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GptOssForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "DeepseekV3ForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Glm4MoeForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "GlmMoeDsaForCausalLM": ATOM_MOE_CAUSAL_LM_MODEL_WRAPPER,
    "Qwen3NextForCausalLM": "atom.models.qwen3_next:Qwen3NextForCausalLMVllm",
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration": "atom.models.qwen3_5:Qwen3_5MoeForConditionalGeneration",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration",
}


def _set_plugin_mode() -> None:
    _set_framework_backbone("vllm")


def _vllm_at_least(major: int, minor: int) -> bool:
    """Best-effort vllm version gate. Returns True on parse failure to be
    conservative (the caller's compat path is a no-op on older vllm anyway)."""
    try:
        import vllm

        v = vllm.__version__.split("+")[0]
        parts = v.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except Exception:
        return True


def _maybe_install_vllm_020_attention_compat(vllm_attention_cls) -> None:
    """Replace `vllm_attention_cls.forward` body to skip query/key/value
    reshape when the backend opted out via `accept_output_buffer = False`.

    vLLM 0.19 had an else branch in `Attention.forward` (gated by the
    backend's `accept_output_buffer = False`) that skipped the per-tensor
    `query/key/value.view(-1, num_heads, head_size)` reshape and called
    `unified_attention(query, key, value, layer_name)` (the no-output-buffer
    variant of the custom op) directly. ATOM relies on this gated path to
    smuggle non-tensor data through the Attention call (positions through
    `key`, fused qkv through `value` when
    ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1) and let its own backend
    impl unpack them inside `forward_impl_plugin_mode`.

    vLLM 0.20+ removed both the else branch and the standalone
    `unified_attention` op (vllm-project/vllm#39125,
    "[Attention][V0 Deprecation] Deprecate accept output buffer"), so every
    Attention.forward call unconditionally reshapes. ATOM's smuggled key
    (= positions, 1-D length=num_tokens) is then `view(-1, num_kv_heads,
    head_size)`-ed, which fails for any num_tokens not divisible by
    num_kv_heads * head_size:

      RuntimeError: shape '[-1, 8, 64]' is invalid for input of size 1008

    Fix: monkey-patch Attention.forward so that for backends with
    `accept_output_buffer = False` (i.e. ATOM plugin backends), we skip
    the reshape entirely and call `unified_attention_with_output` directly.
    The output buffer is still allocated (vLLM 0.20 has no
    no-output-buffer custom op anymore), but we leave it un-reshaped, just
    like the query/key/value, so the backend's `impl.forward` receives all
    four tensors in their raw layout. ATOM's own backend handles unpacking
    and reshaping inside `atom/plugin/attention_mha.forward_impl_plugin_mode`
    and `atom/plugin/attention_mla.forward_impl_plugin_mode`.

    For backends with `accept_output_buffer = True` (the vLLM default), we
    delegate to the original forward — no behavior change.

    The patched forward is wrapped with `torch._dynamo.disable` so neither
    branch's body gets traced into a compiled fx graph (avoids the upstream
    view-meta divisibility bug from being baked into inductor sub-graphs
    via any other code path that might trace through Attention.forward).

    No-op on vLLM 0.19, where the original else branch is still present.

    Caller imports the Attention class and passes it (avoids a circular
    import; call from register_model() not register_platform).

    See https://github.com/sunway513/ATOM/issues/61.
    """
    if not _vllm_at_least(0, 20):
        return

    try:
        import torch._dynamo as _dynamo
        from vllm.model_executor.layers.attention.attention import (
            _encode_layer_name,
            _resolve_layer_name,
            get_attention_context,
        )

        if getattr(vllm_attention_cls.forward, "_atom_compat_patched", False):
            return  # idempotent

        _orig_forward = vllm_attention_cls.forward

        def _patched_forward(
            self,
            query: "torch.Tensor",
            key: "torch.Tensor",
            value: "torch.Tensor",
            output_shape=None,
        ) -> "torch.Tensor":
            accept_output_buffer = getattr(
                self.attn_backend, "accept_output_buffer", True
            )
            if accept_output_buffer:
                # vLLM 0.20 default — original behavior (with reshapes).
                return _orig_forward(self, query, key, value, output_shape=output_shape)

            # ATOM-style: backend opted out of vLLM-managed output buffer.
            # Mirror vLLM 0.19's else-branch behavior — skip the reshapes
            # of query/key/value (they may carry smuggled positions/qkv),
            # and call the backend's impl.forward DIRECTLY. We can't go
            # through unified_attention_with_output because that custom op is
            # void (returns None); ATOM's PagedAttentionImpl.forward_impl_plugin_mode
            # ignores any passed `output` buffer and allocates+returns its
            # own. If we pre-allocated an output and passed it through the
            # void custom op, the backend's allocated tensor would be lost
            # and we'd return uninitialized garbage — which is exactly the
            # gsm8k=0% accuracy regression observed in the first iteration of
            # this patch (sunway513/ATOM#61 follow-up).
            if self.calculate_kv_scales:
                torch.ops.vllm.maybe_calc_kv_scales(
                    query, key, value, _encode_layer_name(self.layer_name)
                )
            if (
                self.query_quant is not None
                and self.impl.supports_quant_query_input
            ):
                query, _ = self.query_quant(query, self._q_scale)

            # Fetch attn_metadata + kv_cache the same way vllm 0.20's
            # unified_attention_with_output does.
            layer_name = _resolve_layer_name(self.layer_name)
            attn_metadata, attn_layer, kv_cache, _ = get_attention_context(
                layer_name
            )

            # Direct dispatch — backend allocates output internally and
            # returns it (raw layout, e.g. [num_tokens, num_heads*head_size]).
            output = attn_layer.impl.forward(
                attn_layer, query, key, value, kv_cache, attn_metadata
            )

            # Flatten to 2-D [num_tokens, hidden_size] for the caller
            # (matches the original forward's final `output.view(-1, hidden_size)`).
            if output is None:
                raise RuntimeError(
                    "ATOM[vllm-0.20-compat]: backend "
                    f"{type(attn_layer.impl).__name__}.forward returned None "
                    "but the patched Attention.forward expects a tensor return. "
                    "Backend may need to be updated to return the allocated output."
                )
            if output.dim() > 2:
                output = output.reshape(output.shape[0], -1)
            return output

        wrapped = _dynamo.disable(_patched_forward)
        try:
            wrapped._atom_compat_patched = True
        except (AttributeError, TypeError):
            pass
        vllm_attention_cls.forward = wrapped
        logger.warning(
            "ATOM[vllm-0.20-compat]: replaced vllm %s.forward to skip "
            "query/key/value reshape when backend.accept_output_buffer=False "
            "(restores vLLM 0.19 else-branch semantics that ATOM relies on "
            "for smuggling positions/qkv through Attention call args). "
            "See https://github.com/sunway513/ATOM/issues/61.",
            vllm_attention_cls.__name__,
        )
    except Exception as e:
        logger.warning(
            "ATOM[vllm-0.20-compat]: failed to install %s.forward body "
            "replacement: %s",
            getattr(vllm_attention_cls, "__name__", "<unknown>"),
            e,
        )


def _maybe_install_atom_dispatcher_compat() -> None:
    """Disable ATOM's @support_torch_compile decoration on vLLM 0.20+.

    ATOM's compile pipeline (TorchCompileWrapperWithCustomDispatcher /
    VllmBackend / dispatch_to_code) bakes shape and rank assumptions from
    the first forward call into the inductor-compiled sub-graphs and
    relies on bypassing dynamo guards (via `forward.__code__` swap) for
    speed on subsequent calls. vLLM 0.20's `gpu_model_runner.profile_run`
    drives the model with multiple shape buckets — including different
    input ranks — in rapid succession, and the cached inductor sub-graphs
    fire `assert_size_stride(arg0_1, (s72,), (1,))` (rank-mismatch) on
    the very first profile-run forward pass after the trace.

    Earlier attempts in this PR's history tried narrower fixes:
      - filter cudagraph_capture_sizes to multiples of 512 (insufficient,
        the bug surfaces in the inductor general-shape compile too)
      - mark `Attention.forward` opaque to dynamo (necessary but
        the broken view-meta is also baked at the model-wrapper level)
      - force `use_custom_dispatcher=False` (necessary but the cached
        inductor sub-graphs still assert before dynamo would invalidate)

    The robust fix is to opt out of ATOM's compile decoration entirely
    on vLLM 0.20+ and let the model run eagerly. AITER kernels and ATOM
    model implementations still execute (they're plain PyTorch ops);
    cudagraph capture continues to work at vLLM's worker layer. We give
    up ATOM's inductor + piecewise-compile optimization in exchange for
    end-to-end correctness on the new vLLM contract.

    Implementation: monkey-patch `atom.utils.decorators._support_torch_compile`
    to return the class unchanged. Any module imported AFTER this patch
    runs (e.g. `atom.models.gpt_oss` when vLLM later instantiates the
    model) sees the no-op decorator and skips the wrapper installation.
    Model classes fall back to `nn.Module.__call__` which invokes
    `forward` eagerly.

    No-op on vLLM 0.19, where ATOM's compile path works correctly.
    See https://github.com/sunway513/ATOM/issues/61.
    """
    if not _vllm_at_least(0, 20):
        return

    try:
        import atom.utils.decorators as _atom_dec

        if getattr(_atom_dec, "_atom_compile_decoration_disabled", False):
            return  # idempotent

        def _patched_support_torch_compile(cls, dynamic_arg_dims):
            # Return cls unchanged — no @support_torch_compile decoration.
            # cls.__call__ then resolves to nn.Module.__call__ which calls
            # cls.forward(*args, **kwargs) eagerly with no compile.
            return cls

        _atom_dec._support_torch_compile = _patched_support_torch_compile
        _atom_dec._atom_compile_decoration_disabled = True
        logger.warning(
            "ATOM[vllm-0.20-compat]: replaced atom.utils.decorators."
            "_support_torch_compile with a no-op so subsequent model imports"
            " skip ATOM's torch.compile pipeline and run eager. AITER kernels"
            " and ATOM model implementations are unaffected; cudagraph capture"
            " still happens at the vLLM worker layer. See"
            " https://github.com/sunway513/ATOM/issues/61."
        )
    except Exception as e:
        logger.warning(
            "ATOM[vllm-0.20-compat]: failed to install dispatcher compat: %s",
            e,
        )


def register_platform() -> Optional[str]:

    if disable_vllm_plugin:
        # return None instead of error because the flag can be used to
        # run pure vllm mode without ATOM plugin
        logger.info("Disable ATOM OOT plugin platforms")
        return None

    # Do not call _set_plugin_mode() here. SGLang (and other stacks) discover
    # vllm.platform_plugins and would set atom's backbone to "vllm" before
    # importing SGLang plugin modules — then atom.models.qwen3_5's ``if is_vllm():``
    # branch runs and requires vllm.model_executor.models.qwen3_5, which may be
    # absent. Backbone is set in register_model() for real vLLM runs.

    # NOTE: do NOT install the vllm-0.20 Attention.forward dynamo-disable
    # wrapper here. register_platform() runs during plugin discovery while
    # vllm.config / vllm.distributed are still partially initialized; importing
    # vllm.attention.layer at this point triggers a circular import. The
    # install is deferred to register_model() (called later), at which point
    # vllm has finished bringing up its core modules — the existing
    # _patch_vllm_attention_process_weights_after_loading hook already proves
    # vllm.attention.layer is importable from there.

    # return the ATOM platform to vllm
    return "atom.plugin.vllm.platform.ATOMPlatform"


def _patch_vllm_attention_process_weights_after_loading(attention) -> None:
    orig = attention.process_weights_after_loading

    if getattr(orig, "_atom_default_act_dtype_patched", False):
        return

    try:
        import inspect

        sig = inspect.signature(orig)
        act_dtype_param = sig.parameters.get("act_dtype")
        if (
            act_dtype_param is not None
            and act_dtype_param.default is not inspect._empty
        ):
            return
    except Exception:
        pass

    import functools

    @functools.wraps(orig)
    def wrapped(self, act_dtype: "torch.dtype" = torch.bfloat16):
        return orig(self, act_dtype)

    setattr(wrapped, "_atom_default_act_dtype_patched", True)
    attention.process_weights_after_loading = wrapped


def register_model() -> None:
    if disable_vllm_plugin:
        logger.info("Disable ATOM model register")
        return

    _set_plugin_mode()

    import vllm.model_executor.models.registry as vllm_model_registry

    any_updated = False
    for arch, qual in _VLLM_MODEL_REGISTRY_OVERRIDES.items():
        module_name, class_name = qual.split(":", 1)
        existing = vllm_model_registry.ModelRegistry.models.get(arch)
        if existing is not None:
            # If already overridden to the same target, skip re-registering.
            if (
                getattr(existing, "module_name", None) == module_name
                and getattr(existing, "class_name", None) == class_name
            ):
                continue

        logger.info(f"Register model {arch} to vLLM with {qual}")
        vllm_model_registry.ModelRegistry.register_model(arch, qual)
        any_updated = True

    # clear lru cache
    if any_updated:
        vllm_model_registry._try_load_model_cls.cache_clear()
        vllm_model_registry._try_inspect_model_cls.cache_clear()

    patch_vllm_mla_attention()
    # patch attention process weights after loading
    # to avoid the specific handle in ATOM loader
    try:
        from vllm.attention.layer import Attention, MLAAttention
    except ImportError:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

    _patch_vllm_attention_process_weights_after_loading(Attention)
    _patch_vllm_attention_process_weights_after_loading(MLAAttention)

    # Install vLLM 0.20+ Attention.forward dynamo-disable wrapper here (NOT in
    # register_platform); vllm.attention.layer is safely importable from this
    # lifecycle hook (the lines above prove it). Idempotent and gated on
    # vllm >= 0.20.
    _maybe_install_vllm_020_attention_compat(Attention)

    # Force ATOM's @support_torch_compile decorator to use the dynamo-guarded
    # dispatch path (instead of its custom bytecode-swap fast path) so the
    # model can re-trace when vLLM 0.20+ profile_run feeds different input
    # ranks across buckets. See sunway513/ATOM#61.
    _maybe_install_atom_dispatcher_compat()

    # Patch vLLM graph_capture to also enter aiter's ca_comm.capture(),
    # avoiding hipMemcpyAsync in fused_allreduce_rmsnorm when model uses aiter collectives
    from atom.plugin.vllm.graph_capture_patch import apply_graph_capture_patch

    apply_graph_capture_patch()
