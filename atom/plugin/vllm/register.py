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
    """Mark `vllm_attention_cls.forward` as opaque to torch._dynamo.

    vLLM 0.20+ moved the per-tensor reshape (`query/key/value.view(-1, num_heads, head_size)`)
    out of the registered custom op (`unified_attention_with_output`) and into
    the surrounding Python forward body, with the comment "We do this outside
    the custom op to minimize the CPU overheads from the non-CUDA-graph regions."
    Under torch.compile + AOT autograd, those reshapes get traced into the
    compiled fx graph with a pathological symbolic shape annotation
    `(s80//512), 8, (s80//(8*((s80//512))))` — `head_size` is derived from a
    KV-cache layout symbol rather than carried as a constant. At runtime
    AOT autograd's `apply_view_meta_sequence` replays the recorded view and
    crashes whenever the intermediate tensor numel isn't divisible by
    `num_kv_heads * head_dim` (e.g. 1599 for a mid-batch GPT-OSS request):

      RuntimeError: shape '[-1, 8, 64]' is invalid for input of size 1599

    Wrapping `Attention.forward` with `torch._dynamo.disable` makes the entire
    forward (reshapes + custom op call) opaque to dynamo, so the broken view
    never enters any compiled fx graph (captured cudagraphs OR inductor
    general-shape fallback). The reshape executes eagerly with concrete
    shapes at runtime, which always satisfies the divisibility constraint.

    Cost: a graph break at every attention call. vLLM's compile pipeline is
    already piecewise (one fx graph per stretch of model code between attention
    layers), so this just makes that boundary explicit. No measured perf
    regression vs the upstream PR-#39125 reshape-outside-custom-op design.

    No-op on vLLM 0.19, where the reshapes were gated behind
    `accept_output_buffer=False` and ATOM never hit this path.

    Caller is responsible for importing `vllm.attention.layer.Attention` and
    passing it here (avoids a circular import if this is called from
    register_platform during early plugin discovery — call it from
    register_model() instead).

    See https://github.com/sunway513/ATOM/issues/61.
    """
    if not _vllm_at_least(0, 20):
        return

    try:
        import torch._dynamo as _dynamo

        if getattr(vllm_attention_cls.forward, "_atom_dynamo_disabled", False):
            return  # idempotent

        _orig_forward = vllm_attention_cls.forward
        _wrapped = _dynamo.disable(_orig_forward)
        try:
            _wrapped._atom_dynamo_disabled = True
        except (AttributeError, TypeError):
            pass
        vllm_attention_cls.forward = _wrapped
        logger.warning(
            "ATOM[vllm-0.20-compat]: marked vllm %s.forward as opaque "
            "to torch._dynamo to keep upstream view-meta divisibility bug "
            "(attention.py reshapes baked with symbolic head_size) out of "
            "compiled fx graphs. See https://github.com/sunway513/ATOM/issues/61.",
            vllm_attention_cls.__name__,
        )
    except Exception as e:
        logger.warning(
            "ATOM[vllm-0.20-compat]: failed to install %s.forward "
            "dynamo-disable wrapper: %s",
            getattr(vllm_attention_cls, "__name__", "<unknown>"),
            e,
        )


def _maybe_install_atom_dispatcher_compat() -> None:
    """Force ATOM's @support_torch_compile decorator off the bytecode-swap
    dispatch path on vLLM 0.20+.

    ATOM's `TorchCompileWrapperWithCustomDispatcher` (atom/utils/decorators.py)
    runs the first forward call through dynamo + AOT autograd + inductor and
    captures the resulting bytecode in `self.compiled_codes`. From the second
    call onward it bypasses dynamo entirely by swapping
    `forward.__code__ = self.compiled_codes[0]` via `dispatch_to_code(0)` —
    no guards, no recompile checks.

    vLLM 0.20's `gpu_model_runner.profile_run` exercises the model with
    multiple shape buckets in succession, and at least one bucket passes a
    rank-2 input_ids where the first trace recorded rank-1 (`i32[s72]`).
    The bytecode-swap path executes the rank-1 inductor compile against the
    rank-2 input and immediately fails:

      AssertionError: wrong number of dimensions 2
      (assert_size_stride(arg0_1, (s72, ), (1, )))

    Setting `use_custom_dispatcher = False` after the original __init__
    forces every call to go through `self.compiled_callable(*args, **kwargs)`
    instead of `dispatch_to_code(0)`. Dynamo's guard mechanism then handles
    rank/shape changes correctly via on-demand recompilation. ATOM's
    inductor + cudagraph optimization paths are still used; we just give up
    the bytecode-swap micro-optimization on vLLM 0.20+.

    No-op on vLLM 0.19, where profile_run apparently never feeds a different
    rank than the first trace.
    """
    if not _vllm_at_least(0, 20):
        return

    try:
        import functools

        from atom.utils.decorators import (
            TorchCompileWrapperWithCustomDispatcher as _Wrapper,
        )

        if getattr(_Wrapper.__init__, "_atom_dispatcher_force_dynamo", False):
            return  # idempotent

        _orig_init = _Wrapper.__init__

        @functools.wraps(_orig_init)
        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # Force the slow (dynamo-guarded) dispatch path. The custom
            # bytecode-swap path is rank-fragile under vLLM 0.20 profile_run.
            self.use_custom_dispatcher = False

        _patched_init._atom_dispatcher_force_dynamo = True
        _Wrapper.__init__ = _patched_init
        logger.warning(
            "ATOM[vllm-0.20-compat]: forced TorchCompileWrapperWithCustomDispatcher"
            ".use_custom_dispatcher=False to keep dynamo guards in the dispatch"
            " loop and let it recompile when vLLM 0.20 profile_run feeds inputs"
            " of different rank than the first trace. See"
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
