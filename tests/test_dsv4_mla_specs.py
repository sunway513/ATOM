# SPDX-License-Identifier: MIT
# Layer 1 unit tests for atom/models/deepseek_v4_mla.py — the MLA paged
# KV-cache variant of the DeepSeek-V4 model. CPU-only, no GPU required.
#
# Coverage:
#   * Module imports cleanly without GPU.
#   * `DeepseekV4MLAAttention.get_kv_cache_spec()` returns the expected
#     MLAAttentionSpec for each layer flavor (dense / C4 / C128).
#   * The new attention class does NOT register a flat
#     `kv_cache` buffer (the whole point of this rewrite).
#   * The marker fields (`base_attention`, `use_mla`, `kv_cache`,
#     `layer_num`) needed by ModelRunner's KV binding loop are present.
#
# IMPORTANT — conftest interaction:
#   /home/pensun/ATOM/tests/conftest.py installs a *stub* aiter module to
#   keep the bare-import-tree working without a real aiter wheel. When the
#   real aiter IS installed (rocm/atom-dev image), the stub overwrites it
#   and breaks ``from aiter import QuantType, ...`` inside
#   ``atom.model_ops.attention_mla``. We work around that by re-importing
#   the real aiter from disk before any atom.* import — the test only runs
#   in the GPU-capable container where the real aiter exists.

import importlib
import importlib.util
import sys

import pytest
import torch

# Force-restore the real aiter module if conftest replaced it with a stub.
def _restore_real_aiter() -> None:
    cur = sys.modules.get("aiter")
    # Heuristic: stubbed aiter has very few attributes; real one has
    # hundreds. ``QuantType`` is a sentinel that only the real one exports.
    if cur is not None and getattr(cur, "QuantType", None) is not None:
        return
    # Drop stub + reload from disk. find_spec raises ValueError when the
    # stub module's __spec__ is None (the case for types.ModuleType()), so
    # we must remove the stub from sys.modules FIRST before doing any spec
    # lookup.
    for mod_name in list(sys.modules):
        if mod_name == "aiter" or mod_name.startswith("aiter."):
            del sys.modules[mod_name]
    try:
        importlib.import_module("aiter")
    except ImportError:
        # Real aiter not available — restore an empty stub so atom imports
        # at least don't raise NameError on `aiter` lookups (tests below
        # that need real aiter symbols will skip via pytest.importorskip).
        sys.modules["aiter"] = type(sys)("aiter")

_restore_real_aiter()


def _restore_real_atom_module(prefix: str) -> None:
    """Drop conftest's stubbed ``atom.<prefix>`` module from sys.modules so
    the next import re-loads the real one from disk. Used for ``atom`` and
    ``atom.config`` which conftest replaces with bare ModuleType stubs to
    keep BlockManager unit tests fast — but our DSV4 MLA tests need the
    real ones (real Config / real torch tensor types / real package path).
    """
    for mod_name in list(sys.modules):
        if mod_name == prefix or mod_name.startswith(prefix + "."):
            mod = sys.modules.get(mod_name)
            # Only drop *stub* modules (those without a __spec__ from disk).
            if mod is not None and getattr(mod, "__spec__", None) is None:
                del sys.modules[mod_name]


_restore_real_atom_module("atom")
_restore_real_atom_module("atom.config")
_restore_real_atom_module("atom.utils.custom_register")


def _ensure_dist_initialized() -> None:
    """Initialize the bare-minimum distributed state so layers that touch
    ``aiter.dist.parallel_state.get_tp_group()`` (VocabParallelEmbedding,
    ColumnParallelLinear, etc.) can be constructed at test time.

    The CPU torch backend works for a 1-process world; we point all groups
    at the local single-rank process so ``rank_in_group`` returns 0.
    """
    import os as _os

    import torch.distributed as _dist

    if _dist.is_initialized():
        return
    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29555")
    _os.environ.setdefault("WORLD_SIZE", "1")
    _os.environ.setdefault("RANK", "0")
    _os.environ.setdefault("LOCAL_RANK", "0")
    try:
        _dist.init_process_group(backend="gloo", world_size=1, rank=0)
    except Exception:  # pragma: no cover  (best-effort)
        return

    # Mirror the minimum aiter.dist.parallel_state setup that
    # VocabParallelEmbedding / ColumnParallelLinear consult.
    try:
        from aiter import init_dist_env

        init_dist_env(tp=1, ep=1, dp=1, pp=1)
    except Exception:
        # If aiter's init helper isn't usable in CPU mode, fall back to
        # patching the module's globals directly.
        try:
            from aiter.dist import parallel_state as _ps
            from aiter.dist.parallel_state import GroupCoordinator
        except Exception:
            return

        # Build a single-rank GroupCoordinator-like sentinel.
        class _OneRank:
            world_size = 1
            rank_in_group = 0

            def all_reduce(self, x, op=None):
                return x

            def all_gather(self, x, dim=0):
                return x

        sentinel = _OneRank()
        _ps._TP = sentinel
        _ps._EP = sentinel
        _ps._DP = sentinel
        _ps._PP = sentinel

# Some atom files import a few helpers from `aiter.dist.parallel_state` at
# module load. With the real aiter restored these resolve naturally; if
# we're somehow still on a stub we provide the minimal shim.
if not hasattr(sys.modules.get("aiter"), "QuantType"):  # pragma: no cover
    pytest.skip(
        "real aiter required for DSV4 MLA tests (no QuantType in stub)",
        allow_module_level=True,
    )

from atom.v1.kv_cache_interface import (  # noqa: E402
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    physical_pool_key,
)


# Lightweight stand-in for atom.config.Config — avoid pulling the real
# Config (which triggers HF / GPU init paths). Only needs to be a dummy
# object that wrapper.atom_config can hold without callers introspecting it.
class _LocalMockConfig:
    pass


# ── DeepseekV4MLAAttention.get_kv_cache_spec ────────────────────────────────


class TestDeepseekV4MLAAttentionSpec:
    def _make_attn(self, compress_ratio: int):
        from atom.models.deepseek_v4_mla import DeepseekV4MLAAttention

        attn = DeepseekV4MLAAttention.__new__(DeepseekV4MLAAttention)
        attn.head_dim = 512
        attn.compress_ratio = compress_ratio
        return attn

    def test_dense_layer_compress_ratio_zero_maps_to_one(self):
        spec = self._make_attn(compress_ratio=0).get_kv_cache_spec()
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.compress_ratio == 1
        assert spec.storage_block_size == 256

    def test_c4_layer(self):
        spec = self._make_attn(compress_ratio=4).get_kv_cache_spec()
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.compress_ratio == 4
        assert spec.storage_block_size == 64
        assert spec.head_size == 512
        assert spec.num_kv_heads == 1

    def test_c128_layer(self):
        spec = self._make_attn(compress_ratio=128).get_kv_cache_spec()
        assert spec.compress_ratio == 128
        assert spec.storage_block_size == 2  # 256 // 128

    def test_dtype_default_bfloat16(self):
        spec = self._make_attn(compress_ratio=4).get_kv_cache_spec()
        assert spec.dtype == torch.bfloat16

    def test_returns_mla_not_sliding_window(self):
        # Distinct from Compressor.get_kv_cache_spec which returns the
        # SlidingWindowMLASpec subclass — the *attention* spec is plain MLA.
        spec = self._make_attn(compress_ratio=4).get_kv_cache_spec()
        assert isinstance(spec, MLAAttentionSpec)
        assert not isinstance(spec, SlidingWindowMLASpec)

    def test_two_attentions_same_ratio_share_pool_key(self):
        a = self._make_attn(compress_ratio=4).get_kv_cache_spec()
        b = self._make_attn(compress_ratio=4).get_kv_cache_spec()
        assert physical_pool_key(a) == physical_pool_key(b)

    def test_distinct_ratios_distinct_pool_keys(self):
        c4 = self._make_attn(compress_ratio=4).get_kv_cache_spec()
        c128 = self._make_attn(compress_ratio=128).get_kv_cache_spec()
        assert physical_pool_key(c4) != physical_pool_key(c128)


# ── No flat `kv_cache` buffer ──────────────────────────────────────────────


class TestNoFlatKvCacheBuffer:
    """The whole point of the MLA paged variant: the legacy attention's
    ``register_buffer("kv_cache", torch.zeros(max_batch_size, kv_cache_size,
    head_dim))`` MUST be absent. The runtime KV substrate now lives in
    ``forward_context.kv_cache_data[f"layer_{i}"].k_cache``.
    """

    def _build_attention(self, compress_ratio: int = 4):
        _ensure_dist_initialized()
        from atom.models.deepseek_v4 import DeepseekV4Args
        from atom.models.deepseek_v4_mla import DeepseekV4MLAAttention

        # Toy config — enough layers for compress_ratios indexing.
        args = DeepseekV4Args(
            vocab_size=128,
            dim=64,
            n_layers=4,
            n_heads=4,
            head_dim=32,
            rope_head_dim=8,
            q_lora_rank=16,
            o_lora_rank=8,
            o_groups=2,
            window_size=8,
            compress_ratios=(0, 0, 4, 128),
            index_n_heads=2,
            index_head_dim=16,
            index_topk=4,
            moe_inter_dim=32,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            hc_mult=2,
            max_seq_len=64,
            max_batch_size=2,
            n_hash_layers=0,
        )
        layer_id = {0: 0, 4: 2, 128: 3}[compress_ratio]
        return DeepseekV4MLAAttention(layer_id, args, prefix=f"layers.{layer_id}.attn")

    def test_no_kv_cache_buffer_for_dense_layer(self):
        attn = self._build_attention(compress_ratio=0)
        buffers = dict(attn.named_buffers())
        # The ONLY persistent-flag buffer left should be `freqs_cis` (RoPE
        # cache, persistent=False) — never a `kv_cache` buffer of any kind.
        # named_buffers() returns even non-persistent buffers, so the assertion
        # is a strict "no key called kv_cache".
        assert "kv_cache" not in buffers, (
            f"DeepseekV4MLAAttention must NOT register a flat kv_cache "
            f"buffer; found buffers: {list(buffers.keys())}"
        )

    def test_no_kv_cache_buffer_for_c4_layer(self):
        attn = self._build_attention(compress_ratio=4)
        buffers = dict(attn.named_buffers())
        # The Compressor and Indexer sub-modules still keep their *small*
        # auxiliary state buffers (kv_state, score_state, indexer.kv_cache),
        # but the attention itself's own `kv_cache` buffer must be gone.
        assert "kv_cache" not in buffers, (
            "DeepseekV4MLAAttention must not own a `kv_cache` buffer at the "
            "top of the named_buffers tree (compressor.* / indexer.* are OK)"
        )

    def test_no_kv_cache_buffer_for_c128_layer(self):
        attn = self._build_attention(compress_ratio=128)
        buffers = dict(attn.named_buffers())
        assert "kv_cache" not in buffers


# ── Marker fields for ModelRunner KV binding ───────────────────────────────


class TestModelRunnerBindingMarkers:
    """ModelRunner walks ``model.modules()`` looking for ``base_attention``
    and ``use_mla`` to know which submodules need a paged-KV binding. The
    new attention class MUST set these so that loop picks it up exactly
    like :class:`atom.model_ops.paged_attention.PagedAttention` does.
    """

    def _build_attention(self):
        _ensure_dist_initialized()
        from atom.models.deepseek_v4 import DeepseekV4Args
        from atom.models.deepseek_v4_mla import DeepseekV4MLAAttention

        args = DeepseekV4Args(
            vocab_size=128,
            dim=64,
            n_layers=4,
            n_heads=4,
            head_dim=32,
            rope_head_dim=8,
            q_lora_rank=16,
            o_lora_rank=8,
            o_groups=2,
            window_size=8,
            compress_ratios=(0, 0, 4, 128),
            index_n_heads=2,
            index_head_dim=16,
            index_topk=4,
            moe_inter_dim=32,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            hc_mult=2,
            max_seq_len=64,
            max_batch_size=2,
            n_hash_layers=0,
        )
        return DeepseekV4MLAAttention(2, args, prefix="layers.2.attn")

    def test_base_attention_marker_exists(self):
        attn = self._build_attention()
        assert hasattr(attn, "base_attention")  # value can be None

    def test_use_mla_is_true(self):
        attn = self._build_attention()
        assert attn.use_mla is True

    def test_kv_cache_placeholder_exists(self):
        attn = self._build_attention()
        # Placeholder; ModelRunner overwrites with the paged tensor view.
        assert isinstance(attn.kv_cache, torch.Tensor)
        # Should be empty placeholder (numel == 0) at construction time —
        # production binding happens later in ModelRunner.
        assert attn.kv_cache.numel() == 0

    def test_layer_num_set(self):
        attn = self._build_attention()
        assert attn.layer_num == 2

    def test_indexer_attribute_exists(self):
        # Required by the V3.2 sparse-attn path's introspection in ModelRunner.
        # CSA layer (compress_ratio=4) → indexer is non-None; HCA (>=8) → None.
        attn = self._build_attention()  # compress_ratio=4 (CSA)
        assert hasattr(attn, "indexer")
        assert attn.indexer is not None  # CSA layer has an indexer


# ── Top-level wrapper sanity ───────────────────────────────────────────────


class TestForCausalLMWrapper:
    """Construct the full ``DeepseekV4ForCausalLM_MLA`` against a tiny stub
    config and assert it builds without GPU and exposes the standard ATOM
    ``model_engine`` contract (``forward``, ``compute_logits``,
    ``get_expert_mapping``, ``load_weights``, ``weights_mapper``).
    """

    def _make_wrapper(self):
        _ensure_dist_initialized()
        from atom.models.deepseek_v4 import DeepseekV4Args, make_v4_quant_config
        from atom.models.deepseek_v4_mla import (
            DeepseekV4ForCausalLM_MLA,
            DeepseekV4ModelMLA,
        )

        # Minimal hf_config — only the attrs DeepseekV4Args.from_hf_config
        # touches via getattr(default).
        class HF:
            vocab_size = 128
            hidden_size = 64
            num_hidden_layers = 2
            num_attention_heads = 4
            head_dim = 32
            qk_rope_head_dim = 8
            q_lora_rank = 16
            o_lora_rank = 8
            o_groups = 2
            sliding_window = 8
            compress_ratios = (0, 0)
            index_n_heads = 2
            index_head_dim = 16
            index_topk = 4
            moe_intermediate_size = 32
            n_routed_experts = 2
            n_shared_experts = 1
            num_experts_per_tok = 1
            scoring_func = "sqrtsoftplus"
            routed_scaling_factor = 1.0
            swiglu_limit = 0.0
            hc_mult = 2
            hc_sinkhorn_iters = 4
            hc_eps = 1e-6
            rope_theta = 10000.0
            compress_rope_theta = 160000.0
            rms_norm_eps = 1e-6
            rope_scaling = {}
            num_nextn_predict_layers = 0
            num_hash_layers = 0
            max_position_embeddings = 64
            quantization_config = None

        # Construct directly (avoid the full Config plumbing).
        wrapper = DeepseekV4ForCausalLM_MLA.__new__(DeepseekV4ForCausalLM_MLA)
        torch.nn.Module.__init__(wrapper)
        wrapper.atom_config = _LocalMockConfig()
        wrapper.hf_config = HF
        wrapper.args = DeepseekV4Args.from_hf_config(HF)
        wrapper.args.max_seq_len = 64
        wrapper.args.max_batch_size = 2
        # Force tuple form (HF defaults can yield lists when read by getattr).
        wrapper.args.compress_ratios = (0, 0)
        wrapper.args.quant_config = make_v4_quant_config(HF)
        wrapper.model = DeepseekV4ModelMLA(args=wrapper.args)
        return wrapper

    def test_constructible_without_gpu(self):
        wrapper = self._make_wrapper()
        assert wrapper.model is not None
        # Two layers of BlockMLA, each with a DeepseekV4MLAAttention.
        from atom.models.deepseek_v4_mla import (
            BlockMLA,
            DeepseekV4MLAAttention,
        )
        assert len(wrapper.model.layers) == 2
        assert all(isinstance(b, BlockMLA) for b in wrapper.model.layers)
        assert all(
            isinstance(b.attn, DeepseekV4MLAAttention) for b in wrapper.model.layers
        )

    def test_no_top_level_kv_cache_buffer(self):
        # Walk every named_buffer in the full model and verify the legacy
        # flat `kv_cache` buffer name is absent at every BlockMLA.attn
        # nesting level. Compressor/Indexer kv_cache buffers under
        # `layers.{i}.attn.compressor.kv_cache` etc. are still allowed
        # (they're auxiliary state, not the dominant stream).
        wrapper = self._make_wrapper()
        bad = [
            name
            for name, _ in wrapper.model.named_buffers()
            if name.endswith(".attn.kv_cache")
        ]
        assert bad == [], (
            f"Found legacy `attn.kv_cache` buffers — paged variant must not "
            f"register them: {bad}"
        )

    def test_compute_logits_passthrough(self):
        wrapper = self._make_wrapper()
        h = torch.randn(2, 8)
        out = wrapper.compute_logits(h)
        assert out is h  # V4 fuses LM head into model.forward; passthrough.

    def test_weights_mapper_present(self):
        from atom.models.deepseek_v4_mla import DeepseekV4ForCausalLM_MLA

        wm = DeepseekV4ForCausalLM_MLA.weights_mapper
        # Same prefix renames as legacy V4.
        assert "embed." in wm.orig_to_new_prefix
        assert wm.orig_to_new_prefix["embed."] == "model.embed."


# ── Architecture registration in support_model_arch_dict ───────────────────


class TestArchRegistration:
    def test_arch_registered(self):
        # Source-level grep — model_runner imports a heavy chain so we read
        # the dict via AST instead of importing the module directly.
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / (
            "atom/model_engine/model_runner.py"
        )
        text = src.read_text()
        tree = ast.parse(text)
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Name)
                        and tgt.id == "support_model_arch_dict"
                        and isinstance(node.value, ast.Dict)
                    ):
                        for k, v in zip(node.value.keys, node.value.values):
                            if (
                                isinstance(k, ast.Constant)
                                and k.value == "DeepseekV4ForCausalLM_MLA"
                                and isinstance(v, ast.Constant)
                                and "deepseek_v4_mla" in v.value
                            ):
                                found = True
        assert found, (
            "Expected `DeepseekV4ForCausalLM_MLA` entry pointing at "
            "atom.models.deepseek_v4_mla.* in support_model_arch_dict"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
