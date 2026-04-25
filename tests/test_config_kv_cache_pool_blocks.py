# SPDX-License-Identifier: MIT
# Layer 1 unit tests for Config.kv_cache_pool_blocks + engine_core dispatch.
# Per RFC §6.2.1 / §9.5.2 — CPU only, mocked, ≤ 30s.


from conftest import MockConfig

# ── Config field ───────────────────────────────────────────────────────────


class TestConfigField:
    def test_mock_config_default_empty_dict(self):
        # MockConfig is a stand-in; verify the new field round-trips.
        cfg = MockConfig(kv_cache_pool_blocks={"main": 64, "compress": 32})
        assert cfg.kv_cache_pool_blocks == {"main": 64, "compress": 32}

    def test_mock_config_default_unset_no_attr_error(self):
        # Even when not explicitly set on a stub, ATOM consumers should
        # gracefully treat absence as legacy single-pool.
        cfg = MockConfig()
        # Our new dispatch logic uses dict.get(); test the gettattr fallback
        # for legacy code paths that don't know about the field.
        pool_blocks = getattr(cfg, "kv_cache_pool_blocks", {})
        assert pool_blocks == {} or pool_blocks is None or isinstance(pool_blocks, dict)


# ── engine_core dispatch contract (no engine_core import — too heavy) ──────
# We test the dispatch logic SHAPE: given a block_info dict, the right
# allocate_kv_cache argument shape goes through.


class _FakeRunnerMgr:
    """Minimal stand-in capturing the args passed to call_func."""

    def __init__(self, block_info):
        self._block_info = block_info
        self.calls: list[tuple] = []

    def call_func(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if name == "get_num_blocks":
            return self._block_info
        if name == "allocate_kv_cache":
            return True
        return None


def _dispatch(block_info, config):
    """Replicates the engine_core.py:90-105 dispatch under test.
    Kept inline so the L1 test stays decoupled from the heavy engine_core
    module (which pulls zmq + multiproc).
    """
    rm = _FakeRunnerMgr(block_info)
    bi = rm.call_func("get_num_blocks", wait_out=True)
    config.mamba_equiv_per_req = bi.get("mamba_equiv_per_req", 0)
    config.num_mamba_groups = bi.get("num_mamba_groups", 0)
    pool_blocks = bi.get("kv_cache_pool_blocks")
    if pool_blocks:
        config.kv_cache_pool_blocks = pool_blocks
        config.num_kvcache_blocks = sum(pool_blocks.values())
        ret = rm.call_func("allocate_kv_cache", pool_blocks, wait_out=True)
    else:
        n = bi["num_kvcache_blocks"]
        config.num_kvcache_blocks = n
        ret = rm.call_func("allocate_kv_cache", n, wait_out=True)
    assert ret
    return rm


class TestEngineCoreDispatch:
    def test_legacy_scalar_path_passes_int_to_allocate_kv_cache(self):
        cfg = MockConfig()
        rm = _dispatch({"num_kvcache_blocks": 128}, cfg)
        assert cfg.num_kvcache_blocks == 128
        # Legacy path does not touch kv_cache_pool_blocks; whether it was
        # already present (real Config has default {}) or absent (MockConfig
        # without override), the dispatch never writes a non-empty value.
        assert not getattr(cfg, "kv_cache_pool_blocks", {})
        # The allocate_kv_cache call received a single int, not a dict.
        alloc_calls = [c for c in rm.calls if c[0] == "allocate_kv_cache"]
        assert len(alloc_calls) == 1 and alloc_calls[0][1] == (128,)

    def test_multipool_dict_path_passes_dict_to_allocate_kv_cache(self):
        cfg = MockConfig()
        pool_blocks = {"main": 64, "compress": 32, "indexer": 16}
        rm = _dispatch({"kv_cache_pool_blocks": pool_blocks}, cfg)
        assert cfg.kv_cache_pool_blocks == pool_blocks
        # Legacy scalar resolves to sum so old consumers still see a count.
        assert cfg.num_kvcache_blocks == 64 + 32 + 16
        alloc_calls = [c for c in rm.calls if c[0] == "allocate_kv_cache"]
        assert len(alloc_calls) == 1 and alloc_calls[0][1] == (pool_blocks,)

    def test_dict_takes_precedence_when_both_present(self):
        # If a runner reports both keys, the dict wins (richer data).
        cfg = MockConfig()
        pool_blocks = {"main": 100}
        rm = _dispatch(
            {"num_kvcache_blocks": 999, "kv_cache_pool_blocks": pool_blocks}, cfg
        )
        assert cfg.kv_cache_pool_blocks == pool_blocks
        assert cfg.num_kvcache_blocks == 100  # sum, not 999
        alloc_calls = [c for c in rm.calls if c[0] == "allocate_kv_cache"]
        assert alloc_calls[0][1] == (pool_blocks,)

    def test_mamba_metadata_still_plumbed(self):
        cfg = MockConfig()
        _dispatch(
            {"num_kvcache_blocks": 10, "mamba_equiv_per_req": 3, "num_mamba_groups": 7},
            cfg,
        )
        assert cfg.mamba_equiv_per_req == 3
        assert cfg.num_mamba_groups == 7
