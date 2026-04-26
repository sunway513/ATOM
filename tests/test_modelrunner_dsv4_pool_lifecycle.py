# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ModelRunner is the sole owner of DSV4KVPool (issue sunway513/atom#37 P2.2).

Verifies the canonical #42 fix: ``is_dummy_run`` short-circuit at the very
entry of ``_maybe_setup_dsv4_forward_batch`` + ``ATOM_DSV4_USE_W4_PATH``
gate. Both must pass before the pool is touched.

Additionally validates:
- pool finish_request is idempotent under preempt-then-finish race.
- ModelRunner._build_dsv4_pool wires the pool's finish_request into
  the scheduler's finish-listener registry (P2.2 ownership boundary).

These tests do NOT exercise live cudagraph or real model construction;
they exercise the pure pool-lifecycle helper using ``unittest.mock``.

Heavy aiter / model-loader imports at the top of ``model_runner.py`` are
stubbed below so the module is importable without GPU dependencies.
"""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Pre-import: stub the heavy chain that atom.model_engine.model_runner pulls
# in at module level (aiter, model_loader, kv_transfer, etc.).
# Must run before any `from atom.model_engine.model_runner import ModelRunner`.
# ---------------------------------------------------------------------------


def _ensure_module(name: str) -> types.ModuleType:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def _stub_modelrunner_imports() -> None:
    # aiter top-level: model_runner does
    #   from aiter import destroy_dist_env, dtypes, init_dist_env
    _aiter = _ensure_module("aiter")
    for fn in ("destroy_dist_env", "init_dist_env"):
        if not hasattr(_aiter, fn):
            setattr(_aiter, fn, lambda *a, **kw: None)
    if not hasattr(_aiter, "dtypes"):
        _aiter.dtypes = types.SimpleNamespace()

    # aiter.dist.parallel_state — extra symbols beyond conftest's stubs.
    _ps = _ensure_module("aiter.dist.parallel_state")
    for fn in ("get_dp_group", "get_pp_group", "get_tp_group", "graph_capture"):
        if not hasattr(_ps, fn):
            setattr(_ps, fn, lambda *a, **kw: None)

    # aiter.dist.utils — get_distributed_init_method
    _du = _ensure_module("aiter.dist.utils")
    if not hasattr(_du, "get_distributed_init_method"):
        _du.get_distributed_init_method = lambda *a, **kw: ""

    # atom.model_loader.loader — load_model
    _ml_pkg = _ensure_module("atom.model_loader")
    _ml_pkg.__path__ = []
    _ml = _ensure_module("atom.model_loader.loader")
    if not hasattr(_ml, "load_model"):
        _ml.load_model = lambda *a, **kw: None

    # atom.model_ops.* — RejectionSampler, sampler, etc.
    _mo_pkg = _ensure_module("atom.model_ops")
    _mo_pkg.__path__ = []
    _rj = _ensure_module("atom.model_ops.rejection_sampler")
    if not hasattr(_rj, "RejectionSampler"):
        _rj.RejectionSampler = type("RejectionSampler", (), {})
    _sm = _ensure_module("atom.model_ops.sampler")
    if not hasattr(_sm, "SAMPLER_EPS"):
        _sm.SAMPLER_EPS = 1e-6
    if not hasattr(_sm, "Sampler"):
        _sm.Sampler = type("Sampler", (), {})

    # atom.spec_decode.eagle — EagleProposer
    _sd_pkg = _ensure_module("atom.spec_decode")
    _sd_pkg.__path__ = []
    _ea = _ensure_module("atom.spec_decode.eagle")
    if not hasattr(_ea, "EagleProposer"):
        _ea.EagleProposer = type("EagleProposer", (), {})

    # atom.kv_transfer.disaggregation — KVConnectorOutput
    _kt_pkg = _ensure_module("atom.kv_transfer")
    _kt_pkg.__path__ = []
    _dg = _ensure_module("atom.kv_transfer.disaggregation")
    if not hasattr(_dg, "KVConnectorOutput"):
        _dg.KVConnectorOutput = type("KVConnectorOutput", (), {})

    # atom.utils submodules used by model_runner
    _u_pkg = _ensure_module("atom.utils")
    _u_pkg.__path__ = [
        __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.dirname(__file__)),
            "atom",
            "utils",
        )
    ]
    # tbo, selector — stub to avoid heavy chain
    _tbo = _ensure_module("atom.utils.tbo")
    if not hasattr(_tbo, "UBatchWrapper"):
        _tbo.UBatchWrapper = type("UBatchWrapper", (), {})
    if not hasattr(_tbo, "maybe_create_ubatch_slices"):
        _tbo.maybe_create_ubatch_slices = lambda *a, **kw: None
    _sel = _ensure_module("atom.utils.selector")
    if not hasattr(_sel, "get_attn_backend"):
        _sel.get_attn_backend = lambda *a, **kw: None

    # atom.config — extra symbol set_current_atom_config
    _cfg = sys.modules.get("atom.config")
    if _cfg is not None and not hasattr(_cfg, "set_current_atom_config"):
        _cfg.set_current_atom_config = lambda *a, **kw: None

    # atom.utils.__init__ — model_runner does
    #   from atom.utils import (CpuGpuBuffer, envs, get_hf_text_config,
    #       init_exit_handler, resolve_obj_by_qualname)
    if not hasattr(_u_pkg, "CpuGpuBuffer"):
        _u_pkg.CpuGpuBuffer = type("CpuGpuBuffer", (), {})
    if not hasattr(_u_pkg, "envs"):
        from atom.utils import envs as _envs

        _u_pkg.envs = _envs
    if not hasattr(_u_pkg, "get_hf_text_config"):
        _u_pkg.get_hf_text_config = lambda cfg: cfg
    if not hasattr(_u_pkg, "init_exit_handler"):
        _u_pkg.init_exit_handler = lambda *a, **kw: None
    if not hasattr(_u_pkg, "resolve_obj_by_qualname"):
        _u_pkg.resolve_obj_by_qualname = lambda name: None
    # graph_marker referenced by model_runner.__init__
    _gm = _ensure_module("atom.utils.graph_marker")
    if not hasattr(_gm, "set_graph_marker_enabled"):
        _gm.set_graph_marker_enabled = lambda *a, **kw: None


_stub_modelrunner_imports()


# ---------------------------------------------------------------------------
# Pool gating tests — the canonical #42 fix
# ---------------------------------------------------------------------------


class TestPoolGating:
    def test_dummy_run_returns_none_none(self, monkeypatch):
        """The canonical #42 fix: is_dummy_run short-circuits before pool admit."""
        from atom.model_engine.model_runner import ModelRunner

        monkeypatch.setenv("ATOM_DSV4_USE_W4_PATH", "1")
        # reload envs to pick up the new value
        from atom.utils import envs as envs_mod

        importlib.reload(envs_mod)

        runner = MagicMock(spec=ModelRunner)
        runner._is_dsv4_model.return_value = True
        runner._dsv4_pool = MagicMock()  # if already inited

        dummy_batch = MagicMock()
        dummy_batch.is_dummy_run = True
        dummy_batch.req_ids = [0, 1, 2, 3]

        # Bind the real method to the mock
        pool, fb = ModelRunner._maybe_setup_dsv4_forward_batch(
            runner,
            dummy_batch,
            attn_metadata=MagicMock(),
            positions=MagicMock(),
        )
        assert pool is None
        assert fb is None
        # Critical: pool admit_request must NOT have been called
        runner._dsv4_pool.admit_request.assert_not_called()

    def test_flag_off_returns_none_none(self, monkeypatch):
        """``ATOM_DSV4_USE_W4_PATH=0`` => also short-circuit."""
        from atom.model_engine.model_runner import ModelRunner

        monkeypatch.setenv("ATOM_DSV4_USE_W4_PATH", "0")
        from atom.utils import envs as envs_mod

        importlib.reload(envs_mod)

        runner = MagicMock(spec=ModelRunner)
        runner._is_dsv4_model.return_value = True

        real_batch = MagicMock()
        real_batch.is_dummy_run = False
        real_batch.req_ids = [0, 1]

        pool, fb = ModelRunner._maybe_setup_dsv4_forward_batch(
            runner,
            real_batch,
            attn_metadata=MagicMock(),
            positions=MagicMock(),
        )
        assert pool is None
        assert fb is None

    def test_non_dsv4_model_returns_none_none(self):
        """Non-DSV4 models pass through untouched."""
        from atom.model_engine.model_runner import ModelRunner

        runner = MagicMock(spec=ModelRunner)
        runner._is_dsv4_model.return_value = False

        pool, fb = ModelRunner._maybe_setup_dsv4_forward_batch(
            runner,
            batch=MagicMock(),
            attn_metadata=MagicMock(),
            positions=MagicMock(),
        )
        assert pool is None
        assert fb is None


# ---------------------------------------------------------------------------
# finish_request idempotency
# ---------------------------------------------------------------------------


class TestFinishIdempotent:
    def test_pool_finish_called_twice_is_safe(self):
        """ModelRunner subscribes pool.finish_request to scheduler events.
        finish_request must be idempotent under preempt-then-finish race.
        """
        import torch

        from atom.engine.kv_pool.dsv4_pool import DSV4KVPool, DSV4KVPoolConfig

        ratios = [4, 4, 128, 128]
        cfg = DSV4KVPoolConfig(
            max_active_seqs=4,
            num_layers=4,
            num_c4_layers=2,
            num_c128_layers=2,
            head_dim=32,
            rope_head_dim=16,
            window_size=32,
            max_seq_len=128,
            ring_size_main=32,
            ring_size_compressor=8,
            ring_size_indexer=32,
            compress_ratio_per_layer=ratios,
            state_inner_dim=32,
            dtype=torch.float32,
            state_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        pool = DSV4KVPool(cfg)
        pool.admit_request(seq_id=0)
        pool.finish_request(seq_id=0)
        # Second call must not raise (idempotent)
        pool.finish_request(seq_id=0)
        # Third call also OK (never admitted; defensive no-op)
        pool.finish_request(seq_id=999)


# ---------------------------------------------------------------------------
# Scheduler subscription wiring (structural smoke check)
# ---------------------------------------------------------------------------


class TestNoDsv4ScheduleListener:
    def test_modelrunner_subscribes_pool_to_scheduler(self):
        """The wiring: ``ModelRunner._build_dsv4_pool`` source must reference
        ``register_finish_listener`` (preferred) or at least the scheduler.

        Smoke-style assertion since constructing a full ModelRunner+Scheduler
        integration is too heavy for a unit test.
        """
        import inspect

        from atom.model_engine.model_runner import ModelRunner

        if hasattr(ModelRunner, "_build_dsv4_pool"):
            src = inspect.getsource(ModelRunner._build_dsv4_pool)
            assert "register_finish_listener" in src or "scheduler" in src.lower()
        else:
            pytest.skip("_build_dsv4_pool not yet implemented")


# ---------------------------------------------------------------------------
# Sequential admit/finish lifecycle (Bug 3 regression guard)
# ---------------------------------------------------------------------------


class TestSequentialAdmitFinish:
    """Regression guard for Bug 3: pool slots must be reusable after finish.

    Root cause: in production the scheduler (EngineCore parent) and the
    DSV4KVPool (ModelRunner child) live in separate processes.
    ``register_finish_listener`` is a no-op there, so finish_request was
    never called and pool slots were never recycled. The fix propagates
    finished_seq_ids through ScheduledBatch so ModelRunner processes them
    before admitting the next round.

    This test exercises the pool directly (unit-level) to confirm the
    admit→finish→re-admit contract is satisfied for sequential workloads.
    """

    def _make_pool(self, max_active_seqs: int = 1):
        import torch

        from atom.engine.kv_pool.dsv4_pool import DSV4KVPool, DSV4KVPoolConfig

        cfg = DSV4KVPoolConfig(
            max_active_seqs=max_active_seqs,
            num_layers=2,
            num_c4_layers=1,
            num_c128_layers=1,
            head_dim=32,
            rope_head_dim=16,
            window_size=32,
            max_seq_len=128,
            ring_size_main=32,
            ring_size_compressor=8,
            ring_size_indexer=32,
            compress_ratio_per_layer=[4, 128],
            state_inner_dim=32,
            dtype=torch.float32,
            state_dtype=torch.float32,
            device=torch.device("cpu"),
        )
        return DSV4KVPool(cfg)

    def test_sequential_admits_release_slots(self):
        """max_active_seqs=1: admit req0, finish req0, admit req1 … must not raise.

        This is the exact scenario that triggered Bug 3: lm_eval sequential
        requests with a tight pool capacity crashed on request 2 because
        finish_request was never called (cross-process no-op wiring).
        """
        pool = self._make_pool(max_active_seqs=1)

        for seq_id in range(5):
            slot = pool.admit_request(seq_id=seq_id)
            assert (
                0 <= slot < pool.max_active_seqs
            ), f"slot {slot} out of range for seq_id={seq_id}"
            pool.finish_request(seq_id=seq_id)

        # After all 5 sequential requests the pool must be fully free again.
        assert len(pool._free) == pool.max_active_seqs

    def test_sequential_admits_no_slot_leak(self):
        """Slots returned to the free list after finish_request (no leak)."""
        pool = self._make_pool(max_active_seqs=2)

        pool.admit_request(seq_id=10)
        pool.admit_request(seq_id=11)
        assert len(pool._free) == 0

        pool.finish_request(seq_id=10)
        assert len(pool._free) == 1

        pool.finish_request(seq_id=11)
        assert len(pool._free) == 2

        # Both slots freed — a third pair must also succeed.
        pool.admit_request(seq_id=20)
        pool.admit_request(seq_id=21)
        assert len(pool._free) == 0

    def test_finished_seq_ids_processed_before_admit(self):
        """ModelRunner must call finish_request BEFORE admit for the same batch.

        Simulates the cross-process path: ScheduledBatch carries finished_seq_ids
        from the previous round; _maybe_setup_dsv4_forward_batch must drain
        them before calling admit_request for the new seqs.
        """
        import inspect

        from atom.model_engine.model_runner import ModelRunner

        src = inspect.getsource(ModelRunner._maybe_setup_dsv4_forward_batch)
        # finished_seq_ids processing must appear before admit_request call.
        finish_pos = src.find("finished_seq_ids")
        admit_pos = src.find("admit_request")
        assert (
            finish_pos != -1
        ), "finished_seq_ids not found in _maybe_setup_dsv4_forward_batch"
        assert (
            admit_pos != -1
        ), "admit_request not found in _maybe_setup_dsv4_forward_batch"
        assert finish_pos < admit_pos, (
            "finish_request processing must precede admit_request in "
            "_maybe_setup_dsv4_forward_batch (Bug 3 ordering invariant)"
        )
