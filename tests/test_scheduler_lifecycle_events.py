"""Scheduler emits seq lifecycle events via callback registry (issue #37 W4.3 P2.2).

Establishes the ownership boundary: Scheduler does not know DSV4KVPool
exists; ModelRunner subscribes pool.admit_request / pool.finish_request
to these events.
"""

from unittest.mock import MagicMock


class TestEventRegistry:
    def test_admit_listener_registry_exists(self, scheduler):
        """Public API: register_admit_listener(fn) appends fn to the registry."""
        listener = MagicMock()
        scheduler.register_admit_listener(listener)
        assert listener in scheduler._admit_listeners

    def test_finish_listener_registry_exists(self, scheduler):
        listener = MagicMock()
        scheduler.register_finish_listener(listener)
        assert listener in scheduler._finish_listeners

    def test_emit_admit_calls_all_admit_listeners(self, scheduler):
        l1, l2 = MagicMock(), MagicMock()
        scheduler.register_admit_listener(l1)
        scheduler.register_admit_listener(l2)
        scheduler._emit_admit(seq_id=42)
        l1.assert_called_once_with(42)
        l2.assert_called_once_with(42)

    def test_emit_finish_calls_all_finish_listeners(self, scheduler):
        l1, l2 = MagicMock(), MagicMock()
        scheduler.register_finish_listener(l1)
        scheduler.register_finish_listener(l2)
        scheduler._emit_finish(seq_id=99)
        l1.assert_called_once_with(99)
        l2.assert_called_once_with(99)

    def test_listener_exception_does_not_propagate(self, scheduler):
        """A bad listener cannot break the scheduler."""
        bad = MagicMock(side_effect=RuntimeError("oops"))
        good = MagicMock()
        scheduler.register_admit_listener(bad)
        scheduler.register_admit_listener(good)
        # Must not raise
        scheduler._emit_admit(seq_id=1)
        good.assert_called_once_with(1)


class TestNoDsv4PoolImport:
    def test_scheduler_source_does_not_reference_dsv4_pool(self):
        """Ownership boundary check: scheduler.py must not import DSV4KVPool.

        ModelRunner subscribes the pool's admit/finish methods to the
        scheduler's events. The scheduler itself is pool-agnostic.
        """
        import atom.model_engine.scheduler as sched_mod

        with open(sched_mod.__file__) as f:
            src = f.read()
        # The scheduler may have generic listener bookkeeping but must not
        # know about DSV4KVPool specifically.
        assert "DSV4KVPool" not in src
        assert "dsv4_pool" not in src.lower()


class TestPendingFinishIdsPipeline:
    """Bug 3 regression: _emit_finish must accumulate into _pending_finish_ids
    so that ScheduledBatch.finished_seq_ids carries them cross-process.
    """

    def test_emit_finish_appends_to_pending(self, scheduler):
        """_emit_finish now accumulates seq_id in _pending_finish_ids."""
        assert hasattr(
            scheduler, "_pending_finish_ids"
        ), "_pending_finish_ids missing — Bug 3 fix not applied to Scheduler"
        scheduler._emit_finish(seq_id=7)
        scheduler._emit_finish(seq_id=8)
        assert 7 in scheduler._pending_finish_ids
        assert 8 in scheduler._pending_finish_ids

    def test_emit_finish_still_calls_listeners(self, scheduler):
        """Accumulation must not suppress existing listener dispatch."""
        from unittest.mock import MagicMock

        cb = MagicMock()
        scheduler.register_finish_listener(cb)
        scheduler._emit_finish(seq_id=42)
        cb.assert_called_once_with(42)
        assert 42 in scheduler._pending_finish_ids

    def test_scheduled_batch_has_finished_seq_ids_field(self):
        """ScheduledBatch must accept and expose finished_seq_ids."""
        from atom.model_engine.scheduler import ScheduledBatch

        batch = ScheduledBatch(
            seqs={},
            num_scheduled_tokens=[],
            total_tokens_num=0,
            finished_seq_ids=[3, 5],
        )
        assert batch.finished_seq_ids == [3, 5]

    def test_scheduled_batch_finished_seq_ids_defaults_to_empty(self):
        """Omitting finished_seq_ids yields an empty list (not None)."""
        from atom.model_engine.scheduler import ScheduledBatch

        batch = ScheduledBatch(
            seqs={},
            num_scheduled_tokens=[],
            total_tokens_num=0,
        )
        assert batch.finished_seq_ids == []
