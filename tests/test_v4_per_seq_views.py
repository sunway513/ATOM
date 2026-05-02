# SPDX-License-Identifier: MIT
# P7-r18 TD-3 Step A: confirm `v4_per_seq_slots_t` and `v4_per_seq_start_pos_t`
# tensor views attach to AttentionMetaData with the right shape/dtype/contents
# semantics, AND that pinned-CPU CpuGpuBuffer staging satisfies the SOP-6
# capture-replay invariants (stable GPU address; sub-view share with parent).
#
# Steps B/C (Indexer + V4Attention loop) extend this file as they land.
# CPU-only — heavy GPU paths live behind the silicon driver.
#
# Module imports are kept narrow: CpuGpuBuffer is loaded by spec from its
# defining file (atom/utils/__init__.py) to avoid the heavy `transformers`
# import that pulling `atom.utils` directly would trigger on a test host.

from dataclasses import make_dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ── Narrow load of CpuGpuBuffer from atom/utils/__init__.py ─────────────────
_ATOM_ROOT = Path(__file__).resolve().parent.parent
_UTILS_INIT = _ATOM_ROOT / "atom" / "utils" / "__init__.py"


def _load_cpu_gpu_buffer_class():
    """Load just the CpuGpuBuffer class from atom/utils/__init__.py without
    importing the (heavy) rest of the package. Walks the source for the class
    block + execs it with a minimal preamble.
    """
    src = _UTILS_INIT.read_text()
    start = src.find("class CpuGpuBuffer:")
    assert start != -1, "CpuGpuBuffer class not found"
    rest = src[start:]
    lines = rest.splitlines(keepends=True)
    end_idx = len(lines)
    for i, ln in enumerate(lines):
        if i == 0:
            continue
        # First top-level (col-0) non-blank, non-comment, non-decorator that
        # ISN'T part of the class body terminates the block.
        if ln and not ln.startswith((" ", "\t", "#", "\n", "@")):
            end_idx = i
            break
    class_src = "".join(lines[:end_idx])
    preamble = (
        "import numpy as np\n" "import torch\n" "from typing import Optional, Union\n"
    )
    ns: dict = {}
    exec(preamble + class_src, ns)
    return ns["CpuGpuBuffer"]


CpuGpuBuffer = _load_cpu_gpu_buffer_class()


# Light-weight AttentionMetaData stand-in for unit-testing the dataclass
# dynamic-attach contract. Attribute names and dynamic attach behavior match
# atom.utils.forward_context.AttentionMetaData (which is also @dataclass).
AttentionMetaData = make_dataclass(
    "AttentionMetaData",
    [
        ("cu_seqlens_q", Optional[torch.Tensor], None),
        ("max_seqlen_q", int, 0),
    ],
)


def _make_cpu_buffer(n: int) -> "CpuGpuBuffer":
    """CPU-stub CpuGpuBuffer (gpu=cpu device for unit-test mocking;
    pin_memory=False because pinned-host alloc requires CUDA/HIP runtime
    which the test host might not have)."""
    return CpuGpuBuffer(
        n, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False
    )


# ── Step A unit tests ──────────────────────────────────────────────────────


class TestStepAMetadataAttach:
    """Step A: per-seq tensor views must be attachable + correct shape."""

    def test_attention_metadata_accepts_per_seq_slot_tensor(self):
        """AttentionMetaData is @dataclass — dynamic attrs work post-init."""
        md = AttentionMetaData()
        slots_t = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        md.v4_per_seq_slots_t = slots_t
        assert md.v4_per_seq_slots_t.shape == (4,)
        assert md.v4_per_seq_slots_t.dtype == torch.int32
        assert md.v4_per_seq_slots_t.tolist() == [0, 1, 2, 3]

    def test_attention_metadata_accepts_per_seq_start_pos_tensor(self):
        md = AttentionMetaData()
        start_pos_t = torch.tensor([10, 20, 30], dtype=torch.int32)
        md.v4_per_seq_start_pos_t = start_pos_t
        assert md.v4_per_seq_start_pos_t.shape == (3,)
        assert md.v4_per_seq_start_pos_t.dtype == torch.int32
        assert md.v4_per_seq_start_pos_t.tolist() == [10, 20, 30]

    def test_per_seq_view_slice_shares_storage(self):
        """A 1-elem slice `view = slots_t[i:i+1]` MUST share storage with
        the parent — replay updates to the parent must be visible through
        the view. This is the core invariant TD-3 relies on for
        capture-replay correctness (`_idx_state_slot_buf.copy_(slots_t[i:i+1])`
        records a captured memcpy whose source pointer is `slots_t.data + i*4`;
        replay reads whatever the parent buffer holds at that offset)."""
        slots_t = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        view = slots_t[1:2]
        assert view.data_ptr() == slots_t.data_ptr() + 1 * slots_t.element_size()
        slots_t[1] = 99
        assert view.item() == 99


class TestStepACpuGpuBufferStaging:
    """Step A: pinned CpuGpuBuffer is the SOP-6 H2D-staging primitive used to
    populate `v4_per_seq_start_pos_t`. Confirm copy_to_gpu returns a sub-view
    of the same underlying tensor (stable address)."""

    def test_buffer_gpu_address_stable_across_writes(self):
        buf = _make_cpu_buffer(8)
        addr_before = buf.gpu.data_ptr()
        # Two writes — address must NOT change (capture-replay guarantee).
        buf.np[:4] = np.array([1, 2, 3, 4], dtype=np.int32)
        buf.copy_to_gpu(4)
        addr_after_1 = buf.gpu.data_ptr()
        buf.np[:4] = np.array([10, 20, 30, 40], dtype=np.int32)
        buf.copy_to_gpu(4)
        addr_after_2 = buf.gpu.data_ptr()
        assert addr_before == addr_after_1 == addr_after_2

    def test_buffer_copy_to_gpu_sub_slice(self):
        buf = _make_cpu_buffer(8)
        buf.np[:3] = np.array([7, 8, 9], dtype=np.int32)
        out = buf.copy_to_gpu(3)
        assert out.shape == (3,)
        assert out.dtype == torch.int32
        assert out.tolist() == [7, 8, 9]
        # Slice's storage MUST be inside parent buffer (sub-view, not copy).
        assert out.data_ptr() == buf.gpu.data_ptr()


# ── Step B unit tests ──────────────────────────────────────────────────────


class TestStepBIndexerSlotTensor:
    """Step B: Indexer.forward must accept BOTH Python-int slot (eager / legacy)
    AND 1-elem GPU tensor slot (capture path). Validated via a focused mock of
    the slot-ingest branch — the full Indexer.forward needs a loaded model and
    GPU and is silicon-only (covered by the silicon driver, not this UT)."""

    @staticmethod
    def _ingest_slot_branch(slot, state_slot_buf):
        """Mirrors the exact Step-B branch in Indexer.forward (lines 1308-1322
        of atom/models/deepseek_v4.py). Pure function for test isolation."""
        if isinstance(slot, torch.Tensor):
            state_slot_buf.copy_(slot, non_blocking=False)
        else:
            state_slot_buf.fill_(slot)
        return state_slot_buf

    def test_int_slot_uses_fill(self):
        buf = torch.zeros(1, dtype=torch.int32)
        out = self._ingest_slot_branch(7, buf)
        assert out.item() == 7
        # Same buffer (fill_ is in-place).
        assert out.data_ptr() == buf.data_ptr()

    def test_tensor_slot_uses_copy(self):
        buf = torch.zeros(1, dtype=torch.int32)
        slots_t = torch.tensor([0, 5, 11, 17], dtype=torch.int32)
        # Take a 1-elem view (mimics V4Attention.forward `slots_t[i:i+1]`).
        slot_view = slots_t[2:3]
        out = self._ingest_slot_branch(slot_view, buf)
        assert out.item() == 11
        assert out.data_ptr() == buf.data_ptr()  # buf address stable
        # Verify view → buf is true copy_ (mutation of buf doesn't write back to slots_t).
        out.fill_(99)
        assert slots_t.tolist() == [0, 5, 11, 17]

    def test_tensor_slot_replay_invariant(self):
        """The TD-3 capture-replay invariant: at replay-time, the parent
        `state_slot_mapping` buffer is updated by `prepare_decode`. A captured
        copy_(slots_t[i:i+1]) records the source POINTER (parent.data + i*4),
        so replay reads whatever the parent buffer holds at that offset.

        This test simulates that invariant by mutating the parent and checking
        that re-running the slot-ingest produces the new value (proving the
        captured copy_ would do the right thing at replay)."""
        buf = torch.zeros(1, dtype=torch.int32)
        slots_t = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        slot_view = slots_t[1:2]
        # First "replay": slots_t holds [0,1,2,3] → view reads 1.
        self._ingest_slot_branch(slot_view, buf)
        assert buf.item() == 1
        # Mutate parent (simulates prepare_decode writing live values).
        slots_t[1] = 42
        # Second "replay": same captured copy_ op — but parent now holds different
        # value at offset 1. Capture-replay would read 42.
        self._ingest_slot_branch(slot_view, buf)
        assert buf.item() == 42

    def test_int_slot_does_not_break_non_capture_path(self):
        """Eager (non-capture) path always called Indexer with a Python int
        before TD-3. After TD-3 Step B, the int branch must remain functional
        so the eager parity gate stays green."""
        buf = torch.zeros(1, dtype=torch.int32)
        for s in (0, 1, 7, 31):
            self._ingest_slot_branch(s, buf)
            assert buf.item() == s


# ── Step C unit tests ──────────────────────────────────────────────────────


class TestStepCV4AttentionSlotDispatch:
    """Step C: V4Attention.forward must thread a 1-elem GPU tensor view of
    the per-seq slot to Indexer when `attn_md.v4_per_seq_slots_t` is
    available, and gracefully fall back to the Python-int `slot` when it
    isn't. Validated via a focused mock of the dispatch logic — full
    V4Attention.forward needs a loaded model and is silicon-only."""

    @staticmethod
    def _select_indexer_slot(indexer_slots_t, slot_int, seq_idx):
        """Mirrors the Step-C selection branch in V4Attention.forward."""
        if indexer_slots_t is not None:
            return indexer_slots_t[seq_idx : seq_idx + 1]
        return slot_int

    def test_tensor_path_returns_view(self):
        slots_t = torch.tensor([4, 5, 6, 7], dtype=torch.int32)
        out = self._select_indexer_slot(slots_t, slot_int=99, seq_idx=2)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1,)
        assert out.item() == 6
        # Must be a view (not a copy) — share storage with parent at offset 2.
        assert out.data_ptr() == slots_t.data_ptr() + 2 * slots_t.element_size()

    def test_fallback_path_returns_int(self):
        out = self._select_indexer_slot(indexer_slots_t=None, slot_int=99, seq_idx=2)
        assert isinstance(out, int)
        assert out == 99

    def test_view_per_seq_isolation(self):
        """Different seq_idx must produce views into different parent offsets,
        so V4Attention's per-seq loop dispatches the right slot per call."""
        slots_t = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
        view0 = self._select_indexer_slot(slots_t, slot_int=0, seq_idx=0)
        view3 = self._select_indexer_slot(slots_t, slot_int=0, seq_idx=3)
        assert view0.item() == 10 and view3.item() == 40
        assert view0.data_ptr() != view3.data_ptr()
        assert view3.data_ptr() == view0.data_ptr() + 3 * slots_t.element_size()

    def test_metadata_attribute_attach_for_step_c_consumer(self):
        """Confirms the V4Attention loop consumer can `getattr` the field
        attached by Step A's prepare_decode/prefill — keeps Step A and Step
        C in sync at the contract level."""
        md = AttentionMetaData()
        # Field absent before Step A code runs (matches non-V4 backends).
        assert getattr(md, "v4_per_seq_slots_t", None) is None
        # After Step A attach.
        md.v4_per_seq_slots_t = torch.tensor([1, 2, 3], dtype=torch.int32)
        slots_t_consumed = getattr(md, "v4_per_seq_slots_t", None)
        assert slots_t_consumed is not None
        assert slots_t_consumed.tolist() == [1, 2, 3]
