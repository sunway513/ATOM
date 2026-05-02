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


# ── r19b unit tests ────────────────────────────────────────────────────────


class TestR19bAttnSlotsHoist:
    """r19b: hoist a general `attn_slots_t` (independent of indexer presence)
    so SWA-only and HCA-only layers also benefit from capture-safe slot
    threading. `indexer_slots_t` becomes a derived alias gated by indexer
    presence (preserves Step C behavior on CSA layers)."""

    @staticmethod
    def _select_attn_slots(
        candidate, num_seqs, attn_md_present, is_dummy_run, dtype, is_cuda
    ):
        """Mirrors the r19b master `attn_slots_t` selection branch."""
        if not (num_seqs > 0 and attn_md_present and not is_dummy_run):
            return None
        if candidate is None:
            return None
        if not (is_cuda and dtype == torch.int32 and candidate.numel() >= num_seqs):
            return None
        return candidate

    @staticmethod
    def _derive_indexer_slots(attn_slots_t, compress_ratio, has_indexer):
        return (
            attn_slots_t
            if (attn_slots_t is not None and compress_ratio and has_indexer)
            else None
        )

    def test_attn_slots_set_for_swa_only_layer(self):
        """SWA-only layer (compress_ratio=0, no indexer): attn_slots_t MUST
        still be populated so the SWA write/read sites get the capture-safe
        path. indexer_slots_t MUST be None (no indexer to feed)."""
        slots = torch.tensor([3, 7, 11], dtype=torch.int32)
        attn = self._select_attn_slots(
            slots,
            num_seqs=3,
            attn_md_present=True,
            is_dummy_run=False,
            dtype=torch.int32,
            is_cuda=True,
        )
        idx = self._derive_indexer_slots(attn, compress_ratio=0, has_indexer=False)
        assert attn is slots
        assert idx is None

    def test_attn_slots_set_for_csa_layer_with_indexer(self):
        """CSA layer (compress_ratio=4, indexer): both attn_slots_t AND
        indexer_slots_t alias the same parent — slicing them per-seq must
        return identical views."""
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        attn = self._select_attn_slots(
            slots,
            num_seqs=4,
            attn_md_present=True,
            is_dummy_run=False,
            dtype=torch.int32,
            is_cuda=True,
        )
        idx = self._derive_indexer_slots(attn, compress_ratio=4, has_indexer=True)
        assert attn is slots and idx is slots
        for i in range(4):
            assert attn[i : i + 1].data_ptr() == idx[i : i + 1].data_ptr()

    def test_dummy_run_disables_both(self):
        """Warmup / dummy run: neither attn_slots_t nor indexer_slots_t is
        populated — both forwards must fall back to Python-int paths so
        warmup (which uses sentinel slot=0) doesn't crash on shape mismatches."""
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        attn = self._select_attn_slots(
            slots,
            num_seqs=4,
            attn_md_present=True,
            is_dummy_run=True,
            dtype=torch.int32,
            is_cuda=True,
        )
        idx = self._derive_indexer_slots(attn, compress_ratio=4, has_indexer=True)
        assert attn is None and idx is None

    def test_undersized_metadata_disables_path(self):
        """Defensive: if the metadata tensor has fewer rows than num_seqs
        (should not happen but guard against builder bugs), fall back to
        Python-int. This prevents OOB index_select / silent index wrap."""
        slots = torch.tensor([0, 1], dtype=torch.int32)  # only 2 rows
        attn = self._select_attn_slots(
            slots,
            num_seqs=4,
            attn_md_present=True,
            is_dummy_run=False,
            dtype=torch.int32,
            is_cuda=True,
        )
        assert attn is None

    def test_wrong_dtype_disables_path(self):
        """Defensive: wrong dtype → fall back. The Triton kernels require
        int32 slot tensors; int64 would silently misindex."""
        slots = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        attn = self._select_attn_slots(
            slots,
            num_seqs=4,
            attn_md_present=True,
            is_dummy_run=False,
            dtype=torch.int64,
            is_cuda=True,
        )
        assert attn is None


class TestR19bSwaWriteCapturePath:
    """r19b: SWA decode write at line ~1932 — must use the `swa_write` Triton
    kernel (capture-safe) when attn_slots_t is available. This is mirrored
    by a pure-PyTorch reference for hermetic CPU testing of the math."""

    @staticmethod
    def _swa_write_ref(seq_kv, seq_positions, slot_view, swa_kv, win):
        """Pure-PyTorch equivalent of the r19b capture-path branch.

        Writes `swa_kv[slot, pos[t] % win, :] = seq_kv_flat[t, :]` for every
        token t in seq_positions. Mirrors `swa_write_reference` in
        atom/model_ops/v4_kernels/state_writes.py."""
        token_num = seq_positions.shape[0]
        seq_kv_flat = seq_kv.squeeze(0).contiguous()  # [token_num, head_dim]
        slot_per_token = slot_view.expand(token_num)  # [token_num]
        ring_idx = seq_positions % win
        swa_kv[slot_per_token.long(), ring_idx.long()] = seq_kv_flat

    def test_decode_write_lands_in_correct_ring_slot(self):
        """Decode token_num=1: write must land at swa_kv[live_slot, pos%win]."""
        win = 8
        head_dim = 4
        num_slots = 4
        swa_kv = torch.zeros(num_slots, win, head_dim)
        slots_t = torch.tensor([0, 5, 11, 7], dtype=torch.int32).clamp(
            max=num_slots - 1
        )
        # seq_idx=2 → live slot = clamped to slots_t[2]=2 (clamp due to test
        # buffer size; in real V4 the slot range is bounded by num_slots).
        seq_idx = 2
        slot_view = slots_t[seq_idx : seq_idx + 1]
        live_slot = int(slot_view.item())
        seq_positions = torch.tensor([13], dtype=torch.int64)  # decode pos
        seq_kv = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # [1, 1, head_dim]
        self._swa_write_ref(seq_kv, seq_positions, slot_view, swa_kv, win)
        assert swa_kv[live_slot, 13 % win].tolist() == [1.0, 2.0, 3.0, 4.0]
        # Other slots unchanged.
        for s in range(num_slots):
            if s != live_slot:
                assert swa_kv[s].abs().sum().item() == 0.0

    def test_per_seq_replay_invariant_via_slot_view(self):
        """Capture-replay invariant: a captured swa_write op records the
        slot_view's parent pointer + offset. At replay, the parent buffer is
        updated by prepare_decode → the same captured op writes to the new
        slot. Mirrored hermetically by mutating the parent before the second
        write call."""
        win = 4
        head_dim = 2
        num_slots = 4
        swa_kv = torch.zeros(num_slots, win, head_dim)
        slots_t = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        slot_view = slots_t[1:2]
        seq_positions = torch.tensor([1], dtype=torch.int64)

        # First "replay": parent[1]=0 → write to swa_kv[0, 1].
        seq_kv = torch.tensor([[[1.0, 1.0]]])
        self._swa_write_ref(seq_kv, seq_positions, slot_view, swa_kv, win)
        assert swa_kv[0, 1].tolist() == [1.0, 1.0]
        # Second "replay" with parent mutated to slot=2.
        slots_t[1] = 2
        seq_kv2 = torch.tensor([[[7.0, 7.0]]])
        self._swa_write_ref(seq_kv2, seq_positions, slot_view, swa_kv, win)
        assert swa_kv[2, 1].tolist() == [7.0, 7.0]
        # Original write still visible at swa_kv[0, 1] (ring not corrupted).
        assert swa_kv[0, 1].tolist() == [1.0, 1.0]


class TestR19bKvSaIndexSelect:
    """r19b: kv_sa SWA-window read at lines ~1968/1971/1973 — must use
    `index_select(0, slot_view)` instead of `swa_kv[slot:slot+1]` (Python-int
    bake). Verify shape and per-replay invariant."""

    def test_index_select_shape_matches_slice(self):
        """`swa_kv.index_select(0, slot_view)` returns shape `[1, n_win,
        head_dim]` — IDENTICAL to legacy `swa_kv[slot:slot+1]`. Drop-in
        replacement (no downstream concat shape adjustment needed)."""
        win = 8
        head_dim = 4
        num_slots = 4
        swa_kv = torch.arange(num_slots * win * head_dim, dtype=torch.float32).reshape(
            num_slots, win, head_dim
        )
        slots_t = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        seq_idx = 2
        legacy = swa_kv[seq_idx : seq_idx + 1]
        new = swa_kv.index_select(0, slots_t[seq_idx : seq_idx + 1].long())
        assert legacy.shape == new.shape == (1, win, head_dim)
        assert torch.equal(legacy, new)

    def test_index_select_per_replay_invariant(self):
        """Mutate the parent slots_t between two index_select calls — the
        second call MUST read the new slot (capture replay simulation)."""
        # Hardcoded swa_kv: shape [num_slots=4, win=2, head_dim=1].
        swa_kv = torch.tensor(
            [[[10.0], [11.0]], [[20.0], [21.0]], [[30.0], [31.0]], [[40.0], [41.0]]]
        )
        slots_t = torch.tensor([0, 1, 0, 0], dtype=torch.int32)
        slot_view = slots_t[1:2]
        first = swa_kv.index_select(0, slot_view.long())
        assert first[0].tolist() == [[20.0], [21.0]]  # slot=1 row
        slots_t[1] = 3
        second = swa_kv.index_select(0, slot_view.long())
        assert second[0].tolist() == [[40.0], [41.0]]  # slot=3 row (live value)

    def test_index_select_int32_to_long_cast(self):
        """`swa_kv.index_select` requires long index. `attn_slots_t` is
        int32; the .long() cast is necessary. Verify the cast preserves
        value semantics for all valid slot ranges."""
        slots_t = torch.tensor([0, 5, 11, 17, 31], dtype=torch.int32)
        for i in range(slots_t.numel()):
            v_int32 = int(slots_t[i].item())
            v_long = int(slots_t[i : i + 1].long().item())
            assert v_int32 == v_long


# ── r20a unit tests ────────────────────────────────────────────────────────


class TestR20aSentinelHelpers:
    """r20a: capture-stable plan format uses `position = -1` sentinel rows
    for tail (over-launched) kernel programs. Verify the helper that fills
    these sentinels into pinned-CPU plan staging buffers."""

    @staticmethod
    def _load_compress_plan_module():
        """Minimal-import load of compress_plan.py (avoids the heavy aiter
        chain pulled in by `import atom.model_ops.v4_kernels`)."""
        import importlib.util
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        path = repo / "atom" / "model_ops" / "v4_kernels" / "compress_plan.py"
        spec = importlib.util.spec_from_file_location("_test_compress_plan", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_sentinel_constant_is_negative(self):
        """Sentinel value MUST be < 0 so the kernel `position < 0` early-
        return fires correctly. Anything >= 0 would be interpreted as a real
        absolute token position."""
        mod = self._load_compress_plan_module()
        assert mod._INVALID_POSITION_SENTINEL < 0

    def test_fill_invalid_rows_writes_sentinel_into_position_column(self):
        """`fill_invalid_plan_rows_inplace_np(plan_np, start, end)` MUST set
        plan_np[start:end, 2] = -1 and zero the other columns (belt + braces
        for any speculative pre-load before the kernel's early-return)."""
        mod = self._load_compress_plan_module()
        plan_np = np.full((6, 4), 7, dtype=np.int32)  # poison initial state
        mod.fill_invalid_plan_rows_inplace_np(plan_np, 2, 5)
        # Rows [2..5) zeroed except col 2 = -1.
        for r in range(2, 5):
            assert plan_np[r, 0] == 0
            assert plan_np[r, 1] == 0
            assert plan_np[r, 2] == -1
            assert plan_np[r, 3] == 0
        # Rows outside the [start, end) range untouched.
        for r in (0, 1, 5):
            for c in range(4):
                assert plan_np[r, c] == 7

    def test_fill_invalid_rows_handles_empty_range(self):
        """end <= start: noop (plan_np unchanged)."""
        mod = self._load_compress_plan_module()
        plan_np = np.full((4, 4), 9, dtype=np.int32)
        snapshot = plan_np.copy()
        mod.fill_invalid_plan_rows_inplace_np(plan_np, 3, 3)  # empty
        mod.fill_invalid_plan_rows_inplace_np(plan_np, 4, 1)  # inverted
        assert np.array_equal(plan_np, snapshot)

    def test_fill_all_invalid_rows_marks_full_buffer(self):
        """Convenience helper for __init__-time setup."""
        mod = self._load_compress_plan_module()
        plan_np = np.full((5, 4), 3, dtype=np.int32)
        mod.fill_all_invalid_plan_rows_inplace_np(plan_np)
        assert (plan_np[:, 2] == -1).all()
        assert (plan_np[:, 0] == 0).all()
        assert (plan_np[:, 1] == 0).all()
        assert (plan_np[:, 3] == 0).all()


class TestR20aCompressPlanCapacity:
    """r20a: CompressPlan grew two optional fields:
    `compress_capacity` and `write_capacity`. When set, the kernel wrappers
    must use them as the launch grid (instead of num_compress / num_write).
    """

    @staticmethod
    def _load_compress_plan_module():
        """See TestR20aSentinelHelpers._load_compress_plan_module."""
        import importlib.util
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        path = repo / "atom" / "model_ops" / "v4_kernels" / "compress_plan.py"
        spec = importlib.util.spec_from_file_location("_test_compress_plan_cap", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_compress_plan_default_capacity_is_none(self):
        """Backward compat: legacy CompressPlan constructions MUST keep
        compress_capacity / write_capacity as None (signalling: use the
        live `num_compress` / `num_write` as the kernel grid)."""
        mod = self._load_compress_plan_module()
        # Mock plan tensor (CPU, since we don't need a GPU for the field check).
        empty = torch.empty((0, 4), dtype=torch.int32)
        cu = np.zeros(2, dtype=np.int32)
        plan = mod.CompressPlan(
            compress_plan_gpu=empty,
            write_plan_gpu=empty,
            num_compress=0,
            num_write=0,
            cu_compress_cpu=cu,
        )
        assert plan.compress_capacity is None
        assert plan.write_capacity is None

    def test_compress_plan_accepts_explicit_capacities(self):
        """Capture-stable construction: capacities passed as kwargs survive."""
        mod = self._load_compress_plan_module()
        empty = torch.empty((0, 4), dtype=torch.int32)
        cu = np.zeros(2, dtype=np.int32)
        plan = mod.CompressPlan(
            compress_plan_gpu=empty,
            write_plan_gpu=empty,
            num_compress=0,
            num_write=0,
            cu_compress_cpu=cu,
            compress_capacity=257,
            write_capacity=8,
        )
        assert plan.compress_capacity == 257
        assert plan.write_capacity == 8


class TestR20aPlanBuilderTailSentinel:
    """r20a: simulate Indexer's `_build_bs1_compress_plan` body — the live
    rows must be in front, the tail must be sentinel, and the live count
    derived from the per-fwd start_pos + token_num is invariant to the
    capacity (= a previous fwd's larger n_compress can NOT leak into the
    new fwd's tail)."""

    @staticmethod
    def _load_compress_plan_module():
        import importlib.util
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        path = repo / "atom" / "model_ops" / "v4_kernels" / "compress_plan.py"
        spec = importlib.util.spec_from_file_location("_test_compress_plan_tail", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _build_plan(self, plan_np, start_pos, token_num, ratio, is_overlap, cap, mod):
        """Mirror of `Indexer._build_bs1_compress_plan` body for the
        compress_plan tensor (write_plan is identical pattern; this UT
        focuses on the compress path)."""
        K = ratio * (2 if is_overlap else 1)
        ragged_ids = np.arange(token_num, dtype=np.int32)
        positions_np = (start_pos + ragged_ids).astype(np.int32)
        window_lens = np.maximum(0, K - np.minimum(ragged_ids + 1, K)).astype(np.int32)
        compress_mask = (positions_np + 1) % ratio == 0
        n_compress = int(compress_mask.sum())
        if n_compress > 0:
            plan_np[:n_compress, 0] = ragged_ids[compress_mask]
            plan_np[:n_compress, 1] = 0
            plan_np[:n_compress, 2] = positions_np[compress_mask]
            plan_np[:n_compress, 3] = window_lens[compress_mask]
        mod.fill_invalid_plan_rows_inplace_np(plan_np, n_compress, cap)
        return n_compress

    def test_tail_sentinel_after_first_fwd(self):
        """Single forward (start_pos large, token_num=1, decode-style):
        tail rows MUST be sentinel; live row carries real position."""
        mod = self._load_compress_plan_module()
        cap = 32
        plan_np = np.zeros((cap, 4), dtype=np.int32)
        mod.fill_all_invalid_plan_rows_inplace_np(plan_np)
        # A decode at start_pos=11, ratio=4 → position=11; (11+1)%4 == 0 → boundary.
        n_c = self._build_plan(plan_np, 11, 1, 4, True, cap, mod)
        assert n_c == 1
        # Front row is live.
        assert plan_np[0, 2] == 11
        # All tail rows are sentinel.
        assert (plan_np[1:, 2] == -1).all()

    def test_tail_sentinel_after_shrinking_fwd(self):
        """Capture-replay simulation: a previous forward wrote n_compress=8
        live rows; the next forward only needs n_compress=2. The tail rows
        [2:capacity) MUST be re-stamped to sentinel (no leakage of the
        previous forward's live data into the new forward's tail kernel
        programs)."""
        mod = self._load_compress_plan_module()
        cap = 16
        plan_np = np.zeros((cap, 4), dtype=np.int32)
        mod.fill_all_invalid_plan_rows_inplace_np(plan_np)
        # Forward A: start_pos=0, token_num=32 with ratio=4 → boundaries at
        # positions 3, 7, 11, ..., 31 → n_compress = 8.
        n_a = self._build_plan(plan_np, 0, 32, 4, True, cap, mod)
        assert n_a == 8
        for r in range(8):
            assert plan_np[r, 2] != -1  # live
        for r in range(8, cap):
            assert plan_np[r, 2] == -1  # tail sentinel
        # Forward B: start_pos=200, token_num=8 with ratio=4 → positions
        # 200..207; (pos+1)%4==0 at positions 203, 207 → n_compress = 2.
        n_b = self._build_plan(plan_np, 200, 8, 4, True, cap, mod)
        assert n_b == 2
        # First 2 rows are live (new positions 203, 207).
        assert plan_np[0, 2] == 203
        assert plan_np[1, 2] == 207
        # CRITICAL: rows 2..7 (which Forward A had populated as live) MUST
        # now be sentinel, not the stale forward-A values.
        for r in range(2, cap):
            assert (
                plan_np[r, 2] == -1
            ), f"row {r} leaked stale forward-A data: {plan_np[r].tolist()}"

    def test_tail_sentinel_after_zero_compress_fwd(self):
        """Forward with n_compress=0 (sentinel decode at non-boundary
        position): all rows must be sentinel; nothing is live."""
        mod = self._load_compress_plan_module()
        cap = 8
        plan_np = np.zeros((cap, 4), dtype=np.int32)
        mod.fill_all_invalid_plan_rows_inplace_np(plan_np)
        # A previous forward populated rows 0..3 with live data.
        plan_np[:4] = np.array([[1, 0, 100, 0]] * 4, dtype=np.int32)
        # New forward: position=1, ratio=4 → (1+1)%4 != 0 → no boundary.
        n = self._build_plan(plan_np, 1, 1, 4, True, cap, mod)
        assert n == 0
        # All rows MUST now be sentinel — including the rows that were live
        # before this forward. This is the "shrink to zero" subcase.
        assert (plan_np[:, 2] == -1).all()


class TestR20aKernelGridConstancy:
    """r20a: the kernel wrapper signatures change to use `compress_capacity`
    / `write_capacity` as the launch grid when set (capture-stable). Verify
    the grid-selection helper produces a CONSTANT grid across forwards with
    differing live counts (= the property cudagraph capture requires)."""

    @staticmethod
    def _grid_for_compress(plan_num_compress, plan_capacity):
        """Mirrors fused_compress.py's `capture_grid = plan.compress_capacity`
        branch."""
        if plan_capacity is not None:
            return int(plan_capacity)
        return int(plan_num_compress)

    @staticmethod
    def _grid_for_write(num_write, write_capacity):
        """Mirrors state_writes.py's `if write_capacity is not None:` branch."""
        if write_capacity is not None:
            return int(write_capacity)
        return int(num_write)

    def test_compress_grid_constant_when_capacity_set(self):
        """Across 3 forwards with n_compress = 0 / 1 / 5, the grid must be
        equal to `compress_capacity` (e.g. 32) for ALL of them. THIS IS THE
        POINT — the captured graph bakes the constant grid value, which
        does not vary by replay."""
        cap = 32
        for n_compress in (0, 1, 5, cap - 1, cap):
            assert self._grid_for_compress(n_compress, cap) == cap

    def test_compress_grid_falls_back_to_num_compress_when_no_capacity(self):
        """Backward compat: legacy plans (capacity=None) get grid = num_compress."""
        for n_compress in (1, 7, 23):
            assert self._grid_for_compress(n_compress, None) == n_compress

    def test_write_grid_constant_when_capacity_set(self):
        cap = 8
        for n_write in (0, 1, 4, cap):
            assert self._grid_for_write(n_write, cap) == cap

    def test_write_grid_falls_back_when_no_capacity(self):
        for n_write in (1, 5, 17):
            assert self._grid_for_write(n_write, None) == n_write


# ── r20b unit tests ────────────────────────────────────────────────────────


class TestR20bBatchedPlanIntoBuffers:
    """r20b: capture-stable batched plan builder
    `make_compress_plans_into_buffers` writes plan rows into caller-provided
    pinned-CPU + GPU buffers and returns CompressPlans with capacity set.

    Validates the same shrink-without-leak invariant as the bs=1 builder,
    plus multi-ratio dispatch (CSA + HCA in one batch)."""

    @staticmethod
    def _load_compress_plan_module():
        """Minimal-import load of compress_plan.py (same pattern as r20a tests)."""
        import importlib.util
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        path = repo / "atom" / "model_ops" / "v4_kernels" / "compress_plan.py"
        spec = importlib.util.spec_from_file_location("_test_compress_plan_r20b", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _make_buffers(self, mod, ratio, cap_c, cap_w):
        """Build the (pin_np, pin, buf, capacity) buffers for one ratio in
        the format `make_compress_plans_into_buffers` expects."""
        pin_c = torch.empty(cap_c, 4, dtype=torch.int32)
        pin_w = torch.empty(cap_w, 4, dtype=torch.int32)
        pin_c_np = pin_c.numpy()
        pin_w_np = pin_w.numpy()
        mod.fill_all_invalid_plan_rows_inplace_np(pin_c_np)
        mod.fill_all_invalid_plan_rows_inplace_np(pin_w_np)
        buf_c = torch.zeros(cap_c, 4, dtype=torch.int32)
        buf_c[:, 2] = -1
        buf_w = torch.zeros(cap_w, 4, dtype=torch.int32)
        buf_w[:, 2] = -1
        return {
            "pin_np": {ratio: pin_c_np},
            "pin_np_w": {ratio: pin_w_np},
            "pin": {ratio: pin_c},
            "pin_w": {ratio: pin_w},
            "buf": {ratio: buf_c},
            "buf_w": {ratio: buf_w},
            "cap_c": {ratio: cap_c},
            "cap_w": {ratio: cap_w},
        }

    def test_returns_capacity_set_plans(self):
        """Plans returned by the batched builder MUST carry both
        `compress_capacity` and `write_capacity` (legacy `make_compress_plans`
        leaves them None)."""
        mod = self._load_compress_plan_module()
        b = self._make_buffers(mod, ratio=4, cap_c=64, cap_w=64)
        extend = np.array([1, 1], dtype=np.int32)  # 2 decode tokens
        seq = np.array([12, 16], dtype=np.int32)  # past + current
        plans = mod.make_compress_plans_into_buffers(
            extend,
            seq,
            [(4, True)],
            compress_pin_per_ratio=b["pin_np"],
            write_pin_per_ratio=b["pin_np_w"],
            compress_buf_per_ratio=b["buf"],
            write_buf_per_ratio=b["buf_w"],
            compress_capacity_per_ratio=b["cap_c"],
            write_capacity_per_ratio=b["cap_w"],
        )
        assert 4 in plans
        plan = plans[4]
        assert plan.compress_capacity == 64
        assert plan.write_capacity == 64
        # Returned plan tensors are the EXACT GPU buffers (stable address).
        assert plan.compress_plan_gpu.data_ptr() == b["buf"][4].data_ptr()
        assert plan.write_plan_gpu.data_ptr() == b["buf_w"][4].data_ptr()

    def test_buffer_address_stable_across_calls(self):
        """The whole point of r20b: buffer addresses MUST be stable across
        multiple builder invocations (= captured graph's pointer args remain
        valid across replays)."""
        mod = self._load_compress_plan_module()
        b = self._make_buffers(mod, ratio=4, cap_c=32, cap_w=32)
        # Call 1.
        plans1 = mod.make_compress_plans_into_buffers(
            np.array([1], dtype=np.int32),
            np.array([8], dtype=np.int32),
            [(4, True)],
            compress_pin_per_ratio=b["pin_np"],
            write_pin_per_ratio=b["pin_np_w"],
            compress_buf_per_ratio=b["buf"],
            write_buf_per_ratio=b["buf_w"],
            compress_capacity_per_ratio=b["cap_c"],
            write_capacity_per_ratio=b["cap_w"],
        )
        addr_c = plans1[4].compress_plan_gpu.data_ptr()
        addr_w = plans1[4].write_plan_gpu.data_ptr()
        # Call 2 with different inputs.
        plans2 = mod.make_compress_plans_into_buffers(
            np.array([4], dtype=np.int32),
            np.array([200], dtype=np.int32),
            [(4, True)],
            compress_pin_per_ratio=b["pin_np"],
            write_pin_per_ratio=b["pin_np_w"],
            compress_buf_per_ratio=b["buf"],
            write_buf_per_ratio=b["buf_w"],
            compress_capacity_per_ratio=b["cap_c"],
            write_capacity_per_ratio=b["cap_w"],
        )
        assert plans2[4].compress_plan_gpu.data_ptr() == addr_c
        assert plans2[4].write_plan_gpu.data_ptr() == addr_w

    def test_tail_sentinel_after_shrinking_call(self):
        """Same shrink-without-leak invariant as the bs=1 path: a previous
        call with large n_compress must not leave live data in the new
        call's tail rows."""
        mod = self._load_compress_plan_module()
        b = self._make_buffers(mod, ratio=4, cap_c=32, cap_w=32)
        # Call A: token_num=32 fresh prefill, ratio=4 → boundaries at
        # positions 3, 7, ..., 31 → n_compress = 8.
        mod.make_compress_plans_into_buffers(
            np.array([32], dtype=np.int32),
            np.array([32], dtype=np.int32),
            [(4, True)],
            compress_pin_per_ratio=b["pin_np"],
            write_pin_per_ratio=b["pin_np_w"],
            compress_buf_per_ratio=b["buf"],
            write_buf_per_ratio=b["buf_w"],
            compress_capacity_per_ratio=b["cap_c"],
            write_capacity_per_ratio=b["cap_w"],
        )
        # Confirm front rows live, tail sentinel.
        pin_np = b["pin_np"][4]
        assert pin_np[0, 2] == 3
        assert pin_np[7, 2] == 31
        for r in range(8, 32):
            assert pin_np[r, 2] == -1
        # Call B (shrink): 1 decode token at position=11 → n_compress = 1.
        mod.make_compress_plans_into_buffers(
            np.array([1], dtype=np.int32),
            np.array([12], dtype=np.int32),
            [(4, True)],
            compress_pin_per_ratio=b["pin_np"],
            write_pin_per_ratio=b["pin_np_w"],
            compress_buf_per_ratio=b["buf"],
            write_buf_per_ratio=b["buf_w"],
            compress_capacity_per_ratio=b["cap_c"],
            write_capacity_per_ratio=b["cap_w"],
        )
        # Front row is the new boundary; rows 1..31 MUST be sentinel
        # (rows 1..7 were previously live in call A).
        assert pin_np[0, 2] == 11
        for r in range(1, 32):
            assert (
                pin_np[r, 2] == -1
            ), f"row {r} leaked stale call-A data: {pin_np[r].tolist()}"

    def test_zero_total_emits_all_sentinel_capacity_plans(self):
        """When extend_lens.sum() == 0 (e.g. dummy batch), the builder must
        STILL emit capacity-set plans whose buffers are all-sentinel — so
        downstream callers always launch the worst-case grid (kernel
        no-ops everywhere)."""
        mod = self._load_compress_plan_module()
        b = self._make_buffers(mod, ratio=4, cap_c=8, cap_w=8)
        # Pre-poison the front of pin_np to detect re-stamp.
        b["pin_np"][4][0, 2] = 999
        plans = mod.make_compress_plans_into_buffers(
            np.array([0, 0], dtype=np.int32),
            np.array([10, 20], dtype=np.int32),
            [(4, True)],
            compress_pin_per_ratio=b["pin_np"],
            write_pin_per_ratio=b["pin_np_w"],
            compress_buf_per_ratio=b["buf"],
            write_buf_per_ratio=b["buf_w"],
            compress_capacity_per_ratio=b["cap_c"],
            write_capacity_per_ratio=b["cap_w"],
        )
        plan = plans[4]
        assert plan.num_compress == 0
        assert plan.num_write == 0
        assert plan.compress_capacity == 8
        # All rows of the underlying pinned buffer MUST be sentinel.
        assert (b["pin_np"][4][:, 2] == -1).all()

    def test_capacity_overflow_raises(self):
        """Defensive: if the live row count exceeds capacity (caller sized
        capacity wrong), raise a clear error rather than silently truncating
        and producing wrong attention output."""
        mod = self._load_compress_plan_module()
        b = self._make_buffers(mod, ratio=4, cap_c=2, cap_w=64)  # cap too small
        # 32 fresh-prefill tokens → 8 compress boundaries > capacity 2.
        try:
            mod.make_compress_plans_into_buffers(
                np.array([32], dtype=np.int32),
                np.array([32], dtype=np.int32),
                [(4, True)],
                compress_pin_per_ratio=b["pin_np"],
                write_pin_per_ratio=b["pin_np_w"],
                compress_buf_per_ratio=b["buf"],
                write_buf_per_ratio=b["buf_w"],
                compress_capacity_per_ratio=b["cap_c"],
                write_capacity_per_ratio=b["cap_w"],
            )
            assert False, "expected RuntimeError on capacity overflow"
        except RuntimeError as e:
            assert "compress plan overflow" in str(e)
            assert "capacity=2" in str(e)


# ── P7-r21 site #5e: sparse-attn layout K_MAX constexpr stability UTs ──────


class TestR21SparseLayoutFixedDecodeK:
    """P7-r21 site #5e: validate that the decode branch of
    `_attach_sparse_layout_metadata` emits the SAME `topk_lens` /
    `kv_offsets` strides regardless of per-seq start_pos, for both ratio=4
    (Indexer-driven) and ratio=128 (HCA-table-driven) layers.

    This stability is what makes the captured `topk_flat` shape and
    `kv_sa` cat shape constant across cudagraph replays. Per EVIDENCE_P720
    §4 5b<->5d co-dependency: if the layout drifts (variable K), the
    captured topk_flat slice strides become wrong and sparse_attn reads
    garbage.

    CPU-only via a minimal builder stand-in that exposes only the surface
    `_attach_sparse_layout_metadata` consumes: window_size, index_topk,
    `_max_n_committed_per_ratio`, and a `model_runner.forward_vars` dict
    of CpuGpuBuffer-shaped objects.
    """

    @staticmethod
    def _make_stub_builder(
        window_size: int, index_topk: int, max_n_csa: int, max_n_hca: int, n_tokens: int
    ):
        """Builds the minimal duck-typed object with the attribute surface
        consumed by `_attach_sparse_layout_metadata` so we can call it
        without importing the heavy `atom.model_ops.attentions.deepseek_v4_attn`
        module (which pulls torch.cuda + transformers).
        """
        # Bring in the actual function source to drive the test against the
        # production layout code (no copy-paste — guarantees future drifts
        # break this test).
        atom_root = Path(__file__).resolve().parent.parent
        attn_mod_src = (
            atom_root / "atom" / "model_ops" / "attentions" / "deepseek_v4_attn.py"
        ).read_text()
        # Locate the def line and the next def/class at column 0 to slice
        # the function body (resilient to small surrounding edits).
        marker = "    def _attach_sparse_layout_metadata("
        i = attn_mod_src.find(marker)
        assert i != -1, "marker not found"
        # Find the next top-of-block sibling method (prefix '    def ').
        j = attn_mod_src.find("\n    def ", i + len(marker))
        assert j != -1
        body = attn_mod_src[i:j]
        # Strip leading 4-space class indent so we can exec at module scope.
        body = "\n".join(
            ln[4:] if ln.startswith("    ") else ln for ln in body.splitlines()
        )
        # Compose: attach a wrapper function that takes self explicitly.
        # Stub the type symbols the function annotates (no runtime use).
        ns = {"np": np, "torch": torch, "AttentionMetaData": object}
        exec(body, ns)
        fn = ns["_attach_sparse_layout_metadata"]

        # Stub forward_vars + builder.
        class _CpuGpuStub:
            def __init__(self, n, dtype):
                self.np = np.zeros(n, dtype=dtype)
                self._dtype = dtype

            def copy_to_gpu(self, n):
                return torch.from_numpy(self.np[:n].copy())

        forward_vars = {}
        for kind in ("dense", "csa", "hca"):
            forward_vars[f"v4_{kind}_sparse_topk_starts"] = _CpuGpuStub(
                n_tokens, np.int64
            )
            forward_vars[f"v4_{kind}_sparse_topk_lens"] = _CpuGpuStub(
                n_tokens, np.int32
            )
            forward_vars[f"v4_{kind}_sparse_kv_offsets"] = _CpuGpuStub(
                n_tokens, np.int32
            )

        class _RunnerStub:
            pass

        runner = _RunnerStub()
        runner.forward_vars = forward_vars

        class _BuilderStub:
            pass

        b = _BuilderStub()
        b.window_size = window_size
        b.index_topk = index_topk
        b.model_runner = runner
        b._max_n_committed_per_ratio = {4: max_n_csa, 128: max_n_hca}
        # Bind the function as a method.
        b._attach_sparse_layout_metadata = fn.__get__(b, _BuilderStub)
        return b

    @staticmethod
    def _make_metadata_stub():
        class _MD:
            pass

        md = _MD()
        md.compress_plans = None  # forces n_compress fallback
        return md

    def test_decode_csa_topk_lens_fixed_at_window_plus_index_topk(self):
        """For ratio==4 decode (start_pos>0), every per-token topk_len MUST
        equal window_size + index_topk regardless of per-seq start_pos."""
        window_size, index_topk, max_n_csa, max_n_hca = 128, 1024, 1024, 32
        bs = 3
        n_tokens = bs  # decode = 1 token per seq
        b = self._make_stub_builder(
            window_size, index_topk, max_n_csa, max_n_hca, n_tokens
        )
        md = self._make_metadata_stub()
        cu_seqlens_q_np = np.array([0, 1, 2, 3], dtype=np.int32)
        # Mix start_pos values to detect Python-int leaks; all > 0 (decode).
        start_pos_per_seq = np.array([5, 200, 4095], dtype=np.int32)
        b._attach_sparse_layout_metadata(
            md, cu_seqlens_q_np, start_pos_per_seq, bs, n_tokens
        )
        layouts = md.v4_sparse_layouts
        csa = layouts[4]
        # All per-token topk_lens equal window_size + index_topk.
        lens_cpu = csa["topk_lens"].cpu().numpy()
        expected = window_size + index_topk
        assert (lens_cpu[:n_tokens] == expected).all(), (
            f"CSA decode topk_lens not invariant: got {lens_cpu[:n_tokens].tolist()}"
            f" expected all == {expected}"
        )
        # max_topk per layout dict reflects the same.
        assert csa["max_topk"] == expected, csa["max_topk"]

    def test_decode_csa_kv_offsets_strides_fixed_at_window_plus_max_n(self):
        """For ratio==4 decode, kv_offsets[i+1] - kv_offsets[i] MUST equal
        window_size + max_n_committed_csa for every consecutive seq pair."""
        window_size, index_topk, max_n_csa, max_n_hca = 128, 1024, 1024, 32
        bs = 4
        n_tokens = bs
        b = self._make_stub_builder(
            window_size, index_topk, max_n_csa, max_n_hca, n_tokens
        )
        md = self._make_metadata_stub()
        cu_seqlens_q_np = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        start_pos_per_seq = np.array([10, 500, 1234, 4000], dtype=np.int32)
        b._attach_sparse_layout_metadata(
            md, cu_seqlens_q_np, start_pos_per_seq, bs, n_tokens
        )
        offsets = md.v4_sparse_layouts[4]["kv_offsets"].cpu().numpy()
        expected_stride = window_size + max_n_csa
        for i in range(bs - 1):
            stride = int(offsets[i + 1]) - int(offsets[i])
            assert stride == expected_stride, (
                f"CSA decode kv_offsets stride seq{i}->{i+1}: got {stride},"
                f" expected {expected_stride}"
            )

    def test_decode_hca_topk_lens_fixed_at_window_plus_max_n_hca(self):
        """For ratio==128 decode, topk_len MUST equal window_size +
        max_n_committed_hca regardless of per-seq start_pos."""
        window_size, index_topk, max_n_csa, max_n_hca = 128, 1024, 1024, 32
        bs = 3
        n_tokens = bs
        b = self._make_stub_builder(
            window_size, index_topk, max_n_csa, max_n_hca, n_tokens
        )
        md = self._make_metadata_stub()
        cu_seqlens_q_np = np.array([0, 1, 2, 3], dtype=np.int32)
        start_pos_per_seq = np.array([129, 1500, 4000], dtype=np.int32)
        b._attach_sparse_layout_metadata(
            md, cu_seqlens_q_np, start_pos_per_seq, bs, n_tokens
        )
        hca = md.v4_sparse_layouts[128]
        lens_cpu = hca["topk_lens"].cpu().numpy()
        expected = window_size + max_n_hca
        assert (lens_cpu[:n_tokens] == expected).all(), lens_cpu[:n_tokens].tolist()
        # And kv_offsets stride
        offsets_cpu = hca["kv_offsets"].cpu().numpy()
        for i in range(bs - 1):
            stride = int(offsets_cpu[i + 1]) - int(offsets_cpu[i])
            assert stride == window_size + max_n_hca, stride

    def test_prefill_csa_keeps_dynamic_shape(self):
        """For ratio==4 prefill (start_pos==0), the layout MUST keep the
        legacy variable-shape K (capture is decode-only). Regression guard
        to ensure r21 didn't accidentally collapse prefill onto fixed K."""
        window_size, index_topk, max_n_csa, max_n_hca = 128, 1024, 1024, 32
        bs = 1
        n_tokens = 16  # prefill: token_num=16
        b = self._make_stub_builder(
            window_size, index_topk, max_n_csa, max_n_hca, n_tokens
        )
        md = self._make_metadata_stub()
        cu_seqlens_q_np = np.array([0, 16], dtype=np.int32)
        start_pos_per_seq = np.array([0], dtype=np.int32)
        b._attach_sparse_layout_metadata(
            md, cu_seqlens_q_np, start_pos_per_seq, bs, n_tokens
        )
        csa = md.v4_sparse_layouts[4]
        lens_cpu = csa["topk_lens"].cpu().numpy()
        # Prefill: window_topk = min(token_num, window_size) = 16,
        # compress_topk = min(index_topk, end_pos // ratio) = min(1024, 4) = 4,
        # topk_len = 20 (NOT the decode-fixed 1152).
        expected = min(n_tokens, window_size) + min(index_topk, n_tokens // 4)
        assert (lens_cpu[:n_tokens] == expected).all(), (
            f"prefill CSA topk_lens={lens_cpu[:n_tokens].tolist()},"
            f" expected all == {expected}"
        )
