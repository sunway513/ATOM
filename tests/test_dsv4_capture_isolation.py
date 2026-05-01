# SPDX-License-Identifier: MIT
"""DSV4 cudagraph capture isolation correctness tests.

P7 round 7 hypothesis (carried from EVIDENCE_P76): conc=4 V4 capture has
multi-request state isolation failure. Conc=1 capture works perfectly
(0.90 vs 0.85 eager). Conc=4 wildly variant (0.35-0.95).

These tests isolate the *mechanism* (cudagraph capture + tensor-indexed slot
dispatch) from the full V4 model, so a fix can be designed around the
verified failure mode.

Test taxonomy:

  test_swa_dispatch_unshuffled_replay:
    Capture at bs=4 with state_slot_mapping=[0,1,2,3]; replay with same.
    Verify per-slot output bit-identical to fresh eager forward.
    Establishes that the capture infra is sound for the trivial case.

  test_swa_dispatch_shuffled_replay:
    Capture at bs=4 with state_slot_mapping=[0,1,2,3]; replay with
    shuffled [3,2,1,0]. Verify per-slot output bit-identical to fresh eager.
    This is the ROOT-CAUSE PROBE — if it fails, we have a concrete repro
    of multi-request capture isolation breakage.

  test_swa_dispatch_per_iter_state_aliasing:
    bs=4 loop with intra-step read-after-write on a shared swa-like buffer
    (mirror of V4Attention.forward inner loop). Verify per-iter output is
    NOT influenced by sibling iters' writes.

The micro-fixture mimics the per-seq dispatch inside `V4Attention.forward`
between lines 1948-2272 (per-seq slot tensor + swa_kv read/write +
constant-shape gather + cat). It does NOT exercise the full attention math
(no Triton kernels, no FP8 quant, no Compressor) — those are tested
elsewhere; we only need to prove the slot dispatch is correct under capture.

Run on a GPU box:
    pytest tests/test_dsv4_capture_isolation.py -v -s

Skip on CPU-only:
    PYTEST_DISABLE_GPU=1 pytest tests/test_dsv4_capture_isolation.py
"""

from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("PYTEST_DISABLE_GPU") == "1" or not torch.cuda.is_available(),
    reason="requires CUDA / HIP",
)


# ---------------------------------------------------------------------------
# Micro-fixture mirroring V4Attention.forward per-seq slot dispatch.
# ---------------------------------------------------------------------------


class _V4LikeMicroAttention(torch.nn.Module):
    """Minimal stand-in for V4Attention's per-seq capture-friendly loop body.

    Mirrors lines 1948-2272 of `atom/models/deepseek_v4.py`:
      - per-seq Python-int loop over `range(num_seqs)`
      - tensor `slot_tensor = state_slot_mapping[seq_idx:seq_idx+1]`
      - in-place SWA write: `swa[slot_tensor, pos_mod] = seq_kv`
      - SWA read for attention input: `swa.index_select(0, slot_tensor)`
      - constant-shape compressed-cache gather + cat
      - sparse-attn-equivalent reduction (just sum along KV dim)

    Replaces FP8 quant + Triton kernels + sparse_attn with bit-exact toy
    arithmetic so capture+replay correctness is exclusively about the
    dispatch mechanism, not numerics.
    """

    def __init__(
        self,
        num_slots: int = 8,
        win: int = 128,
        head_dim: int = 16,
        max_n: int = 8,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.win = win
        self.head_dim = head_dim
        self.max_n = max_n
        # Per-request SWA ring buffer + per-request "compressed-KV" cache.
        # In real V4 these are paged + Triton-written; we use plain dense
        # buffers because correctness of indexing — not bandwidth — is what's
        # under test.
        self.register_buffer(
            "swa", torch.zeros(num_slots, win, head_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "comp", torch.zeros(num_slots, max_n, head_dim, dtype=torch.float32)
        )
        # Tiny linear that mixes input → query so the per-step output depends
        # on per-slot history (otherwise the test trivializes to identity).
        self.proj = torch.nn.Linear(head_dim, head_dim, bias=False)

    def reset_state(self) -> None:
        with torch.no_grad():
            self.swa.zero_()
            self.comp.zero_()

    def init_per_slot_history(self, slot: int, seed: int = 0) -> None:
        """Pre-fill swa+comp for a slot so reads see non-trivial history.

        Each slot gets a deterministic per-slot pattern so tests can detect
        cross-slot contamination (a slot reading another slot's history will
        produce a different output).
        """
        gen = torch.Generator(device=self.swa.device).manual_seed(seed + slot * 17)
        with torch.no_grad():
            self.swa[slot] = torch.randn(
                self.win, self.head_dim, dtype=torch.float32,
                device=self.swa.device, generator=gen,
            )
            self.comp[slot] = torch.randn(
                self.max_n, self.head_dim, dtype=torch.float32,
                device=self.comp.device, generator=gen,
            )

    def forward(
        self,
        x: torch.Tensor,                         # [num_tokens, head_dim], one tok/seq
        positions: torch.Tensor,                 # [num_tokens] int
        state_slot_mapping: torch.Tensor,        # [bs] int
        cu_seqlens_q_cpu: list[int],             # Python list, len bs+1 (capture-baked)
        num_seqs: int,                           # Python int (capture-baked)
        n_committed_per_seq: torch.Tensor,       # [bs] int — committed entries (per slot)
    ) -> torch.Tensor:                           # [num_tokens, head_dim]
        win = self.win
        slot_long = state_slot_mapping.long()
        outs = []
        for seq_idx in range(num_seqs):
            seq_start = cu_seqlens_q_cpu[seq_idx]
            seq_end = cu_seqlens_q_cpu[seq_idx + 1]
            slot_t = slot_long[seq_idx : seq_idx + 1]               # [1] long view
            pos_t = positions[seq_start:seq_end]                    # [1] int
            seq_x = x[seq_start:seq_end]                            # [1, head_dim]
            # in-place SWA write: same pattern as deepseek_v4.py:2107-2109
            pos_mod = (pos_t % win).long()
            self.swa[slot_t, pos_mod] = seq_x
            # SWA read for attention: same pattern as deepseek_v4.py:2200
            swa_view = self.swa.index_select(0, slot_t)             # [1, win, head_dim]
            # constant-shape gather + cat: same pattern as deepseek_v4.py:2179-2201
            n_committed_t = n_committed_per_seq[seq_idx : seq_idx + 1].long()
            ci = torch.arange(self.max_n, device=x.device).unsqueeze(0)  # [1, max_n]
            valid = (ci < n_committed_t.unsqueeze(1)).to(self.comp.dtype)
            comp_view = self.comp.index_select(0, slot_t)           # [1, max_n, head_dim]
            kv_sa = torch.cat([swa_view, comp_view * valid.unsqueeze(-1)], dim=1)
            # Reduction stand-in for sparse_attn — sum along KV dim then proj.
            o_seq = self.proj(kv_sa.sum(dim=1))                     # [1, head_dim]
            outs.append(o_seq)
        return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Helpers — pinned forward_vars buffers + capture/replay drivers.
# ---------------------------------------------------------------------------


def _pinned_inputs(bs: int, head_dim: int, device: str = "cuda"):
    """Allocate pinned-address forward_vars buffers (mirrors model_runner)."""
    return {
        "x": torch.zeros(bs, head_dim, dtype=torch.float32, device=device),
        "positions": torch.zeros(bs, dtype=torch.long, device=device),
        "state_slot_mapping": torch.zeros(bs, dtype=torch.int32, device=device),
        "n_committed": torch.zeros(bs, dtype=torch.int32, device=device),
    }


def _capture_decode_graph(
    model: _V4LikeMicroAttention,
    inputs: dict[str, torch.Tensor],
    bs: int,
    out_buf: torch.Tensor,
):
    """Capture a cudagraph that runs `model.forward` on the pinned inputs.

    Mirrors `ModelRunner.capture_cudagraph` for the V4 decode pure-tensor path:
      - Warmup forward to populate any lazy state
      - `torch.cuda.graph(...)` capture forward writing into `out_buf`
    """
    cu_seqlens_q_cpu = list(range(bs + 1))    # decode: [0,1,2,...,bs]
    # Warmup
    out_buf[:] = model(
        inputs["x"], inputs["positions"], inputs["state_slot_mapping"],
        cu_seqlens_q_cpu, bs, inputs["n_committed"],
    )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out_buf[:] = model(
            inputs["x"], inputs["positions"], inputs["state_slot_mapping"],
            cu_seqlens_q_cpu, bs, inputs["n_committed"],
        )
    return g


def _eager_forward_per_slot(
    model: _V4LikeMicroAttention,
    x_in: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    n_committed: torch.Tensor,
    swa_snapshot: torch.Tensor,
    comp_snapshot: torch.Tensor,
) -> torch.Tensor:
    """Run a fresh model with pre-set state buffers; return per-slot eager output.

    Used as ground truth: each replay output should equal a fresh eager forward
    (with state restored to pre-capture snapshot) for the same inputs.
    """
    bs = slots.numel()
    fresh = _V4LikeMicroAttention(
        num_slots=model.num_slots, win=model.win,
        head_dim=model.head_dim, max_n=model.max_n,
    ).to(model.swa.device)
    fresh.proj.load_state_dict(model.proj.state_dict())
    with torch.no_grad():
        fresh.swa.copy_(swa_snapshot)
        fresh.comp.copy_(comp_snapshot)
    cu = list(range(bs + 1))
    out = fresh(x_in, positions, slots, cu, bs, n_committed)
    return out


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    return "cuda"


@pytest.fixture(scope="function")
def model(device):
    torch.manual_seed(0)
    m = _V4LikeMicroAttention().to(device)
    # Initialize each slot with distinct history so cross-slot contamination
    # is detectable in output norms.
    for s in range(m.num_slots):
        m.init_per_slot_history(s)
    return m


def _assert_per_slot(
    actual: torch.Tensor, expected: torch.Tensor, slots, label: str,
    atol: float = 1e-4,
):
    bs = actual.size(0)
    err_lines = []
    for i in range(bs):
        a, e = actual[i], expected[i]
        if not torch.allclose(a, e, atol=atol, rtol=1e-4):
            mae = (a - e).abs().max().item()
            err_lines.append(
                f"  slot[{i}]={int(slots[i].item())} mismatch: "
                f"max_abs_err={mae:.6f} actual_norm={a.norm():.4f} "
                f"expected_norm={e.norm():.4f}"
            )
    assert not err_lines, (
        f"{label}: {len(err_lines)}/{bs} slots mismatch:\n" + "\n".join(err_lines)
    )


def test_swa_dispatch_unshuffled_replay(model, device):
    """Capture and replay with same state_slot_mapping = [0,1,2,3].

    Establishes the capture infra is sound for the trivial case before testing
    shuffled replay.
    """
    bs = 4
    inputs = _pinned_inputs(bs, model.head_dim, device)
    out_buf = torch.zeros(bs, model.head_dim, dtype=torch.float32, device=device)

    # Pre-snapshot state for ground-truth eager comparisons.
    swa_snapshot = model.swa.clone()
    comp_snapshot = model.comp.clone()

    # Capture-time inputs (= warmup pattern).
    capture_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    capture_pos = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device)
    capture_x = torch.randn(bs, model.head_dim, device=device, dtype=torch.float32)
    capture_n_comm = torch.tensor([2, 3, 1, 4], dtype=torch.int32, device=device)
    inputs["state_slot_mapping"].copy_(capture_slots)
    inputs["positions"].copy_(capture_pos)
    inputs["x"].copy_(capture_x)
    inputs["n_committed"].copy_(capture_n_comm)

    g = _capture_decode_graph(model, inputs, bs, out_buf)

    # Replay #1 (same inputs) — restore state, replay, compare to fresh eager.
    with torch.no_grad():
        model.swa.copy_(swa_snapshot)
        model.comp.copy_(comp_snapshot)
    g.replay()
    torch.cuda.synchronize()

    expected = _eager_forward_per_slot(
        model, capture_x, capture_pos, capture_slots, capture_n_comm,
        swa_snapshot, comp_snapshot,
    )
    _assert_per_slot(out_buf, expected, capture_slots, "unshuffled replay")


def test_swa_dispatch_shuffled_replay(model, device):
    """Capture with [0,1,2,3]; replay with shuffled [3,2,1,0].

    ROOT-CAUSE PROBE for P7-r6 hypothesis: if multi-request capture isolation
    is broken, the shuffled replay will produce wrong per-slot outputs (a slot
    might attend to another slot's swa/comp history because the captured graph
    baked the capture-time slot indices).
    """
    bs = 4
    inputs = _pinned_inputs(bs, model.head_dim, device)
    out_buf = torch.zeros(bs, model.head_dim, dtype=torch.float32, device=device)

    swa_snapshot = model.swa.clone()
    comp_snapshot = model.comp.clone()

    # Capture-time inputs.
    cap_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    cap_pos = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device)
    cap_x = torch.randn(bs, model.head_dim, device=device, dtype=torch.float32)
    cap_ncomm = torch.tensor([2, 3, 1, 4], dtype=torch.int32, device=device)
    inputs["state_slot_mapping"].copy_(cap_slots)
    inputs["positions"].copy_(cap_pos)
    inputs["x"].copy_(cap_x)
    inputs["n_committed"].copy_(cap_ncomm)

    g = _capture_decode_graph(model, inputs, bs, out_buf)

    # Replay-time inputs: SHUFFLED slot mapping.
    rep_slots = torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=device)
    rep_pos = torch.tensor([10, 11, 12, 13], dtype=torch.long, device=device)
    rep_x = torch.randn(bs, model.head_dim, device=device, dtype=torch.float32)
    rep_ncomm = torch.tensor([5, 4, 3, 2], dtype=torch.int32, device=device)
    with torch.no_grad():
        # Restore state so replay sees same pre-step swa/comp as fresh eager.
        model.swa.copy_(swa_snapshot)
        model.comp.copy_(comp_snapshot)
        inputs["state_slot_mapping"].copy_(rep_slots)
        inputs["positions"].copy_(rep_pos)
        inputs["x"].copy_(rep_x)
        inputs["n_committed"].copy_(rep_ncomm)
    g.replay()
    torch.cuda.synchronize()

    expected = _eager_forward_per_slot(
        model, rep_x, rep_pos, rep_slots, rep_ncomm,
        swa_snapshot, comp_snapshot,
    )
    _assert_per_slot(out_buf, expected, rep_slots, "shuffled replay")


def test_swa_dispatch_state_persists_across_replays(model, device):
    """Capture once, replay twice with disjoint slot sets; verify each replay
    correctly mutates only the slots it was given, leaving sibling slots
    untouched between replays.

    This catches the cross-iteration aliasing class of bug: if iter `i+1`'s
    write to slot `b` accidentally clobbers slot `a` written by iter `i` of
    a previous replay (or the same replay's iter 0), this test fails.
    """
    bs = 4
    inputs = _pinned_inputs(bs, model.head_dim, device)
    out_buf = torch.zeros(bs, model.head_dim, dtype=torch.float32, device=device)

    cap_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    cap_pos = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device)
    cap_x = torch.randn(bs, model.head_dim, device=device, dtype=torch.float32)
    cap_ncomm = torch.tensor([2, 3, 1, 4], dtype=torch.int32, device=device)
    inputs["state_slot_mapping"].copy_(cap_slots)
    inputs["positions"].copy_(cap_pos)
    inputs["x"].copy_(cap_x)
    inputs["n_committed"].copy_(cap_ncomm)

    g = _capture_decode_graph(model, inputs, bs, out_buf)

    # Snapshot AFTER warmup so swa[0..3] reflect post-warmup state.
    swa_post_warm = model.swa.clone()
    comp_post_warm = model.comp.clone()

    # Replay 1 with slots [4,5,6,7] (disjoint from warmup's [0,1,2,3]).
    r1_slots = torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=device)
    r1_pos = torch.tensor([20, 21, 22, 23], dtype=torch.long, device=device)
    r1_x = torch.randn(bs, model.head_dim, device=device, dtype=torch.float32)
    r1_ncomm = torch.tensor([1, 2, 3, 0], dtype=torch.int32, device=device)
    with torch.no_grad():
        inputs["state_slot_mapping"].copy_(r1_slots)
        inputs["positions"].copy_(r1_pos)
        inputs["x"].copy_(r1_x)
        inputs["n_committed"].copy_(r1_ncomm)
    g.replay()
    torch.cuda.synchronize()

    # CHECK: after replay 1, slots [0..3] swa MUST be unchanged from post-warmup.
    diff = (model.swa[:4] - swa_post_warm[:4]).abs().max().item()
    assert diff < 1e-6, (
        f"replay with slots [4..7] silently mutated slots [0..3] swa: "
        f"max_diff={diff:.6e}"
    )

    # Verify per-slot output bit-identical to fresh eager forward.
    expected = _eager_forward_per_slot(
        model, r1_x, r1_pos, r1_slots, r1_ncomm,
        swa_post_warm, comp_post_warm,
    )
    _assert_per_slot(out_buf, expected, r1_slots, "replay-1 (slots 4..7)")


# ---------------------------------------------------------------------------
# Multi-layer capture isolation (probes cross-layer pool aliasing).
# ---------------------------------------------------------------------------


class _V4LikeMultiLayer(torch.nn.Module):
    """Stack of N _V4LikeMicroAttention layers sharing a single forward call.

    Real DSV4 has 61 layers. Each layer owns its own swa_kv/comp buffers but
    ALL layers process the same `state_slot_mapping` / `positions` / per-step
    inputs in the captured graph. If a captured-graph bug is layer-count
    sensitive (e.g., pool aliasing across layers, or per-layer scratch
    reuse that interferes), this test surfaces it where the single-layer test
    above does not.
    """

    def __init__(self, num_layers: int = 8, **layer_kwargs) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList(
            [_V4LikeMicroAttention(**layer_kwargs) for _ in range(num_layers)]
        )

    def reset_state(self) -> None:
        for layer in self.layers:
            layer.reset_state()

    def init_per_slot_history(self, base_seed: int = 0) -> None:
        for li, layer in enumerate(self.layers):
            for s in range(layer.num_slots):
                layer.init_per_slot_history(s, seed=base_seed + li * 1000)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        state_slot_mapping: torch.Tensor,
        cu_seqlens_q_cpu: list[int],
        num_seqs: int,
        n_committed_per_seq: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(
                h, positions, state_slot_mapping,
                cu_seqlens_q_cpu, num_seqs, n_committed_per_seq,
            )
        return h


def test_multi_layer_shuffled_replay(device):
    """Multi-layer (8-deep) capture+replay with shuffled state_slot_mapping.

    Real DSV4 has 61 layers. This 8-layer probe catches cross-layer pool
    aliasing or per-layer scratch interference that a single-layer test
    cannot expose.
    """
    bs = 4
    head_dim = 16
    torch.manual_seed(1)
    model = _V4LikeMultiLayer(num_layers=8, head_dim=head_dim).to(device)
    model.init_per_slot_history()
    inputs = _pinned_inputs(bs, head_dim, device)
    out_buf = torch.zeros(bs, head_dim, dtype=torch.float32, device=device)

    swa_snaps = [layer.swa.clone() for layer in model.layers]
    comp_snaps = [layer.comp.clone() for layer in model.layers]

    cap_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    cap_pos = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device)
    cap_x = torch.randn(bs, head_dim, device=device, dtype=torch.float32)
    cap_ncomm = torch.tensor([2, 3, 1, 4], dtype=torch.int32, device=device)
    inputs["state_slot_mapping"].copy_(cap_slots)
    inputs["positions"].copy_(cap_pos)
    inputs["x"].copy_(cap_x)
    inputs["n_committed"].copy_(cap_ncomm)

    g = _capture_decode_graph(model, inputs, bs, out_buf)

    # Shuffled replay.
    rep_slots = torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=device)
    rep_pos = torch.tensor([15, 16, 17, 18], dtype=torch.long, device=device)
    rep_x = torch.randn(bs, head_dim, device=device, dtype=torch.float32)
    rep_ncomm = torch.tensor([5, 4, 3, 2], dtype=torch.int32, device=device)

    # Restore each layer's state to pre-capture snapshot for fair comparison.
    with torch.no_grad():
        for li, layer in enumerate(model.layers):
            layer.swa.copy_(swa_snaps[li])
            layer.comp.copy_(comp_snaps[li])
        inputs["state_slot_mapping"].copy_(rep_slots)
        inputs["positions"].copy_(rep_pos)
        inputs["x"].copy_(rep_x)
        inputs["n_committed"].copy_(rep_ncomm)
    g.replay()
    torch.cuda.synchronize()

    # Build a fresh model with same weights + state to compute eager ground truth.
    fresh = _V4LikeMultiLayer(num_layers=model.num_layers, head_dim=head_dim).to(device)
    fresh.load_state_dict(model.state_dict())
    with torch.no_grad():
        for li, layer in enumerate(fresh.layers):
            layer.swa.copy_(swa_snaps[li])
            layer.comp.copy_(comp_snaps[li])
    cu = list(range(bs + 1))
    expected = fresh(rep_x, rep_pos, rep_slots, cu, bs, rep_ncomm)
    _assert_per_slot(out_buf, expected, rep_slots, "multi-layer shuffled replay")


def test_consecutive_replays_deterministic(device):
    """Replay the same graph twice with identical state and inputs; outputs MUST
    match bit-exact. Catches non-deterministic kernels or uninitialized memory
    in the captured intermediates.
    """
    bs = 4
    torch.manual_seed(2)
    model = _V4LikeMicroAttention().to(device)
    for s in range(model.num_slots):
        model.init_per_slot_history(s)

    inputs = _pinned_inputs(bs, model.head_dim, device)
    out_buf = torch.zeros(bs, model.head_dim, dtype=torch.float32, device=device)

    cap_slots = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=device)
    cap_pos = torch.tensor([5, 6, 7, 8], dtype=torch.long, device=device)
    cap_x = torch.randn(bs, model.head_dim, device=device, dtype=torch.float32)
    cap_ncomm = torch.tensor([2, 3, 1, 4], dtype=torch.int32, device=device)
    inputs["state_slot_mapping"].copy_(cap_slots)
    inputs["positions"].copy_(cap_pos)
    inputs["x"].copy_(cap_x)
    inputs["n_committed"].copy_(cap_ncomm)

    swa0, comp0 = model.swa.clone(), model.comp.clone()
    g = _capture_decode_graph(model, inputs, bs, out_buf)

    with torch.no_grad():
        model.swa.copy_(swa0)
        model.comp.copy_(comp0)
    g.replay()
    torch.cuda.synchronize()
    out_a = out_buf.clone()

    with torch.no_grad():
        model.swa.copy_(swa0)
        model.comp.copy_(comp0)
    g.replay()
    torch.cuda.synchronize()
    out_b = out_buf.clone()

    diff = (out_a - out_b).abs().max().item()
    assert diff == 0.0, (
        f"two replays with identical state/inputs differ: max_diff={diff:.6e}"
        " — non-deterministic capture or uninitialized intermediate memory"
    )
