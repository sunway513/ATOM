"""Diff per-stage Q tensor dumps from two `_forward_w4` runs.

Companion tool to the trace hook in
`atom/models/deepseek_v4.py:DeepseekV4Attention._forward_w4`. The hook
fires when ``ATOM_DSV4_TRACE_QSTAGE_DIR`` is set and writes one file per
(layer, decode-step, TP rank) named ``L{LL}_S{SS}_R{RR}.pt``.

Use to compare the SAME logical row across two runs (e.g. conc=1 with
``ATOM_DSV4_FORCE_W4=1`` vs conc=2 with multi-req enabled). The script
matches by TP rank (R) so cross-rank shard race does not contaminate
the diff.

Usage::

    python tools/dsv4_multireq_debug/diff_qstage.py \\
        traces_qstage_conc1 traces_qstage_conc2 \\
        --layer 0 --target-pos 59
"""
import argparse
from pathlib import Path

import torch


def load(d: Path, layer: int, step: int, rank: int):
    p = d / f"L{layer:02d}_S{step:02d}_R{rank:02d}.pt"
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def find_row_for_pos(rec: dict, target_pos: int) -> int:
    pos = rec["positions"].tolist()
    if target_pos in pos:
        return pos.index(target_pos)
    raise ValueError(f"position {target_pos} not in {pos}")


def main(d1: str, d2: str, layer: int, target_pos: int,
         step: int = 0, ranks=range(8)):
    d1p = Path(d1)
    d2p = Path(d2)
    print(f"d1: {d1p}\nd2: {d2p}\nlayer L{layer:02d}  step S{step:02d}  target_pos={target_pos}\n")

    fields = [
        "x_in", "wq_a_out", "qr_post_qnorm", "wq_b_out",
        "q_pre_rmsnorm", "q_post_rmsnorm", "q_post_rope",
    ]

    print(f"{'Rank':>4}  {'Field':25s}  {'norm_d1':>10}  {'norm_d2':>10}  {'max_diff':>12}  {'mean_diff':>12}")
    print("-" * 90)

    for r in ranks:
        a = load(d1p, layer, step, r)
        b = load(d2p, layer, step, r)
        if a is None or b is None:
            print(f"R{r:02d}  --- missing ({'a' if a is None else 'b'})")
            continue
        try:
            row_a = find_row_for_pos(a, target_pos)
            row_b = find_row_for_pos(b, target_pos)
        except ValueError as e:
            print(f"R{r:02d}  --- {e}")
            continue
        for fld in fields:
            ta = a[fld][row_a:row_a+1].float()
            tb = b[fld][row_b:row_b+1].float()
            if ta.shape != tb.shape:
                print(f"R{r:02d}  {fld:25s}  shape mismatch a={tuple(ta.shape)} b={tuple(tb.shape)}")
                continue
            d = (ta - tb).abs()
            print(f"R{r:02d}  {fld:25s}  {ta.norm().item():10.4f}  {tb.norm().item():10.4f}  {d.max().item():12.6f}  {d.mean().item():12.6f}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir1", help="trace dir for run 1 (e.g. conc=1 FORCE_W4)")
    ap.add_argument("dir2", help="trace dir for run 2 (e.g. conc=2)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--target-pos", type=int, required=True,
                    help="Absolute position of the same logical token to diff (e.g. 59)")
    ap.add_argument("--step", type=int, default=0)
    args = ap.parse_args()
    main(args.dir1, args.dir2, args.layer, args.target_pos, args.step)
