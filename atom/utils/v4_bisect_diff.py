"""V4 bisect dump diff tool — capture vs eager.

Loads two .pt dumps produced by `atom.utils.v4_bisect_dump` (one from
eager run, one from captured run, both bs=4 same-prompt same-seed) and
reports the FIRST DIVERGENT (call_idx, layer, stage) tuple plus
top-K most divergent stages.

Usage:
  python -m atom.utils.v4_bisect_diff <eager.pt> <capture.pt> [--top 10]

Stage map (12 stages per layer):
  Block (0..5):
    0=in, 1=after_attn_norm, 2=after_attention,
    3=after_attn_hc_post, 4=after_ffn, 5=after_ffn_hc_post
  V4Attention (6..11):
    6=after_q_norm, 7=after_kv_norm_pre_rope, 8=after_rope_q,
    9=kv_sa_input_to_sparse_attn (LAST seq only),
    10=sparse_attn_output (LAST seq only),
    11=after_wo_b (final attn output)
"""
import argparse
import sys
import torch


STAGE_NAME = {
    0: "Block.in",
    1: "Block.after_attn_norm",
    2: "Block.after_attention",
    3: "Block.after_attn_hc_post",
    4: "Block.after_ffn",
    5: "Block.after_ffn_hc_post",
    6: "Attn.after_q_norm",
    7: "Attn.after_kv_norm_pre_rope",
    8: "Attn.after_rope_q",
    9: "Attn.kv_sa_input_to_sparse_attn",
    10: "Attn.sparse_attn_output",
    11: "Attn.after_wo_b",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eager", help="eager dump .pt")
    ap.add_argument("capture", help="capture dump .pt")
    ap.add_argument("--top", type=int, default=15, help="top-K divergent stages")
    ap.add_argument("--rel-thresh", type=float, default=1e-2,
                    help="relative norm-diff threshold")
    ap.add_argument("--abs-thresh", type=float, default=1e-3,
                    help="absolute norm-diff threshold (for tiny norms)")
    args = ap.parse_args()

    e = torch.load(args.eager, map_location="cpu", weights_only=False)
    c = torch.load(args.capture, map_location="cpu", weights_only=False)

    en = e.get("inline_norms")
    cn = c.get("inline_norms")
    eh = e.get("inline_heads")
    ch = c.get("inline_heads")

    if en is None or cn is None or en.numel() == 0 or cn.numel() == 0:
        print("ERROR: inline_norms missing/empty in one of the dumps", file=sys.stderr)
        sys.exit(2)

    print(f"eager  inline_norms: {tuple(en.shape)}  call_norms: {len(e.get('call_norms', []))}")
    print(f"capture inline_norms: {tuple(cn.shape)}  call_norms: {len(c.get('call_norms', []))}")

    n_calls = min(en.shape[0], cn.shape[0])
    n_layers = min(en.shape[1], cn.shape[1])
    n_stages = min(en.shape[2], cn.shape[2])

    e_calls = en[:n_calls, :n_layers, :n_stages].float()
    c_calls = cn[:n_calls, :n_layers, :n_stages].float()

    # Per-call_norms summary first (call-level signal)
    print()
    print("=== Per-call hidden_states norm diff ===")
    e_hn = e.get("call_norms", [])
    c_hn = c.get("call_norms", [])
    e_pf = e.get("call_is_prefill", [])
    e_nt = e.get("call_n_tokens", [])
    c_nt = c.get("call_n_tokens", [])
    for i in range(min(len(e_hn), len(c_hn))):
        en_v = float(e_hn[i])
        cn_v = float(c_hn[i])
        d = abs(en_v - cn_v)
        rel = d / max(abs(en_v), 1e-9)
        marker = "  " if rel < args.rel_thresh else "!!"
        ip = e_pf[i] if i < len(e_pf) else "?"
        en_tk = e_nt[i] if i < len(e_nt) else "?"
        cn_tk = c_nt[i] if i < len(c_nt) else "?"
        print(f"  {marker} call[{i:02d}] prefill={ip} ntok=e:{en_tk}/c:{cn_tk}  norm e={en_v:.5f} c={cn_v:.5f} diff={d:.5e} rel={rel:.3e}")

    # Per (call, layer, stage) abs/rel diff
    abs_d = (e_calls - c_calls).abs()
    rel_d = abs_d / e_calls.abs().clamp(min=1e-9)

    diverge = (abs_d >= args.abs_thresh) & (rel_d >= args.rel_thresh)

    # Find first divergent in (call, layer, stage) lex order
    print()
    print("=== First divergent (call, layer, stage) — lex order ===")
    found = False
    for i in range(n_calls):
        for L in range(n_layers):
            for S in range(n_stages):
                if diverge[i, L, S]:
                    print(f"  FIRST DIVERGENT: call={i} layer={L} stage={S} ({STAGE_NAME.get(S, '?')})")
                    print(f"    eager_norm={e_calls[i,L,S].item():.5f}")
                    print(f"    capture_norm={c_calls[i,L,S].item():.5f}")
                    print(f"    abs_diff={abs_d[i,L,S].item():.5e}")
                    print(f"    rel_diff={rel_d[i,L,S].item():.3e}")
                    if eh is not None and ch is not None and eh.numel() > 0 and ch.numel() > 0:
                        eh_v = eh[i, L, S].float()
                        ch_v = ch[i, L, S].float()
                        head_diff = (eh_v - ch_v).abs()
                        print(f"    head[0:8] eager  : {eh_v[:8].tolist()}")
                        print(f"    head[0:8] capture: {ch_v[:8].tolist()}")
                        print(f"    head_diff_max: {head_diff.max().item():.5e}")
                    found = True
                    break
            if found:
                break
        if found:
            break
    if not found:
        print(f"  NO divergent stage found above thresholds rel>{args.rel_thresh} abs>{args.abs_thresh}")

    # Top-K divergent across all (call, layer, stage)
    print()
    print(f"=== Top-{args.top} divergent (rel norm-diff) across all (call, layer, stage) ===")
    flat_rel = rel_d.flatten()
    flat_abs = abs_d.flatten()
    # Mask tiny abs (avoid huge rel from 0 norms)
    mask = flat_abs >= args.abs_thresh
    flat_rel_masked = torch.where(mask, flat_rel, torch.zeros_like(flat_rel))
    topk = torch.topk(flat_rel_masked, k=min(args.top, flat_rel_masked.numel()))
    for rank, (val, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
        if val == 0:
            break
        i = idx // (n_layers * n_stages)
        L = (idx // n_stages) % n_layers
        S = idx % n_stages
        print(f"  #{rank+1:2d} call={i:2d} layer={L:2d} stage={S:2d} ({STAGE_NAME.get(S, '?'):35s}) rel={val:.3e} abs={flat_abs[idx].item():.4e}")

    # Per-stage aggregate: how often does this stage diverge across (call, layer)?
    print()
    print("=== Per-stage divergence count (over all call*layer) ===")
    stage_counts = diverge.sum(dim=(0, 1)).tolist()
    total = n_calls * n_layers
    for S in range(n_stages):
        cnt = stage_counts[S]
        bar = "#" * min(50, cnt * 50 // max(total, 1))
        print(f"  stage {S:2d} ({STAGE_NAME.get(S, '?'):35s}) {cnt:5d}/{total} {bar}")

    # Per-layer aggregate
    print()
    print("=== Per-layer divergence count (over all call*stage) ===")
    layer_counts = diverge.sum(dim=(0, 2)).tolist()
    total_l = n_calls * n_stages
    for L in range(n_layers):
        cnt = layer_counts[L]
        if cnt > 0:
            bar = "#" * min(50, cnt * 50 // max(total_l, 1))
            print(f"  layer {L:3d}  {cnt:5d}/{total_l} {bar}")


if __name__ == "__main__":
    main()
