"""V4 capture-vs-eager bisect dump utility.

Activated by env ATOM_V4_BISECT_DUMP=<path>. Two complementary mechanisms:

1) PyTorch forward hooks on each Block — fire only in EAGER mode (do NOT
   fire when running a captured cudagraph; PyTorch hook callbacks live in
   Python `__call__` while replay is GPU-side IR replay). Per-layer norms
   + first-N tokens, indexed by step counter.

2) **CAPTURE-FRIENDLY inline scratch dump** — Block.forward writes its
   output (first head_size elements + norm) into a pre-allocated scratch
   tensor `[num_layers, max_steps, head_size]` indexed by a GPU step
   counter that is incremented inside the captured graph. Because the
   scratch tensor address is baked at capture time, replay rewrites the
   same memory; we then `.cpu()` snapshot AFTER each captured replay (in
   `record_hidden`) to preserve per-step trajectories.

   The Python-side gating (env check + attribute lookup) happens at
   ATTRIBUTE TIME — once `_v4_dump_inline_buf` is set on a Block, the
   forward write becomes a deterministic in-place op suitable for capture.

Usage:
  ATOM_V4_BISECT_DUMP=/tmp/eager.pt    [run server eager]
  ATOM_V4_BISECT_DUMP=/tmp/capture.pt  [run server capture]
  python -c "import torch; e=torch.load('/tmp/eager.pt'); c=torch.load('/tmp/capture.pt');
             d=(e['inline_norms']-c['inline_norms']).abs(); print(d)"
"""
import os
import torch


# Inline-capture scratch sizes — kept small to fit in the captured stream
# without dominating memory. head_size is the number of leading-elements
# preview per layer per step.
INLINE_HEAD_SIZE = 16


class V4BisectDump:
    def __init__(self, num_layers: int, max_steps: int = 64,
                 head_size: int = 16, device: str = "cuda"):
        self.norms = torch.zeros(
            num_layers, max_steps, dtype=torch.float32, device=device
        )
        self.head_tokens = torch.zeros(
            num_layers, max_steps, head_size,
            dtype=torch.float32, device=device,
        )
        # GPU step counter (single scalar tensor — capture-friendly)
        self.step_t = torch.zeros((), dtype=torch.long, device=device)
        self.max_steps = max_steps
        self.head_size = head_size
        self.num_layers = num_layers
        self._saved = False

        # Inline (capture-friendly) scratch tensors. These are written
        # INSIDE the captured graph by Block.forward via in-place ops on
        # baked addresses. After replay, we snapshot to CPU lists.
        self.inline_head_size = INLINE_HEAD_SIZE
        # Buffer holds CURRENT-step values (overwritten every replay).
        # bf16 to match model dtype; cast to fp32 on snapshot.
        # dim 1 of size N_STAGES: per-stage snapshot inside Block.forward.
        # Block-level stages 0..5 + V4Attention sub-stages 6..11.
        N_STAGES = 12
        self.n_stages = N_STAGES
        self.inline_buf = torch.zeros(
            num_layers, N_STAGES, INLINE_HEAD_SIZE, dtype=torch.bfloat16, device=device
        )
        self.inline_norm_buf = torch.zeros(
            num_layers, N_STAGES, dtype=torch.float32, device=device
        )
        # P7-r5: topk_idxs preview buffer (first TOPK_PREV elements per layer).
        TOPK_PREV = 32
        self.topk_prev_size = TOPK_PREV
        self.topk_buf = torch.zeros(
            num_layers, TOPK_PREV, dtype=torch.int64, device=device
        )
        # Per-call snapshot lists (CPU-side, accumulated per replay):
        self._inline_heads = []   # list of [num_layers, INLINE_HEAD_SIZE] cpu fp32
        self._inline_norms = []   # list of [num_layers] cpu fp32
        self._topk_snapshots = [] # list of [num_layers, TOPK_PREV] cpu int64

    def _save_path(self):
        return os.environ.get("ATOM_V4_BISECT_DUMP")

    def make_hook(self, layer_id: int):
        max_steps = self.max_steps
        head_size = self.head_size
        norms = self.norms
        head_tokens = self.head_tokens
        step_t = self.step_t
        is_last = (layer_id == self.num_layers - 1)
        outer = self

        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            # Capture-safe path: skip if cudagraph is currently capturing.
            try:
                if torch.cuda.is_current_stream_capturing():
                    return
            except Exception:
                pass
            # Clamp step to avoid OOB
            s_idx = step_t.clamp(max=max_steps - 1)
            with torch.no_grad():
                n_bf16 = torch.linalg.vector_norm(output.detach()).to(torch.float32)
                norms[layer_id].scatter_(
                    0, s_idx.unsqueeze(0), n_bf16.reshape(1)
                )
                flat = output.detach().reshape(-1)[:head_size].to(torch.float32)
                if flat.numel() < head_size:
                    flat = torch.nn.functional.pad(
                        flat, (0, head_size - flat.numel())
                    )
                head_tokens[layer_id, s_idx, :] = flat
                if is_last:
                    step_t.add_(1)
                    try:
                        outer.maybe_save(force=False)
                    except Exception:
                        pass
        return hook

    def install(self, model):
        installed = 0
        for name, mod in model.named_modules():
            tname = type(mod).__name__
            if tname == "Block":
                lid = getattr(mod, "layer_id", None)
                if lid is not None and 0 <= lid < self.num_layers:
                    mod.register_forward_hook(self.make_hook(lid))
                    mod._v4_dump_inline_buf = self.inline_buf
                    mod._v4_dump_inline_norm = self.inline_norm_buf
                    mod._v4_dump_inline_head_size = self.inline_head_size
                    installed += 1
            elif tname == "DeepseekV4Attention":
                # Also tag the V4Attention module so its forward can write
                # to sub-stage slots (6..11) of the same per-layer scratch.
                lid = getattr(mod, "layer_id", None)
                if lid is not None and 0 <= lid < self.num_layers:
                    mod._v4_dump_inline_buf = self.inline_buf
                    mod._v4_dump_inline_norm = self.inline_norm_buf
                    mod._v4_dump_inline_head_size = self.inline_head_size
                    mod._v4_dump_topk_buf = self.topk_buf
                    mod._v4_dump_topk_prev_size = self.topk_prev_size
        return installed

    def snapshot_inline(self):
        """Snapshot the inline scratch tensors (called from run_model AFTER
        each model forward / cudagraph replay). Stores fp32 cpu copies.
        Capture-safe: only called outside graph context.
        """
        try:
            head_cpu = self.inline_buf.detach().to(torch.float32).cpu().clone()
            norm_cpu = self.inline_norm_buf.detach().clone().cpu()
            self._inline_heads.append(head_cpu)
            self._inline_norms.append(norm_cpu)
        except Exception:
            pass

    def record_hidden(self, hidden_states, is_prefill):
        """Per-call dump of hidden_states preview (called from run_model).
        Records first 32 elements + norm + per-layer inline scratch
        snapshot.
        """
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.numel() == 0:
            return
        if not hasattr(self, "_call_norms"):
            self._call_norms = []
            self._call_heads = []
            self._call_is_prefill = []
            self._call_n_tokens = []
        try:
            n = torch.linalg.vector_norm(hidden_states.detach()).to(torch.float32).item()
            head = hidden_states.detach().reshape(-1)[:32].to(torch.float32).cpu().tolist()
            self._call_norms.append(n)
            self._call_heads.append(head)
            self._call_is_prefill.append(bool(is_prefill))
            self._call_n_tokens.append(int(hidden_states.shape[0]))
            # Inline per-layer snapshot — captured via baked-address writes
            # inside Block.forward. Snapshot AFTER the model forward
            # completed (this code path is post-forward in run_model).
            self.snapshot_inline()
            # P7-r5: also snapshot topk_idxs preview
            try:
                self._topk_snapshots.append(self.topk_buf.detach().clone().cpu())
            except Exception:
                pass
            try:
                self.maybe_save(force=False)
            except Exception:
                pass
        except Exception as e:
            pass

    def maybe_save(self, force: bool = False):
        path = self._save_path()
        if path is None or self._saved:
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        cur_step = int(self.step_t.item())
        n_calls = len(getattr(self, "_call_norms", []))
        if not force and cur_step < self.max_steps // 2 and n_calls < 16:
            return
        # Stack inline snapshots into [n_calls, num_layers, head_size]
        if self._inline_heads:
            inline_heads_t = torch.stack(self._inline_heads, dim=0)
            inline_norms_t = torch.stack(self._inline_norms, dim=0)
        else:
            inline_heads_t = torch.empty(0)
            inline_norms_t = torch.empty(0)
        d = {
            "norms": self.norms.cpu().contiguous(),
            "head_tokens": self.head_tokens.cpu().contiguous(),
            "step": cur_step,
            "num_layers": self.num_layers,
            "head_size": self.head_size,
            "call_norms": getattr(self, "_call_norms", []),
            "call_heads": getattr(self, "_call_heads", []),
            "call_is_prefill": getattr(self, "_call_is_prefill", []),
            "call_n_tokens": getattr(self, "_call_n_tokens", []),
            # P7-r5 inline (capture-friendly per-layer dump):
            "inline_heads": inline_heads_t,    # [n_calls, num_layers, INLINE_HEAD_SIZE]
            "inline_norms": inline_norms_t,    # [n_calls, num_layers]
            "inline_head_size": self.inline_head_size,
            "topk_snapshots": (torch.stack(self._topk_snapshots, dim=0) if self._topk_snapshots else torch.empty(0)),
            "topk_prev_size": self.topk_prev_size,
        }
        torch.save(d, path)
        self._saved = True
        print(f"[V4BisectDump] saved {cur_step} steps, {len(self._inline_heads)} inline snapshots -> {path}")


def maybe_install(model_runner, model):
    """Hook installer — called from ModelRunner after model load."""
    path = os.environ.get("ATOM_V4_BISECT_DUMP")
    if not path:
        return None
    n_layers = None
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            n_layers = len(layers)
    if n_layers is None:
        print("[V4BisectDump] could not find model.model.layers — skipping")
        return None
    dump = V4BisectDump(num_layers=n_layers, max_steps=128)
    n_installed = dump.install(model)
    print(f"[V4BisectDump] installed {n_installed} hooks (+inline scratch) on {n_layers} layers, save -> {path}")
    model_runner._v4_bisect_dump = dump
    return dump
