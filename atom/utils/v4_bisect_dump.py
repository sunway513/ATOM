"""V4 capture-vs-eager bisect dump utility.

Activated by env ATOM_V4_BISECT_DUMP=<path>. Hooks all DSV4 Block.forward
calls. Per-layer + per-step buffers:
- norms[layer, step] = scalar norm of layer output
- head_tokens[layer, step, :HEAD_SIZE] = first HEAD_SIZE elements of output

After each forward, if step counter reaches max, dumps to <path>.

Capture-friendly: in-place writes to pre-allocated buffer, GPU-side step
counter incremented by tensor add (no Python int leak).

Usage:
  ATOM_V4_BISECT_DUMP=/tmp/eager.pt    [run server eager]   ... 
  ATOM_V4_BISECT_DUMP=/tmp/capture.pt  [run server capture] ...
  python -c "import torch; e=torch.load('/tmp/eager.pt'); c=torch.load('/tmp/capture.pt'); 
             d=(e['norms']-c['norms']).abs(); print(d.argmax(), d.max())"
"""
import os
import torch


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
            # We use this dump only in eager / decode-replay (which runs the
            # captured graph but our hook callbacks DO NOT fire during
            # replay because pytorch only fires hooks in the eager-trace
            # path). This avoids HIP "operation not permitted when stream is
            # capturing" failures during the capture warmup forward.
            try:
                if torch.cuda.is_current_stream_capturing():
                    return
            except Exception:
                pass
            # Clamp step to avoid OOB
            s_idx = step_t.clamp(max=max_steps - 1)
            # Compute scalar norm in BF16 directly to avoid allocating an
            # FP32 cast tensor; casts in capture context may trigger HIP
            # alloc not permitted under stream capture.
            with torch.no_grad():
                n_bf16 = torch.linalg.vector_norm(output.detach()).to(torch.float32)
                norms[layer_id].scatter_(
                    0, s_idx.unsqueeze(0), n_bf16.reshape(1)
                )
                # First-N preview
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
            if type(mod).__name__ == "Block":
                lid = getattr(mod, "layer_id", None)
                if lid is not None and 0 <= lid < self.num_layers:
                    mod.register_forward_hook(self.make_hook(lid))
                    installed += 1
        return installed

    def record_hidden(self, hidden_states, is_prefill):
        """Per-call dump of hidden_states preview (called from run_model).
        Records first 16 elements of each row, plus norm.
        Capture-friendly because called outside graph context (eager prefill +
        eager logits-after-replay path).
        """
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.numel() == 0:
            return
        # Initialize lazy per-call buffers on first call
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
            # Try to save once we have enough data
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
        # Avoid .item() sync inside cudagraph capture region.
        # Capture is detected by current_stream != default stream OR by
        # checking if we're inside torch.cuda.graph context.
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        cur_step = int(self.step_t.item())
        n_calls = len(getattr(self, "_call_norms", []))
        # Save when EITHER step counter is high enough OR we have many run_model calls
        if not force and cur_step < self.max_steps // 2 and n_calls < 16:
            return
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
        }
        torch.save(d, path)
        self._saved = True
        print(f"[V4BisectDump] saved {cur_step} steps -> {path}")


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
    print(f"[V4BisectDump] installed {n_installed} hooks on {n_layers} layers, save -> {path}")
    model_runner._v4_bisect_dump = dump
    return dump
