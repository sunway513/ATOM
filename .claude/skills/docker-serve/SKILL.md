# Skill: ATOM Docker Build, Serve, and Debug

## Description
End-to-end workflow for building, patching, serving, and debugging ATOM Docker images for LLM inference on AMD GPUs. Covers the Docker run template, in-container patching, image tagging, server launch, and health checking.

## When to Use
- Setting up ATOM inference for a new model or Docker image
- Patching an existing Docker image without rebuilding from scratch
- Debugging server startup failures or configuration issues
- Comparing behavior between two Docker images

---

## 1. Docker Run Template

### Full server launch
```bash
docker run -d \
    --name atom-server \
    --network host \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc host \
    --shm-size 256g \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /path/to/models:/models:ro \
    -v /path/to/results:/workspace/results \
    IMAGE_NAME \
    python3 -m atom.entrypoints.openai_server \
        --model /models/MODEL_NAME \
        --server-port 8000 \
        -tp 8 \
        --kv_cache_dtype fp8 \
        --block-size 16 \
        --max-model-len 4096 \
        --trust-remote-code \
        --enforce-eager
```

### Debug container (sleep infinity for patching)
```bash
docker run -d \
    --name atom-debug \
    --network host \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --ipc host \
    --shm-size 256g \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v /path/to/models:/models:ro \
    IMAGE_NAME \
    sleep infinity
```

### Required flags explained
| Flag | Why |
|------|-----|
| `--device /dev/kfd --device /dev/dri` | GPU access on ROCm |
| `--group-add video` | GPU device permissions |
| `--ipc host` | Shared memory for multi-GPU |
| `--shm-size 256g` | Large shared memory for tensor parallel |
| `--cap-add SYS_PTRACE` | ROCm profiler/debugger access |
| `--security-opt seccomp=unconfined` | Required for some ROCm operations |
| `--network host` | Direct port access (no -p mapping needed) |

---

## 2. Patch-Commit-Test Workflow

**Never rebuild Docker for debugging.** Patch inside running containers instead.

### Step 1: Start debug container
```bash
docker run -d --name atom-debug ... IMAGE sleep infinity
```

### Step 2: Apply patches

**Option A: sed for simple string replacements**
```bash
docker exec atom-debug sed -i 's/old_string/new_string/' /app/ATOM/atom/model_ops/moe.py
```

**Option B: Python script for complex patches**
```bash
docker exec atom-debug python3 -c "
content = open('/app/ATOM/atom/model_ops/moe.py').read()
content = content.replace('old_pattern', 'new_pattern')
open('/app/ATOM/atom/model_ops/moe.py', 'w').write(content)
print('Patched')
"
```

**Option C: docker cp for file replacement**
```bash
docker cp local_fixed_file.py atom-debug:/app/ATOM/atom/model_ops/moe.py
```

### Step 3: Commit as new image
```bash
docker commit atom-debug IMAGE:patched-v1
```

### Step 4: Test with new image
```bash
docker rm -f atom-debug
docker run -d --name atom-test ... IMAGE:patched-v1 \
    python3 -m atom.entrypoints.openai_server --model /models/MODEL ...
```

### Step 5: Verify
```bash
# Wait for server health
for i in $(seq 1 60); do
    curl -s http://localhost:8000/health > /dev/null 2>&1 && break
    sleep 5
done

# Test completions
curl -s http://localhost:8000/v1/completions \
  -d '{"model":"/models/MODEL","prompt":"The capital of France is","max_tokens":30,"temperature":0}'
```

---

## 3. Server Configuration

### Model-specific configurations

**DeepSeek-R1 671B (MXFP4)**
```bash
python3 -m atom.entrypoints.openai_server \
    --model /models/DeepSeek-R1-0528-MXFP4 \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --max-model-len 4096   # or 10240 for long context
```

**GPT-OSS 120B (MXFP4)**
```bash
python3 -m atom.entrypoints.openai_server \
    --model /models/gpt-oss-120b \
    -tp 8 \                         # or tp 1 for single-GPU
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --max-model-len 4096
```

### Common server flags
| Flag | Description | Default |
|------|------------|---------|
| `-tp N` | Tensor parallel degree | 1 |
| `--kv_cache_dtype fp8` | FP8 KV cache (saves memory) | auto |
| `--block-size 16` | KV cache block size | 16 |
| `--max-model-len N` | Max sequence length | Model default |
| `--enforce-eager` | Disable CUDA graphs (debug) | False |
| `--trust-remote-code` | Allow custom model code | False |

### Health check
```bash
curl -s http://localhost:8000/health
# Returns 200 when ready
```

### Server startup wait pattern
```bash
start_server() {
    python3 -m atom.entrypoints.openai_server ... &
    SERVER_PID=$!
    for i in $(seq 1 240); do
        curl -s http://localhost:8000/health > /dev/null 2>&1 && return 0
        kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; return 1; }
        sleep 5
    done
    echo "Timeout (20 min)"; return 1
}
```

---

## 4. Image Tagging Convention

```
username_date_rocmversion_aiterversion_atomversion
```

**Examples:**
- `pensun_20260219_rocm720_aiter011_atom010` — base clean image
- `IMAGE:patched-v1` — patched variant

**Registry:** `rocm/atom-private`

---

## 5. GPU Cleanup Between Runs

When restarting the server inside the same container:
```bash
# Kill all related processes
pkill -9 -f "atom.entrypoints" 2>/dev/null || true
pkill -9 -f "ModelRunner" 2>/dev/null || true
pkill -9 -f "from multiprocessing" 2>/dev/null || true
sleep 5

# Clear GPU memory
python3 -c "
import torch
for i in range(8):
    try:
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(i)
    except: pass
"
sleep 15  # Wait for GPU memory release
```

---

## 6. Updating ATOM/AITER from GitHub Inside Container

```bash
# Update ATOM
cd /app/ATOM
pip uninstall -y atom 2>/dev/null || true
git fetch origin main && git checkout origin/main
pip install -e . --no-deps

# Update AITER
cd /app/aiter-test  # or /root/aiter
pip uninstall -y amd-aiter 2>/dev/null || true
git fetch origin main && git checkout origin/main
rm -rf aiter/jit/build aiter/jit/*.so
git submodule sync && git submodule update --init --recursive
python setup.py develop
```

---

## 7. Troubleshooting

### Server won't start
```bash
# Check logs
docker logs atom-server 2>&1 | tail -100

# Common issues:
# - OOM: reduce --max-model-len or increase --shm-size
# - GPU not found: check --device flags and --group-add video
# - Module import error: check AITER/ATOM versions match
```

### Server starts but output is wrong
Use the `triage-accuracy` skill.

### Server starts but output is slow
Use the `triage-perf` skill.

### Remote node deployment
```bash
# Deploy scripts to remote node
scp bench.sh NODE:/path/to/bench.sh

# Launch via SSH (never inline complex Docker commands)
ssh NODE 'bash /path/to/bench.sh'

# Monitor
ssh NODE 'docker logs -f atom-server 2>&1 | tail -50'
```

---

## Anti-Patterns

1. **Don't rebuild Docker for each test** — patch + commit is 100x faster
2. **Don't inline complex Docker commands via SSH** — shell escaping breaks. Write scripts, scp, then `ssh node 'bash script.sh'`
3. **Don't forget GPU cleanup between server restarts** — leftover processes hold GPU memory
4. **Don't skip the health check** — server takes 5-20 min to load large models
5. **Don't mount model directories read-write** — use `:ro` to prevent accidental modification
