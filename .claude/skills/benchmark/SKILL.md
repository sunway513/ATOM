# Skill: ATOM Inference Benchmarking

## Description
Run ATOM inference benchmarks using InferenceX benchmark_serving, compare results across Docker images, and analyze throughput/latency metrics. Covers the full workflow from server launch to results analysis.

## When to Use
- Benchmarking a new ATOM Docker image against a reference
- Comparing performance across different TP configurations, sequence lengths, or concurrency levels
- Validating that performance fixes actually improve throughput
- Generating benchmark data for reports

---

## 1. Benchmark Setup

### Prerequisites
```bash
# Inside Docker container
pip install -q aiohttp requests tqdm transformers
export OMP_NUM_THREADS=1
```

### Directory structure
```
/workspace/results/           # Inside container
/mnt/m2m_nobackup/pensun/results/  # Host mount
    public/                   # Reference image results
    clean_v7/                 # Clean image results
    dsr1_public/              # DeepSeek-R1 reference
```

---

## 2. Running Benchmarks

### Single benchmark
```bash
python3 -m atom.benchmarks.benchmark_serving \
    --model /models/MODEL \
    --backend vllm \
    --base-url http://localhost:8000 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --random-range-ratio 0.8 \
    --num-prompts 80 \
    --max-concurrency 8 \
    --request-rate inf \
    --ignore-eos \
    --save-result \
    --result-dir /workspace/results \
    --result-filename result_tp8_isl1024_osl1024_conc8.json \
    --percentile-metrics ttft,tpot,itl,e2el
```

### Key parameters
| Parameter | Description | Typical Values |
|-----------|------------|----------------|
| `--random-input-len` | Input sequence length | 1024, 8192 |
| `--random-output-len` | Output sequence length | 1024, 8192 |
| `--max-concurrency` | Concurrent requests | 4, 8, 16, 32, 64, 128 |
| `--num-prompts` | Total requests to send | concurrency * 10 |
| `--random-range-ratio` | Min/max ratio for random lengths | 0.8 |
| `--request-rate inf` | Send requests as fast as possible | inf |
| `--ignore-eos` | Don't stop at EOS (measure full output length) | Always use |

### Benchmark matrix
Standard benchmark suite covers these configurations:

**Phase 1: Short context (1k/1k)**
```
TP=8, ISL=1024, OSL=1024, CONC={4,8,16,32}
```

**Phase 2: Long output (1k/8k)**
```
TP=8, ISL=1024, OSL=8192, CONC={4,8,16,32,64,128}
```

**Phase 3: Long input (8k/1k)**
```
TP=8, ISL=8192, OSL=1024, CONC={4,8,16,32,64,128}
```

**Phase 4: Lower TP (if model fits)**
```
TP=4 or TP=1, same ISL/OSL/CONC combos
```

---

## 3. Helper Functions

### start_server
```bash
start_server() {
    local tp=$1; shift; local extra_args="$@"

    # Kill previous server
    pkill -9 -f "atom.entrypoints" 2>/dev/null || true
    pkill -9 -f "ModelRunner" 2>/dev/null || true
    sleep 5

    # Clear GPU memory
    python3 -c "
import torch
for i in range(8):
    try:
        torch.cuda.set_device(i); torch.cuda.empty_cache()
    except: pass
" 2>/dev/null || true
    sleep 15

    # Launch server
    python3 -m atom.entrypoints.openai_server \
        --model $MODEL --server-port $PORT -tp $tp \
        --kv_cache_dtype fp8 --block-size 16 \
        $extra_args > /workspace/server.log 2>&1 &
    SERVER_PID=$!

    # Wait for health (up to 20 min)
    for i in $(seq 1 240); do
        curl -s http://localhost:$PORT/health > /dev/null 2>&1 && return 0
        kill -0 $SERVER_PID 2>/dev/null || return 1
        sleep 5
    done
    return 1
}
```

### run_bench
```bash
run_bench() {
    local isl=$1 osl=$2 conc=$3 tp=$4
    local num_prompts=$((conc * 10))
    local result_file="${MODEL_PREFIX}_tp${tp}_isl${isl}_osl${osl}_conc${conc}"

    # Skip if already done
    [ -f "$RESULT_DIR/${result_file}.json" ] && return 0

    timeout 7200 python3 -m atom.benchmarks.benchmark_serving \
        --model $MODEL --backend vllm --base-url http://localhost:$PORT \
        --dataset-name random \
        --random-input-len $isl --random-output-len $osl \
        --random-range-ratio 0.8 \
        --num-prompts $num_prompts --max-concurrency $conc \
        --request-rate inf --ignore-eos \
        --save-result --result-dir $RESULT_DIR \
        --result-filename ${result_file}.json \
        --percentile-metrics ttft,tpot,itl,e2el
}
```

---

## 4. Results Analysis

### Result JSON format
Each benchmark produces a JSON file with:
```json
{
    "output_throughput": 1234.5,        // output tokens/sec
    "total_token_throughput": 2345.6,   // total tokens/sec (input+output)
    "mean_ttft_ms": 45.2,              // time to first token (ms)
    "median_tpot_ms": 8.1,             // time per output token (ms)
    "mean_itl_ms": 8.3,                // inter-token latency (ms)
    "mean_e2el_ms": 850.0              // end-to-end latency (ms)
}
```

### Key metrics
| Metric | What It Measures | Lower/Higher Better |
|--------|-----------------|-------------------|
| `output_throughput` | Output tokens per second | Higher |
| `total_token_throughput` | All tokens per second | Higher |
| `mean_ttft_ms` | Time to first token | Lower |
| `median_tpot_ms` | Time per output token | Lower |

### Quick summary script
```bash
for f in /workspace/results/*.json; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .json)
    python3 -c "
import json; d=json.load(open('$f'))
print(f'$base: output={d.get(\"output_throughput\",0):.1f} total={d.get(\"total_token_throughput\",0):.1f} ttft={d.get(\"mean_ttft_ms\",0):.1f}ms tpot={d.get(\"median_tpot_ms\",0):.1f}ms')
"
done
```

### Comparison table generator
```python
import json, glob, os

def load_results(result_dir, prefix):
    results = {}
    for f in glob.glob(f"{result_dir}/{prefix}_*.json"):
        base = os.path.basename(f).replace('.json', '')
        # Extract config from filename: prefix_tpN_islN_oslN_concN
        parts = base.split('_')
        key = '_'.join(parts[1:])  # remove prefix
        with open(f) as fh:
            results[key] = json.load(fh)
    return results

public = load_results('/workspace/results', 'public')
clean = load_results('/workspace/results', 'clean')

print(f"{'Config':<35} {'Public':>10} {'Clean':>10} {'Ratio':>8}")
print("-" * 65)
for key in sorted(public.keys()):
    if key in clean:
        pub_tp = public[key].get('output_throughput', 0)
        cln_tp = clean[key].get('output_throughput', 0)
        ratio = cln_tp / pub_tp if pub_tp > 0 else 0
        print(f"{key:<35} {pub_tp:>10.1f} {cln_tp:>10.1f} {ratio:>7.1%}")
```

---

## 5. Remote Node Execution

### Deploy and run on remote nodes
```bash
NODE="uswslocpm2m-106-1236.amd.com"

# 1. Write benchmark script locally
cat > /tmp/bench.sh << 'EOF'
#!/bin/bash
# ... benchmark script content ...
EOF

# 2. Deploy
scp /tmp/bench.sh ${NODE}:/mnt/m2m_nobackup/pensun/bench_serving/

# 3. Launch (never inline complex Docker commands via SSH)
ssh ${NODE} 'bash /mnt/m2m_nobackup/pensun/bench_serving/bench.sh'

# 4. Monitor
ssh ${NODE} 'docker logs -f atom-bench 2>&1 | tail -20'

# 5. Collect results
scp ${NODE}:/mnt/m2m_nobackup/pensun/results/*.json /home/pensun/results/
```

### Multi-node deployment
When running on multiple nodes (e.g., public vs clean on separate nodes):
```bash
NODE1="uswslocpm2m-106-881.amd.com"    # Public image
NODE2="uswslocpm2m-106-1236.amd.com"   # Clean image

# Deploy to both
for NODE in $NODE1 $NODE2; do
    scp /tmp/bench.sh ${NODE}:/mnt/m2m_nobackup/pensun/bench_serving/
done

# Launch in parallel
ssh ${NODE1} 'nohup bash /mnt/.../bench.sh --prefix public > /mnt/.../run.log 2>&1 &'
ssh ${NODE2} 'nohup bash /mnt/.../bench.sh --prefix clean > /mnt/.../run.log 2>&1 &'
```

---

## 6. Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Server OOM | Killed during model load | Reduce `--max-model-len`, or increase TP |
| Benchmark timeout | `timeout 7200` exits 124 | High concurrency + long output overwhelms server. Reduce concurrency or output length |
| Stale GPU processes | Server won't start, GPU memory full | `pkill -9 -f atom.entrypoints && sleep 15` |
| Results already exist | `run_bench` silently skips | Delete old JSON or use different prefix |
| SSH escaping | `Ambiguous output redirect` | Never inline Docker commands via SSH. Use script files. |
| Node 1 GPU issues | Memory faults on all shapes | Prefer Node 2 (1236) for reliable benchmarks |

---

## Anti-Patterns

1. **Don't compare results from different hardware** — MI300X vs MI355X have different performance profiles
2. **Don't skip the warmup** — first few requests are always slower (JIT compilation, Triton caching)
3. **Don't forget `--ignore-eos`** — without it, output length varies and throughput numbers aren't comparable
4. **Don't use `--request-rate` below inf for throughput tests** — rate limiting artificially caps throughput
5. **Don't compare decode and prefill numbers directly** — they measure fundamentally different things
6. **Don't inline Docker commands via SSH** — write scripts, scp them, launch via `bash /path/to/script.sh`
