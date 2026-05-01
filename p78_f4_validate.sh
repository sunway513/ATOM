#!/bin/bash
# F4 fix silicon validation: 3x capture conc=4 (must mean ≥0.85, std ≤0.05)
#                           + 1x eager conc=4 regression (must ≥0.85)
set -e

# Capture mode 3 runs (F4 fix is now default behavior, no env var needed)
/workspace/ATOM-lingpeng/p78_run_set.sh f4_cap 3
echo

# Eager mode 1 run for regression check.
LOGD=/workspace/ATOM-lingpeng/logs/p78_f4_eager
mkdir -p $LOGD

cleanup() {
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null || true
  pkill -9 -f "spawn_main" 2>/dev/null || true
  pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
  pkill -9 -f "atom" 2>/dev/null || true
  sleep 15
}

cleanup
TAG=f4_eager_r1
echo "=== $TAG START $(date -Iseconds) ==="
rm -rf /root/.cache/atom/* 2>/dev/null || true

env \
  ATOM_USE_TRITON_MOE=1 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.entrypoints.openai_server \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --max-model-len 4096 --enforce-eager \
    --gpu-memory-utilization 0.85 --port 8000 \
    > $LOGD/oai_server_$TAG.log 2>&1 &
SRV=$!

for i in $(seq 1 240); do
  sleep 5
  curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && { echo "[$TAG] ready ${i}*5s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "FAIL"; tail -50 $LOGD/oai_server_$TAG.log; SRV=""; break; }
done

if [ -n "$SRV" ] && curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
  HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    /opt/venv/bin/python /workspace/ATOM-lingpeng/tools/dsv4_multireq_debug/v16_gsm8k_direct.py \
    > $LOGD/gsm8k_$TAG.log 2>&1
  SCORE=$(grep "flexible-extract:" $LOGD/gsm8k_$TAG.log | tail -1)
  echo "[$TAG] $SCORE"
fi

cleanup

echo "=== F4 SUMMARY ==="
echo "Capture conc=4 (F4 fix):"
for r in 1 2 3; do
  f=/workspace/ATOM-lingpeng/logs/p78_f4_cap/gsm8k_f4_cap_r${r}.log
  [ -f "$f" ] && grep "flexible-extract:" $f | tail -1 | sed "s/^/  f4_cap_r${r}: /"
done
echo "Eager conc=4 regression:"
grep "flexible-extract:" $LOGD/gsm8k_f4_eager_r1.log 2>/dev/null | tail -1 | sed "s/^/  f4_eager_r1: /"
