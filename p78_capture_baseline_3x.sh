#!/bin/bash
# P7-r8 Phase 1: re-baseline conc=4 capture mode 3 runs to confirm/falsify
# whether P7-r6's reported 0.35-0.50 variance still reproduces.
# No code changes — pure measurement.
set -e
LOGD=/workspace/ATOM-lingpeng/logs/p78_baseline
mkdir -p $LOGD

for RUN in 1 2 3; do
  TAG=p78_baseline_r${RUN}
  echo "=== $TAG START $(date -Iseconds) ==="
  rm -rf /root/.cache/atom/* 2>/dev/null || true
  /opt/venv/bin/python -c "import atom; print('[$TAG] atom package:', atom.__file__)"

  env \
    ATOM_USE_TRITON_MOE=1 \
    AITER_LOG_LEVEL=WARNING \
    /opt/venv/bin/python -m atom.entrypoints.openai_server \
      --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
      --kv_cache_dtype fp8 -tp 8 \
      --max-num-seqs 4 --max-num-batched-tokens 4096 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 --port 8000 \
      --cudagraph-capture-sizes "[1,2,4]" \
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
    echo "rc=$?" > $LOGD/gsm8k_${TAG}_rc
    SCORE=$(grep "flexible-extract:" $LOGD/gsm8k_$TAG.log | tail -1)
    echo "[$TAG] $SCORE"
  fi

  [ -n "$SRV" ] && kill -9 $SRV 2>/dev/null
  sleep 8
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null || true
  sleep 5
  echo "=== $TAG DONE $(date -Iseconds) ==="
done
echo "=== ALL 3 RUNS DONE ==="
echo "=== SUMMARY ==="
for r in 1 2 3; do
  grep "flexible-extract:" $LOGD/gsm8k_p78_baseline_r${r}.log 2>/dev/null | tail -1 | sed "s/^/r${r}: /"
done
