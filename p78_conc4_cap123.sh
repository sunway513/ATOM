#!/bin/bash
# conc=4 server, capture [1,2,3] — does cap_size=3 work?
set -e
PREFIX=conc4_cap123
N=3
LOGD=/workspace/ATOM-lingpeng/logs/p78_${PREFIX}
mkdir -p $LOGD

cleanup() {
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null || true
  pkill -9 -f "spawn_main" 2>/dev/null || true
  pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
  pkill -9 -f "atom" 2>/dev/null || true
  sleep 15
}

for RUN in $(seq 1 $N); do
  TAG=${PREFIX}_r${RUN}
  echo "=== $TAG START $(date -Iseconds) ==="
  cleanup
  rm -rf /root/.cache/atom/* 2>/dev/null || true

  env \
    ATOM_USE_TRITON_MOE=1 \
    AITER_LOG_LEVEL=WARNING \
    /opt/venv/bin/python -m atom.entrypoints.openai_server \
      --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
      --kv_cache_dtype fp8 -tp 8 \
      --max-num-seqs 4 --max-num-batched-tokens 4096 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 --port 8000 \
      --cudagraph-capture-sizes "[1,2,3]" \
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
    echo "[$TAG] $(grep 'flexible-extract:' $LOGD/gsm8k_$TAG.log | tail -1)"
  fi

  cleanup
done

echo "=== ${PREFIX} SUMMARY ==="
for r in 1 2 3; do
  f=$LOGD/gsm8k_${PREFIX}_r${r}.log
  [ -f "$f" ] && grep "flexible-extract:" $f | tail -1 | sed "s/^/  ${PREFIX}_r${r}: /"
done
