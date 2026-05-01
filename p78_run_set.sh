#!/bin/bash
# P7-r8 reusable run-set with robust cleanup. Args:
#   $1 = TAG_PREFIX (e.g. baseline, hypA)
#   $2 = N runs (e.g. 2 or 3)
#   $3+ = extra env vars to propagate (e.g. ATOM_V4_DISABLE_INDEXER_INNER_COMPRESSOR=1)
set -e
PREFIX=$1
N=$2
shift 2
EXTRA_ENV="$@"
LOGD=/workspace/ATOM-lingpeng/logs/p78_${PREFIX}
mkdir -p $LOGD
echo "=== p78 run-set PREFIX=$PREFIX N=$N EXTRA=[$EXTRA_ENV] ==="

cleanup() {
  echo "[cleanup] killing remaining python/server procs..."
  pkill -9 -f "atom.entrypoints.openai_server" 2>/dev/null || true
  pkill -9 -f "engine_core" 2>/dev/null || true
  pkill -9 -f "ModelRunner" 2>/dev/null || true
  pkill -9 -f "spawn_main" 2>/dev/null || true
  pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true
  pkill -9 -f "multiprocessing-fork" 2>/dev/null || true
  sleep 5
  # Re-sweep in case spawn_main respawned
  pkill -9 -f "spawn_main" 2>/dev/null || true
  pkill -9 -f "atom" 2>/dev/null || true
  sleep 15
}

for RUN in $(seq 1 $N); do
  TAG=${PREFIX}_r${RUN}
  echo "=== $TAG START $(date -Iseconds) ==="
  cleanup
  rm -rf /root/.cache/atom/* 2>/dev/null || true
  /opt/venv/bin/python -c "import atom; print('[$TAG] atom package:', atom.__file__)"

  env $EXTRA_ENV \
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
    kill -0 $SRV 2>/dev/null || { echo "[$TAG] FAIL — server died"; tail -50 $LOGD/oai_server_$TAG.log; SRV=""; break; }
  done

  if [ -n "$SRV" ] && curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      /opt/venv/bin/python /workspace/ATOM-lingpeng/tools/dsv4_multireq_debug/v16_gsm8k_direct.py \
      > $LOGD/gsm8k_$TAG.log 2>&1
    RC=$?
    echo "rc=$RC" > $LOGD/gsm8k_${TAG}_rc
    SCORE=$(grep "flexible-extract:" $LOGD/gsm8k_$TAG.log | tail -1)
    echo "[$TAG] $SCORE"
  fi

  cleanup
  echo "=== $TAG DONE $(date -Iseconds) ==="
done

echo "=== $PREFIX SUMMARY ==="
for r in $(seq 1 $N); do
  grep "flexible-extract:" $LOGD/gsm8k_${PREFIX}_r${r}.log 2>/dev/null | tail -1 | sed "s/^/${PREFIX}_r${r}: /"
done
