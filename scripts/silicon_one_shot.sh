#!/bin/bash
# silicon_one_shot.sh — one-cmd DSV4 silicon test
# Usage: ./silicon_one_shot.sh <mode> <use_w4_path> <num_prompts> <max_tokens> <tag>
# Example: ./silicon_one_shot.sh single 1 1 32 w4_bug3_fix
#
# Auto: GPU preflight, kill orphans, run inside container, parse JSON, verify VRAM cleanup.
set -e
MODE=${1:-single}
W4=${2:-1}
NPROMPT=${3:-1}
MAXTOK=${4:-32}
TAG=${5:-test}

CONTAINER=atom_dsv4_feat
LOGFILE=/workspace/ATOM-lingpeng/logs/silicon_${TAG}.log
JSONFILE=/workspace/ATOM-lingpeng/logs/silicon_${TAG}.json
RCFILE=/workspace/ATOM-lingpeng/logs/silicon_${TAG}_rc

echo "=== [silicon_one_shot] tag=${TAG} mode=${MODE} W4=${W4} prompts=${NPROMPT} tok=${MAXTOK} ==="

# Preflight: kill GPU orphans + verify VRAM clean
PIDS=$(rocm-smi --showpids 2>&1 | grep -E '^[0-9]+ +python' | awk '{print $1}' | tr '\n' ' ')
if [ -n "$PIDS" ]; then
  echo "[preflight] killing GPU orphans: $PIDS"
  sudo kill -9 $PIDS 2>/dev/null || true
  sleep 4
fi
USED=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -8 | awk -F, '{sum+=$3} END {print sum/1024/1024 " MB"}')
echo "[preflight] VRAM total used: $USED"

# Run silicon test inside container
docker exec $CONTAINER bash -lc "
  rm -f $RCFILE $JSONFILE $LOGFILE
  ATOM_DSV4_USE_W4_PATH=$W4 \
  ATOM_DSV4_UNSAFE_MULTIREQ_DEV=$W4 \
  AITER_CONFIG_FMOE=/workspace/aiter-lingpeng/aiter/configs/model_configs/dsv4_fp4_tuned_fmoe.csv \
  AITER_LOG_LEVEL=WARNING \
  PYTHONPATH=/workspace/ATOM-lingpeng \
  /opt/venv/bin/python -m tests.silicon.silicon_fact_multireq \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs $([ \"$MODE\" = single ] && echo 1 || echo 4) \
    --max-model-len 2048 --enforce-eager \
    --gpu-memory-utilization 0.85 \
    --num-prompts $NPROMPT --max-tokens $MAXTOK \
    --out $JSONFILE \
    > $LOGFILE 2>&1
  echo rc=\$? > $RCFILE
"

# Parse + report
docker exec $CONTAINER bash -lc "
  echo '=== RC ==='; cat $RCFILE
  echo '=== JSON ==='; cat $JSONFILE 2>/dev/null | head -100
  echo '=== HITS/MISSES ==='
  echo HIT=\$(grep -c HIT $LOGFILE 2>/dev/null || echo 0)
  echo MISS=\$(grep -c MISS $LOGFILE 2>/dev/null || echo 0)
  echo '=== ERRORS ==='
  grep -E 'Error|Traceback|RuntimeError' $LOGFILE 2>/dev/null | head -5
"

# Postflight cleanup
PIDS=$(rocm-smi --showpids 2>&1 | grep -E '^[0-9]+ +python' | awk '{print $1}' | tr '\n' ' ')
if [ -n "$PIDS" ]; then
  echo "[postflight] killing residual GPU pids: $PIDS"
  sudo kill -9 $PIDS 2>/dev/null || true
  sleep 4
fi
USED=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -8 | awk -F, '{sum+=$3} END {print sum/1024/1024 " MB"}')
echo "[postflight] VRAM final: $USED"
