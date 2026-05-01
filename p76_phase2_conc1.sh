#!/bin/bash
# P7-r6 Phase 2: CONC=1 capture mode test.
# If stable + >=0.85, problem is multi-request only.
# If still variant, fundamental capture issue.
set -u
cd /workspace/ATOM-feat/.claude/worktrees/agent-af6dd88bcf35d287a
LOGD=/workspace/ATOM-lingpeng/logs/p76_phase2_conc1
mkdir -p $LOGD
run_gsm8k() {
  local TAG=$1
  local PT=$2
  local CAP=$3   # 1=capture 0=eager
  echo "===== ${TAG} START $(date -Iseconds) ====="
  rm -rf /root/.cache/atom/* 2>/dev/null || true
  CAP_ARG="--cudagraph-capture-sizes [4]"
  [ "$CAP" = "0" ] && CAP_ARG="--enforce-eager"
  env ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING \
      ATOM_V4_PYTORCH_STATE_WRITE=$PT \
    /opt/venv/bin/python -m atom.entrypoints.openai_server \
      --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
      --kv_cache_dtype fp8 -tp 8 \
      --max-num-seqs 4 --max-num-batched-tokens 4096 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 --port 8000 \
      $CAP_ARG > $LOGD/server_${TAG}.log 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do
    sleep 5
    curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && { echo "[${TAG}] ready"; break; }
    kill -0 $SRV 2>/dev/null || { echo DIED; SRV=""; break; }
  done
  if [ -n "${SRV:-}" ]; then
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      /opt/venv/bin/python /workspace/ATOM-lingpeng/tools/dsv4_multireq_debug/v16_gsm8k_conc1.py \
      > $LOGD/gsm8k_${TAG}.log 2>&1
    echo "[${TAG}] gsm8k rc=$?"
    grep -E 'flexible-extract|strict-match' $LOGD/gsm8k_${TAG}.log | head -4
  fi
  if [ -n "${SRV:-}" ]; then
    kill -9 $SRV 2>/dev/null || true; pkill -9 -f openai_server 2>/dev/null || true; pkill -9 -f spawn_main 2>/dev/null || true; sleep 10
  fi
  echo "===== ${TAG} DONE $(date -Iseconds) ====="
}

echo "=== P7-r6 Phase 2 CONC=1 CAPTURE TEST START $(date -Iseconds) ==="
run_gsm8k r6c_conc1_pt_cap_run1 1 1
run_gsm8k r6c_conc1_pt_cap_run2 1 1
run_gsm8k r6c_conc1_tri_cap_run1 0 1
run_gsm8k r6c_conc1_pt_eager_run1 1 0
echo
echo "=== SUMMARY ==="
for TAG in r6c_conc1_pt_cap_run1 r6c_conc1_pt_cap_run2 r6c_conc1_tri_cap_run1 r6c_conc1_pt_eager_run1; do
  S=$(grep "strict-match" $LOGD/gsm8k_${TAG}.log 2>/dev/null | head -1)
  F=$(grep "flexible-extract" $LOGD/gsm8k_${TAG}.log 2>/dev/null | head -1)
  echo "${TAG}: $S | $F"
done
echo "=== END $(date -Iseconds) ==="
