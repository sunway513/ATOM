#!/bin/bash
# P7-r6 Phase 1B: Test the slot-per-token bake fix.
# Hypothesis: torch.full(..., slot, ...) baked capture-time slot, causing
# swa_write to write to WRONG slots at replay → cross-request contamination
# → wildly variant gsm8k. Fix: use slot_tensor.expand(N) instead.
#
# Run with both PYTORCH_STATE_WRITE=1 (fully PyTorch path) to also keep
# Phase 1's intervention. If variance gone → slot bake was the culprit.
set -u
cd /workspace/ATOM-feat/.claude/worktrees/agent-af6dd88bcf35d287a
LOGD=/workspace/ATOM-lingpeng/logs/p76_phase1b
mkdir -p $LOGD
run_gsm8k() {
  local TAG=$1
  local PT=$2
  echo "===== ${TAG} START $(date -Iseconds) ====="
  rm -rf /root/.cache/atom/* 2>/dev/null || true
  env ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING \
      ATOM_V4_PYTORCH_STATE_WRITE=$PT \
    /opt/venv/bin/python -m atom.entrypoints.openai_server \
      --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
      --kv_cache_dtype fp8 -tp 8 \
      --max-num-seqs 4 --max-num-batched-tokens 4096 \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.85 --port 8000 \
      --cudagraph-capture-sizes [4] > $LOGD/server_${TAG}.log 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do
    sleep 5
    curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && { echo "[${TAG}] ready"; break; }
    kill -0 $SRV 2>/dev/null || { echo DIED; SRV=""; break; }
  done
  if [ -n "${SRV:-}" ]; then
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      /opt/venv/bin/python /workspace/ATOM-lingpeng/tools/dsv4_multireq_debug/v16_gsm8k_direct.py \
      > $LOGD/gsm8k_${TAG}.log 2>&1
    echo "[${TAG}] gsm8k rc=$?"
    grep -E 'flexible-extract|strict-match' $LOGD/gsm8k_${TAG}.log | head -4
  fi
  if [ -n "${SRV:-}" ]; then
    kill -9 $SRV 2>/dev/null || true; pkill -9 -f openai_server 2>/dev/null || true; pkill -9 -f spawn_main 2>/dev/null || true; sleep 10
  fi
  echo "===== ${TAG} DONE $(date -Iseconds) ====="
}

echo "=== P7-r6 Phase 1B SLOT-BAKE FIX TEST START $(date -Iseconds) ==="
echo "=== With slot_per_token = slot_tensor.expand(N) — fixes WRONG slot writes ==="
run_gsm8k r6b_pt_slotfix_run1 1
run_gsm8k r6b_pt_slotfix_run2 1
run_gsm8k r6b_pt_slotfix_run3 1
echo
echo "=== ALSO TEST WITH TRITON STATE WRITES (keep Triton, just slot fix) ==="
run_gsm8k r6b_tri_slotfix_run1 0
run_gsm8k r6b_tri_slotfix_run2 0
run_gsm8k r6b_tri_slotfix_run3 0
echo
echo "=== SUMMARY ==="
for TAG in r6b_pt_slotfix_run1 r6b_pt_slotfix_run2 r6b_pt_slotfix_run3 r6b_tri_slotfix_run1 r6b_tri_slotfix_run2 r6b_tri_slotfix_run3; do
  S=$(grep "strict-match" $LOGD/gsm8k_${TAG}.log 2>/dev/null | head -1)
  F=$(grep "flexible-extract" $LOGD/gsm8k_${TAG}.log 2>/dev/null | head -1)
  echo "${TAG}: $S | $F"
done
echo "=== END $(date -Iseconds) ==="
