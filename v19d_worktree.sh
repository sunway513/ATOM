#!/bin/bash
# v19d: capture-mode smoke + gsm8k quant gate (PR4 acceptance)
# Same config as v19b (NO --enforce-eager) but adds the v19a smoke + gsm8k
# tests so we can verify capture preserves model correctness, not just
# "server starts without crash".
set -e
cd /workspace/ATOM-feat/.claude/worktrees/agent-af6dd88bcf35d287a
LOGD=/workspace/ATOM-lingpeng/logs
mkdir -p $LOGD
TAG=v19d_wt
echo "=== $TAG START $(date -Iseconds) ==="
rm -rf /root/.cache/atom/* 2>/dev/null || true

/opt/venv/bin/python -c "import atom; print('[$TAG] atom package:', atom.__file__)"
echo "[$TAG] git HEAD: (worktree skipped)"

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
  PREFIX='Question: A box has 48 chocolates split equally among 8 friends. How many each?
Answer: 48 / 8 = 6. The answer is 6.

'
  PROMPTS=("Question: Janet is 36 years old now. In 4 years she will be how old?
Answer:" "Question: A train travels 60 miles in 2 hours. What is its speed in mph?
Answer:" "Question: Bob has 5 apples and buys 7 more. How many does he have total?
Answer:" "Question: What is 17 multiplied by 23?
Answer:")
  EXPECTED=("40" "30" "12" "391")
  mkdir -p $LOGD/${TAG}_smoke
  CURL_PIDS=()
  for i in 0 1 2 3; do
    P=$(/opt/venv/bin/python -c "import json,sys; print(json.dumps({'model':'/data/hf_models/deepseek-ai/DeepSeek-V4-Pro','prompt':sys.argv[1],'max_tokens':80,'temperature':0.0}))" "${PREFIX}${PROMPTS[$i]}")
    curl -s -m 180 http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "$P" > $LOGD/${TAG}_smoke/p$i.json &
    CURL_PIDS+=($!)
  done
  for pid in "${CURL_PIDS[@]}"; do wait $pid || true; done
  PASS=0
  for i in 0 1 2 3; do
    TXT=$(/opt/venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("choices",[{}])[0].get("text","NO_TEXT"))' $LOGD/${TAG}_smoke/p$i.json 2>&1)
    EXP=${EXPECTED[$i]}
    if echo "$TXT" | grep -q "$EXP"; then PASS=$((PASS+1)); fi
    echo "[p$i exp=$EXP] $(echo "$TXT" | head -c 100)"
  done
  echo "$PASS/4" > $LOGD/${TAG}_smoke_score
  echo "$TAG SMOKE: $PASS/4"
  if [ "$PASS" -ge 2 ]; then
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      /opt/venv/bin/python /workspace/ATOM-lingpeng/tools/dsv4_multireq_debug/v16_gsm8k_direct.py \
      > $LOGD/gsm8k_$TAG.log 2>&1
    echo "rc=$?" > $LOGD/gsm8k_${TAG}_rc
  fi
fi
[ -n "$SRV" ] && kill -9 $SRV 2>/dev/null
sleep 5
echo "=== $TAG DONE $(date -Iseconds) ==="
