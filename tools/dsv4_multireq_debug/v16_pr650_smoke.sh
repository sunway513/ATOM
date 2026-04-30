#!/bin/bash
# v16: Test if Lingpeng's PR #650 (feat/deepseek-v4-pr1-skeleton) fixes
# multi-concurrency. Replicates v11/v15 conditions: 4 parallel curls
# with --max-num-seqs 4 (full conc=4). Uses PR650's recommended invocation.
set -e
cd /workspace/ATOM-feat
LOGD=/workspace/ATOM-lingpeng/logs
mkdir -p $LOGD
echo "=== v16 START $(date -Iseconds) ==="
rm -rf /root/.cache/atom/* 2>/dev/null || true

/opt/venv/bin/python -c "import atom; print('[v16] atom package:', atom.__file__)"
echo "[v16] git HEAD:"; git -C /workspace/ATOM-feat log --oneline -1

env \
  ATOM_USE_TRITON_MOE=1 \
  AITER_LOG_LEVEL=WARNING \
  /opt/venv/bin/python -m atom.entrypoints.openai_server \
    --model /data/hf_models/deepseek-ai/DeepSeek-V4-Pro \
    --kv_cache_dtype fp8 -tp 8 \
    --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --max-model-len 4096 --enforce-eager \
    --gpu-memory-utilization 0.85 --port 8000 \
    > $LOGD/oai_server_v16.log 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  sleep 5
  curl -sf http://localhost:8000/v1/models > /dev/null 2>&1 && { echo "[v16] ready ${i}*5s"; break; }
  kill -0 $SRV 2>/dev/null || { echo "FAIL"; tail -30 $LOGD/oai_server_v16.log; SRV=""; break; }
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
  mkdir -p $LOGD/v16_smoke
  CURL_PIDS=()
  for i in 0 1 2 3; do
    P=$(/opt/venv/bin/python -c "import json,sys; print(json.dumps({'model':'/data/hf_models/deepseek-ai/DeepSeek-V4-Pro','prompt':sys.argv[1],'max_tokens':80,'temperature':0.0}))" "${PREFIX}${PROMPTS[$i]}")
    curl -s -m 180 http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "$P" > $LOGD/v16_smoke/p$i.json &
    CURL_PIDS+=($!)
  done
  for pid in "${CURL_PIDS[@]}"; do wait $pid || true; done
  PASS=0
  for i in 0 1 2 3; do
    TXT=$(/opt/venv/bin/python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("choices",[{}])[0].get("text","NO_TEXT"))' $LOGD/v16_smoke/p$i.json 2>&1)
    EXP=${EXPECTED[$i]}
    echo "[p$i exp=$EXP] $(echo "$TXT" | head -c 200)"
    if echo "$TXT" | grep -q "$EXP"; then PASS=$((PASS+1)); fi
  done
  echo "v16 SMOKE: $PASS/4"
  echo "$PASS/4" > $LOGD/v16_smoke_score
  if [ "$PASS" -ge 2 ]; then
    /opt/venv/bin/lm_eval \
      --model local-completions \
      --model_args "model=/data/hf_models/deepseek-ai/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,num_concurrent=4,max_retries=3,tokenized_requests=False" \
      --tasks gsm8k --num_fewshot 5 --limit 20 --batch_size 4 \
      --gen_kwargs 'max_gen_toks=1024,temperature=0.0,do_sample=False' \
      > $LOGD/lm_eval_v16.log 2>&1
    echo "rc=$?" > $LOGD/lm_eval_v16_rc
  fi
fi
[ -n "$SRV" ] && kill -9 $SRV 2>/dev/null
sleep 5
echo "=== v16 DONE $(date -Iseconds) ==="
