#!/usr/bin/env bash
# Pull DeepSeek-V4-Pro weights to /data/DeepSeek-V4-Pro on the host where this runs.
# Idempotent + resume-safe. Designed for mi355-15 / any MI355X dev box.
#
# Usage:
#   bash recipes/pull_dsv4_pro_weights.sh
#
# Override target dir:
#   DSV4_TARGET=/scratch/DeepSeek-V4-Pro bash recipes/pull_dsv4_pro_weights.sh
#
# Prereq: huggingface_hub or hf-transfer installed. Will install if missing.

set -euo pipefail

DSV4_REPO="${DSV4_REPO:-deepseek-ai/DeepSeek-V4-Pro}"
DSV4_TARGET="${DSV4_TARGET:-/data/DeepSeek-V4-Pro}"
DSV4_CACHE="${DSV4_CACHE:-/data/.hf_cache}"

echo "=== DeepSeek-V4-Pro weights pull ==="
echo "  repo:   ${DSV4_REPO}"
echo "  target: ${DSV4_TARGET}"
echo "  cache:  ${DSV4_CACHE}"

# Disk-space sanity check (~1.6 TB total weights, FP4-packed; expect ~400-800 GB on disk)
parent="$(dirname "${DSV4_TARGET}")"
mkdir -p "${parent}" "${DSV4_CACHE}"
avail_gib=$(df -BG "${parent}" | tail -1 | awk '{print $4}' | tr -d 'G')
if [[ "${avail_gib}" -lt 1000 ]]; then
    echo "WARN: only ${avail_gib} GiB free at ${parent}; DSV4-Pro needs ~600-800 GiB. Continuing anyway."
fi

# Install hf-transfer if not already present (much faster multi-stream downloads)
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install --quiet --upgrade "huggingface_hub[hf_transfer,cli]>=0.20.0"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME="${DSV4_CACHE}"

# Download — uses --local-dir-use-symlinks=False to materialize files (some loaders won't follow symlinks)
echo "=== starting download (resume-safe; ctrl-c then re-run to continue) ==="
huggingface-cli download "${DSV4_REPO}" \
    --local-dir "${DSV4_TARGET}" \
    --local-dir-use-symlinks False \
    --max-workers 8 \
    --resume-download

# Sanity: count safetensors + verify config.json
echo "=== verification ==="
config="${DSV4_TARGET}/config.json"
if [[ ! -f "${config}" ]]; then
    echo "FAIL: ${config} missing — download incomplete"
    exit 1
fi

n_safetensors=$(find "${DSV4_TARGET}" -maxdepth 2 -name '*.safetensors' | wc -l)
total_bytes=$(du -sb "${DSV4_TARGET}" | awk '{print $1}')
total_gib=$((total_bytes / 1024 / 1024 / 1024))
echo "  config.json    : present"
echo "  *.safetensors  : ${n_safetensors} files"
echo "  total on disk  : ${total_gib} GiB"

if [[ "${n_safetensors}" -lt 50 ]]; then
    echo "WARN: only ${n_safetensors} safetensors files — DSV4-Pro typically has 100+. Re-run to resume."
    exit 2
fi

echo "=== DONE ==="
echo "Weights ready at ${DSV4_TARGET}"
echo "Next: ATOM_USE_TRITON_MOE=1 AITER_LOG_LEVEL=WARNING \\"
echo "      python -m atom.examples.simple_inference \\"
echo "      --model ${DSV4_TARGET} --kv_cache_dtype fp8 -tp 8 \\"
echo "      --max-num-seqs 4 --max-model-len 1024 --enforce-eager \\"
echo "      --temperature 0.0 --max-tokens 512"
