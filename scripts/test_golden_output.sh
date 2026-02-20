#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Quick golden output sanity test for ATOM.
#
# Runs simple_inference with temperature=0 and compares output
# against golden reference files. Catches silent numeric regressions
# from model_ops changes (wrong kernel routing, broken dtype conversion, etc).
#
# Usage:
#   # Default: Meta-Llama-3-8B-Instruct from HuggingFace cache
#   ./scripts/test_golden_output.sh
#
#   # With local model path
#   ./scripts/test_golden_output.sh /models/meta-llama/Meta-Llama-3-8B-Instruct
#
#   # With extra args (e.g. fp8 kv cache)
#   ./scripts/test_golden_output.sh /models/meta-llama/Meta-Llama-3-8B-Instruct --kv_cache_dtype fp8
#
# Requirements:
#   - 1 GPU (set HIP_VISIBLE_DEVICES if needed)
#   - Meta-Llama-3-8B-Instruct model (auto-downloads if not local)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GOLDEN_DIR="${ATOM_ROOT}/.github/workflows/golden_outputs"

# Parse args: first positional arg is model path, rest are extra args
MODEL_PATH="${1:-meta-llama/Meta-Llama-3-8B-Instruct}"
shift 2>/dev/null || true
EXTRA_ARGS="$*"

MODEL_NAME="Meta-Llama-3-8B-Instruct"
GOLDEN_FILE="${GOLDEN_DIR}/${MODEL_NAME}_golden_output.txt"

if [[ ! -f "${GOLDEN_FILE}" ]]; then
    echo "ERROR: Golden output file not found: ${GOLDEN_FILE}"
    exit 1
fi

echo "========== ATOM Golden Output Test =========="
echo "Model:       ${MODEL_PATH}"
echo "Golden file: ${GOLDEN_FILE}"
echo "Extra args:  ${EXTRA_ARGS:-none}"
echo "============================================="

# Run inference
TMPFILE=$(mktemp /tmp/atom_golden_test.XXXXXX)
trap "rm -f ${TMPFILE}" EXIT

echo ""
echo "Running simple_inference..."
python3 -m atom.examples.simple_inference \
    --model "${MODEL_PATH}" \
    ${EXTRA_ARGS} \
    --temperature 0 \
    2>&1 | grep -E '^Prompt: |^Completion:' > "${TMPFILE}"

echo ""
echo "========== Test Output =========="
cat "${TMPFILE}"

echo ""
echo "========== Comparing with golden output =========="
if diff -u -B -w --strip-trailing-cr "${TMPFILE}" "${GOLDEN_FILE}"; then
    echo ""
    echo "SUCCESS: Output matches golden reference."
    exit 0
else
    echo ""
    echo "FAILED: Output does not match golden reference."
    echo ""
    echo "If this is expected (e.g. model update), regenerate golden output:"
    echo "  cp ${TMPFILE} ${GOLDEN_FILE}"
    exit 1
fi
