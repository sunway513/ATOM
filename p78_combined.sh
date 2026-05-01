#!/bin/bash
# Run 1 final baseline + 3 hypothesis A runs
set -e
echo "=== START $(date -Iseconds) ==="

# Final baseline run (no env)
/workspace/ATOM-lingpeng/p78_run_set.sh baseline_p3 1
echo

# Hypothesis A: 3 runs with ATOM_V4_DISABLE_INDEXER_INNER_COMPRESSOR=1
/workspace/ATOM-lingpeng/p78_run_set.sh hypA 3 ATOM_V4_DISABLE_INDEXER_INNER_COMPRESSOR=1

echo "=== ALL DONE $(date -Iseconds) ==="
echo
echo "=== FINAL TABLE ==="
echo "Baseline (no env):"
for tag in p78_baseline_r1 p78_baseline_p2_r1 p78_baseline_p3_r1; do
  for d in p78_baseline p78_baseline_p2 p78_baseline_p3; do
    f=/workspace/ATOM-lingpeng/logs/$d/gsm8k_${tag#p78_}.log
    [ -f "$f" ] && grep "flexible-extract:" $f | tail -1 | sed "s/^/  $tag: /"
  done
done
echo "Hypothesis A (ATOM_V4_DISABLE_INDEXER_INNER_COMPRESSOR=1):"
for r in 1 2 3; do
  f=/workspace/ATOM-lingpeng/logs/p78_hypA/gsm8k_hypA_r${r}.log
  [ -f "$f" ] && grep "flexible-extract:" $f | tail -1 | sed "s/^/  hypA_r${r}: /"
done
