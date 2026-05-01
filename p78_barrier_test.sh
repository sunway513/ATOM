#!/bin/bash
# Try existing ATOM_V4_CAPTURE_FIX=1 barrier flag at conc=4 capture (3 runs)
set -e
/workspace/ATOM-lingpeng/p78_run_set.sh barrier 3 ATOM_V4_CAPTURE_FIX=1
echo "=== barrier SUMMARY ==="
for r in 1 2 3; do
  f=/workspace/ATOM-lingpeng/logs/p78_barrier/gsm8k_barrier_r${r}.log
  [ -f "$f" ] && grep "flexible-extract:" $f | tail -1 | sed "s/^/  barrier_r${r}: /"
done
