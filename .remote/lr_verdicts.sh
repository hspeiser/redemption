#!/bin/bash
base=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs
for d in control_proof_20260730T155232Z control_proof_20260730T152303Z \
         control_proof_20260730T145832Z control_proof_20260730T145415Z; do
  echo "===== $d ====="
  cat "$base/$d/result.json" 2>/dev/null | head -c 700
  echo
  cat "$base/$d/verdict.json" 2>/dev/null | head -c 500
  echo
done
echo "===== submission_log tail ====="
tail -12 "$base/submission_log.txt" 2>/dev/null
