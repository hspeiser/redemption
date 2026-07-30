#!/bin/bash
f=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs/control_proof_20260730T155232Z/verdict.json
grep -o '"phase": "[^"]*"' "$f" | head -40
echo "== result metrics =="
grep -A2 '"gates\|maximum\|region\|progress' \
  /mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs/control_proof_20260730T155232Z/result.json | head -30
