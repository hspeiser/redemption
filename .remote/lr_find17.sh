#!/bin/bash
base=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs
grep -l '"official_maximum_active_gate": 17' "$base"/control_proof_*/result.json 2>/dev/null | while read f; do
  d=$(dirname "$f")
  col=$(grep -o '"no_collision_or_safety_abort": [a-z]*' "$f" | grep -o '[a-z]*$')
  dur=$(grep -o '"duration_s": [0-9.]*' "$f" | head -1 | grep -o '[0-9.]*$')
  echo "$(basename $d): FINISH-17 no_collision=$col dur=${dur}s"
  ls "$d" | grep -E "poselog|ekfpose|replay" | head -3
done
