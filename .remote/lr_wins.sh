#!/bin/bash
base=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs
echo "== scanning recent control_proof for official finishes =="
for d in $(ls -dt "$base"/control_proof_* | head -60); do
  r="$d/result.json"
  [ -f "$r" ] || continue
  mx=$(grep -o '"official_maximum_active_gate": [0-9]*' "$r" | grep -o '[0-9]*$')
  col=$(grep -o '"no_collision_or_safety_abort": [a-z]*' "$r" | grep -o '[a-z]*$')
  dur=$(grep -o '"duration_s": [0-9.]*' "$r" | head -1 | grep -o '[0-9.]*$')
  echo "$(basename $d): max_gate=$mx no_collision=$col dur=${dur}s"
done
