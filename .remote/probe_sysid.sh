#!/bin/bash
BASE=/mnt/c/Users/henry/lrspeiser/ai-grand-prix/telemetry
echo "== telemetry subdirs by size (top 12) =="
du -sh "$BASE"/*/ 2>/dev/null | sort -rh | head -12
for d in plan_sysid sysid plan; do
  if [ -d "$BASE/$d" ]; then
    echo "== $d: entry count + first entries =="
    ls "$BASE/$d" | wc -l
    ls "$BASE/$d" | head -10
  fi
done
