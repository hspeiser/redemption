#!/bin/bash
set -e
cd ~/aigp
nohup nice -n 15 .venv/bin/python scripts/fastsim_lake_sysid.py \
  --lake /mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs/captures_dl \
  --limit 800 --out data/fastsim_lake_sysid.json \
  > data/lake_sysid.log 2>&1 &
echo "launched captures_dl sysid pid $!"
sleep 20
tail -3 data/lake_sysid.log
