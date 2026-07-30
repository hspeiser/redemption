#!/bin/bash
set -e
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_probe.tgz
echo "== lake run dirs =="
for d in lake/*/; do ls "$d" 2>/dev/null | head -3; echo "-- $d"; break; done
find lake -maxdepth 2 -name imu.jsonl 2>/dev/null | head -3
# also check local VQ2 captures dirs on the windows side
ls /mnt/c/Users/henry/lrspeiser/ai-grand-prix/outputs 2>/dev/null | grep -i raw | head -3
nohup nice -n 15 .venv/bin/python scripts/fastsim_lake_sysid.py \
  --lake lake --limit 500 --out data/fastsim_lake_sysid.json \
  > data/lake_sysid.log 2>&1 &
echo "launched lake sysid pid $!"
