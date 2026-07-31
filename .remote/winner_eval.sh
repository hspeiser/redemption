#!/bin/bash
cd ~/aigp
for ck in "$@"; do
  [ -f "$ck" ] || continue
  echo "===== $ck ====="
  .venv/bin/python scripts/fastsim_eval.py --ckpt "$ck" \
    --model data/fastsim_model.json --map data/vq2_map_winner_train.json \
    --demo-npz data/fastsim_demo_winner.npz --reloc-events \
    --demo-corridor 2.0 --speed-cap 10 \
    --n-envs 256 2>&1 | grep -E "FINISHED|lap time|failures|gate"
done
