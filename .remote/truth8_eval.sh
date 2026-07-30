#!/bin/bash
cd ~/aigp
for ck in data/fastsim_runs/ppo_truth8/finish_100.pt \
          data/fastsim_runs/ppo_truth8/finish_200.pt \
          data/fastsim_runs/ppo_truth8/latest.pt; do
  [ -f "$ck" ] || continue
  echo "===== $ck ====="
  .venv/bin/python scripts/fastsim_eval.py --ckpt "$ck" \
    --model data/fastsim_model.json --map data/vq2_map_truth.json \
    --obstacles data/vq2_obstacles.json \
    --demo-npz data/fastsim_demo_truthpool.npz --reloc-events \
    --demo-corridor 3.0 --speed-cap 8 \
    --n-envs 512 2>&1 | grep -E "FINISHED|lap time|failures"
done
