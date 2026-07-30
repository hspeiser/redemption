#!/bin/bash
set -e
cd ~/aigp
echo "== checkpoints =="
ls data/fastsim_runs/ppo_v1/*.pt
for ck in data/fastsim_runs/ppo_v1/finish_*.pt data/fastsim_runs/ppo_v1/latest.pt; do
  [ -f "$ck" ] || continue
  echo "===== $ck ====="
  .venv/bin/python scripts/fastsim_eval.py --ckpt "$ck" \
    --model data/fastsim_model.json --map data/vq2_map_final.json \
    --demo-npz data/fastsim_demo_states.npz \
    --n-envs 1024 2>&1 | grep -E "FINISHED|lap time|checkpoint"
done
