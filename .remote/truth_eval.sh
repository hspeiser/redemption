#!/bin/bash
cd ~/aigp
.venv/bin/python scripts/fastsim_eval.py \
  --ckpt data/fastsim_runs/ppo_truth/latest.pt \
  --model data/fastsim_model.json --map data/vq2_map_truth.json \
  --obstacles data/vq2_obstacles.json \
  --demo-npz data/fastsim_demo_truthpool.npz --reloc-events \
  --demo-corridor 3.0 --speed-cap 12 \
  --n-envs 512 2>&1 | grep -E "FINISHED|lap time|failures|checkpoint"
