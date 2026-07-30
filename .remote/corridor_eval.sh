#!/bin/bash
cd ~/aigp
.venv/bin/python scripts/fastsim_eval.py \
  --ckpt data/fastsim_runs/ppo_corridor/latest.pt \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --demo-npz data/fastsim_demo_states.npz --reloc-events \
  --n-envs 512 2>&1 | grep -E "FINISHED|lap time|failures|checkpoint"
