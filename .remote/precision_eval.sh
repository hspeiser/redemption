#!/bin/bash
cd ~/aigp
.venv/bin/python scripts/fastsim_eval.py \
  --ckpt data/fastsim_runs/ppo_precision/latest.pt \
  --model data/fastsim_model.json --map data/vq2_map_belief.json \
  --demo-npz data/fastsim_demo_states.npz --reloc-events \
  --demo-corridor 2.5 --speed-cap 13 \
  --n-envs 512 2>&1 | grep -E "FINISHED|lap time|failures|checkpoint"
