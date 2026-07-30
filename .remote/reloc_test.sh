#!/bin/bash
set -e
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_ship.tgz
echo "== conservative under STRUCTURED reloc noise =="
.venv/bin/python scripts/fastsim_eval.py \
  --ckpt data/fastsim_runs/ppo_v1/finish_100.pt \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --demo-npz data/fastsim_demo_states.npz --reloc-events \
  --n-envs 1024 2>&1 | grep -E "FINISHED|lap time|failures"
echo "== champion under STRUCTURED reloc noise =="
.venv/bin/python scripts/fastsim_eval.py \
  --ckpt data/fastsim_runs/ppo_v1/finish_1600.pt \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --demo-npz data/fastsim_demo_states.npz --reloc-events \
  --n-envs 1024 2>&1 | grep -E "FINISHED|lap time|failures"
