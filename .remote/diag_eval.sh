#!/bin/bash
set -e
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_ship.tgz
.venv/bin/python scripts/fastsim_eval.py \
  --ckpt data/fastsim_runs/ppo_v1/latest.pt \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --n-envs 512 2>&1 | tail -8
