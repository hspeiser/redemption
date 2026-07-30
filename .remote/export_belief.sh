#!/bin/bash
set -e
cd ~/aigp
.venv/bin/python scripts/fastsim_export.py \
  --ckpt data/fastsim_runs/ppo_precision/finish_200.pt \
  --out /mnt/c/Users/henry/vq2_ppo_belief200.pt
.venv/bin/python scripts/fastsim_export.py \
  --ckpt data/fastsim_runs/ppo_precision/finish_100.pt \
  --out /mnt/c/Users/henry/vq2_ppo_belief100.pt
pkill -f fastsim_train_ppo || true
echo "exported; training stopped"
