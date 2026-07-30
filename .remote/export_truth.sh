#!/bin/bash
set -e
cd ~/aigp
pkill -f fastsim_train_ppo || true
sleep 2
.venv/bin/python scripts/fastsim_export.py \
  --ckpt data/fastsim_runs/ppo_truth/finish_300.pt \
  --out /mnt/c/Users/henry/vq2_ppo_truth300.pt
echo exported
