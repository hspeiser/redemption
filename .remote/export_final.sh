#!/bin/bash
set -e
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_probe.tgz 2>/dev/null || true
.venv/bin/python scripts/fastsim_export.py \
  --ckpt data/fastsim_runs/ppo_v1/finish_1600.pt \
  --out /mnt/c/Users/henry/vq2_ppo_champion.pt
.venv/bin/python scripts/fastsim_export.py \
  --ckpt data/fastsim_runs/ppo_v1/finish_100.pt \
  --out /mnt/c/Users/henry/vq2_ppo_conservative.pt
pkill -f fastsim_train_ppo || true
echo "training stopped; artifacts staged"
ls -la /mnt/c/Users/henry/vq2_ppo_*.pt
