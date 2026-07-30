#!/bin/bash
set -e
cd ~/aigp
mkdir -p data/fastsim_runs/ppo_reloc
cp data/fastsim_runs/ppo_v1/finish_1600.pt data/fastsim_runs/ppo_reloc/seed.pt
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 2400 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_reloc \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_states.npz \
  --reloc-events --lr 1e-4 \
  --resume data/fastsim_runs/ppo_reloc/seed.pt \
  > data/fastsim_runs/ppo_reloc/stdout.log 2>&1 &
echo "launched reloc fine-tune pid $!"
sleep 40
tail -3 data/fastsim_runs/ppo_reloc/stdout.log
