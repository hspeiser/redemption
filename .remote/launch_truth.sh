#!/bin/bash
set -e
set -o pipefail
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_ship.tgz
pkill -f fastsim_train_ppo || true
sleep 2
mkdir -p data/fastsim_runs/ppo_truth8
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 2600 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_truth8 \
  --model data/fastsim_model.json --map data/vq2_map_truth.json --obstacles data/vq2_obstacles.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_truthpool.npz \
  --bc-init data/vq2_sac_clean_demo.npz --entropy 3e-3 --reloc-events \
  --demo-corridor 3.0 --speed-cap 8 \
  > data/fastsim_runs/ppo_truth8/stdout.log 2>&1 &
echo "launched round3 pid $!"
sleep 60
tail -3 data/fastsim_runs/ppo_truth8/stdout.log
