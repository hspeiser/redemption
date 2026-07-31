#!/bin/bash
set -e
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_v3_ship.tgz
pkill -f fastsim_train_ppo || true
sleep 2
mkdir -p data/fastsim_runs/ppo_v3fov
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 600 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_v3fov \
  --model data/fastsim_model_v2.json --map data/vq2_map_winner_train.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_winner.npz \
  --bc-init "data/vq2_sac_runs/gate3_nstep_v60b/20260730_161745/episode_0002.npz" \
  --entropy 3e-3 --reloc-events --noise-era 10hz --fov-vision \
  --demo-corridor 2.0 --speed-cap 12 \
  > data/fastsim_runs/ppo_v3fov/stdout.log 2>&1 &
echo "launched v3fov pid $!"
sleep 75
tail -3 data/fastsim_runs/ppo_v3fov/stdout.log
