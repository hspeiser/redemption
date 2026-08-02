#!/bin/bash
set -e
cd ~/aigp
cp /mnt/c/Users/henry/vq2_obstacles_inflated.json data/vq2_obstacles_inflated.json
pkill -f fastsim_train_ppo || true
sleep 2
mkdir -p data/fastsim_runs/ppo_v8c
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 600 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_v8c \
  --model data/fastsim_model_v2.json --map data/vq2_map_winner_train.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_winner.npz \
  --bc-init "data/vq2_sac_runs/gate3_nstep_v60b/20260730_161745/episode_0002.npz" \
  --entropy 3e-3 --reloc-events --noise-era 10hz --fov-vision \
  --action-smoothness 0.06 --act-delay-min 1 \
  --obstacles data/vq2_obstacles_inflated.json \
  --demo-corridor 1.5 --speed-cap 8 \
  > data/fastsim_runs/ppo_v8c/stdout.log 2>&1 &
echo "launched v6obs pid $!"
sleep 75
tail -2 data/fastsim_runs/ppo_v8c/stdout.log
