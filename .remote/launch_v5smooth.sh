#!/bin/bash
set -e
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_v5_ship.tgz
pkill -f fastsim_train_ppo || true
sleep 2
mkdir -p data/fastsim_runs/ppo_v5smooth
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 600 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_v5smooth \
  --model data/fastsim_model_v2.json --map data/vq2_map_winner_train.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_winner.npz \
  --bc-init "data/vq2_sac_runs/gate3_nstep_v60b/20260730_161745/episode_0002.npz" \
  --entropy 3e-3 --reloc-events --noise-era 10hz --fov-vision \
  --action-smoothness 0.12 --act-delay-min 1 \
  --demo-corridor 1.5 --speed-cap 8 \
  > data/fastsim_runs/ppo_v5smooth/stdout.log 2>&1 &
echo "launched v5smooth pid $!"
sleep 75
tail -2 data/fastsim_runs/ppo_v5smooth/stdout.log
