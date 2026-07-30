#!/bin/bash
set -e
set -o pipefail
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_ship.tgz
pkill -f fastsim_train_ppo || true
sleep 2
mkdir -p data/fastsim_runs/ppo_winner
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 2000 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_winner \
  --model data/fastsim_model.json --map data/vq2_map_winner_train.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_winner.npz \
  --bc-init "data/vq2_sac_runs/gate3_nstep_v60b/20260730_161745/episode_0002.npz" \
  --entropy 3e-3 --reloc-events \
  --demo-corridor 2.0 --speed-cap 10 \
  > data/fastsim_runs/ppo_winner/stdout.log 2>&1 &
echo "launched winner-frame run pid $!"
sleep 60
tail -3 data/fastsim_runs/ppo_winner/stdout.log
