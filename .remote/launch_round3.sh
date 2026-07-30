#!/bin/bash
set -e
set -o pipefail
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_ship.tgz
pkill -f fastsim_train_ppo || true
sleep 2
mkdir -p data/fastsim_runs/ppo_round3
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 2600 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_round3 \
  --model data/fastsim_model.json --map data/vq2_map_hybrid.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_states.npz \
  --bc-init data/vq2_sac_clean_demo.npz --entropy 3e-3 --reloc-events \
  --demo-corridor 2.5 --speed-cap 13 \
  > data/fastsim_runs/ppo_round3/stdout.log 2>&1 &
echo "launched round3 pid $!"
sleep 60
tail -3 data/fastsim_runs/ppo_round3/stdout.log
