#!/bin/bash
set -e
set -o pipefail
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_ship.tgz
pkill -f fastsim_train_ppo || true
sleep 2
.venv/bin/python scripts/fastsim_train_ppo.py --iters 2 --n-envs 512 \
  --horizon 16 --minibatch 2048 --device cuda \
  --run-dir data/fastsim_runs/smoke \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_states.npz
echo "== SMOKE OK, launching long run =="
mkdir -p data/fastsim_runs/ppo_v1
nohup .venv/bin/python scripts/fastsim_train_ppo.py \
  --iters 6000 --n-envs 4096 --horizon 64 --minibatch 16384 \
  --device cuda --run-dir data/fastsim_runs/ppo_v1 \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --rate-sign 1 --demo-npz data/fastsim_demo_states.npz \
  --bc-init data/vq2_sac_clean_demo.npz --entropy 3e-3 \
  > data/fastsim_runs/ppo_v1/stdout.log 2>&1 &
echo "launched pid $!"
sleep 60
tail -6 data/fastsim_runs/ppo_v1/stdout.log
