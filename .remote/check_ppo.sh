#!/bin/bash
cd ~/aigp
tar xzf /mnt/c/Users/henry/fastsim_ship.tgz scripts/fastsim_eval.py 2>/dev/null || true
echo "== proc =="
pgrep -af fastsim_train_ppo | head -2
echo "== last log =="
tail -3 data/fastsim_runs/ppo_v1/train_log.jsonl 2>/dev/null
echo "== eval latest checkpoint (spawn starts, noise+DR on) =="
.venv/bin/python scripts/fastsim_eval.py \
  --ckpt data/fastsim_runs/ppo_v1/latest.pt \
  --model data/fastsim_model.json --map data/vq2_map_final.json \
  --n-envs 1024 2>&1 | tail -8
