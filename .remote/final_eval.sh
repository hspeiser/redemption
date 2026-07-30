#!/bin/bash
set -e
cd ~/aigp
pgrep -f fastsim_train_ppo && echo "STILL RUNNING" || echo "fine-tune done"
tail -2 data/fastsim_runs/ppo_reloc/stdout.log | head -1
for tag in "STRUCTURED --reloc-events" "CLEAN "; do
  set -- $tag
  echo "===== fine-tuned latest ($1) ====="
  .venv/bin/python scripts/fastsim_eval.py \
    --ckpt data/fastsim_runs/ppo_reloc/latest.pt \
    --model data/fastsim_model.json --map data/vq2_map_final.json \
    --demo-npz data/fastsim_demo_states.npz $2 \
    --n-envs 1024 2>&1 | grep -E "FINISHED|lap time|failures"
done
