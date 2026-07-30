#!/bin/bash
cd ~/aigp
for ck in data/fastsim_runs/ppo_round3/finish_100.pt \
          data/fastsim_runs/ppo_round3/finish_300.pt \
          data/fastsim_runs/ppo_round3/latest.pt; do
  [ -f "$ck" ] || continue
  echo "===== $ck ====="
  .venv/bin/python scripts/fastsim_eval.py --ckpt "$ck" \
    --model data/fastsim_model.json --map data/vq2_map_hybrid.json \
    --demo-npz data/fastsim_demo_states.npz --reloc-events \
    --demo-corridor 2.5 --speed-cap 13 \
    --n-envs 512 2>&1 | grep -E "FINISHED|lap time|failures"
done
