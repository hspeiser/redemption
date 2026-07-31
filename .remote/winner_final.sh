#!/bin/bash
cd ~/aigp
pkill -f fastsim_train_ppo || true
sleep 2
best_ck=""; best_n=0
for ck in data/fastsim_runs/ppo_winner/finish_*.pt data/fastsim_runs/ppo_winner/latest.pt; do
  [ -f "$ck" ] || continue
  n=$(.venv/bin/python scripts/fastsim_eval.py --ckpt "$ck" \
    --model data/fastsim_model.json --map data/vq2_map_winner_train.json \
    --demo-npz data/fastsim_demo_winner.npz --reloc-events \
    --demo-corridor 2.0 --speed-cap 10 \
    --n-envs 768 2>/dev/null | grep -o 'FINISHED FULL COURSE: [0-9]*' | grep -o '[0-9]*$')
  echo "$ck -> $n/768"
  if [ -n "$n" ] && [ "$n" -gt "$best_n" ]; then best_n=$n; best_ck=$ck; fi
done
echo "BEST: $best_ck ($best_n/768)"
.venv/bin/python scripts/fastsim_export.py --ckpt "$best_ck" \
  --out /mnt/c/Users/henry/vq2_ppo_winner.pt
