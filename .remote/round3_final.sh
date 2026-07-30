#!/bin/bash
cd ~/aigp
pkill -f fastsim_train_ppo || true
sleep 2
best_ck=""; best_n=0
for ck in data/fastsim_runs/ppo_round3/finish_*.pt; do
  [ -f "$ck" ] || continue
  n=$(.venv/bin/python scripts/fastsim_eval.py --ckpt "$ck" \
    --model data/fastsim_model.json --map data/vq2_map_hybrid.json \
    --demo-npz data/fastsim_demo_states.npz --reloc-events \
    --demo-corridor 2.5 --speed-cap 13 \
    --n-envs 512 2>/dev/null | grep -o 'FINISHED FULL COURSE: [0-9]*' | grep -o '[0-9]*$')
  echo "$ck -> $n/512"
  if [ -n "$n" ] && [ "$n" -gt "$best_n" ]; then best_n=$n; best_ck=$ck; fi
done
echo "BEST: $best_ck ($best_n/512)"
.venv/bin/python scripts/fastsim_export.py --ckpt "$best_ck" \
  --out /mnt/c/Users/henry/vq2_ppo_round3.pt
