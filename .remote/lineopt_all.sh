#!/bin/bash
# Line-optimizer CEM: three configs sequentially on the 5090.
cd ~/aigp
mkdir -p data/lineopt
run() {
  cap=$1; clr=$2; tag=$3
  echo "=== $tag (cap $cap, clearance $clr) ==="
  .venv/bin/python scripts/fastsim_line_opt.py \
    --speed-cap "$cap" --clearance "$clr" \
    --n-envs 256 --pop 32 --elite 8 --iters 14 --device cuda \
    --out-prefix "data/lineopt/$tag" > "data/lineopt/$tag.log" 2>&1
  echo "=== $tag done rc=$? ==="
}
run 8 0.25 r1_cap8
run 10 0.15 a1_cap10
run 12 0.15 a2_cap12
echo ALLDONE
