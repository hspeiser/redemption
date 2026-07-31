#!/bin/bash
set -e
cd ~/aigp
.venv/bin/python - <<'EOF'
import numpy as np
d = np.load("data/labels_posetrack/vq2_close.npz")
print("label npz:", {k: d[k].shape for k in d.files})
EOF
pkill -f "train_net.py --resume" || true
sleep 2
nohup .venv/bin/python scripts/train_net.py \
  --resume data/models/gatenet_v7_best.pt \
  --tag v12close \
  --labels-dir data/labels \
  --labels-dir data/labels_posetrack \
  --path-map "C:\\Users\\henry\\Downloads\\AI-GP Simulator v1.0.3379\\ai-grand-prix\\outputs\\captures::/home/henry/aigp/captures" \
  --epochs 12 --lr 1e-4 --workers 8 \
  > data/train_v12close.log 2>&1 &
echo "v12close training launched pid $!"
sleep 60
tail -3 data/train_v12close.log
