#!/bin/bash
set -e
cd ~/aigp
pkill -f "train_net.py --resume" || true
sleep 2
.venv/bin/python - <<'EOF'
import numpy as np
path = "data/labels_posetrack/vq2_droughts.npz"
d = dict(np.load(path, allow_pickle=False))
d["gate_idx"] = np.zeros_like(d["gate_idx"])
np.savez_compressed(path, **d)
print("droughts npz rows:", len(d["gate_idx"]))
EOF
nohup .venv/bin/python scripts/train_net.py \
  --resume data/models/gatenet_v7_best.pt \
  --tag v13drought \
  --labels-dir data/labels \
  --labels-dir data/labels_posetrack \
  --path-map "C:\\Users\\henry\\Downloads\\AI-GP Simulator v1.0.3379\\ai-grand-prix\\outputs\\captures::/home/henry/aigp/captures" \
  --epochs 6 --lr 1e-4 --workers 8 \
  > data/train_v13drought.log 2>&1 &
echo "v13drought training launched pid $!"
sleep 60
grep -v "findDecoder\|WARN" data/train_v13drought.log | tail -3
