#!/bin/bash
set -e
cd ~/aigp
pkill -f "train_net.py --resume" || true
sleep 2
.venv/bin/python - <<'EOF'
import numpy as np
path = "data/labels_posetrack/vq2_close.npz"
d = dict(np.load(path, allow_pickle=False))
# the gate-cls head is VQ1-sized (n_gates=6); class supervision is an
# auxiliary the live stack never uses -- zero it like the v7 VQ2 labels
d["gate_idx"] = np.zeros_like(d["gate_idx"])
np.savez_compressed(path, **d)
print("gate_idx zeroed; rows:", len(d["gate_idx"]))
EOF
nohup .venv/bin/python scripts/train_net.py \
  --resume data/models/gatenet_v7_best.pt \
  --tag v12close \
  --labels-dir data/labels \
  --labels-dir data/labels_posetrack \
  --path-map "C:\\Users\\henry\\Downloads\\AI-GP Simulator v1.0.3379\\ai-grand-prix\\outputs\\captures::/home/henry/aigp/captures" \
  --epochs 12 --lr 1e-4 --workers 8 \
  > data/train_v12close.log 2>&1 &
echo "relaunched pid $!"
sleep 90
grep -v "findDecoder\|WARN" data/train_v12close.log | tail -4
