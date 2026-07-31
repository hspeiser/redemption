#!/bin/bash
set -e
set -o pipefail
cd ~/aigp
tar xzf /mnt/c/Users/henry/labelgen_ship.tgz
mkdir -p data/labels_posetrack
echo "=== label generation (5090) ==="
.venv/bin/python scripts/vq2_labels_posetrack.py \
  --frames-root /mnt/c/Users/henry/aigp_raw/raw_sessions \
  --pose-tracks data/pose_tracks_200049.npz data/pose_tracks_200627.npz \
                data/pose_tracks_212220.npz data/pose_tracks_213340.npz \
  --map data/vq2_runtime_map_g9g15fix.json \
  --primary data/models/gatenet_v7_best.pt \
  --refiner data/models/gatenet_v10strict_ep0.pt \
  --out data/labels_posetrack/vq2_close.npz \
  2>&1 | tail -5
echo "=== v12close finetune ==="
mkdir -p data/fastsim_runs
nohup .venv/bin/python scripts/train_net.py \
  --resume data/models/gatenet_v7_best.pt \
  --tag v12close \
  --labels-dir data/labels \
  --labels-dir data/labels_posetrack \
  --epochs 12 --lr 1e-4 --workers 8 \
  > data/train_v12close.log 2>&1 &
echo "training launched pid $!"
sleep 90
tail -5 data/train_v12close.log
