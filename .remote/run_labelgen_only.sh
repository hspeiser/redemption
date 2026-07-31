#!/bin/bash
set -e
cd ~/aigp
mkdir -p data/labels_posetrack
.venv/bin/python scripts/vq2_labels_posetrack.py \
  --frames-root /mnt/c/Users/henry/aigp_raw/raw_sessions \
  --pose-tracks data/pose_tracks_200049.npz data/pose_tracks_200627.npz \
                data/pose_tracks_212220.npz data/pose_tracks_213340.npz \
  --map data/vq2_runtime_map_g9g15fix.json \
  --primary data/models/gatenet_v7_best.pt \
  --refiner data/models/gatenet_v10strict_ep0.pt \
  --out data/labels_posetrack/vq2_close.npz 2>&1 | tail -4
