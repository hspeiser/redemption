#!/bin/bash
set -euo pipefail

cd /home/henry/aigp
PY=/home/henry/aigp/.venv/bin/python

"$PY" scripts/train_crop_gatenet.py \
  --epochs 18 \
  --batch 96 \
  --workers 12 \
  --lr 1.5e-4 \
  --negative-ratio 0.20 \
  --init data/models/gatenet_v7_best.pt \
  --tag v11crop \
  --labels-dir data/labels_v10_vq1 \
  --labels-dir data/labels_v10_vq2 \
  --labels-dir data/labels_v10_vq2t \
  --path-map 'C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379\ai-grand-prix\outputs\captures::/home/henry/aigp/captures' \
  > data/models/train_v11crop.log \
  2> data/models/train_v11crop.err
