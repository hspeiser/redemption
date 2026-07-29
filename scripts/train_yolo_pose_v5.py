"""Fine-tune a YOLO gate-pose checkpoint for precise VQ2 corners.

This is the high-precision domain-adaptation stage used after building a
dataset with ``build_yolo_pose_dataset.py``. The defaults deliberately use
less destructive geometry augmentation than the broad-coverage Lars V4
training run: we already have range/viewpoint diversity and now care about
sub-pixel corner placement and stable PnP depth.

Example
-------

    python scripts\\train_yolo_pose_v5.py ^
      --data dataset_v5vq2\\gates.yaml ^
      --resume runs\\gatepose_v4\\weights\\best.pt ^
      --project runs --name gatepose_v5vq2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", default="gatepose_v5vq2")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    model = YOLO(str(args.resume.resolve()))
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
        patience=7,
        optimizer="AdamW",
        lr0=args.lr,
        lrf=0.08,
        warmup_epochs=1.0,
        weight_decay=5e-4,
        cos_lr=True,
        deterministic=True,
        seed=7,
        # Preserve exact projective geometry while retaining enough
        # photometric/range diversity to generalize between VQ2 runs.
        degrees=3.0,
        translate=0.03,
        scale=0.20,
        shear=0.5,
        perspective=0.0001,
        fliplr=0.5,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        hsv_h=0.008,
        hsv_s=0.35,
        hsv_v=0.35,
        pose=30.0,
        kobj=1.5,
        plots=False,
        save=True,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
