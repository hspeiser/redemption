"""Thin inference wrapper around an Ultralytics YOLO-Pose model.

Turns raw ``Results`` into a clean list of :class:`Detection` objects carrying
the 4 predicted inner corners (pixel coords) with per-keypoint confidences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    box_conf: float
    kpts_px: np.ndarray   # (4, 2) pixel coordinates
    kpt_conf: np.ndarray  # (4,) per-keypoint confidence

    @property
    def center_px(self) -> np.ndarray:
        return self.kpts_px.mean(axis=0)


def load_model(weights: str):
    """Load a YOLO-Pose model (import kept local so config-only tools stay light)."""
    from ultralytics import YOLO

    return YOLO(weights)


def infer_image(model, image, conf: float = 0.25, imgsz: int = 640,
                device: int | str = 0, verbose: bool = False) -> list[Detection]:
    """Run the model on one image (path or BGR array) -> list of detections."""
    results = model.predict(source=image, conf=conf, imgsz=imgsz, device=device, verbose=verbose)
    if not results:
        return []
    res = results[0]
    dets: list[Detection] = []
    if res.keypoints is None or res.boxes is None:
        return dets

    kdata = res.keypoints.data.cpu().numpy()  # (n, K, 3) -> x, y, conf
    box_conf = res.boxes.conf.cpu().numpy()   # (n,)
    for i in range(kdata.shape[0]):
        kpts = kdata[i]
        dets.append(Detection(
            box_conf=float(box_conf[i]),
            kpts_px=kpts[:, :2].astype(np.float64),
            kpt_conf=kpts[:, 2].astype(np.float64),
        ))
    return dets
