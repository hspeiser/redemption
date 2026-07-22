"""Overlay + montage helpers for qualitative inspection.

Draws predicted vs ground-truth inner corners on images and tiles them into a
montage for the training-progress report.
"""

from __future__ import annotations

import cv2
import numpy as np

from .gate import CORNER_NAMES

# BGR colours
_GT_COLOR = (0, 255, 0)      # green = ground truth
_PRED_COLOR = (0, 128, 255)  # orange = prediction
_CORNER_COLORS = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]


def draw_corners(img: np.ndarray, corners_px: np.ndarray, color, radius: int = 4,
                 label: bool = False, thickness: int = 2) -> None:
    """Draw an ordered corner set + connecting quad in place."""
    pts = np.asarray(corners_px, float)
    poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=1)
    for i, (x, y) in enumerate(pts):
        c = _CORNER_COLORS[i % 4] if label else color
        cv2.circle(img, (int(round(x)), int(round(y))), radius, c, thickness)
        if label:
            cv2.putText(img, CORNER_NAMES[i], (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA)


def overlay_pred_gt(img: np.ndarray, preds: list[np.ndarray], gts: list[np.ndarray]) -> np.ndarray:
    """Return a copy of ``img`` with GT (green) and predicted (orange) corners."""
    out = img.copy()
    for g in gts:
        draw_corners(out, g, _GT_COLOR, radius=5)
    for p in preds:
        draw_corners(out, p, _PRED_COLOR, radius=3)
    return out


def montage(images: list[np.ndarray], cols: int = 4, pad: int = 4,
            bg: int = 30) -> np.ndarray:
    """Tile images into a grid montage (all resized to the first image's size)."""
    if not images:
        return np.zeros((10, 10, 3), np.uint8)
    h, w = images[0].shape[:2]
    tiles = [cv2.resize(im, (w, h)) for im in images]
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = tile
    return canvas
