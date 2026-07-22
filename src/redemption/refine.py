"""Sub-pixel corner refinement for detected gate corners.

The pose model localizes the 4 inner corners to ~2 px; snapping each predicted
corner to the true image corner (frame/aperture edge intersection) with
``cv2.cornerSubPix`` tightens that toward sub-pixel, which directly improves the
recovered pose. Same primitive Henry uses when auto-labeling; here it runs at
inference time on the model's predictions.
"""

from __future__ import annotations

import cv2
import numpy as np

_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03)


def refine_corners(image: np.ndarray, corners_px: np.ndarray, win: int = 4,
                   max_shift: float = 8.0) -> np.ndarray:
    """Return sub-pixel-refined (4,2) corners.

    ``win`` is the half-size of the cornerSubPix search window. Any corner the
    refinement drags more than ``max_shift`` px is reverted to its original
    prediction (guards against snapping to a nearby stronger corner, e.g. gate
    graphics/text).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    orig = np.asarray(corners_px, np.float64).reshape(-1, 2)
    pts = orig.astype(np.float32).reshape(-1, 1, 2)
    try:
        cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), _CRIT)
    except cv2.error:
        return orig
    out = pts.reshape(-1, 2).astype(np.float64)
    d = np.linalg.norm(out - orig, axis=1)
    out[d > max_shift] = orig[d > max_shift]
    return out
