"""Gate geometry: the 3D model of the orange square gate.

All coordinates are in METERS, in the gate's own object frame (centered at the
origin, lying on the plane ``Z = 0`` with +Z as the forward normal). The frame
convention (+X right, +Y down) matches the camera frame so that a recovered PnP
pose ``(R, t)`` maps object points to the camera frame directly:

    X_cam = R @ X_obj + t

Corner ordering is SEMANTIC and fixed (tied to the physical corner, not the
view), which is what YOLO-Pose keypoints and PnP correspondences both require::

    0 = TL (top-left)     1 = TR (top-right)
    3 = BL (bottom-left)  2 = BR (bottom-right)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DotDict

# Keypoint names in index order -- also used for report labels and data.yaml.
CORNER_NAMES = ("TL", "TR", "BR", "BL")

# Unit corner layout (before scaling by half-size). Order: TL, TR, BR, BL.
# +X right, +Y down.
_UNIT_CORNERS = np.array(
    [
        [-1.0, -1.0],  # TL
        [+1.0, -1.0],  # TR
        [+1.0, +1.0],  # BR
        [-1.0, +1.0],  # BL
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Gate:
    """Gate geometry from ``configs/gate.toml`` (all sizes stored in meters)."""

    outer_w: float
    outer_h: float
    inner_w: float
    inner_h: float
    depth: float
    color_bgr: tuple[int, int, int]
    side_face_darkness: float

    @classmethod
    def from_config(cls, gate_cfg: DotDict) -> "Gate":
        d = gate_cfg.dimensions_mm
        ap = gate_cfg.appearance
        return cls(
            outer_w=float(d.outer_width) / 1000.0,
            outer_h=float(d.outer_height) / 1000.0,
            inner_w=float(d.inner_width) / 1000.0,
            inner_h=float(d.inner_height) / 1000.0,
            depth=float(d.depth) / 1000.0,
            color_bgr=tuple(int(c) for c in ap.color_bgr),
            side_face_darkness=float(ap.side_face_darkness),
        )

    # ---- 3D corner sets (object frame, meters) ----------------------------
    def inner_corners(self, z: float = 0.0) -> np.ndarray:
        """(4, 3) inner-opening corners -- the KEYPOINTS and PnP object points."""
        xy = _UNIT_CORNERS * np.array([self.inner_w / 2.0, self.inner_h / 2.0])
        return np.column_stack([xy, np.full(len(xy), z)])

    def outer_corners(self, z: float = 0.0) -> np.ndarray:
        """(4, 3) outer-boundary corners (used for the frame polygon & bbox)."""
        xy = _UNIT_CORNERS * np.array([self.outer_w / 2.0, self.outer_h / 2.0])
        return np.column_stack([xy, np.full(len(xy), z)])

    @property
    def object_points(self) -> np.ndarray:
        """PnP object points = the 4 inner corners on the front face (Z=0)."""
        return self.inner_corners(z=0.0)

    @property
    def inner_side(self) -> float:
        """Side length of the (square) inner opening, meters -- for IPPE_SQUARE."""
        return float(self.inner_w)

    @property
    def n_keypoints(self) -> int:
        return 4
