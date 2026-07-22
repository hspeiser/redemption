"""Pose sampling and 3D<->2D geometry helpers.

A gate pose maps object points to the camera frame via ``X_cam = R @ X_obj + t``
where ``R`` is built from pitch (rotation about camera +X) and yaw (about camera
+Y); roll is fixed at 0 per the spec. ``t`` is the gate center in the camera
frame, placed by back-projecting a randomly chosen image pixel to a random
depth so that image-plane coverage is directly controllable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import PinholeCamera
from .config import DotDict
from .gate import Gate


def rotation_pitch_yaw(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Rotation matrix ``R = Ry(yaw) @ Rx(pitch)`` (roll = 0)."""
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)
    cx, sx = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return Ry @ Rx


@dataclass
class GatePose:
    """A sampled gate pose plus cached rotation/translation."""

    pitch_deg: float
    yaw_deg: float
    center: np.ndarray  # (3,) camera-frame translation t
    R: np.ndarray       # (3, 3)

    @property
    def t(self) -> np.ndarray:
        return self.center

    @property
    def depth(self) -> float:
        return float(self.center[2])

    def transform(self, points_obj: np.ndarray) -> np.ndarray:
        """Object-frame points (N,3) -> camera-frame points (N,3)."""
        pts = np.asarray(points_obj, dtype=np.float64).reshape(-1, 3)
        return (self.R @ pts.T).T + self.center

    def rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth (rvec, tvec) as OpenCV would return them."""
        rvec, _ = cv2.Rodrigues(self.R)
        return rvec.reshape(3), self.center.reshape(3)


def sample_pose(camera: PinholeCamera, pose_cfg: DotDict, rng: np.random.Generator) -> GatePose:
    """Sample one random :class:`GatePose` from the datagen ``[pose]`` config."""
    # --- depth ---
    dmin, dmax = float(pose_cfg.depth_min), float(pose_cfg.depth_max)
    if str(pose_cfg.depth_sampling).lower() == "log_uniform":
        depth = float(np.exp(rng.uniform(np.log(dmin), np.log(dmax))))
    else:
        depth = float(rng.uniform(dmin, dmax))

    # --- center placement via back-projected target pixel ---
    m = float(pose_cfg.center_pixel_margin)
    u = rng.uniform(-m * camera.width, camera.width * (1.0 + m))
    v = rng.uniform(-m * camera.height, camera.height * (1.0 + m))
    center = camera.backproject(np.array([u, v]), depth)

    # --- orientation ---
    pitch = float(rng.uniform(pose_cfg.pitch_min, pose_cfg.pitch_max))
    yaw = float(rng.uniform(pose_cfg.yaw_min, pose_cfg.yaw_max))
    R = rotation_pitch_yaw(pitch, yaw)

    return GatePose(pitch_deg=pitch, yaw_deg=yaw, center=center, R=R)


def project_corners(
    camera: PinholeCamera, gate: Gate, pose: GatePose, which: str = "inner"
) -> tuple[np.ndarray, np.ndarray]:
    """Project a gate's corners to pixels.

    Returns ``(pixels (4,2), in_front (4,))``. ``which`` is ``"inner"`` or ``"outer"``.
    """
    corners_obj = gate.inner_corners() if which == "inner" else gate.outer_corners()
    corners_cam = pose.transform(corners_obj)
    return camera.project(corners_cam)


def polygon_contains(polygon_px: np.ndarray, point_px: np.ndarray) -> bool:
    """True if ``point_px`` lies inside (or on) the closed polygon ``polygon_px``."""
    poly = np.asarray(polygon_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(poly, (float(point_px[0]), float(point_px[1])), False) >= 0
