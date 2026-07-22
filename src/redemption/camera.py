"""Pinhole camera model (no lens distortion).

Convention (OpenCV): camera frame has +X right, +Y down, +Z into the scene; a
point is in front of the camera when ``Z > 0``. Projection::

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DotDict


@dataclass(frozen=True)
class PinholeCamera:
    """Immutable pinhole camera built from ``configs/camera.toml``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray  # shape (5,), all zeros for this project

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, camera_cfg: DotDict) -> "PinholeCamera":
        res = camera_cfg.resolution
        intr = camera_cfg.intrinsics
        return cls(
            width=int(res.width),
            height=int(res.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.cx),
            cy=float(intr.cy),
            dist_coeffs=np.asarray(intr.dist_coeffs, dtype=np.float64).reshape(-1),
        )

    # ---- intrinsics -------------------------------------------------------
    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsics matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan2(self.width / 2.0, self.fx)))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan2(self.height / 2.0, self.fy)))

    # ---- projection -------------------------------------------------------
    def project(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D points given in the CAMERA frame to pixels.

        Parameters
        ----------
        points_cam : (N, 3) array
            Points already expressed in the camera coordinate frame.

        Returns
        -------
        pixels : (N, 2) array of (u, v)
        in_front : (N,) bool array, True where Z > 0 (projectable)
        """
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = pts[:, 2]
        in_front = z > 1e-6
        safe_z = np.where(in_front, z, 1.0)
        u = self.fx * pts[:, 0] / safe_z + self.cx
        v = self.fy * pts[:, 1] / safe_z + self.cy
        pixels = np.stack([u, v], axis=1)
        # Points behind the camera get NaN so callers can't misuse them.
        pixels[~in_front] = np.nan
        return pixels, in_front

    def in_frame(self, pixels: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of pixels inside the image (optionally expanded by margin px)."""
        px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        u, v = px[:, 0], px[:, 1]
        return (
            (u >= -margin)
            & (u < self.width + margin)
            & (v >= -margin)
            & (v < self.height + margin)
        )

    def backproject(self, pixel: np.ndarray, depth: float) -> np.ndarray:
        """Return the camera-frame 3D point at ``pixel`` and the given ``depth`` (Z)."""
        u, v = float(pixel[0]), float(pixel[1])
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth], dtype=np.float64)
