"""Validate the upright-constrained PnP vs free IPPE (rotation-ambiguity fix).

Monte-Carlo: upright 1.5 m gates at random range/heading under varied camera
attitude, projected + pixel noise, solved three ways, compared to ground truth.
Reproduces Henry's finding that pinning "up" from the IMU collapses the free
solver's rotation p90 blow-up. Config-free study.

Run:  uv run python scripts/validate_upright_pnp.py
"""

from __future__ import annotations

import math
import numpy as np

from redemption.config import load_toml
from redemption.camera import PinholeCamera
from redemption.pnp import solve_pnp
from redemption.upright import solve_upright, vertical_basis_camera, corners_camera
from redemption.metrics import rotation_error_deg, translation_error

HALF = 0.75
OBJ = np.array([[-HALF, HALF, 0.0], [HALF, HALF, 0.0],
                [HALF, -HALF, 0.0], [-HALF, -HALF, 0.0]])  # TL,TR,BR,BL (+Y up)


def project(pts_c, K):
    z = np.maximum(pts_c[:, 2], 1e-6)
    return np.stack([K[0, 0] * pts_c[:, 0] / z + K[0, 2],
                     K[1, 1] * pts_c[:, 1] / z + K[1, 2]], axis=1)


def run(n=2500, noises=(0.5, 1.0, 2.0), seed=7):
    cam = PinholeCamera.from_config(load_toml("camera.toml"))
    K, dist = cam.K, cam.dist_coeffs
    W, H = cam.width, cam.height
    rs = np.random.RandomState(seed)

    cases = []
    while len(cases) < n:
        # camera attitude: down direction tilted from (0,1,0) by roll/pitch
        roll = math.radians(rs.uniform(-25, 25))
        pitch = math.radians(rs.uniform(-20, 20))
        down_cam = np.array([math.sin(roll), math.cos(roll) * math.cos(pitch),
                             math.cos(roll) * math.sin(pitch)])
        down_cam /= np.linalg.norm(down_cam)
        up_c, h1, h2 = vertical_basis_camera(down_cam)
        u, v = rs.uniform(60, W - 60), rs.uniform(40, H - 40)
        rng = rs.uniform(2.5, 40.0)
        ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        center = ray / np.linalg.norm(ray) * rng
        psi = rs.uniform(-math.pi, math.pi)
        gt = corners_camera(center, psi, up_c, h1, h2, HALF)
        if (gt[:, 2] < 0.5).any():
            continue
        px = project(gt, K)
        if ((px[:, 0] < 2) | (px[:, 0] > W - 2) | (px[:, 1] < 2) | (px[:, 1] > H - 2)).any():
            continue
        lat_c = math.cos(psi) * h1 + math.sin(psi) * h2
        R_gt = np.column_stack([lat_c, up_c, np.cross(lat_c, up_c)])
        cases.append((px, center, R_gt, down_cam, psi))

    def q(a, f):
        return round(float(np.percentile(a, f)), 3) if len(a) else None

    for s in noises:
        rows = {"free": {"rot": [], "trans": []},
                "up4": {"rot": [], "trans": []},
                "up3": {"rot": [], "trans": []}}
        for px, center, R_gt, down_cam, psi in cases:
            obs = px + rs.randn(4, 2) * s
            f = solve_pnp(OBJ, obs, K, dist, "IPPE", True, False, np.ones(4))
            if f is not None:
                rows["free"]["rot"].append(rotation_error_deg(f["R"], R_gt))
                rows["free"]["trans"].append(translation_error(f["tvec"], center))
            u4 = solve_upright(obs, down_cam, K, HALF)
            if u4 is not None:
                rows["up4"]["rot"].append(rotation_error_deg(u4["R"], R_gt))
                rows["up4"]["trans"].append(translation_error(u4["tvec"], center))
            u3 = solve_upright(obs, down_cam, K, HALF, psi_fixed=psi)
            if u3 is not None:
                rows["up3"]["rot"].append(rotation_error_deg(u3["R"], R_gt))
                rows["up3"]["trans"].append(translation_error(u3["tvec"], center))
        print(f"\n=== pixel noise {s}px  (n={len(cases)}) ===")
        print(f"{'solver':>6} | {'rotMed':>7} {'rotP90':>7} (deg) | {'transMed':>8} {'transP90':>8} (m)")
        for m in ("free", "up4", "up3"):
            rot, tr = np.array(rows[m]["rot"]), np.array(rows[m]["trans"])
            print(f"{m:>6} | {q(rot,50):>7} {q(rot,90):>7}       | {q(tr,50):>8} {q(tr,90):>8}")


if __name__ == "__main__":
    run()
