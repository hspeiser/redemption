"""Upright-constrained gate pose solver (fixes the free-PnP rotation ambiguity).

Our free IPPE solver (:func:`redemption.pnp.solve_pnp`) suffers the classic
planar-square two-fold ambiguity: for near-fronto-parallel / far gates the
recovered rotation can flip to the mirror solution, blowing up the rotation
error (our range eval showed rotation p90 exploding while translation stayed
sane).

Premise (following Henry's `upright_pnp.py`): a racing gate is a **vertical,
unrolled square** and the camera's gravity/down direction is observable (IMU).
With "up" pinned in the camera frame the pose is no longer 6-DoF:

  * **4-DoF** (``solve_upright``)      : gate center (3) + heading psi (1)
  * **3-DoF** (``psi_fixed`` supplied) : gate center only, heading from the map

Because "up" cannot flip, the mirror ambiguity is gone. Inputs are pixel
corners (image order TL, TR, BR, BL) and the down direction in the camera frame
``down_cam`` (unit vector; e.g. (0, 1, 0) for a level camera in our +Y-down
convention). The optional free-PnP seed comes from :func:`redemption.pnp.solve_pnp`.
"""

from __future__ import annotations

import math

import numpy as np

# Corner sign pattern for image order TL, TR, BR, BL as (lateral, up).
_SIGNS = np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])


def vertical_basis_camera(down_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (up_c, h1, h2): world-up + a horizontal orthonormal pair, in cam frame."""
    up_c = -np.asarray(down_cam, float)
    up_c = up_c / (np.linalg.norm(up_c) or 1.0)
    # any vector not parallel to up_c, projected to the horizontal plane
    ref = np.array([1.0, 0.0, 0.0]) if abs(up_c[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    h1 = ref - (ref @ up_c) * up_c
    h1 /= (np.linalg.norm(h1) or 1.0)
    h2 = np.cross(up_c, h1)
    return up_c, h1, h2


def corners_camera(center_c, psi, up_c, h1, h2, half):
    lat_c = math.cos(psi) * h1 + math.sin(psi) * h2
    return (np.asarray(center_c, float)[None, :]
            + half * _SIGNS[:, :1] * lat_c[None, :]
            + half * _SIGNS[:, 1:] * up_c[None, :])


def _project(pts_c, K):
    z = np.maximum(pts_c[:, 2], 1e-6)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([fx * pts_c[:, 0] / z + cx, fy * pts_c[:, 1] / z + cy], axis=1)


def _lm(residual_fn, x0, iters=25):
    """Tiny Levenberg-Marquardt with a numerical Jacobian (from Henry's solver)."""
    x = np.array(x0, float)
    lam = 1e-3
    r = residual_fn(x)
    cost = float(r @ r)
    for _ in range(iters):
        jac = np.empty((r.size, x.size))
        for j in range(x.size):
            dx = np.zeros_like(x)
            dx[j] = 1e-5
            jac[:, j] = (residual_fn(x + dx) - r) / 1e-5
        jtj = jac.T @ jac
        g = jac.T @ r
        for _ in range(8):
            try:
                step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x - step
            rn = residual_fn(xn)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost, lam = xn, rn, cn, max(lam * 0.3, 1e-7)
                break
            lam *= 10
        if cost < 1e-10:
            break
    return x, math.sqrt(cost / max(r.size / 2, 1))


def solve_upright(image_pts, down_cam, K, half=0.75, psi_fixed=None,
                  weights=None, seed_rt=None):
    """Upright-constrained gate pose.

    Parameters
    ----------
    image_pts : (4,2) pixel corners, image order TL,TR,BR,BL.
    down_cam  : (3,) gravity/down direction in the CAMERA frame (unit).
    K         : (3,3) intrinsics.
    half      : half the aperture side (0.75 for a 1.5 m gate).
    psi_fixed : if given, solve 3-DoF (center only) with this heading.
    weights   : optional (4,) per-corner weights (e.g. keypoint confidence).
    seed_rt   : optional (R (3,3), t (3,)) free-PnP seed to initialise from.

    Returns
    -------
    dict(center (3,), psi, R (3,3), tvec (3,), rms_px) or None.
    """
    obs = np.asarray(image_pts, float).reshape(4, 2)
    if not np.isfinite(obs).all():
        return None
    up_c, h1, h2 = vertical_basis_camera(down_cam)
    w = np.ones(4) if weights is None else np.sqrt(np.clip(np.asarray(weights, float), 1e-3, None))
    w = np.repeat(w, 2)  # per (u,v) residual

    # --- seed ---
    if seed_rt is not None:
        R0, t0 = seed_rt
        center0 = np.asarray(t0, float).reshape(3)
        lat0 = np.asarray(R0, float)[:, 0]
        psi0 = math.atan2(lat0 @ h2, lat0 @ h1)
    else:
        edge = max(np.linalg.norm(obs[a] - obs[b]) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)))
        if edge < 6:
            return None
        rng0 = 2 * half * K[0, 0] / edge
        cpx = obs.mean(0)
        ray = np.array([(cpx[0] - K[0, 2]) / K[0, 0], (cpx[1] - K[1, 2]) / K[1, 1], 1.0])
        center0 = ray / np.linalg.norm(ray) * rng0
        psi0 = psi_fixed if psi_fixed is not None else 0.0

    if psi_fixed is None:
        def res(x):
            pts = corners_camera(x[:3], x[3], up_c, h1, h2, half)
            return ((_project(pts, K) - obs).ravel()) * w
        if seed_rt is None:
            best = None
            for p0 in np.linspace(-math.pi, math.pi, 8, endpoint=False):
                xc, rc = _lm(res, np.array([*center0, p0]), iters=15)
                if best is None or rc < best[1]:
                    best = (xc, rc)
            x, rms = _lm(res, best[0], iters=10)
        else:
            x, rms = _lm(res, np.array([*center0, psi0]))
        center, psi = x[:3], float(x[3])
    else:
        # resolve the 180-degree convention: at the map heading TL projects left of TR
        trial = _project(corners_camera(center0, psi_fixed, up_c, h1, h2, half), K)
        if trial[0, 0] > trial[1, 0]:
            psi_fixed = psi_fixed + math.pi

        def res(x):
            pts = corners_camera(x, psi_fixed, up_c, h1, h2, half)
            return ((_project(pts, K) - obs).ravel()) * w
        x, rms = _lm(res, center0.copy())
        center, psi = x, float(psi_fixed)

    if not (0.3 <= float(np.linalg.norm(center)) <= 80.0):
        return None
    lat_c = math.cos(psi) * h1 + math.sin(psi) * h2
    R = np.column_stack([lat_c, up_c, np.cross(lat_c, up_c)])
    return {"center": center, "psi": psi, "R": R, "tvec": center, "rms_px": rms}
