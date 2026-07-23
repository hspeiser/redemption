"""Gate pose with an honest, anisotropic covariance (the depth-tail fix).

The EKF hand-off shows the vision limit is the *monocular depth ambiguity of a
single planar gate*: a good 12 cm median but a 7 m p90 tail, and confidence that
does NOT correlate with that error. The tail is geometrically predictable -- near
fronto-parallel / small / overflowing gates have a badly-conditioned range -- so
we can report it analytically without any retraining or higher resolution.

This module wires the runtime path the estimator asked for:

    model corners -> sub-pixel refine -> overflow flag + calibrated per-corner sigma
        -> gravity-constrained (upright) PnP -> pose + 6-DoF covariance (huge along depth)

The covariance is first-order error propagation through the PnP Jacobian:
    Cov_pose = (J^T W J)^-1 ,  J = d(reprojection)/d(pose params),  W = diag(1/sigma_px^2)
which automatically blows up along depth exactly when the view can't constrain it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .refine import refine_corners
from .upright import (corners_camera, corners_camera_model, model_offsets_8,
                      solve_upright, solve_upright8, vertical_basis_camera)


def _project(pts_c, K):
    z = np.maximum(pts_c[:, 2], 1e-6)
    return np.stack([K[0, 0] * pts_c[:, 0] / z + K[0, 2],
                     K[1, 1] * pts_c[:, 1] / z + K[1, 2]], axis=1)


# --------------------------------------------------------------------------
# Calibrated per-corner pixel sigma  (the input noise for the covariance)
# --------------------------------------------------------------------------
# Fit on the real val set (predicted-vs-labeled corner error binned by apparent
# gate size); see scripts/measure_vision.py. Corner error is ~1-1.5 px in the
# mid/far band, rises as the gate fills the frame, and explodes once it overflows.
SIGMA_BASE = 1.3          # px, well-conditioned mid/far detections (post sub-pixel)
SIGMA_NEAR_K = 90.0       # px*px : growth as the gate gets large in frame
SIGMA_OVERFLOW = 18.0     # px, any corner at/over the image border
OVERFLOW_FRAC = 0.70      # gate subtending > this fraction of the frame => untrusted


def corner_sigma(apparent_px: float, overflow: bool) -> float:
    """Calibrated 1-sigma corner pixel error for a detection (scalar, all 4 corners)."""
    if overflow:
        return SIGMA_OVERFLOW
    # grows mildly as the gate gets big (oblique/close, before true overflow)
    return float(SIGMA_BASE + SIGMA_NEAR_K / max(apparent_px, 1.0)) if apparent_px < 40 \
        else SIGMA_BASE + 0.02 * apparent_px


def apparent_size_px(corners_px: np.ndarray) -> float:
    """Mean edge length of the (ordered) inner-corner quad, in pixels."""
    c = np.asarray(corners_px, float)
    e = [np.linalg.norm(c[a] - c[b]) for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))]
    return float(np.mean(e))


def parallax_score(corners_px: np.ndarray) -> float:
    """Perspective foreshortening of the quad in [0, 1]. ~0 = fronto-parallel.

    Opposite edges of a square differ in projected length under perspective (the
    nearer edge is longer). Near-fronto-parallel views have almost equal opposite
    edges -> low score -> DEPTH is ill-conditioned (the wrong-basin risk the EKF
    flagged). Complements the analytic covariance with a cheap pre-solve signal.
    """
    c = np.asarray(corners_px, float)  # TL,TR,BR,BL
    top, bot = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[3])
    left, right = np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1])
    fh = abs(top - bot) / max(top, bot, 1e-6)      # vertical-tilt foreshortening
    fv = abs(left - right) / max(left, right, 1e-6)  # horizontal-tilt foreshortening
    return float(max(fh, fv))


def is_overflow(corners_px: np.ndarray, W: int, H: int, margin: float = 1.5) -> bool:
    """True if any corner is at/over the frame border or the gate ~fills the frame."""
    c = np.asarray(corners_px, float)
    if (c[:, 0] < margin).any() or (c[:, 0] > W - margin).any() \
            or (c[:, 1] < margin).any() or (c[:, 1] > H - margin).any():
        return True
    span = max(np.ptp(c[:, 0]), np.ptp(c[:, 1]))
    return span > OVERFLOW_FRAC * max(W, H)


# --------------------------------------------------------------------------
# Analytic pose covariance
# --------------------------------------------------------------------------
def pose_covariance(center: np.ndarray, psi: float, down_cam: np.ndarray,
                    K: np.ndarray, half: float, sigma_px) -> np.ndarray:
    """4x4 covariance of the upright pose params [cx, cy, cz, psi] (camera frame).

    ``sigma_px`` is a scalar or (4,) per-corner 1-sigma pixel noise. Returned
    order: indices 0..2 = gate-center position (m), index 3 = heading psi (rad).
    """
    up_c, h1, h2 = vertical_basis_camera(down_cam)
    x0 = np.array([center[0], center[1], center[2], psi], float)

    def g(x):
        pts = corners_camera(x[:3], x[3], up_c, h1, h2, half)
        return _project(pts, K).ravel()  # (8,)

    r0 = g(x0)
    J = np.empty((r0.size, 4))
    for j in range(4):
        step = 1e-4 if j < 3 else 1e-4
        dx = np.zeros(4)
        dx[j] = step
        J[:, j] = (g(x0 + dx) - r0) / step

    sig = np.full(4, float(sigma_px)) if np.isscalar(sigma_px) else np.asarray(sigma_px, float)
    var = np.repeat(sig ** 2, 2)                      # per residual element
    A = J.T @ (J / var[:, None])                      # J^T W J
    try:
        cov = np.linalg.inv(A + 1e-9 * np.eye(4))
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.inf)
    return cov


def pose_covariance8(center: np.ndarray, psi: float, down_cam: np.ndarray,
                     K: np.ndarray, sigma_px, inner_half=0.75, outer_half=1.35) -> np.ndarray:
    """4x4 covariance of [cx,cy,cz,psi] for the 8-corner (inner+outer) fit.

    ``sigma_px`` is scalar or (8,). The outer corners' larger metric baseline makes
    the depth block far tighter than the 4-corner version.
    """
    up_c, h1, h2 = vertical_basis_camera(down_cam)
    offsets = model_offsets_8(inner_half, outer_half)
    x0 = np.array([center[0], center[1], center[2], psi], float)

    def g(x):
        pts = corners_camera_model(x[:3], x[3], up_c, h1, h2, offsets)
        return _project(pts, K).ravel()  # (16,)

    r0 = g(x0)
    J = np.empty((r0.size, 4))
    for j in range(4):
        dx = np.zeros(4)
        dx[j] = 1e-4
        J[:, j] = (g(x0 + dx) - r0) / 1e-4

    sig = np.full(8, float(sigma_px)) if np.isscalar(sigma_px) else np.asarray(sigma_px, float)
    var = np.repeat(sig ** 2, 2)
    A = J.T @ (J / var[:, None])
    try:
        cov = np.linalg.inv(A + 1e-9 * np.eye(4))
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.inf)
    return cov


def depth_sigma_m(center: np.ndarray, pos_cov: np.ndarray) -> float:
    """1-sigma position uncertainty projected along the viewing ray (the depth axis)."""
    ray = np.asarray(center, float)
    ray = ray / (np.linalg.norm(ray) or 1.0)
    return float(np.sqrt(max(ray @ pos_cov[:3, :3] @ ray, 0.0)))


# --------------------------------------------------------------------------
# Interface-contract detection
# --------------------------------------------------------------------------
@dataclass
class GateDetection:
    corners_px: np.ndarray                 # (4,2) refined inner corners TL,TR,BR,BL
    corner_sigma_px: float                 # calibrated 1-sigma corner noise
    box_conf: float
    kpt_conf: np.ndarray                   # (4,)
    overflow: bool                         # truncated / gate fills frame -> distrust
    apparent_px: float
    low_parallax: bool = False             # large + near-fronto-parallel -> depth unreliable
    parallax: float = 0.0                  # perspective foreshortening [0,1] (0 = fronto-parallel)
    center_cam: np.ndarray | None = None   # gate center in camera frame (m)
    psi: float | None = None               # gate heading (rad)
    R: np.ndarray | None = None            # gate rotation (cam<-gate)
    pos_cov: np.ndarray | None = None      # 3x3 position covariance (camera frame, m^2)
    heading_var: float | None = None       # psi variance (rad^2)
    depth_sigma: float | None = None       # 1-sigma along the viewing ray (m)
    extra: dict = field(default_factory=dict)


def detect_gates(model, image, K, dist, half=0.75, down_cam=None,
                 conf=0.25, imgsz=640, device=0, refine=True) -> list[GateDetection]:
    """Full runtime path: detect -> refine -> flag -> upright pose + covariance.

    ``down_cam`` is the gravity (down) direction in the CAMERA frame, from the
    drone IMU. If None, the pose fields are left unset (corners + flags + sigma
    are still returned, which is enough for the estimator to consume).
    """
    from .infer import infer_image
    from .pnp import solve_pnp
    H, W = image.shape[:2]
    obj_up = np.array([[-half, half, 0.], [half, half, 0.], [half, -half, 0.], [-half, -half, 0.]])
    dets = infer_image(model, image, conf=conf, imgsz=imgsz, device=device)
    out: list[GateDetection] = []
    for d in dets:
        kpts = refine_corners(image, d.kpts_px) if refine else np.asarray(d.kpts_px, float)
        n_kp = len(kpts)
        use_outer = n_kp >= 8                    # 8-kpt model: inner 0-3, outer 4-7
        corners = kpts[:4]                        # inner quad drives flags/contract
        kconf = np.asarray(d.kpt_conf, float)
        ap = apparent_size_px(corners)
        of = is_overflow(corners, W, H)
        par = parallax_score(corners)
        low_par = (par < 0.05) and (ap > 90.0)   # big + fronto-parallel: wrong-basin risk
        sig = corner_sigma(ap, of)
        gd = GateDetection(corners_px=corners, corner_sigma_px=sig, box_conf=float(d.box_conf),
                           kpt_conf=kconf[:4], overflow=of, apparent_px=ap,
                           low_parallax=low_par, parallax=par)
        if use_outer:
            gd.extra["outer_px"] = kpts[4:8]
        if down_cam is not None and not of:
            if use_outer:
                # 8-corner upright fit: the outer square's larger baseline tightens depth.
                sol = solve_upright8(kpts[:8], down_cam, K, weights=kconf[:8])
                cov_fn = lambda s: pose_covariance8(s["center"], s["psi"], down_cam, K, sig)
            else:
                # seed the 4-pt upright solve with IPPE PnP (EKF team's §3a robustness ask)
                seed = solve_pnp(obj_up, corners, K, dist, "IPPE", True, True, kconf[:4])
                seed_rt = (seed["R"], seed["tvec"]) if seed is not None else None
                sol = solve_upright(corners, down_cam, K, half, weights=kconf[:4], seed_rt=seed_rt)
                cov_fn = lambda s: pose_covariance(s["center"], s["psi"], down_cam, K, half, sig)
            if sol is not None:
                cov4 = cov_fn(sol)
                gd.center_cam = sol["center"]
                gd.psi = sol["psi"]
                gd.R = sol["R"]
                gd.pos_cov = cov4[:3, :3]
                gd.heading_var = float(cov4[3, 3])
                gd.depth_sigma = depth_sigma_m(sol["center"], cov4[:3, :3])
        out.append(gd)
    return out
