"""Error metrics for corner localisation and PnP pose recovery."""

from __future__ import annotations

import cv2
import numpy as np


def corner_rmse(pred_px: np.ndarray, gt_px: np.ndarray) -> float:
    """RMS pixel distance between two ordered (4,2) corner sets."""
    d = np.asarray(pred_px, float) - np.asarray(gt_px, float)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Euclidean translation error in meters."""
    return float(np.linalg.norm(np.asarray(t_est, float).ravel() - np.asarray(t_gt, float).ravel()))


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angular distance between two rotations, in degrees."""
    R = np.asarray(R_est, float) @ np.asarray(R_gt, float).T
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def reprojection_rmse(object_pts: np.ndarray, image_pts: np.ndarray,
                      rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray,
                      dist: np.ndarray | None = None) -> float:
    """RMS reprojection error (px) of ``object_pts`` under (rvec, tvec) vs ``image_pts``."""
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    proj, _ = cv2.projectPoints(np.asarray(object_pts, float), rvec, tvec, np.asarray(K, float), dist)
    proj = proj.reshape(-1, 2)
    d = proj - np.asarray(image_pts, float).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(d ** 2, axis=1))))


def match_by_center(pred_centers: np.ndarray, gt_centers: np.ndarray,
                    max_dist: float) -> list[tuple[int, int]]:
    """Greedy nearest-neighbour matching of predictions to GT by inner-center distance.

    Returns a list of ``(pred_idx, gt_idx)`` pairs within ``max_dist`` pixels.
    """
    pred_centers = np.asarray(pred_centers, float).reshape(-1, 2)
    gt_centers = np.asarray(gt_centers, float).reshape(-1, 2)
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return []
    dmat = np.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], axis=2)
    pairs: list[tuple[int, int]] = []
    used_p, used_g = set(), set()
    order = np.dstack(np.unravel_index(np.argsort(dmat, axis=None), dmat.shape))[0]
    for p, g in order:
        if p in used_p or g in used_g:
            continue
        if dmat[p, g] > max_dist:
            break
        pairs.append((int(p), int(g)))
        used_p.add(p)
        used_g.add(g)
    return pairs
