"""Classical red/orange gate-quad detector used to bootstrap camera calibration.

Finds the gate border as a contour-with-hole, fits quads to the outer boundary
and the hole, and sub-pixel refines the corners. Accuracy target here is only
~0.5 px — the bundle adjustment averages over thousands of observations.
"""

import cv2
import numpy as np


def gate_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_ok = (h <= 25) | (h >= 160)          # red through orange, wrapping
    mask = (hue_ok & (s >= 90) & (v >= 70)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _fit_quad(cnt):
    peri = cv2.arcLength(cnt, True)
    for eps in (0.02, 0.04, 0.06):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float64)
    return None


def _order_cw(quad):
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    return quad[np.argsort(ang)]


def detect_gates(bgr, min_area=250.0, border_margin=3):
    """Returns list of dicts: {outer (4,2), inner (4,2) or None, area, center}."""
    hgt, wid = bgr.shape[:2]
    mask = gate_mask(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    if hierarchy is None:
        return out
    hierarchy = hierarchy[0]
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    def refine(quad):
        pts = quad.astype(np.float32).reshape(-1, 1, 2)
        try:
            cv2.cornerSubPix(gray, pts, (3, 3), (-1, -1), term)
        except cv2.error:
            return quad
        return pts.reshape(4, 2).astype(np.float64)

    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:      # only top-level contours
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        quad = _fit_quad(cnt)
        if quad is None:
            continue
        if (quad.min() < border_margin or quad[:, 0].max() > wid - border_margin
                or quad[:, 1].max() > hgt - border_margin):
            continue                    # touches image edge; partial gate
        # largest hole child = inner quad
        inner = None
        child = hierarchy[i][2]
        best_hole = 0.0
        while child != -1:
            ha = cv2.contourArea(contours[child])
            if ha > best_hole and ha > 0.2 * area:
                q = _fit_quad(contours[child])
                if q is not None:
                    inner = q
                    best_hole = ha
            child = hierarchy[child][0]
        out.append({
            "outer": _order_cw(refine(quad)),
            "inner": _order_cw(refine(inner)) if inner is not None else None,
            "area": area,
            "center": quad.mean(axis=0),
        })
    return out


def debug_draw(bgr, dets):
    img = bgr.copy()
    for d in dets:
        cv2.polylines(img, [d["outer"].astype(np.int32)], True, (0, 255, 0), 1)
        if d["inner"] is not None:
            cv2.polylines(img, [d["inner"].astype(np.int32)], True, (0, 255, 255), 1)
        for p in d["outer"]:
            cv2.circle(img, tuple(np.round(p).astype(int)), 2, (255, 0, 255), -1)
    return img
