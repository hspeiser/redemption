"""Grouped multi-gate corner association for the VQ2 live localizer.

The stock matcher considers gates {active-1, active, active+1}, matches
each gate to dense corner peaks independently (nearest-in-radius), and
lets two gates claim the same image peak.  This module upgrades that to
course-wide association with global exclusivity:

  1. Project every mapped gate through the EKF prediction; cull gates
     that are behind the camera, outside the frame margin, or beyond
     range.
  2. Resolve each gate's apparent-class flip (viewing side) exactly as
     the stock matcher does — count/cost of nearest matches.
  3. Assign peaks to gate corners with a global one-to-one assignment
     per corner class (Hungarian), so no image peak is used twice.
  4. Accept a gate only with enough exclusive corners (more required at
     range, where attitude lag moves projections by meters).
  5. Return one joint observation set; the EKF's batched chi-square
     update is the joint solve.  Attitude correction is allowed only
     when the accepted constellation has real angular separation —
     one planar square cannot safely correct gravity, two separated
     gates can.

Opt-in from the localizer via AIGP_MULTIGATE=1 (association upgrade,
position/velocity only) and additionally AIGP_MULTIGATE_ATT=1 (attitude
correction when the constellation supports it).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

FLIP = (1, 0, 3, 2, 5, 4, 7, 6)


def _project_gate(ekf, corners8):
    """Project all 8 corners; list of (uv or None, cam_z)."""
    out = []
    for corner in corners8:
        uv, Xc = ekf.predict_pixel(corner)
        out.append((uv, float(Xc[2])))
    return out


def cull_visible_gates(
    ekf,
    gate_world,
    frame_wh=(640, 360),
    margin_px=48.0,
    min_range_m=1.2,
    max_range_m=45.0,
    min_corners_in_frame=4,
):
    """Gate indices plausibly visible under the EKF prediction."""
    width, height = frame_wh
    visible = []
    for gate_index, corners8 in enumerate(gate_world):
        center = np.mean(corners8, axis=0)
        rng = float(np.linalg.norm(center - ekf.p))
        if not min_range_m <= rng <= max_range_m:
            continue
        in_frame = 0
        for uv, _z in _project_gate(ekf, corners8):
            if uv is None:
                continue
            if (-margin_px <= uv[0] <= width + margin_px
                    and -margin_px <= uv[1] <= height + margin_px):
                in_frame += 1
        if in_frame >= min_corners_in_frame:
            visible.append((gate_index, rng))
    return visible


def _resolve_flip(ekf, corners8, peaks, radius):
    """Stock flip resolution: more matches, then lower cost."""
    hypotheses = []
    for flipped in (False, True):
        count, cost = 0, 0.0
        for corner_class in range(8):
            physical = FLIP[corner_class] if flipped else corner_class
            uv_pred, _ = ekf.predict_pixel(corners8[physical])
            if uv_pred is None:
                continue
            best = min(
                (float(np.hypot(p[0] - uv_pred[0], p[1] - uv_pred[1]))
                 for p in peaks[corner_class]),
                default=None,
            )
            if best is not None and best < radius:
                count += 1
                cost += best
        hypotheses.append((count, cost, flipped))
    count, cost, flipped = min(hypotheses, key=lambda h: (-h[0], h[1]))
    return flipped, count


def associate_multigate(
    ekf,
    gate_world,
    peaks,
    radius_px,
    frame_wh=(640, 360),
    max_range_m=45.0,
    far_range_m=18.0,
    min_corners_near=2,
    min_corners_far=3,
    attitude_min_gates=2,
    attitude_min_rows=6,
    attitude_min_separation_deg=12.0,
    exclude_far=False,
):
    """Global one-to-one gate/peak association.

    Returns (observations, debug_matches, constellation) where
    observations is [(Xw, uv)] ready for GateEKF.update_corners and
    constellation carries per-gate acceptance plus whether an attitude
    update is geometrically safe.
    """
    visible = cull_visible_gates(
        ekf, gate_world, frame_wh=frame_wh, max_range_m=max_range_m)
    if exclude_far:
        visible = [(g, rng) for g, rng in visible if rng <= far_range_m]
    flips = {}
    for gate_index, _rng in visible:
        flipped, count = _resolve_flip(
            ekf, gate_world[gate_index], peaks, radius_px)
        if count >= 1:
            flips[gate_index] = flipped

    # per-class global assignment: rows = (gate, physical, uv_pred),
    # cols = detected peaks of that class
    assigned = {}          # (gate, class) -> (uv_obs, dist, score)
    BIG = 1e6
    for corner_class in range(8):
        rows = []
        for gate_index, _rng in visible:
            if gate_index not in flips:
                continue
            physical = (FLIP[corner_class] if flips[gate_index]
                        else corner_class)
            uv_pred, _ = ekf.predict_pixel(
                gate_world[gate_index][physical])
            if uv_pred is not None:
                rows.append((gate_index, physical, uv_pred))
        class_peaks = peaks[corner_class]
        if not rows or not class_peaks:
            continue
        cost = np.full((len(rows), len(class_peaks)), BIG)
        for i, (_g, _p, uv_pred) in enumerate(rows):
            for j, peak in enumerate(class_peaks):
                d = float(np.hypot(peak[0] - uv_pred[0],
                                   peak[1] - uv_pred[1]))
                if d < radius_px:
                    cost[i, j] = d
        row_idx, col_idx = linear_sum_assignment(cost)
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] >= BIG:
                continue
            gate_index, physical, uv_pred = rows[i]
            peak = class_peaks[j]
            assigned[(gate_index, corner_class)] = (
                physical, uv_pred,
                np.asarray(peak[:2], float), cost[i, j],
                float(peak[2]) if len(peak) > 2 else 0.0,
            )

    # gate-level acceptance
    range_of = dict(visible)
    per_gate = {}
    for (gate_index, corner_class), row in assigned.items():
        per_gate.setdefault(gate_index, []).append((corner_class, row))
    observations = []
    debug_matches = []
    accepted_gates = []
    for gate_index, rows in sorted(per_gate.items()):
        rng = range_of[gate_index]
        need = min_corners_far if rng > far_range_m else min_corners_near
        if len(rows) < need:
            continue
        accepted_gates.append(gate_index)
        for corner_class, (physical, uv_pred, uv_obs, dist,
                           score) in rows:
            observations.append((
                gate_world[gate_index][physical], uv_obs,
            ))
            debug_matches.append({
                "gate": int(gate_index),
                "class": int(corner_class),
                "physical": int(physical),
                "flipped": bool(flips[gate_index]),
                "predicted": [float(uv_pred[0]), float(uv_pred[1])],
                "observed": [float(uv_obs[0]), float(uv_obs[1])],
                "distance_px": float(dist),
                "score": float(score),
                "range_m": float(rng),
            })

    # constellation conditioning for attitude safety
    attitude_ok = False
    max_sep_deg = 0.0
    if len(accepted_gates) >= attitude_min_gates and \
            len(observations) >= attitude_min_rows:
        bearings = []
        for gate_index in accepted_gates:
            center = np.mean(gate_world[gate_index], axis=0)
            vec = center - ekf.p
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                bearings.append(vec / norm)
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                cosang = float(np.clip(
                    np.dot(bearings[i], bearings[j]), -1.0, 1.0))
                max_sep_deg = max(max_sep_deg,
                                  float(np.degrees(np.arccos(cosang))))
        attitude_ok = max_sep_deg >= attitude_min_separation_deg

    constellation = {
        "visible_gates": [int(g) for g, _ in visible],
        "accepted_gates": [int(g) for g in accepted_gates],
        "rows": len(observations),
        "max_separation_deg": round(max_sep_deg, 1),
        "attitude_ok": bool(attitude_ok),
    }
    return observations, debug_matches, constellation
