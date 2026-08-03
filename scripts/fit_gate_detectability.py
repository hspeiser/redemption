"""Fit P(gate accepted by association) from live localizer debug logs.

Codex's caution for the co-visibility line search: a gate that merely
projects into the frustum is not a usable landmark.  This fits a small
logistic model on MEASURED association outcomes -- for every dense
inference and every gate the multigate associator considered visible,
the label is whether that gate ended up in accepted_gates.

Features (all computable from a candidate reference pose at plan time):
    log projected span (px), border margin (px, clipped),
    range (m), |cos| viewing angle vs gate normal.

Output: data/gate_detectability_v1.json with weights + calibration
table, consumed by the line optimizer's dual-gate coverage term.

    python scripts/fit_gate_detectability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.vq2_map import gate_quads_world_vq2  # noqa: E402

FRAME_W, FRAME_H = 640, 360


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-glob", default=r"D:\ai-gp\raw_sessions"
                    r"\vq2_20260802_*\localizer_debug\debug.jsonl")
    ap.add_argument("--map", default=str(
        REPO / "data/vq2_runtime_map_g9g15fix.json"))
    ap.add_argument("--out", default=str(
        REPO / "data/gate_detectability_v1.json"))
    args = ap.parse_args()

    gates = json.loads(Path(args.map).read_text())["gates"]
    corners = [np.concatenate(gate_quads_world_vq2(g)) for g in gates[:17]]
    centers = [np.mean(c, axis=0) for c in corners]
    normals = []
    for g in gates[:17]:
        qw, qx, qy, qz = g["quat_wxyz"]
        normals.append(
            Rotation.from_quat([qx, qy, qz, qw]).as_matrix()[:, 1])

    import glob
    X, y, meta = [], [], []
    for f in sorted(glob.glob(args.sessions_glob)):
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line).get("debug") or {}
            except Exception:
                continue
            mg = d.get("multigate")
            if not mg:
                continue
            pos = np.asarray(d.get("position", []), float)
            if pos.size != 3:
                continue
            accepted = set(mg.get("accepted_gates", []))
            expected = d.get("expected", [])
            by_gate = {}
            for e in expected:
                by_gate.setdefault(int(e["gate"]), []).append(e["pixel"])
            for gate_index in mg.get("visible_gates", []):
                if not 0 <= int(gate_index) < 17:
                    continue
                pix = np.asarray(by_gate.get(int(gate_index), []), float)
                if len(pix) < 2:
                    continue
                span = float(np.max(
                    np.linalg.norm(pix[:, None] - pix[None, :], axis=-1)))
                margin = float(min(
                    pix[:, 0].min(), FRAME_W - pix[:, 0].max(),
                    pix[:, 1].min(), FRAME_H - pix[:, 1].max()))
                rng = float(np.linalg.norm(
                    centers[gate_index] - pos))
                view = centers[gate_index] - pos
                view /= np.linalg.norm(view) + 1e-9
                cosang = abs(float(np.dot(view, normals[gate_index])))
                X.append([np.log(max(span, 2.0)),
                          np.clip(margin, -60.0, 120.0) / 60.0,
                          rng / 20.0, cosang])
                y.append(1.0 if int(gate_index) in accepted else 0.0)
                meta.append(int(gate_index))
    X = np.asarray(X)
    y = np.asarray(y)
    print(f"rows {len(y)}, acceptance rate {y.mean():.3f}")

    # logistic regression via Newton iterations
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(60):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) + 1e-3 * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + 1e-3 * np.eye(len(w))
        step = np.linalg.solve(H, grad)
        w -= step
        if np.linalg.norm(step) < 1e-8:
            break
    p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
    # simple discrimination + calibration report
    order = np.argsort(p)
    auc_num = auc_den = 0.0
    pos_ranks = np.searchsorted(np.sort(p), p[y == 1], side="right")
    neg = float((y == 0).sum())
    posn = float((y == 1).sum())
    auc = (pos_ranks.sum() - posn * (posn + 1) / 2) / max(posn * neg, 1)
    print(f"AUC {auc:.3f}")
    calib = []
    for lo in np.arange(0.0, 1.0, 0.1):
        m = (p >= lo) & (p < lo + 0.1)
        if m.sum() >= 20:
            calib.append({"bin": round(lo, 1),
                          "predicted": round(float(p[m].mean()), 3),
                          "observed": round(float(y[m].mean()), 3),
                          "n": int(m.sum())})
    for row in calib:
        print(row)
    Path(args.out).write_text(json.dumps({
        "weights": w.tolist(),
        "features": ["log_span_px", "border_margin_60px",
                     "range_20m", "abs_cos_view_angle", "bias"],
        "frame": [FRAME_W, FRAME_H],
        "rows": int(len(y)),
        "acceptance_rate": float(y.mean()),
        "auc": float(auc),
        "calibration": calib,
    }, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
