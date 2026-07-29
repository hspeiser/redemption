"""Place human corner-fits directly: re-solve each journal fit's 4 hole
clicks (same IPPE_SQUARE + LM recipe as the editor) and place the gate
with a TRUSTED trace pose at that frame — no session-trace inversion.

    .venv-train\\Scripts\\python.exe scripts\\vq2_fits_place.py ^
      --journal j1.jsonl [--journal j2.jsonl] --trace trusted.npz ^
      [--session-trace editor.npz] [--max-rms 1.0] [--max-sig 0.5]

--session-trace guards frame-index alignment: if given, the frame paths
of both traces must match at every fit index.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.vision.labels import load_calib  # noqa: E402

HOLE = 0.75
SQ_HOLE = np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE],
                    [HOLE, 0, HOLE], [-HOLE, 0, HOLE]])
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def circ_median(deg):
    deg = np.asarray(deg, float)
    best, bcost = None, None
    for c in deg:
        cost = np.abs(wrap(deg - c)).sum()
        if bcost is None or cost < bcost:
            best, bcost = c, cost
    return float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", action="append", required=True)
    ap.add_argument("--trace", required=True,
                    help="trusted trace: pose used to place the fits")
    ap.add_argument("--session-trace", default=None,
                    help="editor's trace, for frame-path alignment check")
    ap.add_argument("--max-rms", type=float, default=1.0)
    ap.add_argument("--max-sig", type=float, default=0.5,
                    help="skip fits where the trusted trace sigma_p (m) "
                         "exceeds this")
    ap.add_argument("--gates", default="9-16")
    args = ap.parse_args()
    g_lo, g_hi = [int(x) for x in args.gates.split("-")]

    calib = load_calib(REPO / "data/calib/calib.json")
    fx, fy, cx, cy = calib["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R_cb = np.asarray(calib["R_cb"])
    tr = np.load(args.trace, allow_pickle=True)
    st = np.load(args.session_trace, allow_pickle=True) \
        if args.session_trace else None
    obj = np.ascontiguousarray(SQ_HOLE @ RX90.T)

    by_gate = {}
    n_skip_sig = n_skip_rms = 0
    for jn in args.journal:
        for ln in Path(jn).read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if not r.get("ok") or "clicks" not in r:
                continue
            g = int(r["gate"])
            if not (g_lo <= g <= g_hi):
                continue
            fidx = int(r["frame"])
            if fidx >= len(tr["t"]):
                continue
            if st is not None and (fidx >= len(st["t"]) or
                                   str(st["path"][fidx]) !=
                                   str(tr["path"][fidx])):
                print(f"  frame {fidx}: trace paths differ, skipped")
                continue
            sig = float(tr["sigma"][fidx])
            if sig > args.max_sig:
                n_skip_sig += 1
                continue
            ip = np.ascontiguousarray(
                r["clicks"], np.float64).reshape(-1, 1, 2)
            try:
                _n, rv, tv, _e = cv2.solvePnPGeneric(
                    obj, ip, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error:
                continue
            p_b = np.asarray(tr["pos"][fidx], float)
            qw, qx, qy, qz = tr["quat"][fidx]
            R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            R_wc = R_wb @ R_cb.T
            best = None
            for r0, t0 in zip(rv, tv):
                try:
                    r0, t0 = cv2.solvePnPRefineLM(obj, ip, K, None, r0, t0)
                except cv2.error:
                    continue
                pr, _ = cv2.projectPoints(obj, r0, t0, K, None)
                rms = float(np.sqrt(((pr - ip) ** 2).sum(axis=2).mean()))
                R0, _ = cv2.Rodrigues(r0)
                R_gw = R_wc @ (R0 @ RX90)
                up_err = abs(float(R_gw[2, 2]) - 1.0)
                score = rms + 5.0 * up_err
                if best is None or score < best[0]:
                    best = (score, rms, t0.ravel(), R_gw)
            if best is None:
                continue
            _sc, rms, t_c, R_gw = best
            if rms > args.max_rms:
                n_skip_rms += 1
                continue
            p_g = p_b + R_wc @ t_c
            yaw = float(np.degrees(np.arctan2(R_gw[1, 0], R_gw[0, 0])))
            by_gate.setdefault(g, []).append(
                {"pos": p_g, "yaw": yaw, "rms": rms, "sig": sig,
                 "frame": fidx})
    print(f"skipped: {n_skip_sig} loose-sigma, {n_skip_rms} high-rms")
    for g in sorted(by_gate):
        fits = by_gate[g]
        P = np.array([f["pos"] for f in fits])
        med = np.median(P, axis=0)
        spread = np.linalg.norm(P - med, axis=1)
        yaw = circ_median([f["yaw"] for f in fits])
        sig_med = np.median([f["sig"] for f in fits])
        print(f"g{g:2d}: n={len(fits):2d}  pos {np.round(med, 2)}  "
              f"yaw {yaw:+7.1f}  spread med {np.median(spread)*100:4.0f}cm "
              f"max {spread.max()*100:4.0f}cm  trace-sig {sig_med*100:.0f}cm")


if __name__ == "__main__":
    main()
