"""Render a map-verification video: every map gate wireframe projected
from the saved EKF trajectory, every frame. No detections, no net -
this shows exactly where the MAP is right and wrong.

    .venv-train\\Scripts\\python.exe scripts\\vq2_map_video.py ^
        --trace data\\vq2_trace_r4.npz --map data\\vq2_map_shift.json ^
        --out data\\vq2_map_check.mp4
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

W, H = 640, 360
HOLE, PANEL = 0.75, 1.36
SQ = {"hole": np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE],
                        [HOLE, 0, HOLE], [-HOLE, 0, HOLE]]),
      "panel": np.array([[-PANEL, 0, -PANEL], [PANEL, 0, -PANEL],
                         [PANEL, 0, PANEL], [-PANEL, 0, PANEL]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gate", type=int, default=16,
                    help="highest race gate to draw (17 is the non-visual "
                         "finish marker and is hidden by default)")
    args = ap.parse_args()

    calib = load_calib(REPO / "data/calib/calib.json")
    fx, fy, cx, cy = calib["K"]
    R_cb = np.asarray(calib["R_cb"])
    tr = np.load(args.trace, allow_pickle=False)
    gates = json.loads(Path(args.map).read_text())["gates"]
    gR = []
    for g in gates:
        q = g["quat_wxyz"]
        gR.append(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix())

    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                         (W, H))
    n = len(tr["path"])
    for i in range(n):
        img = cv2.imread(str(tr["path"][i]))
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        p_b = tr["pos"][i]
        qw, qx, qy, qz = tr["quat"][i]
        R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        R_cw = R_cb @ R_wb.T
        for gi, g in enumerate(gates):
            if gi > args.max_gate:
                continue
            pts_all = []
            ok = True
            for kind in ("hole", "panel"):
                pts = []
                for c in SQ[kind]:
                    Xc = R_cw @ (np.asarray(g["pos"]) + gR[gi] @ c - p_b)
                    if Xc[2] < 0.4:
                        ok = False
                        break
                    u = fx * Xc[0] / Xc[2] + cx
                    v = fy * Xc[1] / Xc[2] + cy
                    if not (-4000 <= u <= 4000 and -4000 <= v <= 4000):
                        ok = False
                        break
                    pts.append([u, v])
                if not ok:
                    break
                pts_all.append(np.array(pts, np.int32))
            if ok:
                for k, quad in enumerate(pts_all):
                    cv2.polylines(img, [quad], True, (0, 255, 0),
                                  2 if k == 0 else 1)
                cc = pts_all[0].mean(axis=0)
                if 0 <= cc[0] < W - 20 and 12 <= cc[1] < H:
                    cv2.putText(img, str(gi), tuple(cc.astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 200, 255), 2)
        if "splice_weight" in tr.files:
            mode = "HYBRID VISUAL LOCK"
        elif "visual_smooth_delta" in tr.files:
            mode = "SPARSE VISUAL SMOOTH"
        else:
            mode = f"sigma {tr['sigma'][i]*100:6.1f}cm"
        cv2.putText(img, f"t {tr['t'][i]:5.1f}s  {mode}  MAP CHECK",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        vw.write(img)
    vw.release()
    print(f"wrote {args.out} ({n} frames)")


if __name__ == "__main__":
    main()
