"""Human-in-the-loop VQ2 map editor.

Projects the gate map over replay frames using the EKF's saved trajectory
(vq2_align.py --dump-trace). You nudge gates until the wireframes sit on
the real gates; frame 0 is the parked spawn view where the pose is exact.

    .venv-train\\Scripts\\python.exe scripts\\vq2_map_editor.py ^
        --trace data\\vq2_trace.npz --map data\\vq2_map_zfix.json ^
        --out data\\vq2_map_human.json

Keys:
  A / D      prev / next frame          W / E    jump -30 / +30 frames
  TAB        select next gate (of those in view)
  arrows     nudge selected gate N/S/E/W (local frame), 10 cm
  PgUp/PgDn  nudge selected gate up / down, 10 cm
  , / .      rotate selected gate yaw -2 / +2 deg
  (hold SHIFT with any nudge: 4x step)
  R          reset selected gate to loaded value
  S          save corrected map to --out
  Q / ESC    quit
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
    args = ap.parse_args()

    calib = load_calib(REPO / "data/calib/calib.json")
    fx, fy, cx, cy = calib["K"]
    R_cb = np.asarray(calib["R_cb"])

    tr = np.load(args.trace, allow_pickle=False)
    t_arr, paths = tr["t"], tr["path"]
    pos_arr, quat_arr, sig_arr = tr["pos"], tr["quat"], tr["sigma"]
    m = json.loads(Path(args.map).read_text())
    gates = m["gates"]
    orig = json.loads(json.dumps(gates))
    n_g = len(gates)

    fi = 0
    sel = 0
    dirty = set()

    def gate_R(g):
        q = g["quat_wxyz"]
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    def gate_yaw(g):
        return float(Rotation.from_matrix(gate_R(g)).as_euler(
            "zyx", degrees=True)[0])

    def set_yaw(g, yaw):
        q = Rotation.from_euler("z", yaw, degrees=True).as_quat()
        g["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    def render():
        img = cv2.imread(str(paths[fi]))
        if img is None:
            img = np.zeros((H, W, 3), np.uint8)
        p_b = pos_arr[fi]
        qw, qx, qy, qz = quat_arr[fi]
        R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        R_cw = R_cb @ R_wb.T
        vis_g = []

        def proj(Xw):
            Xc = R_cw @ (np.asarray(Xw) - p_b)
            if Xc[2] < 0.3:
                return None
            return np.array([fx * Xc[0] / Xc[2] + cx,
                             fy * Xc[1] / Xc[2] + cy])

        for gi, g in enumerate(gates):
            cc = proj(g["pos"])
            if cc is None or not (-200 <= cc[0] <= W + 200
                                  and -150 <= cc[1] <= H + 150):
                continue
            vis_g.append(gi)
            Rg = gate_R(g)
            col = (0, 255, 255) if gi == sel else (
                (0, 255, 0) if gi in dirty else (180, 180, 180))
            for kind, th in (("hole", 2), ("panel", 1)):
                pts = []
                ok = True
                for c in SQ[kind]:
                    uv = proj(np.asarray(g["pos"]) + Rg @ c)
                    if uv is None or abs(uv[0]) > 4000 or abs(uv[1]) > 4000:
                        ok = False
                        break
                    pts.append(uv)
                if ok:
                    cv2.polylines(img, [np.array(pts, np.int32)], True,
                                  col, th)
            if 0 <= cc[0] < W and 0 <= cc[1] < H:
                cv2.putText(img, str(gi), tuple(cc.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        g = gates[sel]
        hud1 = (f"frame {fi}/{len(paths)-1}  t={t_arr[fi]:5.1f}s  "
                f"sigma {sig_arr[fi]*100:6.1f}cm"
                + ("  [POSE TRUSTED]" if sig_arr[fi] < 0.12 else
                   "  [pose rough - align elsewhere]"))
        hud2 = (f"sel g{sel}  pos=({g['pos'][0]:.2f},{g['pos'][1]:.2f},"
                f"{g['pos'][2]:.2f})  yaw={gate_yaw(g):.1f}  "
                f"edited: {sorted(dirty)}")
        big = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(big, (0, 0), (1280, 46), (20, 20, 20), -1)
        cv2.putText(big, hud1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1)
        cv2.putText(big, hud2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        return big, vis_g

    cv2.namedWindow("vq2 map editor", cv2.WINDOW_AUTOSIZE)
    while True:
        frame, vis_g = render()
        cv2.imshow("vq2 map editor", frame)
        k = cv2.waitKeyEx(0)
        step = 0.10
        ystep = 2.0
        kc = k & 0xFFFF
        if kc in (27, ord('q'), ord('Q')):
            break
        elif kc in (ord('a'),):
            fi = max(0, fi - 1)
        elif kc in (ord('d'),):
            fi = min(len(paths) - 1, fi + 1)
        elif kc in (ord('A'),):
            fi = max(0, fi - 5)
        elif kc in (ord('D'),):
            fi = min(len(paths) - 1, fi + 5)
        elif kc in (ord('w'), ord('W')):
            fi = max(0, fi - 30)
        elif kc in (ord('e'), ord('E')):
            fi = min(len(paths) - 1, fi + 30)
        elif kc == 9:                     # TAB
            if vis_g:
                if sel in vis_g:
                    sel = vis_g[(vis_g.index(sel) + 1) % len(vis_g)]
                else:
                    sel = vis_g[0]
        elif k == 2424832:                # left
            gates[sel]["pos"][1] -= step
            dirty.add(sel)
        elif k == 2555904:                # right
            gates[sel]["pos"][1] += step
            dirty.add(sel)
        elif k == 2490368:                # up = +north (away)
            gates[sel]["pos"][0] += step
            dirty.add(sel)
        elif k == 2621440:                # down
            gates[sel]["pos"][0] -= step
            dirty.add(sel)
        elif k == 2162688:                # PgUp = up in world (z down!)
            gates[sel]["pos"][2] -= step
            dirty.add(sel)
        elif k == 2228224:                # PgDn
            gates[sel]["pos"][2] += step
            dirty.add(sel)
        elif kc == ord(','):
            set_yaw(gates[sel], gate_yaw(gates[sel]) - ystep)
            dirty.add(sel)
        elif kc == ord('.'):
            set_yaw(gates[sel], gate_yaw(gates[sel]) + ystep)
            dirty.add(sel)
        elif kc in (ord('r'), ord('R')):
            gates[sel] = json.loads(json.dumps(orig[sel]))
            dirty.discard(sel)
        elif kc in (ord('s'), ord('S')):
            Path(args.out).write_text(json.dumps(
                {"frame": "local spawn (human-aligned)", "gates": gates},
                indent=1))
            print(f"saved -> {args.out} (edited gates: {sorted(dirty)})")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
