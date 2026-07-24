"""Live localization accuracy: overlay estimates vs absolute odometry.

Run while the sim streams (overlay must be running). Compares, over N seconds:
  - vision PnP absolute position (per tracked gate + static map)
  - pose-head absolute position
against ODOMETRY ground truth, paired by wall clock.

    uv run python scripts/probe_accuracy.py [--secs 40]
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp.mavlink_io import MavIO
from aigp.vision.labels import load_calib, gate_quads_world
from aigp.calib.solve import project
from scipy.spatial.transform import Rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=40.0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    mav = MavIO()
    calib = load_calib(repo / "data" / "calib" / "calib.json")
    fx, fy, cx, cy = calib["K"]
    gate_map = json.loads((repo / "data" / "episodes" / "calib01" /
                           "gates.json").read_text())
    print("sampling...", flush=True)
    rows_pnp = []   # (err_3d, range)
    rows_head = []
    rows_joint = []
    rows_rot = []
    corner_res = []  # per-corner px residual: tracker corner vs truth projection
    t_end = time.time() + args.secs
    while time.time() < t_end:
        try:
            with urllib.request.urlopen(
                    "http://localhost:8899/state.json", timeout=2) as r:
                st = json.loads(r.read())
        except Exception:
            time.sleep(0.2)
            continue
        od = mav.latest_odom()
        if od is None or not st:
            time.sleep(0.2)
            continue
        age_ms = (time.time_ns() - st.get("wall_ns", 0)) / 1e6
        if age_ms > 400:
            time.sleep(0.2)
            continue
        gt = np.array(od["pos"])
        ph = np.array(st["pose_head_world"])
        rows_head.append(float(np.linalg.norm(ph - gt)))
        if st.get("joint_world"):
            jw = np.array(st["joint_world"])
            rows_joint.append(float(np.linalg.norm(jw - gt)))
            if st.get("joint_R_wb"):
                q = od["quat_wxyz"]
                R_gt = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                R_est = np.array(st["joint_R_wb"])
                cosang = (np.trace(R_est.T @ R_gt) - 1) / 2
                rows_rot.append(float(np.degrees(
                    np.arccos(np.clip(cosang, -1, 1)))))
        # truth-projected corners from odometry pose
        q = od["quat_wxyz"]
        rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        for g in st.get("gates", []):
            est = np.array(g["cam_world"])
            rows_pnp.append((float(np.linalg.norm(est - gt)), g["range_m"],
                             g["gid"]))
            if gate_map and g.get("corners"):
                gm = gate_map[g["gid"]]
                hole_w, panel_w = gate_quads_world(gm)
                pts_w = np.concatenate([hole_w, panel_w])
                uv, valid, _ = project(pts_w, gt, rot, calib["R_cb"],
                                       calib["t_cb"], fx, fy, cx, cy, True)
                for ks, (cu, cv) in g["corners"].items():
                    k = int(ks)
                    if valid[k]:
                        corner_res.append((k, cu - uv[k, 0], cv - uv[k, 1]))
        time.sleep(0.25)

    mav.close()
    print(f"\nsamples: pose-head {len(rows_head)}, PnP {len(rows_pnp)}")
    if rows_head:
        a = np.array(rows_head)
        print(f"POSE-HEAD abs err: median {np.median(a):.2f} m  "
              f"p90 {np.percentile(a, 90):.2f} m")
    if rows_joint:
        a = np.array(rows_joint)
        print(f"JOINT MULTI-GATE abs err: median {np.median(a)*100:.1f} cm  "
              f"p90 {np.percentile(a, 90)*100:.1f} cm  (n={len(a)})")
    if rows_rot:
        a = np.array(rows_rot)
        print(f"JOINT MULTI-GATE rot err: median {np.median(a):.3f} deg  "
              f"p90 {np.percentile(a, 90):.3f} deg")
    if rows_pnp:
        r = np.array([x[:2] for x in rows_pnp])
        print(f"VISION-PnP abs err: median {np.median(r[:,0])*100:.1f} cm  "
              f"p90 {np.percentile(r[:,0],90)*100:.1f} cm")
        for lo, hi in [(0, 10), (10, 25), (25, 60)]:
            m = (r[:, 1] >= lo) & (r[:, 1] < hi)
            if m.sum() > 3:
                print(f"  gate range {lo:2d}-{hi:2d} m: n={m.sum():4d}  "
                      f"median {np.median(r[m,0])*100:6.1f} cm  "
                      f"p90 {np.percentile(r[m,0],90)*100:6.1f} cm")
        for gid in sorted(set(int(x[2]) for x in rows_pnp)):
            e = np.array([x[0] for x in rows_pnp if int(x[2]) == gid])
            print(f"  via gate {gid}: n={len(e):4d} median {np.median(e)*100:.1f} cm")
    if corner_res:
        a = np.array([(du, dv) for (_, du, dv) in corner_res])
        print(f"\nTRACKER-CORNER vs TRUTH-PROJECTION residuals "
              f"(n={len(a)}):")
        print(f"  du: median {np.median(a[:,0]):+.2f}px  "
              f"dv: median {np.median(a[:,1]):+.2f}px  "
              f"|d|: median {np.median(np.linalg.norm(a,axis=1)):.2f}px")
        for k in sorted(set(x[0] for x in corner_res)):
            b = np.array([(du, dv) for (kk, du, dv) in corner_res if kk == k])
            print(f"  corner {k}: du {np.median(b[:,0]):+6.2f} "
                  f"dv {np.median(b[:,1]):+6.2f}  n={len(b)}")


if __name__ == "__main__":
    main()
