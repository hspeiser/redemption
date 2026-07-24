"""Replay recorded racing frames through the FULL live tracking stack and
measure absolute localization error against time-aligned odometry truth.

Uses the same Annotator as the live overlay (net + tracks + One-Euro +
branch disambiguation + joint multi-gate solve), paced at ~30 fps so the
temporal filters behave as in real time. Ground truth comes from the label
files (exact sim-clock alignment — no wall-clock pairing error).

    .venv-train\\Scripts\\python.exe scripts\\replay_eval.py [--episodes ...]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.live_overlay import Annotator

DEFAULT_EPS = ("rc_20260723_022140", "rc_20260723_022654", "rc_20260723_024016")


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def run_episode(ann, npz_path, pace_fps=30.0):
    d = np.load(npz_path)
    n = len(d["path"])
    rows = []          # (speed, err_joint or nan, n_tracked_gates)
    singles = []       # (range_m, err_single)
    t_next = time.perf_counter()
    prev_cam = None    # (R_cam_from_world, p_world)
    for i in range(n):
        bgr = cv2.imread(str(d["path"][i]))
        if bgr is None:
            continue
        # inter-frame camera motion from truth (gyro/EKF proxy: the sim IMU
        # is noiseless, so a deployed integrator supplies exactly this)
        Rwb = quat_to_R(d["quat"][i].astype(np.float64))
        R_cam = ann.R_cb @ Rwb.T
        p = d["pos"][i].astype(np.float64)
        motion = None
        if prev_cam is not None:
            R1, p1 = prev_cam
            R_dc = R_cam @ R1.T
            t_dc = R_cam @ (p1 - p)
            motion = (R_dc, t_dc)
        prev_cam = (R_cam, p)
        ann.process(bgr, motion=motion)
        st = ann.state
        gt = d["pos"][i].astype(np.float64)
        speed = float(np.linalg.norm(d["vel"][i]))
        ej = np.nan
        if st.get("joint_world"):
            ej = float(np.linalg.norm(np.array(st["joint_world"]) - gt))
        rows.append((speed, ej, len(st.get("gates", []))))
        for g in st.get("gates", []):
            singles.append((g["range_m"],
                            float(np.linalg.norm(np.array(g["cam_world"]) - gt))))
        t_next += 1.0 / pace_fps
        dts = t_next - time.perf_counter()
        if dts > 0:
            time.sleep(dts)
    return rows, singles


def report(name, rows, singles):
    r = np.array(rows)
    have = np.isfinite(r[:, 1])
    print(f"\n=== {name}: {len(r)} frames, joint-fix availability "
          f"{100*have.mean():.1f}% ===")
    if have.sum() > 5:
        e = r[have, 1]
        print(f"JOINT abs err: median {np.median(e)*100:6.1f} cm  "
              f"p90 {np.percentile(e,90)*100:6.1f} cm  "
              f"p99 {np.percentile(e,99)*100:6.1f} cm")
        for lo, hi in [(0, 4), (4, 10), (10, 25)]:
            m = have & (r[:, 0] >= lo) & (r[:, 0] < hi)
            if m.sum() > 5:
                e = r[m, 1]
                print(f"  speed {lo:2d}-{hi:2d} m/s: n={m.sum():5d}  "
                      f"median {np.median(e)*100:6.1f} cm  "
                      f"p90 {np.percentile(e,90)*100:6.1f} cm")
    if singles:
        s = np.array(singles)
        print(f"SINGLE-GATE abs err by range:")
        for lo, hi in [(0, 8), (8, 15), (15, 30), (30, 60)]:
            m = (s[:, 0] >= lo) & (s[:, 0] < hi)
            if m.sum() > 5:
                print(f"  range {lo:2d}-{hi:2d} m: n={m.sum():5d}  "
                      f"median {np.median(s[m,1])*100:6.1f} cm  "
                      f"p90 {np.percentile(s[m,1],90)*100:6.1f} cm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", default=list(DEFAULT_EPS))
    ap.add_argument("--pace", type=float, default=30.0)
    args = ap.parse_args()

    ann = Annotator()
    for stem in args.episodes:
        npz = REPO / "data" / "labels" / f"{stem}.npz"
        if not npz.exists():
            npz = REPO / "data" / "labels_quarantine" / f"{stem}.npz"
        if not npz.exists():
            print(f"{stem}: no labels file")
            continue
        # fresh tracker state per episode
        ann.tracks = {}
        rows, singles = run_episode(ann, npz, args.pace)
        report(stem, rows, singles)


if __name__ == "__main__":
    main()
