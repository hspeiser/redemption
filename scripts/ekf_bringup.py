"""EKF bring-up step 1: find IMU axis conventions by dead-reckoning.

Sweeps gyro/accel sign combinations, integrating IMU-only over short windows
started from ground-truth state, ranked by end-of-window drift vs truth.
Uses banked/fast segments (conventions are unobservable at rest).

    uv run python scripts/ekf_bringup.py [--episode rc_20260723_022654]
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ingest import load_training_episode
from aigp.ekf import GateEKF

ROOT = Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379\ai-grand-prix\outputs\captures")


def load_imu(ep_dir):
    rows = []
    with open(Path(ep_dir) / "imu.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "time_usec" in r:
                rows.append((r["time_usec"] * 1e-6, *r["accel"], *r["gyro"]))
    a = np.array(rows)
    # dedupe repeated samples
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])
    return a[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="rc_20260723_022654")
    ap.add_argument("--win", type=float, default=1.5)
    ap.add_argument("--nwin", type=int, default=24)
    args = ap.parse_args()

    b = load_training_episode(ROOT / args.episode)
    od = b["odom"]           # corrected quats, t_us col 0, body vel 8:11
    imu = load_imu(ROOT / args.episode)
    print(f"odom {len(od)} rows, imu {len(imu)} rows")

    # keep the longest monotonic clock segment (sim resets restart time)
    t_all = od[:, 0]
    breaks = np.where(np.diff(t_all) <= 0)[0] + 1
    segs = np.split(np.arange(len(od)), breaks)
    seg = max(segs, key=len)
    od = od[seg]
    t_od = od[:, 0] * 1e-6
    m_imu = (imu[:, 0] >= t_od[0]) & (imu[:, 0] <= t_od[-1])
    imu = imu[m_imu]
    print(f"longest segment: odom {len(od)} rows, imu {len(imu)} rows, "
          f"span {t_od[-1]-t_od[0]:.1f}s")

    def gt_state(t):
        i = np.searchsorted(t_od, t)
        i = np.clip(i, 1, len(od) - 1)
        a = (t - t_od[i - 1]) / max(t_od[i] - t_od[i - 1], 1e-9)
        p = (1 - a) * od[i - 1, 1:4] + a * od[i, 1:4]
        qr = od[i, 4:8]
        R = Rotation.from_quat([qr[1], qr[2], qr[3], qr[0]])
        # world velocity from position derivative (frame-free; the raw
        # ODOMETRY vel vector's axis convention is untrusted)
        j0 = max(0, i - 3)
        j1 = min(len(od) - 1, i + 3)
        v_world = (od[j1, 1:4] - od[j0, 1:4]) / max(
            (t_od[j1] - t_od[j0]), 1e-9)
        return p, v_world, qr, R

    # pick windows with high rotation rates (banked segments)
    gyro_mag = np.linalg.norm(imu[:, 4:7], axis=1)
    order = np.argsort(-gyro_mag)
    starts = []
    for idx in order:
        t0 = imu[idx, 0]
        if t0 < t_od[0] + 1 or t0 > t_od[-1] - args.win - 1:
            continue
        if all(abs(t0 - s) > args.win for s in starts):
            starts.append(t0)
        if len(starts) >= args.nwin:
            break
    print(f"{len(starts)} windows, median |gyro| at start: "
          f"{np.median([gyro_mag[np.searchsorted(imu[:,0], s)] for s in starts]):.2f} rad/s")

    signs = [(a, c, d) for a in (1, -1) for c in (1, -1) for d in (1, -1)]
    results = []
    for gsign in signs:
        for asign in signs:
            errs = []
            for t0 in starts:
                p0, v0, q0, _ = gt_state(t0)
                ekf = GateEKF(np.eye(3), np.eye(3), gyro_sign=gsign,
                              accel_sign=asign)
                ekf.init_state(p0, v0, q0, t0)
                i0 = np.searchsorted(imu[:, 0], t0)
                i1 = np.searchsorted(imu[:, 0], t0 + args.win)
                for i in range(i0, i1):
                    ekf.propagate(imu[i, 0], imu[i, 1:4], imu[i, 4:7])
                p1, v1, q1, R1 = gt_state(imu[i1 - 1, 0])
                perr = np.linalg.norm(ekf.p - p1)
                Re = ekf.q.as_matrix()
                cosang = (np.trace(Re.T @ R1.as_matrix()) - 1) / 2
                aerr = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                errs.append((perr, aerr))
            e = np.array(errs)
            results.append((float(np.median(e[:, 0])), float(np.median(e[:, 1])),
                            gsign, asign))
    results.sort()
    print(f"\n{'pos_med(m)':>10s} {'att_med(deg)':>12s}  gyro_sign  accel_sign")
    for (pe, ae, gs, asn) in results[:8]:
        print(f"{pe:10.3f} {ae:12.2f}  {str(gs):10s} {str(asn):10s}")
    print("...")
    for (pe, ae, gs, asn) in results[-2:]:
        print(f"{pe:10.3f} {ae:12.2f}  {str(gs):10s} {str(asn):10s}")


if __name__ == "__main__":
    main()
