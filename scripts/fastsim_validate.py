"""Build the physically-anchored surrogate model and validate by
closed-form rollouts against (a) VQ1 odometry episodes and (b) the VQ2
clean lap with its cm-grade EKF trace as truth.

Model anchors: thrust 40 m/s^2 per wire unit (VQ1 calibration, hover
0.245), rate loop from the gyro fit, g = 9.81 z-down, weak linear drag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.data import Episode, load_episode, _read_jsonl  # noqa: E402
from aigp.fastsim.sysid import (  # noqa: E402
    SurrogateModel,
    fit_rate_loop,
    rollout,
)


def vq2_episode_from_trace(
    episode_dir: Path, trace_path: Path, hz: float = 100.0
) -> Episode | None:
    """Episode with truth pos/quat from an EKF trace (VQ2 has no odom)."""
    from scipy.spatial.transform import Rotation, Slerp

    base = load_episode(episode_dir, hz=hz, require_odometry=False)
    if base is None:
        return None
    tr = np.load(trace_path, allow_pickle=True)
    imu_rows = _read_jsonl(episode_dir / "imu.jsonl")
    t_imu = np.array([
        r["time_usec"] * 1e-6 for r in imu_rows if "time_usec" in r
    ])
    brk = np.where(np.diff(t_imu) < -0.5)[0]
    if len(brk):
        segs = np.split(np.arange(len(t_imu)), brk + 1)
        t_imu = t_imu[max(segs, key=len)]
    t0 = t_imu[0]
    t_tr = np.asarray(tr["t"], float) + t0
    pos_tr = np.asarray(tr["pos"], float)
    q = np.asarray(tr["quat"], float)   # wxyz
    quat_xyzw = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)
    inc = np.zeros(len(t_tr), bool)
    last = -np.inf
    for i, ti in enumerate(t_tr):
        if ti > last + 1e-6:
            inc[i] = True
            last = ti
    lo = max(base.t[0], t_tr[inc][0])
    hi = min(base.t[-1], t_tr[inc][-1])
    keep = (base.t >= lo) & (base.t <= hi)
    grid = base.t[keep]
    pos = np.stack([
        np.interp(grid, t_tr[inc], pos_tr[inc][:, c]) for c in range(3)
    ], axis=1)
    slerp = Slerp(t_tr[inc], Rotation.from_quat(quat_xyzw[inc]))
    quat = slerp(grid).as_quat()
    vel = savgol_filter(pos, 21, 3, deriv=1, delta=1.0 / hz, axis=0)
    return Episode(
        name=base.name + "(trace)",
        t=grid,
        cmd=base.cmd[keep],
        gyro=base.gyro[keep],
        accel=base.accel[keep],
        pos=pos,
        vel_world=vel,
        quat_wb=quat,
        rates_body=base.gyro[keep],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captures",
        default=r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
                r"\ai-grand-prix\outputs\captures",
    )
    parser.add_argument(
        "--out", default=str(REPO / "data" / "fastsim_model.json")
    )
    args = parser.parse_args()
    caps = Path(args.captures)

    fit_eps = []
    for name in ("rc_20260723_090226", "rc_20260723_090007",
                 "rc_20260723_085904", "rc_20260723_085752"):
        ep = load_episode(caps / name, hz=100.0, require_odometry=True)
        if ep is not None:
            fit_eps.append(ep)
    gains, taus, delay, _ = fit_rate_loop(fit_eps)
    print(f"rate loop: gain {np.round(gains, 3)} tau {np.round(taus, 4)} "
          f"delay {delay*1000:.0f}ms")
    model = SurrogateModel(
        rate_gain=gains.tolist(),
        rate_tau=[float(np.clip(t, 0.02, 0.12)) for t in taus],
        rate_delay=float(delay),
        thrust_gain=40.0,
        thrust_quad=0.0,
        drag_lin=[-0.30, -0.30, -0.30],
        g_vec=[0.0, 0.0, 9.81],
        hz=100.0,
    )
    model.save(args.out)
    print(f"wrote {args.out}")

    horizons = (2.0, 4.0)
    print("\n== VQ1 odometry validation ==")
    for ep in fit_eps[:3]:
        speed = np.linalg.norm(ep.vel_world, axis=1)
        flying = np.flatnonzero(speed > 4.0)
        if not len(flying):
            continue
        starts = np.linspace(flying[0], flying[-1] - 400, 4).astype(int)
        for h in horizons:
            rms = []
            for s in starts:
                r = rollout(model, ep, float(ep.t[s]), h,
                            attitude_from_truth=True)
                if r:
                    rms.append(r["pos_rmse_m"])
            if rms:
                print(f"  {ep.name} {h:.0f}s true-att rollouts: "
                      f"rmse median {np.median(rms):5.2f}m "
                      f"max {max(rms):5.2f}m (n={len(rms)})")

    print("\n== VQ2 clean lap (trace truth) ==")
    vq2 = vq2_episode_from_trace(
        caps / "rc_20260724_003101",
        REPO / "data" / "vq2_trace_101_v3.npz",
    )
    if vq2 is not None:
        speed = np.linalg.norm(vq2.vel_world, axis=1)
        flying = np.flatnonzero(speed > 4.0)
        starts = np.linspace(flying[0], flying[-1] - 500, 6).astype(int)
        for h in horizons:
            rows = []
            for s in starts:
                r = rollout(model, vq2, float(vq2.t[s]), h,
                            attitude_from_truth=True)
                ro = rollout(model, vq2, float(vq2.t[s]), h)
                if r and ro:
                    rows.append((r["pos_rmse_m"], ro["pos_rmse_m"]))
            if rows:
                ta = [r[0] for r in rows]
                oa = [r[1] for r in rows]
                print(f"  003101 {h:.0f}s: true-att rmse median "
                      f"{np.median(ta):5.2f}m max {max(ta):5.2f}m | "
                      f"open median {np.median(oa):5.2f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
