"""Episode loading + time alignment for surrogate-dynamics identification.

All streams are resampled onto a uniform sim-clock grid:
  * cmd.jsonl   -- wire commands (roll, pitch, yaw rate, thrust), wall clock
  * imu.jsonl   -- HIGHRES_IMU accel/gyro, sim clock + wall (pairing source)
  * mav.jsonl   -- ODOMETRY rows (VQ1 only): pos, quat, body vel, body rates

VQ1 odometry conventions (measured, see memory/aigp-sim-conventions):
  quat is Y-flipped left-handed; body->world = (w, -x, y, -z).
  vel is BODY frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Episode:
    name: str
    t: np.ndarray            # uniform sim-clock grid [s]
    cmd: np.ndarray          # (N,4) wire command, sampled-and-held
    gyro: np.ndarray         # (N,3) raw HIGHRES_IMU gyro
    accel: np.ndarray        # (N,3) raw HIGHRES_IMU accel (specific force)
    pos: np.ndarray | None   # (N,3) odometry position (VQ1)
    vel_world: np.ndarray | None
    quat_wb: np.ndarray | None  # (N,4) xyzw scipy order, body->world
    rates_body: np.ndarray | None


def _read_jsonl(path: Path):
    rows = []
    with open(path) as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _longest_monotonic(t: np.ndarray) -> np.ndarray:
    breaks = np.where(np.diff(t) < -0.5)[0]
    if not len(breaks):
        return np.arange(len(t))
    segments = np.split(np.arange(len(t)), breaks + 1)
    return max(segments, key=len)


def load_episode(
    episode_dir: str | Path,
    hz: float = 100.0,
    require_odometry: bool = False,
) -> Episode | None:
    ep = Path(episode_dir)
    imu_rows = _read_jsonl(ep / "imu.jsonl")
    imu_rows = [r for r in imu_rows if "time_usec" in r and "wall" in r]
    if len(imu_rows) < 200:
        return None
    t_imu = np.array([r["time_usec"] * 1e-6 for r in imu_rows])
    keep = _longest_monotonic(t_imu)
    imu_rows = [imu_rows[i] for i in keep]
    t_imu = t_imu[keep]
    wall_imu = np.array([r["wall"] for r in imu_rows])
    gyro = np.array([r["gyro"] for r in imu_rows], float)
    accel = np.array([r["accel"] for r in imu_rows], float)

    cmd_rows = _read_jsonl(ep / "cmd.jsonl")
    cmd_rows = [r for r in cmd_rows if "cmd" in r and "wall" in r]
    if len(cmd_rows) < 50:
        return None
    wall_cmd = np.array([r["wall"] for r in cmd_rows])
    cmd = np.array([r["cmd"] for r in cmd_rows], float)
    # wall -> sim clock via the IMU pairing
    t_cmd = np.interp(wall_cmd, wall_imu, t_imu)

    odom = None
    mav = ep / "mav.jsonl"
    if mav.exists():
        rows = []
        with open(mav) as stream:
            for line in stream:
                if '"ODOMETRY"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(r)
        if len(rows) > 200:
            odom = rows
    if require_odometry and odom is None:
        return None

    t_lo = t_imu[0]
    t_hi = t_imu[-1]
    if odom is not None:
        t_od = np.array([r["time_usec"] * 1e-6 for r in odom])
        keep = _longest_monotonic(t_od)
        odom = [odom[i] for i in keep]
        t_od = t_od[keep]
        t_lo = max(t_lo, t_od[0])
        t_hi = min(t_hi, t_od[-1])
    t_lo = max(t_lo, t_cmd[0])
    t_hi = min(t_hi, t_cmd[-1])
    if t_hi - t_lo < 5.0:
        return None
    grid = np.arange(t_lo, t_hi, 1.0 / hz)

    def interp_cols(tq, ts, X):
        return np.stack(
            [np.interp(tq, ts, X[:, c]) for c in range(X.shape[1])], axis=1
        )

    # commands are zero-order-held between packets
    idx = np.clip(np.searchsorted(t_cmd, grid, side="right") - 1, 0, None)
    cmd_grid = cmd[idx]
    gyro_grid = interp_cols(grid, t_imu, gyro)
    accel_grid = interp_cols(grid, t_imu, accel)

    pos = vel_world = quat = rates = None
    if odom is not None:
        pos_raw = np.array([r["pos"] for r in odom], float)
        vel_raw = np.array([r["vel"] for r in odom], float)
        quat_raw = np.array([r["quat_wxyz"] for r in odom], float)
        rate_raw = np.array(
            [r.get("body_rates", [0, 0, 0]) for r in odom], float
        )
        # Empirically verified on these captures (frame_probe): the NAIVE
        # (w,x,y,z) reading rotates body vel onto d(pos)/dt at 0.03 m/s rms.
        w, x, y, z = (
            quat_raw[:, 0], quat_raw[:, 1], quat_raw[:, 2], quat_raw[:, 3]
        )
        quat_xyzw = np.stack([x, y, z, w], axis=1)
        rot = Rotation.from_quat(quat_xyzw)
        vel_world_raw = rot.apply(vel_raw)

        pos = interp_cols(grid, t_od, pos_raw)
        vel_world = interp_cols(grid, t_od, vel_world_raw)
        rates = interp_cols(grid, t_od, rate_raw)
        # slerp quats onto grid
        from scipy.spatial.transform import Slerp
        # strictly increasing subsequence (clock jitter can repeat stamps)
        inc = np.zeros(len(t_od), bool)
        last = -np.inf
        for i, ti in enumerate(t_od):
            if ti > last + 1e-6:
                inc[i] = True
                last = ti
        slerp = Slerp(t_od[inc], Rotation.from_quat(quat_xyzw[inc]))
        quat = slerp(np.clip(grid, t_od[inc][0], t_od[inc][-1])).as_quat()

    return Episode(
        name=ep.name,
        t=grid,
        cmd=cmd_grid,
        gyro=gyro_grid,
        accel=accel_grid,
        pos=pos,
        vel_world=vel_world,
        quat_wb=quat,
        rates_body=rates,
    )
