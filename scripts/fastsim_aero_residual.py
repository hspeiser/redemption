"""Aero-residual audit: measured specific force vs surrogate prediction.

Uses full-record sessions (10Hz-era belief): IMU accel (specific force,
body frame), wire thrust from commands.jsonl, attitude+velocity from
pose tracks / steps.  Residual r = f_meas - [thrust(u) * axis_hat -
diag(drag) * v_body], binned by airspeed -- the measured correction the
surrogate needs in the 4-12 m/s racing band, and the direct test of the
known +4.6 m/s^2 z-aero once faster data exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel  # noqa: E402


def load_imu(session: Path) -> np.ndarray:
    from pymavlink import mavutil

    header = struct.Struct("<QHI")
    data = (session / "mavlink_rx.bin").read_bytes()
    parser = mavutil.mavlink.MAVLink(None)
    parser.robust_parsing = True
    rows = []
    with (session / "mavlink_index.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["message_type"] != "HIGHRES_IMU":
                continue
            offset = int(row["offset"])
            wall_ns, kind_len, raw_len = header.unpack_from(data, offset)
            start = offset + header.size + kind_len
            for m in parser.parse_buffer(data[start:start + raw_len]) or []:
                if m.get_type() == "HIGHRES_IMU":
                    rows.append((
                        float(wall_ns), m.xacc, m.yacc, m.zacc,
                        m.xgyro, m.ygyro, m.zgyro,
                    ))
    return np.asarray(rows, np.float64)


def load_thrust(session: Path) -> np.ndarray:
    rows = []
    for line in (session / "commands.jsonl").open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "rates":
            continue
        payload = r.get("payload")
        if not isinstance(payload, (list, tuple)) or len(payload) < 4:
            continue
        rows.append((float(r["wall_ns"]), float(payload[3])))
    return np.asarray(rows, np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=Path, nargs="+", required=True)
    parser.add_argument("--tracks", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path,
                        default=REPO / "data/fastsim_model.json")
    args = parser.parse_args()

    model = SurrogateModel.load(args.model)
    drag = np.asarray(model.drag_lin, float)

    f_meas_all, f_pred_all, spd_all, vbody_all = [], [], [], []
    axis_votes = []
    for session, track_path in zip(args.sessions, args.tracks):
        imu = load_imu(session)
        thrust = load_thrust(session)
        track = np.load(track_path)
        t_wall = track["wall_ns"].astype(float)
        order = np.argsort(t_wall)
        t_wall = t_wall[order]
        quat = track["quat_wxyz"][order]
        pos = track["pos"][order]
        sig = track["sigma"][order]
        # velocity: central difference of belief positions
        vel = np.gradient(pos, t_wall * 1e-9, axis=0)

        # hover-axis calibration: near-zero speed rows
        for i in range(0, len(imu), 997):
            wall = imu[i, 0]
            j = int(np.searchsorted(t_wall, wall))
            if not 0 < j < len(t_wall):
                continue
            if abs(t_wall[j] - wall) > 40e6 or sig[j] > 0.15:
                continue
            speed = float(np.linalg.norm(vel[j]))
            if speed < 0.7:
                f = imu[i, 1:4]
                axis_votes.append(f / (np.linalg.norm(f) + 1e-9))
        for i in range(len(imu)):
            wall = imu[i, 0]
            j = int(np.searchsorted(t_wall, wall))
            if not 0 < j < len(t_wall):
                continue
            if abs(t_wall[j] - wall) > 40e6 or sig[j] > 0.15:
                continue
            k = int(np.searchsorted(thrust[:, 0], wall))
            if not 0 < k < len(thrust):
                continue
            if abs(thrust[k, 0] - wall) > 80e6:
                continue
            u = float(thrust[k, 1])
            qw, qx, qy, qz = quat[j]
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            v_world = vel[j]
            v_body = R.T @ v_world
            f_meas_all.append(imu[i, 1:4])
            spd_all.append(float(np.linalg.norm(v_world)))
            vbody_all.append(v_body)
            thrust_mag = model.thrust_gain * u + model.thrust_quad * u * u
            f_pred_all.append((thrust_mag, v_body))

    axis = np.mean(axis_votes, axis=0)
    axis /= np.linalg.norm(axis) + 1e-9
    print(f"hover thrust axis (body): {np.round(axis, 3)} "
          f"(n={len(axis_votes)})")

    f_meas = np.asarray(f_meas_all)
    spd = np.asarray(spd_all)
    vb = np.asarray(vbody_all)
    f_pred = np.asarray([
        tm * axis - drag * v for (tm, v) in f_pred_all
    ])
    resid = f_meas - f_pred
    print(f"samples: {len(resid)}")
    print("\nspeed | n     | residual body x/y/z m/s^2 (median) | |r| p50")
    for lo in range(0, 13, 2):
        m = (spd >= lo) & (spd < lo + 2)
        if m.sum() < 50:
            continue
        med = np.median(resid[m], axis=0)
        mag = np.median(np.linalg.norm(resid[m], axis=1))
        print(f"{lo:2d}-{lo+2:2d}  | {m.sum():5d} | "
              f"({med[0]:+.2f}, {med[1]:+.2f}, {med[2]:+.2f}) | {mag:.2f}")
    # forward-axis drag check: residual along v_body vs speed
    with np.errstate(invalid="ignore"):
        vhat = vb / (np.linalg.norm(vb, axis=1, keepdims=True) + 1e-9)
    along = np.sum(resid * vhat, axis=1)
    print("\nspeed | residual along velocity (median m/s^2)")
    for lo in range(0, 13, 2):
        m = (spd >= lo) & (spd < lo + 2)
        if m.sum() >= 50:
            print(f"{lo:2d}-{lo+2:2d}: {np.median(along[m]):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
