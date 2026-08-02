"""Fit the body-rate loop directly from full-record VQ2 sessions.

The full-session archive stores exact transmitted rate commands and raw
HIGHRES_IMU packets.  Align those streams by wall timestamp and fit

    gyro[t+1] = a * gyro[t] + b * command[t] + c

per axis.  The steady-state gain is b/(1-a), and the time constant is
-dt/log(a).  This needs no absolute pose and therefore remains valid in VQ2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from scripts.fastsim_aero_residual import load_imu  # noqa: E402


def load_rates(session: Path) -> tuple[np.ndarray, np.ndarray]:
    wall, command = [], []
    with (session / "commands.jsonl").open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = row.get("payload")
            if row.get("kind") != "rates" or not isinstance(payload, list):
                continue
            if len(payload) < 4:
                continue
            wall.append(int(row["wall_ns"]))
            command.append(payload[:3])
    return np.asarray(wall, np.int64), np.asarray(command, np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--trim-quantile", type=float, default=0.90,
        help="Refit after dropping the largest one-step residuals.",
    )
    args = parser.parse_args()

    features = [[], [], []]
    targets = [[], [], []]
    sample_dt = []
    rows_used = 0
    for session in args.sessions:
        imu = load_imu(session)
        command_wall, command = load_rates(session)
        if len(imu) < 100 or len(command) < 20:
            print(f"skip {session}: insufficient IMU/command rows")
            continue
        imu_wall = imu[:, 0].astype(np.int64)
        index = np.clip(
            np.searchsorted(command_wall, imu_wall, side="right") - 1,
            0,
            len(command_wall) - 1,
        )
        aligned_command = command[index]
        gyro = imu[:, 4:7]
        dt = np.diff(imu_wall) * 1e-9
        command_age = (imu_wall[:-1] - command_wall[index[:-1]]) * 1e-9
        valid = (
            (dt > 0.004)
            & (dt < 0.020)
            & (command_age >= 0.0)
            & (command_age < 0.10)
        )
        sample_dt.extend(dt[valid])
        rows_used += int(valid.sum())
        for axis in range(3):
            features[axis].append(np.column_stack([
                gyro[:-1, axis][valid],
                aligned_command[:-1, axis][valid],
                np.ones(int(valid.sum())),
            ]))
            targets[axis].append(gyro[1:, axis][valid])
        print(f"{session.name}: {int(valid.sum())} aligned transitions")

    if rows_used < 1000:
        raise RuntimeError(f"only {rows_used} usable transitions")
    dt = float(np.median(sample_dt))
    gains, taus = [], []
    for axis in range(3):
        x = np.concatenate(features[axis])
        y = np.concatenate(targets[axis])
        coefficient, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ coefficient
        threshold = np.quantile(np.abs(residual), args.trim_quantile)
        keep = np.abs(residual) <= threshold
        coefficient, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
        a, b, bias = map(float, coefficient)
        a = float(np.clip(a, 0.001, 0.9999))
        gain = b / (1.0 - a)
        tau = -dt / np.log(a)
        gains.append(gain)
        taus.append(tau)
        print(
            f"axis {axis}: gain={gain:+.4f} tau={tau:.5f}s "
            f"bias={bias:+.6f} n={int(keep.sum())}"
        )

    model = SurrogateModel.load(args.base_model)
    model.rate_gain = gains
    model.rate_tau = taus
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"wrote {args.out} from {rows_used} aligned transitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
