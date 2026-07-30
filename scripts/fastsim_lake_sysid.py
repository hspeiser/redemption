"""Bulk dynamics identification over the VQ2 run lake (target build).

No odometry needed: per run, fit the cmd->gyro rate loop (gain/tau per
axis, one-step discrete form) and measure the hover/thrust balance from
specific force on low-rotation segments. Emits one JSON row per run and
a distribution summary -- the direct check that the training DR ranges
cover the exact simulator build the policy will fly in.

    python scripts/fastsim_lake_sysid.py --lake ~/aigp/lake --limit 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.data import load_episode  # noqa: E402


def fit_run(ep) -> dict | None:
    gyro_mag = np.linalg.norm(ep.gyro, axis=1)
    if gyro_mag.max() < 0.5:
        return None                      # never flew
    row = {"name": ep.name, "rows": len(ep.t)}
    hz = 1.0 / (ep.t[1] - ep.t[0])
    gains, taus = [], []
    for axis in range(3):
        u = ep.cmd[:, axis]
        w = ep.gyro[:, axis]
        if np.std(u) < 5e-3:
            gains.append(np.nan)
            taus.append(np.nan)
            continue
        A = np.stack([w[:-1], u[:-1]], axis=1)
        b = w[1:]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        a, bb = coef
        a = float(np.clip(a, 0.02, 0.999))
        taus.append(-1.0 / (hz * np.log(a)))
        gains.append(float(bb / (1.0 - a)))
    row["gain"] = [round(g, 3) if np.isfinite(g) else None for g in gains]
    row["tau"] = [round(t, 4) if np.isfinite(t) else None for t in taus]
    # thrust balance: low-rotation, thrusting rows; specific force
    # magnitude ~ thrust accel (drag small at low speed)
    m = (gyro_mag < 0.6) & (ep.cmd[:, 3] > 0.15)
    if m.sum() > 100:
        f = np.linalg.norm(ep.accel[m], axis=1)
        u = ep.cmd[m, 3]
        coef = np.polyfit(u, f, 1)
        row["thrust_slope"] = round(float(coef[0]), 2)
        row["thrust_icpt"] = round(float(coef[1]), 2)
        row["thrust_n"] = int(m.sum())
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake", required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--out", default="data/fastsim_lake_sysid.json")
    args = parser.parse_args()
    lake = Path(args.lake).expanduser()
    run_dirs = sorted(
        d for d in lake.iterdir()
        if d.is_dir() and (d / "imu.jsonl").exists()
        and (d / "cmd.jsonl").exists()
    )
    print(f"{len(run_dirs)} candidate runs with imu+cmd")
    rows = []
    for d in run_dirs[: args.limit]:
        try:
            ep = load_episode(d, hz=100.0, require_odometry=False)
        except Exception as error:
            continue
        if ep is None:
            continue
        row = fit_run(ep)
        if row:
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"  {len(rows)} runs fitted...")
    Path(args.out).write_text(json.dumps(rows, indent=1))
    print(f"fitted {len(rows)} runs -> {args.out}")

    for axis, name in enumerate(("roll", "pitch", "yaw")):
        arr = np.array([
            r["gain"][axis] for r in rows
            if r["gain"][axis] is not None
            and -6 < r["gain"][axis] < 6
        ])
        if len(arr):
            print(f"gain {name}: n={len(arr)} median {np.median(arr):+.3f} "
                  f"iqr {np.percentile(arr, 25):+.3f}"
                  f"..{np.percentile(arr, 75):+.3f} "
                  f"p5..p95 {np.percentile(arr, 5):+.3f}"
                  f"..{np.percentile(arr, 95):+.3f}")
    sl = np.array([
        r["thrust_slope"] for r in rows
        if r.get("thrust_slope") is not None
        and 0 < r["thrust_slope"] < 100
    ])
    if len(sl):
        print(f"thrust slope |f| vs u: n={len(sl)} "
              f"median {np.median(sl):.2f} "
              f"iqr {np.percentile(sl, 25):.2f}"
              f"..{np.percentile(sl, 75):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
