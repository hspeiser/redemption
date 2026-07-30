"""Big physics validation: sweep ALL odometry episodes in the captures
folder. For each: frame self-check, per-episode re-fit of the rate loop
and thrust balance (parameter STABILITY across flights = the physics is
one consistent law), and model rollout errors at several horizons.

    python scripts/fastsim_validate_big.py --limit 120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.data import load_episode  # noqa: E402
from aigp.fastsim.sysid import (  # noqa: E402
    SurrogateModel,
    fit_rate_loop,
    rollout,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captures",
        default=r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
                r"\ai-grand-prix\outputs\captures",
    )
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument(
        "--model", default=str(REPO / "data" / "fastsim_model.json")
    )
    parser.add_argument(
        "--out", default=str(REPO / "data" / "fastsim_bigval.json")
    )
    args = parser.parse_args()
    model = SurrogateModel.load(args.model)

    rows = []
    n_checked = 0
    for ep_dir in sorted(Path(args.captures).glob("rc_*"), reverse=True):
        if n_checked >= args.limit:
            break
        try:
            ep = load_episode(ep_dir, hz=100.0, require_odometry=True)
        except Exception as error:
            print(f"{ep_dir.name}: loader error {error!r}")
            continue
        if ep is None:
            continue
        n_checked += 1
        speed = np.linalg.norm(ep.vel_world, axis=1)
        if speed.max() < 3.0:
            continue
        dp = np.diff(ep.pos, axis=0) * 100.0
        frame_rms = float(np.sqrt(np.mean(np.sum(
            (dp - ep.vel_world[:-1]) ** 2, axis=1
        ))))
        if frame_rms > 1.0:
            print(f"{ep.name}: frame check FAIL ({frame_rms:.2f})")
            continue
        # per-episode rate refit (stability check)
        try:
            gains, taus, _d, _r = fit_rate_loop([ep])
        except Exception:
            gains = taus = np.full(3, np.nan)
        # thrust balance on flying rows
        rot = Rotation.from_quat(ep.quat_wb)
        dv = savgol_filter(ep.vel_world, 25, 3, deriv=1, delta=0.01,
                           axis=0)
        a_body = rot.inv().apply(dv - np.asarray(model.g_vec))
        m = (speed > 3.0) & (ep.cmd[:, 3] > 0.05)
        thrust_res = float(np.mean(
            a_body[m, 2] + model.thrust_gain * ep.cmd[m, 3]
        )) if m.sum() > 100 else np.nan
        # rollouts
        flying = np.flatnonzero(speed > 4.0)
        r_short, r_med = [], []
        if len(flying) > 300:
            for s in np.linspace(flying[0], flying[-1] - 300, 5).astype(int):
                r1 = rollout(model, ep, float(ep.t[s]), 1.0,
                             attitude_from_truth=True)
                r2 = rollout(model, ep, float(ep.t[s]), 3.0,
                             attitude_from_truth=True)
                if r1:
                    r_short.append(r1["pos_rmse_m"])
                if r2:
                    r_med.append(r2["pos_rmse_m"])
        row = {
            "name": ep.name,
            "vmax": float(speed.max()),
            "frame_rms": frame_rms,
            "rate_gain": [round(float(g), 3) for g in gains],
            "rate_tau": [round(float(t), 4) for t in taus],
            "thrust_residual_mps2": (
                round(thrust_res, 2) if np.isfinite(thrust_res) else None
            ),
            "roll1s_rmse": (
                round(float(np.median(r_short)), 2) if r_short else None
            ),
            "roll3s_rmse": (
                round(float(np.median(r_med)), 2) if r_med else None
            ),
        }
        rows.append(row)
        print(json.dumps(row))
    Path(args.out).write_text(json.dumps(rows, indent=1))

    # summary
    def col(key):
        return np.array([
            r[key] for r in rows if r.get(key) is not None
        ], float)

    print(f"\n=== SUMMARY over {len(rows)} odometry episodes ===")
    g0 = np.array([r["rate_gain"][0] for r in rows
                   if np.isfinite(r["rate_gain"][0])])
    g1 = np.array([r["rate_gain"][1] for r in rows
                   if np.isfinite(r["rate_gain"][1])])
    g2 = np.array([r["rate_gain"][2] for r in rows
                   if np.isfinite(r["rate_gain"][2])])
    for name, arr in (("roll", g0), ("pitch", g1), ("yaw", g2)):
        if len(arr):
            print(f"rate gain {name}: median {np.median(arr):+.3f}  "
                  f"iqr {np.percentile(arr, 25):+.3f}"
                  f"..{np.percentile(arr, 75):+.3f}")
    tr = col("thrust_residual_mps2")
    if len(tr):
        print(f"thrust residual: median {np.median(tr):+.2f} m/s^2  "
              f"iqr {np.percentile(tr, 25):+.2f}"
              f"..{np.percentile(tr, 75):+.2f}")
    for key in ("roll1s_rmse", "roll3s_rmse"):
        arr = col(key)
        if len(arr):
            print(f"{key}: median {np.median(arr):.2f} m  "
                  f"p90 {np.percentile(arr, 90):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
