"""Fit the surrogate dynamics model from VQ1 odometry episodes.

    .venv-train\\Scripts\\python.exe scripts\\fastsim_fit.py \
        --captures "C:\\...\\outputs\\captures" --max-episodes 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.data import load_episode  # noqa: E402
from aigp.fastsim.sysid import (  # noqa: E402
    SurrogateModel,
    fit_rate_loop,
    fit_translation,
    rollout,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captures",
        default=r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
                r"\ai-grand-prix\outputs\captures",
    )
    parser.add_argument("--max-episodes", type=int, default=12)
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument(
        "--out", default=str(REPO / "data" / "fastsim_model.json")
    )
    args = parser.parse_args()

    episodes = []
    caps = sorted(Path(args.captures).glob("rc_*"), reverse=True)
    for ep_dir in caps:
        ep = load_episode(ep_dir, hz=args.hz, require_odometry=True)
        if ep is None:
            continue
        speed = np.linalg.norm(ep.vel_world, axis=1)
        if speed.max() < 3.0:
            continue          # parked/hover logs carry no dynamics info
        # frame self-check: d(pos)/dt must match world velocity
        dp = np.diff(ep.pos, axis=0) * args.hz
        verr = np.sqrt(np.mean(np.sum(
            (dp - ep.vel_world[:-1]) ** 2, axis=1
        )))
        print(f"{ep.name}: {len(ep.t)} rows, vmax {speed.max():5.1f} m/s, "
              f"dpos-vs-vel rms {verr:6.3f} m/s")
        if verr > 1.0:
            print("   frame check FAILED, skipping")
            continue
        episodes.append(ep)
        if len(episodes) >= args.max_episodes:
            break
    if len(episodes) < 2:
        print("not enough usable odometry episodes")
        return 1
    holdout = episodes[-1]
    fit_eps = episodes[:-1]
    print(f"\nfitting on {len(fit_eps)} episodes, holdout {holdout.name}")

    gains, taus, delay, rate_report = fit_rate_loop(fit_eps)
    print(f"rate loop: gain {np.round(gains, 3)}  tau {np.round(taus, 4)}s "
          f"delay {delay*1000:.0f}ms")
    trans, trans_report = fit_translation(fit_eps)
    print(f"translation: thrust {trans['thrust_gain']:.2f}*u "
          f"{trans['thrust_quad']:+.2f}*u^2; "
          f"drag {np.round(trans['drag_lin'], 4)}; "
          f"g {np.round(trans['g_vec'], 3)} "
          f"(f_z fit rms {trans_report['accel_fit_rms_mps2']:.3f} m/s^2, "
          f"g mad {np.round(trans_report['g_mad'], 3)})")

    model = SurrogateModel(
        rate_gain=gains.tolist(),
        rate_tau=taus.tolist(),
        rate_delay=delay,
        thrust_gain=trans["thrust_gain"],
        thrust_quad=trans["thrust_quad"],
        drag_lin=trans["drag_lin"],
        g_vec=trans["g_vec"],
        hz=args.hz,
    )
    model.save(args.out)
    print(f"wrote {args.out}")

    print("\nvalidation rollouts (logged commands -> position error):")
    for ep in [*fit_eps[:3], holdout]:
        tag = "HOLDOUT" if ep is holdout else "fit"
        for t0 in np.arange(ep.t[0] + 2.0, ep.t[-1] - 6.0,
                            max((ep.t[-1] - ep.t[0]) / 4.0, 6.0)):
            r = rollout(model, ep, float(t0), 5.0)
            ra = rollout(model, ep, float(t0), 5.0,
                         attitude_from_truth=True)
            if r and ra:
                print(f"  [{tag}] {ep.name} t0={r['t0']:7.1f} 5s: "
                      f"open rmse {r['pos_rmse_m']:6.2f}m | "
                      f"true-att rmse {ra['pos_rmse_m']:6.2f}m "
                      f"final {ra['pos_final_err_m']:6.2f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
