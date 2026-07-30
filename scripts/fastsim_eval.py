"""Evaluate a fastsim PPO checkpoint: spawn-start full-lap completion.

The acceptance metric for the whole build: deterministic policy, all
envs starting parked at spawn, estimator noise ON, domain randomization
ON. Reports finish rate, lap times, and where failures happen.

    python scripts/fastsim_eval.py --ckpt data/fastsim_runs/ppo_v1/latest.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import (  # noqa: E402
    ACT_DIM,
    OBS_DIM,
    FastEnvConfig,
    FastVQ2Env,
    N_GATES,
)
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.rl.sac import GaussianActor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n-envs", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=str(REPO / "data" /
                                               "fastsim_model.json"))
    parser.add_argument("--map", default=str(REPO / "data" /
                                             "vq2_map_final.json"))
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--no-dr", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1600)
    args = parser.parse_args()
    device = torch.device(args.device)

    cfg = FastEnvConfig(
        n_envs=args.n_envs,
        random_start_frac=0.0,          # ALL spawn starts
        rate_gain_sign=1.0,
    )
    if args.no_noise:
        cfg.pos_noise_lo = cfg.pos_noise_hi = 0.0
    if args.no_dr:
        cfg.dr_thrust = (1.0, 1.0)
        cfg.dr_rate_gain = (1.0, 1.0)
        cfg.dr_rate_tau = (1.0, 1.0)
        cfg.dr_drag = (0.3, 0.3)
        cfg.act_delay_steps_max = 0
    model = SurrogateModel.load(args.model)
    env = FastVQ2Env(model, args.map, demo_states=None, config=cfg,
                     device=str(device))

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    obs_mean = ck["obs_mean"].to(device)
    obs_var = ck["obs_var"].to(device)
    print(f"checkpoint iter {ck.get('iter')}")

    n = args.n_envs
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    fail_gate = torch.full((n,), -1, dtype=torch.long, device=device)
    fail_kind = torch.zeros(n, dtype=torch.long, device=device)  # 1 hit
    lap_time = torch.zeros(n, device=device)
    steps_alive = torch.zeros(n, device=device)

    obs = env.observations()
    dt = 1.0 / cfg.control_hz
    with torch.no_grad():
        for _step in range(args.max_steps):
            o = torch.clamp(
                (obs - obs_mean) / torch.sqrt(obs_var + 1e-6), -8, 8
            )
            act = actor.deterministic(o)
            obs, _r, done, info = env.step(act)
            live = ~(finished | failed)
            steps_alive += live.float()
            newly_fin = info["finished"] & live
            lap_time = torch.where(newly_fin, steps_alive * dt, lap_time)
            finished |= newly_fin
            newly_fail = done & live & ~info["finished"]
            fail_gate = torch.where(
                newly_fail, info["target"], fail_gate
            )
            fail_kind = torch.where(
                newly_fail & info["hit"], torch.ones_like(fail_kind),
                fail_kind,
            )
            failed |= newly_fail
            if bool((finished | failed).all()):
                break

    fin = int(finished.sum())
    print(f"\nspawn-start eval over {n} randomized worlds "
          f"(noise={'off' if args.no_noise else 'on'}, "
          f"dr={'off' if args.no_dr else 'on'}):")
    print(f"  FINISHED FULL COURSE: {fin}/{n} = {100*fin/n:.1f}%")
    if fin:
        lt = lap_time[finished]
        print(f"  lap time: median {lt.median():.2f}s  "
              f"best {lt.min():.2f}s  p90 {lt.quantile(0.9):.2f}s")
    if int(failed.sum()):
        fg = fail_gate[failed].cpu().numpy()
        kinds = fail_kind[failed].cpu().numpy()
        hist = {}
        for g, k in zip(fg, kinds):
            key = f"g{g}" + ("/hit" if k == 1 else "")
            hist[key] = hist.get(key, 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
        print("  failures by gate:", dict(top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
