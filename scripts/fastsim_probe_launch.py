"""Print the trained policy's actual behavior for the first seconds of a
spawn start: position, speed, wire thrust, target. Diagnoses WHY launch
fails (sit / tumble / overshoot / mis-aim).

    python scripts/fastsim_probe_launch.py --ckpt .../latest.pt
"""

from __future__ import annotations

import argparse
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
)
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.rl.sac import GaussianActor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default=str(REPO / "data" /
                                               "fastsim_model.json"))
    parser.add_argument("--map", default=str(REPO / "data" /
                                             "vq2_map_final.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    device = torch.device(args.device)
    cfg = FastEnvConfig(
        n_envs=2, random_start_frac=0.0, rate_gain_sign=1.0,
        pos_noise_lo=0.0, pos_noise_hi=0.0,
        dr_thrust=(1.0, 1.0), dr_rate_gain=(1.0, 1.0),
        dr_rate_tau=(1.0, 1.0), dr_drag=(0.3, 0.3),
        act_delay_steps_max=0,
    )
    demo_path = REPO / "data" / "fastsim_demo_states.npz"
    demo = dict(np.load(demo_path)) if demo_path.exists() else None
    env = FastVQ2Env(SurrogateModel.load(args.model), args.map,
                     demo_states=demo, config=cfg, device=str(device))
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    mean = ck["obs_mean"].to(device)
    var = ck["obs_var"].to(device)
    obs = env.observations()
    print("step |    x      y      z  | speed | thr_a | pitch_a roll_a "
          "| tgt gate_dist")
    with torch.no_grad():
        for s in range(args.steps):
            o = torch.clamp((obs - mean) / torch.sqrt(var + 1e-6), -8, 8)
            act = actor.deterministic(o)
            obs, r, done, info = env.step(act)
            if s % 6 == 0:
                p = env.p[0].cpu().numpy()
                v = float(torch.linalg.norm(env.v[0]))
                a = act[0].cpu().numpy()
                tgt = int(env.target[0])
                gd = float(torch.linalg.norm(
                    env.gate_pos[min(tgt, 16)] - env.p[0]
                ))
                print(f"{s:4d} | {p[0]:6.2f} {p[1]:6.2f} {p[2]:6.2f} | "
                      f"{v:5.2f} | {a[3]:+5.2f} | {a[1]:+6.2f} {a[0]:+6.2f} "
                      f"| {tgt:2d}  {gd:5.1f}m")
            if bool(done[0]):
                print(f"   done at step {s}: hit={bool(info['hit'][0])} "
                      f"off={bool(info['off'][0])} "
                      f"overspeed={bool(info['overspeed'][0])} "
                      f"timeout={bool(info['timeout'][0])}")
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
