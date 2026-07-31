"""Backbone-only sanity: run RefController with zero residual in the
surrogate and report finish rate + failure map.  Also used to calibrate
feedback sign/gain choices before residual training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.refctl import load_winner_backbone  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402


def run(lat_sign: float, n_envs: int, steps: int, device: str,
        args) -> dict:
    cfg = FastEnvConfig(
        n_envs=n_envs, random_start_frac=0.0, rate_gain_sign=1.0,
        reloc_events=True, demo_corridor_m=2.5, speed_cap_mps=8.0,
        spawn_at_rest=True,
    )
    cfg.apply_vision10hz()
    cfg.fov_vision = True
    cfg.act_delay_steps_min = 1
    cfg.residual_scale = 0.0
    backbone = load_winner_backbone(
        args.demo_npz, args.episode_npz, n_envs, device=device,
        lat_pos_gain=abs(args.lat_gain) * lat_sign,
        lat_vel_gain=abs(args.lat_vel_gain) * lat_sign,
    )
    model = SurrogateModel.load(args.model)
    demo = dict(np.load(args.demo_npz))
    env = FastVQ2Env(model, args.map, demo_states=demo, config=cfg,
                     device=device, backbone=backbone)
    zeros = torch.zeros(n_envs, 4, device=device)
    fin = hit = off = to = eps = 0
    obs = env.observations()
    for _ in range(steps):
        obs, r, d, info = env.step(zeros)
        fin += int(info["finished"].sum())
        hit += int(info["hit"].sum())
        eps += int(d.sum())
    return {"finish": fin, "hit": hit, "episodes": eps}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(
        REPO / "data/fastsim_model_v2.json"))
    parser.add_argument("--map", default=str(
        REPO / "data/vq2_map_winner_train.json"))
    parser.add_argument("--demo-npz", default=str(
        REPO / "data/fastsim_demo_winner.npz"))
    parser.add_argument("--episode-npz", default=str(
        REPO / "data/vq2_sac_runs/gate3_nstep_v60b/20260730_161745"
               "/episode_0002.npz"))
    parser.add_argument("--lat-gain", type=float, default=0.35)
    parser.add_argument("--lat-vel-gain", type=float, default=0.12)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for sign in (+1.0, -1.0):
        out = run(sign, args.n_envs, args.steps, args.device, args)
        rate = out["finish"] / max(out["episodes"], 1)
        print(f"lat_sign {sign:+.0f}: {out}  finish/ep={rate:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
