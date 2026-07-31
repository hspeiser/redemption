"""Live-vs-surrogate observation divergence for a policy artifact.

Rolls the policy deterministically in its own training env (rebuilt
from the artifact's embedded train_config) from a spawn start, then
compares each observation channel against a live-recorded obs sequence
step by step.  The first strongly diverging channel is the transfer
gap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.rl.sac import GaussianActor  # noqa: E402

CHANNEL_NAMES = (
    [f"rel{g}_{ax}" for g in range(3) for ax in "xyz"]
    + [f"tan{g}_{ax}" for g in range(3) for ax in "xyz"]
    + [f"vbody_{ax}" for ax in "xyz"]
    + [f"R0_{ax}" for ax in "xyz"] + [f"R1_{ax}" for ax in "xyz"]
    + [f"rate_{ax}" for ax in "xyz"]
    + [f"prev_a{k}" for k in range(4)]
    + [f"gate1hot_{k}" for k in range(17)]
    + ["sigma_conf", "progress"]
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--live-obs", required=True)
    parser.add_argument("--steps", type=int, default=53)
    args = parser.parse_args()

    art = torch.load(args.policy, map_location="cpu", weights_only=False)
    cfg_dict = art.get("train_config", {})
    actor = GaussianActor(53, 4)
    actor.load_state_dict(art["actor"])
    actor.eval()
    mean = art["obs_mean"]
    var = art["obs_var"]

    def act(obs):
        no = torch.clamp(
            (torch.tensor(obs, dtype=torch.float32) - mean)
            / torch.sqrt(var + 1e-6), -8, 8,
        )
        with torch.no_grad():
            m, _ = actor.distribution(no.unsqueeze(0))
        return torch.tanh(m)[0].numpy()

    cfg = FastEnvConfig(
        n_envs=4, random_start_frac=0.0, rate_gain_sign=1.0,
        reloc_events=True, demo_corridor_m=cfg_dict.get(
            "demo_corridor", 1.5),
        speed_cap_mps=cfg_dict.get("speed_cap", 8.0),
    )
    cfg.apply_vision10hz()
    if cfg_dict.get("fov_vision"):
        cfg.fov_vision = True
        cfg.coast_speed_diffuse = 0.005
        cfg.coast_speed_bias = 0.015
    if cfg_dict.get("act_delay_min") is not None:
        cfg.act_delay_steps_min = cfg_dict["act_delay_min"]
    model = SurrogateModel.load(
        REPO / "data" / Path(str(cfg_dict.get(
            "model", "data/fastsim_model_v2.json"))).name)
    demo_name = Path(str(cfg_dict.get(
        "demo_npz", "data/fastsim_demo_winner.npz"))).name
    demo = dict(np.load(REPO / "data" / demo_name))
    map_name = Path(str(cfg_dict.get(
        "map", "data/vq2_map_winner_train.json"))).name
    env = FastVQ2Env(model, str(REPO / "data" / map_name),
                     demo_states=demo, config=cfg, device="cpu")

    sim_obs = []
    obs = env.observations()
    for _ in range(args.steps):
        sim_obs.append(obs[0].numpy().copy())
        a = act(obs[0].numpy())
        actions = torch.tensor(
            np.tile(a, (4, 1)), dtype=torch.float32)
        obs, _r, _d, _i = env.step(actions)
    sim_obs = np.asarray(sim_obs)

    live = np.load(args.live_obs)["observation"][:args.steps]
    n = min(len(live), len(sim_obs))
    diff = np.abs(live[:n] - sim_obs[:n])
    scale = np.maximum(np.abs(sim_obs[:n]).mean(0), 0.05)
    rel = diff.mean(0) / scale
    order = np.argsort(-rel)
    print(f"steps compared: {n}")
    print("worst channels (mean |live - sim| / scale):")
    for k in order[:12]:
        name = CHANNEL_NAMES[k] if k < len(CHANNEL_NAMES) else f"ch{k}"
        print(f"  {name:14s} rel={rel[k]:6.2f} "
              f"live_mean={live[:n, k].mean():+.3f} "
              f"sim_mean={sim_obs[:n, k].mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
