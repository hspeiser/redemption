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
    parser.add_argument("--demo-npz", default=str(
        REPO / "data" / "fastsim_demo_states.npz"))
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--reloc-events", action="store_true",
                        help="structured coast-and-snap estimator error "
                             "(measured reloc statistics)")
    parser.add_argument("--demo-corridor", type=float, default=2.0)
    parser.add_argument("--speed-cap", type=float, default=16.0)
    parser.add_argument("--obstacles", default="")
    parser.add_argument("--no-dr", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--noise-era", choices=["3hz", "10hz"],
                        default="3hz")
    parser.add_argument("--fov-vision", action="store_true")
    parser.add_argument("--action-smoothness", type=float, default=None)
    parser.add_argument("--act-delay-min", type=int, default=None)
    parser.add_argument("--spawn-at-rest", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--bc-init", default=str(
        REPO / "data/vq2_sac_runs/gate3_nstep_v60b/20260730_161745"
               "/episode_0002.npz"),
        help="episode npz for backbone feedforward in residual mode")
    args = parser.parse_args()
    device = torch.device(args.device)

    cfg = FastEnvConfig(
        n_envs=args.n_envs,
        random_start_frac=0.0,          # ALL spawn starts
        rate_gain_sign=1.0,
        reloc_events=args.reloc_events,
        demo_corridor_m=args.demo_corridor,
        speed_cap_mps=args.speed_cap,
    )
    if args.noise_era == "10hz":
        cfg.apply_vision10hz()
    if args.fov_vision:
        cfg.fov_vision = True
        cfg.coast_speed_diffuse = 0.005
        cfg.coast_speed_bias = 0.015
    if args.action_smoothness is not None:
        cfg.action_smoothness = args.action_smoothness
    if args.act_delay_min is not None:
        cfg.act_delay_steps_min = args.act_delay_min
    if args.spawn_at_rest:
        cfg.spawn_at_rest = True
    if args.no_noise:
        cfg.pos_noise_lo = cfg.pos_noise_hi = 0.0
    if args.no_dr:
        cfg.dr_thrust = (1.0, 1.0)
        cfg.dr_rate_gain = (1.0, 1.0)
        cfg.dr_rate_tau = (1.0, 1.0)
        cfg.dr_drag = (0.3, 0.3)
        cfg.act_delay_steps_max = 0
    model = SurrogateModel.load(args.model)
    demo = None
    if args.demo_npz:
        raw_demo = dict(np.load(args.demo_npz, allow_pickle=False))
        if {"position", "velocity", "observation", "gate_index"} <= raw_demo.keys():
            # Live-training demos use descriptive field names and encode the
            # body-to-course attitude as the first two rotation columns in
            # the 53-D observation.  FastVQ2Env expects compact truth-state
            # keys.  Convert only those numeric arrays instead of forwarding
            # string metadata (source_episode/source_trace) into torch.
            from scipy.spatial.transform import Rotation

            observation = np.asarray(raw_demo["observation"], np.float32)
            first = observation[:, 21:24].astype(np.float64)
            first /= np.linalg.norm(first, axis=1, keepdims=True) + 1e-9
            second = observation[:, 24:27].astype(np.float64)
            second -= first * np.sum(first * second, axis=1, keepdims=True)
            second /= np.linalg.norm(second, axis=1, keepdims=True) + 1e-9
            third = np.cross(first, second)
            matrix = np.stack([first, second, third], axis=2)
            quat_xyzw = Rotation.from_matrix(matrix).as_quat()
            demo = {
                "pos": np.asarray(raw_demo["position"], np.float32),
                "vel": np.asarray(raw_demo["velocity"], np.float32),
                "quat": np.asarray(
                    quat_xyzw[:, [3, 0, 1, 2]], np.float32
                ),
                "gate": np.asarray(raw_demo["gate_index"], np.float32),
            }
        else:
            demo = raw_demo
    backbone = None
    if args.residual:
        from aigp.fastsim.refctl import load_winner_backbone
        cfg.residual_scale = args.residual_scale
        backbone = load_winner_backbone(
            args.demo_npz, args.bc_init, cfg.n_envs, device=str(device)
        )
    env = FastVQ2Env(model, args.map, demo_states=demo, config=cfg,
                     device=str(device),
                     obstacles_path=args.obstacles or None,
                     backbone=backbone)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    actor.load_state_dict(ck["actor"])
    actor.eval()
    if ck.get("kind") == "vq2_residual_sac":
        obs_mean = torch.as_tensor(
            ck["observation_mean"], device=device
        )
        obs_scale = torch.as_tensor(
            ck["observation_std"], device=device
        ).clamp_min(1e-6)
        print(f"SAC checkpoint updates {ck.get('updates')}")
    else:
        obs_mean = ck["obs_mean"].to(device)
        obs_scale = torch.sqrt(ck["obs_var"].to(device) + 1e-6)
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
                (obs - obs_mean) / obs_scale, -8, 8
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
            for code, key in ((1, "hit"), (2, "off"), (3, "overspeed"),
                              (4, "timeout")):
                fail_kind = torch.where(
                    newly_fail & info[key],
                    torch.full_like(fail_kind, code), fail_kind,
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
        names = {0: "?", 1: "hit", 2: "off", 3: "overspeed", 4: "timeout"}
        hist = {}
        for g, k in zip(fg, kinds):
            key = f"g{g}/{names.get(int(k), '?')}"
            hist[key] = hist.get(key, 0) + 1
        top = sorted(hist.items(), key=lambda kv: -kv[1])[:8]
        print("  failures by gate:", dict(top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
