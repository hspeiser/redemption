"""Fly the fastsim-trained PPO policy in the REAL simulator, eval-only.

DO NOT run while the live SAC trainer or a manual session owns the sim.
Uses the live stack unchanged: MavIO + LiveVQ2Localizer + VQ2LiveEnv.
The policy is the full controller (no teacher, no residual scaffolding).

    python scripts/eval_vq2_ppo_live.py --episodes 5 \
        --policy data/models/vq2_ppo_policy.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.mavlink_io import MavIO  # noqa: E402
from aigp.rl.sac import GaussianActor  # noqa: E402
from aigp.rl.vq2_env import VQ2EnvConfig, VQ2LiveEnv  # noqa: E402
from aigp.vq2_live_localizer import LiveVQ2Localizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--policy",
        default=str(REPO / "data" / "models" / "vq2_ppo_policy.pt"),
    )
    parser.add_argument(
        "--map", default=str(REPO / "data" / "vq2_map_final.json")
    )
    parser.add_argument("--mav-port", type=int, default=14550)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument(
        "--log", default=str(REPO / "data" / "ppo_live_eval.jsonl")
    )
    args = parser.parse_args()

    art = torch.load(args.policy, map_location="cpu", weights_only=False)
    actor = GaussianActor(art["obs_dim"], art["act_dim"])
    actor.load_state_dict(art["actor"])
    actor.eval()
    obs_mean = art["obs_mean"].numpy()
    obs_std = np.sqrt(art["obs_var"].numpy() + 1e-6)
    print(f"policy from iter {art.get('source_iter')}")

    mav = MavIO(port=args.mav_port)
    mav.start()
    localizer = LiveVQ2Localizer(
        mavlink=mav, map_path=args.map, camera_port=args.camera_port
    )
    env = VQ2LiveEnv(mav, localizer, VQ2EnvConfig())

    results = []
    try:
        for ep_i in range(args.episodes):
            obs, info = env.reset()
            done = False
            steps = 0
            ep_reward = 0.0
            while not done:
                o = np.clip((obs - obs_mean) / obs_std, -8, 8)
                with torch.no_grad():
                    act = actor.deterministic(
                        torch.from_numpy(o[None].astype(np.float32))
                    )[0].numpy()
                obs, reward, term, trunc, step_info = env.step(act)
                ep_reward += reward
                steps += 1
                done = term or trunc
            row = {
                "episode": ep_i,
                "reward": ep_reward,
                "steps": steps,
                "gate_reached": step_info["target"],
                "finished": step_info["finished"],
                "failure": step_info["failure"],
            }
            results.append(row)
            print("EP", json.dumps(row))
            with open(args.log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            env.park_after_episode()
            time.sleep(1.0)
    finally:
        env.shutdown_to_spawn()
    n_fin = sum(1 for r in results if r["finished"])
    print(f"\n{n_fin}/{len(results)} finished; gates: "
          f"{[r['gate_reached'] for r in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
