"""Fly the fastsim-trained PPO policy in the REAL simulator, eval-only.

Construction mirrors scripts/train_vq2_sac_live.py exactly (v29+ process
isolation defaults that produced 2 stale episodes in 500). DO NOT run
while the live SAC trainer or a manual session owns the sim.

Dry run (no sim, no sockets -- verifies the policy pipeline end to end
against logged observations):

    python scripts/eval_vq2_ppo_live.py --dry-run

Real flight (sim must be free):

    python scripts/eval_vq2_ppo_live.py --episodes 5 \
        --policy data/models/vq2_ppo_conservative.pt
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

from aigp.rl.sac import GaussianActor  # noqa: E402


def load_policy(path: str):
    art = torch.load(path, map_location="cpu", weights_only=False)
    actor = GaussianActor(art["obs_dim"], art["act_dim"])
    actor.load_state_dict(art["actor"])
    actor.eval()
    mean = art["obs_mean"].numpy().astype(np.float32)
    std = np.sqrt(art["obs_var"].numpy().astype(np.float32) + 1e-6)

    def act(observation: np.ndarray) -> np.ndarray:
        o = np.clip((observation - mean) / std, -8.0, 8.0)
        with torch.no_grad():
            a = actor.deterministic(
                torch.from_numpy(o[None].astype(np.float32))
            )[0].numpy()
        return a

    return act, art


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--policy",
        default=str(REPO / "data" / "models" / "vq2_ppo_conservative.pt"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--map", type=Path,
        default=REPO / "data" / "vq2_map_final_live.json",
    )
    parser.add_argument(
        "--primary", type=Path,
        default=REPO / "data/models/gatenet_v7_best.pt",
    )
    parser.add_argument(
        "--refiner", type=Path,
        default=REPO / "data/models/gatenet_v10strict_ep0.pt",
    )
    parser.add_argument(
        "--crop", type=Path,
        default=REPO / "data/models/crop_gatenet_v11crop_ep13.pt",
    )
    parser.add_argument(
        "--proposal", type=Path,
        default=REPO / "data/models/gatepose_v5vq2b_best.pt",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=REPO / "data/calib/calib.json",
    )
    parser.add_argument("--mav-port", type=int, default=14550)
    parser.add_argument("--camera-port", type=int, default=5600)
    # the isolation configuration that fixed real-time health (v29+)
    parser.add_argument("--vision-hz", type=float, default=2.0)
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument("--max-vision-result-age", type=float, default=1.0)
    parser.add_argument("--vision-process-isolation", default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--vision-worker-threads", type=int, default=8)
    parser.add_argument("--vision-worker-affinity", default="0x03FC")
    parser.add_argument("--cpu-affinity", default="0xC000")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--log", default=str(REPO / "data" / "ppo_live_eval.jsonl")
    )
    args = parser.parse_args()

    act, art = load_policy(args.policy)
    print(f"policy: {args.policy} (source iter {art.get('source_iter')})")

    if args.dry_run:
        demo = np.load(REPO / "data" / "vq2_sac_clean_demo.npz")
        obs = demo["observation"]
        acts = demo["action"]
        pred = np.stack([act(o) for o in obs])
        for axis, name in enumerate(("roll", "pitch", "yaw", "thrust")):
            c = np.corrcoef(pred[:, axis], acts[:, axis])[0, 1]
            sat = float(np.mean(np.abs(pred[:, axis]) > 0.98))
            print(f"  {name:6s}: corr(policy, Henry) {c:+.3f}  "
                  f"|a|>0.98 frac {sat:.3f}  "
                  f"mean|a| {np.abs(pred[:, axis]).mean():.3f}")
        finite = np.isfinite(pred).all()
        print(f"  all outputs finite: {finite}")
        print("DRY RUN OK -- pipeline verified, no sim contact")
        return 0

    # ---- real flight ----
    from scripts.train_vq2_sac_live import (
        acquire_single_instance_guard,
        set_process_cpu_affinity,
    )
    from aigp.mavlink_io import MavIO
    from aigp.rl.vq2_env import VQ2EnvConfig, VQ2LiveEnv
    from aigp.vision_io import VisionRX
    from aigp.vq2_live_localizer import LiveVQ2Localizer

    guard = acquire_single_instance_guard()
    set_process_cpu_affinity(args.cpu_affinity)
    torch.set_num_threads(args.torch_threads)

    mavlink = MavIO(port=args.mav_port)
    vision = VisionRX(port=args.camera_port)
    localizer = LiveVQ2Localizer(
        mavlink=mavlink,
        vision=vision,
        map_path=args.map,
        primary_checkpoint=args.primary,
        refine_checkpoint=args.refiner,
        crop_checkpoint=args.crop,
        proposal_checkpoint=args.proposal,
        calibration_path=args.calibration,
        async_interval_s=1.0 / args.vision_hz,
        max_async_result_age_s=args.max_vision_result_age,
        dense_device=args.vision_device,
        dense_process_isolation=args.vision_process_isolation,
        dense_worker_threads=args.vision_worker_threads,
        dense_worker_affinity=args.vision_worker_affinity,
    )
    environment = VQ2LiveEnv(mavlink, localizer, VQ2EnvConfig())

    results = []
    try:
        for ep_i in range(args.episodes):
            observation, info = environment.reset()
            done = False
            steps = 0
            ep_reward = 0.0
            step_info = {}
            while not done:
                action = act(observation)
                observation, reward, term, trunc, step_info = \
                    environment.step(action)
                ep_reward += float(reward)
                steps += 1
                done = term or trunc
            row = {
                "episode": ep_i,
                "reward": round(ep_reward, 1),
                "steps": steps,
                "gate_reached": int(step_info.get("target", -1)),
                "finished": bool(step_info.get("finished")),
                "failure": step_info.get("failure"),
                "sim_time_s": step_info.get("sim_time_s"),
            }
            results.append(row)
            print("EP", json.dumps(row))
            with open(args.log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            environment.park_after_episode()
            time.sleep(1.0)
    finally:
        environment.shutdown_to_spawn()
    n_finished = sum(1 for r in results if r["finished"])
    print(f"\n{n_finished}/{len(results)} finished; gates: "
          f"{[r['gate_reached'] for r in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
