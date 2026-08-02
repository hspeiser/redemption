"""Audit one g0-g4 controller candidate on paired randomized worlds.

Unlike the optimizer's nominal evaluation, this preserves learned dynamics,
plant variation, and estimator realism.  Intentional velocity impulses are an
independent switch so timed qualification is not conflated with disturbance
training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aigp.fastsim.env import ACT_DIM
from aigp.fastsim.sysid import SurrogateModel
from aigp.fastsim.worldmodel import ResidualEnsemble
from aigp.rl.sac import GaussianActor
from scripts.optimize_vq2_g0g4_worldmodel import (
    decoded_demo,
    evaluate,
    release_states,
)


def candidate_theta(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    values = (
        payload["leads"]
        + payload["thrust_scales"]
        + payload["velocity_scales"]
        + payload["trajectory_blends"]
        + payload.get("lateral_offsets_m", [0.0] * 5)
        + payload.get("vertical_offsets_m", [0.0] * 5)
        + payload.get("rate_scales", [1.0] * 5)
    )
    theta = np.asarray(values, np.float64)[None]
    if theta.shape != (1, 35):
        raise ValueError(f"candidate must decode to shape (1, 35), got {theta.shape}")
    return theta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260842)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aleatoric-scale", type=float, default=1.25)
    parser.add_argument("--impulse-rate-hz", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument("--geometry-limit", type=float, default=0.35)
    parser.add_argument("--tier-target", type=float, default=8.7)
    parser.add_argument("--multigate-vision", action="store_true")
    parser.add_argument("--live-estimator-realism", action="store_true")
    parser.add_argument("--vision-outcome-model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.worlds < 1:
        parser.error("--worlds must be positive")

    model = SurrogateModel.load(args.model)
    controller_model = SurrogateModel.load(
        args.controller_model or args.model
    )
    checkpoint = torch.load(
        args.actor, map_location=args.device, weights_only=False
    )
    actor = GaussianActor(53, ACT_DIM).to(args.device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    obs_mean = checkpoint["obs_mean"].to(args.device)
    obs_var = checkpoint["obs_var"].to(args.device)
    demo_states = decoded_demo(args.demo, args.map)
    spawn_states = release_states(args.dataset)
    theta = candidate_theta(args.candidate)

    reports = []
    metadata = []
    for ensemble_path in args.ensemble:
        ensemble, ensemble_metadata = ResidualEnsemble.load(
            ensemble_path, args.device
        )
        ensemble.eval()
        torch.manual_seed(args.seed)
        report = evaluate(
            theta,
            worlds=args.worlds,
            model=model,
            ensemble=ensemble,
            demo_path=args.demo,
            map_path=args.map,
            obstacles=args.obstacles,
            teacher_config=args.teacher_config,
            demo_states=demo_states,
            spawn_states=spawn_states,
            device=args.device,
            robust=True,
            controller_rate_gain=controller_model.rate_gain,
            aleatoric_scale=args.aleatoric_scale,
            live_estimator_realism=args.live_estimator_realism,
            impulse_rate_hz=args.impulse_rate_hz,
            actor=actor,
            actor_obs_mean=obs_mean,
            actor_obs_var=obs_var,
            residual_scale=args.residual_scale,
            multigate_vision=args.multigate_vision,
            vision_outcome_model=args.vision_outcome_model,
            geometry_limit=args.geometry_limit,
            tier_target=args.tier_target,
        )[0]
        reports.append(report)
        metadata.append({
            "path": str(ensemble_path),
            "dataset": ensemble_metadata.get("dataset"),
        })
        print(json.dumps({
            "ensemble": str(ensemble_path),
            "tier_success_rate": report["tier_success_rate"],
            "finish_rate": report["finish_rate"],
            "median_s": report["median_s"],
            "p90_s": report["p90_s"],
            "clearance_p10_m": report["clearance_p10_m"],
        }), flush=True)

    output = {
        "candidate": str(args.candidate),
        "worlds": args.worlds,
        "seed": args.seed,
        "tier_target_s": args.tier_target,
        "aleatoric_scale": args.aleatoric_scale,
        "impulse_rate_hz": args.impulse_rate_hz,
        "live_estimator_realism": args.live_estimator_realism,
        "multigate_vision": args.multigate_vision,
        "vision_outcome_model": (
            str(args.vision_outcome_model.resolve())
            if args.vision_outcome_model else None
        ),
        "reports_by_model": reports,
        "world_models": metadata,
        "worst_tier_success_rate": min(
            report["tier_success_rate"] for report in reports
        ),
        "worst_finish_rate": min(report["finish_rate"] for report in reports),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
