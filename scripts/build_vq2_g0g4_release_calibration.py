"""Calibrate g0-g4 policies with the authoritative release-state evaluator.

The older calibration path spawned every policy from one ideal rest state and
used a different estimator model from the g0-g4 optimizer.  This script uses
recorded release-state clouds, the same multigate/live-estimator realism, and
the same pooled dynamics ensembles used for candidate promotion.
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

from aigp.fastsim.sysid import SurrogateModel
from aigp.fastsim.worldmodel import ResidualEnsemble
from scripts.build_vq2_offline_live_calibration import (
    fit_binomial,
    gate4_outcomes,
    identity_digest,
    json_default,
    optional_path,
    policy_signature,
    runtime_stack_identity,
)
from scripts.export_vq2_config_candidate import candidate_payload
from scripts.optimize_vq2_g0g4_worldmodel import (
    decoded_demo,
    evaluate,
    release_states,
)
from scripts.optimize_vq2_residual_schedule import load_actor


def theta_from_config(config: Path, map_path: Path) -> np.ndarray:
    payload = candidate_payload(config, map_path)
    values = (
        payload["leads"]
        + payload["thrust_scales"]
        + payload["velocity_scales"]
        + payload["trajectory_blends"]
        + payload["lateral_offsets_m"]
        + payload["vertical_offsets_m"]
        + payload["rate_scales"]
    )
    theta = np.asarray(values, np.float64)[None]
    if theta.shape != (1, 35):
        raise ValueError(f"expected one 35-value schedule, got {theta.shape}")
    return theta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--runtime-stack-like", type=Path, required=True)
    parser.add_argument("--require-multigate", action="store_true")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260947)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aleatoric-scale", type=float, default=1.5)
    parser.add_argument("--impulse-rate-hz", type=float, default=0.08)
    parser.add_argument("--vision-outcome-model", type=Path)
    parser.add_argument("--residual-scale-default", type=float, default=0.20)
    parser.add_argument("--geometry-limit", type=float, default=0.50)
    parser.add_argument("--tier-target", type=float, default=8.7)
    parser.add_argument("--max-live-sim-step-p95", type=float, default=0.055)
    parser.add_argument("--max-live-sim-step-max", type=float, default=0.30)
    parser.add_argument("--max-live-step-p95-ms", type=float, default=60.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.worlds < 1:
        parser.error("--worlds must be positive")
    required_stack = identity_digest(
        runtime_stack_identity(args.runtime_stack_like)
    )

    groups: dict[str, dict] = {}
    for log in sorted(args.training_root.glob("**/episodes.jsonl")):
        runtime_config = log.parent / "config.json"
        if not runtime_config.is_file():
            continue
        runtime_payload = json.loads(runtime_config.read_text())
        runtime_args = runtime_payload.get("args", runtime_payload)
        arms = set()
        for line in log.read_text().splitlines():
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                continue
            arms.add(str(episode.get("schedule_arm", "candidate")))
        if not arms:
            continue
        for arm in sorted(arms):
            controller_config = runtime_config
            force_actor_inactive = False
            reference_demo = optional_path(runtime_args.get("demo"))
            if arm == "protected_champion":
                champion = optional_path(
                    runtime_args.get("interleave_champion_config")
                )
                if champion is None or not champion.is_file():
                    raise FileNotFoundError(
                        "protected_champion rows require an existing "
                        f"interleave_champion_config: {runtime_config}"
                    )
                controller_config = champion
                force_actor_inactive = True
                reference_demo = optional_path(
                    runtime_args.get("interleave_champion_demo")
                )
                if reference_demo is None:
                    champion_payload = json.loads(champion.read_text())
                    champion_args = champion_payload.get(
                        "args", champion_payload
                    )
                    reference_demo = optional_path(champion_args.get("demo"))
            outcome = gate4_outcomes(
                log,
                schedule_arm=arm,
                max_sim_step_p95_s=args.max_live_sim_step_p95,
                max_sim_step_max_s=args.max_live_sim_step_max,
                max_step_p95_ms=args.max_live_step_p95_ms,
            )
            signature, identity = policy_signature(
                controller_config,
                map_path=args.map,
                multigate=outcome["multigate"],
                runtime_config_path=runtime_config,
                force_actor_inactive=force_actor_inactive,
                reference_demo_path=reference_demo,
            )
            if identity["runtime_stack_sha256"] != required_stack:
                continue
            if args.require_multigate and not outcome["multigate"]:
                continue
            group = groups.setdefault(signature, {
                "policy_sha256": signature,
                "identity": identity,
                "config_path": str(controller_config),
                "runtime_config_paths": [],
                "schedule_arm": arm,
                "sessions": [],
                "live_attempts": 0,
                "live_successes": 0,
                "live_healthy_attempts": 0,
                "live_healthy_successes": 0,
                "live_times_s": [],
            })
            group["runtime_config_paths"].append(str(runtime_config))
            group["sessions"].append(log.parent.name)
            group["live_attempts"] += outcome["attempts"]
            group["live_successes"] += outcome["successes"]
            group["live_healthy_attempts"] += outcome["healthy_attempts"]
            group["live_healthy_successes"] += outcome["healthy_successes"]
            group["live_times_s"].extend(outcome["times"])

    model = SurrogateModel.load(args.model)
    controller_model = SurrogateModel.load(
        args.controller_model or args.model
    )
    ensembles = []
    ensemble_metadata = []
    for path in args.ensemble:
        ensemble, metadata = ResidualEnsemble.load(path, args.device)
        ensemble.eval()
        ensembles.append(ensemble)
        ensemble_metadata.append({
            "path": str(path),
            "dataset": metadata.get("dataset"),
        })
    demo_states = decoded_demo(args.demo, args.map)
    spawn_states = release_states(args.dataset)
    actor_cache = {}
    rows = []

    for index, row in enumerate(sorted(groups.values(), key=lambda x: x["policy_sha256"])):
        schedule_path = optional_path(row["identity"]["schedule_path"])
        if schedule_path is not None:
            row["skipped"] = "open_loop_residual_schedule_not_supported"
            rows.append(row)
            continue
        actor = obs_mean = obs_var = None
        if row["identity"]["actor_is_active"]:
            actor_path = optional_path(row["identity"]["actor_path"])
            if actor_path is None:
                row["skipped"] = "active_actor_artifact_missing"
                rows.append(row)
                continue
            cache_key = str(actor_path)
            if cache_key not in actor_cache:
                actor_cache[cache_key] = load_actor(actor_path, args.device)
            actor, obs_mean, obs_var = actor_cache[cache_key]

        config = Path(row["config_path"])
        theta = theta_from_config(config, args.map)
        reports = []
        for ensemble in ensembles:
            torch.manual_seed(args.seed)
            reports.append(evaluate(
                theta,
                worlds=args.worlds,
                model=model,
                ensemble=ensemble,
                demo_path=args.demo,
                map_path=args.map,
                obstacles=args.obstacles,
                teacher_config=config,
                demo_states=demo_states,
                spawn_states=spawn_states,
                device=args.device,
                robust=True,
                controller_rate_gain=controller_model.rate_gain,
                aleatoric_scale=args.aleatoric_scale,
                live_estimator_realism=True,
                impulse_rate_hz=args.impulse_rate_hz,
                actor=actor,
                actor_obs_mean=obs_mean,
                actor_obs_var=obs_var,
                residual_scale=float(row["identity"].get(
                    "residual_scale", args.residual_scale_default
                )),
                multigate_vision=bool(row["identity"]["multigate"]),
                vision_outcome_model=args.vision_outcome_model,
                geometry_limit=args.geometry_limit,
                tier_target=args.tier_target,
            )[0])
        row["reports_by_model"] = reports
        row["offline_finish_rate"] = min(r["finish_rate"] for r in reports)
        row["offline_tier_rate"] = min(r["tier_success_rate"] for r in reports)
        row["offline_median_s"] = max(
            r["median_s"] if r["median_s"] is not None else 99.0
            for r in reports
        )
        row["live_finish_rate"] = (
            row["live_successes"] / row["live_attempts"]
            if row["live_attempts"] else None
        )
        row["live_healthy_finish_rate"] = (
            row["live_healthy_successes"] / row["live_healthy_attempts"]
            if row["live_healthy_attempts"] else None
        )
        row["live_median_s"] = (
            float(np.median(row["live_times_s"]))
            if row["live_times_s"] else None
        )
        rows.append(row)
        print(
            f"[{index + 1}/{len(groups)}] {row['policy_sha256'][:10]} "
            f"offline={row['offline_finish_rate']:.3f} "
            f"healthy={row['live_healthy_successes']}/"
            f"{row['live_healthy_attempts']}",
            flush=True,
        )

    fitted_rows = [row for row in rows if "offline_finish_rate" in row]
    output = {
        "evaluator": "g0g4_recorded_release_state_live_estimator",
        "worlds_per_policy": args.worlds,
        "seed": args.seed,
        "runtime_stack_sha256": required_stack,
        "runtime_stack_like": str(args.runtime_stack_like),
        "require_multigate": bool(args.require_multigate),
        "dataset": str(args.dataset),
        "world_models": ensemble_metadata,
        "aleatoric_scale": args.aleatoric_scale,
        "impulse_rate_hz": args.impulse_rate_hz,
        "vision_outcome_model": (
            str(args.vision_outcome_model.resolve())
            if args.vision_outcome_model else None
        ),
        "live_timing_thresholds": {
            "sim_step_p95_s": args.max_live_sim_step_p95,
            "sim_step_max_s": args.max_live_sim_step_max,
            "step_p95_ms": args.max_live_step_p95_ms,
        },
        "rows": rows,
        "calibration": fit_binomial(fitted_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=json_default))
    print(json.dumps(output["calibration"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
