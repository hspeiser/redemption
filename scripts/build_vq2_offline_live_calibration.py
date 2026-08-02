"""Calibrate frozen-world-model finish rates against archived live attempts.

Each distinct deployed controller/actor/schedule signature is evaluated on
the same frozen v13 worlds. Live outcomes are aggregated from its archived
sessions, then a grouped-binomial logistic calibration curve is fitted.
The output is checkpointed after every signature so a long run is resumable.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel
from aigp.fastsim.worldmodel import ResidualEnsemble
from scripts.fastsim_train_ppo import load_demo_states, load_live_teacher_config
from scripts.optimize_vq2_residual_schedule import evaluate, load_actor


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value))


def artifact_hash(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_path(value) -> Path | None:
    if value in {None, "", "None"}:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def gate4_outcomes(
    log: Path,
    *,
    schedule_arm: str | None = None,
    max_sim_step_p95_s: float = 0.055,
    max_sim_step_max_s: float = 0.30,
    max_step_p95_ms: float = 60.0,
) -> dict:
    attempts = successes = healthy_attempts = healthy_successes = 0
    times = []
    multigate_values = []
    for line in log.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            schedule_arm is not None
            and str(row.get("schedule_arm", "candidate")) != schedule_arm
        ):
            continue
        attempts += 1
        multigate_values.append(bool(row.get("multigate", False)))
        healthy = bool(row.get("timing_healthy", True))
        if (
            max_sim_step_p95_s > 0.0
            and row.get("sim_step_p95_s") is not None
        ):
            healthy &= float(row["sim_step_p95_s"]) <= max_sim_step_p95_s
        if (
            max_sim_step_max_s > 0.0
            and row.get("sim_step_max_s") is not None
        ):
            healthy &= float(row["sim_step_max_s"]) <= max_sim_step_max_s
        if (
            max_step_p95_ms > 0.0
            and row.get("step_p95_ms") is not None
        ):
            healthy &= float(row["step_p95_ms"]) <= max_step_p95_ms
        if healthy:
            healthy_attempts += 1
        crossing = next((
            item for item in row.get("crossing_offsets", [])
            if int(item.get("gate", -1)) == 4
        ), None)
        if crossing is not None:
            successes += 1
            if healthy:
                healthy_successes += 1
            times.append((int(crossing["step"]) + 1) / 30.0)
    return {
        "attempts": attempts,
        "successes": successes,
        "healthy_attempts": healthy_attempts,
        "healthy_successes": healthy_successes,
        "times": times,
        "multigate": bool(multigate_values and all(multigate_values)),
    }


def parse_gate_set(value) -> set[int]:
    result = set()
    for item in str(value or "").split(","):
        try:
            result.add(int(item.strip()))
        except ValueError:
            pass
    return result


def runtime_stack_identity(config_path: Path) -> dict:
    """Return only runtime factors that can alter offline/live transfer.

    Controller parameters remain in the policy signature separately.  This
    identity prevents a calibration fit from pooling policies that were flown
    with incompatible perception, map, or timing stacks.
    """
    config = json.loads(config_path.read_text())
    args = config.get("args", config)

    def hashed(key: str) -> str | None:
        return artifact_hash(optional_path(args.get(key)))

    return {
        "map_sha256": hashed("map"),
        "gate_primary_sha256": hashed("gate_primary"),
        "crop_sha256": hashed("crop"),
        "line_model_sha256": hashed("line_model"),
        "vision_hz": float(args.get("vision_hz", 0.0)),
        "vision_device": str(args.get("vision_device", "")),
        "max_vision_result_age": float(
            args.get("max_vision_result_age", 0.0)
        ),
        "vision_process_isolation": bool(
            args.get("vision_process_isolation", False)
        ),
        "crop_tracker": bool(args.get("crop_tracker", False)),
        "crop_tracker_hz": float(args.get("crop_tracker_hz", 0.0)),
        "crop_direct_position_pins": bool(
            args.get("crop_direct_position_pins", False)
        ),
        "control_hz": float(args.get("control_hz", 30.0)),
        "wait_for_official_start": bool(
            args.get("wait_for_official_start", False)
        ),
    }


def identity_digest(identity: dict) -> str:
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
        default=json_default,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def policy_signature(
    config_path: Path,
    *,
    map_path: Path,
    multigate: bool,
    runtime_config_path: Path | None = None,
    force_actor_inactive: bool = False,
    reference_demo_path: Path | None = None,
) -> tuple[str, dict]:
    config = json.loads(config_path.read_text())
    args = config.get("args", config)
    arrays, fixed = load_live_teacher_config(
        config_path, 1, map_path=map_path
    )
    schedule_path = optional_path(args.get("ppo_residual_schedule"))
    residual_gates = parse_gate_set(args.get("residual_gates"))
    actor_is_active = bool(
        schedule_path is not None
        or residual_gates.intersection(range(5))
    ) and not force_actor_inactive
    if force_actor_inactive:
        schedule_path = None
    actor_path = optional_path(args.get("ppo_residual_checkpoint"))
    if actor_path is None and actor_is_active:
        actor_path = optional_path(args.get("seed_checkpoint"))
    runtime_stack = runtime_stack_identity(runtime_config_path or config_path)
    reference_demo = reference_demo_path or optional_path(args.get("demo"))
    payload = {
        "teacher_arrays": {
            key: np.asarray(value[0]).tolist()
            for key, value in arrays.items()
        },
        "teacher_fixed": fixed,
        "actor_sha256": artifact_hash(actor_path),
        "schedule_sha256": artifact_hash(schedule_path),
        "residual_scale": float(args.get("residual_scale", 0.2)),
        "reference_demo_sha256": artifact_hash(reference_demo),
        "multigate": bool(multigate),
        "runtime_stack_sha256": identity_digest(runtime_stack),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=json_default
    )
    return hashlib.sha256(canonical.encode()).hexdigest(), {
        **payload,
        "actor_path": str(actor_path) if actor_path else None,
        "schedule_path": str(schedule_path) if schedule_path else None,
        "reference_demo_path": (
            str(reference_demo) if reference_demo else None
        ),
        "actor_is_active": actor_is_active,
        "runtime_stack": runtime_stack,
    }


def fit_binomial(rows: list[dict]) -> dict:
    valid = [
        row for row in rows
        if row.get("offline_finish_rate") is not None
        and row.get("live_healthy_attempts", row["live_attempts"]) > 0
    ]
    x = np.asarray([row["offline_finish_rate"] for row in valid], float)
    successes = np.asarray([
        row.get("live_healthy_successes", row["live_successes"])
        for row in valid
    ], float)
    trials = np.asarray([
        row.get("live_healthy_attempts", row["live_attempts"])
        for row in valid
    ], float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    ridge = np.diag([1e-6, 1e-4])
    for _ in range(50):
        eta = np.clip(design @ beta, -20.0, 20.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(trials * probability * (1.0 - probability), 1e-6)
        target = eta + (successes - trials * probability) / weight
        lhs = design.T @ (weight[:, None] * design) + ridge
        rhs = design.T @ (weight * target)
        updated = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(updated - beta)) < 1e-9:
            beta = updated
            break
        beta = updated
    grid = np.linspace(0.75, 1.0, 26)
    predicted = 1.0 / (1.0 + np.exp(-(
        beta[0] + beta[1] * grid
    )))
    live_rate = successes / np.maximum(trials, 1.0)
    weighted_x_mean = np.average(x, weights=trials)
    weighted_y_mean = np.average(live_rate, weights=trials)
    covariance = np.average(
        (x - weighted_x_mean) * (live_rate - weighted_y_mean),
        weights=trials,
    )
    variance_x = np.average((x - weighted_x_mean) ** 2, weights=trials)
    variance_y = np.average(
        (live_rate - weighted_y_mean) ** 2, weights=trials
    )
    weighted_correlation = covariance / max(
        np.sqrt(variance_x * variance_y), 1e-12
    )
    required_offline = None
    if beta[1] > 0.0:
        required_offline = float(
            (np.log(0.9 / 0.1) - beta[0]) / beta[1]
        )
    return {
        "groups": len(valid),
        "live_attempts": int(trials.sum()),
        "coefficients": beta.tolist(),
        "weighted_offline_live_correlation": float(weighted_correlation),
        "offline_rate_for_predicted_live_90pct": required_offline,
        "curve": [
            {"offline_finish_rate": float(a), "predicted_live_rate": float(b)}
            for a, b in zip(grid, predicted)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--fallback-actor", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20268131)
    parser.add_argument("--aleatoric-scale", type=float, default=1.5)
    parser.add_argument("--impulse-rate-hz", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--runtime-stack-like",
        type=Path,
        help=(
            "Only calibrate sessions whose perception/map/timing stack "
            "matches this config. Controller settings may still differ."
        ),
    )
    parser.add_argument(
        "--require-multigate",
        action="store_true",
        help="When filtering a runtime stack, retain only multigate sessions.",
    )
    parser.add_argument("--max-live-sim-step-p95", type=float, default=0.055)
    parser.add_argument("--max-live-sim-step-max", type=float, default=0.30)
    parser.add_argument("--max-live-step-p95-ms", type=float, default=60.0)
    args = parser.parse_args()

    required_stack = (
        identity_digest(runtime_stack_identity(args.runtime_stack_like))
        if args.runtime_stack_like is not None else None
    )

    groups: dict[str, dict] = {}
    for log in sorted(args.training_root.glob("g0g4*/**/episodes.jsonl")):
        config_path = log.parent / "config.json"
        if not config_path.is_file():
            continue
        outcome = gate4_outcomes(
            log,
            max_sim_step_p95_s=args.max_live_sim_step_p95,
            max_sim_step_max_s=args.max_live_sim_step_max,
            max_step_p95_ms=args.max_live_step_p95_ms,
        )
        signature, identity = policy_signature(
            config_path,
            map_path=args.map,
            multigate=outcome["multigate"],
        )
        if required_stack is not None:
            if identity["runtime_stack_sha256"] != required_stack:
                continue
            if args.require_multigate and not outcome["multigate"]:
                continue
        group = groups.setdefault(signature, {
            "policy_sha256": signature,
            "identity": identity,
            "config_path": str(config_path),
            "sessions": [],
            "live_attempts": 0,
            "live_successes": 0,
            "live_times_s": [],
            "live_healthy_attempts": 0,
            "live_healthy_successes": 0,
        })
        group["sessions"].append(log.parent.name)
        group["live_attempts"] += outcome["attempts"]
        group["live_successes"] += outcome["successes"]
        group["live_healthy_attempts"] += outcome["healthy_attempts"]
        group["live_healthy_successes"] += outcome["healthy_successes"]
        group["live_times_s"].extend(outcome["times"])

    prior = {}
    if args.out.is_file():
        try:
            prior = {
                row["policy_sha256"]: row
                for row in json.loads(args.out.read_text()).get("rows", [])
                if row.get("offline_finish_rate") is not None
            }
        except (json.JSONDecodeError, KeyError):
            prior = {}

    ensemble, metadata = ResidualEnsemble.load(args.ensemble, args.device)
    ensemble.eval()
    model = SurrogateModel.load(args.model)
    controller_model = SurrogateModel.load(args.controller_model)
    demo = load_demo_states(args.demo, args.map)
    keep = np.asarray(demo["gate"]) < 5
    demo = {key: np.asarray(value)[keep] for key, value in demo.items()}
    actor_cache = {}

    rows = []
    ordered = sorted(groups.values(), key=lambda row: row["policy_sha256"])
    output = {
        "frozen_stack": {
            "ensemble": str(args.ensemble),
            "world_model_dataset": metadata.get("dataset"),
            "worlds_per_policy": args.worlds,
            "seed": args.seed,
            "aleatoric_scale": args.aleatoric_scale,
            "impulse_rate_hz": args.impulse_rate_hz,
            "runtime_stack_like": (
                str(args.runtime_stack_like)
                if args.runtime_stack_like is not None else None
            ),
            "runtime_stack_sha256": required_stack,
            "require_multigate": bool(args.require_multigate),
            "live_timing_thresholds": {
                "sim_step_p95_s": args.max_live_sim_step_p95,
                "sim_step_max_s": args.max_live_sim_step_max,
                "step_p95_ms": args.max_live_step_p95_ms,
            },
        },
        "completed": 0,
        "total": len(ordered),
        "rows": rows,
    }
    for index, row in enumerate(ordered):
        signature = row["policy_sha256"]
        if signature in prior:
            saved = prior[signature]
            for key in (
                "offline_finish_rate", "offline_median_s", "offline_report"
            ):
                row[key] = saved.get(key)
            row["live_finish_rate"] = (
                row["live_successes"] / row["live_attempts"]
                if row["live_attempts"] else None
            )
            row["live_healthy_finish_rate"] = (
                row["live_healthy_successes"]
                / row["live_healthy_attempts"]
                if row["live_healthy_attempts"] else None
            )
            rows.append(row)
            continue
        actor_path = optional_path(row["identity"]["actor_path"])
        cache_key = str(actor_path) if actor_path else "protected_zero"
        if cache_key not in actor_cache:
            source = actor_path or args.fallback_actor
            actor, obs_mean, obs_var = load_actor(source, args.device)
            if actor_path is None:
                actor = copy.deepcopy(actor)
                actor.mean.weight.data.zero_()
                actor.mean.bias.data.zero_()
            actor_cache[cache_key] = (actor, obs_mean, obs_var)
        actor, obs_mean, obs_var = actor_cache[cache_key]
        schedule_path = optional_path(row["identity"]["schedule_path"])
        if schedule_path is None:
            knots = 3
            theta = np.zeros((1, 5 * knots * 4), np.float64)
        else:
            schedule = json.loads(schedule_path.read_text())
            knots = int(schedule.get("knots", 3))
            theta = np.asarray(
                schedule["residual_schedule"], np.float64
            ).reshape(1, -1)
        report = evaluate(
            theta,
            worlds=args.worlds,
            knots=knots,
            actor=actor,
            obs_mean=obs_mean,
            obs_var=obs_var,
            model=model,
            controller_model=controller_model,
            ensemble=ensemble,
            demo_path=args.demo,
            demo_states=demo,
            map_path=args.map,
            obstacles=args.obstacles,
            teacher_config=Path(row["config_path"]),
            device=args.device,
            aleatoric_scale=args.aleatoric_scale,
            impulse_rate_hz=args.impulse_rate_hz,
            seed=args.seed,
            multigate_estimator=bool(row["identity"]["multigate"]),
            residual_scale=float(row["identity"]["residual_scale"]),
        )[0]
        row["offline_finish_rate"] = report["finish_rate"]
        row["offline_median_s"] = report["median_s"]
        row["offline_report"] = report
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
        output["completed"] = len(rows)
        output["rows"] = rows
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2, default=json_default))
        print(
            f"[{index + 1}/{len(ordered)}] {signature[:10]} "
            f"offline={report['finish_rate']:.3f} "
            f"live={row['live_successes']}/{row['live_attempts']}",
            flush=True,
        )

    output["calibration"] = fit_binomial(rows)
    output["completed"] = len(rows)
    args.out.write_text(json.dumps(output, indent=2, default=json_default))
    print(json.dumps(output["calibration"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
