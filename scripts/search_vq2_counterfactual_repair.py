"""Search robust smooth residual repairs from reproduced real failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.branching import (  # noqa: E402
    calibrated_position_sigma,
    restore_branch_cloud,
)
from aigp.fastsim.env import HOLE_HALF, FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.live_teacher import LiveTeacherController  # noqa: E402
from aigp.fastsim.lineopt import load_oriented_gates  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from scripts.fastsim_train_ppo import (  # noqa: E402
    load_demo_states,
    load_live_teacher_config,
    load_schedule_gate_positions,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spline_action(
    table: torch.Tensor,
    step: int,
    total_steps: int,
    worlds: int,
) -> torch.Tensor:
    expanded = table.repeat_interleave(worlds, dim=0)
    coordinate = min(1.0, step / max(total_steps - 1, 1)) * (
        table.shape[1] - 1
    )
    left = int(np.floor(coordinate))
    right = min(left + 1, table.shape[1] - 1)
    fraction = float(coordinate - left)
    return expanded[:, left] + fraction * (
        expanded[:, right] - expanded[:, left]
    )


def pair_worlds(env: FastVQ2Env, worlds: int, candidates: int) -> None:
    names = (
        "p", "v", "q", "w", "target", "prev_action", "noise_pos",
        "vis_age", "progress", "best_prog", "t_best", "dr_thrust",
        "dr_K", "dr_tau", "dr_drag", "residual_member", "act_delay",
    )
    for name in names:
        value = getattr(env, name)
        base = value[:worlds].clone()
        repeats = (candidates,) + (1,) * (value.ndim - 1)
        value.copy_(base.repeat(*repeats))
    env.act_buf.copy_(
        env.act_buf[:, :worlds].repeat(1, candidates, 1)
    )
    if env.backbone is not None:
        ids = torch.arange(env.cfg.n_envs, device=env.device)
        env.backbone.reset_nearest(ids, env.p + env.noise_pos)
        env.backbone.set_target(env.target)
        env.backbone.set_previous_action(env.prev_action)


@torch.no_grad()
def evaluate_model(
    theta: np.ndarray,
    *,
    source: dict[str, np.ndarray],
    branch_step: int,
    failure_target_gate: int,
    geometry_gate: int,
    ensemble_path: Path,
    base: SurrogateModel,
    demo_states: dict,
    demo_path: Path,
    map_path: Path,
    obstacles_path: Path,
    teacher_config: Path,
    worlds: int,
    knots: int,
    authority: float,
    geometry_search: bool,
    max_lateral_offset: float,
    max_vertical_offset: float,
    max_speed_scale_delta: float,
    post_steps: int,
    aleatoric_scale: float,
    position_sigma_scale: float,
    position_sigma_floor_m: float,
    seed: int,
    device: str,
) -> list[dict]:
    torch.manual_seed(seed)
    candidates = len(theta)
    n = candidates * worlds
    tail_steps = len(source["action"]) - branch_step
    total_steps = max(tail_steps + post_steps, knots)
    action_dimensions = (knots - 1) * 4
    values = theta[:, :action_dimensions].reshape(candidates, knots - 1, 4)
    values = np.clip(values, -authority, authority)
    values = np.concatenate([
        values, np.zeros((candidates, 1, 4), np.float64)
    ], axis=1)
    table = torch.as_tensor(values, dtype=torch.float32, device=device)
    cfg = FastEnvConfig(
        n_envs=n,
        race_gates=max(5, failure_target_gate + 1),
        random_start_frac=0.0,
        spawn_at_rest=True,
        max_episode_s=4.0,
        residual_scale=0.25,
        act_delay_steps_min=0,
        act_delay_steps_max=1,
        reloc_events=False,
        fov_vision=False,
        world_model_aleatoric_scale=aleatoric_scale,
        impulse_rate_hz=0.0,
        demo_corridor_m=3.0,
        dr_thrust=(0.97, 1.03),
        dr_rate_gain=(0.97, 1.03),
        dr_rate_tau=(0.95, 1.05),
        dr_drag=(0.22, 0.34),
    )
    arrays, fixed = load_live_teacher_config(
        teacher_config, n, map_path=map_path
    )
    geometry = np.zeros((candidates, 3), np.float64)
    if geometry_search:
        geometry = theta[:, action_dimensions:action_dimensions + 3].copy()
        geometry[:, 0] = np.clip(
            geometry[:, 0], -max_lateral_offset, max_lateral_offset
        )
        geometry[:, 1] = np.clip(
            geometry[:, 1], -max_vertical_offset, max_vertical_offset
        )
        geometry[:, 2] = np.clip(
            geometry[:, 2], -max_speed_scale_delta, max_speed_scale_delta
        )
        gate_count = arrays["action_leads"].shape[1]
        offsets = arrays.setdefault(
            "reference_gate_offsets_world",
            np.zeros((n, gate_count, 3), np.float32),
        )
        _gate_position, gate_rotation = load_oriented_gates(map_path)
        lateral_axis = gate_rotation[geometry_gate, :, 0]
        vertical_axis = gate_rotation[geometry_gate, :, 2]
        per_candidate_offset = (
            geometry[:, 0, None] * lateral_axis[None]
            + geometry[:, 1, None] * vertical_axis[None]
        )
        offsets[:, geometry_gate] += np.repeat(
            per_candidate_offset, worlds, axis=0
        ).astype(np.float32)
        speed_multiplier = np.repeat(
            1.0 + geometry[:, 2], worlds
        ).astype(np.float32)
        arrays["trajectory_velocity_scales"][:, geometry_gate] *= (
            speed_multiplier
        )
    fixed.setdefault(
        "schedule_gate_positions", load_schedule_gate_positions(map_path)
    )
    backbone = LiveTeacherController(
        demo_path,
        n,
        device=device,
        rate_gain=np.asarray(base.rate_gain),
        **arrays,
        **fixed,
    )
    ensemble, metadata = ResidualEnsemble.load(ensemble_path, device)
    ensemble.eval()
    env = FastVQ2Env(
        base,
        map_path,
        demo_states=demo_states,
        config=cfg,
        device=device,
        obstacles_path=obstacles_path,
        backbone=backbone,
        residual_ensemble=ensemble,
    )
    sigma = calibrated_position_sigma(
        float(source["position_sigma_m"][branch_step]),
        scale=position_sigma_scale,
        floor_m=position_sigma_floor_m,
    )
    age = float(source["landmark_age_s"][branch_step])
    restore_branch_cloud(
        env,
        position=source["position"][branch_step],
        velocity=source["velocity"][branch_step],
        rotation=source["rotation"][branch_step],
        rates=source["rates"][branch_step],
        previous_action=source["previous_action"][branch_step],
        target_gate=int(source["gate_index"][branch_step]),
        position_sigma_m=sigma,
        landmark_age_s=age if np.isfinite(age) else 0.25,
        seed=seed,
    )
    pair_worlds(env, worlds, candidates)
    passed = torch.zeros(n, dtype=torch.bool, device=device)
    failed = torch.zeros_like(passed)
    pass_step = torch.full((n,), -1, dtype=torch.long, device=device)
    clearance = torch.full((n,), -1.0, device=device)
    # Keep a dense near-miss signal for CEM.  A gate-frame collision still
    # reports cross_r, even though it is not a pass; without this all early
    # populations are a flat wall of zero-success candidates and CEM has no
    # direction in which to move.
    crossing_radius = torch.full((n,), float("inf"), device=device)
    support = torch.zeros(n, device=device)
    disagreement = torch.zeros(n, device=device)
    post_count = torch.zeros(n, dtype=torch.long, device=device)
    for step in range(total_steps):
        residual = spline_action(table, step, total_steps, worlds)
        _obs, _reward, done, info = env.step(residual)
        target_pass = (
            (info["cross_gate"] == failure_target_gate)
            & info["passed"]
            & ~info["hit"]
        )
        first_pass = target_pass & ~passed & ~failed
        target_cross = info["cross_gate"] == failure_target_gate
        crossing_radius = torch.where(
            target_cross,
            torch.minimum(crossing_radius, info["cross_r"]),
            crossing_radius,
        )
        passed |= first_pass
        pass_step = torch.where(
            first_pass, torch.full_like(pass_step, step), pass_step
        )
        clearance = torch.where(
            first_pass, HOLE_HALF - info["cross_r"], clearance
        )
        failed |= done & ~passed
        failed |= info["hit"]
        post_count += (passed & ~failed).long()
        support = torch.maximum(support, info["world_model_support_z"])
        disagreement = torch.maximum(
            disagreement, info["world_model_disagreement"]
        )
    success = passed & ~failed & (post_count >= post_steps)
    next_gate = min(failure_target_gate + 1, 16)
    next_vector = env.gate_pos[next_gate] - env.p
    next_distance = torch.linalg.norm(next_vector, dim=1)
    next_direction = next_vector / next_distance[:, None].clamp(min=1e-6)
    entry_speed = (env.v * next_direction).sum(1)
    result = []
    for candidate in range(candidates):
        sl = slice(candidate * worlds, (candidate + 1) * worlds)
        ok = success[sl]
        clear = clearance[sl][ok]
        pass_times = pass_step[sl][ok].float() / 30.0
        radii = crossing_radius[sl]
        finite_radii = radii[torch.isfinite(radii)]
        result.append({
            "finish_rate": float(ok.float().mean()),
            "hit_rate": float(failed[sl].float().mean()),
            "clearance_p10_m": (
                float(torch.quantile(clear, 0.1)) if len(clear) else -1.0
            ),
            "crossing_radius_p50_m": (
                float(torch.quantile(finite_radii, 0.5))
                if len(finite_radii) else None
            ),
            "crossing_radius_p90_m": (
                float(torch.quantile(finite_radii, 0.9))
                if len(finite_radii) else None
            ),
            "pass_time_median_s": (
                float(pass_times.median()) if len(pass_times) else None
            ),
            "next_distance_p90_m": (
                float(torch.quantile(next_distance[sl][ok], 0.9))
                if bool(ok.any()) else None
            ),
            "next_entry_speed_p10_mps": (
                float(torch.quantile(entry_speed[sl][ok], 0.1))
                if bool(ok.any()) else None
            ),
            "support_p90": float(torch.quantile(support[sl], 0.9)),
            "disagreement_p90": float(torch.quantile(
                disagreement[sl], 0.9
            )),
            "residual_rms": float(np.sqrt(np.mean(values[candidate] ** 2))),
            "residual_jerk_rms": float(np.sqrt(np.mean(
                np.diff(values[candidate], axis=0) ** 2
            ))),
            "reference_lateral_offset_m": float(geometry[candidate, 0]),
            "reference_vertical_offset_m": float(geometry[candidate, 1]),
            "reference_speed_scale": float(1.0 + geometry[candidate, 2]),
            "reference_geometry_gate": int(geometry_gate),
            "source_dataset": metadata.get("dataset"),
        })
    return result


def objective(report: dict) -> float:
    rate = report["finish_rate"]
    clearance = report["clearance_p10_m"]
    pass_time = report["pass_time_median_s"] or 9.0
    next_distance = report["next_distance_p90_m"] or 50.0
    next_speed = report["next_entry_speed_p10_mps"] or -5.0
    cross_p50 = report["crossing_radius_p50_m"] or 5.0
    cross_p90 = report["crossing_radius_p90_m"] or 5.0
    return (
        12000.0 * rate
        # Smoothly guide a zero-success population toward the hole before
        # the discontinuous pass bonus becomes available.
        - 900.0 * cross_p50
        - 350.0 * cross_p90
        - 2500.0 * max(0.0, 0.10 - clearance)
        - 250.0 * report["hit_rate"]
        - 25.0 * pass_time
        - 2.0 * next_distance
        + 2.0 * next_speed
        - 40.0 * report["support_p90"]
        - 150.0 * report["disagreement_p90"]
        - 30.0 * report["residual_rms"]
        - 20.0 * report["residual_jerk_rms"]
        - 20.0 * abs(report["reference_lateral_offset_m"])
        - 20.0 * abs(report["reference_vertical_offset_m"])
        - 20.0 * abs(report["reference_speed_scale"] - 1.0)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--reproduction-report", type=Path, required=True)
    parser.add_argument(
        "--only-rollback-steps",
        default="",
        help="Optional comma-separated subset of reproduction-eligible horizons.",
    )
    parser.add_argument("--ensemble", type=Path, action="append", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument(
        "--initial-report",
        type=Path,
        help="Optional prior repair report used only to warm-start CEM.",
    )
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--selection-worlds", type=int, default=128)
    parser.add_argument("--audit-worlds", type=int, default=512)
    parser.add_argument("--knots", type=int, default=5)
    parser.add_argument("--authority", type=float, default=0.20)
    parser.add_argument("--geometry-search", action="store_true")
    parser.add_argument("--max-lateral-offset", type=float, default=0.35)
    parser.add_argument("--max-vertical-offset", type=float, default=0.25)
    parser.add_argument("--max-speed-scale-delta", type=float, default=0.15)
    parser.add_argument("--post-steps", type=int, default=15)
    parser.add_argument("--aleatoric-scale", type=float, default=1.0)
    parser.add_argument("--max-support-p90", type=float, default=3.0)
    parser.add_argument("--max-disagreement-p90", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable repair search output exists: {args.out}")
    metadata = json.loads(
        (args.snapshot_artifact / "metadata.json").read_text()
    )
    source_npz = np.load(
        args.snapshot_artifact / "source_trajectory.npz", allow_pickle=False
    )
    source = {key: np.asarray(source_npz[key]) for key in source_npz.files}
    reproduction = json.loads(args.reproduction_report.read_text())
    position_sigma_scale = float(
        reproduction.get("position_sigma_scale", 1.0)
    )
    position_sigma_floor_m = float(
        reproduction.get("position_sigma_floor_m", 0.0)
    )
    eligible_rollbacks = set(reproduction["eligible_rollback_steps"])
    eligible_snapshots = [
        row for row in metadata["snapshots"]
        if row["rollback_steps"] in eligible_rollbacks
    ]
    if args.only_rollback_steps.strip():
        requested_rollbacks = {
            int(value.strip())
            for value in args.only_rollback_steps.split(",")
            if value.strip()
        }
        eligible_snapshots = [
            row for row in eligible_snapshots
            if row["rollback_steps"] in requested_rollbacks
        ]
    if not eligible_snapshots:
        raise ValueError("reproduction report admits no repair snapshots")
    base = SurrogateModel.load(args.base_model)
    demo = load_demo_states(args.demo, args.map)
    failure_gate = int(metadata["classification"]["target_gate"])
    action_dimensions = (args.knots - 1) * 4
    dimensions = action_dimensions + (3 if args.geometry_search else 0)
    bounds = np.full(dimensions, args.authority, np.float64)
    if args.geometry_search:
        bounds[-3:] = (
            args.max_lateral_offset,
            args.max_vertical_offset,
            args.max_speed_scale_delta,
        )
    searches = []
    rng = np.random.default_rng(args.seed)
    initial_theta = None
    initial_report_sha256 = None
    if args.initial_report is not None:
        prior = json.loads(args.initial_report.read_text())
        initial_report_sha256 = sha256(args.initial_report)
        prior_knots = np.asarray(
            prior["chosen_residual_knots"], np.float64
        )
        if prior_knots.shape != (args.knots, 4):
            raise ValueError(
                "initial report knot shape does not match this search: "
                f"{prior_knots.shape} vs {(args.knots, 4)}"
            )
        pieces = [prior_knots[:-1].reshape(-1)]
        if args.geometry_search:
            prior_search = next(
                row for row in prior["searches"]
                if int(row["snapshot"]["rollback_steps"])
                in {int(s["rollback_steps"]) for s in eligible_snapshots}
            )
            geometry = prior_search["reference_geometry"]
            pieces.append(np.asarray([
                geometry["lateral_offset_m"],
                geometry["vertical_offset_m"],
                geometry["speed_scale"] - 1.0,
            ]))
        initial_theta = np.clip(np.concatenate(pieces), -bounds, bounds)

    def evaluate_all(theta, snapshot, worlds, seed):
        return [
            evaluate_model(
                theta,
                source=source,
                branch_step=int(snapshot["branch_step"]),
                failure_target_gate=failure_gate,
                geometry_gate=int(source["gate_index"][snapshot["branch_step"]]),
                ensemble_path=path,
                base=base,
                demo_states=demo,
                demo_path=args.demo,
                map_path=args.map,
                obstacles_path=args.obstacles,
                teacher_config=args.teacher_config,
                worlds=worlds,
                knots=args.knots,
                authority=args.authority,
                geometry_search=args.geometry_search,
                max_lateral_offset=args.max_lateral_offset,
                max_vertical_offset=args.max_vertical_offset,
                max_speed_scale_delta=args.max_speed_scale_delta,
                post_steps=args.post_steps,
                aleatoric_scale=args.aleatoric_scale,
                position_sigma_scale=position_sigma_scale,
                position_sigma_floor_m=position_sigma_floor_m,
                seed=seed,
                device=args.device,
            )
            for path in args.ensemble
        ]

    for snapshot_index, snapshot in enumerate(eligible_snapshots):
        mean = (
            initial_theta.copy()
            if initial_theta is not None
            else np.zeros(dimensions, np.float64)
        )
        std = bounds * (0.18 if initial_theta is not None else 0.45)
        protected = mean.copy()
        candidates = [("zero", mean.copy())]
        history = []
        for iteration in range(args.iterations):
            theta = mean + std * rng.standard_normal(
                (args.population, dimensions)
            )
            theta = np.clip(theta, -bounds, bounds)
            theta[0] = protected
            reports_by_model = evaluate_all(
                theta,
                snapshot,
                args.worlds,
                args.seed + 1009 * iteration + 1000003 * snapshot_index,
            )
            scores_by_model = np.asarray([
                [objective(row) for row in reports]
                for reports in reports_by_model
            ])
            scores = scores_by_model.min(0)
            order = np.argsort(-scores)
            winner = int(order[0])
            protected = theta[winner].copy()
            elite = theta[order[:args.elite]]
            mean = 0.30 * mean + 0.70 * elite.mean(0)
            std = 0.45 * std + 0.55 * (elite.std(0) + 0.005)
            candidates.append((f"iteration_{iteration}", protected.copy()))
            row = {
                "iteration": iteration,
                "winner_reports": [reports[winner] for reports in reports_by_model],
                "worst_score": float(scores[winner]),
            }
            history.append(row)
            print(json.dumps({
                "rollback_steps": snapshot["rollback_steps"], **row
            }), flush=True)
        selection_rows = []
        for candidate_index, (label, theta) in enumerate(candidates):
            reports = evaluate_all(
                theta[None],
                snapshot,
                args.selection_worlds,
                args.seed + 77191 + 1000003 * snapshot_index,
            )
            reports = [rows[0] for rows in reports]
            score = min(objective(row) for row in reports)
            selection_rows.append({
                "label": label,
                "theta": theta,
                "reports": reports,
                "score": score,
            })
        selected = max(selection_rows, key=lambda row: row["score"])
        audit = evaluate_all(
            selected["theta"][None],
            snapshot,
            args.audit_worlds,
            args.seed + 221137 + 1000003 * snapshot_index,
        )
        audit = [rows[0] for rows in audit]
        minimum_rate = min(row["finish_rate"] for row in audit)
        overall_rate = float(np.mean([row["finish_rate"] for row in audit]))
        minimum_clearance = min(row["clearance_p10_m"] for row in audit)
        maximum_support = max(row["support_p90"] for row in audit)
        maximum_disagreement = max(
            row["disagreement_p90"] for row in audit
        )
        accepted = (
            minimum_rate >= 0.90
            and overall_rate >= 0.95
            and minimum_clearance > 0.0
            and maximum_support <= args.max_support_p90
            and maximum_disagreement <= args.max_disagreement_p90
        )
        searches.append({
            "snapshot": snapshot,
            "selection_label": selected["label"],
            "residual_knots": np.concatenate([
                selected["theta"][:action_dimensions].reshape(
                    args.knots - 1, 4
                ),
                np.zeros((1, 4)),
            ]).tolist(),
            "reference_geometry": {
                "gate": int(source["gate_index"][snapshot["branch_step"]]),
                "lateral_offset_m": (
                    float(selected["theta"][-3])
                    if args.geometry_search else 0.0
                ),
                "vertical_offset_m": (
                    float(selected["theta"][-2])
                    if args.geometry_search else 0.0
                ),
                "speed_scale": (
                    float(1.0 + selected["theta"][-1])
                    if args.geometry_search else 1.0
                ),
            },
            "selection_reports": selected["reports"],
            "fresh_audit_reports": audit,
            "minimum_family_finish_rate": minimum_rate,
            "overall_finish_rate": overall_rate,
            "minimum_clearance_p10_m": minimum_clearance,
            "maximum_support_p90": maximum_support,
            "maximum_disagreement_p90": maximum_disagreement,
            "accepted": accepted,
            "history": history,
        })
    accepted = [row for row in searches if row["accepted"]]
    chosen = min(
        accepted,
        key=lambda row: row["snapshot"]["rollback_steps"],
        default=max(searches, key=lambda row: row["minimum_family_finish_rate"]),
    )
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_id": metadata["artifact_id"],
        "source_metadata_sha256": sha256(
            args.snapshot_artifact / "metadata.json"
        ),
        "reproduction_report_sha256": sha256(args.reproduction_report),
        "ensemble_sha256": [sha256(path) for path in args.ensemble],
        "config": {
            "population": args.population,
            "elite": args.elite,
            "iterations": args.iterations,
            "worlds": args.worlds,
            "selection_worlds": args.selection_worlds,
            "audit_worlds": args.audit_worlds,
            "knots": args.knots,
            "authority": args.authority,
            "geometry_search": args.geometry_search,
            "max_lateral_offset": args.max_lateral_offset,
            "max_vertical_offset": args.max_vertical_offset,
            "max_speed_scale_delta": args.max_speed_scale_delta,
            "post_steps": args.post_steps,
            "aleatoric_scale": args.aleatoric_scale,
            "position_sigma_scale": position_sigma_scale,
            "position_sigma_floor_m": position_sigma_floor_m,
            "max_support_p90": args.max_support_p90,
            "max_disagreement_p90": args.max_disagreement_p90,
            "residual_scale": 0.25,
            "direct_action_authority": 0.25 * args.authority,
            "seed": args.seed,
            "initial_report_sha256": initial_report_sha256,
        },
        "searches": searches,
        "accepted": bool(accepted),
        "chosen_rollback_steps": chosen["snapshot"]["rollback_steps"],
        "chosen_residual_knots": chosen["residual_knots"],
        "decision": "accepted_for_distillation" if accepted else "rejected",
    }
    args.out.mkdir(parents=True)
    report_path = args.out / "repair_search_report.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "accepted": result["accepted"],
        "chosen_rollback_steps": result["chosen_rollback_steps"],
        "report_sha256": sha256(report_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
