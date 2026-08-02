"""Export accepted counterfactual branches as actor-only synthetic replay.

The generated rows are deliberately marked ineligible for dynamics training
and critic updates.  They may supervise a copied residual actor at reduced
weight after the repair has passed the independent audit.
"""

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
    repair_race_gates,
    restore_branch_cloud,
)
from aigp.fastsim.env import FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.lineopt import load_oriented_gates  # noqa: E402
from aigp.fastsim.live_teacher import LiveTeacherController  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from scripts.fastsim_train_ppo import (  # noqa: E402
    load_demo_states,
    load_live_teacher_config,
    load_schedule_gate_positions,
)
from scripts.search_vq2_counterfactual_repair import spline_action  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def collect_family(
    *,
    source: dict[str, np.ndarray],
    snapshot: dict,
    search: dict,
    failure_gate: int,
    ensemble_path: Path,
    base: SurrogateModel,
    demo_states: dict,
    demo_path: Path,
    map_path: Path,
    obstacles_path: Path,
    teacher_config: Path,
    worlds: int,
    post_steps: int,
    aleatoric_scale: float,
    seed: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict]:
    torch.manual_seed(seed)
    branch_step = int(snapshot["branch_step"])
    knots_np = np.asarray(search["residual_knots"], np.float32)
    knots = torch.as_tensor(knots_np[None], device=device)
    geometry = search["reference_geometry"]
    geometry_gate = int(geometry["gate"])
    tail_steps = len(source["action"]) - branch_step
    total_steps = max(tail_steps + post_steps, len(knots_np))
    cfg = FastEnvConfig(
        n_envs=worlds,
        race_gates=repair_race_gates(failure_gate),
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
        teacher_config,
        worlds,
        gate_count=repair_race_gates(failure_gate),
        map_path=map_path,
    )
    gate_count = arrays["action_leads"].shape[1]
    offsets = arrays.setdefault(
        "reference_gate_offsets_world",
        np.zeros((worlds, gate_count, 3), np.float32),
    )
    _positions, rotations = load_oriented_gates(map_path)
    offsets[:, geometry_gate] += (
        float(geometry["lateral_offset_m"])
        * rotations[geometry_gate, :, 0]
        + float(geometry["vertical_offset_m"])
        * rotations[geometry_gate, :, 2]
    ).astype(np.float32)
    arrays["trajectory_velocity_scales"][:, geometry_gate] *= float(
        geometry["speed_scale"]
    )
    fixed.setdefault(
        "schedule_gate_positions", load_schedule_gate_positions(map_path)
    )
    backbone = LiveTeacherController(
        demo_path,
        worlds,
        device=device,
        rate_gain=np.asarray(base.rate_gain),
        **arrays,
        **fixed,
    )
    ensemble, ensemble_metadata = ResidualEnsemble.load(
        ensemble_path, device
    )
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
    sigma = float(source["position_sigma_m"][branch_step])
    age = float(source["landmark_age_s"][branch_step])
    restore_branch_cloud(
        env,
        position=source["position"][branch_step],
        velocity=source["velocity"][branch_step],
        rotation=source["rotation"][branch_step],
        rates=source["rates"][branch_step],
        previous_action=source["previous_action"][branch_step],
        target_gate=int(source["gate_index"][branch_step]),
        position_sigma_m=sigma if np.isfinite(sigma) else 0.10,
        landmark_age_s=age if np.isfinite(age) else 0.25,
        seed=seed,
    )
    passed = torch.zeros(worlds, dtype=torch.bool, device=device)
    failed = torch.zeros_like(passed)
    post_count = torch.zeros(worlds, dtype=torch.long, device=device)
    trajectory = []
    for step in range(total_steps):
        active_before = ~failed
        observation = env.observations().clone()
        gate_before = env.target.clone()
        residual = spline_action(knots, step, total_steps, worlds)
        next_observation, reward, done, info = env.step(residual)
        target_pass = (
            (info["cross_gate"] == failure_gate)
            & info["passed"]
            & ~info["hit"]
        )
        passed |= target_pass & ~failed
        failed |= done & ~passed
        failed |= info["hit"]
        post_count += (passed & ~failed).long()
        trajectory.append({
            "observation": observation.cpu().numpy(),
            "action": residual.cpu().numpy(),
            "reward": reward.cpu().numpy(),
            "next_observation": next_observation.cpu().numpy(),
            "gate_index": gate_before.cpu().numpy(),
            "active": active_before.cpu().numpy(),
        })
    success = (passed & ~failed & (post_count >= post_steps)).cpu().numpy()
    payload: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "observation", "action", "reward", "next_observation",
            "gate_index", "world_id", "branch_step_index",
        )
    }
    for step, row in enumerate(trajectory):
        keep = success & row["active"]
        count = int(keep.sum())
        if not count:
            continue
        for key in (
            "observation", "action", "reward", "next_observation",
            "gate_index",
        ):
            payload[key].append(row[key][keep])
        payload["world_id"].append(np.flatnonzero(keep).astype(np.int32))
        payload["branch_step_index"].append(
            np.full(count, step, np.int16)
        )
    result = {
        key: np.concatenate(value, axis=0)
        for key, value in payload.items()
    }
    count = len(result["reward"])
    result.update({
        "done": np.zeros(count, bool),
        "discount": np.ones(count, np.float32),
        "is_demo": np.ones(count, bool),
        "synthetic_repair": np.ones(count, bool),
        "actor_weight": np.full(count, 0.25, np.float32),
        "critic_weight": np.zeros(count, np.float32),
        "dynamics_eligible": np.zeros(count, bool),
    })
    return result, {
        "ensemble": str(ensemble_path.resolve()),
        "ensemble_sha256": sha256(ensemble_path),
        "source_dataset": ensemble_metadata.get("dataset"),
        "worlds": worlds,
        "successful_worlds": int(success.sum()),
        "success_rate": float(success.mean()),
        "exported_rows": count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, action="append", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable export exists: {args.out}")
    repair = json.loads(args.repair_report.read_text())
    if not repair.get("accepted"):
        raise ValueError("repair report was not accepted for distillation")
    metadata = json.loads(
        (args.snapshot_artifact / "metadata.json").read_text()
    )
    source_npz = np.load(
        args.snapshot_artifact / "source_trajectory.npz", allow_pickle=False
    )
    source = {key: np.asarray(source_npz[key]) for key in source_npz.files}
    chosen_rollback = int(repair["chosen_rollback_steps"])
    search = next(
        row for row in repair["searches"]
        if int(row["snapshot"]["rollback_steps"]) == chosen_rollback
    )
    base = SurrogateModel.load(args.base_model)
    demo_states = load_demo_states(args.demo, args.map)
    families = []
    reports = []
    for index, ensemble in enumerate(args.ensemble):
        rows, report = collect_family(
            source=source,
            snapshot=search["snapshot"],
            search=search,
            failure_gate=int(metadata["classification"]["target_gate"]),
            ensemble_path=ensemble,
            base=base,
            demo_states=demo_states,
            demo_path=args.demo,
            map_path=args.map,
            obstacles_path=args.obstacles,
            teacher_config=args.teacher_config,
            worlds=args.worlds,
            post_steps=int(repair["config"]["post_steps"]),
            aleatoric_scale=float(repair["config"]["aleatoric_scale"]),
            seed=args.seed + 100003 * index,
            device=args.device,
        )
        rows["model_family"] = np.full(
            len(rows["reward"]), index, np.int16
        )
        families.append(rows)
        reports.append(report)
    keys = families[0].keys()
    merged = {key: np.concatenate([row[key] for row in families]) for key in keys}
    args.out.mkdir(parents=True)
    data_path = args.out / "synthetic_actor_replay.npz"
    np.savez_compressed(data_path, **merged)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repair_report_sha256": sha256(args.repair_report),
        "source_artifact_id": metadata["artifact_id"],
        "source_metadata_sha256": sha256(
            args.snapshot_artifact / "metadata.json"
        ),
        "rows": len(merged["reward"]),
        "actor_weight": 0.25,
        "critic_weight": 0.0,
        "dynamics_eligible": False,
        "families": reports,
        "payload": data_path.name,
        "payload_sha256": sha256(data_path),
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "rows": manifest["rows"],
        "family_success_rates": [row["success_rate"] for row in reports],
        "manifest_sha256": sha256(manifest_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
