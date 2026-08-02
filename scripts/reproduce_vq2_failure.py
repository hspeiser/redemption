"""Gate counterfactual search on faithful reproduction of a real failure."""

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

from aigp.fastsim.branching import (
    calibrated_position_sigma,
    restore_branch_cloud,
)
from aigp.fastsim.env import FastEnvConfig, FastVQ2Env
from aigp.fastsim.sysid import SurrogateModel
from aigp.fastsim.worldmodel import ResidualEnsemble
from scripts.fastsim_train_ppo import load_demo_states


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def reproduce(
    source: dict[str, np.ndarray],
    branch_step: int,
    *,
    failure_target_gate: int,
    ensemble_path: Path,
    base_model: SurrogateModel,
    demo_states: dict,
    map_path: Path,
    obstacles_path: Path,
    worlds: int,
    aleatoric_scale: float,
    position_sigma_scale: float,
    position_sigma_floor_m: float,
    seed: int,
    device: str,
) -> dict:
    cfg = FastEnvConfig(
        n_envs=worlds,
        # Counterfactual repair originally targeted gates 0-4.  Keeping this
        # hard-coded at five makes any branch already targeting gate 5+ end
        # immediately as a wrong terminal on its first simulated step.
        race_gates=max(5, failure_target_gate + 1),
        random_start_frac=0.0,
        spawn_at_rest=True,
        max_episode_s=3.0,
        act_delay_steps_min=0,
        act_delay_steps_max=0,
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
    ensemble, metadata = ResidualEnsemble.load(ensemble_path, device)
    ensemble.eval()
    env = FastVQ2Env(
        base_model,
        map_path,
        demo_states=demo_states,
        config=cfg,
        device=device,
        obstacles_path=obstacles_path,
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
    active = torch.ones(worlds, dtype=torch.bool, device=device)
    hit = torch.zeros_like(active)
    passed = torch.zeros_like(active)
    wrong_terminal = torch.zeros_like(active)
    time_to_terminal = torch.full(
        (worlds,), float("nan"), dtype=torch.float32, device=device
    )
    cross_r = torch.full(
        (worlds,), float("nan"), dtype=torch.float32, device=device
    )
    branch_target_gate = int(source["gate_index"][branch_step])
    target_gate = int(failure_target_gate)
    actions = source["wire_action"][branch_step:]
    if len(actions):
        actions = np.concatenate([
            actions,
            np.repeat(actions[-1][None], 6, axis=0),
        ])
    for action in actions:
        action_batch = torch.as_tensor(
            action, dtype=torch.float32, device=device
        ).expand(worlds, -1)
        current_target = env.target.clone()
        _obs, _reward, done, info = env.step(action_batch)
        same_gate_cross = active & (info["cross_gate"] == target_gate)
        passed |= same_gate_cross & info["passed"] & ~info["hit"]
        hit_now = active & info["hit"] & (current_target == target_gate)
        hit |= hit_now
        wrong_now = active & done & ~hit_now & ~passed
        wrong_terminal |= wrong_now
        first_terminal = active & (hit_now | passed | wrong_now)
        time_to_terminal = torch.where(
            first_terminal, info["t_ep"], time_to_terminal
        )
        cross_r = torch.where(
            same_gate_cross, info["cross_r"], cross_r
        )
        active &= ~first_terminal
        if not bool(active.any()):
            break
    hit_np = hit.cpu().numpy()
    terminal_np = time_to_terminal.cpu().numpy()
    cross_np = cross_r.cpu().numpy()
    valid_terminal = terminal_np[np.isfinite(terminal_np)]
    valid_cross = cross_np[np.isfinite(cross_np)]
    return {
        "ensemble": str(ensemble_path.resolve()),
        "ensemble_sha256": sha256(ensemble_path),
        "source_dataset": metadata.get("dataset"),
        "worlds": worlds,
        "collision_reproduction_rate": float(hit.float().mean().cpu()),
        "same_gate_pass_rate": float(passed.float().mean().cpu()),
        "wrong_terminal_rate": float(wrong_terminal.float().mean().cpu()),
        "unresolved_rate": float(active.float().mean().cpu()),
        "time_to_terminal_median_s": (
            float(np.median(valid_terminal)) if len(valid_terminal) else None
        ),
        "cross_r_median_m": (
            float(np.median(valid_cross)) if len(valid_cross) else None
        ),
        "collision_count": int(hit_np.sum()),
        "branch_target_gate": branch_target_gate,
        "failure_target_gate": target_gate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-artifact", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, action="append", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--aleatoric-scale", type=float, default=1.0)
    parser.add_argument("--position-sigma-scale", type=float, default=1.0)
    parser.add_argument("--position-sigma-floor-m", type=float, default=0.0)
    parser.add_argument("--minimum-family-rate", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable reproduction output exists: {args.out}")

    metadata_path = args.snapshot_artifact / "metadata.json"
    source_path = args.snapshot_artifact / "source_trajectory.npz"
    metadata = json.loads(metadata_path.read_text())
    source_npz = np.load(source_path, allow_pickle=False)
    source = {key: np.asarray(source_npz[key]) for key in source_npz.files}
    base = SurrogateModel.load(args.base_model)
    demo = load_demo_states(args.demo, args.map)
    rows = []
    failure_target_gate = int(metadata["classification"]["target_gate"])
    for snapshot_index, branch_step in enumerate(source["branch_step"]):
        by_model = []
        for model_index, ensemble_path in enumerate(args.ensemble):
            by_model.append(reproduce(
                source,
                int(branch_step),
                failure_target_gate=failure_target_gate,
                ensemble_path=ensemble_path,
                base_model=base,
                demo_states=demo,
                map_path=args.map,
                obstacles_path=args.obstacles,
                worlds=args.worlds,
                aleatoric_scale=args.aleatoric_scale,
                position_sigma_scale=args.position_sigma_scale,
                position_sigma_floor_m=args.position_sigma_floor_m,
                seed=args.seed + 1009 * snapshot_index + 100003 * model_index,
                device=args.device,
            ))
        minimum = min(row["collision_reproduction_rate"] for row in by_model)
        rows.append({
            "snapshot": metadata["snapshots"][snapshot_index],
            "models": by_model,
            "minimum_family_reproduction_rate": minimum,
            "eligible_for_repair_search": minimum >= args.minimum_family_rate,
        })
    eligible = [row for row in rows if row["eligible_for_repair_search"]]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_artifact_id": metadata["artifact_id"],
        "source_metadata_sha256": sha256(metadata_path),
        "source_payload_sha256": sha256(source_path),
        "seed": args.seed,
        "worlds_per_model": args.worlds,
        "minimum_family_rate": args.minimum_family_rate,
        "position_sigma_scale": args.position_sigma_scale,
        "position_sigma_floor_m": args.position_sigma_floor_m,
        "snapshots": rows,
        "eligible": bool(eligible),
        "eligible_rollback_steps": [
            row["snapshot"]["rollback_steps"] for row in eligible
        ],
    }
    args.out.mkdir(parents=True)
    report_path = args.out / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "eligible": report["eligible"],
        "eligible_rollback_steps": report["eligible_rollback_steps"],
        "report_sha256": sha256(report_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
