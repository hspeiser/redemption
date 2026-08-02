"""Export immutable multi-horizon snapshots from one real VQ2 failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.worldmodel import decode_observations


PHYSICAL_FAILURES = {"collision", "wrong_side", "off_course", "overspeed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path: Path, episode: int) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("episode", -1)) == episode:
            rows.append(row)
    return sorted(rows, key=lambda row: int(row.get("step", -1)))


def episode_summary(path: Path, episode: int) -> dict:
    rows = jsonl_rows(path, episode)
    if not rows:
        raise ValueError(f"episode {episode} missing from {path}")
    return rows[-1] if len(rows) == 1 else next(
        (
            json.loads(line) for line in path.read_text().splitlines()
            if json.loads(line).get("episode") == episode
        ),
        rows[-1],
    )


def summary_from_log(path: Path, episode: int) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(row.get("episode", -1)) == episode:
            return row
    raise ValueError(f"episode {episode} missing from {path}")


def configured_asset(config: dict, key: str) -> Path | None:
    value = config.get("args", config).get(key)
    if value in {None, "", "None"}:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--matching-success-episode", type=int)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--crop-detector", type=Path)
    parser.add_argument("--world-model", type=Path, action="append", required=True)
    parser.add_argument("--calibration-report", type=Path)
    parser.add_argument("--rollback-steps", default="6,12,21,30")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable repair output already exists: {args.out}")

    episode_path = args.run_dir / f"episode_{args.episode:04d}.npz"
    config_path = args.run_dir / "config.json"
    episodes_log = args.run_dir / "episodes.jsonl"
    steps_log = args.run_dir / "steps.jsonl"
    for path in (episode_path, config_path, episodes_log, steps_log, args.map):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text())
    summary = summary_from_log(episodes_log, args.episode)
    steps = jsonl_rows(steps_log, args.episode)
    payload = np.load(episode_path, allow_pickle=False)
    observation = np.asarray(payload["observation"], np.float32)
    next_observation = np.asarray(payload["next_observation"], np.float32)
    gate = np.asarray(payload["gate_index"], np.int16)
    count = len(observation)
    if count != len(steps):
        raise ValueError(f"NPZ has {count} rows but step log has {len(steps)}")

    gate_payload = json.loads(args.map.read_text())
    gate_positions = np.asarray(
        [row["pos"] for row in gate_payload["gates"]], np.float32
    )
    state = decode_observations(observation, gate_positions)
    next_state = decode_observations(next_observation, gate_positions)
    post_position = np.asarray(payload["position"], np.float32)
    position = np.vstack([state.position[:1], post_position[:-1]])
    terminal_step = count - 1
    rollback_steps = sorted({
        int(value.strip()) for value in args.rollback_steps.split(",")
        if value.strip() and int(value.strip()) > 0
    })
    branch_steps = np.asarray([
        max(0, terminal_step - value) for value in rollback_steps
    ], np.int32)

    timing_healthy = bool(
        summary.get("timing_healthy", True)
        and np.asarray(payload.get("timing_healthy", np.ones(count, bool))).all()
    )
    terminal_cause = summary.get("failure")
    target_gate = int(summary.get("gate_reached", gate[-1]))
    reasons = []
    if terminal_cause not in PHYSICAL_FAILURES:
        reasons.append(f"unsupported_terminal:{terminal_cause}")
    if not timing_healthy:
        reasons.append("timing_unhealthy")
    if not np.isfinite(observation).all():
        reasons.append("non_finite_observation")

    args.out.mkdir(parents=True)
    data_path = args.out / "source_trajectory.npz"
    np.savez_compressed(
        data_path,
        branch_step=branch_steps,
        rollback_steps=np.asarray(rollback_steps, np.int16),
        observation=observation,
        next_observation=next_observation,
        position=position.astype(np.float32),
        next_position=post_position.astype(np.float32),
        velocity=state.velocity.astype(np.float32),
        next_velocity=next_state.velocity.astype(np.float32),
        rotation=state.rotation.astype(np.float32),
        next_rotation=next_state.rotation.astype(np.float32),
        rates=state.rates.astype(np.float32),
        next_rates=next_state.rates.astype(np.float32),
        previous_action=state.previous_action.astype(np.float32),
        action=np.asarray(payload["action"], np.float32),
        wire_action=np.asarray(payload["wire_action"], np.float32),
        protected_action=np.asarray(payload["teacher_action"], np.float32),
        gate_index=gate,
        gates_passed=np.asarray(payload["gates_passed"], np.int16),
        timing_healthy=np.asarray(payload["timing_healthy"], bool),
        position_sigma_m=np.asarray([
            row.get("position_sigma_m", np.nan) for row in steps
        ], np.float32),
        landmark_age_s=np.asarray([
            row.get("visual_age_s", np.nan) for row in steps
        ], np.float32),
        sim_time_s=np.asarray([
            row.get("sim_time_s", np.nan) for row in steps
        ], np.float64),
        localizer_source=np.asarray([
            str(row.get("localizer_source", "")) for row in steps
        ]),
    )

    config_identity = config.get("identity", {})
    snapshots = []
    for rollback, branch in zip(rollback_steps, branch_steps):
        row = steps[int(branch)]
        snapshots.append({
            "rollback_steps": int(rollback),
            "rollback_s": float(rollback / args.control_hz),
            "branch_step": int(branch),
            "target_gate": int(gate[branch]),
            "position_sigma_m": (
                float(row["position_sigma_m"])
                if row.get("position_sigma_m") is not None else None
            ),
            "landmark_age_s": (
                float(row["visual_age_s"])
                if row.get("visual_age_s") is not None else None
            ),
            "localizer_source": row.get("localizer_source"),
        })
    matching_hash = None
    if args.matching_success_episode is not None:
        matching_path = args.run_dir / (
            f"episode_{args.matching_success_episode:04d}.npz"
        )
        if not matching_path.is_file():
            raise FileNotFoundError(matching_path)
        matching_hash = sha256(matching_path)
    identity_material = (
        sha256(episode_path)
        + sha256(config_path)
        + sha256(args.map)
        + "".join(sha256(path) for path in args.world_model)
        + ",".join(str(value) for value in rollback_steps)
    )
    artifact_id = hashlib.sha256(identity_material.encode()).hexdigest()
    metadata = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "synthetic_repair": False,
        "source": {
            "session": args.run_dir.name,
            "episode": args.episode,
            "run_dir": str(args.run_dir.resolve()),
            "episode_npz_sha256": sha256(episode_path),
            "matching_success_episode": args.matching_success_episode,
            "matching_success_npz_sha256": matching_hash,
        },
        "identity": {
            "config_sha256": sha256(config_path),
            "runtime_config_sha256": config_identity.get("config_sha256"),
            "map_sha256": sha256(args.map),
            "reference_sha256": config_identity.get("candidate_reference_sha256"),
            "detector_sha256": (
                sha256(args.detector)
                if args.detector and args.detector.is_file()
                else (
                    sha256(configured_asset(config, "gate_primary"))
                    if configured_asset(config, "gate_primary") else None
                )
            ),
            "crop_detector_sha256": (
                sha256(args.crop_detector)
                if args.crop_detector and args.crop_detector.is_file()
                else (
                    sha256(configured_asset(config, "crop"))
                    if configured_asset(config, "crop") else None
                )
            ),
            "world_model_sha256": [sha256(path) for path in args.world_model],
            "calibration_report_sha256": (
                sha256(args.calibration_report)
                if args.calibration_report and args.calibration_report.is_file()
                else None
            ),
        },
        "classification": {
            "terminal_cause": terminal_cause,
            "target_gate": max(0, min(16, target_gate)),
            "timing_healthy": timing_healthy,
            "eligible": not reasons,
            "rejection_reasons": reasons,
            "state_source": "localizer_belief_proxy",
        },
        "snapshots": snapshots,
        "payload": {
            "file": data_path.name,
            "sha256": sha256(data_path),
            "action_order": ["roll_rate", "pitch_rate", "yaw_rate", "thrust"],
            "state_source": "localizer_belief_proxy",
        },
    }
    metadata_path = args.out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "artifact_id": artifact_id,
        "eligible": not reasons,
        "target_gate": target_gate,
        "snapshots": len(snapshots),
        "metadata_sha256": sha256(metadata_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
