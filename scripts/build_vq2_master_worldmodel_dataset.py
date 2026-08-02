"""Build a deduplicated, session-split world-model corpus from all live runs.

Unlike the original single-run builder, this consumes every compatible
episode NPZ under a training root. The logged post-step localizer position is
used as the transition target; it is a better label than reconstructing the
next position from a possibly historical gate map. Entire sessions stay in a
single split and recent interleaved probes are frozen as test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.worldmodel import decode_observations  # noqa: E402


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(text: str) -> float:
    value = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
    return value / float(16 ** 16)


def normalized_path(path: str | Path) -> str:
    return str(path).replace("/", "\\").rstrip("\\").lower()


def load_split_registry(path: Path) -> tuple[dict[str, str], dict]:
    payload = json.loads(path.read_text())
    assignments: dict[str, str] = {}
    for group in payload.get("groups", []):
        split = str(group.get("split", ""))
        if split not in {"train", "validation", "policy_selection", "final_test"}:
            raise ValueError(f"invalid registry split {split!r}")
        for source_path in group.get("paths", []):
            key = normalized_path(source_path)
            previous = assignments.get(key)
            if previous is not None and previous != split:
                raise ValueError(
                    f"split registry assigns {source_path!r} to both "
                    f"{previous!r} and {split!r}"
                )
            assignments[key] = split
    return assignments, payload


def session_split(path: Path, registry: dict[str, str] | None = None) -> str:
    if registry is not None:
        registered = registry.get(normalized_path(path))
        if registered is None:
            return "train"
        if registered == "policy_selection":
            return "test"
        return registered
    lowered = str(path).lower()
    if "interleave_" in lowered or "schedule_v3_counterexample" in lowered:
        return "test"
    value = stable_fraction(str(path.resolve()).lower())
    if value < 0.10:
        return "test"
    if value < 0.20:
        return "validation"
    return "train"


def summaries(path: Path) -> dict[int, dict]:
    rows = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
            rows[int(row["episode"])] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return rows


def outcome_code(summary: dict) -> int:
    crossed = (
        bool(summary.get("poc_completed"))
        or int(summary.get("gate_reached", -1)) >= 5
        or any(
            int(row.get("gate", -1)) == 4
            for row in summary.get("crossing_offsets", [])
        )
    )
    if crossed:
        return 0
    failure = str(summary.get("failure") or "unknown")
    return {
        "collision": 1,
        "stale_sensor_stream": 2,
        "overspeed": 3,
        "gate_timeout": 4,
        "stagnation": 5,
    }.get(failure, 6)


def episode_timing_eligible(summary: dict) -> bool:
    """Reject episodes quarantined by aggregate live timing checks.

    Older recordings do not carry an episode-level timing verdict, so they
    retain the existing transition-level filtering behavior.  A modern
    explicit ``False`` is authoritative: simulator-step p95/max can make an
    episode unhealthy even when every row's packet-age flag is individually
    true.
    """
    return summary.get("timing_healthy") is not False


def extract_episode(
    path: Path,
    summary: dict,
    gate_positions: np.ndarray,
    episode_id: int,
    session_id: int,
    max_gate: int,
) -> dict[str, np.ndarray] | None:
    if not episode_timing_eligible(summary):
        return None
    try:
        payload = np.load(path, allow_pickle=False)
        observation = np.asarray(payload["observation"], np.float32)
        next_observation = np.asarray(payload["next_observation"], np.float32)
        wire_action = np.asarray(payload["wire_action"], np.float32)
        gate = np.asarray(payload["gate_index"], np.int16)
    except (OSError, KeyError, ValueError):
        return None
    count = len(observation)
    if (
        count < 2
        or next_observation.shape != observation.shape
        or len(wire_action) != count
        or len(gate) != count
    ):
        return None
    current = decode_observations(observation, gate_positions)
    following = decode_observations(next_observation, gate_positions)
    # The trainer records info["position"] after environment.step(). It is
    # therefore the exact next-state position for this action. Reconstruct
    # only the very first current position; every later current state is the
    # preceding transition's logged post-step position.
    if "position" in payload.files:
        post_position = np.asarray(payload["position"], np.float32)
        if post_position.shape == (count, 3):
            current_position = np.vstack([
                current.position[:1], post_position[:-1]
            ])
            next_position = post_position
        else:
            current_position = current.position
            next_position = following.position
    else:
        current_position = current.position
        next_position = following.position
    healthy = np.asarray(
        payload["timing_healthy"]
        if "timing_healthy" in payload.files
        else np.ones(count, bool),
        bool,
    )
    done = np.asarray(payload["done"], np.float32)
    valid = (
        healthy
        & (gate >= 0)
        & (gate <= max_gate)
        & np.isfinite(observation).all(1)
        & np.isfinite(next_observation).all(1)
        & np.isfinite(wire_action).all(1)
        & np.isfinite(current_position).all(1)
        & np.isfinite(next_position).all(1)
    )
    row = np.flatnonzero(valid)
    if not len(row):
        return None
    result = {
        "position": current_position[row].astype(np.float32),
        "velocity": current.velocity[row].astype(np.float32),
        "rotation": current.rotation[row].astype(np.float32),
        "rates": current.rates[row].astype(np.float32),
        "previous_action": current.previous_action[row].astype(np.float32),
        "confidence": current.confidence[row].astype(np.float32),
        "next_position": next_position[row].astype(np.float32),
        "next_velocity": following.velocity[row].astype(np.float32),
        "next_rotation": following.rotation[row].astype(np.float32),
        "next_rates": following.rates[row].astype(np.float32),
        "action": wire_action[row],
        "gate_index": gate[row],
        "done": done[row],
        "gates_passed": np.asarray(
            payload["gates_passed"]
            if "gates_passed" in payload.files else gate,
            np.int16,
        )[row],
        "episode": np.full(len(row), episode_id, np.int32),
        "session": np.full(len(row), session_id, np.int16),
        "step": row.astype(np.int16),
        "outcome": np.full(len(row), outcome_code(summary), np.int8),
    }
    speed_bin = np.minimum(
        (np.linalg.norm(result["velocity"], axis=1) / 2.0).astype(np.int8),
        7,
    )
    action_bin = np.minimum(
        (np.linalg.norm(result["action"] - result["previous_action"], axis=1)
         / 0.15).astype(np.int8),
        7,
    )
    # Physics is course/gate independent. Including gate identity here makes
    # inverse-frequency sampling massively over-weight the handful of late
    # gate rows and hurts the common flight envelope. Balance only genuinely
    # physical coverage: outcome, speed, and command-change magnitude.
    result["stratum"] = (
        result["outcome"].astype(np.int16) * 64
        + speed_bin.astype(np.int16) * 8
        + action_bin.astype(np.int16)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--train-max-gate", type=int,
        help="largest gate included in training (default: final mapped gate)",
    )
    parser.add_argument(
        "--eval-max-gate", type=int, default=4,
        help="largest gate included in validation/test",
    )
    parser.add_argument("--require-vision-hz", type=float)
    parser.add_argument("--require-vision-device")
    parser.add_argument("--require-crop-tracker", action="store_true")
    parser.add_argument("--require-map-basename")
    parser.add_argument(
        "--require-primary-basename",
        help=(
            "only include sessions whose primary gate detector checkpoint "
            "has this basename"
        ),
    )
    parser.add_argument(
        "--split-registry", type=Path,
        help=(
            "immutable session split registry; known final_test sessions are "
            "excluded and newly collected sessions are train-only"
        ),
    )
    args = parser.parse_args()
    gates = json.loads(args.map.read_text())["gates"]
    gate_positions = np.asarray([gate["pos"] for gate in gates], np.float64)
    train_max_gate = (
        len(gates) - 1 if args.train_max_gate is None else args.train_max_gate
    )
    if not 0 <= args.eval_max_gate <= train_max_gate < len(gates):
        parser.error("require 0 <= eval-max-gate <= train-max-gate < gate count")
    all_logs = sorted(args.training_root.glob("**/episodes.jsonl"))
    logs = []
    for log in all_logs:
        try:
            config = json.loads((log.parent / "config.json").read_text())
            run_args = config.get("args", config)
        except (OSError, json.JSONDecodeError):
            run_args = {}
        if (args.require_vision_hz is not None
                and float(run_args.get("vision_hz", -1))
                != args.require_vision_hz):
            continue
        if (args.require_vision_device is not None
                and str(run_args.get("vision_device", ""))
                != args.require_vision_device):
            continue
        if (args.require_crop_tracker
                and not bool(run_args.get("crop_tracker", False))):
            continue
        if (args.require_map_basename is not None
                and Path(str(run_args.get("map", ""))).name
                != args.require_map_basename):
            continue
        if (args.require_primary_basename is not None
                and Path(str(run_args.get("primary", ""))).name
                != args.require_primary_basename):
            continue
        logs.append(log)
    registry_assignments = None
    registry_payload = None
    if args.split_registry is not None:
        registry_assignments, registry_payload = load_split_registry(
            args.split_registry
        )
    split_chunks: dict[str, dict[str, list[np.ndarray]]] = {
        name: defaultdict(list) for name in ("train", "validation", "test")
    }
    seen_hashes = set()
    manifest_sessions = []
    global_episode = 0
    skipped = Counter()
    for session_id, log in enumerate(logs):
        source_split = session_split(log.parent, registry_assignments)
        if source_split == "final_test":
            skipped["frozen_final_test_session"] += 1
            continue
        split = source_split
        rows = summaries(log)
        session_rows = session_episodes = 0
        for episode, summary in sorted(rows.items()):
            path = log.parent / f"episode_{episode:04d}.npz"
            if not path.is_file():
                skipped["missing_npz"] += 1
                continue
            digest = file_hash(path)
            if digest in seen_hashes:
                skipped["duplicate_npz"] += 1
                continue
            seen_hashes.add(digest)
            data = extract_episode(
                path, summary, gate_positions, global_episode, session_id,
                train_max_gate if split == "train" else args.eval_max_gate,
            )
            if data is None:
                skipped["incompatible_or_empty"] += 1
                continue
            for key, value in data.items():
                split_chunks[split][key].append(value)
            session_rows += len(data["action"])
            session_episodes += 1
            global_episode += 1
        if session_episodes:
            manifest_sessions.append({
                "session_id": session_id,
                "path": str(log.parent),
                "split": split,
                "registry_split": (
                    registry_assignments.get(normalized_path(log.parent))
                    if registry_assignments is not None else None
                ),
                "episodes": session_episodes,
                "transitions": session_rows,
            })

    args.out.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split, chunks in split_chunks.items():
        data = {key: np.concatenate(values) for key, values in chunks.items()}
        if split == "train":
            counts = Counter(data["stratum"].tolist())
            weight = np.asarray([
                1.0 / np.sqrt(counts[int(value)]) for value in data["stratum"]
            ], np.float32)
            weight /= np.mean(weight)
            data["sample_weight"] = np.clip(weight, 0.25, 8.0)
        else:
            data["sample_weight"] = np.ones(len(data["action"]), np.float32)
        np.savez_compressed(args.out / f"{split}.npz", **data)
        split_stats[split] = {
            "transitions": int(len(data["action"])),
            "episodes": int(len(np.unique(data["episode"]))),
            "sessions": int(len(np.unique(data["session"]))),
            "gate_counts": {
                str(gate): int(np.sum(data["gate_index"] == gate))
                for gate in range(
                    (train_max_gate if split == "train" else args.eval_max_gate)
                    + 1
                )
            },
            "outcome_counts": {
                str(code): int(np.sum(data["outcome"] == code))
                for code in np.unique(data["outcome"])
            },
        }
        print(split, split_stats[split], flush=True)
    manifest = {
        "training_root": str(args.training_root.resolve()),
        "map": str(args.map.resolve()),
        "source_logs": len(logs),
        "discovered_logs": len(all_logs),
        "unique_episode_files": len(seen_hashes),
        "usable_episodes": global_episode,
        "skipped": dict(skipped),
        "splits": split_stats,
        "sessions": manifest_sessions,
        "position_label": "logged post-step localizer position",
        "split_policy": "whole sessions; interleaved/counterexample probes frozen test",
        "split_registry": (
            {
                "path": str(args.split_registry.resolve()),
                "sha256": file_hash(args.split_registry),
                "generation": registry_payload.get("generation"),
                "new_session_policy": "train",
                "policy_selection_mapping": "test",
                "final_test_policy": "excluded",
            }
            if args.split_registry is not None else None
        ),
        "balance_policy": "outcome x speed x action-change; no gate identity",
        "era_filter": {
            "vision_hz": args.require_vision_hz,
            "vision_device": args.require_vision_device,
            "crop_tracker": True if args.require_crop_tracker else None,
            "map_basename": args.require_map_basename,
            "primary_basename": args.require_primary_basename,
        },
        "train_max_gate": train_max_gate,
        "eval_max_gate": args.eval_max_gate,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: manifest[key] for key in (
        "source_logs", "unique_episode_files", "usable_episodes", "skipped"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
