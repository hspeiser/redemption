"""Build an immutable golden-action fixture from a deployed VQ2 episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TELEMETRY_FLOAT_FIELDS = (
    "selected_reference_row",
    "effective_lateral_gain_scale",
    "effective_reference_action_lead",
    "effective_reference_velocity_scale",
    "effective_reference_sequential_speed",
    "effective_reference_thrust_scale",
    "effective_residual_scale",
    "effective_teacher_blend",
    "effective_trajectory_blend",
    "raw_lateral_feedback",
    "clipped_lateral_feedback",
    "raw_longitudinal_feedback",
    "clipped_longitudinal_feedback",
    "lateral_position_error_m",
    "lateral_velocity_error_mps",
    "gate_center_funnel_weight",
)

TELEMETRY_INT_FIELDS = (
    "control_gate_index",
    "reference_segment_start",
    "reference_segment_end",
    "residual_schedule_gate",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fixture(
    episode_path: Path,
    steps_path: Path,
    episode_index: int,
) -> dict[str, np.ndarray]:
    episode = np.load(episode_path, allow_pickle=False)
    rows = []
    with steps_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["episode"]) == episode_index:
                rows.append(row)
    n = len(episode["observation"])
    if len(rows) != n:
        raise ValueError(
            f"telemetry/episode length mismatch: {len(rows)} != {n}"
        )

    final_action = np.asarray([r["action"] for r in rows], np.float32)
    archived_final = np.asarray(episode["wire_action"], np.float32)
    action_error = float(np.max(np.abs(final_action - archived_final)))
    if action_error > 1e-6:
        raise ValueError(
            f"steps action does not match archived final action: {action_error}"
        )

    output: dict[str, np.ndarray] = {
        "observation": np.asarray(episode["observation"], np.float32),
        "final_action": final_action,
        "teacher_action": np.asarray(episode["teacher_action"], np.float32),
        "actor_mean": np.asarray(episode["actor_mean"], np.float32),
        "selected_reference_row_int": np.asarray(
            episode["selected_reference_row"], np.int32
        ),
        "reference_row_int": np.asarray(episode["reference_row"], np.int32),
        "position": np.asarray(episode["position"], np.float32),
        "gate_index": np.asarray(episode["gate_index"], np.int16),
        "actor_source": np.asarray(
            [str(r["residual_actor_source"]) for r in rows], dtype="U24"
        ),
        "trajectory_blend_active": np.asarray(
            [bool(r["trajectory_blend_active"]) for r in rows], bool
        ),
        "residual_phase_active": np.asarray(
            [bool(r["residual_phase_active"]) for r in rows], bool
        ),
        "residual_schedule_active": np.asarray(
            [bool(r["residual_schedule_active"]) for r in rows], bool
        ),
    }
    for field in TELEMETRY_FLOAT_FIELDS:
        output[field] = np.asarray(
            [np.nan if r.get(field) is None else r[field] for r in rows],
            np.float32,
        )
    for field in TELEMETRY_INT_FIELDS:
        output[field] = np.asarray(
            [-1 if r.get(field) is None else r[field] for r in rows],
            np.int32,
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    payload = build_fixture(args.episode, args.steps, args.episode_index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    manifest_path = args.manifest or args.out.with_suffix(".json")
    manifest = {
        "schema": "vq2_teacher_parity_v1",
        "rows": int(len(payload["observation"])),
        "observation_dim": int(payload["observation"].shape[1]),
        "episode_index": args.episode_index,
        "episode_source": str(args.episode.resolve()),
        "episode_sha256": sha256(args.episode),
        "steps_source": str(args.steps.resolve()),
        "steps_sha256": sha256(args.steps),
        "fixture": str(args.out.resolve()),
        "fixture_sha256": sha256(args.out),
        "config_source": str(args.config.resolve()) if args.config else None,
        "config_sha256": sha256(args.config) if args.config else None,
        "required_action_max_abs_error": 1e-5,
        "required_exact_fields": [
            "actor_source",
            "gate_index",
            "selected_reference_row_int",
            "reference_segment_start",
            "reference_segment_end",
            "trajectory_blend_active",
            "residual_phase_active",
            "residual_schedule_active",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
