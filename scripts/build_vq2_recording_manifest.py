"""Inventory VQ flight recordings without copying their heavy payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRUNE = {
    ".git", ".pytest_cache", "__pycache__", "node_modules", "site-packages",
    "frames", "models", "worldmodel", "checkpoints",
}


def file_hash(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def episode_summary(path: Path) -> dict[str, Any]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except (OSError, UnicodeDecodeError):
        pass
    failures = Counter(str(row.get("failure")) for row in rows if row.get("failure"))
    provenance_fields = (
        "config_sha256", "controller_config_sha256",
        "schedule_sha256", "actor_sha256", "secondary_actor_sha256",
        "reference_sha256", "seed_checkpoint_sha256", "map_sha256",
        "primary_detector_sha256", "refiner_detector_sha256",
        "gate_primary_detector_sha256", "crop_detector_sha256",
        "proposal_model_sha256", "calibration_sha256",
        "line_model_sha256",
    )
    provenance = {
        name: sorted({
            str(row.get(name)) for row in rows if row.get(name)
        })
        for name in provenance_fields
    }
    return {
        "episodes": len(rows),
        "finished": sum(bool(row.get("finished")) for row in rows),
        "timing_healthy": sum(row.get("timing_healthy") is True for row in rows),
        "max_gate_reached": max(
            (int(row.get("gate_reached", -1)) for row in rows), default=-1
        ),
        "duration_min_s": min(
            (float(row["duration_s"]) for row in rows if row.get("duration_s") is not None),
            default=None,
        ),
        "failures": dict(sorted(failures.items())),
        **provenance,
    }


def selected_args(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    args = config.get("args", config)
    if not isinstance(args, dict):
        return {}
    names = (
        "demo", "map", "primary", "refiner", "gate_primary", "crop",
        "proposal", "calibration", "line_model", "seed_checkpoint",
        "ppo_residual_checkpoint", "secondary_ppo_residual_checkpoint",
        "ppo_residual_schedule", "eval_only",
        "vision_hz", "vision_device", "crop_tracker", "crop_tracker_hz",
        "teacher_blend", "residual_scale", "poc_stop_after_gate",
    )
    return {name: args.get(name) for name in names if name in args}


def training_entry(host: str, directory: Path) -> dict[str, Any]:
    episodes_path = directory / "episodes.jsonl"
    config_path = directory / "config.json"
    config = read_json(config_path)
    identity = config.get("identity", {}) if isinstance(config, dict) else {}
    summary = episode_summary(episodes_path)
    npz = list(directory.glob("episode_*.npz"))
    raw_pointer = directory / "raw_archive_path.txt"
    try:
        raw_path = raw_pointer.read_text(encoding="utf-8").strip()
    except OSError:
        raw_path = None
    return {
        "kind": "training_session",
        "host": host,
        "path": str(directory.resolve()),
        "session_id": directory.name,
        "config_file_sha256": file_hash(config_path) if config_path.exists() else None,
        "config": selected_args(config),
        "identity": identity if isinstance(identity, dict) else {},
        "episode_npz_count": len(npz),
        "episode_npz_bytes": sum(path.stat().st_size for path in npz),
        "has_steps": (directory / "steps.jsonl").exists(),
        "raw_archive_path": raw_path,
        "summary": summary,
        "eligibility": {
            "dynamics": bool(npz) and summary["timing_healthy"] > 0,
            "localization": bool(raw_path),
            "event_heads": summary["episodes"] > 0,
            "policy": summary["finished"] > 0 or summary["max_gate_reached"] >= 0,
        },
    }


def raw_entry(host: str, directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    config = manifest.get("config") if isinstance(manifest, dict) else None
    frame_index = directory / "frames_index.csv"
    mavlink = directory / "mavlink_rx.bin"
    localizer = directory / "localizer_debug"
    return {
        "kind": "raw_session",
        "host": host,
        "path": str(directory.resolve()),
        "session_id": directory.name,
        "manifest_sha256": file_hash(manifest_path),
        "config": selected_args(config),
        "training_run_dir": (
            manifest.get("training_run_dir") if isinstance(manifest, dict) else None
        ),
        "has_frames_index": frame_index.exists(),
        "has_camera_frames": (directory / "frames").is_dir(),
        "has_mavlink": mavlink.exists(),
        "has_localizer_debug": localizer.is_dir(),
        "eligibility": {
            "dynamics": mavlink.exists(),
            "localization": frame_index.exists() and localizer.is_dir(),
            "event_heads": mavlink.exists(),
            "policy": False,
        },
    }


def control_proof_entry(host: str, directory: Path) -> dict[str, Any]:
    result_path = directory / "result.json"
    replay_path = directory / "replay.jsonl"
    imu_path = directory / "imu_raw.jsonl"
    result = read_json(result_path)
    metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
    checks = result.get("checks", {}) if isinstance(result, dict) else {}
    return {
        "kind": "control_proof_session",
        "host": host,
        "path": str(directory.resolve()),
        "session_id": directory.name,
        "result_sha256": file_hash(result_path),
        "replay_bytes": replay_path.stat().st_size,
        "imu_bytes": imu_path.stat().st_size if imu_path.exists() else 0,
        "decision": result.get("decision") if isinstance(result, dict) else None,
        "duration_s": metrics.get("duration_s"),
        "official_maximum_active_gate": metrics.get(
            "official_maximum_active_gate"
        ),
        "unique_camera_frames_recorded": checks.get(
            "unique_camera_frames_recorded"
        ),
        "eligibility": {
            "dynamics": imu_path.exists(),
            "localization": bool(checks.get("unique_camera_frames_recorded")),
            "event_heads": True,
            "policy": True,
        },
    }


def scan(host: str, roots: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root in roots:
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root):
            directory = Path(current)
            dirs[:] = [
                name for name in dirs
                if name not in PRUNE and not name.startswith(".venv")
            ]
            file_set = set(files)
            if "episodes.jsonl" in file_set:
                key = ("training_session", str(directory.resolve()))
                if key not in seen:
                    entries.append(training_entry(host, directory))
                    seen.add(key)
            if (
                "manifest.json" in file_set
                and ("frames_index.csv" in file_set or "mavlink_rx.bin" in file_set)
            ):
                key = ("raw_session", str(directory.resolve()))
                if key not in seen:
                    entries.append(raw_entry(host, directory))
                    seen.add(key)
                dirs[:] = []
            if "replay.jsonl" in file_set and "result.json" in file_set:
                key = ("control_proof_session", str(directory.resolve()))
                if key not in seen:
                    entries.append(control_proof_entry(host, directory))
                    seen.add(key)
            for name in files:
                if name == "vq2_13finish_demo_v1.npz":
                    path = directory / name
                    key = ("expert_corpus", str(path.resolve()))
                    if key not in seen:
                        entries.append({
                            "kind": "expert_corpus",
                            "host": host,
                            "path": str(path.resolve()),
                            "bytes": path.stat().st_size,
                            "sha256": file_hash(path),
                            "eligibility": {
                                "dynamics": False,
                                "localization": False,
                                "event_heads": False,
                                "policy": True,
                            },
                        })
                        seen.add(key)
    return sorted(entries, key=lambda row: (row["host"], row["kind"], row["path"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("COMPUTERNAME", "unknown"))
    parser.add_argument("--root", action="append", type=Path, default=[])
    parser.add_argument("--merge", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    entries = scan(args.host, args.root)
    for path in args.merge:
        payload = read_json(path)
        if isinstance(payload, dict):
            entries.extend(payload.get("entries", []))
    unique = {
        (str(row.get("host")), str(row.get("kind")), str(row.get("path"))): row
        for row in entries
    }
    entries = sorted(unique.values(), key=lambda row: (
        str(row.get("host")), str(row.get("kind")), str(row.get("path"))
    ))
    payload = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hosts": sorted({str(row.get("host")) for row in entries}),
        "entry_count": len(entries),
        "kind_counts": dict(sorted(Counter(
            str(row.get("kind")) for row in entries
        ).items())),
        "entries": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "entry_count": len(entries),
        "kind_counts": payload["kind_counts"],
        "sha256": file_hash(args.out),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
