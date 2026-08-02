"""Rebuild the shared gates-0..4 dashboard history from archived run logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path


def content_hash(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict) -> str | None:
    if not config:
        return None
    payload = dict(config)
    payload.pop("identity", None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def configured_path(config: dict, key: str) -> Path | None:
    value = config.get("args", {}).get(key)
    if value in {None, "", "None"}:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def official_episode_times(session: Path) -> list[float | None]:
    """Decode exact gate-4 crossing times, preserving failed reset groups."""
    index_path = session / "mavlink_index.csv"
    raw_path = session / "mavlink_rx.bin"
    if not index_path.is_file() or not raw_path.is_file():
        return []
    from pymavlink import mavutil

    header = struct.Struct("<QHI")
    payload = raw_path.read_bytes()
    parser = mavutil.mavlink.MAVLink(None)
    parser.robust_parsing = True
    groups: list[dict] = []
    current: dict | None = None
    saw_pending_countdown = False
    previous_start_ms: int | None = None
    with index_path.open() as stream:
        for row in csv.DictReader(stream):
            if row["message_type"] != "ENCAPSULATED_DATA":
                continue
            offset = int(row["offset"])
            _wall_ns, kind_len, raw_len = header.unpack_from(payload, offset)
            start = offset + header.size + kind_len
            for message in parser.parse_buffer(
                payload[start:start + raw_len]
            ) or []:
                raw = bytes(message.data)
                if not raw or raw[0] != 1:
                    continue
                _, boot_ms, start_ms, _finish_ns, gate, last_gate_ns = (
                    struct.unpack_from("<BQqqIq", raw)
                )
                if int(start_ms) < 0:
                    saw_pending_countdown = True
                starts_race = bool(
                    int(start_ms) >= 0
                    and (
                        previous_start_ms is not None
                        and previous_start_ms < 0
                        or current is None
                        and saw_pending_countdown
                        or current is None
                        and int(boot_ms) < int(start_ms)
                        and int(boot_ms) < 1_000
                    )
                )
                if starts_race:
                    if current is not None and current["race_active"]:
                        groups.append(current)
                    current = {
                        "active": False,
                        "race_active": False,
                        "time_s": None,
                    }
                    saw_pending_countdown = False
                previous_start_ms = int(start_ms)
                if current is None:
                    continue
                if int(start_ms) >= 0 and int(boot_ms) >= int(start_ms):
                    current["race_active"] = True
                if int(gate) > 0:
                    current["active"] = True
                if int(gate) >= 5 and current["time_s"] is None:
                    current["time_s"] = int(last_gate_ns) * 1e-9
    if current is not None and current["race_active"]:
        groups.append(current)
    return [group["time_s"] for group in groups]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--raw-root", type=Path, default=None)
    args = parser.parse_args()
    logs = sorted(
        path for path in args.training_root.rglob("episodes.jsonl")
        if path.is_file()
        and any("g0g4" in part.lower() for part in path.parts)
    )
    rows = []
    for log in logs:
        config_path = log.parent / "config.json"
        try:
            config = json.loads(config_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}
        config_sha256 = config_hash(config)
        schedule_path = configured_path(config, "ppo_residual_schedule")
        actor_path = configured_path(config, "ppo_residual_checkpoint")
        schedule_sha256 = content_hash(schedule_path)
        actor_sha256 = content_hash(actor_path)
        default_arm = (
            "candidate" if schedule_path is not None or actor_path is not None
            else "protected_champion"
        )
        official_times = []
        if args.raw_root is not None:
            official_times = official_episode_times(
                args.raw_root / f"vq2_{log.parent.name}"
            )
        for line in log.read_text().splitlines():
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
            crossing = next((
                item for item in summary.get("crossing_offsets", [])
                if int(item.get("gate", -1)) == 4
            ), None)
            episode = int(summary.get("episode", -1))
            backfilled_official = (
                official_times[episode]
                if 0 <= episode < len(official_times) else None
            )
            official_time = summary.get("official_elapsed_s")
            if official_time is None:
                official_time = backfilled_official
            rows.append({
                "run_number": len(rows),
                "episode": episode,
                "session": log.parent.name,
                "run_id": summary.get("run_id", log.parent.name),
                "config_sha256": summary.get(
                    "config_sha256", config_sha256
                ),
                "schedule_arm": summary.get("schedule_arm", default_arm),
                "schedule_sha256": summary.get(
                    "schedule_sha256",
                    schedule_sha256 or "protected_base",
                ),
                "actor_sha256": summary.get(
                    "actor_sha256", actor_sha256
                ),
                "success": crossing is not None,
                "time_s": (
                    float(official_time)
                    if official_time is not None else
                    (int(crossing["step"]) + 1) / args.control_hz
                    if crossing is not None else None
                ),
                "time_source": (
                    "official"
                    if official_time is not None else
                    "control_step" if crossing is not None else None
                ),
                "failure": summary.get("failure"),
                "deterministic": bool(summary.get("deterministic", False)),
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(
        json.dumps(row, separators=(",", ":")) + "\n" for row in rows
    ))
    print(json.dumps({
        "runs": len(rows),
        "successes": sum(row["success"] for row in rows),
        "best_s": min(
            (row["time_s"] for row in rows if row["time_s"] is not None),
            default=None,
        ),
        "out": str(args.out.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
