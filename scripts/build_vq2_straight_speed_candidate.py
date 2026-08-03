"""Create a live/fastsim config with time-compressed straight segments.

Each ``--segment G:S`` enables sequential reference playback for target gate
G at S demo rows per control step. Body-rate feed-forward scales linearly and
thrust-above-hover scales quadratically, matching the runtime controller's
time-compression convention. Unmentioned gates remain byte-for-byte
behaviorally identical to the base configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIELDS = {
    "reference_sequential_speeds": 1.0,
    "reference_rate_scales": 1.0,
    "reference_thrust_scales": 1.0,
    "reference_velocity_scales": 1.0,
}


def parse_gate_values(text: object) -> dict[int, float]:
    values: dict[int, float] = {}
    for item in str(text or "").split(","):
        if not item.strip():
            continue
        gate, value = item.split(":", 1)
        values[int(gate)] = float(value)
    return values


def format_gate_values(values: dict[int, float]) -> str:
    return ",".join(
        f"{gate}:{values[gate]:.9g}" for gate in sorted(values)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--segment", action="append", default=[])
    args = parser.parse_args()
    if not args.segment:
        parser.error("at least one --segment G:S is required")

    segments: dict[int, float] = {}
    for item in args.segment:
        gate_text, scale_text = item.split(":", 1)
        gate, scale = int(gate_text), float(scale_text)
        if not 0 <= gate <= 16:
            parser.error(f"gate must be in 0..16, got {gate}")
        if not 1.0 < scale <= 2.0:
            parser.error(f"scale must be in (1, 2], got {scale}")
        segments[gate] = scale

    payload = json.loads(args.base.read_text(encoding="utf-8"))
    config = payload.setdefault("args", payload)
    tables = {
        field: parse_gate_values(config.get(field, ""))
        for field in FIELDS
    }
    defaults = {
        "reference_rate_scales": float(
            config.get("reference_rate_scale", 1.0)
        ),
        "reference_thrust_scales": float(
            config.get("reference_thrust_scale", 1.0)
        ),
        "reference_velocity_scales": float(
            config.get("reference_velocity_scale", 1.0)
        ),
    }
    applied: dict[str, dict[str, float]] = {}
    for gate, scale in sorted(segments.items()):
        tables["reference_sequential_speeds"][gate] = scale
        tables["reference_rate_scales"][gate] = (
            tables["reference_rate_scales"].get(
                gate, defaults["reference_rate_scales"]
            ) * scale
        )
        tables["reference_thrust_scales"][gate] = (
            tables["reference_thrust_scales"].get(
                gate, defaults["reference_thrust_scales"]
            ) * scale * scale
        )
        tables["reference_velocity_scales"][gate] = (
            tables["reference_velocity_scales"].get(
                gate, defaults["reference_velocity_scales"]
            ) * scale
        )
        applied[str(gate)] = {
            "time_compression": scale,
            "rate_scale": tables["reference_rate_scales"][gate],
            "thrust_scale": tables["reference_thrust_scales"][gate],
            "velocity_scale": tables["reference_velocity_scales"][gate],
        }
    for field, values in tables.items():
        config[field] = format_gate_values(values)

    payload.pop("identity", None)
    payload["straight_speed_candidate"] = {
        "base": str(args.base.resolve()),
        "base_sha256": sha256(args.base),
        "segments": applied,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "sha256": sha256(args.out),
        "segments": applied,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
