"""Copy a complete live config while replacing only its reference demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gate_values(text: object) -> dict[int, float]:
    result: dict[int, float] = {}
    for item in str(text or "").split(","):
        if item.strip():
            gate, value = item.split(":", 1)
            result[int(gate)] = float(value)
    return result


def format_gate_values(values: dict[int, float]) -> str:
    return ",".join(
        f"{gate}:{values[gate]:.9g}" for gate in sorted(values)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trajectory-blend-gates", default="")
    parser.add_argument("--trajectory-blend", type=float, default=0.0)
    parser.add_argument(
        "--lateral-gain-overrides", default="",
        help="comma-separated gate:value updates",
    )
    parser.add_argument(
        "--lateral-bias-overrides", default="",
        help="comma-separated gate:value updates",
    )
    parser.add_argument(
        "--velocity-scale-overrides", default="",
        help="comma-separated gate:value reference velocity updates",
    )
    parser.add_argument(
        "--speed-cap", type=float, default=None,
        help="optional live overspeed termination threshold",
    )
    args = parser.parse_args()
    payload = json.loads(args.base.read_text())
    config = payload.setdefault("args", payload)
    config["demo"] = str(args.demo.resolve(strict=True))
    config["line"] = None
    if args.speed_cap is not None:
        if args.speed_cap <= 0.0:
            parser.error("--speed-cap must be positive")
        config["speed_cap"] = float(args.speed_cap)
    blend_gates = [
        int(value.strip())
        for value in args.trajectory_blend_gates.split(",")
        if value.strip()
    ]
    if blend_gates:
        if not 0.0 <= args.trajectory_blend <= 1.0:
            parser.error("--trajectory-blend must be in [0, 1]")
        blends = parse_gate_values(config.get("trajectory_blends", ""))
        for gate in blend_gates:
            if not 0 <= gate <= 16:
                parser.error("trajectory blend gate must be in 0..16")
            blends[gate] = args.trajectory_blend
        config["trajectory_blends"] = format_gate_values(blends)
    gain_overrides = parse_gate_values(args.lateral_gain_overrides)
    bias_overrides = parse_gate_values(args.lateral_bias_overrides)
    velocity_overrides = parse_gate_values(args.velocity_scale_overrides)
    for name, overrides in (
        ("lateral_gain_scales", gain_overrides),
        ("extra_lateral_biases", bias_overrides),
        ("reference_velocity_scales", velocity_overrides),
    ):
        values = parse_gate_values(config.get(name, ""))
        values.update(overrides)
        config[name] = format_gate_values(values)
    payload.pop("identity", None)
    payload["demo_candidate"] = {
        "base": str(args.base.resolve(strict=True)),
        "base_sha256": sha256(args.base),
        "demo": str(args.demo.resolve(strict=True)),
        "demo_sha256": sha256(args.demo),
        "trajectory_blend_gates": blend_gates,
        "trajectory_blend": float(args.trajectory_blend),
        "lateral_gain_overrides": gain_overrides,
        "lateral_bias_overrides": bias_overrides,
        "velocity_scale_overrides": velocity_overrides,
        "speed_cap": args.speed_cap,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
