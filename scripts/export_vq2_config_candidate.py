"""Export one live config into the g0-g4 schedule-candidate format."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aigp.fastsim.lineopt import load_oriented_gates
from scripts.fastsim_train_ppo import load_live_teacher_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate_values(values: np.ndarray, *, integer: bool = False) -> str:
    if integer:
        return ",".join(
            f"{gate}:{int(round(float(value)))}"
            for gate, value in enumerate(values)
        )
    return ",".join(
        f"{gate}:{float(value):.9g}"
        for gate, value in enumerate(values)
    )


def candidate_payload(config: Path, map_path: Path) -> dict:
    """Convert an exact live config to optimizer schedule arrays."""
    arrays, fixed = load_live_teacher_config(config, 1, map_path=map_path)
    leads = np.asarray(arrays["action_leads"][0], float)
    thrust = np.asarray(arrays["thrust_scales"][0], float)
    velocity = np.asarray(arrays["trajectory_velocity_scales"][0], float)
    blend = np.asarray(arrays["trajectory_blends"][0], float)
    rate = np.asarray(fixed.get("reference_rate_scales", np.ones(5)), float)
    world_offset = np.asarray(
        arrays.get("reference_gate_offsets_world", np.zeros((1, 5, 3)))[0],
        float,
    )
    _positions, frames = load_oriented_gates(map_path)
    lateral = np.sum(world_offset * frames[:5, :, 0], axis=1)
    vertical = np.sum(world_offset * frames[:5, :, 2], axis=1)
    return {
        "leads": np.rint(leads).astype(int).tolist(),
        "thrust_scales": thrust.tolist(),
        "velocity_scales": velocity.tolist(),
        "trajectory_blends": blend.tolist(),
        "lateral_offsets_m": lateral.tolist(),
        "vertical_offsets_m": vertical.tolist(),
        "rate_scales": rate.tolist(),
        "live_overrides": {
            "reference_action_leads": gate_values(leads, integer=True),
            "reference_thrust_scales": gate_values(thrust),
            "reference_velocity_scales": gate_values(velocity),
            "reference_rate_scales": gate_values(rate),
            "trajectory_blends": gate_values(blend),
            "reference_lateral_offsets": gate_values(lateral),
            "reference_vertical_offsets": gate_values(vertical),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = args.config.resolve(strict=True)
    map_path = args.map.resolve(strict=True)
    out = args.out.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing artifact: {out}")

    payload = candidate_payload(config, map_path)
    payload["exported_live_config"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config),
        "config_sha256": sha256(config),
        "map": str(map_path),
        "map_sha256": sha256(map_path),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), **payload["exported_live_config"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
