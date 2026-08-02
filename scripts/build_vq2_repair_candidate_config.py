"""Apply an accepted repair's deployable geometry and actor to a config."""

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


def gate_values(text: str) -> dict[int, float]:
    result = {}
    for item in str(text).split(","):
        if not item.strip():
            continue
        gate, value = item.split(":", 1)
        result[int(gate)] = float(value)
    return result


def format_gate_values(values: dict[int, float]) -> str:
    return ",".join(
        f"{gate}:{value:.9g}" for gate, value in sorted(values.items())
    )


def gate_phase_windows(text: str) -> dict[int, tuple[float, float]]:
    """Parse ``gate:start:end`` windows without dropping protected gates."""
    result = {}
    for item in str(text).split(","):
        if not item.strip():
            continue
        gate, start, end = item.split(":", 2)
        result[int(gate)] = (float(start), float(end))
    return result


def format_gate_phase_windows(
    values: dict[int, tuple[float, float]],
) -> str:
    return ",".join(
        f"{gate}:{start:.9g}:{end:.9g}"
        for gate, (start, end) in sorted(values.items())
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--distilled-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--residual-phase-start", type=float, default=0.75)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable candidate config exists: {args.out}")
    config = json.loads(args.base_config.read_text())
    repair = json.loads(args.repair_report.read_text())
    if not repair.get("accepted"):
        raise ValueError("repair report is not accepted")
    search = next(
        row for row in repair["searches"]
        if row["snapshot"]["rollback_steps"]
        == repair["chosen_rollback_steps"]
    )
    geometry = search["reference_geometry"]
    gate = int(geometry["gate"])
    cfg = config["args"]
    lateral = gate_values(cfg.get("reference_lateral_offsets", ""))
    vertical = gate_values(cfg.get("reference_vertical_offsets", ""))
    velocity = gate_values(cfg.get("reference_velocity_scales", ""))
    lateral[gate] = lateral.get(gate, 0.0) + float(
        geometry["lateral_offset_m"]
    )
    vertical[gate] = vertical.get(gate, 0.0) + float(
        geometry["vertical_offset_m"]
    )
    baseline_velocity = velocity.get(
        gate, float(cfg.get("reference_velocity_scale", 1.0))
    )
    velocity[gate] = baseline_velocity * float(geometry["speed_scale"])
    residual_gates = {
        int(value) for value in str(cfg.get("residual_gates", "")).split(",")
        if value.strip()
    }
    residual_gates.add(gate)
    phase_windows = gate_phase_windows(
        cfg.get("residual_phase_windows", "")
    )
    phase_windows[gate] = (float(args.residual_phase_start), 1.0)
    cfg.update({
        "reference_lateral_offsets": format_gate_values(lateral),
        "reference_vertical_offsets": format_gate_values(vertical),
        "reference_velocity_scales": format_gate_values(velocity),
        "residual_gates": ",".join(map(str, sorted(residual_gates))),
        "residual_phase_windows": format_gate_phase_windows(phase_windows),
        "seed_checkpoint": str(args.distilled_checkpoint.resolve()),
        "output_root": str(args.output_root.resolve()),
        "episodes": int(args.episodes),
        "eval_only": True,
        "offline_critic_warmup_updates": 0,
        "offline_updates": 0,
    })
    config["counterfactual_repair"] = {
        "repair_report": str(args.repair_report.resolve()),
        "repair_report_sha256": sha256(args.repair_report),
        "distilled_checkpoint_sha256": sha256(args.distilled_checkpoint),
        "geometry_gate": gate,
        "added_lateral_offset_m": float(geometry["lateral_offset_m"]),
        "added_vertical_offset_m": float(geometry["vertical_offset_m"]),
        "segment_speed_multiplier": float(geometry["speed_scale"]),
        "residual_phase_window": [args.residual_phase_start, 1.0],
        "synthetic_dynamics_eligible": False,
        "live_promotion_status": "not_tested",
    }
    # Identity is recomputed by the live launcher; stale hashes are unsafe.
    config.pop("identity", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "config_sha256": sha256(args.out),
        "geometry_gate": gate,
        "reference_lateral_offsets": cfg["reference_lateral_offsets"],
        "reference_vertical_offsets": cfg["reference_vertical_offsets"],
        "reference_velocity_scales": cfg["reference_velocity_scales"],
        "residual_gates": cfg["residual_gates"],
        "residual_phase_windows": cfg["residual_phase_windows"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
