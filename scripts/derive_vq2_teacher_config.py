"""Inject an optimized first-section controller into a full live config.

The optimizer emits compact per-gate arrays.  This utility preserves every
unrelated live setting and every downstream gate override while replacing the
selected prefix.  The result can therefore serve both fastsim and the live
full-course harness without hand-editing a large campaign config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _parse_gate_values(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in str(text or "").split(","):
        if not item.strip():
            continue
        gate, value = item.split(":", 1)
        result[int(gate)] = value
    return result


def _replace_prefix(text: str, values: list[float], count: int) -> str:
    merged = _parse_gate_values(text)
    for gate, value in enumerate(values[:count]):
        merged[gate] = f"{float(value):.9g}"
    return ",".join(f"{gate}:{merged[gate]}" for gate in sorted(merged))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates", type=int, default=5)
    args = parser.parse_args()

    payload = json.loads(args.base.read_text())
    candidate = json.loads(args.candidate.read_text())
    config = payload.setdefault("args", payload)
    replacements = {
        "reference_action_leads": candidate["leads"],
        "reference_thrust_scales": candidate["thrust_scales"],
        "reference_velocity_scales": candidate["velocity_scales"],
        "reference_rate_scales": candidate.get(
            "rate_scales", [1.0] * args.gates
        ),
        "trajectory_blends": candidate["trajectory_blends"],
        "reference_lateral_offsets": candidate.get(
            "lateral_offsets_m", [0.0] * args.gates
        ),
        "reference_vertical_offsets": candidate.get(
            "vertical_offsets_m", [0.0] * args.gates
        ),
    }
    for key, values in replacements.items():
        config[key] = _replace_prefix(config.get(key, ""), values, args.gates)

    # The controller arrays above changed, so the base deployment identity is
    # no longer valid.  Preserve it as provenance but never expose it as the
    # identity of the derived artifact; a live launcher must compute a fresh
    # identity from the final deployment config.
    base_identity = payload.pop("identity", None)
    payload["derived_teacher"] = {
        "base": str(args.base.resolve()),
        "base_sha256": _sha256(args.base),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": _sha256(args.candidate),
        "prefix_gate_count": args.gates,
        "base_identity": base_identity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
