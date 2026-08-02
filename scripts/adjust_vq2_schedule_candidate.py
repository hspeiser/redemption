"""Create an immutable, provenance-stamped schedule variant.

This deliberately changes only explicitly requested gate-indexed values.  It is
used for paired counterfactual audits where rewriting a full optimizer artifact
by hand would make it too easy to change unrelated controller parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_FIELDS = {
    "lead": ("leads", int),
    "thrust": ("thrust_scales", float),
    "velocity": ("velocity_scales", float),
    "blend": ("trajectory_blends", float),
    "lateral": ("lateral_offsets_m", float),
    "vertical": ("vertical_offsets_m", float),
    "rate": ("rate_scales", float),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_assignment(text: str) -> tuple[str, int, float | int]:
    try:
        name, gate_text, value_text = text.split(":", 2)
        field, cast = SUPPORTED_FIELDS[name]
        gate = int(gate_text)
        value = cast(value_text)
    except (KeyError, TypeError, ValueError) as exc:
        choices = ", ".join(sorted(SUPPORTED_FIELDS))
        raise argparse.ArgumentTypeError(
            f"expected FIELD:GATE:VALUE where FIELD is one of {choices}"
        ) from exc
    if gate < 0:
        raise argparse.ArgumentTypeError("gate index must be non-negative")
    return field, gate, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="assignments",
        type=parse_assignment,
        action="append",
        required=True,
        metavar="FIELD:GATE:VALUE",
    )
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    out = args.out.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing artifact: {out}")

    payload = json.loads(source.read_text(encoding="utf-8"))
    changes = []
    for field, gate, value in args.assignments:
        values = payload.get(field)
        if not isinstance(values, list) or gate >= len(values):
            parser.error(f"{field}[{gate}] is not present in the source artifact")
        before = values[gate]
        values[gate] = value
        changes.append({
            "field": field,
            "gate": gate,
            "before": before,
            "after": value,
        })

    # Keep the embedded live overrides synchronized with the numeric arrays.
    overrides = payload.get("live_overrides")
    if isinstance(overrides, dict):
        override_fields = {
            "leads": "reference_action_leads",
            "thrust_scales": "reference_thrust_scales",
            "velocity_scales": "reference_velocity_scales",
            "trajectory_blends": "trajectory_blends",
            "lateral_offsets_m": "reference_lateral_offsets",
            "vertical_offsets_m": "reference_vertical_offsets",
            "rate_scales": "reference_rate_scales",
        }
        for field, override in override_fields.items():
            if field in payload:
                overrides[override] = ",".join(
                    f"{gate}:{value:.9g}" if isinstance(value, float)
                    else f"{gate}:{value}"
                    for gate, value in enumerate(payload[field])
                )

    payload["manual_variant"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": sha256(source),
        "reason": args.reason,
        "changes": changes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "changes": changes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
