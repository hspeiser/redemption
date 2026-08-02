"""Splice gate sections from two VQ2 demonstration NPZ files.

The output takes gates listed by ``--primary-gates`` from ``--primary`` and
all remaining gates from ``--secondary``.  It preserves each source section's
observations/actions while rebuilding a monotonic wall clock and transition
links so the result remains safe to load as either a controller reference or
a replay demonstration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROW_FIELDS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "done",
    "wall",
    "gate_index",
    "position",
    "velocity",
    "sigma",
)


def observation_rotation(observation: np.ndarray) -> np.ndarray:
    first = np.asarray(observation[21:24], float)
    second = np.asarray(observation[24:27], float)
    first /= np.linalg.norm(first) + 1e-9
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second) + 1e-9
    return np.column_stack([first, second, np.cross(first, second)])


def materialize(path: Path) -> dict[str, np.ndarray]:
    """Normalize either a demo archive or a live episode into demo fields."""
    archive = np.load(path, allow_pickle=False)
    output = {key: np.asarray(archive[key]) for key in archive.files}
    count = len(output["gate_index"])
    if "wire_action" in output:
        output["action"] = np.asarray(output["wire_action"], np.float32)
    if "wall" not in output:
        output["wall"] = np.arange(count, dtype=np.float64) / 30.0
    if "velocity" not in output:
        rotations = np.asarray([
            observation_rotation(row) for row in output["observation"]
        ])
        output["velocity"] = np.einsum(
            "nij,nj->ni", rotations, output["observation"][:, 18:21] * 10.0
        ).astype(np.float32)
    if "sigma" not in output:
        output["sigma"] = np.asarray(
            output["observation"][:, -2] * 0.50, np.float32
        )
    if "source_episode" not in output:
        output["source_episode"] = np.full(count, str(path), dtype="U512")
    return output


def parse_gates(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def source_rows(data: dict[str, np.ndarray], count: int) -> np.ndarray:
    if "source_episode" not in data:
        return np.full(count, "unknown", dtype="U256")
    source = np.asarray(data["source_episode"])
    if source.ndim == 0:
        return np.full(count, str(source.item()), dtype="U512")
    return source.astype("U512", copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--primary-gates", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=30.0)
    args = parser.parse_args()

    primary = materialize(args.primary)
    secondary = materialize(args.secondary)
    primary_gates = parse_gates(args.primary_gates)
    all_gates = sorted(
        set(np.asarray(primary["gate_index"], int).tolist())
        | set(np.asarray(secondary["gate_index"], int).tolist())
    )

    chunks: dict[str, list[np.ndarray]] = {field: [] for field in ROW_FIELDS}
    sources: list[np.ndarray] = []
    primary_sources = source_rows(primary, len(primary["gate_index"]))
    secondary_sources = source_rows(secondary, len(secondary["gate_index"]))
    section_report: list[str] = []

    for gate in all_gates:
        data = primary if gate in primary_gates else secondary
        source = primary_sources if gate in primary_gates else secondary_sources
        indices = np.flatnonzero(np.asarray(data["gate_index"], int) == gate)
        if not len(indices):
            raise ValueError(f"source has no rows for gate {gate}")
        for field in ROW_FIELDS:
            chunks[field].append(np.asarray(data[field])[indices])
        sources.append(source[indices])
        section_report.append(
            f"gate {gate}: {'primary' if gate in primary_gates else 'secondary'} "
            f"({len(indices)} rows)"
        )

    output = {field: np.concatenate(parts, axis=0) for field, parts in chunks.items()}
    output["source_episode"] = np.concatenate(sources, axis=0)
    count = len(output["gate_index"])
    output["wall"] = np.arange(count, dtype=np.float64) / args.control_hz
    output["next_observation"] = np.concatenate(
        [output["observation"][1:], output["observation"][-1:]], axis=0
    )
    output["done"] = np.zeros(count, dtype=np.float32)
    output["done"][-1] = 1.0
    output["source_trace"] = np.asarray(
        f"gate-splice primary={args.primary} gates={sorted(primary_gates)}; "
        f"secondary={args.secondary}"
    )
    output["source_map"] = np.asarray(
        str(secondary["source_map"].item())
        if "source_map" in secondary
        else ""
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"wrote {args.output} ({count} rows)")
    print("\n".join(section_report))


if __name__ == "__main__":
    main()
