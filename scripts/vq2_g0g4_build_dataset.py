"""Freeze episode-level splits and decode the gates 0-4 POC dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.worldmodel import decode_observations  # noqa: E402


def read_summaries(run: Path) -> dict[int, dict]:
    result = {}
    with (run / "episodes.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            result[int(row["episode"])] = row
    return result


def make_splits(summaries: dict[int, dict], seed: int) -> dict[str, list[int]]:
    """Stratify successful g4 crossings and failures at episode granularity."""
    groups: dict[str, list[int]] = defaultdict(list)
    for episode, row in summaries.items():
        crossed = (
            any(c.get("gate") == 4
                for c in row.get("crossing_offsets", []))
            or int(row.get("gate_reached", -1)) >= 5
        )
        groups["crossed_g4" if crossed else "failed_before_g4"].append(episode)
    rng = np.random.default_rng(seed)
    split = {"train": [], "validation": [], "test": []}
    for values in groups.values():
        values = np.asarray(values, int)
        rng.shuffle(values)
        n = len(values)
        n_test = max(1, int(round(0.15 * n)))
        n_validation = max(1, int(round(0.15 * n)))
        split["test"].extend(values[:n_test].tolist())
        split["validation"].extend(
            values[n_test:n_test + n_validation].tolist()
        )
        split["train"].extend(values[n_test + n_validation:].tolist())
    for values in split.values():
        values.sort()
    return split


def extract(run: Path, episodes: list[int], gate_positions: np.ndarray) -> dict:
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for episode in episodes:
        path = run / f"episode_{episode:04d}.npz"
        if not path.exists():
            continue
        payload = np.load(path)
        observation = np.asarray(payload["observation"], np.float32)
        next_observation = np.asarray(payload["next_observation"], np.float32)
        current = decode_observations(observation, gate_positions)
        following = decode_observations(next_observation, gate_positions)
        gate = np.asarray(payload["gate_index"], np.int16)
        healthy = np.asarray(
            payload["timing_healthy"] if "timing_healthy" in payload
            else np.ones(len(gate), bool), bool,
        )
        valid = (gate <= 4) & healthy
        row = np.nonzero(valid)[0]
        if not len(row):
            continue
        def add(name: str, value: np.ndarray) -> None:
            chunks[name].append(np.asarray(value)[row])
        add("position", current.position)
        add("velocity", current.velocity)
        add("rotation", current.rotation)
        add("rates", current.rates)
        add("previous_action", current.previous_action)
        add("confidence", current.confidence)
        add("next_position", following.position)
        add("next_velocity", following.velocity)
        add("next_rotation", following.rotation)
        add("next_rates", following.rates)
        add("action", np.asarray(payload["wire_action"], np.float32))
        add("gate_index", gate)
        add("done", np.asarray(payload["done"], np.float32))
        # Older campaigns did not persist the redundant gates_passed vector.
        # For an in-order course the current target gate is exactly the number
        # of gates already passed, so gate_index is the lossless fallback.
        add("gates_passed", np.asarray(
            payload["gates_passed"]
            if "gates_passed" in payload.files else gate,
            np.int16,
        ))
        chunks["episode"].append(np.full(len(row), episode, np.int16))
        chunks["step"].append(row.astype(np.int16))
    return {key: np.concatenate(value) for key, value in chunks.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    summaries = read_summaries(args.run)
    gates = json.loads(args.map.read_text())["gates"]
    gate_positions = np.asarray([g["pos"] for g in gates], np.float64)
    split = make_splits(summaries, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_run": str(args.run.resolve()),
        "map": str(args.map.resolve()),
        "seed": args.seed,
        "splits": split,
        "counts": {},
    }
    for name, episodes in split.items():
        data = extract(args.run, episodes, gate_positions)
        np.savez_compressed(args.out / f"{name}.npz", **data)
        crossed = sum(
            any(c.get("gate") == 4
                for c in summaries[e].get("crossing_offsets", []))
            or int(summaries[e].get("gate_reached", -1)) >= 5
            for e in episodes
        )
        manifest["counts"][name] = {
            "episodes": len(episodes),
            "crossed_gate4": crossed,
            "transitions": int(len(data["action"])),
            "continuous_transitions": int(np.sum(data["done"] == 0)),
        }
        print(name, manifest["counts"][name])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote frozen dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
