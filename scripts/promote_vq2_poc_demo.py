"""Promote a proven live gates-0..N segment into the full-course demo.

The selected live episode replaces the matching prefix of the existing demo.
Later gates remain available so the regular trainer/controller can load the
artifact without a special short-course code path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.worldmodel import decode_observations  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--through-gate", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = np.load(args.base, allow_pickle=False)
    episode = np.load(args.episode, allow_pickle=False)
    gates = json.loads(args.map.read_text())["gates"]
    gate_positions = np.asarray([gate["pos"] for gate in gates], np.float64)
    keep = np.asarray(base["gate_index"]) > args.through_gate
    observation = np.asarray(episode["observation"], np.float32)
    decoded = decode_observations(observation, gate_positions)
    n_live = len(observation)
    n_tail = int(keep.sum())
    source = np.asarray(
        [str(args.episode.resolve())] * n_live
        + [str(args.base.resolve())] * n_tail,
    )

    def merged(live: np.ndarray, key: str) -> np.ndarray:
        return np.concatenate([np.asarray(live), np.asarray(base[key])[keep]])

    payload = {
        "observation": merged(observation, "observation").astype(np.float32),
        "action": merged(episode["wire_action"], "action").astype(np.float32),
        "reward": merged(episode["reward"], "reward").astype(np.float32),
        "next_observation": merged(
            episode["next_observation"], "next_observation"
        ).astype(np.float32),
        "done": merged(episode["done"], "done").astype(np.float32),
        "wall": np.arange(n_live + n_tail, dtype=np.float64) / 30.0,
        "gate_index": merged(
            episode["gate_index"], "gate_index"
        ).astype(np.int64),
        "position": merged(
            episode["position"], "position"
        ).astype(np.float64),
        "velocity": merged(decoded.velocity, "velocity").astype(np.float64),
        "sigma": merged(
            observation[:, 51] * 0.5, "sigma"
        ).astype(np.float64),
        "source_episode": source,
        "source_trace": str(args.episode.resolve()),
        "source_map": str(args.map.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    report = {
        "out": str(args.out.resolve()),
        "promoted_episode": str(args.episode.resolve()),
        "through_gate": args.through_gate,
        "promoted_rows": n_live,
        "preserved_tail_rows": n_tail,
        "total_rows": n_live + n_tail,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
