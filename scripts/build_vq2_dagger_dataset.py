"""Combine Henry's clean full lap with deterministic tracker rollouts.

The clean lap supplies coverage through every gate. Tracker evaluations
supply real recovery states with the corrective action actually chosen there;
unlike synthetic state noise, these are valid DAgger supervision pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FIELDS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "done",
    "gate_index",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-demo", type=Path, required=True)
    parser.add_argument("--eval-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean = np.load(args.clean_demo, allow_pickle=False)
    pieces: dict[str, list[np.ndarray]] = {
        key: [np.asarray(clean[key])] for key in FIELDS
    }
    source = [np.zeros(len(clean["reward"]), np.int8)]

    episode_paths = sorted(args.eval_run.glob("episode_*.npz"))
    if not episode_paths:
        raise RuntimeError(f"no episode NPZ files in {args.eval_run}")
    for path in episode_paths:
        episode = np.load(path, allow_pickle=False)
        pieces["observation"].append(
            np.asarray(episode["observation"], np.float32)
        )
        pieces["action"].append(
            np.asarray(episode["wire_action"], np.float32)
        )
        pieces["reward"].append(np.asarray(episode["reward"], np.float32))
        pieces["next_observation"].append(
            np.asarray(episode["next_observation"], np.float32)
        )
        pieces["done"].append(np.asarray(episode["done"], np.float32))
        pieces["gate_index"].append(
            np.asarray(episode["gate_index"], np.int16)
        )
        source.append(np.ones(len(episode["reward"]), np.int8))

    combined = {
        key: np.concatenate(value, axis=0)
        for key, value in pieces.items()
    }
    combined["source"] = np.concatenate(source)
    combined["source_clean_demo"] = np.asarray(str(args.clean_demo))
    combined["source_eval_run"] = np.asarray(str(args.eval_run))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **combined)
    gate, count = np.unique(combined["gate_index"], return_counts=True)
    print(f"wrote {len(combined['reward'])} rows -> {args.output}")
    print(
        f"clean={int(np.sum(combined['source'] == 0))} "
        f"dagger={int(np.sum(combined['source'] == 1))}"
    )
    print("samples per gate:", dict(zip(gate.tolist(), count.tolist())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
