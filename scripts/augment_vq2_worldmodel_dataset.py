"""Append a small adversarial episode split to a frozen world-model set.

Validation and test remain byte-for-byte semantically identical to the base
dataset.  Repetition is intentional importance weighting for rare live policy
counterexamples; episode ids are remapped so rollout fine-tuning never joins
unrelated copies into one trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path)
    return {key: np.asarray(payload[key]) for key in payload.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--append", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("repeat must be positive")
    base_train = load(args.base / "train.npz")
    extra = load(args.append / "train.npz")
    if set(base_train) != set(extra):
        raise ValueError(
            f"dataset columns differ: base={sorted(base_train)}, "
            f"append={sorted(extra)}"
        )
    chunks: dict[str, list[np.ndarray]] = {
        key: [value] for key, value in base_train.items()
    }
    episode_floor = int(base_train["episode"].max()) + 1
    episode_span = int(extra["episode"].max()) + 1
    for repeat in range(args.repeat):
        for key, value in extra.items():
            copy = value.copy()
            if key == "episode":
                copy = copy.astype(np.int64)
                copy += episode_floor + repeat * episode_span
            chunks[key].append(copy)
    merged = {key: np.concatenate(values) for key, values in chunks.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "train.npz", **merged)
    for split in ("validation", "test"):
        np.savez_compressed(args.out / f"{split}.npz", **load(
            args.base / f"{split}.npz"
        ))
    base_manifest = json.loads((args.base / "manifest.json").read_text())
    manifest = {
        "base": str(args.base.resolve()),
        "map": base_manifest["map"],
        "append": str(args.append.resolve()),
        "append_split": "train",
        "repeat": args.repeat,
        "base_train_rows": int(len(base_train["action"])),
        "append_rows": int(len(extra["action"])),
        "merged_train_rows": int(len(merged["action"])),
        "held_out_counterexample_validation": str(
            (args.append / "validation.npz").resolve()
        ),
        "held_out_counterexample_test": str(
            (args.append / "test.npz").resolve()
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
