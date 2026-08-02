"""Mix a comprehensive corpus with a repeated current-policy anchor set.

Validation and test always come from the comprehensive corpus. Repetition of
the anchor is intentional importance weighting; episode ids are remapped so
multi-step windows never bridge dataset or copy boundaries.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    return {key: np.asarray(payload[key]) for key in payload.files}


def fill_column(key: str, length: int, template: np.ndarray) -> np.ndarray:
    shape = (length,) + template.shape[1:]
    if key == "sample_weight":
        return np.ones(shape, template.dtype)
    return np.full(shape, -1, template.dtype)


def align(data: dict[str, np.ndarray], templates: dict[str, np.ndarray]) \
        -> dict[str, np.ndarray]:
    length = len(data["action"])
    return {
        key: data[key] if key in data else fill_column(key, length, value)
        for key, value in templates.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comprehensive", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--anchor-repeat", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.anchor_repeat < 1:
        parser.error("--anchor-repeat must be positive")

    broad = load(args.comprehensive / "train.npz")
    anchor = load(args.anchor / "train.npz")
    templates = dict(broad)
    for key, value in anchor.items():
        templates.setdefault(key, value)
    broad = align(broad, templates)
    anchor = align(anchor, templates)
    chunks = {key: [value] for key, value in broad.items()}
    episode_floor = int(np.max(broad["episode"])) + 1
    anchor_episode = np.asarray(anchor["episode"], np.int64)
    anchor_span = int(np.max(anchor_episode)) + 1
    for repeat in range(args.anchor_repeat):
        for key, value in anchor.items():
            copy = value.copy()
            if key == "episode":
                copy = anchor_episode + episode_floor + repeat * anchor_span
            # Preserve broad-session ids; anchor sessions are deliberately
            # tagged unknown because many rows were already curriculum mixes.
            if key == "session":
                copy.fill(-1)
            chunks[key].append(copy)
    merged = {key: np.concatenate(value) for key, value in chunks.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "train.npz", **merged)
    for split in ("validation", "test"):
        shutil.copy2(
            args.comprehensive / f"{split}.npz", args.out / f"{split}.npz"
        )
    broad_manifest = json.loads(
        (args.comprehensive / "manifest.json").read_text()
    )
    manifest = {
        "comprehensive": str(args.comprehensive.resolve()),
        "anchor": str(args.anchor.resolve()),
        "anchor_repeat": args.anchor_repeat,
        "map": broad_manifest["map"],
        "comprehensive_train_rows": int(len(broad["action"])),
        "anchor_rows": int(len(anchor["action"])),
        "merged_train_rows": int(len(merged["action"])),
        "validation_rows": int(len(load(args.out / "validation.npz")["action"])),
        "test_rows": int(len(load(args.out / "test.npz")["action"])),
        "split_policy": "comprehensive frozen validation/test; anchor training only",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
