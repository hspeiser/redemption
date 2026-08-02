"""Append active live flights to training without contaminating locked splits."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.vq2_g0g4_build_dataset import extract  # noqa: E402


def parse_episodes(text: str) -> list[int]:
    result: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = map(int, chunk.split("-", 1))
            result.extend(range(lo, hi + 1))
        else:
            result.append(int(chunk))
    return sorted(set(result))


def concatenate(left: np.lib.npyio.NpzFile | dict, right: dict) -> dict:
    left_fields = set(left.files) if hasattr(left, "files") else set(left)
    if left_fields != set(right):
        raise ValueError(
            f"dataset fields differ: base={sorted(left_fields)} "
            f"active={sorted(right)}"
        )
    result = {}
    # Avoid episode-number collisions with the frozen source run.
    episode_offset = int(np.max(left["episode"])) + 1
    for key in sorted(left_fields):
        value = np.asarray(right[key])
        if key == "episode":
            value = value.astype(np.int64) + episode_offset
        result[key] = np.concatenate([np.asarray(left[key]), value])
    return result


def repeat_episodes(data: dict, repeat: int) -> dict:
    """Oversample a small active batch without joining rollout boundaries."""
    if repeat <= 1:
        return data
    episode = np.asarray(data["episode"], np.int64)
    span = int(episode.max()) + 1 if len(episode) else 1
    result: dict[str, np.ndarray] = {}
    for key, value in data.items():
        chunks = []
        for copy in range(repeat):
            chunk = np.asarray(value).copy()
            if key == "episode":
                chunk = chunk.astype(np.int64) + copy * span
            chunks.append(chunk)
        result[key] = np.concatenate(chunks)
    return result


def load_entries(
    entries: list[dict], gate_positions: np.ndarray,
) -> tuple[dict, list[dict], int]:
    """Extract and concatenate sessions while preserving episode boundaries."""
    merged: dict | None = None
    reports: list[dict] = []
    total_rows = 0
    for entry in entries:
        run = Path(entry["run"])
        episodes = parse_episodes(str(entry["episodes"]))
        repeat = max(1, int(entry.get("repeat", 1)))
        active = repeat_episodes(
            extract(run, episodes, gate_positions), repeat
        )
        rows = int(len(active["action"]))
        merged = active if merged is None else concatenate(merged, active)
        total_rows += rows
        reports.append({
            "run": str(run.resolve()),
            "episodes": episodes,
            "repeat": repeat,
            "rows": rows,
        })
    if merged is None:
        raise ValueError("split manifest must contain at least one entry")
    return merged, reports, total_rows


def read_manifest(path: Path) -> list[dict]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path} must contain a non-empty JSON list")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--episodes")
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        help=(
            "JSON list of {run, episodes, repeat} entries. This appends many "
            "sessions in one compression pass while preserving locked splits."
        ),
    )
    parser.add_argument(
        "--validation-manifest", type=Path,
        help="Optional session-level validation split replacing the base split.",
    )
    parser.add_argument(
        "--test-manifest", type=Path,
        help="Optional session-level test split replacing the base split.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Training-only oversampling factor for the active episodes.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_manifest is None and (
        args.run is None or args.episodes is None
    ):
        parser.error("--run and --episodes are required without --batch-manifest")
    if args.batch_manifest is not None and (
        args.run is not None or args.episodes is not None
    ):
        parser.error("use either --batch-manifest or --run/--episodes")

    gates = json.loads(args.map.read_text())["gates"]
    gate_positions = np.asarray([g["pos"] for g in gates], np.float64)
    if args.batch_manifest is None:
        entries = [{
            "run": str(args.run),
            "episodes": args.episodes,
            "repeat": max(1, args.repeat),
        }]
    else:
        try:
            entries = read_manifest(args.batch_manifest)
        except ValueError as exc:
            parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    base_train = np.load(args.base / "train.npz")
    merged: np.lib.npyio.NpzFile | dict = base_train
    active, entry_reports, active_rows = load_entries(entries, gate_positions)
    merged = concatenate(merged, active)
    np.savez_compressed(args.out / "train.npz", **merged)
    split_reports: dict[str, dict] = {}
    for split, manifest_path in (
        ("validation", args.validation_manifest),
        ("test", args.test_manifest),
    ):
        if manifest_path is None:
            # Default remains byte-for-byte locked-split preservation.
            shutil.copy2(args.base / f"{split}.npz", args.out / f"{split}.npz")
            split_reports[split] = {
                "source": "locked_base",
                "rows": int(len(np.load(args.out / f"{split}.npz")["action"])),
            }
            continue
        try:
            split_entries = read_manifest(manifest_path)
        except ValueError as exc:
            parser.error(str(exc))
        split_data, reports, rows = load_entries(
            split_entries, gate_positions
        )
        np.savez_compressed(args.out / f"{split}.npz", **split_data)
        split_reports[split] = {
            "source": str(manifest_path.resolve()),
            "rows": rows,
            "entries": reports,
        }
    manifest = {
        "base_dataset": str(args.base.resolve()),
        "active_entries": entry_reports,
        "map": str(args.map.resolve()),
        "base_train_rows": int(len(base_train["action"])),
        "active_train_rows": active_rows,
        "merged_train_rows": int(len(merged["action"])),
        "splits": split_reports,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
