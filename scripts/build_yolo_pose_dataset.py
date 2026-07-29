"""Convert GateNet NPZ labels into a YOLO four-corner pose dataset.

The converter is designed for domain-adapting the large Lars/Arsenal
gate-pose model to our VQ2 camera stream. It:

* converts physical GateNet corner order to apparent TL,TR,BR,BL order;
* preserves per-corner visibility for partial/clipped gates;
* keeps sessions disjoint between train and validation;
* de-duplicates frames across base and temporal label directories, with
  earlier ``--labels`` arguments taking priority;
* hard-links images by default, so building a dataset does not duplicate
  gigabytes of JPEGs on the same drive.

Example
-------

    python scripts\\build_yolo_pose_dataset.py ^
      --labels data\\labels_vq2 ^
      --labels data\\labels_vq2t ^
      --frame-root C:\\Users\\henry\\vq2_stage ^
      --output C:\\Users\\henry\\vq2_yolo ^
      --val-session rc_20260724_003101
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

W, H = 640, 360


def canonical_indices(quad: np.ndarray) -> np.ndarray | None:
    """Indices that put an arbitrary quad in apparent TL,TR,BR,BL order."""
    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    sums = quad.sum(axis=1)
    diffs = quad[:, 0] - quad[:, 1]
    indices = np.array([
        np.argmin(sums),
        np.argmax(diffs),
        np.argmax(sums),
        np.argmin(diffs),
    ])
    return indices if len(set(indices.tolist())) == 4 else None


def resolve_source(stored_path: str, frame_root: Path | None) -> Path | None:
    original = Path(stored_path)
    if original.is_file():
        return original
    if frame_root is None:
        return None
    # Stored paths end in .../captures/<session>/frames/<frame>.jpg while
    # shipped frame archives start directly at <session>/frames/<frame>.jpg.
    parts = original.parts
    for marker in ("captures", "outputs"):
        if marker in parts:
            index = max(i for i, part in enumerate(parts) if part == marker)
            candidate = frame_root.joinpath(*parts[index + 1:])
            if candidate.is_file():
                return candidate
    if len(parts) >= 3:
        candidate = frame_root.joinpath(*parts[-3:])
        if candidate.is_file():
            return candidate
    return None


def link_or_copy(source: Path, destination: Path, mode: str):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    if mode == "symlink":
        try:
            destination.symlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def gate_to_yolo(inner, visible) -> str | None:
    order = canonical_indices(inner)
    if order is None:
        return None
    corners = np.asarray(inner, np.float64)[order]
    visible = np.asarray(visible, bool)[order]
    in_frame = (
        (corners[:, 0] >= 0.0) & (corners[:, 0] < W)
        & (corners[:, 1] >= 0.0) & (corners[:, 1] < H)
    )
    keypoint_visible = visible & in_frame
    if keypoint_visible.sum() < 2:
        return None

    # The box encloses the complete projected aperture where available, with
    # a small margin. Out-of-frame points are clipped only for the box and
    # visibility-zero keypoint serialization.
    x0 = float(np.clip(corners[:, 0].min(), 0, W - 1))
    x1 = float(np.clip(corners[:, 0].max(), 0, W - 1))
    y0 = float(np.clip(corners[:, 1].min(), 0, H - 1))
    y1 = float(np.clip(corners[:, 1].max(), 0, H - 1))
    bw = max(x1 - x0, 2.0)
    bh = max(y1 - y0, 2.0)
    margin_x = max(2.0, 0.08 * bw)
    margin_y = max(2.0, 0.08 * bh)
    x0 = max(0.0, x0 - margin_x)
    x1 = min(W - 1.0, x1 + margin_x)
    y0 = max(0.0, y0 - margin_y)
    y1 = min(H - 1.0, y1 + margin_y)
    cx = (x0 + x1) / (2.0 * W)
    cy = (y0 + y1) / (2.0 * H)
    nw = (x1 - x0) / W
    nh = (y1 - y0) / H
    fields = ["0", f"{cx:.7f}", f"{cy:.7f}", f"{nw:.7f}", f"{nh:.7f}"]
    for point, is_visible in zip(corners, keypoint_visible):
        u = float(np.clip(point[0] / W, 0.0, 1.0))
        v = float(np.clip(point[1] / H, 0.0, 1.0))
        fields.extend((f"{u:.7f}", f"{v:.7f}", "2" if is_visible else "0"))
    return " ".join(fields)


def frame_name(session: str, source: Path) -> str:
    return f"{session}_{source.stem}{source.suffix.lower()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", action="append", required=True, type=Path,
        help="directory of NPZ labels; repeat, highest priority first",
    )
    parser.add_argument(
        "--frame-root", type=Path, default=None,
        help="root containing shipped <session>/frames/*.jpg archives",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-session", action="append", default=[])
    parser.add_argument(
        "--mode", choices=("hardlink", "symlink", "copy"), default="hardlink",
    )
    parser.add_argument(
        "--include-empty", action="store_true",
        help="include empty-label frames as trusted backgrounds",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    seen = set()
    stats = Counter()
    source_stats = {}
    for label_dir in args.labels:
        source_counter = Counter()
        files = sorted(label_dir.resolve().glob("*.npz"))
        for npz_path in files:
            session = npz_path.stem
            split = "val" if session in args.val_session else "train"
            with np.load(npz_path, allow_pickle=False) as data:
                paths = np.asarray(data["path"])
                inner = np.asarray(data["inner"])
                visibility = np.asarray(data["vis_inner"])
                for row_index, stored_path in enumerate(paths):
                    source = resolve_source(
                        str(stored_path),
                        args.frame_root.resolve() if args.frame_root else None,
                    )
                    if source is None:
                        stats["missing_images"] += 1
                        source_counter["missing_images"] += 1
                        continue
                    key = (session, source.name)
                    if key in seen:
                        stats["deduplicated"] += 1
                        source_counter["deduplicated"] += 1
                        continue

                    labels = []
                    for gate_inner, gate_visible in zip(
                        inner[row_index], visibility[row_index],
                    ):
                        encoded = gate_to_yolo(gate_inner, gate_visible)
                        if encoded is not None:
                            labels.append(encoded)
                    if not labels and not args.include_empty:
                        stats["unlabeled_skipped"] += 1
                        source_counter["unlabeled_skipped"] += 1
                        continue

                    name = frame_name(session, source)
                    image_out = output / "images" / split / name
                    label_out = output / "labels" / split / f"{Path(name).stem}.txt"
                    link_or_copy(source, image_out, args.mode)
                    label_out.write_text(
                        "\n".join(labels) + ("\n" if labels else ""),
                        encoding="utf-8",
                    )
                    seen.add(key)
                    stats[f"{split}_frames"] += 1
                    stats[f"{split}_instances"] += len(labels)
                    source_counter[f"{split}_frames"] += 1
                    source_counter[f"{split}_instances"] += len(labels)
        source_stats[str(label_dir)] = dict(source_counter)

    yaml_path = output / "gates.yaml"
    yaml_path.write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [1, 0, 3, 2]\n"
        "names:\n"
        "  0: gate\n",
        encoding="utf-8",
    )
    report = {
        "output": str(output),
        "label_sources_priority_order": [str(p.resolve()) for p in args.labels],
        "frame_root": str(args.frame_root.resolve()) if args.frame_root else None,
        "validation_sessions": args.val_session,
        "stats": dict(stats),
        "by_source": source_stats,
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"dataset yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
