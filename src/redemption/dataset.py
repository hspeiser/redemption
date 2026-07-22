"""Dataset layout, YOLO-Pose label formatting and ``data.yaml`` generation.

Directory layout produced (Ultralytics-compatible)::

    datasets/<name>/
        data.yaml
        images/{train,val,test}/*.png
        labels/{train,val,test}/*.txt      # YOLO-Pose: cls cx cy w h (px py v)x4
        meta/{train,val,test}/*.json       # ground-truth poses + corner pixels
"""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from .config import DotDict
from .gate import CORNER_NAMES

SPLITS = ("train", "val", "test")

# Horizontal-flip keypoint permutation for TL,TR,BR,BL -> TR,TL,BL,BR.
FLIP_IDX = [1, 0, 3, 2]


def dataset_root(datagen_cfg: DotDict) -> Path:
    return Path(datagen_cfg.dataset.output_root) / datagen_cfg.dataset.name


def split_paths(root: Path, split: str) -> dict[str, Path]:
    return {
        "images": root / "images" / split,
        "labels": root / "labels" / split,
        "meta": root / "meta" / split,
    }


def prepare_dirs(root: Path, overwrite: bool) -> None:
    """Create the dataset tree, optionally wiping an existing one of the same name."""
    if overwrite and root.exists():
        shutil.rmtree(root)
    for split in SPLITS:
        for sub in split_paths(root, split).values():
            sub.mkdir(parents=True, exist_ok=True)


def format_label_line(class_id: int, bbox_norm: tuple[float, float, float, float],
                      kpts_norm: list[tuple[float, float, int]]) -> str:
    """Build one YOLO-Pose label line.

    ``bbox_norm`` is (cx, cy, w, h) normalized; ``kpts_norm`` is a list of
    (x, y, v) with normalized x/y and integer visibility flag.
    """
    parts = [str(int(class_id))]
    parts += [f"{v:.6f}" for v in bbox_norm]
    for x, y, v in kpts_norm:
        parts += [f"{x:.6f}", f"{y:.6f}", str(int(v))]
    return " ".join(parts)


def write_data_yaml(root: Path) -> Path:
    """Write ``data.yaml`` describing the pose dataset for Ultralytics."""
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "gate"},
        "kpt_shape": [len(CORNER_NAMES), 3],
        "flip_idx": FLIP_IDX,
    }
    out = root / "data.yaml"
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out
