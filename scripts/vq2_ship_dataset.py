"""Collect the frames referenced by data/labels_vq2/*.npz into a tar
(paths kept relative to the captures root) for shipping to the training
box. Only labeled frames ship — a fraction of the raw recordings.

    .venv-train\\Scripts\\python.exe scripts\\vq2_ship_dataset.py
        -> data/vq2_train_frames.tar + a manifest printout
"""

import sys
import tarfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CAPTURES = Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
                r"\ai-grand-prix\outputs\captures")


def main():
    labels_dir = REPO / "data" / "labels_vq2"
    out_tar = REPO / "data" / "vq2_train_frames.tar"
    paths = set()
    n_files = 0
    for f in sorted(labels_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        for p in d["path"]:
            paths.add(str(p))
        n_files += 1
    print(f"{n_files} label files, {len(paths)} unique frames")
    total = 0
    with tarfile.open(out_tar, "w") as tf:
        for p in sorted(paths):
            pp = Path(p)
            try:
                rel = pp.relative_to(CAPTURES)
            except ValueError:
                print(f"  outside captures root, skipped: {p}")
                continue
            if not pp.exists():
                continue
            tf.add(pp, arcname=str(rel).replace("\\", "/"))
            total += pp.stat().st_size
    print(f"wrote {out_tar} ({total/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
