"""Build auto-labeled vision dataset from all usable (native-timestamp VQ1)
episodes.

    uv run python scripts/build_dataset.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp.ingest import usable_training_episodes
from aigp.vision.labels import load_calib, build_episode_labels

ROOT = r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379\ai-grand-prix\outputs\captures"
CALIB = Path(__file__).resolve().parents[1] / "data" / "calib" / "calib.json"
OUT = Path(__file__).resolve().parents[1] / "data" / "labels"


def main():
    calib = load_calib(CALIB)
    eps = usable_training_episodes(ROOT)
    print(f"{len(eps)} usable episodes", flush=True)
    total = 0
    for ep in eps:
        try:
            name, n = build_episode_labels(ep, calib, OUT)
            total += n
            print(f"{name}: {n} labeled frames", flush=True)
        except Exception as e:
            print(f"{ep.name}: FAILED {e}", flush=True)
            traceback.print_exc()
    print(f"\nTOTAL labeled frames: {total}", flush=True)


if __name__ == "__main__":
    main()
