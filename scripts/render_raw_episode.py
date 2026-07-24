"""Render a recorder episode's raw frames through the live overlay stack to
MP4 — no labels required (works on old-format/VQ2 episodes).

    .venv-train\\Scripts\\python.exe scripts\\render_raw_episode.py \\
        --episode-dir <captures\\rc_...> [--out ...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.live_overlay import Annotator

W, H = 640, 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pace", action="store_true")
    args = ap.parse_args()

    ep = Path(args.episode_dir)
    out = Path(args.out) if args.out else \
        REPO / "data" / f"replay_{ep.name}_raw.mp4"

    # unique frames in order (first row per frame_id)
    seen = set()
    frames = []
    with open(ep / "frames.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["frame_id"] in seen:
                continue
            seen.add(r["frame_id"])
            frames.append((r["sim_time_ns"], ep / "frames" / f"{r['idx']:06d}.jpg"))
    frames.sort(key=lambda x: x[0])
    print(f"{len(frames)} unique frames", flush=True)

    ann = Annotator()
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, H))
    t_next = time.perf_counter()
    n = 0
    for (_, p) in frames:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))
        ann.process(bgr)
        if ann.jpeg is not None:
            f = cv2.imdecode(np.frombuffer(ann.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if f is not None:
                vw.write(f)
                n += 1
        if args.pace:
            t_next += 1.0 / args.fps
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
        if n and n % 600 == 0:
            print(f"{n} frames...", flush=True)
    vw.release()
    print(f"wrote {n} frames -> {out} ({out.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
