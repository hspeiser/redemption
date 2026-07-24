"""Render a recorded episode through the live overlay stack into an MP4.

    .venv-train\\Scripts\\python.exe scripts\\make_replay_video.py \\
        [--episode rc_20260723_022654] [--out data\\replay_overlay.mp4]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.live_overlay import Annotator
from scripts.replay_eval import quat_to_R

W, H = 640, 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="rc_20260723_022654")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pace", action="store_true",
                    help="pace at real time (default: as fast as GPU allows, "
                         "with synthetic 33ms timestamps for the filters)")
    ap.add_argument("--debug", action="store_true",
                    help="draw ALL net peaks (thresh 0.05, size ~ score) and "
                         "ground-truth gate ghosts instead of relying on the "
                         "overlay's own drawing")
    args = ap.parse_args()

    npz = REPO / "data" / "labels" / f"{args.episode}.npz"
    if not npz.exists():
        npz = REPO / "data" / "labels_quarantine" / f"{args.episode}.npz"
    d = np.load(npz)
    n = len(d["path"])
    out_path = Path(args.out) if args.out else \
        REPO / "data" / f"replay_{args.episode}_overlay.mp4"

    ann = Annotator()
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, H))
    t_next = time.perf_counter()
    written = 0
    prev_cam = None
    for i in range(n):
        bgr = cv2.imread(str(d["path"][i]))
        if bgr is None:
            continue
        Rwb = quat_to_R(d["quat"][i].astype(np.float64))
        R_cam = ann.R_cb @ Rwb.T
        p = d["pos"][i].astype(np.float64)
        motion = None
        if prev_cam is not None:
            R1, p1 = prev_cam
            motion = (R_cam @ R1.T, R_cam @ (p1 - p))
        prev_cam = (R_cam, p)
        ann.process(bgr, motion=motion)
        frame = None
        if args.debug:
            import torch
            from scripts.train_net import decode_corners, orange_channel
            frame = bgr.copy()
            # ground-truth gate ghosts (thin gray)
            for gi in range(d["inner"].shape[1]):
                for quad_key in ("inner", "outer"):
                    q = d[quad_key][i, gi]
                    if np.isfinite(q).all():
                        cv2.polylines(frame, [q.astype(np.int32)], True,
                                      (110, 110, 110), 1)
            img = np.concatenate([bgr.astype(np.float32) / 255.0,
                                  orange_channel(bgr)[..., None]], 2
                                 ).transpose(2, 0, 1)
            x = torch.from_numpy(img).unsqueeze(0).to(ann.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=ann.device == "cuda"):
                out = ann.model(x)
            dec = decode_corners(out["hm"][0].float().cpu(),
                                 out["off"][0].float().cpu(),
                                 thresh=0.05, topk=24)
            for c in range(8):
                col = (0, 255, 0) if c < 4 else (0, 255, 255)
                for (u, v, s) in dec[c]:
                    cv2.circle(frame, (int(u), int(v)),
                               max(2, int(2 + 6 * s)), col,
                               2 if s > 0.25 else 1)
        elif ann.jpeg is not None:
            frame = cv2.imdecode(np.frombuffer(ann.jpeg, np.uint8),
                                 cv2.IMREAD_COLOR)
        if frame is not None:
            vw.write(frame)
            written += 1
        if args.pace:
            t_next += 1.0 / args.fps
            dt = t_next - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
        if written % 300 == 0 and written:
            print(f"{written} frames...", flush=True)
    vw.release()
    print(f"wrote {written} frames -> {out_path} "
          f"({out_path.stat().st_size/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
