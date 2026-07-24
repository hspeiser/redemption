"""Blue-ribbon robustness eval: corner detection rate and localization error
bucketed by the amount of blue glow near each ground-truth corner.

    .venv-train\\Scripts\\python.exe scripts\\eval_blue.py [--ckpt ...]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.vision.model import GateNet
from scripts.train_net import (GateDataset, decode_corners, orange_channel,
                               VAL_EPISODES)

W, H = 640, 360
BUCKETS = [(0.0, 0.01, "clean       "), (0.01, 0.10, "light blue  "),
           (0.10, 0.30, "medium blue "), (0.30, 1.01, "heavy blue  ")]


def blue_fraction(hsv, u, v, r=10):
    x0, x1 = int(u - r), int(u + r + 1)
    y0, y1 = int(v - r), int(v + r + 1)
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return None
    roi = hsv[max(0, y0):y1, max(0, x0):x1]
    if roi.size == 0:
        return None
    blue = ((roi[..., 0] > 85) & (roi[..., 0] < 135) &
            (roi[..., 1] > 60) & (roi[..., 2] > 60))
    return float(blue.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "data" / "models" / "gatenet_v4_best.pt"))
    ap.add_argument("--max-frames", type=int, default=1200)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GateNet().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck['epoch']} ({Path(args.ckpt).name})")

    files = sorted((REPO / "data" / "labels").glob("*.npz"))
    val_f = [f for f in files if f.stem in VAL_EPISODES] or files[::8]
    ds = GateDataset(val_f, train=False)
    step = max(1, len(ds) // args.max_frames)

    SPANS = [(14, 40, "far  (14-40px)"), (40, 120, "mid  (40-120px)"),
             (120, 10000, "near (>120px)")]
    stats = {(sn, name): {"err": [], "n_gt": 0, "n_det": 0}
             for (_, _, sn) in SPANS for (_, _, name) in BUCKETS}
    for i in range(0, len(ds.items), step):
        it = ds.items[i]
        bgr = cv2.imread(it["path"])
        if bgr is None:
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        img = np.concatenate([bgr.astype(np.float32) / 255.0,
                              orange_channel(bgr)[..., None]], 2
                             ).transpose(2, 0, 1)
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=device == "cuda"):
            out = model(x)
        dec = decode_corners(out["hm"][0].float().cpu(),
                             out["off"][0].float().cpu(), thresh=0.25)
        n_g = it["inner"].shape[0]
        for gi in range(n_g):
            # only score reasonably-sized gates
            pts_o = it["outer"][gi]
            if not np.isfinite(pts_o).all():
                continue
            span = (np.nanmax(pts_o, 0) - np.nanmin(pts_o, 0)).max()
            span_name = None
            for lo, hi, sn in SPANS:
                if lo <= span < hi:
                    span_name = sn
                    break
            if span_name is None:
                continue
            for half, pts, vis in (("i", it["inner"][gi], it["vis_inner"][gi]),
                                   ("o", it["outer"][gi], it["vis_outer"][gi])):
                for c in range(4):
                    if not vis[c]:
                        continue
                    u, v = pts[c]
                    bf = blue_fraction(hsv, u, v)
                    if bf is None:
                        continue
                    bucket = None
                    for lo, hi, name in BUCKETS:
                        if lo <= bf < hi:
                            bucket = name
                            break
                    if bucket is None:
                        continue
                    cls = c if half == "i" else 4 + c
                    best = None
                    for (du, dv, s) in dec[cls]:
                        d = np.hypot(du - u, dv - v)
                        if d < 8.0 and (best is None or d < best):
                            best = d
                    key = (span_name, bucket)
                    stats[key]["n_gt"] += 1
                    if best is not None:
                        stats[key]["n_det"] += 1
                        stats[key]["err"].append(best)

    print(f"\n{'gate size':16s} {'blue':14s} {'GT':>6s} {'det rate':>9s} "
          f"{'err med':>8s} {'err p90':>8s}")
    for (_, _, sn) in SPANS:
        for (_, _, name) in BUCKETS:
            s = stats[(sn, name)]
            if s["n_gt"] < 20:
                continue
            e = np.array(s["err"]) if s["err"] else np.array([np.nan])
            print(f"{sn:16s} {name:14s} {s['n_gt']:6d} "
                  f"{s['n_det']/s['n_gt']*100:8.1f}% "
                  f"{np.median(e):7.2f}px {np.percentile(e,90):7.2f}px")


if __name__ == "__main__":
    main()
