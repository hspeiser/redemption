"""Visual + numeric evaluation of a trained GateNet checkpoint.

    .venv-train\\Scripts\\python.exe scripts\\eval_net.py [--ckpt PATH] [--n 12]

Draws decoded corners (green=inner, yellow=outer, GT in magenta) on val
frames, prints pose-head errors, saves overlays to data/models/eval_overlays.
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
from aigp.vision.model import GateNet, rot6d_to_matrix
from scripts.train_net import (GateDataset, decode_corners, evaluate,
                               VAL_EPISODES, POS_SCALE)
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "data" / "models" / "gatenet_best.pt"))
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GateNet().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck['epoch']}, saved metrics: "
          f"{json.dumps(ck.get('metrics', {}), indent=1)}")

    labels_dir = REPO / "data" / "labels"
    files = sorted(labels_dir.glob("*.npz"))
    val_f = [f for f in files if f.stem in VAL_EPISODES] or files[::8]
    ds = GateDataset(val_f, train=False)
    dl = DataLoader(ds, batch_size=16, num_workers=2)
    print(f"val frames: {len(ds)}")

    metrics = evaluate(model, dl, device, max_batches=250)
    print("FULL VAL METRICS:", json.dumps(metrics, indent=1))

    out_dir = REPO / "data" / "models" / "eval_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(ds) - 1, args.n).astype(int)
    for k, i in enumerate(idxs):
        item = ds[i]
        img = item["img"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=device == "cuda"):
            out = model(img)
        bgr = (item["img"][:3].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
        # GT corners (magenta): recover from offset maps
        om, off = item["om"].numpy(), item["off"].numpy()
        for c in range(8):
            for (ci, cj) in np.argwhere(om[c] > 0):
                u = (cj + off[2 * c, ci, cj]) * 4
                v = (ci + off[2 * c + 1, ci, cj]) * 4
                cv2.drawMarker(bgr, (int(u), int(v)), (255, 0, 255),
                               cv2.MARKER_CROSS, 8, 1)
        dec = decode_corners(out["hm"][0].float().cpu(), out["off"][0].float().cpu())
        for c in range(8):
            color = (0, 255, 0) if c < 4 else (0, 255, 255)
            for (u, v, s) in dec[c]:
                cv2.circle(bgr, (int(u), int(v)), 3, color, 1)
        pos = out["pos"][0].float().cpu().numpy() * POS_SCALE
        gt_pos = item["pos"].numpy() * POS_SCALE
        cv2.putText(bgr, f"pos err {np.linalg.norm(pos-gt_pos):.2f}m",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(out_dir / f"eval_{k:02d}.jpg"), bgr)
    print(f"overlays -> {out_dir}")


if __name__ == "__main__":
    main()
