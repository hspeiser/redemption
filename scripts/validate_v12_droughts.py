"""v7 vs v12 on the exact frames where live vision failed (fused==0).

For each 10Hz-era debug record with zero fused corners, run both
checkpoints' dense peak extraction and count frames where >=4 of the
active gate's expected corners have a matching-class peak within
max(6, 0.08*span) px.  Recovery rate = the fraction of previously
blind frames each detector now sees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.vision.model import GateNet  # noqa: E402
from aigp.vq2_live_localizer import _dense_corner_peaks  # noqa: E402


def load_net(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    net = GateNet()
    net.load_state_dict(payload.get("model", payload))
    return net.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--sessions", nargs="+", default=[
        "vq2_20260730_200627", "vq2_20260730_212220",
        "vq2_20260730_213340",
    ])
    parser.add_argument("--primary-a", type=Path,
                        default=REPO / "data/models/gatenet_v7_best.pt")
    parser.add_argument("--primary-b", type=Path, required=True)
    parser.add_argument("--refiner", type=Path,
                        default=REPO /
                        "data/models/gatenet_v10strict_ep0.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=600)
    args = parser.parse_args()

    device = torch.device(args.device)
    net_a = load_net(args.primary_a, device)
    net_b = load_net(args.primary_b, device)
    refiner = load_net(args.refiner, device)

    records = []
    for name in args.sessions:
        debug = args.sessions_root / name / "localizer_debug/debug.jsonl"
        if not debug.exists():
            continue
        for line in debug.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = r["debug"]
            if d.get("fused", 0) != 0:
                continue
            g = d.get("active_gate")
            exp = [e for e in d.get("expected", []) if e["gate"] == g]
            if len(exp) < 8:
                continue
            img_path = args.sessions_root / name / r["image"]
            if img_path.exists():
                records.append((img_path, exp))
    print(f"blind frames: {len(records)} (evaluating up to {args.limit})")
    if len(records) > args.limit:
        idx = np.linspace(0, len(records) - 1, args.limit).astype(int)
        records = [records[i] for i in idx]

    def recovered(net, image, exp):
        peaks = _dense_corner_peaks(image, net, refiner, device, 0.35, 3.0)
        px = np.array([e["pixel"] for e in exp])
        span = float(np.ptp(px, axis=0).max())
        radius = max(6.0, 0.08 * span)
        hits = 0
        for e in exp:
            c = e["corner"]
            target = np.array(e["pixel"])
            for (u, v, _s) in peaks[c]:
                if np.hypot(u - target[0], v - target[1]) <= radius:
                    hits += 1
                    break
        return hits >= 4

    wins = {"v7": 0, "v12": 0}
    for img_path, exp in records:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        if recovered(net_a, image, exp):
            wins["v7"] += 1
        if recovered(net_b, image, exp):
            wins["v12"] += 1
    n = len(records)
    print(f"recovered blind frames: v7 {wins['v7']}/{n} "
          f"({100*wins['v7']/max(n,1):.0f}%)  "
          f"v12 {wins['v12']}/{n} ({100*wins['v12']/max(n,1):.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
