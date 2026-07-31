"""Close-range VQ2 self-labels from pose tracks + dense peak snapping.

The v7 detector was trained with close/partial gates systematically
unlabeled (proposal recall bias), and live it fails hardest on
large-span views.  This generator manufactures exactly those labels:

- pose per frame from pose_tracks npz (10Hz-era belief, ~8 px seed
  accuracy validated against the localizer's own projections),
- corners projected from the g9/g15-corrected map,
- apparent corner classes via the viewing-side permutation
  [1,0,3,2,5,4,7,6] (validated on 3,389 debug match sets, 100%),
- snapped to dense GateNet peaks where available (sub-pixel), kept as
  raw projections only on high-quality frames,
- output in the vq2_labels npz schema (inner/outer/vis/ignore) plus
  real pose-head supervision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import json  # noqa: E402

import torch  # noqa: E402

from aigp.vision.labels import load_calib  # noqa: E402
from aigp.vision.model import GateNet  # noqa: E402
from aigp.vq2_live_localizer import _dense_corner_peaks  # noqa: E402
from aigp.vq2_map import gate_quads_world_vq2  # noqa: E402

G_SLOTS = 4
W, H = 640, 360
PERM = np.array([1, 0, 3, 2, 5, 4, 7, 6])


def load_net(path: Path, device) -> GateNet:
    payload = torch.load(path, map_location=device, weights_only=False)
    net = GateNet()
    net.load_state_dict(payload.get("model", payload))
    return net.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--pose-tracks", type=Path, nargs="+",
                        required=True)
    parser.add_argument("--map", type=Path,
                        default=REPO / "data/vq2_runtime_map_g9g15fix.json")
    parser.add_argument("--calibration", type=Path,
                        default=REPO / "data/calib/calib.json")
    parser.add_argument("--primary", type=Path,
                        default=REPO / "data/models/gatenet_v7_best.pt")
    parser.add_argument("--refiner", type=Path,
                        default=REPO /
                        "data/models/gatenet_v10strict_ep0.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--stride", type=int, default=1,
                        help="use every Nth pose frame")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    primary = load_net(args.primary, device)
    refiner = load_net(args.refiner, device)
    calib = load_calib(args.calibration)
    R_cb = np.asarray(calib["R_cb"], float)
    fx, fy, cx, cy = calib["K"]

    payload = json.loads(args.map.read_text())
    gate_list = payload["gates"] if isinstance(payload, dict) else payload
    quads = []
    norms = []
    centers = []
    for gate in gate_list[:17]:
        hole, panel = gate_quads_world_vq2(gate)
        quads.append(np.vstack([hole, panel]))
        qw, qx, qy, qz = gate["quat_wxyz"]
        norms.append(
            Rotation.from_quat([qx, qy, qz, qw]).as_matrix()[:, 1]
        )
        centers.append(np.asarray(gate["pos"], float))

    L = {k: [] for k in (
        "path", "inner", "outer", "vis_inner", "vis_outer",
        "pos", "quat", "gate_idx", "next_gate_pos",
        "ig_boxes", "ig_fidx",
    )}
    stats = {"frames": 0, "labeled": 0, "snapped": 0, "projected": 0,
             "close_labeled": 0}

    for track_path in args.pose_tracks:
        track = np.load(track_path, allow_pickle=False)
        session = str(track_path.stem).replace("pose_tracks_", "")
        session_dir = None
        for candidate in args.frames_root.glob(f"*{session}*"):
            session_dir = candidate
            break
        if session_dir is None:
            print(f"no session dir for {track_path.name}, skip")
            continue
        n_rows = len(track["path"])
        for row in range(0, n_rows, args.stride):
            stats["frames"] += 1
            sigma = float(track["sigma"][row])
            gyro = float(track["gyro_norm"][row])
            high_quality = sigma < 0.10 and gyro < 1.2
            img_path = session_dir / str(track["path"][row])
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            pos = track["pos"][row].astype(float)
            qw, qx, qy, qz = track["quat_wxyz"][row].astype(float)
            Rw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

            peaks = _dense_corner_peaks(
                image, primary, refiner, device,
                args.threshold, 3.0,
            )

            candidates = []
            for gi in range(17):
                cam = (R_cb @ Rw.T @ (quads[gi] - pos).T).T
                if np.any(cam[:, 2] < 0.25):
                    continue
                uv = np.stack([
                    fx * cam[:, 0] / cam[:, 2] + cx,
                    fy * cam[:, 1] / cam[:, 2] + cy,
                ], axis=1)
                vis = ((uv[:, 0] >= -8) & (uv[:, 0] < W + 8)
                       & (uv[:, 1] >= -8) & (uv[:, 1] < H + 8))
                if vis.sum() < 2:
                    continue
                span = float(max(np.ptp(uv[vis], axis=0).max(), 1.0))
                if span < 12:
                    continue
                side = float(np.dot(pos - centers[gi], norms[gi]))
                order = PERM if side > 0 else np.arange(8)
                # apparent-class pixel for class c = uv[world index]
                uv_class = np.empty((8, 2), np.float32)
                vis_class = np.zeros(8, bool)
                for c in range(8):
                    uv_class[c] = uv[order[c]]
                    vis_class[c] = vis[order[c]]
                candidates.append((span, gi, uv_class, vis_class))

            candidates.sort(key=lambda t: -t[0])
            gates_f = []
            for span, gi, uv_class, vis_class in candidates[:G_SLOTS]:
                radius = float(np.clip(0.10 * span, 6.0, 20.0))
                snapped = uv_class.copy()
                n_snap = 0
                for c in range(8):
                    if not vis_class[c]:
                        continue
                    best = None
                    for (u, v, score) in peaks[c]:
                        dist = float(np.hypot(
                            u - uv_class[c, 0], v - uv_class[c, 1]
                        ))
                        if dist <= radius and (
                            best is None or dist < best[0]
                        ):
                            best = (dist, u, v)
                    if best is not None:
                        snapped[c] = (best[1], best[2])
                        n_snap += 1
                is_close = span > 140
                if n_snap >= 3 or (is_close and n_snap >= 2) or (
                    is_close and high_quality
                ):
                    gates_f.append((snapped, vis_class, n_snap, is_close))
                    stats["snapped"] += n_snap
                    stats["projected"] += int(vis_class.sum()) - n_snap

            if not gates_f:
                continue
            inner = np.full((G_SLOTS, 4, 2), np.nan, np.float32)
            outer = np.full((G_SLOTS, 4, 2), np.nan, np.float32)
            vi = np.zeros((G_SLOTS, 4), bool)
            vo = np.zeros((G_SLOTS, 4), bool)
            any_close = False
            for slot, (uv8, vis8, _n, is_close) in enumerate(gates_f):
                inner[slot] = uv8[0:4]
                outer[slot] = uv8[4:8]
                vi[slot] = vis8[0:4]
                vo[slot] = vis8[4:8]
                any_close = any_close or is_close
            L["path"].append(str(img_path))
            L["inner"].append(inner)
            L["outer"].append(outer)
            L["vis_inner"].append(vi)
            L["vis_outer"].append(vo)
            L["pos"].append(pos.astype(np.float32))
            L["quat"].append(np.array([qw, qx, qy, qz], np.float32))
            gidx = int(track["gate_idx"][row])
            L["gate_idx"].append(min(gidx, 16))
            L["next_gate_pos"].append(
                centers[min(gidx, 16)].astype(np.float32)
            )
            stats["labeled"] += 1
            if any_close:
                stats["close_labeled"] += 1

    n = len(L["path"])
    np.savez_compressed(
        args.out,
        path=np.array(L["path"]),
        inner=np.array(L["inner"], np.float32),
        outer=np.array(L["outer"], np.float32),
        vis_inner=np.array(L["vis_inner"]),
        vis_outer=np.array(L["vis_outer"]),
        pos=np.array(L["pos"], np.float32),
        vel=np.zeros((n, 3), np.float32),
        quat=np.array(L["quat"], np.float32),
        gate_idx=np.array(L["gate_idx"], np.int64),
        next_gate_pos=np.array(L["next_gate_pos"], np.float32),
        pose_valid=np.ones(n, np.float32),
        ignore_boxes=np.zeros((0, 4), np.float32),
        ignore_frame_idx=np.zeros(0, np.int64),
    )
    print(f"stats: {stats}")
    print(f"labeled frames: {n} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
