"""Map-free VQ2 self-labeling: GateNet v6 corners, geometrically verified.

Per frame: classical detections propose gate regions (plus a whole-frame
proposal when nothing is found); the strongest v6 peak per corner class
inside each region is PnP-verified against the rigid 2.72/1.50 m gate
(>=6 classes, reprojection < 1 px). Verified poses REPROJECT all 8
corners as labels (sub-pixel, includes weak/occluded corners). Proposals
that half-fire but fail verification become ignore boxes. Pose-head
targets are zeroed with pose_valid=0 (masked in training).

    .venv-train\\Scripts\\python.exe scripts\\vq2_labels.py <ep_dir> [...]
        --out data/labels_vq2
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
from aigp.calib.detect import detect_gates  # noqa: E402
from aigp.vision.labels import load_calib  # noqa: E402
from aigp.vision.model import GateNet  # noqa: E402
from scripts.train_net import orange_channel, decode_corners  # noqa: E402

RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
H8, P8 = 0.75, 1.36
OBJ8 = np.array([[-H8, 0, -H8], [H8, 0, -H8], [H8, 0, H8], [-H8, 0, H8],
                 [-P8, 0, -P8], [P8, 0, -P8], [P8, 0, P8], [-P8, 0, P8]])
G_SLOTS = 4          # label slots per frame
W, H = 640, 360


def unique_frames(ep):
    """Unique frame jpgs (the recorder writes duplicate frame_ids)."""
    fj = ep / "frames.jsonl"
    if not fj.exists():
        sub = sorted((ep / "frames").glob("*.jpg"))
        return sub if sub else sorted(ep.glob("*.jpg"))
    rows, seen = [], set()
    with open(fj) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["frame_id"] in seen:
                continue
            seen.add(r["frame_id"])
            rows.append((r["sim_time_ns"], ep / "frames" / f"{r['idx']:06d}.jpg"))
    rows.sort()
    return [p for (_t, p) in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*")
    ap.add_argument("--list", default=None,
                    help="text file with one episode dir per line")
    ap.add_argument("--out", default=str(REPO / "data" / "labels_vq2"))
    ap.add_argument("--ckpt", default=str(
        REPO / "data/models/gatenet_v6wsl_best.pt"))
    ap.add_argument("--stride", type=int, default=2,
                    help="use every Nth unique frame")
    ap.add_argument("--rms-max", type=float, default=1.0)
    args = ap.parse_args()
    if args.list:
        args.episodes = [ln.strip() for ln in
                         Path(args.list).read_text().splitlines()
                         if ln.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = load_calib(REPO / "data/calib/calib.json")
    fx, fy, cx, cy = calib["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = GateNet().to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"labeler net: {Path(args.ckpt).name} epoch {ck['epoch']} on {dev}")

    obj_p_full = np.ascontiguousarray(OBJ8 @ RX90.T)

    def verify(idxs, uvs):
        """PnP-verify >=6 identified corners; returns (rms, all8_uv) or None."""
        obj = np.ascontiguousarray(OBJ8[idxs] @ RX90.T)
        ip = np.ascontiguousarray(uvs, np.float64).reshape(-1, 1, 2)
        try:
            n, rv, tv, errs = cv2.solvePnPGeneric(
                obj, ip, K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        best = None
        for r0, t0 in zip(rv, tv):
            try:
                r0, t0 = cv2.solvePnPRefineLM(obj, ip, K, None, r0, t0)
            except cv2.error:
                continue
            pr, _ = cv2.projectPoints(obj, r0, t0, K, None)
            rms = float(np.sqrt(((pr - ip) ** 2).sum(axis=2).mean()))
            if best is None or rms < best[0]:
                best = (rms, r0, t0)
        if best is None or best[0] > args.rms_max:
            return None
        rms, r0, t0 = best
        depth = float(np.linalg.norm(t0))
        if not (1.2 < depth < 45.0):
            return None
        pr8, _ = cv2.projectPoints(obj_p_full, r0, t0, K, None)
        return rms, pr8.reshape(8, 2), depth

    for ep_s in args.episodes:
        ep = Path(ep_s)
        out_f = out_dir / f"{ep.name}.npz"
        if out_f.exists():
            print(f"{ep.name}: exists, skip")
            continue
        paths = unique_frames(ep)[:: args.stride]
        L = {"path": [], "inner": [], "outer": [], "vis_inner": [],
             "vis_outer": [], "ig_boxes": [], "ig_fidx": []}
        n_lab = 0
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            orange = orange_channel(bgr)
            x = np.concatenate([bgr.astype(np.float32) / 255.0,
                                orange[..., None]], 2).transpose(2, 0, 1)
            xt = torch.from_numpy(x).unsqueeze(0).to(dev)
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=dev == "cuda"):
                o = model(xt)
            peaks = decode_corners(o["hm"][0].float().cpu(),
                                   o["off"][0].float().cpu(), thresh=0.25)
            n_pk = sum(len(peaks[c]) for c in range(8))
            if n_pk == 0:
                continue
            dets = detect_gates(bgr, min_area=250)
            boxes = []
            for d in dets:
                x0, y0 = d["outer"].min(0) - 14
                x1, y1 = d["outer"].max(0) + 14
                boxes.append((x0, y0, x1, y1))
            # net-native proposals from hole-diagonal peak pairs: catches
            # CLOSE gates the classical detector misses (bloom/partial),
            # so close views stop being systematically unlabeled
            for (ca, cb) in ((0, 2), (1, 3)):
                for (ua, va, _sa) in sorted(peaks[ca],
                                            key=lambda q: -q[2])[:3]:
                    for (ub, vb, _sb) in sorted(peaks[cb],
                                                key=lambda q: -q[2])[:3]:
                        span = max(abs(ub - ua), abs(vb - va))
                        if span < 24:
                            continue
                        cx0, cy0 = (ua + ub) / 2, (va + vb) / 2
                        half = span * 1.15
                        boxes.append((cx0 - half, cy0 - half,
                                      cx0 + half, cy0 + half))
            if not boxes:
                boxes.append((-1e9, -1e9, 1e9, 1e9))   # whole frame
            gates_f = []          # (inner4, outer4, vis8)
            ig_f = []
            used_pk = set()
            for (x0, y0, x1, y1) in boxes[:12]:
                idxs, uvs, keys = [], [], []
                for c in range(8):
                    inb = [(u, v, sc) for (u, v, sc) in peaks[c]
                           if x0 <= u <= x1 and y0 <= v <= y1
                           and (c, round(u), round(v)) not in used_pk]
                    if inb:
                        u, v, _ = max(inb, key=lambda q: q[2])
                        idxs.append(c)
                        uvs.append([u, v])
                        keys.append((c, round(u), round(v)))
                if len(idxs) < 6:
                    if x1 - x0 < 1e8:
                        ig_f.append([max(x0, 0), max(y0, 0),
                                     min(x1, W), min(y1, H)])
                    continue
                ver = verify(idxs, uvs)
                if ver is None:
                    if x1 - x0 < 1e8:
                        ig_f.append([max(x0, 0), max(y0, 0),
                                     min(x1, W), min(y1, H)])
                    continue
                rms, uv8, depth = ver
                used_pk.update(keys)
                vis = np.array([(-8 <= u < W + 8 and -8 <= v < H + 8)
                                for (u, v) in uv8])
                gates_f.append((uv8[0:4], uv8[4:8], vis))
                if len(gates_f) >= G_SLOTS:
                    break
            if not gates_f:
                continue
            inner = np.full((G_SLOTS, 4, 2), np.nan, np.float32)
            outer = np.full((G_SLOTS, 4, 2), np.nan, np.float32)
            vi = np.zeros((G_SLOTS, 4), bool)
            vo = np.zeros((G_SLOTS, 4), bool)
            for gi, (i4, o4, vis) in enumerate(gates_f):
                inner[gi] = i4
                outer[gi] = o4
                vi[gi] = vis[0:4]
                vo[gi] = vis[4:8]
            fidx = len(L["path"])
            L["path"].append(str(p))
            L["inner"].append(inner)
            L["outer"].append(outer)
            L["vis_inner"].append(vi)
            L["vis_outer"].append(vo)
            for b in ig_f:
                L["ig_boxes"].append(b)
                L["ig_fidx"].append(fidx)
            n_lab += 1
        n = len(L["path"])
        if n == 0:
            print(f"{ep.name}: 0 labeled frames, skip")
            continue
        np.savez_compressed(
            out_f,
            path=np.array(L["path"]),
            inner=np.array(L["inner"], np.float32),
            outer=np.array(L["outer"], np.float32),
            vis_inner=np.array(L["vis_inner"]),
            vis_outer=np.array(L["vis_outer"]),
            pos=np.zeros((n, 3), np.float32),
            vel=np.zeros((n, 3), np.float32),
            quat=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
            gate_idx=np.zeros(n, np.int64),
            next_gate_pos=np.zeros((n, 3), np.float32),
            pose_valid=np.zeros(n, np.float32),
            ignore_boxes=np.array(L["ig_boxes"], np.float32).reshape(-1, 4),
            ignore_frame_idx=np.array(L["ig_fidx"], np.int64),
        )
        print(f"{ep.name}: {n_lab}/{len(paths)} frames labeled, "
              f"{len(L['ig_boxes'])} ignore boxes -> {out_f.name}")


if __name__ == "__main__":
    main()
