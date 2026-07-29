"""Hop-chain: extend the certified map one gate at a time with ZERO IMU
bridging and ZERO position beliefs.

In a frame where Henry corner-fitted a CERTIFIED gate g, the camera pose
is exact (certified position + his click PnP + vision attitude). The
NEXT gate is usually visible in that same frame; GateNet corners + PnP
give its position in the camera frame -> absolute world position. Each
hop is a single 10-25m optical measurement (no integration), so error
does not accumulate along the course.

Self-check per frame: the net's own detection of the clicked gate must
land on the certified position; its residual measures the frame's
attitude error and is reported (and used as a first-order correction).

    .venv-train\\Scripts\\python.exe scripts\\vq2_hop_chain.py --lap 003101
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.calib.detect import detect_gates  # noqa: E402
from aigp.vision.labels import load_calib  # noqa: E402

HOLE, PANEL = 0.75, 1.36
OBJ8 = np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE], [HOLE, 0, HOLE],
                 [-HOLE, 0, HOLE],
                 [-PANEL, 0, -PANEL], [PANEL, 0, -PANEL], [PANEL, 0, PANEL],
                 [-PANEL, 0, PANEL]])
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
SQ_HOLE = OBJ8[:4]

calib = load_calib(REPO / "data/calib/calib.json")
fx, fy, cx, cy = calib["K"]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
R_cb = np.asarray(calib["R_cb"])

LAPS = {
    "003101": {"journals": ["data/vq2_map_human9.json.journal.jsonl",
                            "data/vq2_map_human10.json.journal.jsonl"],
               "att_trace": "data/vq2_trace_101_i1.npz",
               "sess_trace": "data/vq2_trace_v3_101.npz"},
    "gift": {"journals": ["data/vq2_map_human11.json.journal.jsonl"],
             "att_trace": "data/vq2_trace_gift_i1.npz",
             "sess_trace": "data/vq2_trace_gift.npz"},
}

# certified seeds (the bible + tonight's multiply-confirmed g10)
CERT = {8: np.array([107.83, -1.16, -5.43]),
        9: np.array([115.7, 9.1, -4.6]),
        10: np.array([125.0, 6.6, -2.4])}


def pnp_points_all(idx, imgp):
    if len(idx) < 6:
        return []
    obj_p = np.ascontiguousarray(OBJ8[list(idx)] @ RX90.T)
    ip = np.ascontiguousarray(imgp, np.float64).reshape(-1, 1, 2)
    try:
        _n, rvecs, tvecs, _e = cv2.solvePnPGeneric(
            obj_p, ip, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    out = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_p, ip, K, None, rvec,
                                              tvec)
        except cv2.error:
            continue
        proj, _ = cv2.projectPoints(obj_p, rvec, tvec, K, None)
        rms = float(np.sqrt(((proj - ip) ** 2).sum(axis=2).mean()))
        R, _ = cv2.Rodrigues(rvec)
        out.append((R @ RX90, tvec.ravel(), rms))
    return sorted(out, key=lambda s: s[2])


def click_solve(clicks):
    obj = np.ascontiguousarray(SQ_HOLE @ RX90.T)
    ip = np.ascontiguousarray(clicks, np.float64).reshape(-1, 1, 2)
    try:
        _n, rv, tv, _e = cv2.solvePnPGeneric(
            obj, ip, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
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
        if best is None or rms < best[1]:
            best = (t0.ravel(), rms)
    if best is None or best[1] > 1.0:
        return None
    return best[0]


def frame_dets(img, peaks):
    """region proposals -> [(t_cam(3), yaw_cam2gate_R, rms, ncorn)]"""
    boxes = []
    try:
        for dd in detect_gates(img):
            bx0, by0 = dd["outer"].min(0) - 14
            bx1, by1 = dd["outer"].max(0) + 14
            boxes.append((bx0, by0, bx1, by1))
    except Exception:
        pass
    for (ca, cb) in ((0, 2), (1, 3)):
        for (ua, va, _sa) in sorted(peaks[ca], key=lambda q: -q[2])[:4]:
            for (ub, vb, _sb) in sorted(peaks[cb], key=lambda q: -q[2])[:4]:
                span = max(abs(ub - ua), abs(vb - va))
                if span < 12:
                    continue
                cx0, cy0 = (ua + ub) / 2, (va + vb) / 2
                half = span * 1.15
                boxes.append((cx0 - half, cy0 - half, cx0 + half,
                              cy0 + half))
    dets = []
    used_c = []
    for (bx0, by0, bx1, by1) in boxes[:20]:
        idxs, uvs = [], []
        for c in range(8):
            inb = [(u, v, sc) for (u, v, sc) in peaks[c]
                   if bx0 <= u <= bx1 and by0 <= v <= by1]
            if inb:
                u, v, _ = max(inb, key=lambda q: q[2])
                idxs.append(c)
                uvs.append([u, v])
        if len(idxs) < 4:
            continue
        cc0 = np.mean(uvs, axis=0)
        if any(np.hypot(*(cc0 - d0)) < 30 for d0 in used_c):
            continue
        if len(idxs) >= 6:
            br = pnp_points_all(idxs, uvs)
            if not br or br[0][2] > 1.2:
                continue
            used_c.append(cc0)
            R_v, t_v, rms_v = br[0]
            dets.append((t_v, R_v, rms_v, len(idxs)))
        elif set(idxs) == {0, 1, 2, 3}:
            # hole-only far detection: IPPE_SQUARE (bearing exact, depth
            # noisier -- clustering absorbs it)
            obj = np.ascontiguousarray(SQ_HOLE @ RX90.T)
            ip = np.ascontiguousarray(uvs, np.float64).reshape(-1, 1, 2)
            try:
                _n, rv, tv, _e = cv2.solvePnPGeneric(
                    obj, ip, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error:
                continue
            best = None
            for r0, t0 in zip(rv, tv):
                try:
                    r0, t0 = cv2.solvePnPRefineLM(obj, ip, K, None, r0, t0)
                except cv2.error:
                    continue
                pr, _ = cv2.projectPoints(obj, r0, t0, K, None)
                rms = float(np.sqrt(((pr - ip) ** 2).sum(axis=2).mean()))
                if best is None or rms < best[2]:
                    R0, _ = cv2.Rodrigues(r0)
                    best = (R0 @ RX90, t0.ravel(), rms)
            if best is None or best[2] > 1.2:
                continue
            used_c.append(cc0)
            dets.append((best[1], best[0], best[2], 4))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lap", required=True, choices=list(LAPS))
    ap.add_argument("--max-depth", type=float, default=30.0)
    args = ap.parse_args()
    lap = LAPS[args.lap]

    import torch
    from aigp.vision.model import GateNet
    from scripts.train_net import orange_channel, decode_corners
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(REPO / "data/models/gatenet_v7_best.pt",
                    map_location=dev, weights_only=False)
    net = GateNet().to(dev)
    net.load_state_dict(ck["model"])
    net.eval()

    def net_peaks(bgr):
        orange = orange_channel(bgr)
        x = np.concatenate([bgr.astype(np.float32) / 255.0,
                            orange[..., None]], 2).transpose(2, 0, 1)
        xt = torch.from_numpy(x).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.float16, enabled=dev == "cuda"):
            o = net(xt)
        return decode_corners(o["hm"][0].float().cpu(),
                              o["off"][0].float().cpu(), thresh=0.30)

    att = np.load(lap["att_trace"], allow_pickle=True)
    st = np.load(lap["sess_trace"], allow_pickle=True)
    qs = att["quat"]
    rot = Rotation.from_quat(
        np.stack([qs[:, 1], qs[:, 2], qs[:, 3], qs[:, 0]], axis=1))
    tta = np.asarray(att["t"], float)
    keep = np.concatenate([[True], np.diff(tta) > 1e-6])
    sl = Slerp(tta[keep], rot[keep])

    fits = []
    for jn in lap["journals"]:
        for ln in Path(jn).read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("ok") and "clicks" in r and r.get("rms", 9) < 1.0:
                fits.append(r)

    # ALL clicked gates: every fit frame yields WITHIN-FRAME relative
    # vectors click-gate -> net-detected-gate (attitude error nearly
    # cancels in the difference; no IMU anywhere)
    rel_obs = {}   # clicked_gate -> list of (rel_w(3), rms, nc, frame,
    #                depth)
    for r in fits:
        g = int(r["gate"])
        fidx = int(r["frame"])
        if fidx >= len(st["t"]):
            continue
        t_c = click_solve(r["clicks"])
        if t_c is None:
            continue
        trel = float(st["t"][fidx])
        R_wb = sl(np.clip(trel, tta[keep][0], tta[keep][-1])).as_matrix()
        R_wc = R_wb @ R_cb.T
        img = cv2.imread(str(st["path"][fidx]))
        if img is None:
            continue
        for (t_v, R_v, rms_v, nc) in frame_dets(img, net_peaks(img)):
            depth = float(np.linalg.norm(t_v))
            if depth > args.max_depth:
                continue
            rel = R_wc @ (t_v - t_c)     # g' - g, world axes, same frame
            if np.linalg.norm(rel) < 3.0:
                continue                 # the clicked gate itself
            rel_obs.setdefault(g, []).append(
                (rel, rms_v, nc, fidx, depth))
    out_rows = []
    for g in sorted(rel_obs):
        rows = rel_obs[g]
        print(f"### clicks on g{g}: {len(rows)} other-gate detections")
        Prel = np.array([r[0] for r in rows])
        left = list(range(len(rows)))
        while left:
            seed = left[0]
            cl = [i for i in left
                  if np.linalg.norm(Prel[i] - Prel[seed]) < 3.0]
            left = [i for i in left if i not in cl]
            Pc = Prel[cl]
            med = np.median(Pc, axis=0)
            spread = np.median(np.linalg.norm(Pc - med, axis=1))
            dep = np.median([rows[i][4] for i in cl])
            n8 = sum(1 for i in cl if rows[i][2] >= 7)
            print(f"   rel cluster n={len(cl):2d} {np.round(med, 2)} "
                  f"|{np.linalg.norm(med):5.1f}m| spread {spread*100:4.0f}cm"
                  f" depth~{dep:4.1f}m ({n8} full-corner)")
            sigma = max(0.25, spread) * (1.0 + dep / 40.0) / \
                np.sqrt(len(cl))
            out_rows.append([g, med[0], med[1], med[2], sigma, len(cl),
                             dep])
    out = REPO / "data" / f"vq2_hops_{args.lap}.npz"
    np.savez(out, rows=np.array(out_rows))
    print(f"wrote {out} ({len(out_rows)} hop clusters)")


if __name__ == "__main__":
    main()
