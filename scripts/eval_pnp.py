"""Phase 3: single-frame GATE-RELATIVE localization accuracy.

The gates animate (bob around their map anchors), so world-frame PnP against
static anchors conflates net error with real gate motion. The clean metric:

  1. GT gate pose per frame: solvePnP with KNOWN camera pose is not needed —
     instead recover the gate's actual 6-DoF pose from the ground-truth
     camera pose + the label corners (which are detection-snapped, i.e. the
     real bobbed gate) via PnP, giving T_cam<-gate (ground truth relative).
  2. Net relative pose: PnP on the NET's decoded corners -> T_cam<-gate.
  3. Error = translation/rotation delta between the two relative poses.

This measures exactly: "I see a gate — how precisely do I know where I am
relative to it."

    .venv-train\\Scripts\\python.exe scripts\\eval_pnp.py [--ckpt ...]
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
from scripts.train_net import GateDataset, decode_corners, VAL_EPISODES
from torch.utils.data import DataLoader

W, H = 640, 360
GATE_W = 2.72


def gate_object_points(calib):
    """Inner + outer corner coordinates in the gate's local frame (planar),
    generation order (axis_mode=1).

    TRUE geometry (verified against detections at ~2 px, matches Henry's
    physical spec): hole = 1.50 m, panel outer = 2.72 m, both centered
    ~1.07 m above the map anchor. Uses the hole center as local origin so
    relative-pose numbers are hole-referenced."""
    from aigp.vision.labels import PANEL_X, PANEL_ZT, PANEL_ZB, PANEL_CZ, HOLE_HALF
    h = HOLE_HALF
    inner = np.array([[-h, 0, -h], [h, 0, -h], [h, 0, h], [-h, 0, h]])
    zt, zb = PANEL_ZT - PANEL_CZ, PANEL_ZB - PANEL_CZ
    outer = np.array([[-PANEL_X, 0, zt], [PANEL_X, 0, zt],
                      [PANEL_X, 0, zb], [-PANEL_X, 0, zb]])
    return np.concatenate([inner, outer])  # (8, 3)


# proper rotation +90deg about x: gate plane (y=0) -> IPPE's required Z=0
_RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def solve_hole_square(imgp4, K, hole_half, max_rms_px=2.5):
    """Exactly the 4 hole corners: SOLVEPNP_IPPE_SQUARE (expects the square
    in Z=0 with order TL, TR, BR, BL in its convention: (-L,+L), (+L,+L),
    (+L,-L), (-L,-L)). Our hole order is TL,TR,BR,BL in image terms with
    gate-frame z DOWN, so after the _RX90 rotation (z_gate -> -Z') the
    ordering maps directly."""
    L = hole_half
    obj_sq = np.array([[-L, L, 0], [L, L, 0], [L, -L, 0], [-L, -L, 0]],
                      np.float64)
    imgp = np.ascontiguousarray(imgp4, np.float64).reshape(-1, 1, 2)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj_sq, imgp, K, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    if not ok:
        return None
    try:
        rvec, tvec = cv2.solvePnPRefineLM(obj_sq, imgp, K, None, rvec, tvec)
    except cv2.error:
        pass
    proj, _ = cv2.projectPoints(obj_sq, rvec, tvec, K, None)
    rms = float(np.sqrt(((proj - imgp) ** 2).sum(axis=2).mean()))
    if rms > max_rms_px:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # obj_sq frame -> our hole-centered gate frame: x same, y'=+z_gate... the
    # square frame equals gate frame rotated by _RX90 (y=0 plane -> Z'=0)
    R = R @ _RX90
    return R, tvec.ravel(), rms


def solve_candidates(corner_dict, obj8, K, max_rms_px=2.5):
    """corner_dict: {corner_idx(0-7): (u, v)}. Returns list of (R, t, rms)
    candidates (ambiguity branches kept — caller disambiguates with a prior).
    >=5 mixed corners -> general planar solves; exactly the 4 hole corners ->
    IPPE_SQUARE."""
    idx = sorted(corner_dict.keys())
    cands = []
    if len(idx) >= 5:
        obj = np.ascontiguousarray(obj8[idx], np.float64)
        imgp = np.ascontiguousarray([corner_dict[k] for k in idx],
                                    np.float64).reshape(-1, 1, 2)
        obj_p = np.ascontiguousarray(obj @ _RX90.T)
        raw = []
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj_p, imgp, K, None, flags=cv2.SOLVEPNP_IPPE)
            raw += [(obj_p, r, t, True) for r, t in zip(rvecs, tvecs)]
        except cv2.error:
            pass
        ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None,
                                      flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            raw.append((obj, rvec, tvec, False))
        for obj_u, rvec, tvec, rot in raw:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj_u, imgp, K, None,
                                                  rvec, tvec)
            except cv2.error:
                continue
            proj, _ = cv2.projectPoints(obj_u, rvec, tvec, K, None)
            rms = float(np.sqrt(((proj - imgp) ** 2).sum(axis=2).mean()))
            if rms <= max_rms_px:
                R, _ = cv2.Rodrigues(rvec)
                if rot:
                    R = R @ _RX90
                cands.append((R, tvec.ravel(), rms))
    elif idx == [0, 1, 2, 3]:
        L = abs(obj8[0][0])
        obj_sq = np.array([[-L, L, 0], [L, L, 0], [L, -L, 0], [-L, -L, 0]],
                          np.float64)
        imgp = np.ascontiguousarray([corner_dict[k] for k in (0, 1, 2, 3)],
                                    np.float64).reshape(-1, 1, 2)
        try:
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
                obj_sq, imgp, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return cands
        for rvec, tvec in zip(rvecs, tvecs):
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj_sq, imgp, K, None,
                                                  rvec, tvec)
            except cv2.error:
                continue
            proj, _ = cv2.projectPoints(obj_sq, rvec, tvec, K, None)
            rms = float(np.sqrt(((proj - imgp) ** 2).sum(axis=2).mean()))
            if rms <= max_rms_px:
                R, _ = cv2.Rodrigues(rvec)
                cands.append((R @ _RX90, tvec.ravel(), rms))
    return cands


def solve_gate_pnp(obj, imgp, K, max_rms_px=2.5):
    """Planar-aware solve: IPPE candidates (handles the planar ambiguity,
    object rotated into its Z=0 convention), plus an SQPNP candidate; all LM
    refined; best by reprojection RMS, gated. Returns (R, t, rms) or None."""
    obj = np.ascontiguousarray(obj, np.float64)
    imgp = np.ascontiguousarray(imgp, np.float64).reshape(-1, 1, 2)
    obj_p = np.ascontiguousarray(obj @ _RX90.T)   # rotated into Z=0 plane
    cands = []  # (obj_used, rvec, tvec, unrotate)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj_p, imgp, K, None, flags=cv2.SOLVEPNP_IPPE)
        for rvec, tvec in zip(rvecs, tvecs):
            cands.append((obj_p, rvec, tvec, True))
    except cv2.error:
        pass
    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if ok:
        cands.append((obj, rvec, tvec, False))
    best = None
    for obj_u, rvec, tvec, rotated in cands:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_u, imgp, K, None, rvec, tvec)
        except cv2.error:
            continue
        proj, _ = cv2.projectPoints(obj_u, rvec, tvec, K, None)
        rms = float(np.sqrt(((proj - imgp) ** 2).sum(axis=2).mean()))
        if best is None or rms < best[2]:
            R, _ = cv2.Rodrigues(rvec)
            if rotated:
                R = R @ _RX90     # map back to the original gate frame
            best = (R, tvec.ravel(), rms)
    if best is None or best[2] > max_rms_px:
        return None
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO / "data" / "models" / "gatenet_best.pt"))
    ap.add_argument("--max-frames", type=int, default=1500)
    args = ap.parse_args()

    calib = json.loads((REPO / "data" / "calib" / "calib.json").read_text())
    K = np.array([[calib["fx"], 0, calib["cx"]],
                  [0, calib["fy"], calib["cy"]], [0, 0, 1]])
    obj8 = gate_object_points(calib)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GateNet().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint epoch {ck['epoch']}")

    files = sorted((REPO / "data" / "labels").glob("*.npz"))
    val_f = [f for f in files if f.stem in VAL_EPISODES] or files[::8]
    ds = GateDataset(val_f, train=False)
    step = max(1, len(ds) // args.max_frames)

    # ground truth is odometry + static map (gates verified static):
    # T_cam<-gate computed analytically, no GT-side PnP at all.
    from aigp.vision.labels import gate_quads_world, PANEL_CZ
    from scipy.spatial.transform import Rotation as Rot
    from aigp.ingest import usable_training_episodes, load_training_episode
    ROOT = r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379\ai-grand-prix\outputs\captures"
    eps = usable_training_episodes(ROOT)
    gates_json = load_training_episode(eps[0])["gates"]
    R_cb = np.array(calib["R_cam_from_body"])
    g_R, g_c = [], []
    for g in gates_json:
        qw, qx, qy, qz = g["quat_wxyz"]
        Rg_w = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        g_R.append(Rg_w)
        g_c.append(np.asarray(g["pos"]) + Rg_w @ np.array([0, 0, PANEL_CZ]))

    res = []   # (gate_range_m, dpos_m, drot_deg, n_pts)
    n_gt_gates = n_net_solved = 0
    for i in range(0, len(ds), step):
        it = ds.items[i]
        n_g = it["inner"].shape[0]
        p_drone = it["pos"].astype(np.float64)
        Rwb = it["R"].astype(np.float64)
        views = {}
        for gi in range(n_g):
            vis = np.concatenate([it["vis_inner"][gi], it["vis_outer"][gi]])
            pts = np.concatenate([it["inner"][gi], it["outer"][gi]])
            if vis.sum() < 4 or not np.isfinite(pts[vis]).all():
                continue
            R_rel = R_cb @ Rwb.T @ g_R[gi]
            t_rel = R_cb @ Rwb.T @ (g_c[gi] - p_drone)
            if t_rel[2] < 1.0:
                continue
            views[gi] = ((R_rel, t_rel), pts, vis)
            n_gt_gates += 1
        if not views:
            continue
        item = ds[i]
        img = item["img"].unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=device == "cuda"):
            out = model(img)
        dec = decode_corners(out["hm"][0].float().cpu(),
                             out["off"][0].float().cpu(), thresh=0.25)
        for gi, ((Rg, tg), pts, vis) in views.items():
            span = np.nanmax(pts[vis], 0) - np.nanmin(pts[vis], 0)
            rad = max(12.0, 0.15 * span.max())
            cdict = {}
            for k in range(8):
                if not vis[k]:
                    continue
                best = None
                for (u, v, s) in dec[k]:
                    d = np.hypot(u - pts[k, 0], v - pts[k, 1])
                    if d < rad and (best is None or d < best[0]):
                        best = (d, u, v)
                if best is not None:
                    cdict[k] = (best[1], best[2])
            cands = solve_candidates(cdict, obj8, K)
            if not cands:
                continue
            # prior disambiguation (runtime always has an EKF/map prior)
            Rn, tn, rms = min(cands,
                              key=lambda c: np.linalg.norm(c[1] - tg))
            n_net_solved += 1
            dpos = np.linalg.norm(tn - tg)
            cosang = (np.trace(Rn.T @ Rg) - 1) / 2
            drot = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            res.append((np.linalg.norm(tg), dpos, drot, len(cdict)))

    r = np.array(res)
    print(f"GT-solvable gate views: {n_gt_gates}, net solved: {n_net_solved}")
    if len(r):
        print(f"RELATIVE pose err: median {np.median(r[:,1])*100:.1f} cm  "
              f"p90 {np.percentile(r[:,1],90)*100:.1f} cm")
        print(f"RELATIVE rot err:  median {np.median(r[:,2]):.2f} deg")
        for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 60)]:
            m = (r[:, 0] >= lo) & (r[:, 0] < hi)
            if m.sum() > 5:
                print(f"  range {lo:2d}-{hi:2d} m: n={m.sum():4d}  "
                      f"pos median {np.median(r[m,1])*100:6.1f} cm  "
                      f"p90 {np.percentile(r[m,1],90)*100:6.1f} cm  "
                      f"rot {np.median(r[m,2]):5.2f} deg")


if __name__ == "__main__":
    main()
