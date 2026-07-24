"""EKF full-lap replay: vision (GateNet corners) + IMU only — no odometry.

Ground truth is used ONLY to (a) initialize the state once at t0 and
(b) grade the output afterward. The filter itself sees: IMU samples, camera
frames, the static gate map, and the camera clock offset.

    .venv-train\\Scripts\\python.exe scripts\\ekf_replay.py \\
        [--episode rc_20260723_022654]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ingest import load_training_episode
from aigp.ekf import GateEKF
from aigp.calib.solve import SegmentedInterp
from aigp.vision.labels import load_calib, gate_quads_world
from aigp.vision.model import GateNet
from scripts.train_net import decode_corners, orange_channel
from scripts.ekf_bringup import load_imu

ROOT = Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379\ai-grand-prix\outputs\captures")
W, H = 640, 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="rc_20260723_022654")
    ap.add_argument("--ckpt", default=str(REPO / "data/models/gatenet_v6wsl_best.pt"))
    ap.add_argument("--assoc-px", type=float, default=40.0)
    args = ap.parse_args()

    calib = load_calib(REPO / "data/calib/calib.json")
    fx, fy, cx, cy = calib["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    b = load_training_episode(ROOT / args.episode)
    gates = b["gates"]
    quads = [gate_quads_world(g) for g in gates]
    gate_world = [np.concatenate([h, p]) for (h, p) in quads]  # (8,3) each

    od = b["odom"]
    # longest continuous GT segment: split on clock restarts AND position
    # teleports (mid-segment sim resets jump the truth to the start pad)
    jump = np.linalg.norm(np.diff(od[:, 1:4], axis=0), axis=1)
    breaks = np.where((np.diff(od[:, 0]) <= 0) | (jump > 3.0))[0] + 1
    seg = max(np.split(np.arange(len(od)), breaks), key=len)
    od = od[seg]
    t_od = od[:, 0] * 1e-6

    imu = load_imu(ROOT / args.episode)
    imu = imu[(imu[:, 0] >= t_od[0]) & (imu[:, 0] <= t_od[-1])]

    # camera frame -> telemetry-clock mapping (clock alignment only)
    interp = SegmentedInterp(b["odom"])
    interp.fit_offsets(b["frames"])
    frames = []
    for (fid, sim_ns, wall_ns, path) in b["frames"]:
        t = interp.frame_time(sim_ns, wall_ns)
        if t is None:
            continue
        ts = (t - calib["dt_us"]) * 1e-6
        if t_od[0] + 0.3 < ts < t_od[-1] - 0.3:
            frames.append((ts, path))
    frames.sort()
    print(f"{len(frames)} frames, {len(imu)} imu samples, "
          f"span {t_od[-1]-t_od[0]:.1f}s")

    # net
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = GateNet().to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    def gt_state(t):
        i = np.clip(np.searchsorted(t_od, t), 1, len(od) - 1)
        a = (t - t_od[i - 1]) / max(t_od[i] - t_od[i - 1], 1e-9)
        p = (1 - a) * od[i - 1, 1:4] + a * od[i, 1:4]
        qr = od[i, 4:8]
        R = Rotation.from_quat([qr[1], qr[2], qr[3], qr[0]])
        j0, j1 = max(0, i - 3), min(len(od) - 1, i + 3)
        v = (od[j1, 1:4] - od[j0, 1:4]) / max(t_od[j1] - t_od[j0], 1e-9)
        return p, v, qr, R

    # ---- init once from truth at the first frame time
    t0 = frames[0][0]
    p0, v0, q0, _ = gt_state(t0)
    ekf = GateEKF(K, calib["R_cb"], sigma_px=1.0)
    ekf.init_state(p0, v0, q0, t0)

    # ---- run
    errs = []       # (t, pos_err, att_err_deg, n_upd, gap_since_upd, verr)
    fi = 0
    last_upd_t = t0
    prev_t = None
    t_end = frames[-1][0] + 1.0   # grade only within camera-covered span
    for i in range(len(imu)):
        t_imu = imu[i, 0]
        if t_imu < t0:
            continue
        if t_imu > t_end:
            print(f"  [eval ends at t={t_imu-t0:.1f}s: camera span over]",
                  flush=True)
            break
        if prev_t is not None and t_imu - prev_t > 1.5:
            print(f"  [truncated at t={prev_t-t0:.1f}s: {t_imu-prev_t:.1f}s "
                  f"recorder data hole]", flush=True)
            break
        prev_t = t_imu
        ekf.propagate(t_imu, imu[i, 1:4], imu[i, 4:7])
        # any camera frames due?
        n_upd = 0
        while fi < len(frames) and frames[fi][0] <= t_imu:
            ts, path = frames[fi]
            fi += 1
            img = cv2.imread(str(path))
            if img is None:
                continue
            arr = np.concatenate([img.astype(np.float32) / 255.0,
                                  orange_channel(img)[..., None]], 2
                                 ).transpose(2, 0, 1)
            x = torch.from_numpy(arr).unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                                 enabled=device == "cuda"):
                out = model(x)
            dec = decode_corners(out["hm"][0].float().cpu(),
                                 out["off"][0].float().cpu(), thresh=0.2)
            # covariance-adaptive association radius: 3-sigma of the
            # projected position uncertainty (+ base), per gate depth
            sig_p = float(np.sqrt(max(np.trace(ekf.P[0:3, 0:3]), 0.0)))
            obs = []
            for gi in range(len(gates)):
                c_pred, c_Xc = ekf.predict_pixel(gate_world[gi].mean(0))
                if c_pred is None:
                    continue
                depth = max(c_Xc[2], 2.0)
                rad = np.clip(3.0 * K[0, 0] * sig_p / depth + 12.0,
                              15.0, 160.0)
                for k in range(8):
                    Xw = gate_world[gi][k]
                    uv_pred, Xc = ekf.predict_pixel(Xw)
                    if uv_pred is None:
                        continue
                    if not (-60 < uv_pred[0] < W + 60 and -60 < uv_pred[1] < H + 60):
                        continue
                    best = None
                    for (u, v, s) in dec[k]:
                        d = np.hypot(u - uv_pred[0], v - uv_pred[1])
                        if d < rad and (best is None or d < best[0]):
                            best = (d, u, v)
                    if best is not None:
                        obs.append((Xw, np.array([best[1], best[2]])))
            n_upd += ekf.update_corners(obs)

            # ---- lost-mode relocalization: long gap + nothing accepted,
            # but the net sees corners -> re-seed from pose-head prior +
            # joint PnP over prior-associated peaks
            if n_upd == 0 and (t_imu - last_upd_t) > 1.0:
                ph = out["pos"][0].float().cpu().numpy() * 50.0
                from aigp.vision.model import rot6d_to_matrix
                Rph = rot6d_to_matrix(out["rot6"].float())[0].cpu().numpy()
                R_cw_ph = np.asarray(calib["R_cb"]) @ Rph.T
                objp, imgp = [], []
                for gi in range(len(gates)):
                    for k in range(8):
                        Xc = R_cw_ph @ (gate_world[gi][k] - ph)
                        if Xc[2] < 1.0:
                            continue
                        u = K[0, 0] * Xc[0] / Xc[2] + K[0, 2]
                        v = K[1, 1] * Xc[1] / Xc[2] + K[1, 2]
                        if not (0 <= u < W and 0 <= v < H):
                            continue
                        best = None
                        for (du, dv, s) in dec[k]:
                            d = np.hypot(du - u, dv - v)
                            if d < 80 and (best is None or d < best[0]):
                                best = (d, du, dv)
                        if best is not None:
                            objp.append(gate_world[gi][k])
                            imgp.append([best[1], best[2]])
                if len(objp) >= 10:
                    objp = np.ascontiguousarray(objp, np.float64)
                    imgp2 = np.ascontiguousarray(imgp, np.float64).reshape(-1, 1, 2)
                    okp, rvec, tvec = cv2.solvePnP(objp, imgp2, K, None,
                                                   flags=cv2.SOLVEPNP_SQPNP)
                    if okp:
                        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
                        rms = float(np.sqrt(((proj - imgp2) ** 2)
                                            .sum(axis=2).mean()))
                        if rms < 2.5:
                            R_cw, _ = cv2.Rodrigues(rvec)
                            p_new = (-R_cw.T @ tvec.ravel())
                            R_wb = R_cw.T @ np.asarray(calib["R_cb"])
                            qx = Rotation.from_matrix(R_wb).as_quat()
                            ekf.init_state(p_new, ekf.v,
                                           [qx[3], qx[0], qx[1], qx[2]],
                                           ekf.t, pos_std=0.3, vel_std=1.5,
                                           ang_std=0.05)
                            n_upd += len(objp)
                            print(f"  reloc @t={t_imu-t0:.1f}s "
                                  f"({len(objp)} pts, rms {rms:.2f}px)",
                                  flush=True)
        if n_upd:
            last_upd_t = t_imu
        # grade
        pg, vg, qg, Rg = gt_state(t_imu)
        perr = np.linalg.norm(ekf.p - pg)
        verr = np.linalg.norm(ekf.v - vg)
        cosang = (np.trace(ekf.q.as_matrix().T @ Rg.as_matrix()) - 1) / 2
        aerr = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        errs.append((t_imu - t0, perr, aerr, n_upd, t_imu - last_upd_t, verr))

    e = np.array(errs)
    print(f"\n=== EKF vs truth over {e[-1,0]:.1f}s "
          f"(truth used only at t0) ===")
    print(f"position err: median {np.median(e[:,1])*100:6.1f} cm  "
          f"p90 {np.percentile(e[:,1],90)*100:6.1f} cm  "
          f"max {e[:,1].max()*100:6.1f} cm")
    print(f"attitude err: median {np.median(e[:,2]):6.2f} deg  "
          f"p90 {np.percentile(e[:,2],90):6.2f} deg  "
          f"max {e[:,2].max():6.2f} deg")
    upd_frac = (e[:, 3] > 0).sum() / max((np.diff(e[:, 0]) > 0).sum(), 1)
    print(f"corner updates accepted on {100*(e[:,3]>0).mean():.1f}% of imu ticks; "
          f"mean corners/update {e[e[:,3]>0,3].mean():.1f}")
    m_gap = e[:, 4] > 0.5
    if m_gap.any():
        print(f"during >0.5s vision gaps ({100*m_gap.mean():.1f}% of time): "
              f"pos median {np.median(e[m_gap,1])*100:.1f} cm  "
              f"p90 {np.percentile(e[m_gap,1],90)*100:.1f} cm")
    # error vs time buckets
    for lo, hi in ((0, 20), (20, 50), (50, 1000)):
        m = (e[:, 0] >= lo) & (e[:, 0] < hi)
        if m.sum() > 50:
            print(f"  t {lo:3d}-{hi:3d}s: pos median "
                  f"{np.median(e[m,1])*100:6.1f} cm  p90 "
                  f"{np.percentile(e[m,1],90)*100:6.1f} cm")
    np.save(REPO / "data" / "ekf_trace.npy", e)
    print("trace -> data/ekf_trace.npy "
          "(t, perr, aerr, n_upd, gap, verr)")


if __name__ == "__main__":
    main()
