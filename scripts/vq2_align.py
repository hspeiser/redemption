"""VQ2 anchor + first EKF flight, no ground truth anywhere.

1. Spawn anchor: at rest before the run, PnP the visible start gate
   (classical detector, spec geometry) -> gate-0 pose in the drone's local
   frame (attitude from gravity, local yaw defined = 0 at init) -> solves
   the map anchor (translation + yaw) in closed form. Saved to
   data/vq2_anchor.json.
2. EKF over the flight: IMU propagation + classical hole/panel corners
   against the anchored map. Reports self-consistency: fused-corner rate,
   innovation RMS, per-gate reprojection residuals, gate-pass timing vs
   race status. Optional overlay video from the filter's belief.

    uv run python scripts/vq2_align.py --episode-dir <rc_...> [--video ...]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ekf import GateEKF
from aigp.calib.detect import detect_gates
from aigp.vision.labels import load_calib
from aigp.vq2_map import load_vq2_map, gate_quads_world_vq2
from scripts.ekf_bringup import load_imu

W, H = 640, 360
HOLE, PANEL = 0.75, 1.35
OBJ8 = np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE], [HOLE, 0, HOLE],
                 [-HOLE, 0, HOLE],
                 [-PANEL, 0, -PANEL], [PANEL, 0, -PANEL], [PANEL, 0, PANEL],
                 [-PANEL, 0, PANEL]])
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def load_frames(ep, imu, rig_wall_pair=True):
    """(t_imu_clock_s, path) per unique frame; clock offset via wall pairing
    of the rig's own timestamps."""
    rows = []
    with open(ep / "frames.jsonl") as fh:
        seen = set()
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["frame_id"] in seen:
                continue
            seen.add(r["frame_id"])
            rows.append((r["sim_time_ns"], r["wall"],
                         ep / "frames" / f"{r['idx']:06d}.jpg"))
    rows.sort()
    # imu rows in imu.jsonl carry wall too — reload minimal for pairing
    iw = []
    with open(ep / "imu.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "time_usec" in r and "wall" in r:
                iw.append((r["wall"], r["time_usec"] * 1e-6))
    iw = np.array(iw)
    offs = []
    for (sim_ns, wall, _p) in rows[:: max(1, len(rows) // 300)]:
        k = np.argmin(np.abs(iw[:, 0] - wall))
        offs.append(sim_ns * 1e-9 - iw[k, 1])
    off = float(np.median(offs))
    return [((sim_ns * 1e-9 - off), p) for (sim_ns, _w, p) in rows], off


def load_race_status(ep):
    """(wall, active_gate) rows from the recorder's parsed race status."""
    rows = []
    with open(ep / "mav.jsonl") as fh:
        for line in fh:
            if '"race_status"' not in line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append((r["wall"], int(r["active_gate"]),
                         int(r["race_start_ms"])))
    return rows


def pnp_gate_all(det, K):
    """All refined IPPE branches for an 8-corner detection:
    [(R_g2c, t, rms), ...] sorted by rms."""
    if det["inner"] is None:
        return []
    imgp = np.concatenate([det["inner"], det["outer"]])
    obj_p = np.ascontiguousarray(OBJ8 @ RX90.T)
    ip = np.ascontiguousarray(imgp, np.float64).reshape(-1, 1, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj_p, ip, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return []
    out = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_p, ip, K, None, rvec, tvec)
        except cv2.error:
            continue
        proj, _ = cv2.projectPoints(obj_p, rvec, tvec, K, None)
        rms = float(np.sqrt(((proj - ip) ** 2).sum(axis=2).mean()))
        R, _ = cv2.Rodrigues(rvec)
        out.append((R @ RX90, tvec.ravel(), rms))
    return sorted(out, key=lambda s: s[2])


def pnp_gate(det, K):
    """8-corner PnP (hole+panel, spec geometry). Returns (R_go, t, rms)."""
    if det["inner"] is None:
        return None
    imgp = np.concatenate([det["inner"], det["outer"]])
    # correspondence by construction: detector orders quads consistently
    obj_p = np.ascontiguousarray(OBJ8 @ RX90.T)
    ip = np.ascontiguousarray(imgp, np.float64).reshape(-1, 1, 2)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj_p, ip, K, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    best = None
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_p, ip, K, None, rvec, tvec)
        except cv2.error:
            continue
        proj, _ = cv2.projectPoints(obj_p, rvec, tvec, K, None)
        rms = float(np.sqrt(((proj - ip) ** 2).sum(axis=2).mean()))
        if best is None or rms < best[2]:
            R, _ = cv2.Rodrigues(rvec)
            best = (R @ RX90, tvec.ravel(), rms)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--rest-secs", type=float, default=3.0)
    ap.add_argument("--no-mirror", action="store_true",
                    help="disable the East-mirror map correction (validated "
                         "against co-visible gate pairs: the gate_map.json "
                         "frame is E-flipped vs the sim world)")
    ap.add_argument("--dt-offset", type=float, default=0.0,
                    help="seconds added to camera timestamps (clock probe)")
    args = ap.parse_args()
    args.mirror_e = not args.no_mirror
    ep = Path(args.episode_dir)

    calib = load_calib(REPO / "data/calib/calib.json")
    fx, fy, cx, cy = calib["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R_cb = np.asarray(calib["R_cb"])

    imu = load_imu(ep)
    frames, clock_off = load_frames(ep, imu)
    if args.dt_offset:
        frames = [(t + args.dt_offset, p) for (t, p) in frames]
    print(f"{len(frames)} frames, {len(imu)} imu samples, "
          f"clock offset {clock_off:.3f}s")

    # ---------- rest attitude from gravity (local yaw := 0) ----------
    # still period = from start until the first sustained gyro activity
    # (the arena lights come on some seconds in; the drone sits parked the
    # whole time, so use the ENTIRE parked stretch, not a fixed 3 s)
    t_start = imu[0, 0]
    gmag = np.abs(imu[:, 4:7]).max(axis=1)
    moving = np.convolve((gmag > 0.05).astype(float), np.ones(12) / 12,
                         "same") > 0.5
    k_move = int(np.argmax(moving)) if moving.any() else len(imu)
    t_still_end = imu[max(k_move - 1, 0), 0]
    m_rest = imu[:, 0] <= t_still_end
    if m_rest.sum() < 60:                      # <0.5 s parked: fall back
        m_rest = imu[:, 0] < t_start + args.rest_secs
        t_still_end = t_start + args.rest_secs
    gyro_still = np.abs(imu[m_rest, 4:7]).max() < 0.02
    f_rest = imu[m_rest, 1:4].mean(axis=0) * GateEKF.ACCEL_SIGN
    print(f"still period {t_still_end - t_start:.1f}s ({m_rest.sum()} samples), "
          f"still={gyro_still}, |f|={np.linalg.norm(f_rest):.3f}")
    # find R (yaw=0) with R^T @ (-g z) = f_rest  ->  roll/pitch from gravity
    zb = -f_rest / np.linalg.norm(f_rest)          # body 'down' direction
    # at rest f = R^T (0,0,-g): f_x = g sin(pitch), f_y = -g cos(pitch) sin(roll)
    pitch = np.arcsin(np.clip(f_rest[0] / 9.81, -1, 1))
    roll = np.arctan2(-f_rest[1], -f_rest[2])
    R0 = Rotation.from_euler("ZYX", [0.0, pitch, roll])
    print(f"rest attitude: pitch {np.degrees(pitch):+.2f} deg, "
          f"roll {np.degrees(roll):+.2f} deg")

    # ---------- spawn anchor: PnP the start gate from rest frames ----------
    # scan EVERY frame of the still period (the arena is dark at first; the
    # gate only lights up seconds in) and take the median anchor across all
    # clean detections — the spread is a free self-consistency check
    m0 = json.loads(
        (Path(r"C:\Users\henry\Downloads\gate_map.json")).read_text())
    R_wc = R0.as_matrix() @ R_cb.T           # camera -> world(local)
    sols = []                                 # (p_g0, yaw_g0, rms)
    rest_frames = [(ts, p) for (ts, p) in frames if ts <= t_still_end - 0.1]
    for (ts, path) in rest_frames:
        img = cv2.imread(str(path))
        if img is None:
            continue
        dets = [d for d in detect_gates(img, min_area=500)
                if d["inner"] is not None]
        if not dets:
            continue
        det = max(dets, key=lambda d0: d0["area"])
        sol = pnp_gate(det, K)
        if sol is None or sol[2] > 2.0:
            continue
        R_g2c, t_g2c, rms = sol
        p_g0 = R_wc @ t_g2c                  # gate centre in local frame
        R_g0 = R_wc @ R_g2c                  # gate rotation in local frame
        gx = R_g0[:, 0]                      # gate x-axis -> gate yaw
        sols.append((p_g0, np.degrees(np.arctan2(gx[1], gx[0])), rms))
    if not sols:
        print(f"FAILED: no clean start-gate detection in the "
              f"{len(rest_frames)} still-period frames")
        return 1
    P = np.array([s[0] for s in sols])
    yaws = np.array([s[1] for s in sols])
    yaw_med = np.degrees(np.arctan2(np.median(np.sin(np.radians(yaws))),
                                    np.median(np.cos(np.radians(yaws)))))
    p_med = np.median(P, axis=0)
    spread_p = np.linalg.norm(P - p_med, axis=1)
    map_g0_yaw = -m0["gate_yaw_deg"][0] if args.mirror_e else m0["gate_yaw_deg"][0]
    a_yaw = yaw_med - map_g0_yaw
    anchor = {"anchor_t": [float(v) for v in p_med],
              "anchor_yaw_deg": float(a_yaw),
              "n_anchor_frames": len(sols),
              "pos_spread_p90_m": float(np.percentile(spread_p, 90)),
              "yaw_spread_p90_deg": float(np.percentile(
                  np.abs((yaws - yaw_med + 180) % 360 - 180), 90)),
              "pnp_rms_px": float(np.median([s[2] for s in sols]))}
    print(f"ANCHOR from {len(sols)} still frames: gate0 at local "
          f"{np.round(p_med, 2)}, gate yaw {yaw_med:.1f} deg -> anchor yaw "
          f"{a_yaw:.1f} deg")
    print(f"  spread p90: {anchor['pos_spread_p90_m']*100:.1f}cm / "
          f"{anchor['yaw_spread_p90_deg']:.2f}deg, median pnp rms "
          f"{anchor['pnp_rms_px']:.2f}px")
    (REPO / "data" / "vq2_anchor.json").write_text(json.dumps(anchor, indent=2))

    gates = load_vq2_map(anchor["anchor_t"], anchor["anchor_yaw_deg"],
                         mirror_e=args.mirror_e)
    gate_world = [np.concatenate(gate_quads_world_vq2(g)) for g in gates]
    gate_R = [Rotation.from_quat([g["quat_wxyz"][1], g["quat_wxyz"][2],
                                  g["quat_wxyz"][3], g["quat_wxyz"][0]]
                                 ).as_matrix() for g in gates]

    # race-status active gate on the imu clock (wall-paired)
    iw = []
    with open(ep / "imu.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "time_usec" in r and "wall" in r:
                iw.append((r["wall"], r["time_usec"] * 1e-6))
    iw = np.array(iw)
    rs = load_race_status(ep)
    rs_t = np.interp([w for (w, _a, _s) in rs], iw[:, 0], iw[:, 1])
    rs_ag = np.array([a for (_w, a, _s) in rs])
    # scrub the stale pre-reset tail at the head (ag=17 before race re-arms)
    k0 = int(np.argmax(rs_ag == 0)) if (rs_ag == 0).any() else 0
    rs_t, rs_ag = rs_t[k0:], rs_ag[k0:]

    def active_gate(t):
        if len(rs_t) == 0 or t < rs_t[0]:
            return 0
        return int(rs_ag[min(np.searchsorted(rs_t, t, "right") - 1,
                             len(rs_ag) - 1)])

    # ---------- EKF over the flight (classical corners) ----------
    # classical corners on VQ2 neon gates are 5-15px noisy (bloom), nothing
    # like GateNet's sub-pixel output — χ² gate must reflect that
    ekf = GateEKF(K, R_cb, sigma_px=5.0)
    q0 = R0.as_quat()
    ekf.init_state(np.zeros(3), np.zeros(3),
                   [q0[3], q0[0], q0[1], q0[2]], t_start)
    vw = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                         (W, H)) if args.video else None

    fi = 0
    stats = []   # (t, n_fused, innov_rms, sigma_p, n_clean)
    relocs = []  # (t, gate, att_angle_deg, pnp_rms)
    edge_checks = []  # (t, pinned_gate, other_gate, pixel_err)
    last_upd = t_start
    prev_t = None
    for i in range(len(imu)):
        t_imu = imu[i, 0]
        if prev_t is not None and t_imu - prev_t > 1.5:
            print(f"[data hole at t={t_imu - t_start:.1f}s]", flush=True)
        prev_t = t_imu
        ekf.propagate(t_imu, imu[i, 1:4], imu[i, 4:7])
        while fi < len(frames) and frames[fi][0] <= t_imu:
            ts, path = frames[fi]
            fi += 1
            img = cv2.imread(str(path))
            if img is None:
                continue
            dets = detect_gates(img, min_area=250)
            obs = []
            innovs = []
            n_clean = 0
            clean_dets = []
            sig_p = float(np.sqrt(max(np.trace(ekf.P[0:3, 0:3]), 0)))
            for det in dets:
                # gate sanity: concentric inner+outer with the spec area
                # ratio ((1.35/0.75)^2 = 3.24) — kills the gold "Station"
                # signage and other neon false positives
                if det["inner"] is None:
                    continue
                a_out = cv2.contourArea(det["outer"].astype(np.float32))
                a_in = cv2.contourArea(det["inner"].astype(np.float32))
                if a_in <= 0 or not (2.0 < a_out / a_in < 5.5):
                    continue
                span = np.sqrt(a_out)
                if np.linalg.norm(det["outer"].mean(0) -
                                  det["inner"].mean(0)) > 0.25 * span:
                    continue
                quads = [("outer", det["outer"], 4),
                         ("inner", det["inner"], 0)]
                n_clean += 1
                clean_dets.append(det)
                dc = det["outer"].mean(0)
                # nearest predicted gate among race-status candidates
                ag = active_gate(t_imu)
                cand = [g for g in (ag - 1, ag, ag + 1)
                        if 0 <= g < len(gates)]
                best_gi, best_d = None, 1e9
                for gi in cand:
                    uvp, Xc = ekf.predict_pixel(np.asarray(gates[gi]["pos"]))
                    if uvp is None:
                        continue
                    rad = np.clip(3 * K[0, 0] * sig_p / max(Xc[2], 2.0) + 25,
                                  30, 200)
                    d = np.hypot(uvp[0] - dc[0], uvp[1] - dc[1])
                    if d < rad and d < best_d:
                        best_gi, best_d = gi, d
                if best_gi is None:
                    continue
                for (kind, quad_px, off) in quads:
                    world4 = gate_world[best_gi][off:off + 4] if kind == "inner" \
                        else gate_world[best_gi][4:8]
                    world4 = gate_world[best_gi][0:4] if kind == "inner" else \
                        gate_world[best_gi][4:8]
                    # greedy corner match against prediction
                    for k in range(4):
                        uvp, _ = ekf.predict_pixel(world4[k])
                        if uvp is None:
                            continue
                        dists = np.linalg.norm(quad_px - uvp, axis=1)
                        j = int(np.argmin(dists))
                        if dists[j] < 60:
                            obs.append((world4[k], quad_px[j]))
                            innovs.append(dists[j])
            # filter-independent map-edge check: pose from the active gate's
            # own PnP, then see where the map puts the OTHER co-visible
            # detection. Validates gate-to-gate map geometry directly.
            if len(clean_dets) >= 2:
                dsort = sorted(clean_dets, key=lambda d0: -d0["area"])
                det0 = dsort[0]
                ag = active_gate(t_imu)
                Rb = ekf.q.as_matrix()
                pin = None
                for (R_g2c, t_pnp, rms) in pnp_gate_all(det0, K):
                    if rms > 2.5:
                        continue
                    R_wb = gate_R[ag] @ R_g2c.T @ R_cb
                    cosang = (np.trace(R_wb.T @ Rb) - 1) / 2
                    angd = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                    if angd < 15:
                        p_pin = np.asarray(gates[ag]["pos"]) - \
                            gate_R[ag] @ (R_g2c.T @ t_pnp)
                        pin = (p_pin, R_wb)
                        break
                if pin is not None:
                    p_pin, R_wb = pin
                    for do in dsort[1:2]:
                        dc_o = do["outer"].mean(0)
                        best_e, best_g = None, None
                        for gj in range(len(gates)):
                            if gj == ag:
                                continue
                            Xc = R_cb @ R_wb.T @ (
                                np.asarray(gates[gj]["pos"]) - p_pin)
                            if Xc[2] < 1.0:
                                continue
                            uv = np.array([K[0, 0] * Xc[0] / Xc[2] + K[0, 2],
                                           K[1, 1] * Xc[1] / Xc[2] + K[1, 2]])
                            e = float(np.hypot(*(uv - dc_o)))
                            if best_e is None or e < best_e:
                                best_e, best_g = e, gj
                        if best_e is not None and best_e < 250:
                            edge_checks.append((t_imu - t_start, ag, best_g,
                                                best_e))

            n = ekf.update_corners(obs)
            if n:
                last_upd = t_imu
            elif clean_dets and (t_imu - last_upd) > 0.8:
                # lost -> PnP-relocalize against the race-status active gate;
                # branch/back-face disambiguated by attitude agreement (the
                # gyro is noiseless, so belief attitude is trustworthy)
                det = max(clean_dets, key=lambda d0: d0["area"])
                ag = active_gate(t_imu)
                Rb = ekf.q.as_matrix()
                best = None
                for gi in [g for g in (ag, ag + 1, ag - 1)
                           if 0 <= g < len(gates)]:
                    for (R_g2c, t_pnp, rms) in pnp_gate_all(det, K):
                        if rms > 3.0:
                            continue
                        R_wb = gate_R[gi] @ R_g2c.T @ R_cb
                        cosang = (np.trace(R_wb.T @ Rb) - 1) / 2
                        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                        if ang > 15:
                            continue
                        p_new = np.asarray(gates[gi]["pos"]) - \
                            gate_R[gi] @ (R_g2c.T @ t_pnp)
                        if best is None or ang < best[1]:
                            best = (p_new, ang, gi, rms)
                if best is not None:
                    q = ekf.q.as_quat()
                    ekf.init_state(best[0], ekf.v,
                                   [q[3], q[0], q[1], q[2]], t_imu,
                                   pos_std=0.4, vel_std=0.8, ang_std=0.03)
                    relocs.append((t_imu - t_start, best[2], best[1],
                                   best[3]))
                    last_upd = t_imu
            stats.append((t_imu - t_start, n,
                          float(np.sqrt(np.mean(np.square(innovs)))) if innovs
                          else np.nan, sig_p, n_clean))
            if vw is not None:
                vis = img.copy()
                for gi in range(len(gates)):
                    pts = []
                    ok_all = True
                    for Xw in gate_world[gi][0:4]:
                        uvp, _ = ekf.predict_pixel(Xw)
                        if uvp is None or abs(uvp[0]) > 4000:
                            ok_all = False
                            break
                        pts.append(uvp)
                    if ok_all:
                        cv2.polylines(vis, [np.array(pts, np.int32)], True,
                                      (0, 255, 0), 2)
                cv2.putText(vis, f"t {ts - t_start:5.1f}s fused {n:2d} "
                            f"sigma {sig_p*100:5.1f}cm",
                            (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                vw.write(vis)
    if vw is not None:
        vw.release()

    s = np.array(stats)
    upd = s[:, 1] > 0
    print("\nper-5s: frames | frames-with-clean-det | fused | med innov px | sigma cm")
    for b0 in np.arange(0, s[-1, 0], 5.0):
        mb = (s[:, 0] >= b0) & (s[:, 0] < b0 + 5)
        if not mb.any():
            continue
        det_b = (s[mb, 4] > 0).sum()
        print(f"  t {b0:4.0f}-{b0+5:.0f}s: {mb.sum():4d} | {det_b:4d} | "
              f"{(s[mb,1]>0).sum():4d} | "
              f"{np.nanmedian(s[mb,2]):6.1f} | {np.median(s[mb,3])*100:7.1f}")
    print(f"\n=== VQ2 EKF self-consistency over {s[-1,0]:.1f}s ===")
    print(f"frames with fused corners: {100*upd.mean():.1f}%  "
          f"mean corners/update {s[upd,1].mean():.1f}")
    print(f"innovation rms: median {np.nanmedian(s[:,2]):.2f}px")
    print(f"filter sigma_p: median {np.median(s[:,3])*100:.1f}cm  "
          f"p90 {np.percentile(s[:,3],90)*100:.1f}cm")
    if edge_checks:
        ee = np.array([e[3] for e in edge_checks])
        print(f"map-edge checks (co-visible pairs): {len(ee)}  "
              f"median {np.median(ee):.1f}px  p75 {np.percentile(ee,75):.1f}px  "
              f"<40px: {100*(ee<40).mean():.0f}%")
        pairs = {}
        for (_t, a, b, e) in edge_checks:
            pairs.setdefault((a, b), []).append(e)
        for (a, b), es in sorted(pairs.items()):
            print(f"    gate {a:2d} -> {b:2d}: n={len(es):3d} "
                  f"median {np.median(es):6.1f}px")
    print(f"relocalizations: {len(relocs)}")
    for (tr, gi, ang, rms) in relocs[:40]:
        print(f"  t {tr:5.1f}s -> gate {gi:2d}  att-agree {ang:4.1f}deg  "
              f"pnp {rms:.2f}px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
