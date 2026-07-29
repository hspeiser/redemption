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
OBJ_APPARENT_INNER = np.array([
    [-HOLE, HOLE, 0.0],
    [HOLE, HOLE, 0.0],
    [HOLE, -HOLE, 0.0],
    [-HOLE, -HOLE, 0.0],
], np.float64)


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


def pnp_points_all(idx, imgp, K, obj8=None):
    """All refined IPPE branches for >=6 identified corners (indices into
    the 8-corner model): [(R_g2c, t, rms), ...] sorted by rms."""
    o8 = OBJ8 if obj8 is None else obj8
    if len(idx) < 6:
        return []
    obj_p = np.ascontiguousarray(o8[list(idx)] @ RX90.T)
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


def pnp_apparent_inner_all(imgp, K):
    """PnP branches for apparent TL,TR,BR,BL aperture corners."""
    image = np.ascontiguousarray(imgp, np.float64).reshape(-1, 1, 2)
    try:
        _count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            OBJ_APPARENT_INNER,
            image,
            K,
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return []
    solutions = []
    for rvec, tvec in zip(rvecs, tvecs):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                OBJ_APPARENT_INNER,
                image,
                K,
                None,
                rvec,
                tvec,
            )
        except cv2.error:
            continue
        translation = np.asarray(tvec, np.float64).reshape(3)
        if not np.isfinite(translation).all() or translation[2] <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            OBJ_APPARENT_INNER, rvec, tvec, K, None
        )
        residual = projected.reshape(4, 2) - image.reshape(4, 2)
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        solutions.append((translation, rms))
    return sorted(solutions, key=lambda solution: solution[1])


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
    ap.add_argument("--anchor-yaw", type=float, default=None,
                    help="override the solved anchor yaw (deg), e.g. the "
                         "pair-refined value from a previous pass")
    ap.add_argument("--classical", action="store_true",
                    help="use the classical detector instead of GateNet v6 "
                         "corners for the EKF pass")
    ap.add_argument("--ckpt", default=str(
        REPO / "data/models/gatenet_v6wsl_best.pt"))
    ap.add_argument(
        "--corner-refine-ckpt", default=None,
        help=(
            "optional second GateNet: keep primary checkpoint classes and "
            "availability, but snap accepted peaks to nearby second-model "
            "locations"
        ),
    )
    ap.add_argument(
        "--corner-refine-radius", type=float, default=2.0,
        help="maximum pixel distance for second-model peak snapping",
    )
    ap.add_argument(
        "--crop-gatenet-ckpt", default=None,
        help="optional instance-aware V11 crop-corner checkpoint",
    )
    ap.add_argument(
        "--crop-proposal-ckpt", default=None,
        help="YOLO gate proposal checkpoint required by crop V11",
    )
    ap.add_argument(
        "--crop-proposal-thresh", type=float, default=0.05,
        help="minimum YOLO proposal confidence for crop V11",
    )
    ap.add_argument(
        "--crop-proposal-padding", type=float, default=2.6,
        help="crop side divided by proposal-box side",
    )
    ap.add_argument(
        "--crop-v11-primary", action="store_true",
        help=(
            "use robust inner-corner V11 observations when all four "
            "survive map-prior gating; otherwise fall back to full-frame V7"
        ),
    )
    ap.add_argument(
        "--crop-v11-position-pins", action="store_true",
        help=(
            "keep the V7+V10 corner EKF and add rate-limited V11 aperture-PnP "
            "position measurements using gyro attitude"
        ),
    )
    ap.add_argument(
        "--crop-v11-pin-interval", type=float, default=0.25,
        help="minimum seconds between accepted V11 position measurements",
    )
    ap.add_argument(
        "--crop-v11-inference-interval", type=float, default=0.10,
        help=(
            "minimum seconds between V11 crop inference passes; the default "
            "runs the expensive fallback at 10 Hz while V7+V10 remains the "
            "per-frame tracker"
        ),
    )
    ap.add_argument("--dump-obs", default=None,
                    help="npz path: per-frame single-gate PnP observations "
                         "in gyro-chain world frame (for the joint SLAM "
                         "solve): t, active_gate, rel_world, yaw, rms")
    ap.add_argument("--dump-trace", default=None,
                    help="npz path: per-frame belief (t, frame path, pos, "
                         "quat_wxyz, sigma_p) for the map editor")
    ap.add_argument("--thresh", type=float, default=0.25,
                    help="net corner peak decode threshold")
    ap.add_argument("--label-dump", default=None,
                    help="npz path: temporal-transfer labels — while the "
                         "EKF is locked on an actively-fused gate, project "
                         "its corners every frame (incl. close/partial "
                         "views the geometric verifier cannot certify)")
    ap.add_argument("--dump-pairs", default=None,
                    help="npz path: dump co-visible pair measurements "
                         "(t, active_gate, dp_local, yawA, depths) for the "
                         "measured-map builder")
    ap.add_argument("--map-json", default=None,
                    help="use a measured local-frame map (from "
                         "vq2_build_map.py) instead of the anchored "
                         "gate_map.json")
    ap.add_argument("--no-map-reanchor", action="store_true",
                    help="load --map-json coordinates literally; by default "
                         "its gate-0 pose is rigidly aligned to this episode's "
                         "measured spawn gate")
    ap.add_argument("--write-corrected-map", default=None,
                    help="after the run, apply median locked-state per-gate "
                         "position corrections and write the map json here")
    ap.add_argument("--write-runtime-map", default=None,
                    help="write the episode-local map after spawn re-anchoring "
                         "(use this exact map when rendering the dumped trace)")
    ap.add_argument("--yaw-flip", action="store_true",
                    help="flip gate orientations 180 deg (front/back "
                         "ambiguity of the spawn PnP)")
    ap.add_argument("--decouple-yaw", action="store_true",
                    help="gate orientations use the gate0-observed offset "
                         "instead of the position anchor yaw")
    ap.add_argument("--pins-only", action="store_true",
                    help="disable corner updates; dense PnP pins + IMU "
                         "dead-reckoning between them measure the map's "
                         "gate-to-gate geometry -> anchor-yaw fit")
    ap.add_argument("--max-fuse-gate", type=int, default=None,
                    help="diagnostic: do not fuse or relocalize from gates "
                         "above this race index")
    ap.add_argument("--direct-position-pins", action="store_true",
                    help="use gate PnP for direct world position/velocity "
                         "pins with pure-gyro attitude; disables corner-EKF "
                         "attitude corrections and legacy relocalization")
    ap.add_argument("--gyro-attitude", action="store_true",
                    help="run normal V7 corner fusion but let vision update "
                         "translation/velocity only; attitude remains the "
                         "pure IMU gyro integration")
    args = ap.parse_args()
    if bool(args.crop_gatenet_ckpt) != bool(args.crop_proposal_ckpt):
        ap.error(
            "--crop-gatenet-ckpt and --crop-proposal-ckpt are required together"
        )
    if (
        args.crop_v11_primary or args.crop_v11_position_pins
    ) and not args.crop_gatenet_ckpt:
        ap.error("crop V11 runtime modes require crop V11 checkpoints")
    args.mirror_e = not args.no_mirror
    ep = Path(args.episode_dir)

    calib = load_calib(REPO / "data/calib/calib.json")
    fx, fy, cx, cy = calib["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R_cb = np.asarray(calib["R_cb"])

    imu = load_imu(ep)
    # recordings can span sim resets (clock jumps backward): clip to the
    # LONGEST monotonic segment — frames/race rows outside it are dropped
    brk = np.where(np.diff(imu[:, 0]) < -0.5)[0]
    if len(brk):
        segs = np.split(np.arange(len(imu)), brk + 1)
        seg = max(segs, key=len)
        print(f"clock segments: {len(segs)} -> keeping longest "
              f"({len(seg)}/{len(imu)} samples, "
              f"t {imu[seg[0],0]:.1f}..{imu[seg[-1],0]:.1f})")
        imu = imu[seg]
    frames, clock_off = load_frames(ep, imu)
    t_lo, t_hi = imu[0, 0] - 0.5, imu[-1, 0] + 0.5
    n_all = len(frames)
    frames = [(t, p) for (t, p) in frames if t_lo <= t <= t_hi]
    if len(frames) != n_all:
        print(f"frames clipped to clock segment: {len(frames)}/{n_all}")
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
    yaw_off_solved = a_yaw + (180.0 if args.yaw_flip else 0.0)
    if args.anchor_yaw is not None:
        print(f"anchor yaw OVERRIDE (positions): {a_yaw:.1f} -> "
              f"{args.anchor_yaw:.1f} deg (gate orientations keep "
              f"{yaw_off_solved:.1f})")
        a_yaw = args.anchor_yaw
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

    if args.map_json:
        gates = json.loads(Path(args.map_json).read_text())["gates"]
        print(f"measured map: {args.map_json} ({len(gates)} gates)")
        if not args.no_map_reanchor:
            # Measured maps live in the local frame of the episode that built
            # them. Every new recording defines yaw=0 independently from its
            # rest attitude, so loading those coordinates literally rotates
            # the whole course by the inter-episode spawn-yaw difference.
            # Align the map's gate-0 pose to the gate-0 pose measured above.
            p_map0 = np.asarray(gates[0]["pos"], float)
            # Use the spawn->gate0 bearing, not the planar PnP orientation.
            # The gate is square and its solved in-plane yaw can move several
            # degrees between recordings even when centre translation is
            # repeatable to centimetres. A 2-degree yaw mistake becomes
            # metres of lateral map error at the back of the course.
            yaw_map0 = float(np.degrees(
                np.arctan2(p_map0[1], p_map0[0])))
            yaw_obs0 = float(np.degrees(
                np.arctan2(p_med[1], p_med[0])))
            dyaw = (yaw_obs0 - yaw_map0 + 180.0) % 360.0 - 180.0
            R_anchor_map = Rotation.from_euler(
                "Z", dyaw, degrees=True).as_matrix()
            for gate in gates:
                gate["pos"] = (
                    p_med + R_anchor_map @ (
                        np.asarray(gate["pos"], float) - p_map0)).tolist()
                qg = gate["quat_wxyz"]
                Rg = Rotation.from_quat(
                    [qg[1], qg[2], qg[3], qg[0]]).as_matrix()
                qn = Rotation.from_matrix(
                    R_anchor_map @ Rg).as_quat()
                gate["quat_wxyz"] = [
                    float(qn[3]), float(qn[0]), float(qn[1]), float(qn[2])]
            print(f"  re-anchored gate0: translation "
                  f"{np.round(p_med - p_map0, 3)}m, yaw {dyaw:+.2f}deg")
    else:
        gates = load_vq2_map(anchor["anchor_t"], anchor["anchor_yaw_deg"],
                             mirror_e=args.mirror_e,
                             gate_yaw_offset_deg=(
                                 yaw_off_solved if args.decouple_yaw
                                 else None))
    if args.write_runtime_map:
        Path(args.write_runtime_map).write_text(json.dumps(
            {"source_map": args.map_json,
             "episode_dir": str(ep),
             "anchor": anchor,
             "gates": gates}, indent=2))
        print(f"runtime map -> {args.write_runtime_map}")
    gate_world = [np.concatenate(gate_quads_world_vq2(g)) for g in gates]
    outer_world_keys = {
        tuple(np.round(corner, 6))
        for gate_corners in gate_world
        for corner in gate_corners[4:8]
    }
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
    m_rs = (rs_t >= t_lo) & (rs_t <= t_hi)
    rs_t, rs_ag = rs_t[m_rs], rs_ag[m_rs]
    # UDP/MAV rows can arrive out of timestamp order around a simulator reset.
    # np.searchsorted below requires a sorted clock. Sort stably, discard any
    # stale pre-reset tail before the first armed gate-0 row, then enforce the
    # official monotonic race sequence. Late stale packets with a lower gate
    # index can no longer send association backward by several gates.
    order_rs = np.argsort(rs_t, kind="stable")
    rs_t, rs_ag = rs_t[order_rs], rs_ag[order_rs]
    # scrub the stale pre-reset tail at the head (ag=17 before race re-arms)
    k0 = int(np.argmax(rs_ag == 0)) if (rs_ag == 0).any() else 0
    rs_t, rs_ag = rs_t[k0:], rs_ag[k0:]
    rs_ag = np.maximum.accumulate(rs_ag)

    def active_gate(t):
        if len(rs_t) == 0 or t < rs_t[0]:
            return 0
        return int(rs_ag[min(np.searchsorted(rs_t, t, "right") - 1,
                             len(rs_ag) - 1)])

    # ---------- EKF over the flight (classical corners) ----------
    # corner source: GateNet v6 (sub-pixel, transfers to VQ2) unless
    # --classical. Classical corners on VQ2 neon gates are 5-15px noisy.
    net = None
    refine_net = None
    if not args.classical:
        import torch
        from aigp.vision.model import GateNet
        from scripts.train_net import orange_channel, decode_corners
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        net = GateNet().to(dev)
        net.load_state_dict(ck["model"])
        net.eval()
        print(f"GateNet corners: {Path(args.ckpt).name} "
              f"(epoch {ck['epoch']}) on {dev}")

        if args.corner_refine_ckpt:
            refine_ck = torch.load(
                args.corner_refine_ckpt, map_location=dev,
                weights_only=False,
            )
            refine_net = GateNet().to(dev)
            refine_net.load_state_dict(refine_ck["model"])
            refine_net.eval()
            print(
                f"corner refiner: {Path(args.corner_refine_ckpt).name} "
                f"(epoch {refine_ck['epoch']}, "
                f"radius {args.corner_refine_radius:.1f}px)"
            )

        def run_corner_net(model, tensor):
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.float16, enabled=dev == "cuda"):
                output = model(tensor)
            return decode_corners(
                output["hm"][0].float().cpu(),
                output["off"][0].float().cpu(),
                thresh=args.thresh,
            )

        def net_peaks(bgr):
            orange = orange_channel(bgr)
            x = np.concatenate([bgr.astype(np.float32) / 255.0,
                                orange[..., None]], 2).transpose(2, 0, 1)
            xt = torch.from_numpy(x).unsqueeze(0).to(dev)
            primary = run_corner_net(net, xt)
            if refine_net is None:
                return primary
            refined = run_corner_net(refine_net, xt)
            snapped = []
            radius = args.corner_refine_radius
            for corner_class, class_peaks in enumerate(primary):
                ring = range(0, 4) if corner_class < 4 else range(4, 8)
                candidates = [
                    point for refine_class in ring
                    for point in refined[refine_class]
                ]
                class_snapped = []
                for u, v, score in class_peaks:
                    nearest = min(
                        candidates,
                        key=lambda point: np.hypot(
                            point[0] - u, point[1] - v
                        ),
                        default=None,
                    )
                    if nearest is not None and np.hypot(
                            nearest[0] - u, nearest[1] - v) <= radius:
                        class_snapped.append(
                            (nearest[0], nearest[1], score)
                        )
                    else:
                        class_snapped.append((u, v, score))
                snapped.append(class_snapped)
            return snapped

    crop_net = None
    crop_proposal = None
    if args.crop_gatenet_ckpt:
        import torch
        from ultralytics import YOLO
        from aigp.vision.crop_gate import (
            CropGateNet,
            decode_crop_corners,
            orange_channel as crop_orange_channel,
            proposal_channel,
            warp_gate_crop,
        )

        crop_dev = "cuda" if torch.cuda.is_available() else "cpu"
        crop_ck = torch.load(
            args.crop_gatenet_ckpt,
            map_location=crop_dev,
            weights_only=False,
        )
        crop_net = CropGateNet().to(crop_dev)
        crop_net.load_state_dict(crop_ck["model"])
        crop_net.eval()
        crop_size = int(crop_ck.get("crop_size", 256))
        crop_prior = proposal_channel(crop_size)
        crop_proposal = YOLO(args.crop_proposal_ckpt)
        proposal_dev = crop_dev
        if crop_dev == "cuda":
            try:
                from torchvision.ops import nms

                nms(
                    torch.zeros((1, 4), device="cuda"),
                    torch.ones(1, device="cuda"),
                    0.5,
                )
            except (NotImplementedError, RuntimeError):
                proposal_dev = "cpu"
                print("crop proposal CUDA NMS unavailable; YOLO using CPU")
        print(
            f"crop V11: {Path(args.crop_gatenet_ckpt).name} "
            f"(epoch {crop_ck['epoch']}) + "
            f"{Path(args.crop_proposal_ckpt).name} on {crop_dev}"
        )

        d4_mappings = []
        d4_base = np.arange(4)
        for d4_shift in range(4):
            d4_mappings.append(
                tuple(int(value) for value in np.roll(d4_base, d4_shift))
            )
            d4_mappings.append(
                tuple(
                    int(value)
                    for value in np.roll(d4_base[::-1], d4_shift)
                )
            )
        d4_mappings = list(dict.fromkeys(d4_mappings))

        @torch.inference_mode()
        def crop_v11_observations(bgr, gate_indices):
            result = crop_proposal.predict(
                bgr,
                conf=args.crop_proposal_thresh,
                imgsz=640,
                verbose=False,
                device=proposal_dev,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                return [], [], {}, []
            boxes = result.boxes.xyxy.cpu().numpy()
            gate_predictions = {}
            assignments = []
            for priority, gate_index in enumerate(gate_indices):
                projected = []
                for physical_corner in range(4):
                    pixel, _camera_point = ekf.predict_pixel(
                        gate_world[gate_index][physical_corner]
                    )
                    if pixel is None:
                        projected = []
                        break
                    projected.append(pixel)
                if len(projected) != 4:
                    continue
                projected = np.asarray(projected, np.float32)
                expected_center = projected.mean(axis=0)
                expected_span = float(max(
                    np.ptp(projected[:, 0]),
                    np.ptp(projected[:, 1]),
                ))
                association_radius = max(30.0, 1.25 * expected_span)
                candidates = []
                for box_index, box in enumerate(boxes):
                    box_center = np.asarray([
                        (box[0] + box[2]) * 0.5,
                        (box[1] + box[3]) * 0.5,
                    ], np.float32)
                    distance = float(np.linalg.norm(
                        box_center - expected_center
                    ))
                    if distance <= association_radius:
                        candidates.append(
                            (distance, box_index, box, box_center)
                        )
                if candidates:
                    gate_predictions[gate_index] = (
                        projected,
                        expected_center,
                        expected_span,
                    )
                    best = min(candidates, key=lambda row: row[0])
                    assignments.append((priority, *best, gate_index))
            assignments.sort(key=lambda row: (row[0], row[1]))
            used_boxes = set()
            observations = []
            innovations = []
            matched = {}
            position_pin_candidates = []
            for (
                _priority,
                _proposal_distance,
                box_index,
                box,
                box_center,
                gate_index,
            ) in assignments:
                if box_index in used_boxes:
                    continue
                used_boxes.add(box_index)
                projected, expected_center, expected_span = (
                    gate_predictions[gate_index]
                )
                proposal_span = float(max(
                    box[2] - box[0], box[3] - box[1]
                ))
                side = max(
                    18.0,
                    proposal_span * args.crop_proposal_padding,
                )
                crop, _forward, inverse = warp_gate_crop(
                    bgr, box_center, side, crop_size
                )
                orange = crop_orange_channel(crop)
                network_input = np.concatenate([
                    crop.astype(np.float32) / 255.0,
                    orange[..., None],
                    crop_prior[..., None],
                ], axis=2).transpose(2, 0, 1)
                tensor = torch.from_numpy(network_input).unsqueeze(0).to(
                    crop_dev
                )
                with torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=crop_dev == "cuda",
                ):
                    output = crop_net(tensor)
                decoded = decode_crop_corners(
                    output,
                    crop_size,
                    inverse_affine=inverse,
                )
                measured = np.asarray(
                    decoded["corners"], np.float32
                )[:4]
                accepted = (
                    np.asarray(decoded["scores"])[:4] >= 0.05
                ) & (
                    np.asarray(decoded["visibility"])[:4] >= 0.20
                ) & (
                    float(decoded["presence"]) >= 0.20
                )
                center_radius = max(15.0, 0.35 * expected_span)
                if np.linalg.norm(
                    measured.mean(axis=0) - expected_center
                ) > center_radius:
                    continue
                corner_radius = max(16.0, 0.25 * expected_span)
                hypotheses = []
                for mapping in d4_mappings:
                    rows = []
                    cost = 0.0
                    for apparent_corner, physical_corner in enumerate(mapping):
                        if not accepted[apparent_corner]:
                            continue
                        distance = float(np.linalg.norm(
                            measured[apparent_corner]
                            - projected[physical_corner]
                        ))
                        if distance <= corner_radius:
                            rows.append((
                                gate_world[gate_index][physical_corner],
                                measured[apparent_corner],
                                distance,
                            ))
                            cost += distance
                    hypotheses.append((len(rows), cost, rows))
                count, cost, rows = min(
                    hypotheses,
                    key=lambda hypothesis: (
                        -hypothesis[0], hypothesis[1]
                    ),
                )
                if count < 2:
                    continue
                for world_corner, pixel, distance in rows:
                    observations.append((world_corner, pixel))
                    innovations.append(distance)
                matched[gate_index] = (count, cost)
                if count == 4 and accepted.all():
                    position_pin_candidates.append((
                        gate_index,
                        measured.copy(),
                        expected_span,
                        float(decoded["presence"]),
                    ))
            return (
                observations,
                innovations,
                matched,
                position_pin_candidates,
            )

    ekf = GateEKF(
        K,
        R_cb,
        sigma_px=1.5 if net is not None or crop_net is not None else 5.0,
    )
    q0 = R0.as_quat()
    ekf.init_state(np.zeros(3), np.zeros(3),
                   [q0[3], q0[0], q0[1], q0[2]], t_start)
    vw = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                         (W, H)) if args.video else None

    fi = 0
    # map-independent attitude reference for the pair measurement: pure
    # gyro integration from the rest attitude (sim gyro is noiseless)
    R_gyro = R0.as_matrix().copy()
    t_gyro = None
    stats = []   # (t, n_fused, innov_rms, sigma_p, n_clean)
    relocs = []  # (t, gate, att_angle_deg, pnp_rms, p_before, p_new)
    last_pin = None  # (t, gate, p_new) for pin-pair velocity fixes
    last_crop_position_pin = -np.inf
    last_crop_inference = -np.inf
    crop_position_rows = []
    edge_checks = []  # (t, pinned_gate, other_gate, pixel_err)
    pair_rows = []   # (t, active_gate, dp_local A->B)
    yaw_rows = []    # (gate_id, observed local yaw deg)
    assoc_errs = []  # uncapped nearest-gate pixel error per clean-det frame
    trace = []            # (t, path, p(3), q_wxyz(4), sigma_p)
    obs_rows = []         # (t, ag, rel_world(3), yaw_deg, rms, depth)
    lock_gate = None      # gate currently hard-locked (sigma small)
    edge_resid = {}       # (locked_gate, next_gate) -> list of 3D offsets
    flip_votes = {}       # gate -> list of bool (matched better 180-flipped)
    last_fuse = {}        # gate -> last time its corners were fused
    strong_fuse = {}      # gate -> last STRONG fuse (>=5 corners, tight sig)
    tl_rows = []          # temporal-transfer label rows
    pin_hist = {}         # gate -> recent (time, measured body position)
    direct_pin_rows = []  # (time, gate, jump, rms, depth)
    pending_pin = {}      # gate -> (last_time, last_position, count)
    pass_pin_rows = []    # (time, passed_gate, pre-reset error)
    gyro_translation_pins = []  # generic PnP fallback rows
    last_ag_pin = None
    last_ag_change_t = t_start
    last_pass_pin = None  # (time, pre-reset position error)
    last_pass_event = None  # (time, gate)
    last_upd = t_start
    prev_t = None
    for i in range(len(imu)):
        t_imu = imu[i, 0]
        if prev_t is not None and t_imu - prev_t > 1.5:
            print(f"[data hole at t={t_imu - t_start:.1f}s]", flush=True)
        prev_t = t_imu
        if t_gyro is not None and 0 < t_imu - t_gyro < 0.5:
            w_b = imu[i, 4:7] * np.asarray(GateEKF.GYRO_SIGN)
            R_gyro = R_gyro @ Rotation.from_rotvec(
                w_b * (t_imu - t_gyro)).as_matrix()
        t_gyro = t_imu
        ekf.propagate(t_imu, imu[i, 1:4], imu[i, 4:7])
        if args.direct_position_pins:
            ag_now = active_gate(t_imu)
            if last_ag_pin is not None and ag_now == last_ag_pin + 1:
                # The official event means the drone has just crossed
                # last_ag_pin. Its centre is therefore an authoritative
                # sub-metre position landmark even when vision was absent.
                p_gate_pass = np.asarray(gates[last_ag_pin]["pos"], float)
                err_pass = ekf.p - p_gate_pass
                if last_pass_pin is not None and \
                        np.linalg.norm(err_pass) < 6.0 and \
                        np.linalg.norm(last_pass_pin[1]) < 6.0:
                    dt_pass = t_imu - last_pass_pin[0]
                    if 0.3 < dt_pass < 8.0:
                        v_err = (err_pass - last_pass_pin[1]) / dt_pass
                        if np.linalg.norm(v_err) < 12.0:
                            ekf.v -= 0.8 * v_err
                # Prefer velocity measured from repeated PnP positions while
                # approaching the gate. Fall back to the authoritative
                # gate-to-gate displacement/time average.
                v_reset = None
                hp_pass = pin_hist.get(last_ag_pin, [])
                if len(hp_pass) >= 4:
                    thp = np.array([q0[0] for q0 in hp_pass])
                    php = np.array([q0[1] for q0 in hp_pass])
                    dthp = np.diff(thp)
                    mhp = (dthp > 0.01) & (dthp < 0.15)
                    if mhp.any():
                        vvp = np.diff(php, axis=0)[mhp] / dthp[mhp, None]
                        vvp = vvp[np.linalg.norm(vvp, axis=1) < 25.0]
                        if len(vvp) >= 2:
                            v_reset = np.median(vvp, axis=0)
                if v_reset is None and last_pass_event is not None:
                    dt_leg = t_imu - last_pass_event[0]
                    if 0.3 < dt_leg < 8.0:
                        p_prev = np.asarray(
                            gates[last_pass_event[1]]["pos"], float)
                        v_reset = (p_gate_pass - p_prev) / dt_leg
                if v_reset is not None and np.linalg.norm(v_reset) < 25.0:
                    ekf.v = v_reset
                elif np.linalg.norm(ekf.v) > 18.0:
                    ekf.v *= 18.0 / np.linalg.norm(ekf.v)
                last_pass_pin = (t_imu, err_pass.copy())
                last_pass_event = (t_imu, last_ag_pin)
                ekf.p = p_gate_pass
                ekf.P[0:3, :] = 0.0
                ekf.P[:, 0:3] = 0.0
                ekf.P[0:3, 0:3] = np.eye(3) * 0.45**2
                ekf.P[3:6, 3:6] += np.eye(3) * 0.60**2
                pass_pin_rows.append(
                    (t_imu - t_start, last_ag_pin,
                     float(np.linalg.norm(err_pass))))
                last_ag_change_t = t_imu
            if last_ag_pin is None or ag_now >= last_ag_pin or ag_now == 0:
                last_ag_pin = ag_now
        while fi < len(frames) and frames[fi][0] <= t_imu:
            ts, path = frames[fi]
            fi += 1
            img = cv2.imread(str(path))
            if img is None:
                continue
            if args.direct_position_pins:
                # Keep attitude map-independent. A tilted or mis-oriented
                # gate must never rotate gravity into the translation state.
                ekf.q = Rotation.from_matrix(R_gyro)
            dets = detect_gates(img, min_area=250)
            obs = []
            innovs = []
            n_clean = 0
            clean_dets = []
            peaks = None
            matched_g = {}
            crop_pin_candidates = []
            sig_p = float(np.sqrt(max(np.trace(ekf.P[0:3, 0:3]), 0)))
            if net is not None and not args.pins_only:
                peaks = net_peaks(img)
                if any(len(peaks[c]) for c in range(8)):
                    n_clean = 1
                ag_n = active_gate(t_imu)
                rad_n = float(np.clip(3 * K[0, 0] * sig_p / 6.0 + 20,
                                      25, 120))
                FLIP = (1, 0, 3, 2, 5, 4, 7, 6)   # gate rotated 180 deg
                for gi in [
                        g for g in (ag_n - 1, ag_n, ag_n + 1)
                        if 0 <= g < min(17, len(gates))
                        and (args.max_fuse_gate is None
                             or g <= args.max_fuse_gate)]:
                    # some map gates are 180-flipped (spline heading vs
                    # facing): match under both orientations, keep better
                    cand_obs = {False: [], True: []}
                    cand_cost = {False: 0.0, True: 0.0}
                    for flip in (False, True):
                        for k in range(8):
                            Xw = gate_world[gi][FLIP[k] if flip else k]
                            uvp, Xc = ekf.predict_pixel(Xw)
                            if uvp is None:
                                continue
                            best_pk = None
                            for (u, v, s) in peaks[k]:
                                d = float(np.hypot(u - uvp[0], v - uvp[1]))
                                if d < rad_n and (best_pk is None
                                                  or d < best_pk[0]):
                                    best_pk = (d, u, v)
                            if best_pk is not None:
                                cand_obs[flip].append(
                                    (Xw, np.array(best_pk[1:]),
                                     best_pk[0]))
                                cand_cost[flip] += best_pk[0]
                    pick = False
                    if len(cand_obs[True]) > len(cand_obs[False]) or (
                            len(cand_obs[True]) == len(cand_obs[False])
                            and cand_cost[True] < cand_cost[False]):
                        pick = True
                    for (Xw, uv0, d0) in cand_obs[pick]:
                        obs.append((Xw, uv0))
                        innovs.append(d0)
                    if cand_obs[pick]:
                        flip_votes.setdefault(gi, []).append(pick)
                        matched_g[gi] = (len(cand_obs[pick]),
                                         cand_cost[pick])
            if (
                crop_net is not None
                and not args.pins_only
                and t_imu - last_crop_inference
                >= args.crop_v11_inference_interval
            ):
                last_crop_inference = t_imu
                active_crop_gate = active_gate(t_imu)
                crop_gate_indices = [
                    gate_index
                    for gate_index in (
                        active_crop_gate,
                        active_crop_gate - 1,
                        active_crop_gate + 1,
                    )
                    if 0 <= gate_index < min(17, len(gates))
                    and (
                        args.max_fuse_gate is None
                        or gate_index <= args.max_fuse_gate
                    )
                ]
                (
                    crop_obs,
                    crop_innovs,
                    crop_matched,
                    crop_pin_candidates,
                ) = (
                    crop_v11_observations(img, crop_gate_indices)
                )
                if crop_matched:
                    n_clean = max(n_clean, 1)
                    matched_g.update(crop_matched)
                if args.crop_v11_primary and len(crop_obs) >= 4:
                    # V11 owns the grouped inner aperture. Preserve only V7's
                    # outer-ring constraints so the EKF keeps the wider
                    # geometric baseline without duplicating/conflicting
                    # inner measurements.
                    v7_outer = [
                        (row, innovation)
                        for row, innovation in zip(obs, innovs)
                        if tuple(np.round(row[0], 6)) in outer_world_keys
                    ]
                    obs = crop_obs + [row for row, _ in v7_outer]
                    innovs = crop_innovs + [
                        innovation for _, innovation in v7_outer
                    ]
                elif (
                    not args.crop_v11_primary
                    and not args.crop_v11_position_pins
                ):
                    obs.extend(crop_obs)
                    innovs.extend(crop_innovs)
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
                if net is not None:
                    continue      # net supplies the update corners; the
                                  # classical det is kept for reloc only
                dc = det["outer"].mean(0)
                # nearest predicted gate among race-status candidates
                ag = active_gate(t_imu)
                cand = [
                    g for g in (ag - 1, ag, ag + 1)
                    if 0 <= g < min(17, len(gates))
                    and (args.max_fuse_gate is None
                         or g <= args.max_fuse_gate)
                ]
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
            # global map-fit metric: uncapped distance from the biggest clean
            # detection to the nearest predicted gate centre (any gate)
            if clean_dets:
                dbig = max(clean_dets, key=lambda d0: -0 + d0["area"])
                dcb = dbig["outer"].mean(0)
                emin = None
                for gi2 in range(len(gates)):
                    uvp2, _ = ekf.predict_pixel(np.asarray(gates[gi2]["pos"]))
                    if uvp2 is None:
                        continue
                    e2 = float(np.hypot(uvp2[0] - dcb[0], uvp2[1] - dcb[1]))
                    if emin is None or e2 < emin:
                        emin = e2
                if emin is not None:
                    assoc_errs.append(emin)

            # next-gate first-sight residual: while hard-locked (small sigma,
            # belief pinned to the active gate), PnP any OTHER clean det ->
            # measured world position vs map = per-edge map correction
            if net is not None and peaks is not None and sig_p < 0.35 \
                    and clean_dets:
                agl = active_gate(t_imu)
                R_wc_l = ekf.q.as_matrix() @ R_cb.T
                for dd in clean_dets:
                    x0, y0 = dd["outer"].min(0) - 12
                    x1, y1 = dd["outer"].max(0) + 12
                    idxs, uvs = [], []
                    for c in range(8):
                        inb = [(u, v, sc) for (u, v, sc) in peaks[c]
                               if x0 <= u <= x1 and y0 <= v <= y1]
                        if inb:
                            u, v, _ = max(inb, key=lambda p0: p0[2])
                            idxs.append(c)
                            uvs.append([u, v])
                    if len(idxs) < 6:
                        continue
                    br = pnp_points_all(idxs, uvs, K)
                    if not br or br[0][2] > 1.0:
                        continue
                    p_meas = ekf.p + R_wc_l @ br[0][1]
                    ds = [np.linalg.norm(
                        p_meas - np.asarray(gates[gj]["pos"]))
                        for gj in range(len(gates))]
                    gj = int(np.argmin(ds))
                    if ds[gj] < 10.0:
                        edge_resid.setdefault((agl, gj), []).append(
                            p_meas - np.asarray(gates[gj]["pos"]))

            # SLAM observation dump: PnP of the BIGGEST clean det (assumed
            # = race-status active gate), rotated to the gyro-chain world
            # frame. Map-independent by construction.
            if args.dump_obs and peaks is not None and clean_dets:
                dd0 = max(clean_dets, key=lambda d0: d0["area"])
                x0, y0 = dd0["outer"].min(0) - 12
                x1, y1 = dd0["outer"].max(0) + 12
                idxs0, uvs0 = [], []
                for c in range(8):
                    inb = [(u, v, sc) for (u, v, sc) in peaks[c]
                           if x0 <= u <= x1 and y0 <= v <= y1]
                    if inb:
                        u, v, _ = max(inb, key=lambda q0: q0[2])
                        idxs0.append(c)
                        uvs0.append([u, v])
                if len(idxs0) >= 6:
                    br0 = pnp_points_all(idxs0, uvs0, K)
                    if br0 and br0[0][2] < 1.0:
                        R_o, t_o, rms_o = br0[0]
                        R_wc_o = R_gyro @ R_cb.T
                        relw = R_wc_o @ t_o
                        R_go = R_wc_o @ R_o
                        obs_rows.append((
                            t_imu - t_start, active_gate(t_imu),
                            float(relw[0]), float(relw[1]), float(relw[2]),
                            float(np.degrees(np.arctan2(R_go[1, 0],
                                                        R_go[0, 0]))),
                            rms_o, float(np.linalg.norm(t_o))))

            # convention-neutral pair measurement: PnP two co-visible gates
            # independently -> gate->gate vector in the local frame (drone
            # pose cancels; only belief ATTITUDE is used, which is gyro-
            # driven and map-independent). Branch picked by rms ratio test.
            if len(clean_dets) >= 2:
                dsort2 = sorted(clean_dets, key=lambda d0: -d0["area"])[:2]
                sols2 = []
                for dd in dsort2:
                    if net is not None and peaks is not None:
                        # sub-pixel: strongest net peak per class inside the
                        # classical det's bbox (spatial grouping by the det)
                        x0, y0 = dd["outer"].min(0) - 12
                        x1, y1 = dd["outer"].max(0) + 12
                        idxs, uvs = [], []
                        for c in range(8):
                            inb = [(u, v, sc) for (u, v, sc) in peaks[c]
                                   if x0 <= u <= x1 and y0 <= v <= y1]
                            if inb:
                                u, v, _ = max(inb, key=lambda p0: p0[2])
                                idxs.append(c)
                                uvs.append([u, v])
                        br = pnp_points_all(idxs, uvs, K) \
                            if len(idxs) >= 6 else []
                        sols2.append(br[0] if br and br[0][2] < 1.0
                                     else None)
                    else:
                        br = [s for s in pnp_gate_all(dd, K) if s[2] < 3.0]
                        if not br or (len(br) > 1
                                      and br[1][2] < 1.2 * br[0][2]):
                            sols2.append(None)
                        else:
                            sols2.append(br[0])
                if all(s is not None for s in sols2):
                    R_wc_b = R_gyro @ R_cb.T
                    pA = R_wc_b @ sols2[0][1]
                    pB = R_wc_b @ sols2[1][1]
                    RA = R_wc_b @ sols2[0][0]
                    RB = R_wc_b @ sols2[1][0]
                    ag2 = active_gate(t_imu)
                    pair_rows.append((
                        t_imu - t_start, ag2, pB - pA,
                        float(np.degrees(np.arctan2(RA[1, 0], RA[0, 0]))),
                        float(np.degrees(np.arctan2(RB[1, 0], RB[0, 0]))),
                        float(np.linalg.norm(sols2[0][1])),
                        float(np.linalg.norm(sols2[1][1]))))
                # per-gate yaw observation for the biggest det (identity =
                # active gate), same ratio test
                if sols2[0] is not None:
                    R_g0_obs = (R_gyro @ R_cb.T) @ sols2[0][0]
                    yaw_rows.append((active_gate(t_imu), np.degrees(
                        np.arctan2(R_g0_obs[1, 0], R_g0_obs[0, 0]))))

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

            # Direct gate-relative position/velocity correction. PnP
            # translation plus pure-gyro camera attitude gives body position
            # without using the gate's map orientation:
            #   p_body = p_gate - R_world_camera @ t_camera_gate
            if args.direct_position_pins and peaks is not None and clean_dets \
                    and t_imu - last_ag_change_t > 0.30:
                agp = active_gate(t_imu)
                # A low-RMS PnP fit says "this is a gate", not WHICH gate.
                # The old largest-box rule repeatedly mistook the gate just
                # crossed for the newly-active gate. Evaluate EVERY spatially
                # separated gate detection and keep the active-gate solution
                # whose implied camera position agrees with the pass-anchored
                # inertial prediction. A previous/future gate is displaced by
                # an entire map edge and therefore fails the 3.5 m gate.
                sols_p = []
                R_wc_pin = R_gyro @ R_cb.T
                uv_expect = None
                if 0 <= agp < min(17, len(gates)):
                    uv_expect, _xc_expect = ekf.predict_pixel(
                        np.asarray(gates[agp]["pos"], float))
                for ddp in clean_dets:
                    # The classical detector has already passed concentricity,
                    # area-ratio and inner/outer checks. Its complete 8-corner
                    # geometry is far more available than demanding six
                    # correctly classed neural peaks (which yielded no pins
                    # at all for gates 8-16).
                    branches_p = pnp_gate_all(ddp, K)
                    for _Rp, tcp, rmsp in branches_p:
                        if rmsp >= 3.0 or not (
                                0 <= agp < min(17, len(gates))):
                            continue
                        gp = agp
                        pp = np.asarray(gates[gp]["pos"]) - R_wc_pin @ tcp
                        jp = float(np.linalg.norm(pp - ekf.p))
                        pixp = 0.0 if uv_expect is None else float(
                            np.linalg.norm(ddp["outer"].mean(0) - uv_expect))
                        # Position continuity is authoritative. Pixel bearing
                        # only breaks ties between otherwise plausible gates.
                        costp = jp + 0.001 * min(pixp, 500.0)
                        depthp = float(np.linalg.norm(tcp))
                        sols_p.append(
                            (costp, jp, gp, pp, rmsp, depthp))
                bestp = min(sols_p, key=lambda q0: q0[0]) \
                    if sols_p else None
                if bestp is not None:
                    # Never "recover" from a large jump merely because the
                    # same wrong gate is seen for several frames. Pass pins
                    # bound each dead-reckoning leg, so skipping an uncertain
                    # visual update is much safer than accepting a leg-sized
                    # translation.
                    accept_p = bestp[1] < 3.5
                    if accept_p:
                        _costp, jumpp, gp, ppin, rmsp, depthp = bestp
                        pending_pin.pop(gp, None)
                        hp = pin_hist.setdefault(gp, [])
                        hp.append((ts, ppin.copy()))
                        hp[:] = [(tp, pp) for tp, pp in hp
                                 if ts - tp <= 0.45]

                        # Settle to the landmark in a few frames while
                        # suppressing single-frame planar-depth jitter.
                        ekf.p = 0.20 * ekf.p + 0.80 * ppin

                        # Fit velocity directly from recent world-position
                        # pins. The old filter corrected position but carried
                        # a 1-3 m/s velocity error into every later gate.
                        if len(hp) >= 4 and hp[-1][0] - hp[0][0] >= 0.08:
                            th = np.array([q0[0] for q0 in hp])
                            ph = np.array([q0[1] for q0 in hp])
                            dth = np.diff(th)
                            good_dt = (dth > 0.01) & (dth < 0.15)
                            if good_dt.any():
                                vv = np.diff(ph, axis=0)[good_dt] / \
                                    dth[good_dt, None]
                                vv = vv[np.linalg.norm(vv, axis=1) < 25.0]
                                if len(vv) >= 2:
                                    vpin = np.median(vv, axis=0)
                                    ekf.v = 0.55 * ekf.v + 0.45 * vpin
                                    if np.linalg.norm(ekf.v) > 22.0:
                                        ekf.v *= 22.0 / np.linalg.norm(ekf.v)

                        # Do not report centimetre certainty from a map whose
                        # individual landmarks still have dm-m uncertainty.
                        ekf.P[0:3, :] = 0.0
                        ekf.P[:, 0:3] = 0.0
                        ekf.P[0:3, 0:3] = np.eye(3) * 0.20**2
                        ekf.P[3:6, 3:6] += np.eye(3) * 0.35**2
                        direct_pin_rows.append(
                            (t_imu - t_start, gp, jumpp, rmsp, depthp))
                        last_upd = t_imu

            if args.dump_trace:
                q_tr = ekf.q.as_quat()
                trace.append((t_imu - t_start, str(path),
                              ekf.p.copy(),
                              np.array([q_tr[3], q_tr[0], q_tr[1],
                                        q_tr[2]]), sig_p))
            if args.pins_only:
                obs = []
            if args.direct_position_pins:
                # Translation and velocity came from the explicit pin above.
                # Never let planar gate geometry modify gyro attitude.
                obs = []
            n = ekf.update_corners(
                obs, update_attitude=not args.gyro_attitude)
            if n:
                last_upd = t_imu
                for gi_f, (nm_f, _c_f) in matched_g.items():
                    last_fuse[gi_f] = t_imu
                    if nm_f >= 5 and sig_p < 0.06:
                        strong_fuse[gi_f] = t_imu

            if (
                args.crop_v11_position_pins
                and not n
                and crop_pin_candidates
                and t_imu - last_crop_position_pin
                >= args.crop_v11_pin_interval
            ):
                R_wc_crop = ekf.q.as_matrix() @ R_cb.T
                solved_crop_pins = []
                for (
                    gate_index,
                    apparent_inner,
                    _expected_span,
                    presence_probability,
                ) in crop_pin_candidates:
                    for translation_cg, pnp_rms in pnp_apparent_inner_all(
                        apparent_inner, K
                    ):
                        if pnp_rms > 3.0:
                            continue
                        position_measurement = np.asarray(
                            gates[gate_index]["pos"], float
                        ) - R_wc_crop @ translation_cg
                        jump = float(np.linalg.norm(
                            position_measurement - ekf.p
                        ))
                        solved_crop_pins.append((
                            jump,
                            pnp_rms,
                            -presence_probability,
                            gate_index,
                            position_measurement,
                            float(np.linalg.norm(translation_cg)),
                        ))
                if solved_crop_pins:
                    (
                        jump,
                        pnp_rms,
                        _negative_presence,
                        gate_index,
                        position_measurement,
                        depth,
                    ) = min(
                        solved_crop_pins,
                        key=lambda row: (row[0], row[1], row[2]),
                    )
                    current_sigma = float(np.sqrt(max(
                        np.trace(ekf.P[0:3, 0:3]), 0.0
                    )))
                    maximum_jump = max(
                        0.60,
                        min(1.50, 0.35 + 2.5 * current_sigma),
                    )
                    if jump <= maximum_jump:
                        measurement_sigma = float(np.clip(
                            0.12 + 0.020 * depth, 0.15, 0.45
                        ))
                        measurement_matrix = np.zeros((3, 9), float)
                        measurement_matrix[:, 0:3] = np.eye(3)
                        measurement_noise = (
                            np.eye(3) * measurement_sigma**2
                        )
                        innovation_covariance = (
                            measurement_matrix
                            @ ekf.P
                            @ measurement_matrix.T
                            + measurement_noise
                        )
                        kalman_gain = (
                            ekf.P
                            @ measurement_matrix.T
                            @ np.linalg.inv(innovation_covariance)
                        )
                        if args.gyro_attitude:
                            kalman_gain[6:9, :] = 0.0
                        correction = kalman_gain @ (
                            position_measurement - ekf.p
                        )
                        ekf.p += correction[0:3]
                        ekf.v += correction[3:6]
                        identity = np.eye(9)
                        residual_matrix = (
                            identity - kalman_gain @ measurement_matrix
                        )
                        ekf.P = (
                            residual_matrix
                            @ ekf.P
                            @ residual_matrix.T
                            + kalman_gain
                            @ measurement_noise
                            @ kalman_gain.T
                        )
                        ekf.P = 0.5 * (ekf.P + ekf.P.T)
                        last_crop_position_pin = t_imu
                        last_upd = t_imu
                        n = max(n, 4)
                        crop_position_rows.append((
                            t_imu - t_start,
                            gate_index,
                            jump,
                            pnp_rms,
                            measurement_sigma,
                            depth,
                        ))

            # Generic recovery for fast/blurred approaches: when classified
            # V7 corners fail the batch gate, use a complete clean gate's PnP
            # translation while retaining pure gyro attitude. Race status
            # supplies identity; a previous/future visible gate implies an
            # inter-gate-sized camera jump and fails continuity.
            if args.gyro_attitude and not n and clean_dets and \
                    t_imu - last_upd > 0.15:
                ag_pin = active_gate(t_imu)
                if 0 <= ag_pin < min(17, len(gates)):
                    R_wc_pin = ekf.q.as_matrix() @ R_cb.T
                    pin_candidates = []
                    for det_pin in clean_dets:
                        for _Rgc, t_cg, rms_pin in pnp_gate_all(det_pin, K):
                            if rms_pin > 3.0:
                                continue
                            p_pin = np.asarray(
                                gates[ag_pin]["pos"]) - R_wc_pin @ t_cg
                            jump_pin = float(np.linalg.norm(p_pin - ekf.p))
                            pin_candidates.append(
                                (jump_pin, rms_pin, p_pin))
                    if pin_candidates:
                        jump_pin, rms_pin, p_pin = min(
                            pin_candidates, key=lambda row: row[0])
                        gap_pin = t_imu - last_upd
                        jmax_pin = min(4.0, 1.5 + 0.8 * gap_pin)
                        if jump_pin < jmax_pin:
                            q_pin = ekf.q.as_quat()
                            v_pin = ekf.v.copy()
                            if last_pin is not None and \
                                    last_pin[1] == ag_pin and \
                                    0.06 < t_imu - last_pin[0] < 0.7:
                                v_meas = (p_pin - last_pin[2]) / (
                                    t_imu - last_pin[0])
                                if np.linalg.norm(v_meas) < 25.0:
                                    v_pin = 0.7 * v_pin + 0.3 * v_meas
                            last_pin = (
                                t_imu, ag_pin, p_pin.copy())
                            ekf.init_state(
                                0.2 * ekf.p + 0.8 * p_pin, v_pin,
                                [q_pin[3], q_pin[0], q_pin[1], q_pin[2]],
                                t_imu, pos_std=0.25, vel_std=0.7,
                                ang_std=0.02)
                            gyro_translation_pins.append(
                                (t_imu - t_start, ag_pin,
                                 jump_pin, rms_pin))
                            last_upd = t_imu

            # temporal-transfer labels: gates actively fused moments ago
            # keep exact relative pose through the approach/pass — label
            # their projected corners even when few are visible
            if args.label_dump and sig_p < 0.15 and peaks is not None:
                FLIP_L = (1, 0, 3, 2, 5, 4, 7, 6)
                gates_lab = []
                for gi_l, t_f in last_fuse.items():
                    t_strong = strong_fuse.get(gi_l, -1e9)
                    if t_imu - t_f > 0.35 and t_imu - t_strong > 0.8:
                        continue
                    flip_l = bool(flip_votes.get(gi_l)) and \
                        np.mean(flip_votes[gi_l]) > 0.5
                    uv8 = np.full((8, 2), np.nan, np.float32)
                    vis8 = np.zeros(8, bool)
                    n_in = 0
                    n_snap = 0
                    for k in range(8):
                        Xw = gate_world[gi_l][FLIP_L[k] if flip_l else k]
                        uvp, Xc0 = ekf.predict_pixel(Xw)
                        if uvp is None or Xc0[2] < 0.25:
                            continue
                        # snap to a same-class net peak when one is close
                        best_s = None
                        for (u, v, s0) in peaks[k]:
                            d0 = float(np.hypot(u - uvp[0], v - uvp[1]))
                            if d0 < 16 and (best_s is None
                                            or d0 < best_s[0]):
                                best_s = (d0, u, v)
                        uv = np.array(best_s[1:]) if best_s else uvp
                        if best_s is not None:
                            n_snap += 1
                        uv8[k] = uv
                        inb = (-40 <= uv[0] < W + 40 and
                               -30 <= uv[1] < H + 30)
                        vis8[k] = inb
                        n_in += int(inb)
                    # this-frame corroboration: the projection must agree
                    # with live net evidence, or it does not become a label
                    # (post-fuse drift was producing floating labels)
                    # a STRONG recent lock earns a corroboration-free window
                    # (pure IMU drift over 0.8s is cm-scale) — this is what
                    # labels the pass-through frames the teacher net cannot
                    ok_lab = n_in >= 2 and (
                        n_snap >= 2 or
                        (n_snap >= 1 and t_imu - t_f < 0.12) or
                        t_imu - t_strong < 0.8)
                    if ok_lab:
                        gates_lab.append((gi_l, uv8, vis8))
                if gates_lab:
                    igs = []
                    for dd in clean_dets:
                        dc0 = dd["outer"].mean(0)
                        near = any(
                            np.isfinite(uv8).all(axis=1).any() and
                            np.nanmin(np.linalg.norm(
                                uv8 - dc0, axis=1)) < 60
                            for (_g, uv8, _v) in gates_lab)
                        if not near:
                            x0, y0 = dd["outer"].min(0) - 12
                            x1, y1 = dd["outer"].max(0) + 12
                            igs.append([max(x0, 0), max(y0, 0),
                                        min(x1, W), min(y1, H)])
                    tl_rows.append((str(path), gates_lab[:3], igs))
            if (not args.direct_position_pins) and (not n) and \
                    (clean_dets or n_clean) and (t_imu - last_upd) > (
                    0.3 if args.pins_only else 0.5):
                # lost -> PnP-relocalize against the race-status active gate;
                # branch/back-face disambiguated by attitude agreement (the
                # gyro is noiseless, so belief attitude is trustworthy).
                # Prefer a classical detection (spatially segmented, so no
                # cross-gate corner mixing); fall back to strongest net peaks.
                if clean_dets:
                    det = max(clean_dets, key=lambda d0: d0["area"])
                    branches = pnp_gate_all(det, K)
                elif net is not None:
                    cd = {c: max(peaks[c], key=lambda p0: p0[2])[:2]
                          for c in range(8) if peaks[c]}
                    branches = pnp_points_all(
                        sorted(cd), [cd[k] for k in sorted(cd)], K) \
                        if len(cd) >= 6 else []
                else:
                    branches = []
                ag = active_gate(t_imu)
                Rb = ekf.q.as_matrix()
                cands_r = []
                for gi in [
                        g for g in (ag, ag + 1, ag - 1)
                        if 0 <= g < min(17, len(gates))
                        and (args.max_fuse_gate is None
                             or g <= args.max_fuse_gate)]:
                    for (R_g2c, t_pnp, rms) in branches:
                        if rms > 3.0:
                            continue
                        R_wb = gate_R[gi] @ R_g2c.T @ R_cb
                        cosang = (np.trace(R_wb.T @ Rb) - 1) / 2
                        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                        if ang > 15:
                            continue
                        if args.gyro_attitude:
                            # Translation from PnP does not require trusting
                            # the square gate's ambiguous map orientation.
                            R_wc = Rb @ R_cb.T
                            p_new = np.asarray(
                                gates[gi]["pos"]) - R_wc @ t_pnp
                        else:
                            p_new = np.asarray(gates[gi]["pos"]) - \
                                gate_R[gi] @ (R_g2c.T @ t_pnp)
                        jump = float(np.linalg.norm(p_new - ekf.p))
                        cands_r.append((p_new, ang, gi, rms, jump))
                # gate identity by position-jump plausibility: dead-reckon
                # drift over the vision gap is cm-scale, wrong-gate
                # candidates jump the inter-gate distance (metres)
                gap = t_imu - last_upd
                jmax = 1.0 + 0.6 * gap
                ok_j = [c for c in cands_r if c[4] < jmax]
                pool = ok_j if ok_j else (cands_r if gap > 3.0 else [])
                best = min(pool, key=lambda c: c[1]) if pool else None
                if best is not None:
                    q = ekf.q.as_quat()
                    p_before = ekf.p.copy()
                    # same-gate pin pair within 0.7s -> direct velocity fix
                    v_new = ekf.v
                    if last_pin is not None and last_pin[1] == best[2] and \
                            0.05 < t_imu - last_pin[0] < 0.7:
                        v_new = (best[0] - last_pin[2]) / (t_imu - last_pin[0])
                    last_pin = (t_imu, best[2], best[0].copy())
                    ekf.init_state(best[0], v_new,
                                   [q[3], q[0], q[1], q[2]], t_imu,
                                   pos_std=0.4, vel_std=0.8, ang_std=0.03)
                    relocs.append((t_imu - t_start, best[2], best[1],
                                   best[3], p_before, best[0].copy()))
                    last_upd = t_imu
            stats.append((t_imu - t_start, n,
                          float(np.sqrt(np.mean(np.square(innovs)))) if innovs
                          else np.nan, sig_p, n_clean))
            if vw is not None:
                vis = img.copy()
                if peaks is not None:      # raw net corner detections
                    for c in range(8):
                        col = (0, 255, 0) if c < 4 else (0, 255, 255)
                        for (u, v, s) in peaks[c]:
                            cv2.circle(vis, (int(u), int(v)), 3, col, -1)
                # MEASURED wireframes only: per detection region, PnP the
                # net corners against the rigid gate; draw the reprojected
                # quad only if the fit is tight. Nothing drawn from belief
                # or map -> no floating/ghost gates possible.
                if peaks is not None:
                    boxes_v = []
                    for dd in dets:
                        bx0, by0 = dd["outer"].min(0) - 14
                        bx1, by1 = dd["outer"].max(0) + 14
                        boxes_v.append((bx0, by0, bx1, by1))
                    # net-native proposals: hole-diagonal peak pairs imply a
                    # gate-sized box (classical bboxes miss CLOSE gates -
                    # bloom/partial - so without these, near gates never
                    # even get a lock attempt)
                    for (ca, cb) in ((0, 2), (1, 3)):
                        for (ua, va, sa) in sorted(
                                peaks[ca], key=lambda q: -q[2])[:3]:
                            for (ub, vb, sb) in sorted(
                                    peaks[cb], key=lambda q: -q[2])[:3]:
                                span = max(abs(ub - ua), abs(vb - va))
                                if span < 24:
                                    continue
                                cx0 = (ua + ub) / 2
                                cy0 = (va + vb) / 2
                                half = span * 1.15   # panel ~1.8x hole
                                boxes_v.append((cx0 - half, cy0 - half,
                                                cx0 + half, cy0 + half))
                    if not boxes_v:
                        boxes_v.append((-1e9, -1e9, 1e9, 1e9))
                    drawn_c = []
                    for (bx0, by0, bx1, by1) in boxes_v[:14]:
                        idxs, uvs = [], []
                        for c in range(8):
                            inb = [(u, v, sc) for (u, v, sc) in peaks[c]
                                   if bx0 <= u <= bx1 and by0 <= v <= by1]
                            if inb:
                                u, v, _ = max(inb, key=lambda q: q[2])
                                idxs.append(c)
                                uvs.append([u, v])
                        if len(idxs) < 6:
                            continue
                        cc0 = np.mean(uvs, axis=0)
                        if any(np.hypot(*(cc0 - d0)) < 30 for d0 in drawn_c):
                            continue
                        br = pnp_points_all(idxs, uvs, K)
                        if not br or br[0][2] > 1.5:
                            continue
                        drawn_c.append(cc0)
                        R_v, t_v, rms_v = br[0]
                        obj_v = np.ascontiguousarray(OBJ8 @ RX90.T)
                        rvec_v, _ = cv2.Rodrigues(R_v @ RX90.T)
                        pr8, _ = cv2.projectPoints(
                            obj_v, rvec_v, t_v.reshape(3, 1), K, None)
                        pr8 = pr8.reshape(8, 2)
                        good = len(idxs) >= 7 and rms_v < 0.8
                        col = (0, 255, 0) if good else (0, 200, 255)
                        for quad in (pr8[0:4], pr8[4:8]):
                            cv2.polylines(vis, [quad.astype(np.int32)],
                                          True, col, 2)
                        dist = float(np.linalg.norm(t_v))
                        bear = float(np.degrees(
                            np.arctan2(t_v[0], t_v[2])))
                        top = pr8[np.argmin(pr8[:, 1])]
                        cv2.putText(
                            vis, f"{dist:4.1f}m {bear:+3.0f}deg "
                            f"{len(idxs)}/8 {rms_v:.1f}px",
                            (int(np.clip(top[0] - 55, 4, W - 175)),
                             int(np.clip(top[1] - 8, 14, H - 6))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
                cv2.putText(vis, f"t {ts - t_start:5.1f}s fused {n:2d} "
                            f"sigma {sig_p*100:5.1f}cm",
                            (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1)
                vw.write(vis)
    if vw is not None:
        vw.release()

    # ---------- convention + anchor-yaw refinement from co-visible pairs ----
    # each pair row is a gate->gate vector measured in the local frame with
    # the drone pose cancelled. For each map convention, every row implies an
    # anchor yaw; the correct convention clusters tightly and its circular
    # median IS the refined anchor yaw (immune to gate0's weak single-PnP yaw)
    print(f"\npair rows: {len(pair_rows)}, yaw rows: {len(yaw_rows)}")
    if args.dump_pairs and pair_rows:
        np.savez(args.dump_pairs, rows=np.array(
            [[t0, a0, d0[0], d0[1], d0[2], ya, yb, da, db]
             for (t0, a0, d0, ya, yb, da, db) in pair_rows]))
        print(f"dumped {len(pair_rows)} pair rows -> {args.dump_pairs}")
    if pair_rows:
        rel_raw = np.asarray(m0["gates_ring_center_NED_rel_spawn"], float)
        yaws_raw = np.asarray(m0["gate_yaw_deg"], float)

        def wrap(a):
            return (np.asarray(a) + 180.0) % 360.0 - 180.0

        print(f"\npair rows: {len(pair_rows)}, yaw rows: {len(yaw_rows)}")
        for label, mir in (("as-is", False), ("MIRROR", True)):
            rel_c = rel_raw * (np.array([1, -1, 1]) if mir else 1)
            implied = []
            for (_t, ag2, dp, *_rest) in pair_rows:
                d_obs = np.linalg.norm(dp)
                if d_obs < 3.0:
                    continue
                # the bigger det (A) is the race-status active gate; only
                # the identity of B is open — no ordering ambiguity
                cands = []
                i = ag2
                for j in range(max(0, ag2 - 1), min(len(rel_c), ag2 + 3)):
                    if i == j:
                        continue
                    dm = rel_c[j] - rel_c[i]
                    if abs(np.linalg.norm(dm) - d_obs) < max(
                            0.10 * d_obs, 0.6):
                        cands.append(np.degrees(
                            np.arctan2(dp[1], dp[0]) -
                            np.arctan2(dm[1], dm[0])))
                for a in cands:
                    implied.append((a, 1.0 / len(cands)))
            if not implied:
                print(f"  {label}: no distance-matched pairs")
                continue
            ang = np.radians([a for a, _w in implied])
            w = np.array([wgt for _a, wgt in implied])
            mean_a = np.degrees(np.arctan2((w * np.sin(ang)).sum(),
                                           (w * np.cos(ang)).sum()))
            dev = np.abs(wrap([np.degrees(x) - mean_a for x in ang]))
            order = np.argsort(dev)
            cum = np.cumsum(w[order]) / w.sum()
            mad = float(dev[order][np.searchsorted(cum, 0.5)])
            frac10 = float(w[dev < 10].sum() / w.sum())
            print(f"  {label}: n={len(implied)}  anchor-yaw mean "
                  f"{mean_a:+7.2f} deg  MAD {mad:5.2f} deg  "
                  f"within10deg {100*frac10:.0f}%")
            # per-gate yaw observations under this convention
            ya = wrap([yo - (-y if mir else y)
                       for (g, yo) in yaw_rows
                       for y in [yaws_raw[g]]])
            if len(ya):
                yv = np.radians(ya)
                ymean = np.degrees(np.arctan2(np.sin(yv).mean(),
                                              np.cos(yv).mean()))
                ymad = float(np.median(np.abs(wrap(ya - ymean))))
                print(f"           gate-yaw rows: mean {ymean:+7.2f} deg  "
                      f"MAD {ymad:5.2f} deg")

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
    if assoc_errs:
        ae = np.array(assoc_errs)
        print(f"assoc err (uncapped nearest-gate px, n={len(ae)}): "
              f"median {np.median(ae):.1f}  p75 {np.percentile(ae,75):.1f}  "
              f"<40px {100*(ae<40).mean():.0f}%")
    if edge_resid:
        print("\nlocked-state map residuals (measured - map, per gate):")
        for (i, j), offs in sorted(edge_resid.items()):
            offs = np.asarray(offs)
            med = np.median(offs, axis=0)
            print(f"  lock g{i:2d} sees g{j:2d}: n={len(offs):3d}  "
                  f"off {np.round(med, 2)}  |off| {np.linalg.norm(med):5.2f}m")
    if flip_votes:
        print("per-gate 180-flip votes (fraction matched better flipped):")
        for g, vs in sorted(flip_votes.items()):
            print(f"  g{g:2d}: {100*np.mean(vs):3.0f}% of {len(vs)}")
    if args.write_corrected_map:
        corr = {}
        for (_i, j), offs in edge_resid.items():
            corr.setdefault(j, []).extend(offs)
        n_corr = 0
        for j, offs in corr.items():
            if len(offs) < 5:
                continue
            med = np.median(np.asarray(offs), axis=0)
            if np.linalg.norm(med) < 6.0:
                gates[j]["pos"] = [float(v) for v in
                                   np.asarray(gates[j]["pos"]) + med]
                n_corr += 1
        Path(args.write_corrected_map).write_text(json.dumps(
            {"frame": "local spawn frame", "gates": gates}, indent=1))
        print(f"corrected map ({n_corr} gates nudged) -> "
              f"{args.write_corrected_map}")
    if args.dump_obs and obs_rows:
        np.savez_compressed(args.dump_obs,
                            rows=np.array(obs_rows, np.float64))
        print(f"obs: {len(obs_rows)} rows -> {args.dump_obs}")
    if args.dump_trace and trace:
        np.savez_compressed(
            args.dump_trace,
            t=np.array([r[0] for r in trace]),
            path=np.array([r[1] for r in trace]),
            pos=np.array([r[2] for r in trace], np.float64),
            quat=np.array([r[3] for r in trace], np.float64),
            sigma=np.array([r[4] for r in trace], np.float64))
        print(f"trace: {len(trace)} frames -> {args.dump_trace}")
    if args.label_dump and tl_rows:
        GS = 3
        n_r = len(tl_rows)
        arr_i = np.full((n_r, GS, 4, 2), np.nan, np.float32)
        arr_o = np.full((n_r, GS, 4, 2), np.nan, np.float32)
        v_i = np.zeros((n_r, GS, 4), bool)
        v_o = np.zeros((n_r, GS, 4), bool)
        paths_r, ig_b, ig_f = [], [], []
        for ri, (p_r, gl, igs) in enumerate(tl_rows):
            paths_r.append(p_r)
            for si, (_g, uv8, vis8) in enumerate(gl[:GS]):
                arr_i[ri, si] = uv8[0:4]
                arr_o[ri, si] = uv8[4:8]
                v_i[ri, si] = vis8[0:4]
                v_o[ri, si] = vis8[4:8]
            for b in igs:
                ig_b.append(b)
                ig_f.append(ri)
        np.savez_compressed(
            args.label_dump,
            path=np.array(paths_r), inner=arr_i, outer=arr_o,
            vis_inner=v_i, vis_outer=v_o,
            pos=np.zeros((n_r, 3), np.float32),
            vel=np.zeros((n_r, 3), np.float32),
            quat=np.tile(np.array([1, 0, 0, 0], np.float32), (n_r, 1)),
            gate_idx=np.zeros(n_r, np.int64),
            next_gate_pos=np.zeros((n_r, 3), np.float32),
            pose_valid=np.zeros(n_r, np.float32),
            ignore_boxes=np.array(ig_b, np.float32).reshape(-1, 4),
            ignore_frame_idx=np.array(ig_f, np.int64))
        n_partial = int(((v_i.sum(2) + v_o.sum(2) > 0)
                         & (v_i.sum(2) + v_o.sum(2) < 6)).sum())
        print(f"temporal labels: {n_r} frames ({n_partial} partial-view "
              f"gate slots) -> {args.label_dump}")
    print(f"relocalizations: {len(relocs)}")
    for (tr, gi, ang, rms, _pb, _pn) in relocs[:60]:
        print(f"  t {tr:5.1f}s -> gate {gi:2d}  att-agree {ang:4.1f}deg  "
              f"pnp {rms:.2f}px")
    if args.direct_position_pins:
        print(f"direct position pins: {len(direct_pin_rows)}")
        if direct_pin_rows:
            dp = np.asarray(direct_pin_rows, float)
            print(f"  jump median/p90 {np.median(dp[:,2]):.2f}/"
                  f"{np.percentile(dp[:,2],90):.2f}m, "
                  f"PnP rms median {np.median(dp[:,3]):.2f}px")
            counts = {
                int(g): int((dp[:, 1] == g).sum())
                for g in np.unique(dp[:, 1]).astype(int)}
            print(f"  pins per gate: {counts}")
        print(f"official pass pins: {len(pass_pin_rows)}")
        if pass_pin_rows:
            pp = np.asarray(pass_pin_rows, float)
            print(f"  pre-pin error median/p90 {np.median(pp[:,2]):.2f}/"
                  f"{np.percentile(pp[:,2],90):.2f}m")
    if args.crop_v11_position_pins:
        print(f"crop V11 position pins: {len(crop_position_rows)}")
        if crop_position_rows:
            cp = np.asarray(crop_position_rows, float)
            counts = {
                int(g): int((cp[:, 1] == g).sum())
                for g in np.unique(cp[:, 1]).astype(int)
            }
            print(f"  pins per gate: {counts}")
            print(
                f"  jump median/p90 {np.median(cp[:,2]):.2f}/"
                f"{np.percentile(cp[:,2],90):.2f}m, "
                f"PnP rms median {np.median(cp[:,3]):.2f}px, "
                f"sigma median {np.median(cp[:,4])*100:.1f}cm"
            )
    if args.gyro_attitude:
        print(f"gyro-attitude translation pins: "
              f"{len(gyro_translation_pins)}")
        if gyro_translation_pins:
            gp = np.asarray(gyro_translation_pins, float)
            counts = {
                int(g): int((gp[:, 1] == g).sum())
                for g in np.unique(gp[:, 1]).astype(int)}
            print(f"  pins per gate: {counts}")
            print(f"  jump median/p90 {np.median(gp[:,2]):.2f}/"
                  f"{np.percentile(gp[:,2],90):.2f}m, "
                  f"PnP rms median {np.median(gp[:,3]):.2f}px")

    # pin-chain anchor-yaw fit: dead-reckoned displacement between
    # consecutive pins vs the map's gate-to-gate vector
    if len(relocs) >= 2:
        edges = []
        for k in range(len(relocs) - 1):
            t0r, g0r, _a0, _r0, _pb0, pn0 = relocs[k]
            t1r, g1r, _a1, _r1, pb1, _pn1 = relocs[k + 1]
            dt = t1r - t0r
            if dt > 6.0 or g0r == g1r:
                continue
            d_meas = pb1 - pn0
            d_map = np.asarray(gates[g1r]["pos"]) - \
                np.asarray(gates[g0r]["pos"])
            if np.linalg.norm(d_map[:2]) < 2.0:
                continue
            dyaw = np.degrees(np.arctan2(d_meas[1], d_meas[0]) -
                              np.arctan2(d_map[1], d_map[0]))
            dyaw = (dyaw + 180) % 360 - 180
            L_map = float(np.linalg.norm(d_map))
            dlen = np.linalg.norm(d_meas) - L_map
            # rotation preserves length: an edge whose measured length
            # disagrees with the map is a broken pin/velocity, not yaw info
            if abs(dlen) > max(1.5, 0.10 * L_map):
                continue
            edges.append((g0r, g1r, dyaw, dlen, dt, L_map))
        if edges:
            print("\npin-chain edges (measured vs map):")
            for (g0r, g1r, dyaw, dlen, dt, L) in edges:
                print(f"  g{g0r:2d}->g{g1r:2d}  L={L:5.1f}m  dt={dt:4.1f}s  "
                      f"dyaw {dyaw:+6.2f}deg  dlen {dlen:+5.2f}m")
            w = np.array([e[5] / max(e[4], 0.5) for e in edges])
            dy = np.radians([e[2] for e in edges])
            fit = np.degrees(np.arctan2((w * np.sin(dy)).sum(),
                                        (w * np.cos(dy)).sum()))
            cur = anchor["anchor_yaw_deg"] if args.anchor_yaw is None \
                else args.anchor_yaw
            print(f"pin-chain yaw correction: {fit:+.2f} deg "
                  f"(n={len(edges)}) -> suggested --anchor-yaw "
                  f"{cur + fit:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
