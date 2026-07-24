"""Camera calibration by bundle adjustment against ground-truth odometry.

Solves for: fx, fy, cx, cy, camera-from-body rotation, camera position in body,
camera-vs-telemetry clock offset, and gate border offset — by minimizing
reprojection error of known gate corners over an entire logged episode.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from .detect import detect_gates
from ..ingest import load_training_episode

# base alignment: camera x=body y (right), y=body z (down), z=body x (forward)
R0 = np.array([[0.0, 1.0, 0.0],
               [0.0, 0.0, 1.0],
               [1.0, 0.0, 0.0]])


class PoseInterp:
    def __init__(self, odom):
        t = odom[:, 0]
        keep = np.concatenate([[True], np.diff(t) > 0])
        odom = odom[keep]
        self.t = odom[:, 0]
        self.pos = odom[:, 1:4]
        q_wxyz = odom[:, 4:8]
        self.rot = Rotation.from_quat(q_wxyz[:, [1, 2, 3, 0]])  # to xyzw
        self.slerp = Slerp(self.t, self.rot)
        self.t0, self.t1 = self.t[0], self.t[-1]

    def __call__(self, t_us):
        if t_us <= self.t0 or t_us >= self.t1:
            return None
        i = np.searchsorted(self.t, t_us) - 1
        a = (t_us - self.t[i]) / (self.t[i + 1] - self.t[i])
        pos = (1 - a) * self.pos[i] + a * self.pos[i + 1]
        rot = self.slerp([t_us])[0]
        return pos, rot


class SegmentedInterp:
    """Pose interpolation split at sim resets (clock restarts + teleports).

    Sim resets restart time_boot, so segments are cut at backward time jumps
    and at position teleports. Each segment also gets its own camera-clock
    offset (sim_time_ns is monotonic epoch; time_boot restarts), estimated
    from wall-clock pairing within the segment.
    """

    MARGIN_US = 300_000  # skip queries near segment boundaries

    def __init__(self, odom):
        t = odom[:, 0]
        pos = odom[:, 1:4]
        jump = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        breaks = np.where((np.diff(t) <= 0) | (jump > 5.0))[0] + 1
        self.segs = []
        self.seg_walls = []
        for chunk in np.split(odom, breaks):
            if len(chunk) >= 10:
                self.segs.append(PoseInterp(chunk))
                self.seg_walls.append((chunk[0, 15], chunk[-1, 15], chunk))
        self.offsets = [None] * len(self.segs)

    def fit_offsets(self, frames):
        """Per-segment camera-clock offset: median(sim_ns/1e3 - t_us) over
        frames wall-paired into the segment."""
        for si, (w0, w1, chunk) in enumerate(self.seg_walls):
            ests = []
            ow, ot = chunk[:, 15], chunk[:, 0]
            for (_, sim_ns, wall_ns, _) in frames:
                if not (w0 <= wall_ns <= w1):
                    continue
                i = np.argmin(np.abs(ow - wall_ns))
                ests.append(sim_ns / 1e3 - ot[i])
            if len(ests) >= 8:
                self.offsets[si] = float(np.median(ests))

    def frame_time(self, sim_ns, wall_ns):
        """Map a camera frame to a segment-local telemetry time (or None)."""
        for si, (w0, w1, _) in enumerate(self.seg_walls):
            if self.offsets[si] is None:
                continue
            if w0 - 2e8 <= wall_ns <= w1 + 2e8:
                t = sim_ns / 1e3 - self.offsets[si]
                s = self.segs[si]
                if s.t0 + self.MARGIN_US < t < s.t1 - self.MARGIN_US:
                    return t
        return None

    def __call__(self, t_us):
        if t_us is None or not np.isfinite(t_us):
            return None
        for s in self.segs:
            if s.t0 + self.MARGIN_US < t_us < s.t1 - self.MARGIN_US:
                return s(t_us)
        return None


# ---------------------------------------------------------------- geometry

def gate_corners_world(gate, b, axis_mode):
    """Outer panel corners. b is (b_side, b_top, b_bot): the visible orange
    panel extends past the opening by different amounts on the sides, top
    (AI-GP header band), and bottom (logo band). NED z is down, so the top
    edge is at -z."""
    if np.isscalar(b):
        b = (b, b, b)
    b_side, b_top, b_bot = b
    w2 = gate["width"] / 2.0 + b_side
    z_top = -(gate["height"] / 2.0 + b_top)
    z_bot = gate["height"] / 2.0 + b_bot
    if axis_mode == 0:      # gate plane = local Y-Z (X is flight direction)
        local = np.array([[0, -w2, z_top], [0, w2, z_top],
                          [0, w2, z_bot], [0, -w2, z_bot]], float)
    else:                   # gate plane = local X-Z
        local = np.array([[-w2, 0, z_top], [w2, 0, z_top],
                          [w2, 0, z_bot], [-w2, 0, z_bot]], float)
    qw, qx, qy, qz = gate["quat_wxyz"]
    Rg = Rotation.from_quat([qx, qy, qz, qw])
    return np.asarray(gate["pos"], float) + Rg.apply(local)


def project(pts_w, pos_wb, rot_wb, R_cb, t_cb, fx, fy, cx, cy, q_is_body_to_world):
    Rwb = rot_wb.as_matrix()
    if q_is_body_to_world:
        Xb = (pts_w - pos_wb) @ Rwb          # R_wb^T applied to rows
    else:
        Xb = (pts_w - pos_wb) @ Rwb.T
    Xc = Xb @ R_cb.T + t_cb
    z = Xc[:, 2]
    valid = z > 0.8
    uv = np.empty((len(pts_w), 2))
    uv[:, 0] = fx * Xc[:, 0] / np.maximum(z, 1e-6) + cx
    uv[:, 1] = fy * Xc[:, 1] / np.maximum(z, 1e-6) + cy
    return uv, valid, z


def greedy_match(proj, det):
    """One-to-one greedy corner match; returns (proj_idx, det_idx) pairs."""
    D = np.linalg.norm(proj[:, None, :] - det[None, :, :], axis=2)
    pairs = []
    used_p, used_d = set(), set()
    for _ in range(4):
        idx = np.unravel_index(np.argmin(D), D.shape)
        if D[idx] == np.inf:
            break
        pairs.append((idx[0], idx[1], D[idx]))
        D[idx[0], :] = np.inf
        D[:, idx[1]] = np.inf
    return pairs


# ---------------------------------------------------------------- association

def associate(frame_dets, frame_times, interp, gates, params, hyp, max_center_dist=70.0):
    """Build corner correspondences under the current model.

    Returns list of (frame_idx, gate_idx, corner_idx(0-3), u_obs, v_obs,
    kind) where kind 0 = inner opening (exact known gate dims, no border
    params) and kind 1 = outer panel edge (with border params)."""
    fx, fy, cx, cy = params["fx"], params["fy"], params["cx"], params["cy"]
    R_cb, t_cb = params["R_cb"], params["t_cb"]
    dt = params["dt_us"]
    b = params["b"]
    corr = []
    outer_pts = [gate_corners_world(g, b, hyp["axis_mode"]) for g in gates]
    inner_pts = [gate_corners_world(g, (0.0, 0.0, 0.0), hyp["axis_mode"])
                 for g in gates]
    for fi, dets in enumerate(frame_dets):
        if not dets:
            continue
        pose = interp(frame_times[fi] - dt)
        if pose is None:
            continue
        pos_wb, rot_wb = pose
        projected = []
        for gi in range(len(gates)):
            uv_o, valid_o, _ = project(outer_pts[gi], pos_wb, rot_wb, R_cb, t_cb,
                                       fx, fy, cx, cy, hyp["q_b2w"])
            uv_i, valid_i, _ = project(inner_pts[gi], pos_wb, rot_wb, R_cb, t_cb,
                                       fx, fy, cx, cy, hyp["q_b2w"])
            if not (valid_o.all() and valid_i.all()):
                continue
            span = uv_o.max(axis=0) - uv_o.min(axis=0)
            if span.max() < 10 or span.max() > 900:
                continue
            projected.append((gi, uv_o, uv_i, uv_o.mean(axis=0), span.max()))
        for det in dets:
            dc = det["center"]
            dspan = det["outer"].max(axis=0) - det["outer"].min(axis=0)
            best = None
            for gi, uv_o, uv_i, c, span in projected:
                cd = np.linalg.norm(c - dc)
                ratio = span / max(dspan.max(), 1e-6)
                if cd < max_center_dist and 0.35 < ratio < 2.8:
                    if best is None or cd < best[0]:
                        best = (cd, gi, uv_o, uv_i)
            if best is None:
                continue
            _, gi, uv_o, uv_i = best
            diag = np.linalg.norm(dspan)
            gate_px = max(10.0, 0.35 * diag)
            pairs = greedy_match(uv_o, det["outer"])
            if len(pairs) == 4 and max(p[2] for p in pairs) < gate_px:
                for (pi, di, _) in pairs:
                    corr.append((fi, gi, pi, det["outer"][di, 0],
                                 det["outer"][di, 1], 1))
            if det["inner"] is not None:
                pairs = greedy_match(uv_i, det["inner"])
                if len(pairs) == 4 and max(p[2] for p in pairs) < gate_px:
                    for (pi, di, _) in pairs:
                        corr.append((fi, gi, pi, det["inner"][di, 0],
                                     det["inner"][di, 1], 0))
    return corr


# ---------------------------------------------------------------- residuals

def make_residual_fn(corr, frame_times, interp, gates, hyp):
    corr = np.array(corr, dtype=np.float64)
    fis = corr[:, 0].astype(int)
    gis = corr[:, 1].astype(int)
    cis = corr[:, 2].astype(int)
    obs = corr[:, 3:5]
    kinds = corr[:, 5].astype(int) if corr.shape[1] > 5 else np.ones(len(corr), int)

    def fn(p):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        R_cb = Rotation.from_rotvec(p[4:7]).as_matrix()
        t_cb = p[7:10]
        dt = p[10]
        b = tuple(p[11:14])
        outer = {gi: gate_corners_world(gates[gi], b, hyp["axis_mode"])
                 for gi in np.unique(gis)}
        inner = {gi: gate_corners_world(gates[gi], (0.0, 0.0, 0.0),
                                        hyp["axis_mode"])
                 for gi in np.unique(gis)}
        res = np.zeros((len(corr), 2))
        pose_cache = {}
        for k in range(len(corr)):
            fi = fis[k]
            if fi not in pose_cache:
                pose_cache[fi] = interp(frame_times[fi] - dt)
            pose = pose_cache[fi]
            if pose is None:
                continue
            pos_wb, rot_wb = pose
            pts = outer if kinds[k] == 1 else inner
            pt = pts[gis[k]][cis[k]:cis[k] + 1]
            uv, valid, _ = project(pt, pos_wb, rot_wb, R_cb, t_cb,
                                   fx, fy, cx, cy, hyp["q_b2w"])
            res[k] = uv[0] - obs[k]
        return res.ravel()

    return fn


# ---------------------------------------------------------------- solver

def grid_cost(frame_dets, frame_times, interp, gates, params, hyp):
    corr = associate(frame_dets, frame_times, interp, gates, params, hyp)
    if len(corr) < 40:
        return 1e9, len(corr)
    fn = make_residual_fn(corr, frame_times, interp, gates, hyp)
    p = pack_params(params)
    r = fn(p).reshape(-1, 2)
    d = np.linalg.norm(r, axis=1)
    # robust mean, and reward high match counts
    hub = np.where(d < 3.0, d ** 2 / 2, 3.0 * d - 4.5)
    return hub.mean() + 2000.0 / len(corr), len(corr)


def pack_params(P):
    b = P["b"]
    if np.isscalar(b):
        b = (b, b, b)
    return np.concatenate([
        [P["fx"], P["fy"], P["cx"], P["cy"]],
        Rotation.from_matrix(P["R_cb"]).as_rotvec(),
        P["t_cb"], [P["dt_us"]], list(b),
    ])


def unpack_params(p):
    return {
        "fx": p[0], "fy": p[1], "cx": p[2], "cy": p[3],
        "R_cb": Rotation.from_rotvec(p[4:7]).as_matrix(),
        "t_cb": np.array(p[7:10]), "dt_us": p[10], "b": tuple(p[11:14]),
    }


def solve(ep_dir, sample_step=2, max_frames=1200, verbose=True,
          fix_intrinsics=None):
    """fix_intrinsics: optional (fx, fy, cx, cy) to lock (e.g. the rig meta's
    320/320/320/180) so only rotation, clock offset, and borders are solved."""
    bundle = load_training_episode(ep_dir)
    ep = Path(bundle["dir"])
    odom, gates = bundle["odom"], bundle["gates"]
    if not gates:
        raise RuntimeError("no gate map in episode")
    frames = bundle["frames"][::sample_step][:max_frames]
    interp = SegmentedInterp(odom)
    interp.fit_offsets(bundle["frames"])
    n_off = sum(1 for o in interp.offsets if o is not None)
    print(f"{len(odom)} odom samples in {len(interp.segs)} segments "
          f"({n_off} with clock offsets), {len(frames)} frames sampled, "
          f"{len(gates)} gates", flush=True)

    # detect corners; frame time mapped per clock segment
    frame_dets, frame_times = [], []
    print(f"Detecting gates in {len(frames)} frames...", flush=True)
    for (fid, sim_ns, wall_ns, path) in frames:
        t = interp.frame_time(sim_ns, wall_ns)
        if t is None:
            frame_dets.append([])
            frame_times.append(np.nan)
            continue
        img = cv2.imread(path)
        dets = detect_gates(img) if img is not None else []
        frame_dets.append(dets)
        frame_times.append(t)
    n_det = sum(len(d) for d in frame_dets)
    n_timed = sum(1 for t in frame_times if np.isfinite(t))
    print(f"{n_det} quads detected across {n_timed} time-mapped frames "
          f"(of {len(frames)})", flush=True)

    # ---- hypothesis grid ----
    # For rc bundles the euler sign convention is part of the search: the
    # quats are rebuilt per sign combo from the raw eulers in cols 11:14.
    from ..ingest import apply_euler_signs

    W, H = 640, 360
    base = {"cx": (W - 1) / 2, "cy": (H - 1) / 2, "t_cb": np.zeros(3),
            "dt_us": 0.0, "b": (0.0, 0.0, 0.0)}
    is_rc = (Path(bundle["dir"]) / "mav.jsonl").exists()
    use_signs = is_rc and not bundle.get("native_quat", False)
    sign_combos = ([(-1, -1, 1), (1, 1, 1), (-1, 1, 1), (1, -1, 1),
                    (-1, -1, -1), (1, 1, -1), (-1, 1, -1), (1, -1, -1)]
                   if use_signs else [None])
    best = None
    interps = {}
    for signs in sign_combos:
        if signs is None:
            si = interp
        else:
            od2 = odom.copy()
            apply_euler_signs(od2, *signs)
            si = SegmentedInterp(od2)
            si.offsets = list(interp.offsets)
            si.seg_walls = si.seg_walls  # offsets are sign-independent
        interps[signs] = si
        for f in (240.0, 300.0, 320.0, 360.0, 420.0, 500.0, 620.0):
            for tilt in (-0.45, -0.35, -0.31, 0.0, 0.31, 0.35, 0.45):
                R_cb = (Rotation.from_rotvec([tilt, 0, 0]).as_matrix() @ R0)
                for axis_mode in (0, 1):
                    for q_b2w in ((True, False) if signs is None else (True,)):
                        P = dict(base, fx=f, fy=f, R_cb=R_cb)
                        hyp = {"axis_mode": axis_mode, "q_b2w": q_b2w,
                               "signs": signs}
                        c, n = grid_cost(frame_dets, frame_times, si, gates,
                                         P, hyp)
                        if best is None or c < best[0]:
                            best = (c, n, P, hyp, f, tilt)
    c, n, P, hyp, f_win, tilt_win = best
    interp = interps[hyp["signs"]]
    print(f"grid winner: f={f_win} tilt={tilt_win} axis={hyp['axis_mode']} "
          f"signs={hyp['signs']} cost={c:.2f} matches={n}", flush=True)

    # ---- dt grid on winner ----
    best_dt = (c, 0.0)
    for dt in np.arange(-250_000, 250_001, 2_000):
        P2 = dict(P, dt_us=float(dt))
        cc, nn = grid_cost(frame_dets, frame_times, interp, gates, P2, hyp)
        if cc < best_dt[0]:
            best_dt = (cc, float(dt))
    P["dt_us"] = best_dt[1]
    print(f"dt grid: {best_dt[1]:.0f} us (cost {best_dt[0]:.2f})", flush=True)

    # ---- alternating association / least-squares ----
    for rnd in range(4):
        corr = associate(frame_dets, frame_times, interp, gates, P, hyp,
                         max_center_dist=70.0 if rnd == 0 else 30.0)
        if len(corr) < 40:
            raise RuntimeError(f"too few correspondences: {len(corr)}")
        fn = make_residual_fn(corr, frame_times, interp, gates, hyp)
        p0 = pack_params(P)
        scales = np.array([100, 100, 50, 50, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2,
                           5000, 0.05, 0.05, 0.05])
        # keep the solution physical: principal point near center, camera at
        # the body origin (+-10cm so it can't absorb timing error), wide
        # clock refinement, per-edge panel borders
        lo = np.array([150, 150, 290, 140, -np.inf, -np.inf, -np.inf,
                       -0.10, -0.10, -0.10, -250_000, -1.2, -1.2, -1.2])
        hi = np.array([800, 800, 350, 220, np.inf, np.inf, np.inf,
                       0.10, 0.10, 0.10, 250_000, 1.2, 1.2, 1.2])
        if fix_intrinsics is not None:
            for k, v in enumerate(fix_intrinsics):
                p0[k] = v
                lo[k] = v - 1e-6
                hi[k] = v + 1e-6
            # camera at exact body origin as well
            p0[7:10] = 0.0
            lo[7:10] = -1e-5
            hi[7:10] = 1e-5
        p0 = np.clip(p0, lo + 1e-7, hi - 1e-7)
        sol = least_squares(fn, p0, loss="huber", f_scale=2.0, x_scale=scales,
                            bounds=(lo, hi), max_nfev=200)
        P = unpack_params(sol.x)
        r = sol.fun.reshape(-1, 2)
        d = np.linalg.norm(r, axis=1)
        inl = d < 3.0
        print(f"round {rnd}: {len(corr)} corners, rms={np.sqrt((d**2).mean()):.3f}px, "
              f"inlier rms={np.sqrt((d[inl]**2).mean()):.3f}px "
              f"({inl.mean()*100:.1f}% inliers)", flush=True)

    # ---- visual overlays for sanity checking ----
    ov_dir = Path("data") / "calib" / "overlays"
    ov_dir.mkdir(parents=True, exist_ok=True)
    with_dets = [i for i, d in enumerate(frame_dets) if d and np.isfinite(frame_times[i])]
    picks = with_dets[:: max(1, len(with_dets) // 6)][:6]
    gate_pts = [gate_corners_world(g, P["b"], hyp["axis_mode"]) for g in gates]
    for i in picks:
        img = cv2.imread(frames[i][3])
        if img is None:
            continue
        pose = interp(frame_times[i] - P["dt_us"])
        if pose is None:
            continue
        for d0 in frame_dets[i]:
            cv2.polylines(img, [d0["outer"].astype(np.int32)], True, (0, 255, 0), 1)
        for pts in gate_pts:
            uv, valid, _ = project(pts, pose[0], pose[1], P["R_cb"], P["t_cb"],
                                   P["fx"], P["fy"], P["cx"], P["cy"], hyp["q_b2w"])
            if valid.all() and np.all(np.abs(uv) < 3000):
                cv2.polylines(img, [uv.astype(np.int32)], True, (255, 0, 255), 1)
        cv2.imwrite(str(ov_dir / f"overlay_{i:05d}.jpg"), img)

    # ---- report + save ----
    result = {
        "fx": P["fx"], "fy": P["fy"], "cx": P["cx"], "cy": P["cy"],
        "R_cam_from_body": P["R_cb"].tolist(),
        "cam_tilt_deg": float(np.degrees(
            Rotation.from_matrix(P["R_cb"] @ R0.T).magnitude())),
        "t_cam_in_body": P["t_cb"].tolist(),
        "seg_clock_offsets_us": [o for o in interp.offsets],
        "dt_refine_us": P["dt_us"],
        "border_side_top_bot_m": list(P["b"]),
        "axis_mode": hyp["axis_mode"],
        "euler_signs": hyp["signs"],
        "n_corners": len(corr),
        "rms_px": float(np.sqrt((d ** 2).mean())),
        "inlier_rms_px": float(np.sqrt((d[inl] ** 2).mean())),
        "inlier_frac": float(inl.mean()),
        "episode": str(ep),
    }
    out = Path("data") / "calib"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calib.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "R_cam_from_body"}, indent=2), flush=True)
    print(f"saved {out / 'calib.json'} and overlays to {ov_dir}", flush=True)
    return result
