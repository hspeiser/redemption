"""Auto-label generator: for every frame of every usable (native-timestamp,
VQ1-only) episode, project the known gate map through the calibrated camera at
the ground-truth pose and emit training labels.

Per-frame labels:
  jpg path, drone pos (3), drone quat wxyz (4), drone vel world (3),
  active gate idx, next-gate pos (3),
  per gate: inner corners (4,2), outer corners (4,2), per-corner in-image
  flags, gate center depth.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import cv2

from ..ingest import load_training_episode, is_on_vq1_course
from ..calib.solve import SegmentedInterp, gate_corners_world, project, greedy_match
from ..calib.detect import detect_gates, gate_mask

W, H = 640, 360


def load_calib(calib_path):
    c = json.loads(Path(calib_path).read_text())
    return {
        "K": (c["fx"], c["fy"], c["cx"], c["cy"]),
        "R_cb": np.array(c["R_cam_from_body"]),
        "t_cb": np.array(c["t_cam_in_body"]),
        "b_outer": tuple(c["border_side_top_bot_m"]),
        "axis_mode": c["axis_mode"],
        "dt_us": c["dt_refine_us"],
    }


def _resolve_q_b2w(interp, frames, gates, calib):
    """The calib file doesn't record the quat-direction hypothesis; determine
    it by which direction keeps projected gates on-screen more often."""
    fx, fy, cx, cy = calib["K"]
    scores = {}
    for q_b2w in (True, False):
        on = 0
        for (fid, sim_ns, wall_ns, path) in frames[:: max(1, len(frames) // 200)]:
            t = interp.frame_time(sim_ns, wall_ns)
            pose = interp(t - calib["dt_us"]) if t is not None else None
            if pose is None:
                continue
            for g in gates:
                pts = gate_corners_world(g, (0, 0, 0), calib["axis_mode"])
                uv, valid, z = project(pts, pose[0], pose[1], calib["R_cb"],
                                       calib["t_cb"], fx, fy, cx, cy, q_b2w)
                if valid.all() and (uv[:, 0] > 0).all() and (uv[:, 0] < W).all() \
                        and (uv[:, 1] > 0).all() and (uv[:, 1] < H).all():
                    on += 1
        scores[q_b2w] = on
    return scores[True] >= scores[False]


def refine_episode_dt(b, interp, calib, q_b2w, n_frames=220,
                      grid_ms=(-60, 60, 2)):
    """Gates are STATIC, so projected inner corners must coincide with clean
    detections up to a per-episode clock-offset bias. Grid-search that bias.

    Returns (dt_us, median_px_after, n_pairs) — dt_us is 0 if too little data.
    """
    gates = b["gates"]
    fx, fy, cx, cy = calib["K"]
    panel_pts = [gate_quads_world(g)[1] for g in gates]
    frames = b["frames"]
    step = max(1, len(frames) // n_frames)
    cache = []
    for (fid, sim_ns, wall_ns, path) in frames[::step]:
        t = interp.frame_time(sim_ns, wall_ns)
        if t is None:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        dets = detect_gates(img, min_area=250)
        if dets:
            cache.append((t, dets))
    if len(cache) < 15:
        return 0.0, None, 0

    def cost(dt):
        dists = []
        for (t, dets) in cache:
            pose = interp(t - calib["dt_us"] - dt)
            if pose is None:
                continue
            projs = []
            for pts in panel_pts:
                uv, valid, z = project(pts, pose[0], pose[1], calib["R_cb"],
                                       calib["t_cb"], fx, fy, cx, cy, q_b2w)
                if valid.all() and (np.abs(uv) < 3000).all():
                    projs.append(uv)
            for det in dets:
                dc = det["outer"].mean(0)
                diag = np.linalg.norm(det["outer"].max(0) - det["outer"].min(0))
                gate_px = max(10.0, 0.35 * diag)
                best = None
                for uv in projs:
                    d0 = np.linalg.norm(uv.mean(0) - dc)
                    if d0 < 60 and (best is None or d0 < best[0]):
                        best = (d0, uv)
                if best is None:
                    continue
                pairs = greedy_match(best[1], det["outer"])
                if len(pairs) == 4 and max(p[2] for p in pairs) < gate_px:
                    dists += [p[2] for p in pairs]
        if len(dists) < 40:
            return None, 0
        return float(np.median(dists)), len(dists)

    best = (None, 0.0, 0)
    for dt_ms in np.arange(grid_ms[0], grid_ms[1] + 1e-9, grid_ms[2]):
        c, n = cost(dt_ms * 1000.0)
        if c is not None and (best[0] is None or c < best[0]):
            best = (c, dt_ms * 1000.0, n)
    if best[0] is None:
        return 0.0, None, 0
    # fine pass at 0.5 ms
    coarse = best[1]
    for dt_us in np.arange(coarse - 2000, coarse + 2001, 500.0):
        c, n = cost(dt_us)
        if c is not None and c < best[0]:
            best = (c, dt_us, n)
    return best[1], best[0], best[2]


# TRUE gate geometry (verified 2026-07-23, both quads fit detections ~2 px):
# orange panel outer boundary = 2.72 m (the map's "width" refers to THIS),
# fly-through hole = 1.50 m (Henry's spec), both centered ~1.07 m ABOVE the
# map anchor pos. Solved panel edges in gate frame (axis_mode=1, NED z down):
PANEL_X = 1.324
PANEL_ZT, PANEL_ZB = -2.416, 0.267
PANEL_CZ = 0.5 * (PANEL_ZT + PANEL_ZB)
HOLE_HALF = 0.75


def gate_quads_world(gate):
    """Returns (hole_corners(4,3), panel_corners(4,3)) in world frame."""
    from scipy.spatial.transform import Rotation
    qw, qx, qy, qz = gate["quat_wxyz"]
    Rg = Rotation.from_quat([qx, qy, qz, qw])
    panel = np.array([[-PANEL_X, 0, PANEL_ZT], [PANEL_X, 0, PANEL_ZT],
                      [PANEL_X, 0, PANEL_ZB], [-PANEL_X, 0, PANEL_ZB]])
    hole = np.array([[-HOLE_HALF, 0, PANEL_CZ - HOLE_HALF],
                     [HOLE_HALF, 0, PANEL_CZ - HOLE_HALF],
                     [HOLE_HALF, 0, PANEL_CZ + HOLE_HALF],
                     [-HOLE_HALF, 0, PANEL_CZ + HOLE_HALF]])
    p = np.asarray(gate["pos"], float)
    return p + Rg.apply(hole), p + Rg.apply(panel)


def build_episode_labels(ep_dir, calib, out_dir):
    b = load_training_episode(ep_dir)
    gates = b["gates"]
    interp = SegmentedInterp(b["odom"])
    interp.fit_offsets(b["frames"])
    # Quat convention is a property of the SIM, not the episode: body->world
    # (verified by rate probes AND by both gate quads fitting detections at
    # ~2 px under True). The old per-episode resolver was a coin flip near
    # yaw=pi (R == R^T there) and silently corrupted half the labels.
    q_b2w = True
    dt_ep, agree_px, n_pairs = refine_episode_dt(b, interp, calib, q_b2w)
    print(f"  dt refine: {dt_ep/1000.0:+.1f} ms, agreement "
          f"{agree_px if agree_px is None else round(agree_px, 2)} px "
          f"({n_pairs} corner pairs)", flush=True)

    # race status: (wall_ns, active_gate) from the rc mav log
    race = []
    mav = Path(b["dir"]) / "mav.jsonl"
    if mav.exists():
        with open(mav) as fh:
            for line in fh:
                if '"race_status"' not in line:
                    continue
                try:
                    r = json.loads(line)
                    race.append((r["wall"] * 1e9, r["active_gate"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    race = np.array(race) if race else np.zeros((0, 2))

    fx, fy, cx, cy = calib["K"]
    n_g = len(gates)
    quads = [gate_quads_world(g) for g in gates]
    inner_pts = [q[0] for q in quads]   # 1.5 m hole
    outer_pts = [q[1] for q in quads]   # 2.72 m panel
    gate_pos = np.array([g["pos"] for g in gates])
    dt_total = calib["dt_us"] + dt_ep

    rows = {k: [] for k in ("path", "pos", "quat", "vel", "gate_idx",
                            "next_gate_pos", "inner", "outer", "vis_inner",
                            "vis_outer", "depth")}
    ignore_boxes = []      # (M, 4) across all frames
    ignore_frame_idx = []  # (M,) row index each box belongs to
    for (fid, sim_ns, wall_ns, path) in b["frames"]:
        t = interp.frame_time(sim_ns, wall_ns)
        if t is None:
            continue
        pose = interp(t - dt_total)
        if pose is None:
            continue
        pos_wb, rot_wb = pose
        if not is_on_vq1_course(pos_wb, gates):
            continue
        # velocity: nearest odom row (col 8:11 is body-frame for native quats)
        oi = np.searchsorted(interp.segs[0].t, t) if len(interp.segs) == 1 else None
        vel_w = np.zeros(3)
        od = b["odom"]
        k = np.argmin(np.abs(od[:, 0] - t))
        Rwb = rot_wb.as_matrix() if q_b2w else rot_wb.as_matrix().T
        vel_w = Rwb @ od[k, 8:11]
        # active gate from race status by wall time
        gi_active = 0
        if len(race):
            ri = np.searchsorted(race[:, 0], wall_ns) - 1
            if ri >= 0:
                gi_active = int(min(max(race[ri, 1], 0), n_g - 1))
        # ------------------------------------------------------------------
        # Gates are STATIC: with the per-episode clock refined, the map
        # projection IS the geometric corner truth — including on frames
        # where the classical detector fails (motion blur, blue-ribbon
        # glow). Visibility comes from sampling the orange mask; gates whose
        # projected panel shows no orange (occluded by structures or fully
        # glow-killed) become ignore regions rather than negatives.
        # ------------------------------------------------------------------
        img = cv2.imread(path)
        omask = gate_mask(img) if img is not None else None
        if omask is not None:
            omask = cv2.dilate(omask, np.ones((5, 5), np.uint8))

        inner = np.full((n_g, 4, 2), np.nan, np.float32)
        outer = np.full((n_g, 4, 2), np.nan, np.float32)
        vis_i = np.zeros((n_g, 4), bool)
        vis_o = np.zeros((n_g, 4), bool)
        depth = np.full(n_g, np.nan, np.float32)
        ignore = []  # (u0, v0, u1, v1) full-res boxes

        def orange_at(u, v, r=4):
            if omask is None:
                return False
            x0, x1 = int(u - r), int(u + r + 1)
            y0, y1 = int(v - r), int(v + r + 1)
            if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
                return False
            return omask[max(0, y0):y1, max(0, x0):x1].any()

        for gi in range(n_g):
            uv_i, val_i, z_i = project(inner_pts[gi], pos_wb, rot_wb,
                                       calib["R_cb"], calib["t_cb"],
                                       fx, fy, cx, cy, q_b2w)
            uv_o, val_o, z_o = project(outer_pts[gi], pos_wb, rot_wb,
                                       calib["R_cb"], calib["t_cb"],
                                       fx, fy, cx, cy, q_b2w)
            depth[gi] = z_i.mean()
            if not (val_i.all() and val_o.all()) or z_i.mean() > 60.0:
                continue
            inner[gi] = uv_i
            outer[gi] = uv_o
            inb_o = (uv_o[:, 0] >= 0) & (uv_o[:, 0] < W) & \
                (uv_o[:, 1] >= 0) & (uv_o[:, 1] < H)
            inb_i = (uv_i[:, 0] >= 0) & (uv_i[:, 0] < W) & \
                (uv_i[:, 1] >= 0) & (uv_i[:, 1] < H)
            if not (inb_o.any() or inb_i.any()):
                continue
            # gate-level visibility: orange along the border band (midway
            # between inner and outer corners + edge midpoints)
            band = 0.5 * (uv_i + uv_o)
            samples = list(band) + list(0.5 * (band + np.roll(band, 1, 0)))
            n_hit = sum(orange_at(u, v) for (u, v) in samples)
            gate_visible = n_hit >= max(2, int(0.25 * len(samples)))
            proj_c = uv_o.mean(axis=0)
            proj_span = float((uv_o.max(0) - uv_o.min(0)).max())
            if not gate_visible:
                pad = 0.6 * proj_span + 20
                ignore.append([proj_c[0] - pad, proj_c[1] - pad,
                               proj_c[0] + pad, proj_c[1] + pad])
                continue
            for c in range(4):
                vis_i[gi, c] = bool(inb_i[c]) and orange_at(*uv_i[c], r=6)
                vis_o[gi, c] = bool(inb_o[c]) and orange_at(*uv_o[c], r=6)
        ignore = np.array(ignore, np.float32) if ignore else np.zeros((0, 4), np.float32)
        for box in ignore:
            ignore_boxes.append(box)
            ignore_frame_idx.append(len(rows["path"]))
        q = rot_wb.as_quat()  # xyzw
        rows["path"].append(str(path))
        rows["pos"].append(pos_wb)
        rows["quat"].append([q[3], q[0], q[1], q[2]])
        rows["vel"].append(vel_w)
        rows["gate_idx"].append(gi_active)
        rows["next_gate_pos"].append(gate_pos[gi_active])
        rows["inner"].append(inner)
        rows["outer"].append(outer)
        rows["vis_inner"].append(vis_i)
        rows["vis_outer"].append(vis_o)
        rows["depth"].append(depth)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(b["dir"]).name
    np.savez_compressed(
        out / f"{name}.npz",
        path=np.array(rows["path"]),
        pos=np.array(rows["pos"], np.float32),
        quat=np.array(rows["quat"], np.float32),
        vel=np.array(rows["vel"], np.float32),
        gate_idx=np.array(rows["gate_idx"], np.int64),
        next_gate_pos=np.array(rows["next_gate_pos"], np.float32),
        inner=np.array(rows["inner"], np.float32),
        outer=np.array(rows["outer"], np.float32),
        vis_inner=np.array(rows["vis_inner"]),
        vis_outer=np.array(rows["vis_outer"]),
        depth=np.array(rows["depth"], np.float32),
        ignore_boxes=(np.array(ignore_boxes, np.float32)
                      if ignore_boxes else np.zeros((0, 4), np.float32)),
        ignore_frame_idx=np.array(ignore_frame_idx, np.int64),
        q_b2w=np.array([q_b2w]),
    )
    return name, len(rows["path"])
