"""Loaders that normalize episodes into one bundle format.

Two sources:
  - our own EpisodeLogger output (telemetry.npz + frames_index.csv + gates.json)
  - the user's RC capture rig (rc_*: mav.jsonl raw-hex payloads, frames.jsonl
    with ~40x duplicated rows, imu.jsonl, cmd.jsonl, meta.json)

Bundle:
  odom   : (N,16) float64 [t_us, x,y,z, qw,qx,qy,qz, vx,vy,vz, rs,ps,ys,
                           reset_counter, wall_ns]
  frames : list of (frame_id, sim_time_ns, wall_ns, jpg_path)
  gates  : list of gate dicts (gate_id, pos, quat_wxyz, width, height)
  meta   : dict (rc meta.json contents when available)
"""

import csv
import json
import struct
from pathlib import Path

import numpy as np

MSG_ODOMETRY = 331
MSG_ENCAPSULATED_DATA = 131
MSG_HANDSHAKE = 130


def _decode_gate_payload(payload):
    num_gates, = struct.unpack_from("<H", payload)
    payload = payload[2:]
    gates = []
    for _ in range(num_gates):
        gid, x, y, z, qw, qx, qy, qz, w, h = struct.unpack_from("<Hfffffffff", payload)
        payload = payload[38:]
        gates.append({"gate_id": gid, "pos": [x, y, z],
                      "quat_wxyz": [qw, qx, qy, qz], "width": w, "height": h})
    gates.sort(key=lambda g: g["gate_id"])
    return gates


def load_rc_episode(ep_dir):
    ep = Path(ep_dir)
    meta = json.loads((ep / "meta.json").read_text()) if (ep / "meta.json").exists() else {}

    # ---- frames: dedupe by frame_id, keep first row ----
    frames = []
    seen = set()
    with open(ep / "frames.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = r["frame_id"]
            if fid in seen:
                continue
            seen.add(fid)
            path = ep / "frames" / f"{r['idx']:06d}.jpg"
            frames.append((fid, r["sim_time_ns"], int(r["wall"] * 1e9), str(path)))
    frames.sort(key=lambda f: f[1])

    # ---- mavlink ----
    # The rig decodes known messages to fields and leaves unknown/track
    # messages as raw payload hex. ODOMETRY rows lost their sim timestamp
    # (wall clock only, +-18ms jitter), but ATTITUDE and LOCAL_POSITION_NED
    # kept time_boot_ms (exact sim clock, 1ms resolution) — so the pose
    # stream is rebuilt from those two instead.
    att_rows = []    # (t_us, roll, pitch, yaw, wall_ns)  [reported signs]
    lpos_rows = []   # (t_us, x,y,z, vx,vy,vz, wall_ns)
    odom_native = [] # rows when the rig logged ODOMETRY time_usec (new rigs)
    track_chunks, track_expected = {}, {}
    gate_maps = []  # (wall_ns, gates)
    with open(ep / "mav.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            mid = r.get("msg_id")
            if mid == MSG_ODOMETRY and "time_usec" in r and "pos" in r:
                # SIM QUAT CONVENTION FIX (verified vs PnP-true camera
                # rotation on banked frames, 1.45 deg residual): the sim's
                # quaternion is Y-flipped left-handed (Unreal). Correct
                # reading = (w, -x, y, -z). Identical to the naive reading
                # at zero roll & yaw~pi, which is why rest checks passed.
                qw, qx, qy, qz = r["quat_wxyz"]
                odom_native.append((
                    float(r["time_usec"]),
                    *r["pos"], qw, -qx, qy, -qz, *r["vel"], *r["body_rates"],
                    0, int(r["wall"] * 1e9),
                ))
            elif mid == 30 and "time_boot_ms" in r:
                att_rows.append((r["time_boot_ms"] * 1000.0, r["roll"],
                                 r["pitch"], r["yaw"], int(r["wall"] * 1e9)))
            elif mid == 32 and "time_boot_ms" in r:
                lpos_rows.append((r["time_boot_ms"] * 1000.0, *r["pos"],
                                  *r["vel"], int(r["wall"] * 1e9)))
            elif mid == MSG_HANDSHAKE and r.get("raw"):
                buf = bytes.fromhex(r["raw"])
                if len(buf) >= 10:
                    width, = struct.unpack_from("<H", buf, 4)
                    packets, = struct.unpack_from("<H", buf, 8)
                    track_expected[width] = packets
                    track_chunks[width] = {}
            elif mid == MSG_ENCAPSULATED_DATA and r.get("raw") \
                    and r.get("data_type") == 2:
                buf = bytes.fromhex(r["raw"])
                if len(buf) < 6:
                    continue
                seqnr, = struct.unpack_from("<H", buf, 0)
                data = buf[2:]
                _, transfer_id = struct.unpack_from("<BH", data)
                if transfer_id not in track_expected:
                    continue
                track_chunks[transfer_id][seqnr] = data[3:]
                if len(track_chunks[transfer_id]) == track_expected[transfer_id]:
                    chunks = track_chunks.pop(transfer_id)
                    del track_expected[transfer_id]
                    payload = b"".join(chunks[i] for i in range(len(chunks)))
                    try:
                        gate_maps.append((int(r["wall"] * 1e9),
                                          _decode_gate_payload(payload)))
                    except struct.error:
                        pass

    # majority-vote map (keyed by rounded gate-0 position)
    gates = []
    if gate_maps:
        from collections import Counter
        keys = [tuple(np.round(g[1][0]["pos"], 1)) for g in gate_maps]
        winner = Counter(keys).most_common(1)[0][0]
        for k, (w, gm) in zip(keys, gate_maps):
            if k == winner:
                gates = gm
                break

    if len(odom_native) > 100:
        # native path: exact us timestamps + native quats, no reconstruction.
        # Drop duplicate timestamps (recorder logs some rows repeatedly);
        # keep genuine backward jumps (sim-reset clock restarts) for
        # segmentation downstream.
        odom = np.array(odom_native, dtype=np.float64)
        dup = np.concatenate([[False], np.diff(odom[:, 0]) == 0])
        odom = odom[~dup]
        native = True
    else:
        odom = _pose_from_att_lpos(att_rows, lpos_rows)
        native = False
    return {"odom": odom, "frames": frames, "gates": gates,
            "gate_maps": gate_maps, "meta": meta, "dir": str(ep),
            "native_quat": native}


def split_on_clock_reset(arr, t_col=0, min_len=5):
    """Split a time-stamped array at backward time jumps (sim resets restart
    the time_boot clock)."""
    if len(arr) == 0:
        return []
    dt = np.diff(arr[:, t_col])
    breaks = np.where(dt <= 0)[0] + 1
    return [c for c in np.split(arr, breaks) if len(c) >= min_len]


def _pose_from_att_lpos(att_rows, lpos_rows):
    """Build the (N,16) odom-format array from LOCAL_POSITION_NED positions
    and ATTITUDE eulers on the exact sim clock, per clock segment.

    Sim convention (measured): ATTITUDE roll and pitch are sign-inverted vs
    physical; yaw is correct. True attitude = ZYX euler(yaw, -pitch, -roll).
    """
    from scipy.spatial.transform import Rotation

    att = np.array(att_rows, dtype=np.float64)
    lpos = np.array(lpos_rows, dtype=np.float64)
    if len(att) < 5 or len(lpos) < 5:
        return np.zeros((0, 16))

    att_segs = split_on_clock_reset(att)
    out_rows = []
    for lp in split_on_clock_reset(lpos):
        w0, w1 = lp[0, 7], lp[-1, 7]
        best = None
        for a in att_segs:
            overlap = min(w1, a[-1, 4]) - max(w0, a[0, 4])
            if best is None or overlap > best[0]:
                best = (overlap, a)
        if best is None or best[0] <= 0:
            continue
        a = best[1]
        # dedupe repeated timestamps within the segment
        a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
        lp = lp[np.concatenate([[True], np.diff(lp[:, 0]) > 0])]
        t = lp[:, 0]
        # slerp attitude to position times (euler lerp bends the rotation
        # path during aggressive maneuvers), then extract eulers again so the
        # sign-convention hypothesis can still be applied downstream
        rot_a = Rotation.from_euler("ZYX", a[:, [3, 2, 1]])
        from scipy.spatial.transform import Slerp
        slerp = Slerp(a[:, 0], rot_a)
        tq = np.clip(t, a[0, 0], a[-1, 0])
        eul = slerp(tq).as_euler("ZYX")
        yaw, pitch, roll = eul[:, 0], eul[:, 1], eul[:, 2]
        seg = np.zeros((len(t), 16))
        seg[:, 0] = t
        seg[:, 1:4] = lp[:, 1:4]
        # store RAW interpolated eulers in cols 11:14 (roll, pitch, yaw);
        # quats are built later once the sign convention is chosen
        seg[:, 11] = roll
        seg[:, 12] = pitch
        seg[:, 13] = yaw
        seg[:, 8:11] = lp[:, 4:7]         # world-frame velocity
        seg[:, 15] = lp[:, 7]
        out_rows.append(seg)
    if not out_rows:
        return np.zeros((0, 16))
    out = np.concatenate(out_rows, axis=0)
    apply_euler_signs(out, -1.0, -1.0, 1.0)  # measured default (3385 build)
    return out


def apply_euler_signs(odom, rs, ps, ys):
    """(Re)build quaternion columns 4:8 from the raw eulers in cols 11:14
    under a given sign convention. Mutates and returns odom."""
    from scipy.spatial.transform import Rotation

    if len(odom) == 0:
        return odom
    roll = rs * odom[:, 11]
    pitch = ps * odom[:, 12]
    yaw = ys * odom[:, 13]
    q_xyzw = Rotation.from_euler(
        "ZYX", np.column_stack([yaw, pitch, roll])).as_quat()
    odom[:, 4] = q_xyzw[:, 3]
    odom[:, 5:8] = q_xyzw[:, 0:3]
    return odom


def load_logger_episode(ep_dir):
    ep = Path(ep_dir)
    tel = np.load(ep / "telemetry.npz")
    odom = tel["odom"]
    gates = json.loads((ep / "gates.json").read_text()) if (ep / "gates.json").exists() else []
    frames = []
    with open(ep / "frames_index.csv") as fh:
        for row in csv.DictReader(fh):
            sim_ns = int(row["sim_time_ns"])
            fid = int(row["frame_id"])
            frames.append((fid, sim_ns, int(row["wall_ns"]),
                           str(ep / "frames" / f"{sim_ns}_{fid}.jpg")))
    frames.sort(key=lambda f: f[1])
    return {"odom": odom, "frames": frames, "gates": gates, "meta": {},
            "dir": str(ep)}


def load_episode_auto(ep_dir):
    ep = Path(ep_dir)
    if (ep / "mav.jsonl").exists():
        return load_rc_episode(ep)
    return load_logger_episode(ep)


# ---------------------------------------------------------------------------
# STRICT training loader — Henry's directive (2026-07-23):
# use ONLY episodes with native ODOMETRY time_usec, and ONLY the VQ1 course.
# Everything else is rejected outright.
# ---------------------------------------------------------------------------

VQ1_GATE0_POS = np.array([-23.298, -0.400, -0.032])


def is_vq1_map(gates):
    if not gates:
        return False
    return np.linalg.norm(np.asarray(gates[0]["pos"]) - VQ1_GATE0_POS) < 1.0


def load_training_episode(ep_dir):
    """Load an episode for any calibration/dataset/training use.

    Raises ValueError for: rc episodes without native ODOMETRY time_usec
    (old recorder format) or episodes whose winning gate map is not VQ1.
    Episodes that additionally saw a stray non-VQ1 map broadcast are kept —
    consumers must filter per-frame (drone must be within the VQ1 course
    region, see is_on_vq1_course)."""
    b = load_episode_auto(ep_dir)
    is_rc = (Path(b["dir"]) / "mav.jsonl").exists()
    if is_rc and not b.get("native_quat"):
        raise ValueError(f"EXCLUDED (old format, no ODOMETRY time_usec): {ep_dir}")
    if not is_vq1_map(b["gates"]):
        raise ValueError(f"EXCLUDED (gate map is not VQ1): {ep_dir}")
    return b


def is_on_vq1_course(pos, gates, max_dist=60.0):
    """Per-frame guard: drone position within max_dist of some VQ1 gate.
    Frames flown on another course (e.g. the VQ2 sky layout) fail this."""
    p = np.asarray(pos)
    for g in gates:
        if np.linalg.norm(p - np.asarray(g["pos"])) < max_dist:
            return True
    return False


def usable_training_episodes(root):
    """All strictly-usable episodes under a captures root, sorted by name."""
    out = []
    for ep in sorted(Path(root).iterdir()):
        if not ep.is_dir() or not ep.name.startswith("rc_"):
            continue
        try:
            load_training_episode(ep)
            out.append(ep)
        except (ValueError, FileNotFoundError, KeyError):
            continue
    return out
