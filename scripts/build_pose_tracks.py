"""Build per-frame pose tracks for full-record sessions (label pipeline).

Position: interpolated belief from the run's steps.jsonl (9-16 cm in the
10Hz era).  Attitude: gravity-aligned at the parked spawn, then gyro
integration (the sim IMU is effectively noiseless).  Validation: project
map corners with these poses and compare against the localizer's own
'expected' projections stored in the debug archive.

Output per session: pose_tracks.npz {path, wall_ns, pos, quat_wxyz,
gate_idx, sigma, age} for frames inside healthy-tracking windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FX = FY = 320.0
CX, CY = 320.0, 180.0


def load_imu(session: Path) -> np.ndarray:
    from pymavlink import mavutil

    header = struct.Struct("<QHI")
    data = (session / "mavlink_rx.bin").read_bytes()
    parser = mavutil.mavlink.MAVLink(None)
    parser.robust_parsing = True
    rows = []
    with (session / "mavlink_index.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["message_type"] != "HIGHRES_IMU":
                continue
            offset = int(row["offset"])
            wall_ns, kind_len, raw_len = header.unpack_from(data, offset)
            start = offset + header.size + kind_len
            for message in parser.parse_buffer(
                data[start:start + raw_len]
            ) or []:
                if message.get_type() == "HIGHRES_IMU":
                    rows.append((
                        float(message.time_usec),
                        float(message.xacc), float(message.yacc),
                        float(message.zacc),
                        float(message.xgyro), float(message.ygyro),
                        float(message.zgyro),
                        float(wall_ns),
                    ))
    return np.asarray(rows, np.float64)


def episode_rotations(observations: np.ndarray) -> np.ndarray:
    """Recover belief rotation matrices from 53-D observations.

    build_observation stores rotation[:, 0] at obs[21:24] and
    rotation[:, 1] at obs[24:27]; the third column is their cross
    product.  This is the EKF attitude in its own convention -- no
    integration, no sign guessing.
    """
    c0 = observations[:, 21:24].astype(float)
    c1 = observations[:, 24:27].astype(float)
    c0 /= np.linalg.norm(c0, axis=1, keepdims=True) + 1e-9
    c1 -= np.sum(c0 * c1, axis=1, keepdims=True) * c0
    c1 /= np.linalg.norm(c1, axis=1, keepdims=True) + 1e-9
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--map", type=Path,
        default=REPO / "data/vq2_runtime_map_g9g15fix.json",
    )
    parser.add_argument("--calibration", type=Path,
                        default=REPO / "data/calib/calib.json")
    parser.add_argument("--max-sigma", type=float, default=0.20)
    parser.add_argument("--max-age", type=float, default=0.50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from aigp.vision.labels import load_calib
    calib = load_calib(args.calibration)
    R_cb = np.asarray(calib["R_cb"], float)

    imu_all = load_imu(args.session)
    steps = [json.loads(l) for l in (args.run_dir / "steps.jsonl").open()]
    per_ep: dict[int, list[dict]] = {}
    for s in steps:
        per_ep.setdefault(s["episode"], []).append(s)

    frames = []
    with (args.session / "frames_index.csv").open() as stream:
        seen = set()
        for row in csv.DictReader(stream):
            key = (row["frame_id"], row["sim_time_ns"])
            if key in seen:
                continue
            seen.add(key)
            frames.append((int(row["wall_ns"]), row["path"]))
    frame_wall = np.asarray([f[0] for f in frames], float)

    # IMU segments (sim clock resets per hard reset)
    resets = np.flatnonzero(np.diff(imu_all[:, 0]) < -1e5)
    seg_b = np.concatenate([[0], resets + 1])
    seg_e = np.concatenate([resets + 1, [len(imu_all)]])
    segments = [(int(a), int(b)) for a, b in zip(seg_b, seg_e)
                if b - a > 200]

    out_rows = {k: [] for k in (
        "path", "wall_ns", "pos", "quat_wxyz", "gate_idx", "sigma", "age",
        "gyro_norm",
    )}
    cursor = 0
    matched = 0
    for ep in sorted(per_ep):
        rows = per_ep[ep]
        first_sim, last_sim = rows[0]["sim_time_s"], rows[-1]["sim_time_s"]
        seg = None
        while cursor < len(segments):
            a, b = segments[cursor]
            cursor += 1
            sim = imu_all[a:b, 0] * 1e-6
            if sim[0] <= first_sim + 0.5 and abs(sim[-1] - last_sim) < 1.2:
                seg = (a, b)
                break
        if seg is None:
            continue
        a, b = seg
        imu = imu_all[a:b]
        ep_npz = args.run_dir / f"episode_{ep:04d}.npz"
        if not ep_npz.exists():
            continue
        observations = np.load(ep_npz)["observation"]
        if len(observations) != len(rows):
            # steps and episode rows must align 1:1 for attitude lookup
            continue
        matched += 1
        step_rot = Rotation.from_matrix(episode_rotations(observations))
        imu_sim = imu[:, 0] * 1e-6
        imu_wall = imu[:, 7]

        step_sim = np.asarray([r["sim_time_s"] for r in rows])
        step_pos = np.asarray([r["position"] for r in rows])
        step_sig = np.asarray([r["position_sigma_m"] for r in rows])
        step_age = np.asarray([r["visual_age_s"] for r in rows])
        step_gate = np.asarray([r["target"] for r in rows])
        from scipy.spatial.transform import Slerp
        keep = np.concatenate([[True], np.diff(step_sim) > 1e-6])
        slerp = Slerp(step_sim[keep], step_rot[keep])

        mask = (frame_wall >= imu_wall[0]) & (frame_wall <= imu_wall[-1])
        for fi in np.flatnonzero(mask):
            wall, relpath = frames[fi]
            sim = float(np.interp(wall, imu_wall, imu_sim))
            if not step_sim[0] <= sim <= step_sim[-1]:
                continue
            si = int(np.searchsorted(step_sim, sim))
            si = min(si, len(rows) - 1)
            if step_sig[si] > args.max_sigma or step_age[si] > args.max_age:
                continue
            pos = np.array([
                np.interp(sim, step_sim, step_pos[:, k]) for k in range(3)
            ])
            q = slerp(np.clip(sim, step_sim[0], step_sim[-1])).as_quat()
            out_rows["path"].append(relpath)
            out_rows["wall_ns"].append(wall)
            out_rows["pos"].append(pos)
            out_rows["quat_wxyz"].append([q[3], q[0], q[1], q[2]])
            out_rows["gate_idx"].append(int(step_gate[si]))
            out_rows["sigma"].append(float(step_sig[si]))
            out_rows["age"].append(float(step_age[si]))
            ii = int(np.clip(
                np.searchsorted(imu_sim, sim), 0, len(imu) - 1
            ))
            out_rows["gyro_norm"].append(
                float(np.linalg.norm(imu[ii, 4:7]))
            )

    out = args.out or (args.session / "pose_tracks.npz")
    np.savez_compressed(
        out,
        path=np.array(out_rows["path"]),
        wall_ns=np.array(out_rows["wall_ns"], np.int64),
        pos=np.array(out_rows["pos"], np.float32),
        quat_wxyz=np.array(out_rows["quat_wxyz"], np.float32),
        gate_idx=np.array(out_rows["gate_idx"], np.int64),
        sigma=np.array(out_rows["sigma"], np.float32),
        age=np.array(out_rows["age"], np.float32),
        gyro_norm=np.array(out_rows["gyro_norm"], np.float32),
    )
    print(f"episodes matched: {matched}/{len(per_ep)}; "
          f"pose frames: {len(out_rows['path'])} -> {out}")

    # ---- validation against the localizer's own projections ----
    debug_path = args.session / "localizer_debug" / "debug.jsonl"
    if not debug_path.exists():
        return 0
    gates = json.loads(args.map.read_text())
    gate_list = gates["gates"] if isinstance(gates, dict) else gates
    wall_arr = np.array(out_rows["wall_ns"], np.int64)
    pos_arr = np.array(out_rows["pos"], np.float32)
    quat_arr = np.array(out_rows["quat_wxyz"], np.float32)
    errs = []
    from aigp.vq2_map import gate_quads_world_vq2
    quads = [
        np.vstack(gate_quads_world_vq2(gate)) for gate in gate_list[:17]
    ]
    for line in debug_path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = r["debug"]
        g = d.get("active_gate")
        exp = [e for e in d.get("expected", []) if e["gate"] == g]
        if len(exp) < 8 or g is None or g >= len(quads):
            continue
        i = int(np.searchsorted(wall_arr, r["wall_ns"]))
        if not 0 < i < len(wall_arr):
            continue
        if abs(wall_arr[i] - r["wall_ns"]) > 40e6:
            continue
        q = quat_arr[i]
        Rw = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        cam = (R_cb @ Rw.T @ (
            np.asarray(quads[g][:8]) - pos_arr[i]
        ).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.stack([
                FX * cam[:, 0] / cam[:, 2] + CX,
                FY * cam[:, 1] / cam[:, 2] + CY,
            ], axis=1)
        ref = np.array([e["pixel"] for e in exp])
        if np.all(np.isfinite(uv)) and np.all(cam[:, 2] > 0.3):
            errs.append(float(np.mean(
                np.linalg.norm(uv - ref[:len(uv)], axis=1)
            )))
    if errs:
        errs = np.array(errs)
        print(f"validation vs localizer projections: n={len(errs)} "
              f"p50={np.percentile(errs, 50):.1f}px "
              f"p90={np.percentile(errs, 90):.1f}px")
    else:
        print("validation: no comparable debug frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
