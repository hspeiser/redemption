"""Render a full-lap localizer-view video proving the gate-9/15 map fix.

Overlays on every recorded camera frame of a finished v77 lap:
  red   = original map projected through the belief pose
  green = corrected map (g9/g15 fix) projected through the same pose
  yellow crosses = corners the dense detector actually observed (from
  the localizer debug archive, ~10 Hz)

Everywhere the map is right, red and green coincide and sit on the
gate pixels.  At gate 9 the red frame floats ~0.5 m off the real gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.vision.labels import load_calib  # noqa: E402
from aigp.vq2_map import gate_quads_world_vq2  # noqa: E402


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
            for m in parser.parse_buffer(data[start:start + raw_len]) or []:
                if m.get_type() == "HIGHRES_IMU":
                    rows.append((float(m.time_usec), float(wall_ns)))
    return np.asarray(rows, np.float64)


def episode_rotations(observations: np.ndarray) -> np.ndarray:
    c0 = observations[:, 21:24].astype(float)
    c1 = observations[:, 24:27].astype(float)
    c0 /= np.linalg.norm(c0, axis=1, keepdims=True) + 1e-9
    c1 -= np.sum(c0 * c1, axis=1, keepdims=True) * c0
    c1 /= np.linalg.norm(c1, axis=1, keepdims=True) + 1e-9
    return np.stack([c0, c1, np.cross(c0, c1)], axis=2)


def gate_screen(quads, pos, Rw, R_cb, K):
    fx, fy, cx, cy = K
    cam = (R_cb @ Rw.T @ (quads - pos).T).T
    if np.any(cam[:, 2] < 0.3):
        return None
    uv = np.stack([
        fx * cam[:, 0] / cam[:, 2] + cx,
        fy * cam[:, 1] / cam[:, 2] + cy,
    ], axis=1)
    if not np.all(np.isfinite(uv)):
        return None
    if uv[:, 0].max() < -80 or uv[:, 0].min() > 720 \
            or uv[:, 1].max() < -80 or uv[:, 1].min() > 440:
        return None
    return uv.astype(np.int32)


def draw_gate(img, uv, color, thick=1):
    for ring in (uv[0:4], uv[4:8]):
        for i in range(4):
            cv2.line(img, tuple(ring[i]), tuple(ring[(i + 1) % 4]),
                     color, thick, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session", type=Path,
        default=Path(r"D:\ai-gp\raw_sessions\vq2_20260730_200627"))
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path(r"D:\ai-gp\training\vq2_sac_runs"
                     r"\gate10_left15_gpu10_awr_v77\20260730_200627"))
    parser.add_argument("--episode", type=int, default=3)
    parser.add_argument("--map-old", type=Path,
                        default=REPO /
                        "data/vq2_runtime_map_gift_v11sparse10hz_ep13.json")
    parser.add_argument("--map-new", type=Path,
                        default=REPO / "data/vq2_runtime_map_g9g15fix.json")
    parser.add_argument("--out", type=Path,
                        default=REPO / "vq2_g9fix_proof.mp4")
    args = parser.parse_args()

    calib = load_calib(REPO / "data/calib/calib.json")
    R_cb = np.asarray(calib["R_cb"], float)
    K = calib["K"]

    def load_quads(path):
        payload = json.loads(path.read_text())
        gates = payload["gates"] if isinstance(payload, dict) else payload
        return [np.vstack(gate_quads_world_vq2(g)) for g in gates[:17]]

    quads_old = load_quads(args.map_old)
    quads_new = load_quads(args.map_new)

    steps = [json.loads(l) for l in (args.run_dir / "steps.jsonl").open()
             ]
    rows = [s for s in steps if s["episode"] == args.episode]
    obs = np.load(
        args.run_dir / f"episode_{args.episode:04d}.npz"
    )["observation"]
    assert len(obs) == len(rows), (len(obs), len(rows))
    step_sim = np.asarray([s["sim_time_s"] for s in rows])
    step_pos = np.asarray([s["position"] for s in rows])
    keep = np.concatenate([[True], np.diff(step_sim) > 1e-6])
    rot = Rotation.from_matrix(episode_rotations(obs))
    slerp = Slerp(step_sim[keep], rot[keep])

    imu = load_imu(args.session)
    resets = np.flatnonzero(np.diff(imu[:, 0]) < -1e5)
    seg_b = np.concatenate([[0], resets + 1])
    seg_e = np.concatenate([resets + 1, [len(imu)]])
    segment = None
    for a, b in zip(seg_b, seg_e):
        if b - a < 200:
            continue
        sim = imu[a:b, 0] * 1e-6
        if sim[0] <= step_sim[0] + 0.5 and abs(sim[-1] - step_sim[-1]) < 1.2:
            segment = (int(a), int(b))
            break
    assert segment, "episode segment not found"
    a, b = segment
    imu_sim = imu[a:b, 0] * 1e-6
    imu_wall = imu[a:b, 1]

    frames = []
    seen = set()
    with (args.session / "frames_index.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (row["frame_id"], row["sim_time_ns"])
            if key in seen:
                continue
            seen.add(key)
            wall = int(row["wall_ns"])
            if imu_wall[0] <= wall <= imu_wall[-1]:
                frames.append((wall, row["path"], int(row["frame_id"])))
    print(f"episode frames: {len(frames)}")

    # detector observations by frame_id from the debug archive
    observed = {}
    for line in (args.session / "localizer_debug" / "debug.jsonl").open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        d = r["debug"]
        pts = [m["observed"] for m in d.get("matches", [])]
        pk = [p["pixel"] for p in d.get("peaks", [])]
        if pts or pk:
            observed[int(d.get("frame_id", r.get("frame_id", -1)))] = (
                pts, pk
            )

    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360)
    )
    n_drawn = 0
    for wall, relpath, frame_id in frames:
        img = cv2.imread(str(args.session / relpath))
        if img is None:
            continue
        sim = float(np.interp(wall, imu_wall, imu_sim))
        if not step_sim[0] <= sim <= step_sim[-1]:
            continue
        si = min(int(np.searchsorted(step_sim, sim)), len(rows) - 1)
        pos = np.array([
            np.interp(sim, step_sim, step_pos[:, k]) for k in range(3)
        ])
        Rw = slerp(np.clip(sim, step_sim[0], step_sim[-1])).as_matrix()

        for gi in range(17):
            uv_new = gate_screen(quads_new[gi], pos, Rw, R_cb, K)
            uv_old = gate_screen(quads_old[gi], pos, Rw, R_cb, K)
            if uv_old is not None and uv_new is not None \
                    and np.abs(uv_old - uv_new).max() > 2:
                draw_gate(img, uv_old, (60, 60, 230), 2)
            if uv_new is not None:
                highlight = gi in (9, 15)
                draw_gate(img, uv_new, (80, 220, 80),
                          2 if highlight else 1)
        if frame_id in observed:
            pts, pk = observed[frame_id]
            for (u, v) in pk:
                cv2.drawMarker(img, (int(u), int(v)), (0, 220, 255),
                               cv2.MARKER_CROSS, 7, 1)
        hud = (f"t={sim - step_sim[0]:5.1f}s  gate={rows[si]['target']:2d} "
               f"spd={rows[si]['speed']:4.1f}  age="
               f"{rows[si]['visual_age_s']:.2f}s")
        cv2.rectangle(img, (0, 0), (640, 18), (0, 0, 0), -1)
        cv2.putText(img, hud, (6, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        legend = "GREEN=fixed map  RED=old map (where different)  " \
                 "YELLOW=detector"
        cv2.rectangle(img, (0, 342), (640, 360), (0, 0, 0), -1)
        cv2.putText(img, legend, (6, 355), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
        n_drawn += 1
    writer.release()
    print(f"wrote {args.out} ({n_drawn} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
