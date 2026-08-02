"""Render a recorded VQ2 episode with detector/EKF diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from render_g9fix_video import (
    draw_gate,
    episode_rotations,
    gate_screen,
    load_imu,
)

from aigp.vision.labels import load_calib
from aigp.vq2_map import gate_quads_world_vq2


def put(image, text, xy, color=(235, 235, 235), scale=0.43):
    cv2.putText(
        image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
        cv2.LINE_AA,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=5)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    calib = load_calib(Path(__file__).resolve().parents[1] / "data/calib/calib.json")
    r_cb = np.asarray(calib["R_cb"], float)
    intrinsics = calib["K"]
    map_payload = json.loads(args.map.read_text())
    gates = map_payload["gates"]
    gate_positions = np.asarray([gate["pos"] for gate in gates[:17]], float)
    quads = [np.vstack(gate_quads_world_vq2(gate)) for gate in gates[:17]]

    rows = [
        row for row in map(json.loads, (args.run_dir / "steps.jsonl").open())
        if row["episode"] == args.episode
    ]
    episode = np.load(args.run_dir / f"episode_{args.episode:04d}.npz")
    observations = episode["observation"]
    if len(rows) != len(observations):
        raise ValueError(f"step/observation mismatch: {len(rows)} != {len(observations)}")
    step_sim = np.asarray([row["sim_time_s"] for row in rows])
    step_pos = np.asarray([row["position"] for row in rows])
    keep = np.r_[True, np.diff(step_sim) > 1e-6]
    rotation = Rotation.from_matrix(episode_rotations(observations))
    slerp = Slerp(step_sim[keep], rotation[keep])

    imu = load_imu(args.session)
    cuts = np.flatnonzero(np.diff(imu[:, 0]) < -1e5)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts + 1, len(imu)]
    segment = None
    for start, end in zip(starts, ends):
        if end - start < 200:
            continue
        sim = imu[start:end, 0] * 1e-6
        if sim[0] <= step_sim[0] + 0.5 and abs(sim[-1] - step_sim[-1]) < 1.2:
            segment = (int(start), int(end))
            break
    if segment is None:
        raise RuntimeError("episode IMU segment not found")
    start, end = segment
    imu_sim = imu[start:end, 0] * 1e-6
    imu_wall = imu[start:end, 1]

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

    debug = {}
    with (args.session / "localizer_debug/debug.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            payload = record["debug"]
            debug[int(payload.get("frame_id", record["frame_id"]))] = payload
    debug_ids = np.asarray(sorted(debug), dtype=np.int64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (960, 360)
    )
    drawn = 0
    xy_min = gate_positions[:, :2].min(0) - 5.0
    xy_span = np.maximum(gate_positions[:, :2].max(0) - xy_min + 5.0, 1.0)

    for wall, relpath, frame_id in frames:
        camera = cv2.imread(str(args.session / relpath))
        if camera is None:
            continue
        sim = float(np.interp(wall, imu_wall, imu_sim))
        if not step_sim[0] <= sim <= step_sim[-1]:
            continue
        index = min(int(np.searchsorted(step_sim, sim)), len(rows) - 1)
        row = rows[index]
        pos = np.asarray([
            np.interp(sim, step_sim, step_pos[:, axis]) for axis in range(3)
        ])
        world_rotation = slerp(np.clip(sim, step_sim[0], step_sim[-1])).as_matrix()

        target = int(row["target"])
        for gate_index, quad in enumerate(quads):
            uv = gate_screen(quad, pos, world_rotation, r_cb, intrinsics)
            if uv is not None:
                color = (50, 230, 70) if gate_index == target else (40, 110, 40)
                draw_gate(camera, uv, color, 2 if gate_index == target else 1)

        payload = None
        if len(debug_ids):
            nearest = int(debug_ids[np.argmin(np.abs(debug_ids - frame_id))])
            if abs(nearest - frame_id) <= 4:
                payload = debug[nearest]
                for match in payload.get("matches", []):
                    if not match.get("fused", False):
                        continue
                    observed = tuple(np.rint(match["observed"]).astype(int))
                    predicted = tuple(np.rint(match["predicted"]).astype(int))
                    cv2.line(camera, predicted, observed, (0, 180, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(
                        camera, observed, (0, 230, 255), cv2.MARKER_CROSS, 9, 2
                    )

        canvas = np.zeros((360, 960, 3), np.uint8)
        canvas[:, :640] = camera
        canvas[:, 640:] = (24, 24, 28)
        cv2.rectangle(canvas, (0, 0), (640, 20), (0, 0, 0), -1)
        put(
            canvas,
            f"V13 live | t={sim-step_sim[0]:5.2f}s | target={target} | speed={row['speed']:.1f} m/s",
            (7, 15),
        )

        put(canvas, "V13 + MULTIGATE EKF", (655, 23), (80, 220, 255), 0.48)
        put(canvas, f"position X  {pos[0]:+7.2f} m", (655, 52))
        put(canvas, f"position Y  {pos[1]:+7.2f} m", (655, 72))
        put(canvas, f"position Z  {pos[2]:+7.2f} m", (655, 92))
        put(canvas, f"sigma       {row['position_sigma_m']:.3f} m", (655, 116))
        put(canvas, f"landmark age {row['visual_age_s']:.3f} s", (655, 136))
        put(canvas, f"source  {row['localizer_source']}", (655, 156), (190, 210, 255))
        put(canvas, f"inference {row['vision_inference_ms']:.1f} ms", (655, 176))
        if payload is not None:
            multigate = payload.get("multigate", {})
            put(canvas, f"matched corners {payload.get('fused', 0)}", (655, 196))
            put(canvas, f"accepted gates {multigate.get('accepted_gates', [])}", (655, 216))
            put(canvas, f"attitude update {payload.get('attitude_updated', False)}", (655, 236))
        else:
            put(canvas, "detector: between 10 Hz frames", (655, 205), (150, 150, 150))

        # Compact top-down course position.
        x0, y0, width, height = 660, 252, 275, 90
        cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (55, 55, 62), 1)
        def map_xy(point):
            norm = (np.asarray(point[:2]) - xy_min) / xy_span
            return int(x0 + 4 + norm[0] * (width - 8)), int(y0 + height - 4 - norm[1] * (height - 8))
        course_points = [map_xy(point) for point in gate_positions]
        for first, second in zip(course_points[:-1], course_points[1:]):
            cv2.line(canvas, first, second, (65, 65, 75), 1, cv2.LINE_AA)
        for gate_index, point in enumerate(course_points):
            color = (0, 150, 255) if gate_index == target else (110, 110, 120)
            cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, map_xy(pos), 5, (255, 160, 40), -1, cv2.LINE_AA)
        put(canvas, "top-down course / drone", (665, 350), (175, 175, 185), 0.38)
        put(canvas, "GREEN=EKF gate  YELLOW=v13 match", (8, 354), (230, 230, 230), 0.40)

        writer.write(canvas)
        drawn += 1

    writer.release()
    print(f"wrote {args.out} ({drawn} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
