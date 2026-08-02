"""Replay a raw full-record session episode through the live localizer.

Feeds recorded camera frames and HIGHRES_IMU through
LiveVQ2Localizer.update_async at true wall-clock pacing, so the async
dense worker and the opt-in crop tracker (AIGP_CROP_TRACKER=1) run with
production timing.  Scores landmark age, update sources, and belief
distance to the in-run recorded belief -- all offline, zero sim time.

Usage:
    python scripts/replay_raw_session_bench.py --episode-index 32
    AIGP_CROP_TRACKER=1 python scripts/replay_raw_session_bench.py ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.vq2_live_localizer import LiveVQ2Localizer  # noqa: E402


def gate_crossing_offset(
    gate: dict,
    position_world: np.ndarray,
) -> dict[str, float]:
    """Express an estimated crossing position in the gate aperture frame."""
    qw, qx, qy, qz = np.asarray(gate["quat_wxyz"], float)
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    local = rotation.T @ (
        np.asarray(position_world, float)
        - np.asarray(gate["pos"], float)
    )
    return {
        "lateral_m": float(local[0]),
        "plane_m": float(local[1]),
        "vertical_m": float(local[2]),
    }


class ReplayMavlink:
    def __init__(self):
        self.imu = deque(maxlen=200_000)


class ReplayVision:
    latest = None


def load_imu_cache(session: Path, cache: Path) -> np.ndarray:
    """(time_usec, ax, ay, az, gx, gy, gz, wall_ns) rows, parsed once.

    mavlink_rx.bin records are ``<QHI`` (wall_ns, kind_len, raw_len) +
    kind string + one raw MAVLink v2 packet (full_session_recorder).
    """
    if cache.exists():
        return np.load(cache)["imu"]
    import struct

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
            raw = data[start:start + raw_len]
            messages = parser.parse_buffer(raw) or []
            for message in messages:
                if message.get_type() == "HIGHRES_IMU":
                    rows.append((
                        float(message.time_usec),
                        float(message.xacc), float(message.yacc),
                        float(message.zacc),
                        float(message.xgyro), float(message.ygyro),
                        float(message.zgyro),
                        float(wall_ns),
                    ))
    imu = np.asarray(rows, np.float64)
    np.savez_compressed(cache, imu=imu)
    return imu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session", type=Path,
        default=Path(r"D:\ai-gp\raw_sessions\vq2_20260730_172300"),
    )
    parser.add_argument(
        "--run-dir", type=Path,
        default=Path(
            r"D:\ai-gp\training\vq2_sac_runs"
            r"\gate7_midpoint_fullrecord_v70\20260730_172300"
        ),
    )
    parser.add_argument("--episode-index", type=int, default=32)
    parser.add_argument(
        "--map", type=Path,
        default=REPO / "data/vq2_runtime_map_gift_v11sparse10hz_ep13.json",
    )
    parser.add_argument(
        "--primary", type=Path,
        default=REPO / "data/models/gatenet_v7_best.pt",
    )
    parser.add_argument(
        "--refiner", type=Path,
        default=REPO / "data/models/gatenet_v10strict_ep0.pt",
    )
    parser.add_argument(
        "--gate-primary", type=Path, default=None,
        help="Optional primary detector used only for --gate-primary-gates.",
    )
    parser.add_argument(
        "--gate-primary-gates", default="",
        help="Comma-separated active gates that use --gate-primary.",
    )
    parser.add_argument(
        "--crop", type=Path,
        default=REPO / "data/models/crop_gatenet_v11crop_ep13.pt",
    )
    parser.add_argument(
        "--proposal", type=Path,
        default=REPO / "data/models/gatepose_v5vq2b_best.pt",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=REPO / "data/calib/calib.json",
    )
    parser.add_argument("--vision-hz", type=float, default=3.0)
    parser.add_argument("--vision-device", default="cpu")
    parser.add_argument(
        "--crop-tracker-gates",
        default="",
        help="Optional comma-separated crop-tracker target-gate allowlist.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scratch = Path(__file__).resolve().parent.parent / "data"
    imu_all = load_imu_cache(
        args.session, scratch / f"imu_cache_{args.session.name}.npz"
    )
    print(f"IMU rows: {len(imu_all)}")

    frames = []
    with (args.session / "frames_index.csv").open() as stream:
        for row in csv.DictReader(stream):
            frames.append((
                int(row["frame_id"]), int(row["sim_time_ns"]),
                int(row["wall_ns"]), row["path"],
            ))
    # De-duplicate repeated frame writes (same frame_id + sim_time).
    seen = set()
    unique_frames = []
    for frame in frames:
        key = (frame[0], frame[1])
        if key in seen:
            continue
        seen.add(key)
        unique_frames.append(frame)
    frames = unique_frames

    # Episode segmentation: the sim clock in HIGHRES_IMU restarts on every
    # hard reset (more segments than episodes because countdown retries
    # also reset).  Match steps.jsonl episodes to IMU segments greedily by
    # sim-time coverage, in wall order.
    imu_sim = imu_all[:, 0] * 1e-6
    imu_wall = imu_all[:, 7]
    reset_rows = np.flatnonzero(np.diff(imu_all[:, 0]) < -1e5)
    seg_starts = np.concatenate([[0], reset_rows + 1])
    seg_ends = np.concatenate([reset_rows + 1, [len(imu_all)]])
    imu_segments = [
        (int(start), int(end))
        for start, end in zip(seg_starts, seg_ends)
        if end - start > 200
    ]
    steps = [
        json.loads(line)
        for line in (args.run_dir / "steps.jsonl").open()
    ]
    per_episode: dict[int, list[dict]] = {}
    for step in steps:
        per_episode.setdefault(step["episode"], []).append(step)
    matches: dict[int, tuple[int, int]] = {}
    segment_cursor = 0
    for episode_index in sorted(per_episode):
        rows = per_episode[episode_index]
        first_sim = float(rows[0]["sim_time_s"])
        last_sim = float(rows[-1]["sim_time_s"])
        while segment_cursor < len(imu_segments):
            start, end = imu_segments[segment_cursor]
            segment_cursor += 1
            if (
                imu_sim[start] <= first_sim + 0.5
                and abs(imu_sim[end - 1] - last_sim) < 1.2
            ):
                matches[episode_index] = (start, end)
                break
    print(
        f"IMU segments: {len(imu_segments)}, matched episodes: "
        f"{len(matches)}/{len(per_episode)}"
    )
    if args.episode_index not in matches:
        print(f"episode {args.episode_index} not matched to an IMU segment")
        return 2
    episode_steps = per_episode[args.episode_index]
    start, end = matches[args.episode_index]
    imu_rows = imu_all[start:end]
    wall_low = float(imu_wall[start])
    wall_high = float(imu_wall[end - 1])
    segment_frames = [
        frame for frame in frames if wall_low <= frame[2] <= wall_high
    ]
    frame_sim = np.interp(
        np.asarray([frame[2] for frame in segment_frames], float),
        imu_wall[start:end],
        imu_sim[start:end],
    )
    print(
        f"episode {args.episode_index}: {len(segment_frames)} frames, "
        f"{(wall_high - wall_low) * 1e-9:.1f}s, "
        f"{len(episode_steps)} steps, "
        f"deepest gate {max(s['target'] for s in episode_steps)}"
    )

    step_sim = np.asarray([s["sim_time_s"] for s in episode_steps])
    step_pos = np.asarray([s["position"] for s in episode_steps])
    step_gate = np.asarray([s["target"] for s in episode_steps])
    recorded_age = np.asarray([s["visual_age_s"] for s in episode_steps])

    mavlink = ReplayMavlink()
    vision = ReplayVision()
    localizer = LiveVQ2Localizer(
        mavlink=mavlink,
        vision=vision,
        map_path=args.map,
        primary_checkpoint=args.primary,
        refine_checkpoint=args.refiner,
        gate_primary_checkpoint=args.gate_primary,
        gate_primary_gates=tuple(
            int(value.strip())
            for value in args.gate_primary_gates.split(",")
            if value.strip()
        ),
        crop_checkpoint=args.crop,
        proposal_checkpoint=args.proposal,
        calibration_path=args.calibration,
        async_interval_s=1.0 / args.vision_hz,
        dense_device=args.vision_device,
        dense_process_isolation=False,
        crop_track_gates=(
            tuple(
                int(value.strip())
                for value in args.crop_tracker_gates.split(",")
                if value.strip()
            )
            if args.crop_tracker_gates.strip()
            else None
        ),
    )
    print(f"crop tracker enabled: {localizer.crop_track_enabled}")

    # ---- anchor on the countdown frames (drone parked, gate 0 ahead) ----
    launch_sim_s = float(episode_steps[0]["sim_time_s"])
    countdown = [
        frame for frame, sim in zip(segment_frames, frame_sim)
        if sim < launch_sim_s - 0.15
    ]
    countdown_imu = imu_rows[
        imu_rows[:, 0] * 1e-6 < launch_sim_s - 0.15
    ]
    if len(countdown) < 5 or len(countdown_imu) < 20:
        print("not enough countdown data to anchor")
        return 2
    feeder_stop = threading.Event()

    def feed_anchor():
        index = 0
        while not feeder_stop.is_set():
            frame = countdown[index % len(countdown)]
            vision.latest = (
                1_000_000 + index,
                frame[1],
                (args.session / frame[3]).read_bytes(),
                time.time_ns(),
            )
            row = countdown_imu[index % len(countdown_imu)]
            mavlink.imu.append((*row[:7], time.time_ns()))
            index += 1
            time.sleep(0.01)

    anchor_thread = threading.Thread(target=feed_anchor, daemon=True)
    anchor_thread.start()
    try:
        localizer.initialize(timeout_s=4.0)
    finally:
        feeder_stop.set()
        anchor_thread.join(timeout=0.5)
    diag = localizer.anchor_diagnostics
    print(
        "anchor ok: spread "
        f"{diag.get('translation_spread_p90_m', -1):.3f}m, "
        f"gate0 err {diag.get('gate0_visual_error_m', -1):.3f}m"
    )

    # ---- real-time replay through update_async ----
    mavlink.imu.clear()
    flight_imu = imu_rows[imu_rows[:, 0] * 1e-6 >= launch_sim_s - 0.15]
    flight_frames = [
        frame for frame, sim in zip(segment_frames, frame_sim)
        if sim >= launch_sim_s - 0.15
    ]
    localizer.last_imu_us = int(flight_imu[0][0])
    localizer.ekf.t = flight_imu[0][0] * 1e-6
    localizer.start_async()

    replay_started_ns = time.time_ns()
    wall_base = int(flight_frames[0][2])

    def feed_flight():
        imu_cursor = 0
        frame_cursor = 0
        while not feeder_stop.is_set() and (
            imu_cursor < len(flight_imu)
            or frame_cursor < len(flight_frames)
        ):
            now_rel = time.time_ns() - replay_started_ns
            while (
                imu_cursor < len(flight_imu)
                and flight_imu[imu_cursor][7] - wall_base <= now_rel
            ):
                row = flight_imu[imu_cursor]
                mavlink.imu.append((*row[:7], time.time_ns()))
                imu_cursor += 1
            while (
                frame_cursor < len(flight_frames)
                and flight_frames[frame_cursor][2] - wall_base <= now_rel
            ):
                frame = flight_frames[frame_cursor]
                vision.latest = (
                    frame[0],
                    frame[1],
                    (args.session / frame[3]).read_bytes(),
                    replay_started_ns + (frame[2] - wall_base),
                )
                frame_cursor += 1
            time.sleep(0.002)

    feeder_stop.clear()
    flight_thread = threading.Thread(target=feed_flight, daemon=True)
    flight_thread.start()

    duration_s = (wall_high - wall_base) * 1e-9
    log = []
    while True:
        elapsed = (time.time_ns() - replay_started_ns) * 1e-9
        if elapsed > duration_s + 0.2:
            break
        sim_now = flight_imu[0][0] * 1e-6 + elapsed
        gate = int(step_gate[min(
            np.searchsorted(step_sim, sim_now), len(step_gate) - 1
        )])
        state = localizer.update_async(min(gate, 16))
        log.append({
            "t": elapsed,
            "sim": sim_now,
            "p": state.position.tolist(),
            "sig": state.position_sigma_m,
            "age": state.visual_age_s,
            "fused": state.corners_fused,
            "src": localizer.last_update_source,
            "gate": gate,
        })
        time.sleep(max(0.0, 1.0 / 30.0 - 0.002))
    feeder_stop.set()
    flight_thread.join(timeout=1.0)
    localizer.stop_async()

    # ---- score ----
    ages = np.asarray([row["age"] for row in log])
    positions = np.asarray([row["p"] for row in log])
    sims = np.asarray([row["sim"] for row in log])
    in_window = sims <= step_sim[-1]
    reference = np.vstack([
        np.interp(sims, step_sim, step_pos[:, axis]) for axis in range(3)
    ]).T
    err = np.linalg.norm(positions - reference, axis=1)
    map_payload = json.loads(args.map.read_text())
    gate_map = map_payload["gates"]
    crossing_offsets = []
    for step_index in range(1, len(episode_steps)):
        previous_target = int(episode_steps[step_index - 1]["target"])
        target = int(episode_steps[step_index]["target"])
        if target != previous_target + 1 or previous_target >= len(gate_map):
            continue
        crossing_sim = float(episode_steps[step_index]["sim_time_s"])
        replay_index = int(np.argmin(np.abs(sims - crossing_sim)))
        crossing_offsets.append({
            "gate": previous_target,
            "sim_time_s": crossing_sim,
            **gate_crossing_offset(
                gate_map[previous_target],
                positions[replay_index],
            ),
        })
    print("\n==== replay result ====")
    print(f"mode: crop_tracker={'ON' if localizer.crop_track_enabled else 'OFF'}")
    print(
        f"landmark age s: p50={np.percentile(ages, 50):.3f} "
        f"p90={np.percentile(ages, 90):.3f} max={ages.max():.3f}"
    )
    print(
        f"recorded in-run age: p50={np.percentile(recorded_age, 50):.3f} "
        f"p90={np.percentile(recorded_age, 90):.3f} "
        f"max={recorded_age.max():.3f}"
    )
    print(
        f"belief vs in-run belief m (sanity, not truth): "
        f"p50={np.percentile(err[in_window], 50):.3f} "
        f"p90={np.percentile(err[in_window], 90):.3f}"
    )
    print("update counts:", localizer.update_counts)
    print("replay crossing offsets:", crossing_offsets)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "crop_tracker": localizer.crop_track_enabled,
            "age_p50": float(np.percentile(ages, 50)),
            "age_p90": float(np.percentile(ages, 90)),
            "age_max": float(ages.max()),
            "err_p50": float(np.percentile(err[in_window], 50)),
            "err_p90": float(np.percentile(err[in_window], 90)),
            "counts": localizer.update_counts,
            "crossing_offsets": crossing_offsets,
            "log": log,
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
