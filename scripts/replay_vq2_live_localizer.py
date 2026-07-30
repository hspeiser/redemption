"""Replay the causal live localizer against a verified offline VQ2 trace."""

from __future__ import annotations

import argparse
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


class ReplayMavlink:
    def __init__(self):
        self.imu = deque(maxlen=100_000)


class ReplayVision:
    latest = None


def read_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=Path,
        default=REPO / "data/ep_rc_20260729_000036",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=REPO /
        "data/vq2_trace_gift_v11sparse10hz_ep13.npz",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=REPO /
        "data/vq2_runtime_map_gift_v11sparse10hz_ep13.json",
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=REPO / "data/models/gatenet_v7_best.pt",
    )
    parser.add_argument(
        "--refiner",
        type=Path,
        default=REPO / "data/models/gatenet_v10strict_ep0.pt",
    )
    parser.add_argument(
        "--crop",
        type=Path,
        default=REPO / "data/models/crop_gatenet_v11crop_ep13.pt",
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        default=REPO / "data/models/gatepose_v5vq2b_best.pt",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO / "data/calib/calib.json",
    )
    parser.add_argument("--anchor-seconds", type=float, default=3.0)
    parser.add_argument("--anchor-only", action="store_true")
    args = parser.parse_args()

    imu = []
    for row in read_jsonl(args.episode / "imu.jsonl"):
        imu.append((
            int(row["time_usec"]),
            *map(float, row["accel"]),
            *map(float, row["gyro"]),
            float(row["wall"]),
        ))
    clocks = np.asarray([row[0] for row in imu], np.int64)
    breaks = np.flatnonzero(np.diff(clocks) < -500_000)
    if len(breaks):
        imu = imu[int(breaks[-1] + 1):]
    clock_start = imu[0][0]
    imu_time = np.asarray([
        (row[0] - clock_start) * 1e-6 for row in imu
    ])
    imu_wall = np.asarray([row[-1] for row in imu])

    frame_rows = {
        int(row["idx"]): row
        for row in read_jsonl(args.episode / "frames.jsonl")
    }
    trace = np.load(args.trace, allow_pickle=False)
    frame_indices = np.asarray([
        int(Path(str(path).replace("\\", "/")).stem)
        for path in trace["path"]
    ])
    frame_wall = np.asarray([
        frame_rows[int(index)]["wall"] for index in frame_indices
    ])
    frame_time = np.interp(frame_wall, imu_wall, imu_time)

    race = [
        row for row in read_jsonl(args.episode / "mav.jsonl")
        if row.get("kind") == "race_status"
    ]
    race_wall = np.asarray([row["wall"] for row in race])
    race_gate = np.asarray([row["active_gate"] for row in race])
    order = np.argsort(race_wall, kind="stable")
    race_wall, race_gate = race_wall[order], race_gate[order]
    frame_gate = race_gate[np.clip(
        np.searchsorted(race_wall, frame_wall, side="right") - 1,
        0,
        len(race_gate) - 1,
    )]

    mavlink = ReplayMavlink()
    vision = ReplayVision()
    localizer = LiveVQ2Localizer(
        mavlink=mavlink,
        vision=vision,
        map_path=args.map,
        primary_checkpoint=args.primary,
        refine_checkpoint=args.refiner,
        crop_checkpoint=args.crop,
        proposal_checkpoint=args.proposal,
        calibration_path=args.calibration,
    )
    static_frames = [
        (index, path)
        for index, path, stamp in zip(
            frame_indices, trace["path"], frame_time
        )
        if 0.4 < stamp < args.anchor_seconds
    ]
    static_imu = [
        row for row, stamp in zip(imu, imu_time)
        if 0.4 < stamp < args.anchor_seconds
    ]
    feeder_stop = False

    def feed_anchor():
        index = 0
        while not feeder_stop:
            frame_index, path = static_frames[index % len(static_frames)]
            vision.latest = (
                100_000 + index,
                time.time_ns(),
                Path(str(path)).read_bytes(),
                time.time_ns(),
            )
            row = static_imu[index % len(static_imu)]
            mavlink.imu.append((*row[:7], time.time_ns()))
            index += 1
            time.sleep(0.008)

    feeder = threading.Thread(target=feed_anchor, daemon=True)
    feeder.start()
    localizer.initialize(timeout_s=2.8)
    feeder_stop = True
    feeder.join(timeout=0.2)
    print("anchor", json.dumps(localizer.anchor_diagnostics))
    if args.anchor_only:
        return 0

    # The synthetic anchor feeder loops timestamps.  Resume at the exact
    # recorded clock immediately after the parked interval.
    mavlink.imu.clear()
    imu_cursor = int(np.searchsorted(
        imu_time, args.anchor_seconds, side="right"
    ))
    previous_imu = imu[imu_cursor - 1]
    mavlink.imu.append((*previous_imu[:7], time.time_ns()))
    localizer.last_imu_us = int(previous_imu[0])
    localizer.ekf.t = previous_imu[0] * 1e-6

    error = []
    sigma = []
    fused = []
    gates = []
    target_error = []
    sources = []
    replay_times = []
    target_distances = []
    template_positions = np.asarray([
        gate["pos"] for gate in localizer.map_template[:17]
    ])
    live_positions = np.asarray([
        gate["pos"] for gate in localizer.gates[:17]
    ])
    for trace_row, (
        frame_index, path, stamp, gate,
    ) in enumerate(zip(
        frame_indices, trace["path"], frame_time, frame_gate
    )):
        if stamp <= args.anchor_seconds or gate > 16:
            continue
        while (
            imu_cursor < len(imu)
            and imu_time[imu_cursor] <= stamp
        ):
            row = imu[imu_cursor]
            mavlink.imu.append((*row[:7], time.time_ns()))
            imu_cursor += 1
        vision.latest = (
            int(frame_index),
            int(stamp * 1e9),
            Path(str(path)).read_bytes(),
            time.time_ns(),
        )
        state = localizer.update(int(gate))
        error.append(float(np.linalg.norm(
            state.position - trace["pos"][trace_row]
        )))
        ql = state.quat_wxyz
        qt = trace["quat"][trace_row]
        live_rotation = Rotation.from_quat(
            [ql[1], ql[2], ql[3], ql[0]]
        ).as_matrix()
        trace_rotation = Rotation.from_quat(
            [qt[1], qt[2], qt[3], qt[0]]
        ).as_matrix()
        live_relative = live_rotation.T @ (
            live_positions[int(gate)] - state.position
        )
        trace_relative = trace_rotation.T @ (
            template_positions[int(gate)] - trace["pos"][trace_row]
        )
        target_error.append(float(np.linalg.norm(
            live_relative - trace_relative
        )))
        target_distances.append(float(np.linalg.norm(trace_relative)))
        sigma.append(state.position_sigma_m)
        fused.append(state.corners_fused)
        gates.append(int(gate))
        sources.append(localizer.last_update_source)
        replay_times.append(float(stamp))

    error = np.asarray(error)
    sigma = np.asarray(sigma)
    fused = np.asarray(fused)
    gates = np.asarray(gates)
    sources = np.asarray(sources)
    replay_times = np.asarray(replay_times)
    target_distances = np.asarray(target_distances)
    target_error = np.asarray(target_error)
    print(
        "overall error median/p90/p99/max m",
        np.round(np.percentile(error, [50, 90, 99, 100]), 4),
    )
    print(
        "sigma median/p90/max m",
        np.round(np.percentile(sigma, [50, 90, 100]), 4),
        f"fused={100 * np.mean(fused > 0):.1f}%",
    )
    print(
        "next-gate body-relative error median/p90/p99/max m",
        np.round(np.percentile(
            target_error, [50, 90, 99, 100]
        ), 4),
    )
    print("update counts", localizer.update_counts)
    for gate in range(17):
        mask = gates == gate
        if mask.any():
            print(
                f"gate {gate:2d} n={int(mask.sum()):3d} "
                f"position med/p90="
                f"{np.round(np.percentile(error[mask], [50, 90]), 4)} "
                f"target med/p90="
                f"{np.round(np.percentile(target_error[mask], [50, 90]), 4)} "
                f"sources={dict(zip(*np.unique(sources[mask], return_counts=True)))}"
            )
            if gate in (9, 15, 16):
                rows = np.flatnonzero(mask)
                worst = rows[np.argsort(target_error[rows])[-8:]]
                print("  worst t/dist/error/source", [
                    (
                        round(float(replay_times[row]), 2),
                        round(float(target_distances[row]), 2),
                        round(float(target_error[row]), 3),
                        str(sources[row]),
                    )
                    for row in worst
                ])
    return 0 if np.percentile(target_error, 90) <= 0.40 else 2


if __name__ == "__main__":
    raise SystemExit(main())
