"""Fixed-lag smoothing of a VQ2 pose trace using official gate-pass events.

The online filter can accumulate an almost constant velocity error during a
gate-to-gate leg.  At the end of that leg, the official active-gate increment
provides an authoritative position landmark: the drone is in the aperture of
the gate it just passed.  This tool distributes that endpoint correction
backward over only that leg.  It is intended for map verification and offline
label generation; it does not pretend that the future pass event was available
to a zero-latency live controller.
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
from aigp.vision.labels import load_calib  # noqa: E402
from scripts.vq2_align import load_imu, load_race_status  # noqa: E402

HOLE = 0.75
SQ_HOLE = np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE],
                    [HOLE, 0, HOLE], [-HOLE, 0, HOLE]])
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def pass_events(ep: Path):
    imu_all = load_imu(ep)
    breaks = np.where(np.diff(imu_all[:, 0]) < -0.5)[0]
    segments = np.split(np.arange(len(imu_all)), breaks + 1)
    imu = imu_all[max(segments, key=len)]
    t_start = float(imu[0, 0])
    t_lo, t_hi = t_start - 0.5, float(imu[-1, 0]) + 0.5

    wall_imu = []
    with open(ep / "imu.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if "time_usec" in r and "wall" in r:
                wall_imu.append((r["wall"], r["time_usec"] * 1e-6))
    wall_imu = np.asarray(wall_imu)

    race = load_race_status(ep)
    race_t = np.interp(
        [wall for wall, _active, _start in race],
        wall_imu[:, 0], wall_imu[:, 1])
    active = np.asarray([active for _wall, active, _start in race])
    valid = (race_t >= t_lo) & (race_t <= t_hi)
    race_t, active = race_t[valid], active[valid]
    first_zero = int(np.argmax(active == 0)) if (active == 0).any() else 0
    race_t, active = race_t[first_zero:], active[first_zero:]

    changes = np.where(np.diff(active) == 1)[0] + 1
    return [(float(race_t[k] - t_start), int(active[k] - 1))
            for k in changes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--journal", action="append", default=[],
                    help="optional map-editor click journal from this exact "
                         "trace; sparse PnP positions become visual anchors")
    args = ap.parse_args()

    src = np.load(args.trace, allow_pickle=False)
    t = np.asarray(src["t"], float)
    pos_raw = np.asarray(src["pos"], float)
    pos = pos_raw.copy()
    correction = np.zeros_like(pos)
    gates = json.loads(Path(args.map).read_text())["gates"]
    events = [(tp, g) for tp, g in pass_events(Path(args.episode_dir))
              if 0 <= g < len(gates)]

    start_i = 0
    rows = []
    for pass_t, gate in events:
        end_i = int(np.searchsorted(t, pass_t, side="left") - 1)
        if end_i <= start_i:
            continue
        target = np.asarray(gates[gate]["pos"], float)
        end_delta = target - pos_raw[end_i]
        # The previous pass reset starts this leg at a trusted landmark.
        # A linear ramp is the exact correction for constant velocity bias,
        # which is the observed smooth up/right failure mode.
        u = (t[start_i:end_i + 1] - t[start_i]) / max(
            t[end_i] - t[start_i], 1e-6)
        correction[start_i:end_i + 1] = u[:, None] * end_delta
        pos[start_i:end_i + 1] += correction[start_i:end_i + 1]
        rows.append((gate, pass_t, float(np.linalg.norm(end_delta))))
        start_i = int(np.searchsorted(t, pass_t, side="left"))

    visual_rows = []
    visual_correction = np.zeros_like(pos)
    if args.journal:
        calib = load_calib(REPO / "data/calib/calib.json")
        fx, fy, cx, cy = calib["K"]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        R_cb = np.asarray(calib["R_cb"])
        obj = np.ascontiguousarray(SQ_HOLE @ RX90.T)
        for journal in args.journal:
            for line in Path(journal).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if not row.get("ok") or "clicks" not in row:
                    continue
                frame = int(row["frame"])
                gate = int(row["gate"])
                if not (0 <= frame < len(t) and 0 <= gate < len(gates)):
                    continue
                ip = np.ascontiguousarray(
                    row["clicks"], np.float64).reshape(-1, 1, 2)
                try:
                    _n, rvecs, tvecs, _e = cv2.solvePnPGeneric(
                        obj, ip, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                except cv2.error:
                    continue
                best = None
                for rvec, tvec in zip(rvecs, tvecs):
                    try:
                        rvec, tvec = cv2.solvePnPRefineLM(
                            obj, ip, K, None, rvec, tvec)
                    except cv2.error:
                        continue
                    proj, _ = cv2.projectPoints(
                        obj, rvec, tvec, K, None)
                    rms = float(np.sqrt(
                        ((proj - ip) ** 2).sum(axis=2).mean()))
                    if best is None or rms < best[0]:
                        best = (rms, tvec.ravel())
                if best is None or best[0] > 1.0:
                    continue
                qw, qx, qy, qz = src["quat"][frame]
                R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                R_wc = R_wb @ R_cb.T
                p_visual = np.asarray(
                    gates[gate]["pos"], float) - R_wc @ best[1]
                delta = p_visual - pos[frame]
                # A multi-metre residual here is a mismatched journal/map,
                # not a localization correction.
                if np.linalg.norm(delta) <= 4.0:
                    visual_rows.append(
                        (frame, gate, best[0], delta.copy()))

        if visual_rows:
            visual_rows.sort(key=lambda row: row[0])
            # Collapse duplicate-frame anchors robustly.
            anchors = []
            for frame in sorted({row[0] for row in visual_rows}):
                ds = np.asarray(
                    [row[3] for row in visual_rows if row[0] == frame])
                anchors.append((frame, np.median(ds, axis=0)))
            ai = np.asarray([row[0] for row in anchors], int)
            ad = np.asarray([row[1] for row in anchors], float)
            # Preserve the certified front exactly until one second before
            # the first sparse visual anchor, then interpolate the correction
            # between visual measurements. Holding the final correction is
            # preferable to resuming unconstrained inertial drift.
            boundary = max(0, int(ai[0]) - 30)
            ai = np.r_[boundary, ai]
            ad = np.vstack([np.zeros(3), ad])
            for axis in range(3):
                visual_correction[:, axis] = np.interp(
                    np.arange(len(t)), ai, ad[:, axis],
                    left=0.0, right=ad[-1, axis])
            pos += visual_correction

    payload = {key: src[key] for key in src.files}
    payload["pos"] = pos
    payload["pass_smooth_delta"] = correction
    payload["visual_smooth_delta"] = visual_correction
    np.savez_compressed(args.out, **payload)

    print(f"smoothed {len(rows)} legs -> {args.out}")
    for gate, pass_t, err in rows:
        print(f"  g{gate:02d}  t={pass_t:5.2f}s  endpoint correction={err:5.2f}m")
    if rows:
        err = np.asarray([row[2] for row in rows])
        print(f"endpoint correction median/p90/max: {np.median(err):.2f}/"
              f"{np.percentile(err, 90):.2f}/{err.max():.2f}m")
    if args.journal:
        print(f"visual anchors accepted: {len(visual_rows)}")
        if visual_rows:
            by_gate = {
                gate: sum(row[1] == gate for row in visual_rows)
                for gate in sorted({row[1] for row in visual_rows})}
            norms = np.asarray(
                [np.linalg.norm(row[3]) for row in visual_rows])
            print(f"  per gate: {by_gate}")
            print(f"  added correction median/p90/max: "
                  f"{np.median(norms):.2f}/"
                  f"{np.percentile(norms, 90):.2f}/{norms.max():.2f}m")


if __name__ == "__main__":
    main()
