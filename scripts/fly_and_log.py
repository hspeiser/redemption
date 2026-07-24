"""Calibration data-collection flight.

Reset -> arm -> take off -> (gate map arrives once flying) -> fly a varied
pattern around the first gates while logging every frame + all telemetry.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp.mavlink_io import MavIO
from aigp.vision_io import VisionRX
from aigp.logger import EpisodeLogger
from aigp.flight import Flyer


def quat_yaw(qw, qx, qy, qz):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="episode name")
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1] / "data" / "episodes"))
    args = ap.parse_args()

    mav = MavIO()
    logger = EpisodeLogger(args.data_root, args.out)
    vis = VisionRX(on_frame=logger.on_frame)

    print("Resetting sim...", flush=True)
    mav.reset_sim()
    time.sleep(1.5)

    print("Arming...", flush=True)
    mav.arm()
    time.sleep(1.0)

    od0 = mav.latest_odom()
    if od0 is None:
        print("FATAL: no odometry", flush=True)
        return 1
    start = np.array(od0["pos"])
    print(f"Start pos NED: {start}", flush=True)

    flyer = Flyer(mav)

    # --- takeoff to 2.5 m and verify the controller holds position ---
    print("Taking off (attitude mode, thrust-limited)...", flush=True)
    hover_wp = start + np.array([0, 0, -2.5])
    ok = flyer.goto(hover_wp, timeout=6.0, reach=0.6)
    od = mav.latest_odom()
    print(f"takeoff reached={ok} pos={np.round(np.array(od['pos']), 2)}", flush=True)
    if flyer.abort or (not ok and start[2] - od["pos"][2] < 0.5):
        print("FATAL: takeoff/controller failed", flush=True)
        logger.finalize(mav, {"status": "takeoff_failed"})
        return 1
    flyer.hold(2.0, anchor=hover_wp)
    od = mav.latest_odom()
    hover_err = np.linalg.norm(np.array(od["pos"]) - hover_wp)
    print(f"hover error after 2s: {hover_err:.2f} m", flush=True)
    if hover_err > 2.0:
        print("FATAL: hover unstable, aborting flight", flush=True)
        logger.finalize(mav, {"status": "hover_unstable"})
        return 1

    # --- gate map (arrives once flying / after reset) ---
    gates = None
    for _ in range(15):
        with mav.lock:
            gates = mav.gate_map
        if gates:
            break
        flyer.hold(1.0, anchor=hover_wp)
    if not gates:
        print("WARNING: no gate map received; flying generic pattern", flush=True)

    if gates:
        g0 = np.array(gates[0]["pos"])
        g1 = np.array(gates[1]["pos"]) if len(gates) > 1 else g0
        print(f"Gate 0 at {g0}, gate 1 at {g1}, {len(gates)} gates total", flush=True)
        d = g0 - np.array(mav.latest_odom()["pos"])
        d[2] = 0
        dist0 = np.linalg.norm(d)
        dhat = d / max(dist0, 1e-6)
        side = np.array([-dhat[1], dhat[0], 0.0])
        zg = g0[2]

        # varied standoff points in front of gate 0: distances, laterals, verticals
        pattern = [
            (14.0,  0.0,  0.0, 0.35),
            (11.0, -3.0,  0.6, 0.3),
            (11.0,  3.0, -0.6, 0.3),
            (9.0,  -2.0,  0.8, 0.3),
            (9.0,   2.0, -0.8, 0.3),
            (7.0,  -3.0,  0.0, 0.25),
            (7.0,   3.0,  0.0, 0.25),
            (6.0,   0.0,  0.8, 0.2),
            (5.0,  -1.5, -0.5, 0.2),
            (5.0,   1.5,  0.5, 0.2),
            (4.0,   0.0,  0.0, 0.15),
            (8.0,  -4.5,  0.0, 0.4),
            (8.0,   4.5,  0.0, 0.4),
            (12.0,  0.0, -1.0, 0.45),
        ]
        for (dist, lat, dz, sweep) in pattern:
            wp = g0 - dhat * dist + side * lat + np.array([0, 0, dz])
            wp[2] = zg + dz
            ok = flyer.goto(wp, look_at=g0, timeout=9.0, sweep_amp=sweep)
            flyer.hold(1.5, look_at=g0, sweep_amp=sweep, sweep_hz=0.5)
            print(f"wp dist={dist} lat={lat} reached={ok} "
                  f"pos={np.round(np.array(mav.latest_odom()['pos']),2)}", flush=True)

        # look toward gate 1 from near gate 0 (off to the side, not through it)
        wp = g0 + side * 4.0
        wp[2] = zg
        flyer.goto(wp, look_at=g1, timeout=9.0)
        flyer.hold(3.0, look_at=g1, sweep_amp=0.3, sweep_hz=0.4)
    else:
        # generic: slow forward drift with yaw sweeps at current heading
        yaw0 = quat_yaw(*mav.latest_odom()["quat_wxyz"])
        fwd = np.array([math.cos(yaw0), math.sin(yaw0), 0.0])
        p = np.array(mav.latest_odom()["pos"])
        for k in range(8):
            wp = p + fwd * (2.0 + 2.0 * k) + np.array([0, 0, 0])
            flyer.goto(wp, look_at=wp + fwd * 10, timeout=8.0, sweep_amp=0.5)

    print("Pattern complete; holding briefly...", flush=True)
    flyer.hold(2.0)

    ep_dir = logger.finalize(mav, {
        "status": "ok",
        "frames": vis.frame_count,
        "dropped": vis.dropped,
        "purpose": "calibration",
    })
    print(f"DONE. Episode: {ep_dir}", flush=True)
    vis.close()
    mav.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
