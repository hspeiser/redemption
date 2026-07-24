"""Passive recorder: logs everything while a human flies the sim.

Sends NO commands (except nothing at all — pure listener). Run it, fly laps,
Ctrl+C to stop and finalize the episode.

    uv run python scripts\\record.py --out manual01
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp.mavlink_io import MavIO
from aigp.vision_io import VisionRX
from aigp.logger import EpisodeLogger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="episode name")
    ap.add_argument("--data-root",
                    default=str(Path(__file__).resolve().parents[1] / "data" / "episodes"))
    args = ap.parse_args()

    mav = MavIO()
    logger = EpisodeLogger(args.data_root, args.out)
    vis = VisionRX(on_frame=logger.on_frame)

    print("\nRECORDING — fly the course! Ctrl+C to stop and save.\n", flush=True)
    t0 = time.time()
    try:
        while True:
            time.sleep(5.0)
            rs = mav.race_status
            gate = rs["active_gate"] if rs else "?"
            print(f"[{time.time()-t0:6.0f}s] frames={vis.frame_count} "
                  f"odom={len(mav.odom)} imu={len(mav.imu)} "
                  f"collisions={len(mav.collisions)} active_gate={gate}",
                  flush=True)
    except KeyboardInterrupt:
        print("\nStopping — finalizing episode...", flush=True)
    finally:
        vis.close()
        logger.finalize(mav, {
            "purpose": "manual_flight",
            "frames": vis.frame_count,
            "dropped": vis.dropped,
            "duration_s": time.time() - t0,
        })
        mav.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
