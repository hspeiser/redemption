import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp.calib.solve import solve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="episode directory")
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--fix-intrinsics", action="store_true",
                    help="lock fx=fy=320, cx=320, cy=180 (rig meta values)")
    args = ap.parse_args()
    fix = (320.0, 320.0, 320.0, 180.0) if args.fix_intrinsics else None
    solve(args.episode, args.step, args.max_frames, fix_intrinsics=fix)
