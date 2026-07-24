"""Record the live overlay's annotated stream to MP4 (until Ctrl+C or
--secs). Reads the MJPEG /stream endpoint of live_overlay.py.

    uv run python scripts/capture_overlay.py --secs 300 --out data\\vq2_live.mp4
"""

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 360


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"data\vq2_live.mp4")
    ap.add_argument("--secs", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (W, H))
    req = urllib.request.urlopen("http://localhost:8899/stream", timeout=10)
    buf = b""
    n = 0
    t_end = time.time() + args.secs
    last_write = 0.0
    try:
        while time.time() < t_end:
            chunk = req.read(16384)
            if not chunk:
                break
            buf += chunk
            while True:
                a = buf.find(b"\xff\xd8")
                b = buf.find(b"\xff\xd9", a + 2)
                if a < 0 or b < 0:
                    if len(buf) > 4_000_000:
                        buf = buf[-100_000:]
                    break
                jpg = buf[a:b + 2]
                buf = buf[b + 2:]
                # pace to target fps (stream may deliver faster/slower)
                now = time.time()
                if now - last_write < 1.0 / args.fps * 0.5:
                    continue
                last_write = now
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    if frame.shape[:2] != (H, W):
                        frame = cv2.resize(frame, (W, H))
                    vw.write(frame)
                    n += 1
                    if n % 300 == 0:
                        print(f"{n} frames", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        vw.release()
        print(f"wrote {n} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
