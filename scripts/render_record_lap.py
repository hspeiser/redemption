"""Render the record lap from its raw recording, with the live overlay.

Reads a full-record session's camera frames plus the localizer debug
stream and draws what the drone actually saw and believed: projected
gates, matched corners, fused-corner count and filter sigma.  Output is
a README-sized GIF.

    python scripts/render_record_lap.py \
        --session D:\\ai-gp\\raw_sessions\\vq2_20260802_191537 \
        --out docs/record_lap.gif
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GREEN = (90, 255, 90)
DIM = (70, 150, 70)
WHITE = (255, 255, 255)


def load_frames(session: Path):
    rows = []
    with (session / "frames_index.csv").open() as stream:
        for row in csv.DictReader(stream):
            rows.append((int(row["wall_ns"]), row["path"]))
    rows.sort()
    seen, out = set(), []
    for wall, path in rows:
        if path in seen:
            continue
        seen.add(path)
        out.append((wall, path))
    return out


def load_debug(session: Path):
    path = session / "localizer_debug" / "debug.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.open(encoding="utf-8"):
        try:
            payload = json.loads(line)
        except Exception:
            continue
        rows.append((int(payload["wall_ns"]), payload.get("debug") or {}))
    rows.sort()
    return rows


def draw(frame, dbg, elapsed):
    if dbg:
        by_gate = {}
        for item in dbg.get("expected", []):
            by_gate.setdefault(int(item["gate"]), []).append(item["pixel"])
        accepted = set(
            (dbg.get("multigate") or {}).get("accepted_gates", []))
        for gate, pixels in by_gate.items():
            pts = np.asarray(pixels, np.float32)
            if len(pts) < 4:
                continue
            inside = (
                (pts[:, 0] > -200) & (pts[:, 0] < frame.shape[1] + 200)
                & (pts[:, 1] > -200) & (pts[:, 1] < frame.shape[0] + 200)
            )
            if inside.sum() < 4:
                continue
            hull = cv2.convexHull(pts[inside].astype(np.int32))
            hot = gate in accepted
            cv2.polylines(frame, [hull], True, GREEN if hot else DIM,
                          2 if hot else 1, cv2.LINE_AA)
            if hot:
                x, y = hull[:, 0, :].min(axis=0)
                cv2.putText(frame, f"G{gate}", (int(x), max(int(y) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1,
                            cv2.LINE_AA)
        for match in dbg.get("matches", []):
            u, v = match["observed"]
            colour = GREEN if match.get("fused") else DIM
            cv2.circle(frame, (int(u), int(v)), 3, colour, -1, cv2.LINE_AA)
        sigma = float(dbg.get("position_sigma_m", 0.0))
        fused = int(dbg.get("fused", 0))
        n_gates = len(accepted)
        hud = (f"t {elapsed:5.2f}s   gates {n_gates}   "
               f"corners fused {fused:2d}   sigma {100 * sigma:4.1f}cm")
    else:
        hud = f"t {elapsed:5.2f}s"
    cv2.rectangle(frame, (0, frame.shape[0] - 22),
                  (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, hud, (8, frame.shape[0] - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--start-s", type=float, default=0.0,
                    help="offset from the first recorded frame")
    ap.add_argument("--duration-s", type=float, default=35.4)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--width", type=int, default=420)
    ap.add_argument("--colors", type=int, default=96)
    ap.add_argument("--frame-ms", type=int, default=55)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    frames = load_frames(args.session)
    debug = load_debug(args.session)
    if not frames:
        raise SystemExit("no frames in session")
    base = frames[0][0] + int(args.start_s * 1e9)
    end = base + int(args.duration_s * 1e9)
    debug_walls = np.asarray([w for w, _ in debug], np.int64) if debug \
        else np.zeros(0, np.int64)

    out_frames = []
    kept = 0
    for index, (wall, rel) in enumerate(frames):
        if wall < base or wall > end:
            continue
        if kept % args.stride:
            kept += 1
            continue
        kept += 1
        image = cv2.imread(str(args.session / rel))
        if image is None:
            continue
        dbg = {}
        if len(debug_walls):
            j = int(np.argmin(np.abs(debug_walls - wall)))
            if abs(debug_walls[j] - wall) < 150_000_000:
                dbg = debug[j][1]
        image = draw(image, dbg, (wall - base) / 1e9)
        if args.title and len(out_frames) < 12:
            cv2.putText(image, args.title, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2,
                        cv2.LINE_AA)
        height = int(image.shape[0] * args.width / image.shape[1])
        small = cv2.resize(image, (args.width, height),
                           interpolation=cv2.INTER_AREA)
        out_frames.append(Image.fromarray(
            cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ).quantize(colors=args.colors, method=Image.MEDIANCUT))

    if not out_frames:
        raise SystemExit("window selected no frames")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(args.out, save_all=True,
                       append_images=out_frames[1:],
                       duration=args.frame_ms, loop=0, optimize=True)
    print(f"{args.out}: {len(out_frames)} frames -> "
          f"{args.out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
