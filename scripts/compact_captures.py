"""Losslessly compact capture episodes: the recorder writes the latest camera
frame every loop tick (~40x duplication). Keep the FIRST jpg per unique
frame_id (the one ingest uses), delete duplicates after verifying they are
identical (size check on all, full byte-compare on a random sample).

frames.jsonl is left untouched — ingest keeps the first row per frame_id,
whose file is exactly the one kept.

Usage: uv run python scripts/compact_captures.py <captures_root> [--age-min 5]
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path


def compact_episode(ep, sample_rate=0.01):
    fj = ep / "frames.jsonl"
    fdir = ep / "frames"
    if not fj.exists() or not fdir.exists():
        return (0, 0, "no frames")
    keep = {}     # frame_id -> idx of first row
    dupes = []    # (keep_idx, dup_idx)
    with open(fj) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid, idx = r["frame_id"], r["idx"]
            if fid in keep:
                dupes.append((keep[fid], idx))
            else:
                keep[fid] = idx

    freed = 0
    removed = 0
    mismatches = 0
    for keep_idx, dup_idx in dupes:
        kf = fdir / f"{keep_idx:06d}.jpg"
        df = fdir / f"{dup_idx:06d}.jpg"
        if not df.exists():
            continue
        if not kf.exists():
            continue
        ks, ds = kf.stat().st_size, df.stat().st_size
        if ks != ds:
            mismatches += 1
            continue  # not identical — keep it, lossless means lossless
        if random.random() < sample_rate:
            if (hashlib.sha256(kf.read_bytes()).digest()
                    != hashlib.sha256(df.read_bytes()).digest()):
                mismatches += 1
                continue
        df.unlink()
        freed += ds
        removed += 1
    return (removed, freed, f"{mismatches} kept-not-identical")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--age-min", type=float, default=5.0,
                    help="skip episodes modified in the last N minutes")
    args = ap.parse_args()
    root = Path(args.root)
    now = time.time()
    total_freed = 0
    for ep in sorted(root.iterdir()):
        if not ep.is_dir() or not ep.name.startswith("rc_"):
            continue
        newest = max((f.stat().st_mtime for f in ep.glob("*.jsonl")),
                     default=0)
        if now - newest < args.age_min * 60:
            print(f"{ep.name}: SKIP (recently active)", flush=True)
            continue
        removed, freed, note = compact_episode(ep)
        total_freed += freed
        print(f"{ep.name}: removed {removed} dupes, freed {freed/1e9:.2f} GB "
              f"({note})", flush=True)
    print(f"\nTOTAL freed: {total_freed/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
