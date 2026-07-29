"""Splice a trusted front trace to a repaired back trace at a gate crossing."""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True)
    ap.add_argument("--back", required=True)
    ap.add_argument("--switch-time", type=float, required=True)
    ap.add_argument("--blend-secs", type=float, default=0.20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    front = np.load(args.front, allow_pickle=False)
    back = np.load(args.back, allow_pickle=False)
    if len(front["t"]) != len(back["t"]) or not np.array_equal(
            front["path"], back["path"]):
        raise ValueError("trace frame paths do not match exactly")

    t = np.asarray(front["t"], float)
    half = args.blend_secs / 2
    weight = np.clip(
        (t - (args.switch_time - half)) / max(args.blend_secs, 1e-6),
        0.0, 1.0)
    payload = {key: front[key] for key in front.files}
    payload["pos"] = (
        (1 - weight[:, None]) * front["pos"] +
        weight[:, None] * back["pos"])

    # Normalized shortest-arc quaternion interpolation is sufficient for the
    # sub-five-degree handoff observed here.
    qa = np.asarray(front["quat"], float)
    qb = np.asarray(back["quat"], float).copy()
    qb[np.sum(qa * qb, axis=1) < 0] *= -1
    q = (1 - weight[:, None]) * qa + weight[:, None] * qb
    payload["quat"] = q / np.linalg.norm(q, axis=1, keepdims=True)
    if "sigma" in back.files:
        payload["sigma"] = (
            (1 - weight) * front["sigma"] + weight * back["sigma"])
    payload["splice_weight"] = weight
    np.savez_compressed(args.out, **payload)
    print(f"spliced {args.front} -> {args.back} at "
          f"{args.switch_time:.2f}s ({args.blend_secs:.2f}s blend)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
