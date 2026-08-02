"""Find g1-approach debug frames from July 30 (v77) and today (104428)
at similar drone positions, copy for visual comparison."""
import json
import shutil
from pathlib import Path

import numpy as np

def pick(session, label, want_pos, count=2):
    root = Path(r"D:\ai-gp\raw_sessions") / session
    best = []
    for line in (root / "localizer_debug/debug.jsonl").open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        d = r["debug"]
        if d.get("active_gate") != 1:
            continue
        p = np.asarray(d.get("position", [0, 0, 0]))
        dist = float(np.linalg.norm(p[:2] - np.asarray(want_pos)))
        best.append((dist, r["image"], p.round(1).tolist()))
    best.sort()
    out_dir = Path(r"data\g1_compare")
    out_dir.mkdir(exist_ok=True)
    for k, (dist, img, pos) in enumerate(best[:count]):
        src = root / img
        dst = out_dir / f"{label}_{k}_d{dist:.1f}.jpg"
        shutil.copy(src, dst)
        print(label, k, "pos", pos, "dist", round(dist, 2), "->", dst.name)

# compare from the same approach point ~(20, 6): mid g0->g1
pick("vq2_20260730_200627", "jul30", (20.0, 6.0))
pick("vq2_20260731_104428", "jul31", (20.0, 6.0))
