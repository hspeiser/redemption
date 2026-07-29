"""Solve gates 11-16 from hop clusters (vq2_hop_chain dumps): purely
optical within-frame relative vectors, both laps, no IMU, no beliefs.
Iterative association (nearest predicted gate) + Huber IRLS. Gates
8/9/10 fixed (certified). Prints every edge, residual, and a per-lap
split so cross-lap disagreement is visible."""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data"

FIX = {8: np.array([107.83, -1.16, -5.43]),
       9: np.array([115.7, 9.1, -4.6]),
       10: np.array([125.0, 6.6, -2.4])}
FREE = [11, 12, 13, 14, 15, 16]

# initial guesses from the unambiguous hop arithmetic (only used to seed
# association; the LSQ owns the final values)
INIT = {11: [140.4, 2.2, -0.8], 12: [153.5, -11.9, -2.5],
        13: [162.7, -21.9, -4.0], 14: [172.8, -16.0, -4.0],
        15: [183.5, -6.4, -2.5], 16: [203.9, -4.5, -3.5]}


def load_hops():
    rows = []
    for lap in ("003101", "gift", "locked"):
        p = D / f"vq2_hops_{lap}.npz"
        if not p.exists():
            continue
        r = np.load(p)["rows"]
        for row in r:
            g = int(row[0])
            rows.append({"g": g, "rel": row[1:4], "sig": max(row[4], 0.3),
                         "n": int(row[5]), "dep": row[6], "lap": lap})
    return rows


def solve(rows, use_laps):
    pos = {**FIX, **{g: np.asarray(INIT[g], float) for g in FREE}}
    gidx = {g: k for k, g in enumerate(FREE)}
    edges = None
    for outer in range(3):
        # association: target = nearest gate to pos[g] + rel
        edges = []
        for r in rows:
            if r["lap"] not in use_laps or r["g"] not in pos:
                continue
            tgt = pos[r["g"]] + r["rel"]
            best, bd = None, 1e9
            for g2 in pos:
                if g2 == r["g"]:
                    continue
                d = np.linalg.norm(pos[g2] - tgt)
                if d < bd:
                    best, bd = g2, d
            if bd > 5.0:
                edges.append((r, None, bd))
                continue
            edges.append((r, best, bd))
        w = {id(r): 1.0 for r, _b, _d in edges}
        for _irls in range(4):
            A_rows, b_rows = [], []
            for r, b, _d in edges:
                if b is None:
                    continue
                row = np.zeros(len(FREE))
                rhs = np.asarray(r["rel"], float).copy()
                if b in gidx:
                    row[gidx[b]] += 1.0
                else:
                    rhs -= FIX[b]
                    rhs = -rhs  # placeholder, fixed below
                if b in FIX:
                    rhs = np.asarray(r["rel"], float) - FIX[b]
                    rhs = -rhs
                # rebuild cleanly: pos[b] - pos[g] = rel
                rhs = np.asarray(r["rel"], float).copy()
                if b in FIX:
                    rhs = rhs - FIX[b]      # -> -pos[g] part
                if r["g"] in gidx:
                    row[gidx[r["g"]]] -= 1.0
                elif r["g"] in FIX:
                    rhs = rhs + FIX[r["g"]]
                ww = w[id(r)] / r["sig"]
                A_rows.append(row * ww)
                b_rows.append(rhs * ww)
            sol, *_ = np.linalg.lstsq(np.stack(A_rows), np.stack(b_rows),
                                      rcond=None)
            for g, k in gidx.items():
                pos[g] = sol[k]
            for r, b, _d in edges:
                if b is None:
                    continue
                resid = np.linalg.norm(
                    (pos[b] - pos[r["g"]]) - r["rel"]) / r["sig"]
                w[id(r)] = 1.0 if resid < 2.5 else np.sqrt(
                    2.5 / max(resid, 1e-9))
    return pos, edges, w


def main():
    rows = load_hops()
    print(f"{len(rows)} hop clusters loaded")
    for laps in (("003101",), ("gift",), ("003101", "gift", "locked")):
        pos, edges, w = solve(rows, set(laps))
        tag = "+".join(laps)
        print(f"\n===== solve [{tag}] =====")
        if len(laps) == 2:
            for r, b, d in edges:
                if b is None:
                    print(f"  UNASSOCIATED g{r['g']} rel "
                          f"{np.round(r['rel'], 1)} (nearest {d:.1f}m off)")
                    continue
                err = np.linalg.norm((pos[b] - pos[r['g']]) - r["rel"])
                dw = " DOWN-W" if w[id(r)] < 1.0 else ""
                print(f"  g{r['g']:2d}->g{b:2d} [{r['lap']:6s} n={r['n']:2d}"
                      f" d{r['dep']:4.0f}m]: err {err*100:5.0f}cm{dw}")
        for g in FREE:
            print(f"  g{g:2d}: {np.round(pos[g], 2)}")
    # write final map from the combined solve
    pos, _e, _w = solve(rows, {"003101", "gift", "locked"})
    base = json.loads((D / "vq2_map_iter1.json").read_text())
    gates = base["gates"]
    # yaws: keep iter1 click-derived yaws (validated separately)
    for g in FREE:
        gates[g]["pos"] = [float(v) for v in pos[g]]
    gates[10]["pos"] = [float(v) for v in FIX[10]]
    (D / "vq2_map_hop.json").write_text(json.dumps(
        {"frame": "hop-solve: optical within-frame hops, both laps, "
                  "gates 8-10 fixed",
         "gates": gates, "certified": list(range(11))}, indent=1))
    print("\nwrote data/vq2_map_hop.json")


if __name__ == "__main__":
    main()
