"""Build the VQ2 gate map MEASURED from recordings (spawn/local frame).

Inputs: pair-row dumps from `vq2_align.py --dump-pairs` (co-visible gate
pairs: sub-pixel PnP of both gates in the same frame -> exact gate->gate
vector in the local frame, drone pose cancelled) + data/vq2_anchor.json
(gate0 position + observed yaw) + gate_map.json (used ONLY for gate-pair
distances during identity resolution and as a rigid-fitted prior to fill
unobserved gates).

    .venv-train\\Scripts\\python.exe scripts\\vq2_build_map.py dump1.npz ...
        -> data/vq2_map_measured.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
MAP_PATH = Path(r"C:\Users\henry\Downloads\gate_map.json")

GATE_SIZE = {"width": 2.72, "height": 2.72, "aperture": 1.5, "depth": 0.26}


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def circ_median(deg):
    deg = np.asarray(deg, float)
    best, bcost = None, None
    for c in deg:
        cost = np.abs(wrap(deg - c)).sum()
        if bcost is None or cost < bcost:
            best, bcost = c, cost
    return float(best)


def main():
    dumps = sys.argv[1:] or [str(REPO / "data/vq2_pairs_003153.npz"),
                             str(REPO / "data/vq2_pairs_003101.npz")]
    m0 = json.loads(MAP_PATH.read_text())
    rel = np.asarray(m0["gates_ring_center_NED_rel_spawn"], float)
    n_g = len(rel)
    anchor = json.loads((REPO / "data/vq2_anchor.json").read_text())
    p0 = np.asarray(anchor["anchor_t"], float)
    yaw0_obs = anchor["anchor_yaw_deg"] + (-m0["gate_yaw_deg"][0])

    # ---------- collect rows ----------
    rows = []
    for d in dumps:
        if not Path(d).exists():
            print(f"  (missing dump {d})")
            continue
        r = np.load(d)["rows"]
        rows.append(r)
        print(f"  {Path(d).name}: {len(r)} rows")
    rows = np.concatenate(rows)

    # ---------- identity resolution by pair distance (convention-free) ----
    # A (bigger det) is assumed = active gate; B resolved by |dp| against
    # the map's inter-gate distances in a window around A.
    edges = {}       # (i, j) -> list of dp
    yaw_obs = {}     # gate -> list of yaw deg
    n_amb = n_used = 0
    for r in rows:
        t0, ag = r[0], int(r[1])
        dp = r[2:5]
        ya, yb, dA, dB = r[5], r[6], r[7], r[8]
        d_obs = np.linalg.norm(dp)
        if d_obs < 3.0:
            continue
        i = ag
        cands = []
        for j in range(max(0, ag - 2), min(n_g, ag + 4)):
            if j == i:
                continue
            if abs(np.linalg.norm(rel[j] - rel[i]) - d_obs) < max(
                    0.08 * d_obs, 0.5):
                cands.append(j)
        if len(cands) != 1:
            n_amb += 1
            continue
        j = cands[0]
        edges.setdefault((i, j), []).append(dp)
        yaw_obs.setdefault(i, []).append(ya)
        yaw_obs.setdefault(j, []).append(yb)
        n_used += 1
    print(f"rows used {n_used}, ambiguous/skipped {n_amb}")

    med_edges = {}
    for (i, j), ds in edges.items():
        ds = np.asarray(ds)
        med = np.median(ds, axis=0)
        spread = np.linalg.norm(ds - med, axis=1)
        med_edges[(i, j)] = (med, len(ds), float(np.median(spread)))
    print("edges:")
    for (i, j), (med, n, sp) in sorted(med_edges.items()):
        print(f"  g{i:2d}->g{j:2d}: n={n:3d}  |d|={np.linalg.norm(med):6.2f}m"
              f"  spread {sp*100:5.1f}cm  d={np.round(med, 2)}")

    # ---------- linear LSQ over gate positions ----------
    # unknowns: pos[0..n_g-1]; constraints: pos0 fixed, edges, weak prior
    # (rigid-fitted raw map) added AFTER a first fit on solved nodes.
    def solve(prior_pos=None, w_prior=0.05):
        A_rows, b_rows = [], []

        def add(i_idx, coef, rhs, w):
            row = np.zeros(n_g)
            for k, c in zip(i_idx, coef):
                row[k] = c
            A_rows.append(row * w)
            b_rows.append(np.asarray(rhs, float) * w)

        add([0], [1.0], p0, 100.0)
        for (i, j), (med, n, sp) in med_edges.items():
            w = np.sqrt(n) / (1.0 + 10 * sp)
            add([j, i], [1.0, -1.0], med, w)
        if prior_pos is not None:
            for g in range(n_g):
                add([g], [1.0], prior_pos[g], w_prior)
        A = np.stack(A_rows)
        B = np.stack(b_rows)
        sol, *_ = np.linalg.lstsq(A, B, rcond=None)
        return sol

    pos = solve()
    # nodes touched by edges (plus gate0) are observed; others sit at 0
    seen = {0}
    for (i, j) in med_edges:
        seen.add(i)
        seen.add(j)
    print(f"observed gates: {sorted(seen)}")

    # rigid-fit the (mirrored) raw map onto observed nodes: rotation about z
    # + translation (mirror already established empirically; try both, keep
    # the better fit)
    best_fit = None
    for s in (-1.0, 1.0):
        relc = rel * np.array([1.0, s, 1.0])
        idx = sorted(seen)
        P = pos[idx]
        Q = relc[idx]
        Pc, Qc = P.mean(0), Q.mean(0)
        num = den = 0.0
        for a, b in zip(P - Pc, Q - Qc):
            num += b[0] * a[1] - b[1] * a[0]
            den += b[0] * a[0] + b[1] * a[1]
        th = np.arctan2(num, den)
        Rz = Rotation.from_euler("z", th).as_matrix()
        Qr = (Rz @ (Q - Qc).T).T
        t = Pc - 0
        resid = P - (Qr + Pc)
        rms = float(np.sqrt((resid ** 2).sum(1).mean()))
        if best_fit is None or rms < best_fit[0]:
            best_fit = (rms, s, th, Rz, Qc, Pc)
    rms, s, th, Rz, Qc, Pc = best_fit
    print(f"rigid prior fit: mirror={'yes' if s < 0 else 'no'} "
          f"rot {np.degrees(th):+.2f}deg  rms {rms*100:.1f}cm")
    prior_pos = (Rz @ ((rel * np.array([1.0, s, 1.0])) - Qc).T).T + Pc

    pos = solve(prior_pos=prior_pos, w_prior=0.2)

    # ---------- per-gate yaw ----------
    # observed circular median where available; prior-derived otherwise.
    # prior yaw: gate yaws rotate with the constellation: s*yaw_map + rot
    yaw_out = []
    for g in range(n_g):
        obs = yaw_obs.get(g, [])
        prior_yaw = s * m0["gate_yaw_deg"][g] + np.degrees(th)
        if len(obs) >= 3:
            ym = circ_median(obs)
            # PnP yaw has a front/back 180 ambiguity per row; snap to the
            # nearer of prior_yaw / prior_yaw+180 clusters
            d0 = abs(wrap(ym - prior_yaw))
            d1 = abs(wrap(ym - prior_yaw - 180))
            yaw_out.append(ym if d0 <= d1 else circ_median(
                [w0 + 180 for w0 in obs]))
        else:
            yaw_out.append(prior_yaw)
    # gate0: trust the direct spawn observation
    yaw_out[0] = yaw0_obs

    gates = []
    for g in range(n_g):
        q = Rotation.from_euler("z", yaw_out[g], degrees=True).as_quat()
        gates.append({
            "gate_id": g,
            "pos": [float(v) for v in pos[g]],
            "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]),
                          float(q[2])],
            "observed": g in seen,
            "n_yaw_obs": len(yaw_obs.get(g, [])),
            **GATE_SIZE,
        })
    out = {"frame": "local spawn frame of the anchor episode",
           "gates": gates,
           "prior_fit": {"mirror": s < 0, "rot_deg": float(np.degrees(th)),
                         "rms_cm": rms * 100}}
    out_p = REPO / "data" / "vq2_map_measured.json"
    out_p.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_p}")
    for g in gates:
        print(f"  g{g['gate_id']:2d} pos {np.round(g['pos'], 2)} "
              f"yaw {Rotation.from_quat([g['quat_wxyz'][1], g['quat_wxyz'][2], g['quat_wxyz'][3], g['quat_wxyz'][0]]).as_euler('zyx', degrees=True)[0]:+7.1f} "
              f"{'obs' if g['observed'] else 'prior'}")


if __name__ == "__main__":
    main()
