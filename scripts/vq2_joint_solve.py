"""Joint back-half solve, NO position beliefs anywhere.

Measurements:
  * click sessions (fresh IPPE re-solve of the journal clicks) -> exact
    camera-rel-gate vectors; per-session regression gives cam_tm, v_tm
  * IMU bridges between consecutive sessions of the SAME lap (accel
    rotated by the trace's vision-corrected ATTITUDE only) -> relative
    gate-to-gate edge vectors
  * co-visible pair rows (vq2_align --dump-pairs) -> pose-free
    gate-to-gate vectors, association by inter-gate distance consistency
Anchors: gates fixed from the certified front (bible). Everything else
floats. Robust IRLS (Huber). Gate 17 excluded (non-visual finish).

    .venv-train\\Scripts\\python.exe scripts\\vq2_joint_solve.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ekf import GateEKF  # noqa: E402
from aigp.vision.labels import load_calib  # noqa: E402
from scripts.ekf_bringup import load_imu  # noqa: E402

G_NED = np.array([0.0, 0.0, 9.81])
HOLE = 0.75
SQ_HOLE = np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE],
                    [HOLE, 0, HOLE], [-HOLE, 0, HOLE]])
RX90 = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])

calib = load_calib(REPO / "data/calib/calib.json")
fx, fy, cx, cy = calib["K"]
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
R_cb = np.asarray(calib["R_cb"])
OBJ = np.ascontiguousarray(SQ_HOLE @ RX90.T)

# certified anchors (the bible end)
FIX = {8: np.array([107.83, -1.16, -5.43]),
       9: np.array([115.7, 9.1, -4.6])}
GATES = list(range(8, 17))          # nodes in the solve
FREE = [g for g in GATES if g not in FIX]

LAPS = [
    {"name": "003101",
     "ep": Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
                r"\ai-grand-prix\outputs\captures\rc_20260724_003101"),
     "journals": ["data/vq2_map_human9.json.journal.jsonl",
                  "data/vq2_map_human10.json.journal.jsonl"],
     "att_trace": "data/vq2_trace_101_i1.npz",
     "sess_trace": "data/vq2_trace_v3_101.npz",
     "crash_t": None},
    {"name": "gift",
     "ep": REPO / "data/ep_rc_20260729_000036",
     "journals": ["data/vq2_map_human11.json.journal.jsonl"],
     "att_trace": "data/vq2_trace_gift_i1.npz",
     "sess_trace": "data/vq2_trace_gift.npz",
     "crash_t": None},
    {"name": "003153",
     "ep": Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
                r"\ai-grand-prix\outputs\captures\rc_20260724_003153"),
     "journals": ["data/vq2_map_human8.json.journal.jsonl"],
     "att_trace": "data/vq2_trace_slam2.npz",
     "sess_trace": "data/vq2_trace_slam2.npz",
     "crash_t": 33.3},
]
PAIR_DUMPS = []  # pair rows are in the per-episode gyro frame, NOT the
# map frame — unusable as world vectors without per-episode yaw alignment
# (verified: g9->g8 pair vector is length-correct but rotated ~the
# anchor yaw). Distances only, if ever.
PRIOR = json.loads((REPO / "data/vq2_map_iter1.json").read_text())["gates"]
REL_PRIOR = {g: np.asarray(PRIOR[g]["pos"]) for g in range(18)}


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


def fresh_solve(clicks, R_wc):
    ip = np.ascontiguousarray(clicks, np.float64).reshape(-1, 1, 2)
    try:
        _n, rv, tv, _e = cv2.solvePnPGeneric(
            OBJ, ip, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    except cv2.error:
        return None
    best = None
    for r0, t0 in zip(rv, tv):
        try:
            r0, t0 = cv2.solvePnPRefineLM(OBJ, ip, K, None, r0, t0)
        except cv2.error:
            continue
        pr, _ = cv2.projectPoints(OBJ, r0, t0, K, None)
        rms = float(np.sqrt(((pr - ip) ** 2).sum(axis=2).mean()))
        R0, _ = cv2.Rodrigues(r0)
        R_gw = R_wc @ (R0 @ RX90)
        up_err = abs(float(R_gw[2, 2]) - 1.0)
        score = rms + 5.0 * up_err
        if best is None or score < best[0]:
            best = (score, rms, t0.ravel(), R_gw)
    if best is None or best[1] > 1.0:
        return None
    _s, rms, t_c, R_gw = best
    yaw = float(np.degrees(np.arctan2(R_gw[1, 0], R_gw[0, 0])))
    return t_c, yaw, rms


def lap_edges(lap):
    """-> (edges [(gi, gj, vec, sigma)], yaw_obs {gate: [deg]})"""
    imu = load_imu(lap["ep"])
    brk = np.where(np.diff(imu[:, 0]) < -0.5)[0]
    if len(brk):
        segs = np.split(np.arange(len(imu)), brk + 1)
        imu = imu[max(segs, key=len)]
    t_imu = imu[:, 0]
    t0 = t_imu[0]
    att = np.load(lap["att_trace"], allow_pickle=True)
    st = np.load(lap["sess_trace"], allow_pickle=True)
    qs = att["quat"]
    rot = Rotation.from_quat(
        np.stack([qs[:, 1], qs[:, 2], qs[:, 3], qs[:, 0]], axis=1))
    tt = np.asarray(att["t"], float) + t0
    keep = np.concatenate([[True], np.diff(tt) > 1e-6])
    sl = Slerp(tt[keep], rot[keep])

    def R_at(t_abs):
        return sl(np.clip(t_abs, tt[keep][0], tt[keep][-1])).as_matrix()

    n_i = len(imu)
    Rws = R_at(t_imu)
    a_w = np.einsum("nij,nj->ni", Rws,
                    imu[:, 1:4] * np.asarray(GateEKF.ACCEL_SIGN)) + G_NED
    dts = np.diff(t_imu, prepend=t_imu[0])
    dts[(dts <= 0) | (dts > 0.5)] = 0
    Vc = np.cumsum(a_w * dts[:, None], axis=0)
    Pc = np.cumsum(Vc * dts[:, None], axis=0)

    def V_at(tr):
        return np.array([np.interp(tr + t0, t_imu, Vc[:, c])
                         for c in range(3)])

    def P_at(tr):
        return np.array([np.interp(tr + t0, t_imu, Pc[:, c])
                         for c in range(3)])

    obs = []
    yaw_obs = {}
    for jn in lap["journals"]:
        for ln in Path(jn).read_text().splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if not r.get("ok") or "clicks" not in r:
                continue
            g = int(r["gate"])
            if g not in GATES:
                continue
            fidx = int(r["frame"])
            if fidx >= len(st["t"]):
                continue
            trel = float(st["t"][fidx])
            R_wb = R_at(trel + t0)
            sol = fresh_solve(r["clicks"], R_wb @ R_cb.T)
            if sol is None:
                continue
            t_c, yaw, rms = sol
            rel_w = R_wb @ R_cb.T @ t_c      # gate - camera, world axes
            obs.append({"t": trel, "gate": g, "rel": rel_w})
            yaw_obs.setdefault(g, []).append(yaw)
    obs.sort(key=lambda o: o["t"])

    sessions = []
    for o in obs:
        if sessions and o["gate"] == sessions[-1]["gate"] and \
                o["t"] - sessions[-1]["obs"][-1]["t"] < 1.5:
            sessions[-1]["obs"].append(o)
        else:
            sessions.append({"gate": o["gate"], "obs": [o]})
    for s in sessions:
        ts = np.array([o["t"] for o in s["obs"]])
        cam = -np.array([o["rel"] for o in s["obs"]])
        tm = ts.mean()
        curv = np.array([P_at(t) - P_at(tm) - V_at(tm) * (t - tm)
                         for t in ts])
        lin = cam - curv
        A = np.stack([np.ones_like(ts), ts - tm], axis=1)
        coef, *_ = np.linalg.lstsq(A, lin, rcond=None)
        s["tm"], s["cam_tm"], s["v_tm"], s["n"] = tm, coef[0], coef[1], \
            len(ts)
        s["span"] = float(ts.max() - ts.min())

    edges = []
    for a, b in zip(sessions, sessions[1:]):
        dt = b["tm"] - a["tm"]
        if not (0 < dt < 6.0) or a["gate"] == b["gate"]:
            continue
        if lap["crash_t"] is not None and a["tm"] < lap["crash_t"] < b["tm"]:
            continue
        d_imu = P_at(b["tm"]) - P_at(a["tm"]) - V_at(a["tm"]) * dt
        cam_disp = a["v_tm"] * dt + d_imu
        vec = cam_disp - b["cam_tm"] + a["cam_tm"]   # g_b - g_a
        # velocity-regression noise dominates when the session is short
        v_sig = 0.10 / max(a["span"], 0.2)
        sigma = 0.10 + 0.05 * dt ** 2 + v_sig * dt
        edges.append((a["gate"], b["gate"], vec, sigma))
    return edges, yaw_obs


def pass_edges(lap):
    """Drone displacement between consecutive gate passes = gate-to-gate
    vector (+/- where in the aperture it crossed). Uses trace positions,
    which are IMU+vision smooth BETWEEN reloc jumps; windows containing a
    jump are dropped."""
    ep = lap["ep"]
    iw = []
    for ln in open(ep / "imu.jsonl"):
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if "time_usec" in r and "wall" in r:
            iw.append((r["wall"], r["time_usec"] * 1e-6))
    iw = np.array(iw)
    imu = load_imu(ep)
    brk = np.where(np.diff(imu[:, 0]) < -0.5)[0]
    if len(brk):
        segs = np.split(np.arange(len(imu)), brk + 1)
        imu = imu[max(segs, key=len)]
    t0 = imu[0, 0]
    iw = iw[(iw[:, 1] >= t0) & (iw[:, 1] <= imu[-1, 0])]
    seen0 = False
    prev = None
    t_pass = {}
    for ln in open(ep / "mav.jsonl"):
        if '"race_status"' not in ln:
            continue
        r = json.loads(ln)
        ag = int(r["active_gate"])
        if ag == 0:
            seen0 = True
        if seen0:
            if prev is not None and ag == prev + 1:
                t_pass[prev] = np.interp(r["wall"], iw[:, 0],
                                         iw[:, 1]) - t0
            prev = ag
    tr = np.load(lap["att_trace"], allow_pickle=True)
    tt = np.asarray(tr["t"], float)
    pos = np.asarray(tr["pos"], float)
    edges = []
    for g in range(8, 16):
        if g not in t_pass or g + 1 not in t_pass:
            continue
        ta, tb = t_pass[g], t_pass[g + 1]
        m = (tt >= ta - 0.05) & (tt <= tb + 0.05)
        if m.sum() < 4:
            continue
        step = np.linalg.norm(np.diff(pos[m], axis=0), axis=1)
        dtf = np.diff(tt[m])
        # jump = frame-to-frame step far beyond physical speed (20 m/s)
        if np.any(step > np.maximum(20.0 * dtf, 0.4)):
            continue
        pa = np.array([np.interp(ta, tt, pos[:, c]) for c in range(3)])
        pb = np.array([np.interp(tb, tt, pos[:, c]) for c in range(3)])
        sig = 1.0 + 0.2 * (tb - ta)
        edges.append((g, g + 1, pb - pa, sig))
    return edges


def pair_edges():
    edges = {}
    for d in PAIR_DUMPS:
        p = REPO / d
        if not p.exists():
            print(f"  (no pair dump {d})")
            continue
        rows = np.load(p)["rows"]
        for r in rows:
            ag = int(r[1])
            dp = r[2:5]
            d_obs = np.linalg.norm(dp)
            if d_obs < 3.0:
                continue
            cands = []
            for i in range(max(0, ag - 1), min(17, ag + 2)):
                for j in range(max(0, i - 2), min(17, i + 4)):
                    if j == i:
                        continue
                    dm = np.linalg.norm(REL_PRIOR[j] - REL_PRIOR[i])
                    if abs(dm - d_obs) < max(0.06 * d_obs, 0.4):
                        cands.append((i, j))
            # require unambiguous association AND i == active gate
            cands = [c for c in cands if c[0] == ag]
            if len(cands) != 1:
                continue
            i, j = cands[0]
            if i in GATES and j in GATES:
                edges.setdefault((i, j), []).append(dp)
    out = []
    for (i, j), ds in edges.items():
        ds = np.asarray(ds)
        med = np.median(ds, axis=0)
        spread = float(np.median(np.linalg.norm(ds - med, axis=1)))
        sigma = max(0.05, spread) / np.sqrt(len(ds))
        out.append((i, j, med, sigma, len(ds), spread))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--laps", default="003101,gift,003153",
                    help="comma list of lap names to include")
    ap.add_argument("--no-pass-edges", action="store_true")
    ap.add_argument("--no-session-edges", action="store_true")
    args = ap.parse_args()
    want = set(args.laps.split(","))

    all_edges = []
    yaw_all = {}
    for lap in LAPS:
        if lap["name"] not in want:
            continue
        try:
            e, y = lap_edges(lap)
        except FileNotFoundError as ex:
            print(f"lap {lap['name']}: SKIPPED ({ex})")
            continue
        if not args.no_session_edges:
            print(f"lap {lap['name']}: {len(e)} session edges")
            for gi, gj, vec, sig in e:
                print(f"    sess g{gi:2d}->g{gj:2d}: {np.round(vec, 2)} "
                      f"|{np.linalg.norm(vec):5.1f}m| sig {sig*100:.0f}cm")
                all_edges.append((gi, gj, vec, sig, f"s:{lap['name']}"))
        for g, ys in y.items():
            yaw_all.setdefault(g, []).extend(ys)
        if not args.no_pass_edges:
            pe = pass_edges(lap)
            for gi, gj, vec, sig in pe:
                print(f"    pass g{gi:2d}->g{gj:2d}: {np.round(vec, 2)} "
                      f"|{np.linalg.norm(vec):5.1f}m| sig {sig*100:.0f}cm")
                all_edges.append((gi, gj, vec, sig, f"p:{lap['name']}"))

    gidx = {g: k for k, g in enumerate(FREE)}
    n_u = len(FREE)

    def solve(weights):
        A_rows, b_rows = [], []
        for w, (gi, gj, vec, sig, _src) in zip(weights, all_edges):
            row = np.zeros(n_u)
            rhs = np.asarray(vec, float).copy()
            okA = okB = False
            if gj in gidx:
                row[gidx[gj]] += 1.0
                okA = True
            elif gj in FIX:
                rhs -= 0
                rhs = FIX[gj] - vec        # unused branch guard
            if gi in gidx:
                row[gidx[gi]] -= 1.0
                okB = True
            # rebuild rhs properly: gj - gi = vec
            rhs = np.asarray(vec, float).copy()
            if gj in FIX:
                # -gi = vec - FIX[gj]  ->  gi = FIX[gj] - vec
                rhs = rhs - FIX[gj]
            if gi in FIX:
                rhs = rhs + FIX[gi]
            if not (okA or okB):
                continue
            ww = w / sig
            A_rows.append(row * ww)
            b_rows.append(rhs * ww)
        A = np.stack(A_rows)
        B = np.stack(b_rows)
        sol, *_ = np.linalg.lstsq(A, B, rcond=None)
        return sol

    w = np.ones(len(all_edges))
    for it in range(4):
        sol = solve(w)
        pos = dict(FIX)
        for g, k in gidx.items():
            pos[g] = sol[k]
        res = []
        for (gi, gj, vec, sig, _s) in all_edges:
            r = np.linalg.norm((pos[gj] - pos[gi]) - vec) / sig
            res.append(r)
        res = np.asarray(res)
        w = np.where(res < 2.0, 1.0, np.sqrt(2.0 / np.maximum(res, 1e-9)))
    print("\nresiduals after IRLS:")
    for (gi, gj, vec, sig, src), r, wi in zip(all_edges, res, w):
        err = np.linalg.norm((pos[gj] - pos[gi]) - vec)
        flag = " DOWN-WEIGHTED" if wi < 1.0 else ""
        print(f"  g{gi:2d}->g{gj:2d} [{src:8s}]: err {err*100:6.0f}cm "
              f"({r:5.1f} sig){flag}")
    print("\nsanity checks:")
    n_bad = 0
    for g in GATES[:-1]:
        if g + 1 not in pos:
            continue
        step = pos[g + 1] - pos[g]
        d = np.linalg.norm(step)
        msgs = []
        if not (6.0 < d < 26.0):
            msgs.append(f"step length {d:.1f}m outside 6-26m")
        if step[0] < 2.0:
            msgs.append(f"course x-progress {step[0]:+.1f}m (should be >2)")
        if msgs:
            n_bad += 1
            print(f"  g{g}->g{g+1}: " + "; ".join(msgs))
    for g in GATES:
        if not (-9.0 < pos[g][2] < -0.3):
            n_bad += 1
            print(f"  g{g}: altitude z {pos[g][2]:+.2f} outside (-9,-0.3)")
    if not n_bad:
        print("  all pass (step length, x-progress, altitude)")
    print("\nsolved gates:")
    for g in GATES:
        y = circ_median(yaw_all[g]) if g in yaw_all else None
        ys = f" yaw {y:+7.1f} (n={len(yaw_all.get(g, []))})" if y is not \
            None else ""
        tag = "FIX" if g in FIX else "   "
        print(f"  g{g:2d} {tag}: {np.round(pos[g], 2)}{ys}")
    out = {str(g): [float(v) for v in pos[g]] for g in GATES}
    (REPO / "data/vq2_joint_solution.json").write_text(json.dumps(
        {"pos": out,
         "yaw": {str(g): circ_median(v) for g, v in yaw_all.items()}},
        indent=1))
    print("wrote data/vq2_joint_solution.json")


if __name__ == "__main__":
    main()
