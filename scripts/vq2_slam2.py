"""SLAM v2: solve gates 10-16 with gates 0-9 as FIXED anchors.

Unknowns: keyframe p,v (3+3 each), gates 10-16 (3 each), yaw-rate bias b.
Constraints (all linear, one joint sparse system):
  IMU chains between consecutive keyframes (exact integrals, weight high)
  known-gate obs:   p_k = g_fixed - rel  (+ b coupling)   [absolute]
  unknown-gate obs: g_u - p_k = rel      (+ b coupling)   [relative]
Obs weighting split into bearing (tight) / range (loose) via projection
onto the ray direction. One robust re-solve drops >2 m residual obs.

    .venv-train\\Scripts\\python.exe scripts\\vq2_slam2.py
"""
import json
import numpy as np
from pathlib import Path
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ekf import GateEKF  # noqa: E402
from scripts.ekf_bringup import load_imu  # noqa: E402

D = REPO / "data"
EP = Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
          r"\ai-grand-prix\outputs\captures\rc_20260724_003153")
G_NED = np.array([0.0, 0.0, 9.81])
ZHAT = np.array([0.0, 0.0, 1.0])

chain = json.loads((D / "vq2_map_chain.json").read_text())
KNOWN = {i: np.asarray(chain["gates"][i]["pos"]) for i in
         chain["certified"]}
UNK = [g for g in range(10, 17)]

rows = np.load(D / "vq2_obs_003153.npz")["rows"]
# (t, ag, relx..z, yaw, rms, depth)
m = (rows[:, 6] < 1.2) & (rows[:, 7] > 3.0) & (rows[:, 7] < 24.0)
ags = rows[:, 1].astype(int)
t_r = rows[:, 0]
# identity stability: away from transitions
keep = m.copy()
tr_ts = [t_r[i] for i in range(1, len(rows)) if ags[i] != ags[i - 1]]
for tt in tr_ts:
    keep &= ~(np.abs(t_r - tt) < 0.5)
rows = rows[keep]
print(f"obs kept: {len(rows)}  (known-gate: "
      f"{int(sum(1 for r in rows if int(r[1]) in KNOWN))}, unknown: "
      f"{int(sum(1 for r in rows if int(r[1]) in UNK))})")

imu = load_imu(EP)
t_imu = imu[:, 0]
t0 = t_imu[0]
f_rest = imu[t_imu < t0 + 4.0, 1:4].mean(axis=0)
pitch = np.arcsin(np.clip(f_rest[0] / 9.81, -1, 1))
roll = np.arctan2(-f_rest[1], -f_rest[2])
R = Rotation.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()
gs = np.asarray(GateEKF.GYRO_SIGN, float)
n_i = len(imu)
a_w = np.zeros((n_i, 3))
prev = t_imu[0]
for i in range(n_i):
    dt = t_imu[i] - prev
    prev = t_imu[i]
    if 0 < dt < 0.5:
        R = R @ Rotation.from_rotvec(imu[i, 4:7] * gs * dt).as_matrix()
    a_w[i] = R @ (imu[i, 1:4] * np.asarray(GateEKF.ACCEL_SIGN)) + G_NED
dts = np.diff(t_imu, prepend=t_imu[0])
dts[(dts <= 0) | (dts > 0.5)] = 0
Vc = np.cumsum(a_w * dts[:, None], axis=0)
Pc = np.cumsum(Vc * dts[:, None], axis=0)


def V_at(trel):
    return np.array([np.interp(trel + t0, t_imu, Vc[:, c])
                     for c in range(3)])


def P_at(trel):
    return np.array([np.interp(trel + t0, t_imu, Pc[:, c])
                     for c in range(3)])


kf_t = np.unique(np.concatenate([
    np.round(rows[:, 0], 4),
    np.arange(rows[:, 0].min(), rows[:, 0].max(), 0.25)]))
K = len(kf_t)
kf_of = {round(t, 4): k for k, t in enumerate(np.round(kf_t, 4))}
print(f"keyframes: {K}")

NU = 6 * K + 3 * len(UNK) + 1        # p,v per kf + unknown gates + bias
u_of = {g: 6 * K + 3 * j for j, g in enumerate(UNK)}
B_COL = NU - 1


def solve(drop=None):
    eqs = []                          # (cols, coefs, rhs, weight_rows 3x?)
    A = lil_matrix((3 * (K - 1) * 2 + 3 * len(rows) + 6, NU))
    b = np.zeros(A.shape[0])
    r = 0
    WCH = 30.0
    for k in range(K - 1):
        ta, tb = kf_t[k], kf_t[k + 1]
        dt = tb - ta
        Va, Vb = V_at(ta), V_at(tb)
        Pa, Pb = P_at(ta), P_at(tb)
        for ax in range(3):
            A[r, 6 * (k + 1) + 3 + ax] = WCH
            A[r, 6 * k + 3 + ax] = -WCH
            b[r] = (Vb[ax] - Va[ax]) * WCH
            r += 1
            A[r, 6 * (k + 1) + ax] = WCH
            A[r, 6 * k + ax] = -WCH
            A[r, 6 * k + 3 + ax] = -dt * WCH
            b[r] = ((Pb[ax] - Pa[ax]) - Va[ax] * dt) * WCH
            r += 1
    for oi, o in enumerate(rows):
        if drop is not None and oi in drop:
            continue
        trel = round(float(o[0]), 4)
        k = kf_of.get(trel)
        if k is None:
            continue
        gid = int(o[1])
        rel = o[2:5].copy()
        d = float(o[7])
        rhat = rel / max(np.linalg.norm(rel), 1e-6)
        # weights: bearing sigma ~0.003*d ; range sigma ~ 0.015*d^2/2.7
        w_perp = 1.0 / max(0.003 * d, 0.02)
        w_par = 1.0 / max(0.006 * d * d / 2.7, 0.05)
        Pperp = np.eye(3) - np.outer(rhat, rhat)
        W = w_perp * Pperp + w_par * np.outer(rhat, rhat)
        # equation rows: W @ (g - p_k - b*t*(z x rel)) = W @ rel
        cross = np.cross(ZHAT, rel) * float(o[0])
        for ax in range(3):
            wrow = W[ax]
            for c in range(3):
                if gid in KNOWN:
                    A[r, 6 * k + c] += -wrow[c] * -1.0   # -p term sign
                else:
                    A[r, u_of[gid] + c] += wrow[c]
                    A[r, 6 * k + c] += -wrow[c]
            A[r, B_COL] += -float(wrow @ cross)
            if gid in KNOWN:
                b[r] = float(wrow @ (KNOWN[gid] - rel))
            else:
                b[r] = float(wrow @ rel)
            r += 1
    sol = lsqr(A.tocsr()[:r], b[:r], atol=1e-11, btol=1e-11,
               iter_lim=40000)[0]
    return sol


sol = solve()
# robust pass: residuals of unknown-gate obs
drop = set()
for oi, o in enumerate(rows):
    gid = int(o[1])
    if gid not in KNOWN:
        continue          # unknown-gate obs are few and range-soft-weighted
    trel = round(float(o[0]), 4)
    k = kf_of.get(trel)
    if k is None:
        continue
    p_k = sol[6 * k:6 * k + 3]
    res = np.linalg.norm(KNOWN[gid] - p_k - o[2:5])
    if res > 2.0:
        drop.add(oi)
from collections import Counter
dropped_gates = Counter(int(rows[oi][1]) for oi in drop)
print(f'robust pass drop composition: {dict(dropped_gates)}')
for g in [10,12,13,15]:
    print(f'  first-solve g{g}:', sol[u_of[g]:u_of[g]+3].round(2))
print(f"robust pass: dropping {len(drop)} outlier obs; re-solving")
sol = solve(drop=drop)
print(f"yaw-rate bias: {np.degrees(sol[B_COL])*1000:.3f} mdeg/s... "
      f"(raw {sol[B_COL]:.6f})")

out = json.loads(json.dumps(chain["gates"]))
def wrapd(a):
    return (a + 180.0) % 360.0 - 180.0
for g in UNK:
    p = sol[u_of[g]:u_of[g] + 3]
    ys = rows[(rows[:, 1] == g)][:, 5]
    yw = 0.0
    if len(ys) >= 2:
        yr = np.radians(ys)
        yw = float(np.degrees(np.arctan2(np.median(np.sin(yr)),
                                         np.median(np.cos(yr)))))
    q = Rotation.from_euler("z", yw, degrees=True).as_quat()
    out[g]["pos"] = [float(v) for v in p]
    out[g]["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]),
                           float(q[2])]
    n_o = int((rows[:, 1] == g).sum())
    print(f"  g{g}: n_obs={n_o:3d} pos {np.round(p, 2)} yaw {yw:+.1f} "
          f"{'BELOW FLOOR!' if p[2] > -0.2 else ''}")
out[17]["visual"] = False
(D / "vq2_map_slam2.json").write_text(json.dumps(
    {"frame": "chain 0-9 + slam2 10-16", "gates": out,
     "certified": chain["certified"]}, indent=1))
print("wrote vq2_map_slam2.json")
