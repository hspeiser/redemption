"""Fit the VQ2 map rotation from race-status PASS TIMES + IMU integration.

At each active_gate increment the drone is AT that gate (~1-2m). For each
consecutive pass triple (g, g+1, g+2), IMU double-integration gives the
trajectory shape exactly up to the unknown entry velocity, which cancels:

    G(g+2) - G(g+1) - k (G(g+1) - G(g)) = D2 + V1*dt2 - k D1,  k = dt2/dt1

LHS = Rz(A) @ M @ u  with u from the raw map -> per-triple implied A and a
length self-check. No detector, no PnP, no identity guessing.

    .venv-train\\Scripts\\python.exe scripts\\vq2_pass_fit.py <episode_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ekf import GateEKF  # noqa: E402
from scripts.ekf_bringup import load_imu  # noqa: E402

MAP_PATH = Path(r"C:\Users\henry\Downloads\gate_map.json")
G_NED = np.array([0.0, 0.0, 9.81])


def main():
    ep = Path(sys.argv[1])
    m0 = json.loads(MAP_PATH.read_text())
    rel = np.asarray(m0["gates_ring_center_NED_rel_spawn"], float)

    imu = load_imu(ep)
    t = imu[:, 0]

    # rest attitude (same recipe as vq2_align)
    gmag = np.abs(imu[:, 4:7]).max(axis=1)
    moving = np.convolve((gmag > 0.05).astype(float), np.ones(12) / 12,
                         "same") > 0.5
    k_move = int(np.argmax(moving)) if moving.any() else len(imu)
    m_rest = t <= t[max(k_move - 1, 0)]
    if m_rest.sum() < 60:
        m_rest = t < t[0] + 3.0
    f_rest = imu[m_rest, 1:4].mean(axis=0) * GateEKF.ACCEL_SIGN
    pitch = np.arcsin(np.clip(f_rest[0] / 9.81, -1, 1))
    roll = np.arctan2(-f_rest[1], -f_rest[2])
    R = Rotation.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()

    # integrate: attitude chain + world accel -> v(t), p(t) increments
    n = len(imu)
    a_w = np.zeros((n, 3))
    gs = np.asarray(GateEKF.GYRO_SIGN, float)
    as_ = np.asarray(GateEKF.ACCEL_SIGN, float)
    prev = t[0]
    for i in range(n):
        dt = t[i] - prev
        prev = t[i]
        if 0 < dt < 0.5:
            R = R @ Rotation.from_rotvec(imu[i, 4:7] * gs * dt).as_matrix()
        a_w[i] = R @ (imu[i, 1:4] * as_) + G_NED
    dt_s = np.diff(t, prepend=t[0])
    dt_s[(dt_s < 0) | (dt_s > 0.5)] = 0
    v = np.cumsum(a_w * dt_s[:, None], axis=0)
    p = np.cumsum(v * dt_s[:, None], axis=0)

    def interp(arr, tq):
        out = np.empty(3)
        for c in range(3):
            out[c] = np.interp(tq, t, arr[:, c])
        return out

    # pass times: active_gate increments on the imu clock
    iw = []
    with open(ep / "imu.jsonl") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "time_usec" in r and "wall" in r:
                iw.append((r["wall"], r["time_usec"] * 1e-6))
    iw = np.array(iw)
    passes = {}          # gate just PASSED -> imu time
    prev_ag = None
    with open(ep / "mav.jsonl") as fh:
        for line in fh:
            if '"race_status"' not in line:
                continue
            r = json.loads(line)
            ag = int(r["active_gate"])
            ti = float(np.interp(r["wall"], iw[:, 0], iw[:, 1]))
            if prev_ag is not None and ag == prev_ag + 1:
                passes[prev_ag] = ti      # gate prev_ag passed now
            prev_ag = ag if (prev_ag is None or ag >= prev_ag or ag == 0) \
                else prev_ag
            if prev_ag is None or (ag == 0 and prev_ag != 0):
                prev_ag = ag
    print(f"passes detected: {sorted(passes)}")

    seq = sorted(passes)
    rows = []
    for a in range(len(seq) - 2):
        g0, g1, g2 = seq[a], seq[a + 1], seq[a + 2]
        if g1 != g0 + 1 or g2 != g1 + 1:
            continue
        t0, t1, t2 = passes[g0], passes[g1], passes[g2]
        dt1, dt2 = t1 - t0, t2 - t1
        if not (0.3 < dt1 < 8 and 0.3 < dt2 < 8):
            continue
        k = dt2 / dt1
        # D1 = p(t1)-p(t0)-v(t0)dt1 ; D2 likewise from t1
        D1 = interp(p, t1) - interp(p, t0) - interp(v, t0) * dt1
        D2 = interp(p, t2) - interp(p, t1) - interp(v, t1) * dt2
        V1 = interp(v, t1) - interp(v, t0)
        rhs = D2 + 0 * V1 - k * D1   # v cancels: v1 = v0 + V1 already in D2?
        # careful: D2 uses v(t1) which equals v(t0)+V1; eliminating v(t0):
        # G2-G1 = v(t0)dt2 + V1 dt2 + D2  and  G1-G0 = v(t0)dt1 + D1
        # -> (G2-G1) - k (G1-G0) = V1 dt2 + D2 - k D1
        rhs = V1 * dt2 + D2 - k * D1
        for s in (-1.0, 1.0):
            relc = rel * np.array([1.0, s, 1.0])
            u = relc[g2] - relc[g1] - k * (relc[g1] - relc[g0])
            if np.linalg.norm(u[:2]) < 1.0:
                continue
            A = np.degrees(np.arctan2(rhs[1], rhs[0]) -
                           np.arctan2(u[1], u[0]))
            A = (A + 180) % 360 - 180
            dlen = np.linalg.norm(rhs[:2]) - np.linalg.norm(u[:2])
            rows.append((s, g0, A, dlen, float(np.linalg.norm(u[:2]))))

    for s in (-1.0, 1.0):
        rs = [r for r in rows if r[0] == s]
        if not rs:
            continue
        As = np.radians([r[2] for r in rs])
        w = np.array([r[4] / (1 + abs(r[3])) for r in rs])
        mean_A = np.degrees(np.arctan2((w * np.sin(As)).sum(),
                                       (w * np.cos(As)).sum()))
        dev = np.abs((np.degrees(As) - mean_A + 180) % 360 - 180)
        print(f"\nmirror={'yes' if s < 0 else 'no'}: n={len(rs)}  "
              f"A = {mean_A:+7.2f} deg  MAD {np.median(dev):5.2f}  "
              f"len-resid median "
              f"{np.median([abs(r[3]) for r in rs]):.2f}m")
        for (s0, g0, A, dlen, L) in rs:
            print(f"   triple g{g0}-{g0+2}: A {A:+7.1f}  "
                  f"dlen {dlen:+6.2f}m  |u| {L:5.1f}m")


if __name__ == "__main__":
    main()
