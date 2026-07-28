"""Joint linear SLAM for the VQ2 map: solve keyframe positions/velocities
AND all 18 gate positions from IMU integration + PnP gate observations.
No prior map anywhere. Frame: gyro-chain attitude from rest, p0 = 0.

Unknowns: p_k, v_k per keyframe + g_j per gate. All constraints linear:
  p_{k+1} = p_k + v_k dt + D_k     (D_k = IMU double integral, known)
  v_{k+1} = v_k + V_k              (V_k = IMU integral, known)
  g_{obs} = p_k + rel_world        (PnP observation, known vector)
Anchors: p_0 = 0. Sparse lsqr per axis.

    .venv-train\\Scripts\\python.exe scripts\\vq2_slam.py ^
        --episode-dir <rc_...> --obs data\\vq2_obs.npz ^
        --out data\\vq2_map_slam.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ekf import GateEKF  # noqa: E402
from scripts.ekf_bringup import load_imu  # noqa: E402

G_NED = np.array([0.0, 0.0, 9.81])


def wrap(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def circ_median(deg):
    deg = np.asarray(deg, float)
    best, bc = None, None
    for c in deg:
        cost = np.abs(wrap(deg - c)).sum()
        if bc is None or cost < bc:
            best, bc = float(c), cost
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rms-max", type=float, default=0.8)
    ap.add_argument("--depth-max", type=float, default=28.0)
    args = ap.parse_args()

    rows = np.load(args.obs)["rows"]
    # (t, ag, relx, rely, relz, yaw, rms, depth)
    m = (rows[:, 6] < args.rms_max) & (rows[:, 7] < args.depth_max)
    # drop observations within 0.4s of an active-gate transition (identity
    # ambiguous mid-pass)
    ags = rows[:, 1]
    t_r = rows[:, 0]
    keep = m.copy()
    tr_t = [t_r[i] for i in range(1, len(rows)) if ags[i] != ags[i - 1]]
    for tt in tr_t:
        keep &= ~(np.abs(t_r - tt) < 0.4)
    rows = rows[keep]
    print(f"obs kept: {len(rows)} (rms<{args.rms_max}, "
          f"depth<{args.depth_max}, away from transitions)")

    imu = load_imu(Path(args.episode_dir))
    t_imu = imu[:, 0]
    t0 = t_imu[0]

    # gyro-chain attitude + world-accel integrals
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
    dt_s = np.diff(t_imu, prepend=t_imu[0])
    dt_s[(dt_s <= 0) | (dt_s > 0.5)] = 0
    Vc = np.cumsum(a_w * dt_s[:, None], axis=0)          # ∫a dt
    Pc = np.cumsum(Vc * dt_s[:, None], axis=0)           # ∫∫

    # keyframes: unique obs times (relative t) mapped onto imu clock
    kf_t = np.unique(np.round(rows[:, 0], 4))
    K = len(kf_t)
    print(f"keyframes: {K}")

    def interp(arr, tq):
        return np.array([np.interp(tq + t0 - 0.0, t_imu, arr[:, c])
                         for c in range(3)])

    # per-axis sparse system. unknowns: [p(K), v(K), g(18)] per axis
    NU = K + K + 18
    n_gate_obs = len(rows)
    n_eq = 1 + (K - 1) * 2 + n_gate_obs + 1     # p0 anchor + chains + obs
    sols = []
    kf_idx = {round(t, 4): k for k, t in enumerate(kf_t)}
    gate_obs_count = {}
    for ax in range(3):
        A = lil_matrix((n_eq, NU))
        b = np.zeros(n_eq)
        r = 0
        A[r, 0] = 1.0                                    # p0 = 0
        b[r] = 0.0
        r += 1
        W_CHAIN = 10.0
        for k in range(K - 1):
            ta, tb = kf_t[k], kf_t[k + 1]
            dt = tb - ta
            Va = interp(Vc, ta)[ax]
            Vb = interp(Vc, tb)[ax]
            Pa = interp(Pc, ta)[ax]
            Pb = interp(Pc, tb)[ax]
            # v_b = v_a + (Vb - Va)
            A[r, K + k + 1] = 1.0 * W_CHAIN
            A[r, K + k] = -1.0 * W_CHAIN
            b[r] = (Vb - Va) * W_CHAIN
            r += 1
            # p_b = p_a + v_a dt + [(Pb - Pa) - Va*dt]
            A[r, k + 1] = 1.0 * W_CHAIN
            A[r, k] = -1.0 * W_CHAIN
            A[r, K + k] = -dt * W_CHAIN
            b[r] = ((Pb - Pa) - Va * dt) * W_CHAIN
            r += 1
        for o in rows:
            k = kf_idx[round(o[0], 4)]
            gid = int(o[1])
            w = 1.0 / max(0.03 * o[7], 0.05)     # relative accuracy ~ depth
            A[r, 2 * K + gid] = 1.0 * w
            A[r, k] = -1.0 * w
            b[r] = o[2 + ax] * w
            r += 1
            if ax == 0:
                gate_obs_count[gid] = gate_obs_count.get(gid, 0) + 1
        sol = lsqr(A.tocsr(), b, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
        sols.append(sol)
    X = np.stack(sols, axis=1)     # (NU, 3)
    gates_p = X[2 * K:]

    # per-gate yaw from observations
    yaws_out = []
    for gid in range(18):
        ys = rows[rows[:, 1] == gid][:, 5]
        yaws_out.append(circ_median(ys) if len(ys) >= 3 else None)

    gates = []
    for gid in range(18):
        yw = yaws_out[gid] if yaws_out[gid] is not None else 0.0
        q = Rotation.from_euler("z", yw, degrees=True).as_quat()
        gates.append({
            "gate_id": gid,
            "pos": [float(v) for v in gates_p[gid]],
            "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]),
                          float(q[2])],
            "n_obs": int(gate_obs_count.get(gid, 0)),
            "yaw_observed": yaws_out[gid] is not None,
            "width": 2.72, "height": 2.72, "aperture": 1.5, "depth": 0.26})
    Path(args.out).write_text(json.dumps(
        {"frame": "local spawn (joint SLAM, no prior)", "gates": gates},
        indent=1))
    print(f"wrote {args.out}")
    for g in gates:
        print(f"  g{g['gate_id']:2d} n_obs={g['n_obs']:4d} "
              f"pos {np.round(g['pos'], 2)} "
              f"yaw {Rotation.from_quat([g['quat_wxyz'][1], g['quat_wxyz'][2], g['quat_wxyz'][3], g['quat_wxyz'][0]]).as_euler('zyx', degrees=True)[0]:+7.1f}"
              f"{'' if g['yaw_observed'] else '  (no yaw obs)'}")


if __name__ == "__main__":
    main()
