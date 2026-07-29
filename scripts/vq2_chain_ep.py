"""Chain one episode's click sessions over ITS OWN IMU (crash-free lap
back-half repair). Sessions are belief-anchored where the trace was tight
(<20 cm) and short-bridged otherwise. Output merges onto a base map.

    .venv-train\\Scripts\\python.exe scripts\\vq2_chain_ep.py ^
      --episode-dir <rc_...> --journal data\\vq2_map_human9.json.journal.jsonl ^
      --trace data\\vq2_trace_v3_101.npz --base data\\vq2_map_chain_v3.json ^
      --gates 9-16 --out data\\vq2_map_v4.json
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.ekf import GateEKF  # noqa: E402
from aigp.vision.labels import load_calib  # noqa: E402
from scripts.ekf_bringup import load_imu  # noqa: E402

G_NED = np.array([0.0, 0.0, 9.81])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-dir", required=True)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--trace", required=True,
                    help="the trace the editor session ran on (needed to "
                         "invert the click solves)")
    ap.add_argument("--anchor-trace", default=None,
                    help="optionally a BETTER trace of the same episode "
                         "for belief anchors (post-hoc improved map)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--gates", default="9-16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ep = Path(args.episode_dir)
    g_lo, g_hi = [int(x) for x in args.gates.split("-")]
    ALLOW = set(range(g_lo, g_hi + 1))

    calib = load_calib(REPO / "data/calib/calib.json")
    R_cb = np.asarray(calib["R_cb"])
    st = np.load(args.trace, allow_pickle=False)
    at = np.load(args.anchor_trace, allow_pickle=False) \
        if args.anchor_trace else st

    imu = load_imu(ep)
    # clip to longest monotonic segment (same as vq2_align)
    brk = np.where(np.diff(imu[:, 0]) < -0.5)[0]
    if len(brk):
        segs = np.split(np.arange(len(imu)), brk + 1)
        imu = imu[max(segs, key=len)]
    t_imu = imu[:, 0]
    t0 = t_imu[0]
    f_rest = imu[t_imu < t0 + 4.0, 1:4].mean(axis=0)
    pitch = np.arcsin(np.clip(f_rest[0] / 9.81, -1, 1))
    roll = np.arctan2(-f_rest[1], -f_rest[2])
    R = Rotation.from_euler("ZYX", [0.0, pitch, roll]).as_matrix()
    gs = np.asarray(GateEKF.GYRO_SIGN, float)
    n_i = len(imu)
    Rws = np.zeros((n_i, 3, 3))
    a_w = np.zeros((n_i, 3))
    prev = t_imu[0]
    for i in range(n_i):
        dt = t_imu[i] - prev
        prev = t_imu[i]
        if 0 < dt < 0.5:
            R = R @ Rotation.from_rotvec(imu[i, 4:7] * gs * dt).as_matrix()
        Rws[i] = R
        a_w[i] = R @ (imu[i, 1:4] * np.asarray(GateEKF.ACCEL_SIGN)) + G_NED
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
    for ln in Path(args.journal).read_text().splitlines():
        if not ln.strip():
            continue
        r0 = json.loads(ln)
        if not r0.get("ok"):
            continue
        fidx = min(r0["frame"], len(st["t"]) - 1)
        trel = float(st["t"][fidx])
        p_sess = st["pos"][fidx]
        qw, qx, qy, qz = st["quat"][fidx]
        R_sess = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        g_solved = np.asarray(r0["pos"], float)
        t_pnp = R_cb @ R_sess.T @ (g_solved - p_sess)
        k = min(max(int(np.searchsorted(t_imu, trel + t0)), 0), n_i - 1)
        rel_w = Rws[k] @ R_cb.T @ t_pnp
        dyaw = Rotation.from_matrix(Rws[k] @ R_sess.T).as_euler(
            "zyx", degrees=True)[0]
        aidx = min(fidx, len(at["t"]) - 1)
        obs.append({"t": trel, "rel": rel_w, "yaw": r0["yaw"] + dyaw,
                    "gate": int(r0["gate"]),
                    "p_abs": np.asarray(at["pos"][aidx], float),
                    "sig": float(at["sigma"][aidx])})
    obs.sort(key=lambda o: o["t"])
    print(f"fits: {len(obs)}")

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
        s["tm"] = tm
        s["cam_tm"] = coef[0]
        s["v_tm"] = coef[1]
        ys = np.radians([o["yaw"] for o in s["obs"]])
        s["yaw"] = float(np.degrees(np.arctan2(
            np.median(np.sin(ys)), np.median(np.cos(ys)))))
        best_o = min(s["obs"], key=lambda o: o["sig"])
        s["anchor"] = best_o
        res = lin - A @ coef
        print(f"  gate {s['gate']:2d}: n={len(ts):2d} span "
              f"{ts.max()-ts.min():.2f}s fit-rms "
              f"{np.sqrt((res**2).sum(1).mean())*100:.0f}cm "
              f"best-sig {best_o['sig']*100:.0f}cm")

    gates_pos = {}
    gates_yaw = {}
    prev_s = None
    for s in sessions:
        if s["anchor"]["sig"] < 0.20:
            cam_w = s["anchor"]["p_abs"] + s["v_tm"] * (
                s["tm"] - s["anchor"]["t"])
            src = f"belief (sig {s['anchor']['sig']*100:.0f}cm)"
        elif prev_s is not None and s["tm"] - prev_s["tm"] < 4.0 and \
                prev_s["gate"] in gates_pos:
            dt = s["tm"] - prev_s["tm"]
            cam_a = gates_pos[prev_s["gate"]] + prev_s["cam_tm"]
            d_imu = P_at(s["tm"]) - P_at(prev_s["tm"]) - \
                V_at(prev_s["tm"]) * dt
            cam_w = cam_a + prev_s["v_tm"] * dt + d_imu
            src = f"imu bridge {dt:.2f}s"
        else:
            print(f"  gate {s['gate']}: no anchor available, skipped")
            prev_s = s
            continue
        p = cam_w - s["cam_tm"]
        if p[2] > -0.2:
            print(f"  gate {s['gate']}: {np.round(p,2)} below floor "
                  f"[{src}] REJECTED")
            prev_s = s
            continue
        gates_pos[s["gate"]] = p
        gates_yaw[s["gate"]] = s["yaw"]
        print(f"  gate {s['gate']:2d} [{src}]: {np.round(p, 2)} "
              f"yaw {s['yaw']:+.1f}")
        prev_s = s

    base = json.loads(Path(args.base).read_text())
    gates = base["gates"]
    updated = []
    for g, p in sorted(gates_pos.items()):
        if g not in ALLOW:
            continue
        q = Rotation.from_euler("z", gates_yaw[g], degrees=True).as_quat()
        gates[g]["pos"] = [float(v) for v in p]
        gates[g]["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]),
                                 float(q[2])]
        updated.append(g)
    Path(args.out).write_text(json.dumps(
        {"frame": base.get("frame", "") + " + clean-lap repair",
         "gates": gates,
         "certified": sorted(set(base.get("certified", [])) |
                             set(updated))}, indent=1))
    print(f"updated gates {updated} -> {args.out}")


if __name__ == "__main__":
    main()
