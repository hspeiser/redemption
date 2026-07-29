"""Gate map from human clicks + short IMU hops only.

Per click session (several fits of one gate over ~1s): each fit gives the
camera->gate vector in the gyro-attitude world frame; regressing those
vs time gives BOTH the gate-relative camera position AND velocity.
Between consecutive sessions (1-3s): p advances by v*dt + IMU double
integral (cm-level drift). Chaining anchors every gate to parked gate 0.

    .venv-train\\Scripts\\python.exe scripts\\vq2_click_chain.py ^
        --trace data\\vq2_trace_r6.npz --out data\\vq2_map_chain.json
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
from scripts.ekf_bringup import load_imu  # noqa: E402

D = REPO / "data"
EP = Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
          r"\ai-grand-prix\outputs\captures\rc_20260724_003153")
G_NED = np.array([0.0, 0.0, 9.81])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gap", type=float, default=4.0)
    args = ap.parse_args()

    tr = np.load(args.trace, allow_pickle=False)
    t_arr = tr["t"]

    # calib for click->camera vector (clicks stored as solved gate pos in
    # the session's map frame - NOT what we want. Instead re-derive the
    # camera->gate vector: journal stores solved world pos given that
    # session's belief. To stay session-independent we need the RAW
    # camera-frame vector; recover it: rel_cam = R_cb @ R_wb^T (g - p).
    # The journal doesn't store p/R... but the TRACE does, per frame, and
    # solved g = p + R_wc rel  =>  rel_world = g - p_frame. p_frame comes
    # from EACH session's own trace; sessions used different traces. To be
    # exact we store t + rel via the CURRENT trace only for sessions made
    # on it. Simplification: for every fit, rel_world_gyro is recoverable
    # ONLY with the session trace. We therefore recompute rel_world using
    # the session trace files recorded below.
    session_traces = {
        "vq2_map_human.json.journal.jsonl": "vq2_trace_003153.npz",
        "vq2_map_human3.json.journal.jsonl": "vq2_trace_r3.npz",
        "vq2_map_human4.json.journal.jsonl": "vq2_trace_r4.npz",
        "vq2_map_human5.json.journal.jsonl": "vq2_trace_r6.npz",
        "vq2_map_human7.json.journal.jsonl": "vq2_trace_full.npz",
        "vq2_map_human8.json.journal.jsonl": "vq2_trace_slam2.npz",
    }
    # gyro-pure attitude chain (shared, map-independent)
    imu = load_imu(EP)
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
    dt_s = np.diff(t_imu, prepend=t_imu[0])
    dt_s[(dt_s <= 0) | (dt_s > 0.5)] = 0
    Vc = np.cumsum(a_w * dt_s[:, None], axis=0)
    Pc = np.cumsum(Vc * dt_s[:, None], axis=0)

    def V_at(trel):
        return np.array([np.interp(trel + t0, t_imu, Vc[:, c])
                         for c in range(3)])

    def P_at(trel):
        return np.array([np.interp(trel + t0, t_imu, Pc[:, c])
                         for c in range(3)])

    # collect fits as (t_rel, rel_world_gyro, yaw_world, gate_vote)
    # rel_world_gyro = R_gyro(t) @ R_cb^T @ t_pnp; recover t_pnp from the
    # session's stored solved pos minus that session trace's belief pos,
    # then re-rotate out the session's belief attitude and re-apply gyro
    # attitude:  t_pnp = R_cb R_sess^T (g_solved - p_sess)
    from aigp.vision.labels import load_calib
    calib = load_calib(REPO / "data/calib/calib.json")
    R_cb = np.asarray(calib["R_cb"])

    # race status (for votes)
    iw = []
    with open(EP / "imu.jsonl") as fh:
        for line in fh:
            try:
                r0 = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "time_usec" in r0 and "wall" in r0:
                iw.append((r0["wall"], r0["time_usec"] * 1e-6))
    iw = np.array(iw)
    rs_t, rs_ag = [], []
    seen0 = False
    with open(EP / "mav.jsonl") as fh:
        for line in fh:
            if '"race_status"' not in line:
                continue
            r0 = json.loads(line)
            ag = int(r0["active_gate"])
            if ag == 0:
                seen0 = True
            if seen0:
                rs_t.append(float(np.interp(r0["wall"], iw[:, 0],
                                            iw[:, 1])) - t0)
                rs_ag.append(ag)
    rs_t = np.array(rs_t)
    rs_ag = np.array(rs_ag)

    def active_gate(trel):
        if len(rs_t) == 0 or trel < rs_t[0]:
            return 0
        return int(rs_ag[min(np.searchsorted(rs_t, trel, "right") - 1,
                             len(rs_ag) - 1)])

    obs = []
    for jname, tname in session_traces.items():
        jp = D / jname
        tp = D / tname
        if not jp.exists() or not tp.exists():
            continue
        st = np.load(tp, allow_pickle=False)
        for ln in jp.read_text().splitlines():
            if not ln.strip():
                continue
            r0 = json.loads(ln)
            if not r0.get("ok"):
                continue
            fidx = min(r0["frame"], len(st["t"]) - 1)
            trel = float(st["t"][fidx])
            sig_sess = float(st["sigma"][fidx])
            p_sess = st["pos"][fidx]
            qw, qx, qy, qz = st["quat"][fidx]
            R_sess = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            g_solved = np.asarray(r0["pos"], float)
            t_pnp = R_cb @ R_sess.T @ (g_solved - p_sess)
            k = int(np.searchsorted(t_imu, trel + t0))
            k = min(max(k, 0), n_i - 1)
            rel_w = Rws[k] @ R_cb.T @ t_pnp
            yaw_sess = r0["yaw"]
            # yaw in gyro frame: correct by (gyro yaw - session yaw) at k
            dyaw = (Rotation.from_matrix(
                Rws[k] @ R_sess.T).as_euler("zyx", degrees=True)[0])
            obs.append({"t": trel, "rel": rel_w,
                        "yaw": yaw_sess + dyaw,
                        "ag": active_gate(trel),
                        "p_abs": np.asarray(p_sess, float),
                        "sig": sig_sess})
    # append NET PnP observations — OFF by default: net fixes only exist
    # at 8-22m range where 0.8px = ~1m depth noise, which poisons the
    # session velocity regressions (verified empirically: broke gates 1-9)
    obs_npz = D / "vq2_obs_003153.npz"
    if "--with-net-obs" in sys.argv and obs_npz.exists():
        rows = np.load(obs_npz)["rows"]
        n_add = 0
        for r0 in rows:
            if r0[6] > 0.8 or r0[7] > 22.0:
                continue
            obs.append({"t": float(r0[0]), "rel": np.array(r0[2:5]),
                        "yaw": float(r0[5]), "ag": int(r0[1])})
            n_add += 1
        print(f"net observations added: {n_add}")
    obs.sort(key=lambda o: o["t"])
    print(f"total observations: {len(obs)}")

    # group into sessions: same ag & gaps < 1.5s
    sessions = []
    for o in obs:
        if sessions and o["ag"] == sessions[-1]["ag"] and \
                o["t"] - sessions[-1]["obs"][-1]["t"] < 1.5:
            sessions[-1]["obs"].append(o)
        else:
            sessions.append({"ag": o["ag"], "obs": [o]})
    print(f"sessions: {[(s['ag'], len(s['obs'])) for s in sessions]}")

    # per session: camera pos rel gate + velocity (regress rel vs t after
    # removing IMU acceleration curvature)
    for s in sessions:
        ts = np.array([o["t"] for o in s["obs"]])
        RELS = np.array([o["rel"] for o in s["obs"]])
        tm = ts.mean()
        # camera(t) = gate - rel(t); fit camera(t) = c0 + v (t-tm) + curv
        cam = -RELS
        curv = np.array([P_at(t) - P_at(tm) - V_at(tm) * (t - tm)
                         for t in ts])
        lin = cam - curv
        A = np.stack([np.ones_like(ts), ts - tm], axis=1)
        coef, *_ = np.linalg.lstsq(A, lin, rcond=None)
        s["tm"] = tm
        s["cam_tm"] = coef[0]          # camera pos at tm, RELATIVE to gate
        s["v_tm"] = coef[1] + V_at(tm) * 0 + (V_at(tm) - V_at(tm))
        s["v_tm"] = coef[1]
        s["yaw"] = float(np.median([o["yaw"] for o in s["obs"]]))
        n = len(ts)
        res = lin - A @ coef
        s["fit_rms"] = float(np.sqrt((res ** 2).sum(1).mean()))
        print(f"  ag={s['ag']:2d} n={n:2d} span {ts.max()-ts.min():.2f}s "
              f"vel {np.round(s['v_tm'],1)} fit-rms {s['fit_rms']*100:.0f}cm")

    # chain: gate positions. g_first known? anchor: session ag=0 clicked
    # parked -> camera at origin -> gate0 = -cam_tm(rel) ... cam_tm is
    # camera RELATIVE TO GATE, so gate0 = -cam_tm (camera at ~origin).
    gates_pos = {}
    gates_yaw = {}
    s0 = sessions[0]
    cam_world = {}
    # world camera position at s0.tm: parked -> 0 (+tiny)
    cam_world[0] = np.zeros(3)
    gates_pos[s0["ag"]] = cam_world[0] - s0["cam_tm"]
    gates_yaw[s0["ag"]] = s0["yaw"]
    for a, b in zip(range(len(sessions) - 1), range(1, len(sessions))):
        sa, sb = sessions[a], sessions[b]
        dt = sb["tm"] - sa["tm"]
        cam_a = gates_pos.get(sa["ag"])
        if cam_a is None:
            continue
        cam_a_world = gates_pos[sa["ag"]] + sa["cam_tm"]
        # best belief anchor available inside session b (the WORKING
        # 0-9 recipe: a solid absolute anchor beats a long IMU bridge)
        best_o = min(sb["obs"], key=lambda o: o.get("sig", 9e9))
        if dt > args.max_gap and best_o.get("sig", 9e9) < 0.20:
            cam_at_fit = best_o["p_abs"]
            cam_b_world = cam_at_fit + sb["v_tm"] * (sb["tm"]
                                                     - best_o["t"])
            src = f"BELIEF-ANCHOR (sig {best_o['sig']*100:.0f}cm)"
        else:
            if dt > args.max_gap:
                print(f"  gap {dt:.1f}s ag{sa['ag']}->{sb['ag']} exceeds "
                      f"{args.max_gap}s and no tight belief: bridging "
                      f"(drift!)")
            d_imu = P_at(sb["tm"]) - P_at(sa["tm"]) - V_at(sa["tm"]) * dt
            cam_b_world = cam_a_world + sa["v_tm"] * dt + d_imu
            src = f"imu dt {dt:.2f}s"
        gates_pos[sb["ag"]] = cam_b_world - sb["cam_tm"]
        gates_yaw[sb["ag"]] = sb["yaw"]
        print(f"  chained ag {sa['ag']} -> {sb['ag']} [{src}]: "
              f"gate{sb['ag']} at {np.round(gates_pos[sb['ag']], 2)}")

    prior = json.loads((D / "vq2_map_sim_94p62_0p91.json").read_text())[
        "gates"]
    gates = json.loads(json.dumps(prior))
    cert = []
    for gid, p in sorted(gates_pos.items()):
        if p[2] > -0.2:
            print(f"  g{gid} below floor -> dropped")
            continue
        q = Rotation.from_euler("z", gates_yaw[gid], degrees=True).as_quat()
        gates[gid]["pos"] = [float(v) for v in p]
        gates[gid]["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]),
                                   float(q[2])]
        cert.append(gid)
    gates[17]["visual"] = False
    Path(args.out).write_text(json.dumps(
        {"frame": "local spawn (click-chain)", "gates": gates,
         "certified": cert}, indent=1))
    print(f"certified: {cert} -> {args.out}")


if __name__ == "__main__":
    main()
