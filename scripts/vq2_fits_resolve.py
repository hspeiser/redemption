"""Objective identity resolution for ALL human corner-fits: cluster fits
by 3D position (ignore user labels), assign each cluster the race-status
active gate at its frames' times. Outputs the certified map."""
import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data"
EP = Path(r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
          r"\ai-grand-prix\outputs\captures\rc_20260724_003153")

# ---- race-status timeline on the imu clock ----
iw = []
with open(EP / "imu.jsonl") as fh:
    for line in fh:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "time_usec" in r and "wall" in r:
            iw.append((r["wall"], r["time_usec"] * 1e-6))
iw = np.array(iw)
rs_t, rs_ag = [], []
seen0 = False
with open(EP / "mav.jsonl") as fh:
    for line in fh:
        if '"race_status"' not in line:
            continue
        r = json.loads(line)
        ag = int(r["active_gate"])
        if ag == 0:
            seen0 = True
        if seen0:
            rs_t.append(float(np.interp(r["wall"], iw[:, 0], iw[:, 1])))
            rs_ag.append(ag)
rs_t = np.array(rs_t)
rs_ag = np.array(rs_ag)
t0_imu = iw[0, 1]


def active_gate(t_rel):
    t = t_rel + t0_imu
    if len(rs_t) == 0 or t < rs_t[0]:
        return 0
    return int(rs_ag[min(np.searchsorted(rs_t, t, "right") - 1,
                         len(rs_ag) - 1)])


# ---- gather ALL journal fits (every session) ----
tr = np.load(sys.argv[1], allow_pickle=False)   # trace used by the editor
t_arr = tr["t"]
fits = []
for jp in sorted(D.glob("vq2_map_human*.journal.jsonl")):
    if "human2." in jp.name:      # round-2: contaminated trajectory era
        print(f"  (skipping {jp.name} - poisoned-trajectory session)")
        continue
    for ln in jp.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("ok"):
            fits.append(r)
print(f"total journal fits: {len(fits)}")

# ---- cluster by 3D position ----
clusters = []
for r in fits:
    p = np.asarray(r["pos"], float)
    placed = False
    for c in clusters:
        if np.linalg.norm(p - np.mean(c["P"], axis=0)) < 1.2:
            c["P"].append(p)
            c["yaw"].append(r["yaw"])
            c["frames"].append(r["frame"])
            placed = True
            break
    if not placed:
        clusters.append({"P": [p], "yaw": [r["yaw"]],
                         "frames": [r["frame"]]})
print(f"clusters: {len(clusters)}")

# ---- assign race index by race status at the cluster's frames ----
out_gates = {}
for c in clusters:
    if len(c["P"]) < 2:
        continue
    P = np.array(c["P"])
    pm = np.median(P, axis=0)
    spread = float(np.linalg.norm(P - pm, axis=1).max())
    ags = [active_gate(t_arr[min(f, len(t_arr) - 1)]) for f in c["frames"]]
    vals, counts = np.unique(ags, return_counts=True)
    gid = int(vals[np.argmax(counts)])
    ys = np.radians(c["yaw"])
    ym = float(np.degrees(np.arctan2(np.median(np.sin(ys)),
                                     np.median(np.cos(ys)))))
    print(f"  cluster n={len(P):2d} spread {spread*100:5.0f}cm  "
          f"race-gate votes {dict(zip(vals.tolist(), counts.tolist()))} "
          f"-> g{gid}  pos {np.round(pm, 2)} yaw {ym:+.1f}")
    if pm[2] > -0.2 or pm[2] < -15.0:
        print(f"    REJECTED: z={pm[2]:.1f} is below floor / absurd "
              f"(bad PnP face or reflection click)")
        continue
    if spread < 0.7 and len(P) >= 3:
        prev = out_gates.get(gid)
        if prev is None or len(P) > prev["n"]:
            out_gates[gid] = {"pos": pm, "yaw": ym, "n": len(P),
                              "spread": spread}

# ---- build map: resolved gates exact, prior elsewhere; g17 non-visual ----
prior = json.loads((D / "vq2_map_sim_94p62_0p91.json").read_text())["gates"]
gates = json.loads(json.dumps(prior))
cert = []
for gid, v in sorted(out_gates.items()):
    q = Rotation.from_euler("z", v["yaw"], degrees=True).as_quat()
    gates[gid]["pos"] = [float(x) for x in v["pos"]]
    gates[gid]["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]),
                               float(q[2])]
    cert.append(gid)
gates[17]["visual"] = False       # finish line: no physical gate
(D / "vq2_map_resolved.json").write_text(json.dumps(
    {"frame": "local spawn (race-status resolved)", "gates": gates,
     "certified": cert}, indent=1))
print(f"certified (race-indexed): {cert} -> vq2_map_resolved.json")
