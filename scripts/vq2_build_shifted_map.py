"""Final map construction: Henry's measured gates exact; all others from
the +1-shifted rigid fit of the raw file (positions ~2-5m prior, yaws
accurate at offset +179.4). Race gate 17 extrapolated (no entry 18)."""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

truth = json.loads(Path(
    r"C:\Users\henry\Downloads\gate_map.json").read_text())
rel = np.asarray(truth["gates_ring_center_NED_rel_spawn"], float)
yaws = np.asarray(truth["gate_yaw_deg"], float)
D = Path(r"C:\Users\henry\Desktop\ai-gp\data")
human = json.loads((D / "vq2_map_r2.json").read_text())["gates"]
MEAS = [0, 1, 4, 5, 6]

P = np.array([human[i]["pos"] for i in MEAS])
Q = np.array([rel[i + 1] for i in MEAS])
Pc, Qc = P.mean(0), Q.mean(0)
A = P - Pc
B = Q - Qc
num = sum(b[0] * a[1] - b[1] * a[0] for a, b in zip(A, B))
den = sum(b[0] * a[0] + b[1] * a[1] for a, b in zip(A, B))
th = np.arctan2(num, den)
c, s = np.cos(th), np.sin(th)
Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
t0 = Pc - Rz @ Qc

# yaw offset from measured gates
def hyaw(g):
    q = g["quat_wxyz"]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler(
        "zyx", degrees=True)[0]

offs = [(hyaw(human[i]) - yaws[i + 1] + 180) % 360 - 180 for i in MEAS]
yoff = float(np.median(offs))
print(f"fit: rot {np.degrees(th):+.2f} t0 {np.round(t0,2)} "
      f"yaw-offset {yoff:+.1f} (spread {np.ptp(offs):.1f})")

gates = json.loads(json.dumps(human))
for k in range(18):
    if k in MEAS:
        continue
    if k + 1 < 18:
        p = Rz @ rel[k + 1] + t0
        yw = yaws[k + 1] + yoff
    else:                       # race gate 17: extrapolate last segment
        p = Rz @ (rel[17] + (rel[17] - rel[16])) + t0
        yw = yaws[17] + yoff
    q = Rotation.from_euler("z", yw, degrees=True).as_quat()
    gates[k]["pos"] = [float(v) for v in p]
    gates[k]["quat_wxyz"] = [float(q[3]), float(q[0]), float(q[1]),
                             float(q[2])]
(D / "vq2_map_shift.json").write_text(json.dumps(
    {"frame": "local spawn (human + shifted prior)", "gates": gates},
    indent=1))
print("wrote vq2_map_shift.json")
