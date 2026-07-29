"""Branch-disambiguation stills: for each click session on gate g, pin
the camera from the click itself (exact, trajectory-free), then project
gate g+1 from each candidate map in a distinct color. The real next gate
in the image picks the winner. Henry's eyes are the referee."""
import json
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from aigp.vision.labels import load_calib  # noqa: E402
import cv2  # noqa: E402

D = REPO / "data"
OUT = Path(r"C:\Users\henry\.claude\jobs\1eb1f991\tmp\branch")
OUT.mkdir(parents=True, exist_ok=True)
W, H = 640, 360
HOLE, PANEL = 0.75, 1.36
SQ = np.array([[-PANEL, 0, -PANEL], [PANEL, 0, -PANEL],
               [PANEL, 0, PANEL], [-PANEL, 0, PANEL]])
SQH = np.array([[-HOLE, 0, -HOLE], [HOLE, 0, -HOLE],
                [HOLE, 0, HOLE], [-HOLE, 0, HOLE]])

calib = load_calib(REPO / "data/calib/calib.json")
fx, fy, cx, cy = calib["K"]
R_cb = np.asarray(calib["R_cb"])

CANDS = {
    "v8_GREEN": ("vq2_map_v8.json", (0, 255, 0)),
    "v10_YELLOW": ("vq2_map_v10.json", (0, 255, 255)),
    "file_CYAN": ("vq2_map_full.json", (255, 255, 0)),
}
maps = {}
for name, (f, col) in CANDS.items():
    maps[name] = (json.loads((D / f).read_text())["gates"], col)

tr = np.load(D / "vq2_trace_v3_101.npz", allow_pickle=False)

def gate_R(g):
    q = g["quat_wxyz"]
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

# pick per gate the click with the LONGEST next-gate visibility: use the
# EARLIEST fit of each session (gate still ahead, next gate more likely
# in frame)
fits = []
for jn in ("vq2_map_human9.json.journal.jsonl",
           "vq2_map_human10.json.journal.jsonl"):
    for ln in (D / jn).read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("ok"):
            fits.append(r)
by_gate = {}
for r in fits:
    by_gate.setdefault(r["gate"], []).append(r)

for g in range(8, 16):
    if g not in by_gate:
        continue
    r0 = min(by_gate[g], key=lambda r: r["frame"])   # earliest fit
    fidx = min(r0["frame"], len(tr["t"]) - 1)
    img = cv2.imread(str(tr["path"][fidx]))
    if img is None:
        continue
    # camera pose FROM THE CLICK: use the session-trace attitude (belief
    # attitude, vision-corrected, reliable) + click-solved position:
    # p_cam = g_solved_pos - offset ... the click solve stored the gate
    # in world via belief pose, so belief pose IS the camera pose used;
    # projecting candidate NEXT-gate positions RELATIVE to the candidate's
    # own gate-g position cancels absolute error:
    p_sess = tr["pos"][fidx]
    qw, qx, qy, qz = tr["quat"][fidx]
    R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    R_cw = R_cb @ R_wb.T
    g_clicked = np.asarray(r0["pos"], float)   # where HIS click put gate g
    vis = img.copy()
    for name, (gates, col) in maps.items():
        gg = np.asarray(gates[g]["pos"])
        gn = np.asarray(gates[g + 1]["pos"])
        rel_next = gn - gg                      # candidate's g->g+1 vector
        p_next = g_clicked + rel_next           # anchored on HIS gate g
        Rg = gate_R(gates[g + 1])
        for quad, th in ((SQH, 2), (SQ, 1)):
            pts = []
            ok = True
            for c in quad:
                Xc = R_cw @ (p_next + Rg @ c - p_sess)
                if Xc[2] < 0.4:
                    ok = False
                    break
                u = fx * Xc[0] / Xc[2] + cx
                v = fy * Xc[1] / Xc[2] + cy
                if not (-3000 < u < 3000 and -3000 < v < 3000):
                    ok = False
                    break
                pts.append([u, v])
            if ok:
                cv2.polylines(vis, [np.array(pts, np.int32)], True, col, th)
    cv2.putText(vis, f"clicked gate {g} -> candidates for gate {g+1}: "
                "GREEN=v8 YELLOW=v10 CYAN=file", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(OUT / f"edge_{g}_{g+1}.jpg"), vis)
    print(f"edge {g}->{g+1} written (frame {fidx})")
