"""VQ2 gate map: loads the ground-truth map (aperture-centre positions,
NED metres, re-zeroed to gate 0) and expresses it in the drone's local
frame given the anchor transform (solved once from the spawn view).
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

MAP_PATH = Path(r"C:\Users\henry\Downloads\gate_map.json")


def load_vq2_map(anchor_t=(0, 0, 0), anchor_yaw_deg=0.0, path=MAP_PATH,
                 mirror_e=False):
    """Returns gates in our standard format (pos = APERTURE CENTRE — note:
    unlike VQ1 broadcasts, no panel offset applies; use hole-centred local
    corner models directly).

    anchor: local = R_z(anchor_yaw) @ map_rel_spawn + anchor_t
    """
    m = json.loads(Path(path).read_text())
    Rz = Rotation.from_euler("z", anchor_yaw_deg, degrees=True)
    gates = []
    for i, (p_rel, yaw) in enumerate(zip(m["gates_ring_center_NED_rel_spawn"],
                                         m["gate_yaw_deg"])):
        p_rel = np.asarray(p_rel, float)
        if mirror_e:
            p_rel = p_rel * np.array([1.0, -1.0, 1.0])
            yaw = -yaw
        p_local = Rz.apply(p_rel) + np.asarray(anchor_t, float)
        q = Rotation.from_euler("z", yaw + anchor_yaw_deg, degrees=True).as_quat()
        gates.append({
            "gate_id": i,
            "pos": [float(v) for v in p_local],   # aperture centre
            "quat_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
            "width": m["gate_size_m"]["outer"],
            "height": m["gate_size_m"]["outer"],
            "aperture": m["gate_size_m"]["inner_aperture"],
            "depth": m["gate_size_m"]["depth"],
        })
    return gates


def gate_quads_world_vq2(gate, panel_half=1.35, hole_half=0.75):
    """Hole + panel corners, both centred on the aperture centre (per the
    official spec drawing — VQ2 map positions ARE aperture centres)."""
    qw, qx, qy, qz = gate["quat_wxyz"]
    Rg = Rotation.from_quat([qx, qy, qz, qw])
    hole = np.array([[-hole_half, 0, -hole_half], [hole_half, 0, -hole_half],
                     [hole_half, 0, hole_half], [-hole_half, 0, hole_half]])
    panel = np.array([[-panel_half, 0, -panel_half], [panel_half, 0, -panel_half],
                      [panel_half, 0, panel_half], [-panel_half, 0, panel_half]])
    p = np.asarray(gate["pos"], float)
    return p + Rg.apply(hole), p + Rg.apply(panel)
