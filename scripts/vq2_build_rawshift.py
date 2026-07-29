"""Build a non-circular VQ2 map from the simulator's extracted constellation.

Race gate 0 corresponds to entry 1 in gate_map.json.  Fit one rigid XY
transform plus a Z translation from raw entries 1..10 to the independently
measured race gates 0..9, freeze those measured gates, then use the transformed
raw entries 11..17 for race gates 10..16.

No EKF tail pose, IMU integration, or gates 10..16 are used by this fit.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RAW = Path(r"C:\Users\henry\Downloads\gate_map.json")
DEFAULT_ORIENTATION = Path(
    r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
    r"\ai-grand-prix\outputs\gate_map_truth.json"
)


def wrap_deg(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def gate_yaw(gate):
    qw, qx, qy, qz = gate["quat_wxyz"]
    return float(Rotation.from_quat([qx, qy, qz, qw])
                 .as_euler("zyx", degrees=True)[0])


def full_gate_rotation(pitch_deg, through_yaw_deg, roll_deg):
    """Gate local frame -> NED world.

    Our gate model uses local x = frame-right, local y = back-face normal,
    and local z = frame-down.  The pak spline orientation describes the
    forward through-normal in Unreal pitch/yaw/roll convention.
    """
    pitch, yaw, roll = np.radians(
        [pitch_deg, through_yaw_deg, roll_deg])
    normal = np.array([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch),
    ])
    down = np.array([0.0, 0.0, 1.0])
    right = np.cross(down, normal)
    right /= np.linalg.norm(right)
    gate_down = np.cross(normal, right)
    gate_down /= np.linalg.norm(gate_down)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        right, gate_down = (
            c * right + s * gate_down,
            -s * right + c * gate_down,
        )
    return np.column_stack((right, -normal, gate_down))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "data/vq2_map_hop.json"))
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--orientation", default=str(DEFAULT_ORIENTATION),
                    help="pak truth JSON containing full pitch/yaw/roll")
    ap.add_argument("--out",
                    default=str(REPO / "data/vq2_map_rawshift.json"))
    ap.add_argument("--last-anchor", type=int, default=9)
    args = ap.parse_args()

    base_doc = json.loads(Path(args.base).read_text())
    raw_doc = json.loads(Path(args.raw).read_text())
    orientation_doc = json.loads(Path(args.orientation).read_text())
    gates = json.loads(json.dumps(base_doc["gates"]))
    raw = np.asarray(raw_doc["gates_ring_center_NED"], float)
    raw_yaw = np.asarray(raw_doc["gate_yaw_deg"], float)
    raw_pyr = np.asarray(
        orientation_doc["orientation_ue_pitch_yaw_roll_deg"], float)

    ids = np.arange(args.last_anchor + 1)
    if ids[-1] + 1 >= len(raw):
        raise ValueError("shifted raw map does not cover the anchor range")
    measured = np.asarray([gates[i]["pos"] for i in ids], float)
    source = raw[ids + 1]

    # Row-vector Kabsch fit in XY.  Z is already NED, so it needs only an
    # origin translation.
    src_xy = source[:, :2]
    dst_xy = measured[:, :2]
    src_c = src_xy - src_xy.mean(axis=0)
    dst_c = dst_xy - dst_xy.mean(axis=0)
    u, _s, vt = np.linalg.svd(src_c.T @ dst_c)
    r_xy = u @ vt
    if np.linalg.det(r_xy) < 0:
        u[:, -1] *= -1
        r_xy = u @ vt
    t_xy = dst_xy.mean(axis=0) - src_xy.mean(axis=0) @ r_xy
    t_z = float(np.median(measured[:, 2] - source[:, 2]))

    fitted = np.column_stack((src_xy @ r_xy + t_xy,
                              source[:, 2] + t_z))
    err = np.linalg.norm(fitted - measured, axis=1)
    if np.max(err) > 1.0:
        raise RuntimeError(
            f"shift fit failed sanity check: max anchor error {err.max():.2f}m")

    yaw_offsets = wrap_deg(np.array([gate_yaw(gates[i]) for i in ids])
                           - raw_yaw[ids + 1])
    # All trusted gates agree near +/-180; unwrap around the first observation
    # before taking the median.
    yaw_ref = yaw_offsets[0]
    yaw_off = float(yaw_ref + np.median(wrap_deg(yaw_offsets - yaw_ref)))
    yaw_off = float(wrap_deg(yaw_off))

    # Preserve independent positions through last_anchor. Populate only the
    # tail positions. Full pak pitch/roll is restored for every physical gate:
    # race gate 9 (raw entry 10) is tilted -20 degrees and was the exact point
    # where the old yaw-only EKF began to diverge.
    for race_gate in range(args.last_anchor + 1, min(17, len(gates))):
        raw_gate = race_gate + 1
        p = np.array([
            *(raw[raw_gate, :2] @ r_xy + t_xy),
            raw[raw_gate, 2] + t_z,
        ])
        gates[race_gate]["pos"] = [float(v) for v in p]

    position_yaw = float(np.degrees(np.arctan2(
        r_xy[0, 1], r_xy[0, 0])))
    for race_gate in range(min(17, len(gates))):
        raw_gate = race_gate + 1
        pitch, yaw, roll = raw_pyr[raw_gate]
        rg = full_gate_rotation(pitch, yaw + position_yaw, roll)
        q = Rotation.from_matrix(rg).as_quat()
        gates[race_gate]["quat_wxyz"] = [
            float(q[3]), float(q[0]), float(q[1]), float(q[2])
        ]

    out = {
        "frame": (
            "raw-shift candidate: race g0..g9 fixed; race g10..g16 = "
            "rigid transform of extracted raw gates 11..17"
        ),
        "gates": gates,
        "fit": {
            "race_to_raw_index_offset": 1,
            "anchors": [int(i) for i in ids],
            "anchor_error_m": [float(v) for v in err],
            "anchor_error_median": float(np.median(err)),
            "anchor_error_max": float(np.max(err)),
            "xy_rotation_deg": position_yaw,
            "xy_translation": [float(v) for v in t_xy],
            "z_translation": t_z,
            "gate_yaw_offset_deg": yaw_off,
            "full_orientation_source": str(Path(args.orientation)),
            "race_gate_9_pitch_deg": float(raw_pyr[10, 0]),
            "race_gate_16_roll_deg": float(raw_pyr[17, 2]),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"anchors g0..g{ids[-1]}: median {np.median(err):.3f}m, "
          f"max {np.max(err):.3f}m")
    print(f"position rotation {out['fit']['xy_rotation_deg']:+.3f}deg, "
          f"yaw offset {yaw_off:+.3f}deg")
    for i, e in zip(ids, err):
        print(f"  anchor g{i:2d}: {e:.3f}m")
    for i in range(args.last_anchor + 1, min(17, len(gates))):
        print(f"  tail   g{i:2d}: {np.round(gates[i]['pos'], 3)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
