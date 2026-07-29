"""Probe VQ1's command-response signs + frame conventions to build the twin->VQ1 adapter.

For each rate axis (roll, pitch, yaw): hover, pulse the command, measure the body-rate response
sign and gain from odometry. Also sanity-print the obs-frame facts (gravity/gate direction in the
body frame at rest) so the observation adapter can be verified.

  python -m racer_state.probe_signs
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from racer_state.config import CFG
from racer_state.mav_io import MavLink
from racer_state.geom import rot_world_to_body, gate_normal_world
from racer_state.train import wait_reset


def stream(mav, cmd, dur, record=False):
    """Hold a command for dur seconds (continuous att stream + hb). Optionally record rates."""
    out = []
    t0 = time.time(); last_hb = 0.0
    while time.time() - t0 < dur:
        mav.att(*cmd)
        if time.time() - last_hb > 0.3:
            mav.hb(); last_hb = time.time()
        if record:
            out.append(np.array(mav.rates, float))
        time.sleep(0.01)
    return np.array(out) if record else None


def main():
    mav = MavLink(CFG.mav_addr)
    print(f"connected sys={mav.sys}", flush=True)
    ok = mav.capture_gates()
    gates = mav.gates or []
    print(f"gates: {len(gates)} (capture ok={ok})", flush=True)
    wait_reset(mav); time.sleep(CFG.reset_settle_s)

    q = mav.quat.copy()
    grav_body = rot_world_to_body(q, np.array([0.0, 0.0, 1.0]))   # NED down
    print(f"AT REST: grav_body(NED down rotated) = {np.round(grav_body, 3)}  (FRD expects ~[0,0,1])")
    if gates:
        rel = rot_world_to_body(q, gates[0]["pos"] - mav.pos)
        nrm = rot_world_to_body(q, gate_normal_world(gates[0]["quat"]))
        print(f"AT REST: rel_gate0_body = {np.round(rel, 2)}  (expect ~[+23, small, small])")
        print(f"AT REST: gate0_normal_body = {np.round(nrm, 3)}  (expect ~[+1, 0, 0] = forward)")

    hover = CFG.thrust_hover
    mav.arm(True)
    stream(mav, (0, 0, 0, hover + 0.06), 0.8)     # small climb to get off the floor
    stream(mav, (0, 0, 0, hover), 0.5)

    names = ["roll", "pitch", "yaw"]
    results = {}
    for axis in range(3):
        stream(mav, (0, 0, 0, hover), 0.5)                     # settle
        cmd = [0.0, 0.0, 0.0, hover]
        AMP = 0.4
        cmd[axis] = AMP
        rec = stream(mav, tuple(cmd), 0.45, record=True)       # pulse + record
        cmd[axis] = -AMP
        stream(mav, tuple(cmd), 0.25)                          # counter-pulse
        stream(mav, (0, 0, 0, hover), 0.4)
        if rec is None or len(rec) < 5:
            print(f"{names[axis]}: NO DATA"); continue
        resp = rec[len(rec) // 3:, axis]                       # steady part of the pulse
        mean = float(np.mean(resp))
        results[names[axis]] = (np.sign(mean) if abs(mean) > 0.05 else 0.0, mean / AMP)
        print(f"{names[axis]:5s}: cmd +{AMP} -> measured rate {mean:+.2f} rad/s  "
              f"(sign {np.sign(mean):+.0f}, gain {mean / AMP:+.2f}x)", flush=True)

    mav.arm(False)
    for _ in range(6):
        mav.att(0, 0, 0, 0); mav.hb(); time.sleep(0.02)
    print("RESULTS:", results, flush=True)
    mav.stop()


if __name__ == "__main__":
    main()
