"""Decisive frame-convention probe: for each axis, pulse an open-loop rate cmd and compare
  (a) the commanded axis/sign,
  (b) the ODOMETRY-reported body rates,
  (c) the ACTUAL rotation axis derived from the quaternion delta (ground truth), and
  (d) velocity: quat-rotated reported vel vs d(pos)/dt (is vel FRD-body, FLU-body, or world?).

If (b) disagrees with (c), the odometry rate frame is not what the adapter assumes — which
corrupts both the obs and the inner rate loop's feedback, and would explain a transferred
policy tumbling from a clean hover.

  python -m racer_state.probe_frames
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from racer_state.config import CFG
from racer_state.mav_io import MavLink
from racer_state.geom import rot_world_to_body, rot_body_to_world
from racer_state.train import wait_reset
from racer_state.train2 import thrust_cmd, HOVER_CMD

_DOWN = np.array([0.0, 0.0, 1.0])


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def body_rot_axis(q0, q1):
    """Rotation q0 -> q1 expressed in the BODY frame: dq = q0^-1 * q1; axis*angle."""
    dq = qmul(qconj(q0), q1)
    if dq[0] < 0:
        dq = -dq
    ang = 2.0 * np.arccos(np.clip(dq[0], -1, 1))
    s = np.linalg.norm(dq[1:4])
    axis = dq[1:4] / s if s > 1e-9 else np.zeros(3)
    return axis, float(ang)


def level_hover(mav, hb, z_t, dur_max=3.0):
    t0 = time.time()
    while time.time() - t0 < dur_max:
        g = rot_world_to_body(mav.quat, _DOWN)
        des = np.array([-3.0 * float(g[1]), 3.0 * float(g[0]), 0.0])
        vz = float(rot_body_to_world(mav.quat, np.asarray(mav.vel, np.float64))[2])
        err_up = float(mav.pos[2]) - z_t
        a_up = float(np.clip(1.6 * err_up, -2.5, 2.5) + np.clip(1.2 * vz, -2.5, 2.5))
        spec = (9.81 + a_up) / max(0.6, float(g[2]))
        # crude inner loop with prior gains
        err = des - np.asarray(mav.rates, np.float64)
        cmd = np.clip((des + 1.5 * err) / np.array([-2.48, 2.52, -2.31]), -1.9, 1.9)
        mav.att(float(cmd[0]), float(cmd[1]), float(cmd[2]), thrust_cmd(spec / 9.81))
        if time.time() - hb[0] > 0.3:
            mav.hb(); hb[0] = time.time()
        if abs(err_up) < 0.3 and abs(vz) < 0.4 and abs(float(g[0])) < 0.1 \
                and abs(float(g[1])) < 0.1:
            return True
        time.sleep(0.004)
    return False


def main():
    mav = MavLink(CFG.mav_addr)
    print(f"connected sys={mav.sys}", flush=True)
    wait_reset(mav); time.sleep(CFG.reset_settle_s)
    t0 = time.time()
    while mav.t_us == 0 and time.time() - t0 < 3:
        mav.hb(); time.sleep(0.03)
    hb = [0.0]
    mav.arm(True)
    z_t = float(mav.pos[2]) - 3.0
    level_hover(mav, hb, z_t, dur_max=5.0)

    names = ["roll(cmd0)", "pitch(cmd1)", "yaw(cmd2)"]
    for axis in range(3):
        level_hover(mav, hb, z_t)
        q0 = mav.quat.copy()
        rates_acc = []
        t1 = time.time()
        cmd = [0.0, 0.0, 0.0]; cmd[axis] = 0.5
        while time.time() - t1 < 0.25:
            mav.att(cmd[0], cmd[1], cmd[2], HOVER_CMD)
            rates_acc.append(np.asarray(mav.rates, np.float64).copy())
            if time.time() - hb[0] > 0.3:
                mav.hb(); hb[0] = time.time()
            time.sleep(0.004)
        q1 = mav.quat.copy()
        ax, ang = body_rot_axis(q0, q1)
        mean_rates = np.mean(rates_acc[len(rates_acc) // 3:], axis=0)
        print(f"{names[axis]:11s} +0.5: quat-truth axis={np.round(ax, 2)} angle={np.degrees(ang):5.1f}deg"
              f" | reported rates={np.round(mean_rates, 2)}", flush=True)
        # counter-pulse + recover
        cmd[axis] = -0.5
        t1 = time.time()
        while time.time() - t1 < 0.18:
            mav.att(cmd[0], cmd[1], cmd[2], HOVER_CMD)
            if time.time() - hb[0] > 0.3:
                mav.hb(); hb[0] = time.time()
            time.sleep(0.004)

    # velocity frame: descend slowly (known world +z motion), compare reported vel frames
    level_hover(mav, hb, z_t)
    print("descending at low thrust for 0.6 s (world vz should be +down):", flush=True)
    t1 = time.time()
    while time.time() - t1 < 0.6:
        g = rot_world_to_body(mav.quat, _DOWN)
        des = np.array([-3.0 * float(g[1]), 3.0 * float(g[0]), 0.0])
        err = des - np.asarray(mav.rates, np.float64)
        c = np.clip((des + 1.5 * err) / np.array([-2.48, 2.52, -2.31]), -1.9, 1.9)
        mav.att(float(c[0]), float(c[1]), float(c[2]), thrust_cmd(0.6))
        if time.time() - hb[0] > 0.3:
            mav.hb(); hb[0] = time.time()
        time.sleep(0.004)
    p0 = mav.pos.copy(); v_rep = np.asarray(mav.vel, np.float64).copy(); q = mav.quat.copy()
    time.sleep(0.2)
    p1 = mav.pos.copy()
    dposdt = (p1 - p0) / 0.2
    print(f"  d(pos)/dt world = {np.round(dposdt, 2)}", flush=True)
    print(f"  reported vel raw = {np.round(v_rep, 2)}", flush=True)
    print(f"  reported vel rotated body->world = "
          f"{np.round(rot_body_to_world(q, v_rep), 2)}", flush=True)

    mav.arm(False)
    for _ in range(6):
        mav.att(0, 0, 0, 0); mav.hb(); time.sleep(0.02)
    mav.stop()


if __name__ == "__main__":
    main()
