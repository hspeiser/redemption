"""Cascade flight controller for the AI-GP sim.

The sim only accepts body-rate + collective-thrust commands (velocity/position
setpoints are ignored), so this implements position -> acceleration -> attitude
-> body-rate control using ground-truth odometry.

Thrust model measured empirically: ~40 m/s^2 of thrust acceleration per unit
thrust (hover ~= 0.25).
"""

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

G = 9.81


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# Empirically measured sim conventions (rate_probe, 2026-07-23):
#  - odom quaternion: STANDARD body->world, NED world, FRD body, wxyz.
#  - body-rate commands: roll and yaw axes are SIGN-INVERTED vs standard;
#    pitch is normal. Inner rate loop has roughly 2.5x effective gain.
#  - ATTITUDE msg reports roll/pitch sign-inverted; use the quaternion instead.
#  - attitude-quaternion targets misbehave (inverted axes + degenerate near
#    yaw=pi, which is the start pose) — fly rates mode.
#  - rate commands are tracked ~2.56x too fast (constant 0.5 rad/s command
#    yields 1.28 rad/s actual, settling in ~0.1s) — compensate with gain.
#  - IMU gyro y-axis is sign-inverted vs the quaternion derivative.
# UPDATE (quat convention fix): MavIO now corrects the sim's Y-flipped
# left-handed quats to standard body->world. Under the corrected reading the
# rate commands are STANDARD-signed (the old [-1, 1, -1] table was an
# artifact of the misread quats — same flip, both places).
RATE_CMD_SIGN = np.array([1.0, 1.0, 1.0])
RATE_CMD_GAIN = 1.0 / 2.56


class RateController:
    def __init__(self, k_thrust=40.0, thrust_limit=0.7):
        self.k_thrust = k_thrust
        self.thrust_limit = thrust_limit
        self.kp = np.array([1.2, 1.2, 2.0])
        self.kv = np.array([2.0, 2.0, 2.8])
        self.k_att = 4.0
        self.rate_limit = 4.0
        self.zi = 0.0  # z integrator for hover-thrust bias

    def update(self, p, v, Rwb, p_ref, v_ref, yaw_des, dt):
        a_cmd = self.kp * (p_ref - p) + self.kv * (v_ref - v)
        self.zi = float(np.clip(self.zi + 0.6 * (p_ref[2] - p[2]) * dt, -3.0, 3.0))
        a_cmd[2] += self.zi
        # limit commanded acceleration
        ah = np.linalg.norm(a_cmd[:2])
        if ah > 14.0:
            a_cmd[:2] *= 14.0 / ah
        a_cmd[2] = np.clip(a_cmd[2], -18.0, 12.0)

        t_des = a_cmd - np.array([0.0, 0.0, G])   # desired thrust accel vector
        if t_des[2] > -2.0:
            t_des[2] = -2.0                        # never fully cut upward thrust
        zb_des = -t_des / np.linalg.norm(t_des)

        xc = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
        yb = np.cross(zb_des, xc)
        n = np.linalg.norm(yb)
        if n < 1e-6:
            yb = np.array([0.0, 1.0, 0.0])
            n = 1.0
        yb /= n
        xb = np.cross(yb, zb_des)
        R_des = np.column_stack([xb, yb, zb_des])

        w_cmd = self.k_att * Rotation.from_matrix(Rwb.T @ R_des).as_rotvec()
        w_cmd = np.clip(w_cmd, -self.rate_limit, self.rate_limit)
        thrust = float(np.clip(np.linalg.norm(t_des) / self.k_thrust,
                               0.02, self.thrust_limit))
        return w_cmd, thrust, R_des


class Flyer:
    """Waypoint/hold flying on top of RateController."""

    def __init__(self, mav, hz=90.0, mode="rates"):
        # hz capped below 100: official spec (VADR-TS-002 4.4) requires
        # command rate < 100 Hz. (The sample code's 250 Hz is non-compliant.)
        self.mav = mav
        self.hz = hz
        self.mode = mode      # "rates" (default, verified) or "attitude"
        self.ctl = RateController()
        self.q_b2w = True
        self.abort = False


    def _state(self):
        od = self.mav.latest_odom()
        if od is None:
            return None
        q = od["quat_wxyz"]
        Rq = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        Rwb = Rq if self.q_b2w else Rq.T
        # ODOMETRY velocity is BODY-frame (MAVLink child frame) — rotate to world
        v_world = Rwb @ np.array(od["vel"])
        return np.array(od["pos"]), v_world, Rwb

    def _step(self, p_ref, v_ref, yaw_des, dt):
        st = self._state()
        if st is None:
            return None
        p, v, Rwb = st
        w, thrust, R_des = self.ctl.update(p, v, Rwb, np.asarray(p_ref, float),
                                           np.asarray(v_ref, float), yaw_des, dt)
        if self.mode == "attitude":
            q = Rotation.from_matrix(R_des if self.q_b2w else R_des.T).as_quat()
            self.mav.send_attitude_quat([q[3], q[0], q[1], q[2]], thrust)
        else:
            ws = RATE_CMD_SIGN * RATE_CMD_GAIN * w
            self.mav.send_attitude_rates(ws[0], ws[1], ws[2], thrust)
        return p

    def _yaw_to(self, p, Rwb, look_at, sweep_amp, sweep_hz, t):
        la = np.asarray(look_at, float) - p
        if np.linalg.norm(la[:2]) < 1.5:
            # target (nearly) overhead/underneath: hold current yaw
            return math.atan2(Rwb[1, 0], Rwb[0, 0])
        yaw = math.atan2(la[1], la[0])
        if sweep_amp:
            yaw += sweep_amp * math.sin(2 * math.pi * sweep_hz * t)
        return yaw

    def goto(self, wp, look_at=None, timeout=10.0, reach=0.8,
             sweep_amp=0.0, sweep_hz=0.3, max_err_abort=80.0):
        wp = np.asarray(wp, float)
        t0 = time.time()
        dt = 1.0 / self.hz
        while time.time() - t0 < timeout:
            st = self._state()
            if st is None:
                time.sleep(dt)
                continue
            p = st[0]
            if np.linalg.norm(wp - p) > max_err_abort or p[2] > 2.0:
                print("ABORT: position error exploded or below ground", flush=True)
                self.abort = True
                return False
            if np.linalg.norm(wp - p) < reach:
                return True
            yaw = self._yaw_to(p, st[2], look_at if look_at is not None else wp,
                               sweep_amp, sweep_hz, time.time() - t0)
            self._step(wp, np.zeros(3), yaw, dt)
            time.sleep(dt)
        return False

    def hold(self, duration, look_at=None, sweep_amp=0.0, sweep_hz=0.3,
             anchor=None):
        st = self._state()
        if st is None:
            return
        anchor = np.asarray(anchor, float) if anchor is not None else st[0].copy()
        t0 = time.time()
        dt = 1.0 / self.hz
        while time.time() - t0 < duration:
            st = self._state()
            if st is None:
                time.sleep(dt)
                continue
            p = st[0]
            yaw = (self._yaw_to(p, st[2], look_at, sweep_amp, sweep_hz,
                                time.time() - t0)
                   if look_at is not None
                   else math.atan2(st[2][1, 0], st[2][0, 0]))
            self._step(anchor, np.zeros(3), yaw, dt)
            time.sleep(dt)

    def track(self, p_ref, v_ref, yaw_des):
        """Single external-loop control step (for trajectory tracking)."""
        return self._step(p_ref, v_ref, yaw_des, 1.0 / self.hz)
