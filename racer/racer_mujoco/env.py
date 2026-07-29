"""MuJoCo twin of the VQ1 racer — a rate-controlled quadrotor + the REAL 6-gate track, runs FASTER
than real time so we can iterate RL in seconds instead of hours.

Matches the VQ1 interface where it matters: action = (roll_rate, pitch_rate, yaw_rate, thrust) raw
policy units in (-1,1); observation = the same 15-dim body-frame state as racer_state; reward =
exact distance progress + gate bonus + distance-scaled failure. Frame here is z-UP (standard
MuJoCo); obs is built consistently in this frame. Hover thrust ~0.28 (measured on VQ1).

Track captured live from VQ1 (2026-07-23) and mapped NED -> z-up via (x,y,z) = (-nx, -ny, SPAWN_Z-nz):
all 6 gates face straight down-track (+x), 2.72 m square openings, and the course DESCENDS ~26 m
from spawn height by gate 5. Spawn is at z=30 so the whole descending track stays above ground.
On a pass the env retargets gate_pos/gate_normal to the next gate (multi-gate episodes)."""
from __future__ import annotations
import numpy as np
import mujoco

SPAWN_Z = 30.0
# real VQ1 track (NED): g0(-23.3,-0.4,-0.03) g1(-46.89,-2.5,5.07) g2(-74.59,1.2,13.67)
#                       g3(-111.49,-5.1,24.57) g4(-135.49,-0.8,25.36) g5(-159.19,-4.4,25.97)
GATE_POS = np.array([
    [23.30,  0.40, 30.03],
    [46.89,  2.50, 24.93],
    [74.59, -1.20, 16.33],
    [111.49, 5.10,  5.43],
    [135.49, 0.80,  4.64],
    [159.19, 4.40,  4.03],
])
GATE_NORMAL = np.array([1.0, 0.0, 0.0])   # every gate faces down-track (+x); pass = cross -> +x
GATE0 = GATE_POS[0]                       # kept for older callers
GATE_HALF = 1.36            # opening half-width (2.72 m gate, captured)
HOVER = 0.28               # normalized hover thrust (VQ1)
MASS = 1.0
G = 9.81
MAX_THRUST = MASS * G / HOVER      # force at thrust_norm=1
RATE_SCALE = 4.0                    # rad/s at |action|=1
THRUST_SPAN = 0.35                 # hover-centered: thrust_norm = HOVER + a3*span
KP_RATE = 25.0                     # body-rate P gain
IDIAG = np.array([0.01, 0.01, 0.02])


def _gate_body(i, p):
    gh = GATE_HALF
    t = 0.12          # post half-thickness: thick enough to survive 84px downsampling at 23 m
    return f"""
    <body name="gate{i}" pos="{p[0]} {p[1]} {p[2]}">
      <geom type="box" size="{t} {gh + 0.1} {t}" pos="0 0 {gh}" material="gate"/>
      <geom type="box" size="{t} {gh + 0.1} {t}" pos="0 0 -{gh}" material="gate"/>
      <geom type="box" size="{t} {t} {gh}" pos="0 {gh} 0" material="gate"/>
      <geom type="box" size="{t} {t} {gh}" pos="0 -{gh} 0" material="gate"/>
    </body>"""


def _clutter(seed, n=25):
    """Random visual-reference blocks OUTSIDE the flight corridor (|y| <= 8 around the track).
    Random sizes/colors (never gate-yellow) so the vision policy gets parallax cues that
    generalize across env instances."""
    if seed is None:
        return ""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        x = rng.uniform(-15, 175)
        y = rng.choice([-1, 1]) * rng.uniform(9, 35)
        sx, sy = rng.uniform(0.5, 3.0, 2)
        sz = rng.uniform(1.0, 14.0)              # some tall pillars for horizon reference
        r, g, b = rng.uniform(0.15, 0.9, 3)
        if r > 0.6 and g > 0.6 and b < 0.5:      # avoid gate-yellow lookalikes
            b = 0.7
        out.append(f'<geom type="box" pos="{x:.1f} {y:.1f} {sz / 2:.1f}" '
                   f'size="{sx:.2f} {sy:.2f} {sz / 2:.2f}" rgba="{r:.2f} {g:.2f} {b:.2f} 1"/>')
    return "\n    ".join(out)


def build_mjcf(clutter_seed=None):
    return f"""
<mujoco model="racer">
  <option timestep="0.004" gravity="0 0 -{G}" integrator="RK4"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.45 0.6 0.8" rgb2="0.85 0.9 0.95"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.28 0.33 0.38" rgb2="0.42 0.47 0.52"
             width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="80 80" reflectance="0"/>
    <material name="gate" rgba="1.0 0.82 0.12 1" emission="0.9"/>
  </asset>
  <worldbody>
    <light directional="true" pos="0 0 60" dir="0.25 0.1 -1"
           diffuse="0.85 0.85 0.85" ambient="0.45 0.45 0.45"/>
    <geom name="ground" type="plane" pos="0 0 0" size="400 400 1" material="grid"/>
    <body name="drone" pos="0 0 {SPAWN_Z}">
      <freejoint name="root"/>
      <geom name="core" type="box" size="0.12 0.12 0.03" rgba="0.9 0.3 0.2 1"/>
      <inertial pos="0 0 0" mass="{MASS}" diaginertia="{IDIAG[0]} {IDIAG[1]} {IDIAG[2]}"/>
      <camera name="fpv" fovy="90" pos="0.14 0 0.02" xyaxes="0 -1 0 0 0 1"/>
    </body>
    {''.join(_gate_body(i, p) for i, p in enumerate(GATE_POS))}
    {_clutter(clutter_seed)}
  </worldbody>
</mujoco>
"""


_MJCF = build_mjcf(None)


class QuadEnv:
    """Minimal fast quadrotor gate env. Exposes a racer_state-compatible state (pos/quat/vel/rates).
    gate_pos/gate_normal always refer to the ACTIVE gate; a pass retargets them to the next gate."""

    def __init__(self, ctrl_hz=30.0, episode_s=8.0, seed=0, clutter_seed=None):
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(clutter_seed))
        self.data = mujoco.MjData(self.model)
        self.bid = self.model.body("drone").id
        self.spawn = np.array([0.0, 0.0, SPAWN_Z])
        self.gates = [(GATE_POS[i].copy(), GATE_NORMAL.copy()) for i in range(len(GATE_POS))]
        self.ngates = len(self.gates)
        self.gate_pos, self.gate_normal = self.gates[0]
        self.ctrl_dt = 1.0 / ctrl_hz
        self.substeps = max(1, int(round(self.ctrl_dt / self.model.opt.timestep)))
        self.max_steps = int(episode_s * ctrl_hz)
        self.rng = np.random.default_rng(seed)
        self.spawn_dist = float(GATE_POS[0][0])   # curriculum hook: distance behind gate 0
        # racer_state-style live state
        self.pos = self.spawn.copy(); self.quat = np.array([1.0, 0, 0, 0])
        self.vel = np.zeros(3); self.rates = np.zeros(3)
        self.active_gate = 0
        self._prev_side = None

    # ---- state helpers ----
    def _read(self):
        self.pos = self.data.qpos[0:3].copy()
        self.quat = self.data.qpos[3:7].copy()            # w,x,y,z
        self.vel = self.data.qvel[0:3].copy()             # world linvel
        self.rates = self.data.qvel[3:6].copy()           # body angvel

    def _retarget(self):
        self.gate_pos, self.gate_normal = self.gates[self.active_gate]
        self._prev_side = float(np.dot(self.pos - self.gate_pos, self.gate_normal))

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # spawn `spawn_dist` behind gate 0, aligned with the opening + small jitter
        g0 = self.gates[0][0]
        base = np.array([g0[0] - self.spawn_dist, g0[1], g0[2]])
        self.data.qpos[0:3] = base + self.rng.uniform(-0.3, 0.3, 3)
        yaw = 0.0
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._read()
        self.active_gate = 0
        self._retarget()
        self.t_us = 1
        self.collision_epoch = 0
        return self

    def _apply(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        desired = a[:3] * RATE_SCALE
        thrust_norm = np.clip(HOVER + a[3] * THRUST_SPAN, 0, 1)
        F = thrust_norm * MAX_THRUST
        q = self.quat
        thr_world = np.zeros(3); mujoco.mju_rotVecQuat(thr_world, np.array([0.0, 0, F]), q)
        torque_body = IDIAG * KP_RATE * (desired - self.rates)
        tq_world = np.zeros(3); mujoco.mju_rotVecQuat(tq_world, torque_body, q)
        self.data.xfrc_applied[self.bid, 0:3] = thr_world
        self.data.xfrc_applied[self.bid, 3:6] = tq_world

    def step(self, action):
        self._apply(action)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._read()
        # gate pass: crossed the ACTIVE gate plane forward (side<0 -> side>=0) inside the opening
        side = float(np.dot(self.pos - self.gate_pos, self.gate_normal))
        radial = np.linalg.norm((self.pos - self.gate_pos) - side * self.gate_normal)
        passed = (self._prev_side is not None and self._prev_side < 0 <= side and radial < GATE_HALF)
        self._prev_side = side
        if passed:
            self.active_gate += 1
            if self.active_gate < self.ngates:
                self._retarget()          # chase the next gate (obs/reward switch AFTER the pass)
        # crash: contact involving the drone, or below ground
        crashed = self.data.ncon > 0 or self.pos[2] < 0.15
        if crashed:
            self.collision_epoch += 1
        return passed, crashed

    def dist_to_gate(self):
        return float(np.linalg.norm(self.gate_pos - self.pos))
