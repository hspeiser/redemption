"""Identify the surrogate dynamics from recorded episodes.

Model (per-axis first-order rate loop + rigid body + drag):
  rate loop:      dw_i/dt = (K_i * u_i(t - d) - w_i) / tau_i
  attitude:       quaternion integration of body rates
  translation:    dv/dt = g_vec + R_wb @ ( [0,0,-cT*u_t] + D_lin * v_body )
Everything (signs, gains, delays, gravity direction, thrust curve, drag)
is fitted empirically -- zero convention assumptions.

VQ1 odometry supplies ground-truth rates/vel/pos for both fitting and the
closed-loop validation rollout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from aigp.fastsim.data import Episode


@dataclass
class SurrogateModel:
    rate_gain: list[float]       # K per axis (wire -> rad/s steady state)
    rate_tau: list[float]        # s
    rate_delay: float            # s, command transport delay
    thrust_gain: float           # m/s^2 per unit wire thrust (linear term)
    thrust_quad: float           # m/s^2 per unit^2 (motor curve)
    drag_lin: list[float]        # per-axis body linear drag [1/s]
    g_vec: list[float]           # world gravity vector (frame-empirical)
    hz: float
    # quadratic body drag [1/m]: f_body -= c * |v| * v_body.  Fitted
    # jointly with the thrust-curve correction on v77/v79 full-record
    # data (151k samples, 2-12 m/s); zero for legacy models.
    drag_quad: list[float] | None = None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path: str | Path) -> "SurrogateModel":
        return SurrogateModel(**json.loads(Path(path).read_text()))


def _smooth_derivative(x: np.ndarray, hz: float) -> np.ndarray:
    window = max(int(0.15 * hz) | 1, 5)
    return savgol_filter(
        x, window, 3, deriv=1, delta=1.0 / hz, axis=0, mode="interp"
    )


def fit_rate_loop(
    episodes: list[Episode], delays=np.arange(0.0, 0.101, 0.01)
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Fit K_i, tau_i and one shared delay from cmd -> GYRO (dense, native).

    Discrete one-step form avoids differentiating: w[k+1] = a*w[k] + b*u[k]
    with a = exp(-dt/tau), K = b/(1-a).
    """
    best = None
    for delay in delays:
        gains = np.zeros(3)
        taus = np.zeros(3)
        total = 0.0
        for axis in range(3):
            A_rows, b_rows = [], []
            for ep in episodes:
                hz = 1.0 / (ep.t[1] - ep.t[0])
                shift = int(round(delay * hz))
                u = np.roll(ep.cmd[:, axis], shift)
                u[:shift] = ep.cmd[0, axis]
                w = ep.gyro[:, axis]
                A_rows.append(np.stack([w[:-1], u[:-1]], axis=1))
                b_rows.append(w[1:])
            A = np.concatenate(A_rows)
            b = np.concatenate(b_rows)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            a, bb = coef
            a = float(np.clip(a, 0.02, 0.999))
            hz0 = 1.0 / (episodes[0].t[1] - episodes[0].t[0])
            taus[axis] = -1.0 / (hz0 * np.log(a))
            gains[axis] = float(bb / (1.0 - a))
            pred = A @ coef
            total += float(np.mean((pred - b) ** 2))
        if best is None or total < best[0]:
            best = (total, delay, gains.copy(), taus.copy())
    _mse, delay, gains, taus = best
    report = {"delay_s": float(delay),
              "gain": gains.tolist(), "tau_s": taus.tolist()}
    return gains, taus, float(delay), report


def fit_translation(episodes: list[Episode]) -> tuple[dict, dict]:
    """Fit gravity from the odometry/IMU consistency, then thrust + drag
    directly against the MEASURED specific force (noiseless IMU):

      g      = median over flight of (dv_world - R @ f_imu)
      f_z    = -(cT1*u + cT2*u^2) + dz*v_body_z
      f_x/y  = dx*v_body_x / dy*v_body_y            (drag only)
    """
    g_samples = []
    A_z, b_z, A_x, b_x, A_y, b_y = [], [], [], [], [], []
    for ep in episodes:
        if ep.vel_world is None:
            continue
        hz = 1.0 / (ep.t[1] - ep.t[0])
        dv = _smooth_derivative(ep.vel_world, hz)
        rot = Rotation.from_quat(ep.quat_wb)
        f_world = rot.apply(ep.accel)
        g_samples.append(dv - f_world)
    g_all = np.concatenate(g_samples)
    g_vec = np.median(g_all, axis=0)
    for ep in episodes:
        if ep.vel_world is None:
            continue
        hz = 1.0 / (ep.t[1] - ep.t[0])
        dv = _smooth_derivative(ep.vel_world, hz)
        rot = Rotation.from_quat(ep.quat_wb)
        # TRUE total body force from odometry (bypasses the sim
        # accelerometer's scale quirks entirely)
        a_body = rot.inv().apply(dv - g_vec)
        v_body = rot.inv().apply(ep.vel_world)
        u = ep.cmd[:, 3]
        m = u > 0.02
        A_z.append(np.stack(
            [-u[m], -(u[m] ** 2), v_body[m, 2]], axis=1
        ))
        b_z.append(a_body[m, 2])
        A_x.append(v_body[:, 0:1])
        b_x.append(a_body[:, 0])
        A_y.append(v_body[:, 1:2])
        b_y.append(a_body[:, 1])
    cz, *_ = np.linalg.lstsq(np.concatenate(A_z), np.concatenate(b_z),
                             rcond=None)
    cx, *_ = np.linalg.lstsq(np.concatenate(A_x), np.concatenate(b_x),
                             rcond=None)
    cy, *_ = np.linalg.lstsq(np.concatenate(A_y), np.concatenate(b_y),
                             rcond=None)
    pred_z = np.concatenate(A_z) @ cz
    rms = float(np.sqrt(np.mean((pred_z - np.concatenate(b_z)) ** 2)))
    fit = {
        "thrust_gain": float(cz[0]),
        "thrust_quad": float(cz[1]),
        "drag_lin": [float(cx[0]), float(cy[0]), float(cz[2])],
        "g_vec": g_vec.tolist(),
    }
    return fit, {"accel_fit_rms_mps2": rms,
                 "g_mad": np.median(np.abs(g_all - g_vec), 0).tolist()}


def rollout(
    model: SurrogateModel, ep: Episode, t0: float, duration: float,
    attitude_from_truth: bool = False,
) -> dict:
    """Closed-loop integrate the model from odometry initial conditions,
    driven by the LOGGED wire commands; compare against odometry."""
    hz = 1.0 / (ep.t[1] - ep.t[0])
    i0 = int(np.searchsorted(ep.t, t0))
    i1 = min(int(np.searchsorted(ep.t, t0 + duration)), len(ep.t) - 1)
    if i1 - i0 < int(hz):
        return {}
    delay_steps = int(round(model.rate_delay * hz))
    K = np.asarray(model.rate_gain)
    tau = np.asarray(model.rate_tau)
    D = np.asarray(model.drag_lin)
    g = np.asarray(model.g_vec)

    w = ep.rates_body[i0].copy()
    q = Rotation.from_quat(ep.quat_wb[i0])
    v = ep.vel_world[i0].copy()
    p = ep.pos[i0].copy()
    dt = 1.0 / hz
    errs = []
    for i in range(i0, i1):
        j = max(i - delay_steps, 0)
        u = ep.cmd[j]
        w = w + dt * (K * u[:3] - w) / tau
        if attitude_from_truth:
            R = Rotation.from_quat(ep.quat_wb[i]).as_matrix()
        else:
            q = q * Rotation.from_rotvec(w * dt)
            R = q.as_matrix()
        v_body = R.T @ v
        f_body = D * v_body
        f_body[2] += -(model.thrust_gain * u[3]
                       + model.thrust_quad * u[3] ** 2)
        a = g + R @ f_body
        v = v + a * dt
        p = p + v * dt
        errs.append(np.linalg.norm(p - ep.pos[i + 1]))
    errs = np.asarray(errs)
    return {
        "t0": float(ep.t[i0]),
        "duration_s": float(ep.t[i1] - ep.t[i0]),
        "pos_rmse_m": float(np.sqrt(np.mean(errs**2))),
        "pos_final_err_m": float(errs[-1]),
        "pos_max_err_m": float(errs.max()),
    }
