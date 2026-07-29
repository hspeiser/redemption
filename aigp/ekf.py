"""Error-state EKF: vision (gate corners) + IMU, no odometry.

State (nominal): position p (world NED), velocity v (world), attitude q
(body->world, corrected sim convention). Error state: [dp, dv, dtheta] (9).
No bias states — the sim IMU is measured noiseless/bias-free.

IMU conventions are injected via `gyro_sign`/`accel_sign` (diagonal axis
signs mapping raw HIGHRES_IMU vectors into the corrected body frame); they
are determined empirically by the dead-reckoning bring-up.
"""

import numpy as np
from scipy.spatial.transform import Rotation

G_NED = np.array([0.0, 0.0, 9.81])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class GateEKF:
    # Empirically determined (ekf_bringup dead-reckoning sweep on banked
    # flight): gyro is fully negated vs the corrected body frame; accel is
    # direct. 1.5 s IMU-only drift: 7.6 cm position, 0.36 deg attitude.
    GYRO_SIGN = (-1.0, -1.0, -1.0)
    ACCEL_SIGN = (1.0, 1.0, 1.0)

    def __init__(self, K, R_cb, gyro_sign=GYRO_SIGN, accel_sign=ACCEL_SIGN,
                 sigma_px=1.0):
        self.K = np.asarray(K)          # 3x3 intrinsics
        self.R_cb = np.asarray(R_cb)    # body -> camera rotation
        self.gs = np.asarray(gyro_sign, float)
        self.asn = np.asarray(accel_sign, float)
        self.sigma_px = sigma_px

        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = Rotation.identity()
        self.P = np.eye(9) * 1e-4
        self.t = None                    # last IMU time (seconds)

        # process noise (integration error only — tiny, but keep the filter
        # humble enough to accept corrections)
        self.q_acc = 0.15       # m/s^2 / sqrt(Hz) equivalent
        self.q_gyro = 0.005     # rad/s / sqrt(Hz)

    # ------------------------------------------------------------ lifecycle

    def init_state(self, p, v, q_wxyz, t, pos_std=0.05, vel_std=0.05,
                   ang_std=0.02):
        self.p = np.asarray(p, float).copy()
        self.v = np.asarray(v, float).copy()
        w, x, y, z = q_wxyz
        self.q = Rotation.from_quat([x, y, z, w])
        self.P = np.diag([pos_std**2] * 3 + [vel_std**2] * 3 + [ang_std**2] * 3)
        self.t = t

    # ------------------------------------------------------------ propagate

    def propagate(self, t, accel_raw, gyro_raw):
        """One IMU sample: t in seconds (exact sim clock), raw vectors as
        reported by HIGHRES_IMU."""
        if self.t is None:
            self.t = t
            return
        dt = t - self.t
        if dt <= 0 or dt > 0.1:
            self.t = t
            return
        self.t = t

        w = self.gs * np.asarray(gyro_raw, float)      # body rates
        f = self.asn * np.asarray(accel_raw, float)    # specific force, body

        R0 = self.q.as_matrix()
        dq = Rotation.from_rotvec(w * dt)
        q1 = self.q * dq
        # midpoint attitude for velocity integration
        Rm = (self.q * Rotation.from_rotvec(w * dt * 0.5)).as_matrix()
        a_world = Rm @ f + G_NED
        p1 = self.p + self.v * dt + 0.5 * a_world * dt * dt
        v1 = self.v + a_world * dt

        # error-state transition
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -Rm @ skew(f) * dt
        F[6:9, 6:9] = Rotation.from_rotvec(-w * dt).as_matrix()

        Q = np.zeros((9, 9))
        Q[3:6, 3:6] = np.eye(3) * (self.q_acc * dt) ** 2
        Q[6:9, 6:9] = np.eye(3) * (self.q_gyro * dt) ** 2
        Q[0:3, 0:3] = np.eye(3) * (0.5 * self.q_acc * dt * dt) ** 2

        self.P = F @ self.P @ F.T + Q
        self.p, self.v, self.q = p1, v1, q1

    # ------------------------------------------------------------ updates

    def predict_pixel(self, Xw):
        """Project world point through current state. Returns (uv, Xc)."""
        Rwb = self.q.as_matrix()
        Xc = self.R_cb @ Rwb.T @ (np.asarray(Xw) - self.p)
        if Xc[2] <= 0.3:
            return None, Xc
        uv = np.array([self.K[0, 0] * Xc[0] / Xc[2] + self.K[0, 2],
                       self.K[1, 1] * Xc[1] / Xc[2] + self.K[1, 2]])
        return uv, Xc

    def update_corners(self, obs, chi2_gate=9.0, update_attitude=True):
        """obs: list of (Xw (3,), uv_meas (2,)). Batch EKF update with
        per-observation chi-square gating. Returns number accepted.

        If update_attitude is false, vision corrects translation/velocity
        only. This is useful when a low-noise gyro is more trustworthy than
        the planar orientation of a symmetric square landmark."""
        H_rows, r_rows = [], []
        Rwb = self.q.as_matrix()
        for Xw, uv_meas in obs:
            uv_pred, Xc = self.predict_pixel(Xw)
            if uv_pred is None:
                continue
            x, y, z = Xc
            fx, fy = self.K[0, 0], self.K[1, 1]
            # d(uv)/d(Xc)
            J_uv = np.array([[fx / z, 0, -fx * x / z**2],
                             [0, fy / z, -fy * y / z**2]])
            # d(Xc)/d(error state): Xc = R_cb Rwb^T (Xw - p)
            dXc_dp = -self.R_cb @ Rwb.T
            # attitude error (right perturbation on q): d(Rwb^T u)/dtheta
            u = Xw - self.p
            dXc_dth = self.R_cb @ skew(Rwb.T @ u)
            H = np.zeros((2, 9))
            H[:, 0:3] = J_uv @ dXc_dp
            if update_attitude:
                H[:, 6:9] = J_uv @ dXc_dth
            r = np.asarray(uv_meas) - uv_pred
            # innovation gating
            S = H @ self.P @ H.T + np.eye(2) * self.sigma_px**2
            m2 = float(r @ np.linalg.solve(S, r))
            if m2 > chi2_gate:
                continue
            H_rows.append(H)
            r_rows.append(r)
        if not H_rows:
            return 0
        H = np.vstack(H_rows)
        r = np.concatenate(r_rows)
        Rm = np.eye(len(r)) * self.sigma_px**2
        S = H @ self.P @ H.T + Rm
        Kk = self.P @ H.T @ np.linalg.solve(S, np.eye(len(r)))
        if not update_attitude:
            # Cross-covariance can otherwise leak a nominal attitude update
            # even though the measurement Jacobian has no attitude columns.
            Kk[6:9, :] = 0.0
        dx = Kk @ r
        self.p += dx[0:3]
        self.v += dx[3:6]
        if update_attitude:
            self.q = self.q * Rotation.from_rotvec(dx[6:9])
        I_KH = np.eye(9) - Kk @ H
        self.P = I_KH @ self.P @ I_KH.T + Kk @ Rm @ Kk.T
        return len(H_rows)
