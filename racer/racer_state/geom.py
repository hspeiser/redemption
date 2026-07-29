"""Quaternion / rotation helpers. Quaternions are [w, x, y, z] (MAVLink ODOMETRY order).

The ODOMETRY quaternion is the drone orientation in the world (NED) frame: a body-frame vector
rotates to world by R(q). So world->body uses the conjugate. All functions are numpy, no scipy.
"""
from __future__ import annotations
import numpy as np


def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z], np.float64)


def rot_body_to_world(q, v):
    """v_world = R(q) v_body."""
    w, x, y, z = q
    v = np.asarray(v, np.float64)
    # rotation matrix from quaternion
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], np.float64)
    return R @ v


def rot_world_to_body(q, v):
    """v_body = R(q)^T v_world."""
    return rot_body_to_world(quat_conj(q), v)


def gate_normal_world(gq):
    """Gate 'through' axis in world NED. Gates here carry quat (w,z)=(0.71,0.71) ~ +90deg about z;
    the track runs along -X, so the gate plane normal is the gate's local Y rotated to world."""
    return rot_body_to_world(gq, np.array([0.0, 1.0, 0.0]))
