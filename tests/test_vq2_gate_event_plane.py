from types import SimpleNamespace

import numpy as np

from aigp.vq2_live_localizer import LiveVQ2Localizer


def _localizer(position):
    localizer = LiveVQ2Localizer.__new__(LiveVQ2Localizer)
    localizer.ekf = SimpleNamespace(
        p=np.asarray(position, dtype=float),
        P=np.eye(9, dtype=float),
    )
    localizer.gates = [
        {
            "pos": [10.0, 0.0, 0.0],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    localizer.update_counts = {}
    localizer.last_gate_event_correction = None
    return localizer


def test_gate_event_corrects_only_plane_position():
    localizer = _localizer([10.4, 2.0, -0.3])

    result = localizer.apply_gate_plane_event(
        0, forward_offset_m=0.15, sigma_m=0.20
    )

    # Identity gate rotation has local y as its plane normal and the incoming
    # direction is orthogonal in this synthetic case, hence positive offset.
    np.testing.assert_allclose(localizer.ekf.p, [10.4, 0.15, -0.3])
    assert result["before_plane_m"] == 2.0
    assert result["after_plane_m"] == 0.15
    assert localizer.update_counts["gate_event_plane"] == 1
    np.testing.assert_allclose(localizer.ekf.P[1, 1], 0.20**2)
    np.testing.assert_allclose(localizer.ekf.P[0, 0], 1.0)


def test_gate_event_uses_course_direction_sign():
    localizer = _localizer([10.0, 2.0, 0.0])
    localizer.gates[0]["pos"] = [0.0, -10.0, 0.0]

    result = localizer.apply_gate_plane_event(0, forward_offset_m=0.2)

    assert result["after_plane_m"] == -0.2
    np.testing.assert_allclose(localizer.ekf.p[1], -10.2)
