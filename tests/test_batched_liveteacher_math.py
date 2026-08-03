from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from aigp.fastsim.liveteacher import _rotvec_batch


def test_rotvec_preserves_sub_milliradian_skew_when_trace_rounds_high():
    # This is the gate-0 step-0 tracking rotation from the 35.37 s fixture.
    # In float32 its trace rounds to the identity side of acos(), while its
    # skew part still carries a real 0.000406 rad correction.
    matrix = np.asarray([
        [1.00000002, 1.32772362e-05, -2.51343295e-04],
        [-1.31973085e-05, 0.999999962, 3.18003302e-04],
        [2.51345347e-04, -3.17999957e-04, 0.999999985],
    ])
    expected = Rotation.from_matrix(matrix).as_rotvec()
    actual = _rotvec_batch(
        torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    )[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-8)
    assert np.linalg.norm(actual) > 4e-4
