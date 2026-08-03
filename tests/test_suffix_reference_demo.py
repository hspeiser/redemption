from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "build_vq2_suffix_reference_demo.py"
)
SPEC = importlib.util.spec_from_file_location("suffix_reference_demo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_reconstruct_feedforward_is_finite_and_bounded():
    class Model:
        thrust_gain = 13.85
        thrust_quad = 65.19

    n = 20
    velocity = np.zeros((n, 3), dtype=float)
    velocity[:, 0] = np.linspace(3.0, 5.0, n)
    quat = np.zeros((n, 4), dtype=float)
    quat[:, 0] = 1.0

    action, rates = MODULE.reconstruct_feedforward(velocity, quat, Model())

    assert action.shape == (n, 4)
    assert rates.shape == (n, 3)
    assert np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1.0
    assert np.allclose(rates, 0.0)


def test_build_real_geometry_suffix_has_deployable_schema():
    root = Path(__file__).resolve().parents[1]
    payload = MODULE.build_suffix_demo(
        root / "data/lineopt/sfx_geometry_best.npz",
        root / "data/vq2_runtime_map_g9g15fix.json",
        root / "data/fastsim_model_v2.json",
    )

    n = len(payload["gate_index"])
    assert n > 300
    assert payload["observation"].shape == (n, 53)
    assert payload["action"].shape == (n, 4)
    assert payload["next_observation"].shape == (n, 53)
    assert payload["gate_index"].min() == 11
    assert payload["gate_index"].max() == 16
    assert np.all(np.diff(payload["gate_index"]) >= 0)
    assert payload["done"].sum() == 1.0
    assert payload["done"][-1] == 1.0
    assert np.isfinite(payload["observation"]).all()
    assert np.isfinite(payload["action"]).all()
