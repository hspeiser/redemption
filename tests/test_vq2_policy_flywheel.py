import numpy as np

from scripts.build_vq2_policy_sequence_dataset import (
    n_step_targets,
    segment_bounds,
)
from scripts.build_vq2_split_registry import group_key, initial_split


def test_split_registry_links_raw_and_training_timestamp():
    training = {
        "host": "local",
        "kind": "training_session",
        "path": r"D:\ai-gp\training\campaign\20260802_110754",
    }
    raw = {
        "host": "local",
        "kind": "raw_session",
        "path": r"D:\ai-gp\raw_sessions\vq2_20260802_110754",
    }
    assert group_key(training) == group_key(raw) == "local:20260802_110754"
    assert initial_split(group_key(training)) == initial_split(group_key(raw))


def test_segment_bounds_keeps_context_and_next_gate_entry():
    gate = np.asarray([4] * 50 + [5] * 60 + [6] * 50, np.int16)
    assert segment_bounds(gate, 5, 15) == (35, 125)


def test_n_step_targets_stop_at_terminal():
    reward = np.asarray([1.0, 2.0, 3.0], np.float32)
    done = np.asarray([0.0, 1.0, 0.0], np.float32)
    next_observation = np.arange(6, dtype=np.float32).reshape(3, 2)
    total, terminal, discount, following = n_step_targets(
        reward, done, next_observation, gamma=0.5, n_step=3,
    )
    np.testing.assert_allclose(total, [2.0, 2.0, 3.0])
    np.testing.assert_allclose(terminal, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(discount, [0.25, 0.5, 0.5])
    np.testing.assert_allclose(following[0], next_observation[1])
