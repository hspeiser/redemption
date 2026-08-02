import numpy as np

from scripts.build_vq2_master_worldmodel_dataset import (
    episode_timing_eligible,
    load_split_registry,
    session_split,
)
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


def test_worldmodel_builder_uses_immutable_registry(tmp_path):
    session = tmp_path / "training" / "campaign" / "20260802_110754"
    registry_path = tmp_path / "splits.json"
    registry_path.write_text(
        '{"generation": 3, "groups": ['
        '{"split": "policy_selection", "paths": ['
        f'"{str(session).replace(chr(92), chr(92) * 2)}"]}}]}}'
    )
    assignments, payload = load_split_registry(registry_path)
    assert payload["generation"] == 3
    assert session_split(session, assignments) == "test"
    assert session_split(tmp_path / "new_session", assignments) == "train"


def test_worldmodel_builder_keeps_final_test_frozen(tmp_path):
    session = tmp_path / "final"
    registry_path = tmp_path / "splits.json"
    registry_path.write_text(
        '{"groups": [{"split": "final_test", "paths": ['
        f'"{str(session).replace(chr(92), chr(92) * 2)}"]}}]}}'
    )
    assignments, _ = load_split_registry(registry_path)
    assert session_split(session, assignments) == "final_test"


def test_worldmodel_builder_rejects_aggregate_timing_quarantine():
    assert episode_timing_eligible({})
    assert episode_timing_eligible({"timing_healthy": True})
    assert not episode_timing_eligible({
        "timing_healthy": False,
        "timing_health_reasons": ["sim_step_p95"],
    })


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
