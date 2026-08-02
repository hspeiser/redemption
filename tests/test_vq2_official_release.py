from types import SimpleNamespace
import time

from aigp.rl.vq2_env import VQ2LiveEnv


def test_negative_release_margin_predicts_early_start() -> None:
    env = VQ2LiveEnv.__new__(VQ2LiveEnv)
    env.config = SimpleNamespace(
        race_start_timeout_s=1.0,
        official_release_margin_ms=-500.0,
    )
    env.mavlink = SimpleNamespace(race_status={
        "active_gate": 0,
        "race_start_ms": 3000,
        "sim_boot_ms": 2500,
        "race_finish_ns": -1,
        "wall_ns": time.time_ns(),
    })

    status, _ = env._wait_for_official_release()

    assert status["predictive_release"] is True
    assert status["source_sim_boot_ms"] == 2500
    assert 2500 <= status["sim_boot_ms"] < 3000
