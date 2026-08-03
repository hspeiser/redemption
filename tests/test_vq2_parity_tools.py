from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPARE = load_script("compare_vq2_paired_audits.py")


def worlds(finished, times, gates):
    return {
        "world_id": np.arange(len(finished)),
        "finished": np.asarray(finished, bool),
        "finish_time_s": np.asarray(times, float),
        "failure_gate": np.asarray(gates, int),
    }


def test_paired_comparison_reports_discordant_finishes_and_time():
    baseline = worlds(
        [True, True, False, False], [35.0, 36.0, np.nan, np.nan],
        [-1, -1, 5, 9],
    )
    candidate = worlds(
        [True, False, True, False], [34.5, np.nan, 35.0, np.nan],
        [-1, 8, -1, 9],
    )
    report = COMPARE.compare(baseline, candidate, bootstrap=1000, seed=1)
    assert report["paired_finish_rate_delta"] == 0.0
    assert report["candidate_only_finishes"] == 1
    assert report["baseline_only_finishes"] == 1
    assert report["both_finished"] == 1
    assert report["paired_finish_time_delta_median_s"] == -0.5
    assert report["terminal_histogram_js_bits"] > 0.0


def test_paired_comparison_rejects_different_world_order():
    baseline = worlds([True, False], [35.0, np.nan], [-1, 4])
    candidate = worlds([True, False], [35.0, np.nan], [-1, 4])
    candidate["world_id"] = np.array([1, 0])
    with pytest.raises(ValueError, match="not paired"):
        COMPARE.compare(baseline, candidate, bootstrap=10)


def test_identical_paired_audits_pass_all_acceptance_criteria():
    baseline = worlds(
        [True, True, False, False], [35.0, 36.0, np.nan, np.nan],
        [-1, -1, 5, 9],
    )
    report = COMPARE.compare(baseline, baseline, bootstrap=100, seed=3)
    assert report["acceptance"]["pass"] is True
    assert report["terminal_histogram_js_bits"] == 0.0
    assert report["baseline_terminal_histogram"] == {
        "-1": 2,
        "5": 1,
        "9": 1,
    }


def test_paired_comparison_rejects_duplicate_world_ids():
    baseline = worlds([True, False], [35.0, np.nan], [-1, 4])
    candidate = worlds([True, False], [35.0, np.nan], [-1, 4])
    baseline["world_id"] = np.array([7, 7])
    candidate["world_id"] = np.array([7, 7])
    with pytest.raises(ValueError, match="not unique"):
        COMPARE.compare(baseline, candidate, bootstrap=10)
