import json

from aigp.vq2_dashboard import VQ2Dashboard


class _Localizer:
    def dashboard_snapshot(self):
        return {}, {}


def test_dashboard_history_preserves_runtime_provenance(tmp_path):
    history = tmp_path / "history.jsonl"
    dashboard = VQ2Dashboard(
        _Localizer(), [[0.0, 0.0, 0.0]], history_path=history,
    )
    provenance = {
        "config_sha256": "config",
        "controller_config_sha256": "controller",
        "schedule_sha256": "schedule",
        "actor_sha256": "actor",
        "secondary_actor_sha256": "secondary",
        "reference_sha256": "reference",
        "seed_checkpoint_sha256": "critic",
        "map_sha256": "map",
        "primary_detector_sha256": "primary",
        "refiner_detector_sha256": "refiner",
        "gate_primary_detector_sha256": "gate-primary",
        "crop_detector_sha256": "crop",
        "proposal_model_sha256": "proposal",
        "calibration_sha256": "calibration",
        "line_model_sha256": "line-model",
    }
    dashboard.record_episode({
        "episode": 3,
        "run_id": "run",
        "schedule_arm": "candidate",
        "crossing_offsets": [{"gate": 4, "step": 239}],
        "official_elapsed_s": 7.9,
        "failure": None,
        "deterministic": True,
        **provenance,
    })

    row = json.loads(history.read_text().strip())
    for name, value in provenance.items():
        assert row[name] == value
