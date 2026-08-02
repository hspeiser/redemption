import json
from pathlib import Path

from aigp.vq2_pipeline_dashboard import PipelineScanner


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pipeline_dashboard_filters_timing_and_reports_record(tmp_path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    run = data / "training" / "campaign" / "20260802_120000"
    run.mkdir(parents=True)
    rows = [
        {"episode": 0, "finished": True, "official_elapsed_s": 37.2,
         "timing_healthy": True, "gate_reached": 17, "schedule_arm": "champion"},
        {"episode": 1, "finished": True, "official_elapsed_s": 20.1,
         "timing_healthy": False, "gate_reached": 17, "schedule_arm": "candidate"},
        {"episode": 2, "finished": False, "timing_healthy": True,
         "gate_reached": 6, "failure": "collision", "schedule_arm": "candidate"},
    ]
    (run / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    write_json(data / "corpus_manifests" / "vq2_splits_v1.json", {"generation": 1})
    write_json(data / "corpus_manifests" / "vq2_canonical_x.json", {"entry_count": 12})
    write_json(
        data / "worldmodel" / "g0g16_master_currentera_v1_registry" / "manifest.json",
        {"splits": {"train": {"transitions": 1234}}},
    )
    model_dir = data / "worldmodel" / "v1_allgate_registry_flywheel"
    model_dir.mkdir(parents=True)
    (model_dir / "residual_ensemble_v1.pt").write_bytes(b"model")
    write_json(
        model_dir / "fresh_registry_audit_h32.json",
        {"overall": {"v1": {"position_m": {"median": 0.2, "p90": 0.4}}}},
    )
    write_json(repo / "data" / "vq2_flywheel_cycle_x.json", {"cycle": "x"})

    snapshot = PipelineScanner(
        repo=repo,
        data_root=data,
        training_root=data / "training",
        worldmodel_root=data / "worldmodel",
        manifest_root=data / "corpus_manifests",
    ).snapshot()

    assert snapshot["metrics"][0]["value"] == 37.2
    assert snapshot["metrics"][1]["value"] == 50.0
    assert snapshot["metrics"][4]["value"] == 1234
    assert snapshot["metrics"][5]["value"] == 0.4
    assert snapshot["gate_failures"] == [{"gate": "Gate 6", "count": 1}]
    assert len(snapshot["timeline"]) == 1
