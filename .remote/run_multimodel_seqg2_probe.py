"""Re-audit gate-2 sequential modes after exact schedule/demo frame parity."""

import json
from pathlib import Path

import numpy as np
import torch

from aigp.fastsim.sysid import SurrogateModel
from aigp.fastsim.worldmodel import ResidualEnsemble
from scripts.fastsim_train_ppo import load_live_teacher_config
from scripts.optimize_vq2_g0g4_worldmodel import (
    decoded_demo,
    evaluate,
    release_states,
)


ROOT = Path(r"C:\Users\henry\aigp")
DATA = ROOT / "data"
WORLD = ROOT / "worldmodel"
MODEL_PATHS = {
    "v13": WORLD / "multimodel_seqg2_probe" / "residual_ensemble_current_v13.pt",
    "v19": WORLD / "multimodel_seqg2_probe" / "residual_ensemble_v19.pt",
    "v20": WORLD / "v21_full16_baseline" / "residual_ensemble_v20.pt",
    "v21": WORLD / "v21_full16_baseline" / "residual_ensemble_v21.pt",
}
DATASET = WORLD / "g0g4_current_aug_17_full16_baseline"
DEMO = DATA / "vq2_g0g1fast9267_g2plus_clean_demo.npz"
MAP = DATA / "vq2_runtime_map_g9g15fix.json"
OBSTACLES = DATA / "vq2_obstacles_inflated.json"
CONFIG = ROOT / "training" / "fast926" / "config.json"
PLANT = SurrogateModel.load(DATA / "fastsim_model_v2.json")
DEMO_STATES = decoded_demo(DEMO, MAP)
SPAWN_STATES = release_states(DATASET)
ARRAYS, _ = load_live_teacher_config(CONFIG, 1)
THETA = np.r_[
    ARRAYS["action_leads"][0],
    ARRAYS["thrust_scales"][0],
    ARRAYS["trajectory_velocity_scales"][0],
    ARRAYS["trajectory_blends"][0],
][None]


def summarize(report: dict) -> dict:
    return {
        "finish_rate": report["finish_rate"],
        "median_s": report["median_s"],
        "p90_s": report["p90_s"],
        "clearance_p10_m": report["clearance_p10_m"],
        "support_z_mean": report["support_z_mean"],
        "support_z_p90": report["support_z_p90"],
        "disagreement_mean": report["disagreement_mean"],
        "gate_time_median_s": report["gate_time_median_s"],
        "gate_clearance_p10_m": report["gate_clearance_p10_m"],
        "failure_histogram": report["failure_histogram"],
    }


results = {}
for model_name, path in MODEL_PATHS.items():
    ensemble, _metadata = ResidualEnsemble.load(path, "cuda")
    ensemble.eval()
    results[model_name] = {}
    for label, speed in [("nearest", None), *[(f"seq_{s:.2f}", s) for s in (.85, .90, .95, 1.0, 1.05)]]:
        overrides = None
        if speed is not None:
            enabled = np.zeros(5, bool)
            enabled[2] = True
            speeds = np.ones(5, np.float32)
            speeds[2] = speed
            overrides = {
                "reference_sequential_enabled": enabled,
                "reference_sequential_speeds": speeds,
            }
        torch.manual_seed(20260801)
        report = evaluate(
            THETA,
            worlds=512,
            model=PLANT,
            ensemble=ensemble,
            demo_path=DEMO,
            map_path=MAP,
            obstacles=OBSTACLES,
            teacher_config=CONFIG,
            demo_states=DEMO_STATES,
            spawn_states=SPAWN_STATES,
            device="cuda",
            robust=True,
            controller_rate_gain=np.asarray(PLANT.rate_gain),
            aleatoric_scale=1.5,
            live_estimator_realism=True,
            impulse_rate_hz=0.08,
            controller_fixed_overrides=overrides,
        )[0]
        results[model_name][label] = summarize(report)
        print(
            model_name,
            label,
            report["finish_rate"],
            report["median_s"],
            report["gate_clearance_p10_m"],
            flush=True,
        )

out = WORLD / "multimodel_seqg2_probe" / "results_schedule_parity.json"
out.write_text(json.dumps(results, indent=2))
print("WROTE", out, flush=True)
