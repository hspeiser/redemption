"""Paired finite-difference probes around a protected residual schedule.

This is intentionally smaller and more diagnostic than CEM.  Each candidate
changes one action axis for one gate by a fixed amount across all phase knots.
Every candidate is evaluated separately with the same seed under every world
model, so runtime vision/relocation/impulse draws are exactly common-random.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from scripts.fastsim_train_ppo import load_demo_states  # noqa: E402
from scripts.optimize_vq2_residual_schedule import (  # noqa: E402
    evaluate,
    load_actor,
    objective,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gates", default="1,2,3")
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20268251)
    parser.add_argument("--aleatoric-scale", type=float, default=1.5)
    parser.add_argument("--impulse-rate-hz", type=float, default=0.08)
    parser.add_argument("--reliability-floor", type=float, default=0.90)
    parser.add_argument("--time-weight", type=float, default=100.0)
    parser.add_argument("--tier-target", type=float, default=8.7)
    parser.add_argument("--tier-bonus", type=float, default=300.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base_payload = json.loads(args.base.read_text())
    knots = int(base_payload["knots"])
    base = np.asarray(base_payload["residual_schedule"], np.float64)
    if base.shape != (5, knots, 4):
        raise ValueError(f"unexpected base schedule shape {base.shape}")
    gates = [int(value) for value in args.gates.split(",") if value.strip()]
    if not gates or not all(0 <= gate < 5 for gate in gates):
        raise ValueError("--gates must contain indices in 0..4")

    labels = ["baseline"]
    candidates = [base.copy()]
    for gate in gates:
        for axis in range(4):
            for sign in (-1.0, 1.0):
                candidate = base.copy()
                candidate[gate, :, axis] = np.clip(
                    candidate[gate, :, axis] + sign * args.delta,
                    -0.45,
                    0.45,
                )
                labels.append(
                    f"gate{gate}_axis{axis}_{sign * args.delta:+.4f}"
                )
                candidates.append(candidate)

    actor, obs_mean, obs_var = load_actor(args.actor, args.device)
    ensembles = []
    metadata = []
    for path in args.ensemble:
        ensemble, row = ResidualEnsemble.load(path, args.device)
        ensemble.eval()
        ensembles.append(ensemble)
        metadata.append(row)
    model = SurrogateModel.load(args.model)
    controller_model = SurrogateModel.load(args.controller_model)
    demo = load_demo_states(args.demo, args.map)
    keep = np.asarray(demo["gate"]) < 5
    demo = {key: np.asarray(value)[keep] for key, value in demo.items()}

    rows = []
    started = time.time()
    candidate_matrix = np.asarray(candidates).reshape(len(candidates), -1)
    reports_by_model = []
    for model_index, ensemble in enumerate(ensembles):
        torch.manual_seed(args.seed)
        reports_by_model.append(evaluate(
                candidate_matrix,
                worlds=args.worlds,
                knots=knots,
                actor=actor,
                obs_mean=obs_mean,
                obs_var=obs_var,
                model=model,
                controller_model=controller_model,
                ensemble=ensemble,
                demo_path=args.demo,
                demo_states=demo,
                map_path=args.map,
                obstacles=args.obstacles,
                teacher_config=args.teacher_config,
                device=args.device,
                aleatoric_scale=args.aleatoric_scale,
                impulse_rate_hz=args.impulse_rate_hz,
                seed=args.seed,
            ))
        print(json.dumps({
            "completed_model": model_index + 1,
            "models": len(ensembles),
            "probes": len(candidates),
        }), flush=True)
    for index, (label, schedule) in enumerate(zip(labels, candidates)):
        reports = [model_reports[index] for model_reports in reports_by_model]
        scores = [
            objective(
                report,
                args.reliability_floor,
                args.time_weight,
                args.tier_target,
                args.tier_bonus,
            )
            for report in reports
        ]
        row = {
            "index": index,
            "label": label,
            "worst_score": float(min(scores)),
            "reports_by_model": reports,
            "residual_schedule": schedule.tolist(),
        }
        rows.append(row)
        print(json.dumps({
            "probe": index + 1,
            "probes": len(candidates),
            "label": label,
            "worst_score": row["worst_score"],
            "reports_by_model": reports,
        }), flush=True)

    rows.sort(key=lambda row: row["worst_score"], reverse=True)
    winner = rows[0]
    output = {
        "actor": str(args.actor),
        "knots": knots,
        "residual_schedule": winner["residual_schedule"],
        "winner": winner,
        "baseline": next(row for row in rows if row["label"] == "baseline"),
        "ranking": rows,
        "worlds": args.worlds,
        "seed": args.seed,
        "delta": args.delta,
        "gates": gates,
        "world_model_datasets": [row.get("dataset") for row in metadata],
        "elapsed_s": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "winner": winner["label"],
        "winner_reports_by_model": winner["reports_by_model"],
        "elapsed_s": output["elapsed_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
