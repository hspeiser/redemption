"""Conservative one-gate-at-a-time search from a live-validated controller."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from scripts.optimize_vq2_g0g4_worldmodel import (  # noqa: E402
    decoded_demo,
    evaluate,
    release_states,
    unpack,
)


def base_theta(path: Path | None) -> np.ndarray:
    if path is None:
        return np.r_[np.zeros(5), np.ones(5), np.ones(5), np.zeros(5)]
    row = json.loads(path.read_text())
    return np.asarray(
        row["leads"] + row["thrust_scales"]
        + row["velocity_scales"] + row["trajectory_blends"], float,
    )


def candidates(base: np.ndarray, gate: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = [base.copy()]
    # Deterministic low-risk probes first; random rows fill combinations.
    for lead in (-2, -1, 0, 1, 2, 3, 4):
        row = base.copy(); row[gate] = lead; rows.append(row)
    for value in (0.94, 0.98, 1.02, 1.06, 1.10, 1.14):
        row = base.copy(); row[5 + gate] = value; rows.append(row)
    for value in (1.05, 1.15, 1.25, 1.35, 1.50, 1.65):
        for blend in (0.025, 0.05, 0.075, 0.10, 0.125, 0.15):
            row = base.copy()
            row[10 + gate] = value
            row[15 + gate] = blend
            rows.append(row)
    while len(rows) < count:
        row = base.copy()
        row[gate] = rng.integers(-2, 5)
        row[5 + gate] = rng.uniform(0.94, 1.14)
        row[10 + gate] = rng.uniform(1.0, 1.70)
        row[15 + gate] = rng.uniform(0.02, 0.16)
        rows.append(row)
    return np.asarray(rows[:count])


def key(
    row: dict,
    minimum_finish_rate: float,
    minimum_clearance: float,
) -> tuple:
    r = row["report"]
    median = r["median_s"] if r["median_s"] is not None else 99.0
    p90 = r["p90_s"] if r["p90_s"] is not None else 99.0
    safe = (
        r["finish_rate"] >= minimum_finish_rate
        and r["clearance_p10_m"] >= minimum_clearance
    )
    return (
        0 if safe else 1,
        median if safe else -r["finish_rate"],
        p90,
        -r["finish_rate"],
        -r["clearance_p10_m"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path)
    parser.add_argument("--gate", type=int, choices=range(5), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--screen-worlds", type=int, default=64)
    parser.add_argument("--finalists", type=int, default=10)
    parser.add_argument("--finalist-worlds", type=int, default=512)
    parser.add_argument("--locked-worlds", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aleatoric-scale", type=float, default=0.0)
    parser.add_argument("--minimum-finish-rate", type=float, default=0.90)
    parser.add_argument("--minimum-clearance", type=float, default=0.12)
    args = parser.parse_args()

    started = time.time()
    theta = candidates(
        base_theta(args.base), args.gate, args.candidates, args.seed
    )
    model = SurrogateModel.load(args.model)
    controller = SurrogateModel.load(args.controller_model)
    ensemble, metadata = ResidualEnsemble.load(args.ensemble, args.device)
    ensemble.eval()
    common = dict(
        model=model,
        ensemble=ensemble,
        demo_path=args.demo,
        map_path=args.map,
        obstacles=args.obstacles,
        demo_states=decoded_demo(args.demo, args.map),
        spawn_states=release_states(args.dataset),
        device=args.device,
        robust=True,
        controller_rate_gain=controller.rate_gain,
        aleatoric_scale=args.aleatoric_scale,
    )
    rows = []
    for start in range(0, len(theta), args.batch_size):
        stop = min(start + args.batch_size, len(theta))
        reports = evaluate(
            theta[start:stop], worlds=args.screen_worlds, **common
        )
        rows.extend({
            "index": start + i,
            "theta": theta[start + i],
            "report": report,
        } for i, report in enumerate(reports))
        ranking_key = lambda row: key(  # noqa: E731
            row, args.minimum_finish_rate, args.minimum_clearance
        )
        winner = min(rows, key=ranking_key)
        print(json.dumps({
            "screened": stop,
            "best_index": winner["index"],
            "best": winner["report"],
        }), flush=True)
    screen_top = sorted(rows, key=ranking_key)[:args.finalists]
    finalists = np.asarray([row["theta"] for row in screen_top])
    reports = evaluate(
        finalists, worlds=args.finalist_worlds, **common
    )
    final_rows = [{
        "index": screen_top[i]["index"],
        "theta": finalists[i],
        "screen_report": screen_top[i]["report"],
        "report": report,
    } for i, report in enumerate(reports)]
    # The finalist screen is still noisy enough for a marginal candidate to
    # cross the acceptance boundary by luck.  Lock every finalist at the full
    # world count, then rank on that report.  Evaluating one at a time keeps
    # GPU memory bounded even when locked_worlds is large.
    for row in final_rows:
        row["locked_report"] = evaluate(
            row["theta"][None], worlds=args.locked_worlds, **common
        )[0]
    final_rows.sort(key=lambda row: ranking_key({
        **row, "report": row["locked_report"]
    }))
    winner = final_rows[0]
    locked = winner["locked_report"]
    leads, thrust, velocity, blend = unpack(winner["theta"][None])
    output = {
        "gate_changed": args.gate,
        "acceptance": {
            "minimum_finish_rate": args.minimum_finish_rate,
            "minimum_clearance": args.minimum_clearance,
            "aleatoric_scale": args.aleatoric_scale,
        },
        "leads": leads[0].tolist(),
        "thrust_scales": thrust[0].tolist(),
        "velocity_scales": velocity[0].tolist(),
        "trajectory_blends": blend[0].tolist(),
        "robust_eval": locked,
        "finalists": [{
            "source_index": row["index"],
            "theta": row["theta"].tolist(),
            "screen_report": row["screen_report"],
            "finalist_report": row["report"],
            "locked_report": row["locked_report"],
        } for row in final_rows],
        "elapsed_s": time.time() - started,
        "world_model_metadata": {
            "dataset": metadata.get("dataset"),
            "base_model": metadata.get("base_model"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
