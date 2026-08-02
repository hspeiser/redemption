"""Targeted robust search around a gates-0..4 world-model champion.

This is deliberately a local search: broad CEM finds a safe corridor, then this
script looks for the one-frame speed improvements that CEM's quantized timing
objective tends to miss. Candidates are screened in modest world batches and
the best are re-evaluated with many randomized ensemble worlds.
"""

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


def load_theta(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    return np.asarray(
        payload["leads"]
        + payload["thrust_scales"]
        + payload["velocity_scales"]
        + payload["trajectory_blends"],
        dtype=np.float64,
    )


def make_candidates(base: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = [base.copy()]

    # One-knob probes expose which segment can buy a frame without moving the
    # whole trajectory. Positive speed/blend/thrust probes are intentionally
    # denser because the input champion is already safe and only one frame shy.
    for offset, deltas in (
        (5, (-0.03, 0.02, 0.04)),
        (10, (-0.05, 0.03, 0.06, 0.10)),
        (15, (-0.015, 0.008, 0.015, 0.025)),
    ):
        for gate in range(5):
            for delta in deltas:
                row = base.copy()
                row[offset + gate] += delta
                rows.append(row)
    for gate in range(5):
        for delta in (-1.0, 1.0):
            row = base.copy()
            row[gate] += delta
            rows.append(row)

    # Correlated local perturbations can improve the approach and crossing
    # together. Keep them tight enough to stay inside demonstrated support.
    while len(rows) < count:
        row = base.copy()
        row[:5] += rng.choice((-1.0, 0.0, 1.0), 5, p=(0.10, 0.80, 0.10))
        row[5:10] += rng.normal(0.012, 0.030, 5)
        row[10:15] += rng.normal(0.035, 0.065, 5)
        row[15:20] += rng.normal(0.006, 0.018, 5)
        rows.append(row)
    return np.asarray(rows[:count])


def rank_key(row: dict) -> tuple:
    report = row["report"]
    median = report["median_s"] if report["median_s"] is not None else 99.0
    p90 = report["p90_s"] if report["p90_s"] is not None else 99.0
    # Reliability remains the hard requirement. Above 90%, prefer lower
    # median, then margin above the 90% floor, tail time, and clearance.
    eligible = report["finish_rate"] >= 0.90
    return (
        0 if eligible else 1,
        median if eligible else -report["finish_rate"],
        -report["finish_rate"] if eligible else p90,
        p90,
        -report["clearance_p10_m"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--controller-model", type=Path,
        help="Rate-command calibration used by the live trajectory tracker.",
    )
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--screen-worlds", type=int, default=64)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument("--finalist-worlds", type=int, default=512)
    parser.add_argument("--locked-worlds", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aleatoric-scale", type=float, default=0.0)
    args = parser.parse_args()

    started = time.time()
    model = SurrogateModel.load(args.model)
    controller_rate_gain = (
        SurrogateModel.load(args.controller_model).rate_gain
        if args.controller_model is not None else model.rate_gain
    )
    ensemble, metadata = ResidualEnsemble.load(args.ensemble, args.device)
    ensemble.eval()
    demo_states = decoded_demo(args.demo, args.map)
    spawn_states = release_states(args.dataset)
    candidates = make_candidates(load_theta(args.champion), args.candidates, args.seed)
    common = dict(
        model=model,
        ensemble=ensemble,
        demo_path=args.demo,
        map_path=args.map,
        obstacles=args.obstacles,
        demo_states=demo_states,
        spawn_states=spawn_states,
        device=args.device,
        robust=True,
        controller_rate_gain=controller_rate_gain,
        aleatoric_scale=args.aleatoric_scale,
    )

    screened = []
    for start in range(0, len(candidates), args.batch_size):
        stop = min(len(candidates), start + args.batch_size)
        reports = evaluate(
            candidates[start:stop], worlds=args.screen_worlds, **common
        )
        screened.extend(
            {"index": start + i, "theta": candidates[start + i], "report": r}
            for i, r in enumerate(reports)
        )
        best = sorted(screened, key=rank_key)[0]
        print(json.dumps({
            "screened": stop,
            "best_index": best["index"],
            "best": best["report"],
        }), flush=True)

    finalists = sorted(screened, key=rank_key)[:args.finalists]
    final_thetas = np.asarray([row["theta"] for row in finalists])
    final_reports = evaluate(
        final_thetas, worlds=args.finalist_worlds, **common
    )
    reranked = [
        {
            "index": finalists[i]["index"],
            "theta": final_thetas[i],
            "screen_report": finalists[i]["report"],
            "report": report,
        }
        for i, report in enumerate(final_reports)
    ]
    reranked.sort(key=rank_key)
    winner = reranked[0]
    locked = evaluate(
        winner["theta"][None], worlds=args.locked_worlds, **common
    )[0]
    clean_common = dict(common)
    clean_common["robust"] = False
    clean = evaluate(winner["theta"][None], worlds=512, **clean_common)[0]
    leads, thrust, velocity, blend = unpack(winner["theta"][None])
    output = {
        "leads": leads[0].tolist(),
        "thrust_scales": thrust[0].tolist(),
        "velocity_scales": velocity[0].tolist(),
        "trajectory_blends": blend[0].tolist(),
        "robust_eval": locked,
        "mean_model_eval": clean,
        "finalists": [
            {
                "source_index": row["index"],
                "theta": row["theta"].tolist(),
                "screen_report": row["screen_report"],
                "finalist_report": row["report"],
            }
            for row in reranked
        ],
        "search": {
            "candidates": args.candidates,
            "screen_worlds": args.screen_worlds,
            "finalist_worlds": args.finalist_worlds,
            "locked_worlds": args.locked_worlds,
            "seed": args.seed,
            "elapsed_s": time.time() - started,
        },
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
