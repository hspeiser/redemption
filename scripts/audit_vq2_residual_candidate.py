"""Run a never-seen, fixed-seed audit of one residual schedule candidate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel
from aigp.fastsim.worldmodel import ResidualEnsemble
from scripts.fastsim_train_ppo import load_demo_states
from scripts.optimize_vq2_residual_schedule import evaluate, load_actor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path)
    parser.add_argument(
        "--candidate-label",
        help=(
            "optional label from a probe result's ranking; audits that row "
            "instead of the top-level residual_schedule"
        ),
    )
    parser.add_argument(
        "--protected-base", action="store_true",
        help="Zero the residual actor and schedule; audit the reference only.",
    )
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--aleatoric-scale", type=float, default=1.5)
    parser.add_argument("--impulse-rate-hz", type=float, default=0.08)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--residual-gates",
        help=(
            "Comma-separated deployment gate mask. Defaults to the value "
            "stored in --teacher-config."
        ),
    )
    parser.add_argument(
        "--residual-phase-windows",
        help=(
            "Comma-separated gate:start:end windows. Defaults to the "
            "teacher config."
        ),
    )
    parser.add_argument(
        "--multigate-estimator", action="store_true",
        help="Model course-wide mapped-gate landmark availability.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    teacher_payload = json.loads(args.teacher_config.read_text())
    teacher_args = teacher_payload.get("args", teacher_payload)
    residual_gate_text = (
        args.residual_gates
        if args.residual_gates is not None
        else str(teacher_args.get("residual_gates", ""))
    )
    residual_gates = tuple(
        int(value.strip())
        for value in residual_gate_text.split(",")
        if value.strip()
    )
    phase_text = (
        args.residual_phase_windows
        if args.residual_phase_windows is not None
        else str(teacher_args.get("residual_phase_windows", ""))
    )
    residual_phase_windows = {}
    for item in phase_text.split(","):
        if not item.strip():
            continue
        gate, start, end = item.split(":")
        residual_phase_windows[int(gate)] = (float(start), float(end))

    if args.candidate is None:
        knots = 3
        theta = np.zeros((1, 5 * knots * 4), np.float64)
    else:
        payload = json.loads(args.candidate.read_text())
        knots = int(payload["knots"])
        if args.candidate_label:
            matches = [
                row for row in payload.get("ranking", [])
                if row.get("label") == args.candidate_label
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"candidate label {args.candidate_label!r} matched "
                    f"{len(matches)} ranking rows"
                )
            payload = {**payload, "residual_schedule": matches[0][
                "residual_schedule"
            ]}
        theta = np.asarray(
            payload["residual_schedule"], np.float64
        ).reshape(1, -1)
    actor, obs_mean, obs_var = load_actor(args.actor, args.device)
    if args.protected_base:
        actor = copy.deepcopy(actor)
        actor.mean.weight.data.zero_()
        actor.mean.bias.data.zero_()
    ensemble, metadata = ResidualEnsemble.load(args.ensemble, args.device)
    ensemble.eval()
    model = SurrogateModel.load(args.model)
    controller_model = SurrogateModel.load(args.controller_model)
    demo = load_demo_states(args.demo, args.map)
    keep = np.asarray(demo["gate"]) < 5
    demo = {key: np.asarray(value)[keep] for key, value in demo.items()}

    def run(impulse_rate_hz: float, seed: int) -> dict:
        return evaluate(
            theta,
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
            impulse_rate_hz=impulse_rate_hz,
            seed=seed,
            multigate_estimator=args.multigate_estimator,
            residual_scale=args.residual_scale,
            residual_gates=residual_gates,
            residual_phase_windows=residual_phase_windows,
        )[0]

    result = {
        "candidate": str(args.candidate) if args.candidate else None,
        "protected_base": bool(args.protected_base),
        "world_model_dataset": metadata.get("dataset"),
        "worlds_per_arm": args.worlds,
        "aleatoric_scale": args.aleatoric_scale,
        "multigate_estimator": bool(args.multigate_estimator),
        "residual_scale": float(args.residual_scale),
        "residual_gates": list(residual_gates),
        "residual_phase_windows": {
            str(gate): list(window)
            for gate, window in residual_phase_windows.items()
        },
        "clean_seed": args.seed,
        "impulse_seed": args.seed + 1,
        "clean": run(0.0, args.seed),
        "impulse": run(args.impulse_rate_hz, args.seed + 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
