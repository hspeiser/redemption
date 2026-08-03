"""Build a phase-aware residual pulse for one VQ2 course segment.

The schedule is expressed in normalized residual coordinates, matching both
``fastsim_train_ppo.py --fixed-residual-schedule`` and the live teacher's
``--ppo-residual-schedule`` path.  Only the requested target-gate segment is
modified; every other gate remains exactly zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=int, required=True)
    parser.add_argument("--race-gates", type=int, default=17)
    parser.add_argument(
        "--pitch-knots",
        type=float,
        nargs="+",
        required=True,
        help="Normalized pitch residual at uniformly spaced segment phases.",
    )
    parser.add_argument(
        "--thrust-knots",
        type=float,
        nargs="+",
        default=None,
        help="Optional normalized thrust residual at the same phases.",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if not 0 <= args.gate < args.race_gates:
        parser.error("--gate must be inside --race-gates")
    if len(args.pitch_knots) < 2:
        parser.error("at least two --pitch-knots are required")
    if args.thrust_knots is not None and (
        len(args.thrust_knots) != len(args.pitch_knots)
    ):
        parser.error("--thrust-knots must match --pitch-knots in length")

    knots = len(args.pitch_knots)
    schedule = np.zeros((args.race_gates, knots, 4), np.float32)
    schedule[args.gate, :, 1] = np.asarray(args.pitch_knots, np.float32)
    if args.thrust_knots is not None:
        schedule[args.gate, :, 3] = np.asarray(
            args.thrust_knots, np.float32
        )
    if np.max(np.abs(schedule)) > 1.0:
        parser.error("normalized residual knots must remain in [-1, 1]")

    payload = {
        "kind": "vq2_phase_aware_straight_residual",
        "label": args.label,
        "gate": args.gate,
        "knots": knots,
        "phases": np.linspace(0.0, 1.0, knots).tolist(),
        "residual_schedule": schedule.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
