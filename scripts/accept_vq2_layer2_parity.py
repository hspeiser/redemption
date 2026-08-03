"""Gate paired rollout artifacts behind Layer-1 controller parity.

This script does not launch the simulator or generate rollouts. It verifies
that the exact-action oracle passed, checks the required audit size, computes
paired statistics, and writes one hash-addressed acceptance report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.compare_vq2_paired_audits import compare, load_worlds  # noqa: E402


STAGE_WORLDS = {"development": 256, "final": 768}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer1", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=sorted(STAGE_WORLDS), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    parser.add_argument("--finish-rate-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--finish-time-tolerance-s", type=float, default=0.15
    )
    parser.add_argument(
        "--terminal-js-tolerance-bits", type=float, default=0.02
    )
    args = parser.parse_args()

    layer1 = json.loads(args.layer1.read_text())
    if not layer1.get("PASS", False):
        raise SystemExit(
            "Layer 1 has not passed; paired rollout acceptance is blocked"
        )
    baseline = load_worlds(args.baseline)
    candidate = load_worlds(args.candidate)
    minimum_worlds = STAGE_WORLDS[args.stage]
    if len(baseline["world_id"]) < minimum_worlds:
        raise SystemExit(
            f"{args.stage} audit requires at least {minimum_worlds} worlds; "
            f"got {len(baseline['world_id'])}"
        )
    statistics = compare(
        baseline,
        candidate,
        bootstrap=args.bootstrap,
        seed=args.bootstrap_seed,
        finish_rate_tolerance=args.finish_rate_tolerance,
        finish_time_tolerance_s=args.finish_time_tolerance_s,
        terminal_js_tolerance_bits=args.terminal_js_tolerance_bits,
    )
    report = {
        "schema": "vq2_controller_layer2_parity_v1",
        "stage": args.stage,
        "minimum_worlds": minimum_worlds,
        "git_revision": git_revision(),
        "layer1": {
            "path": str(args.layer1.resolve()),
            "sha256": sha256(args.layer1),
            "config_sha256": layer1.get("config_sha256"),
            "action_max_abs_error": layer1.get("final", {}).get("max"),
        },
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sha256": sha256(args.baseline),
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256(args.candidate),
        },
        "statistics": statistics,
        "PASS": bool(statistics["acceptance"]["pass"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
