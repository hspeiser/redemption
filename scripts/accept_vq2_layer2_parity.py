"""Gate paired rollout artifacts behind all controller-parity layers.

This script does not launch the simulator or generate rollouts. It verifies
that the exact-action oracle and same-state shadow probe passed, checks the
required audit size, computes paired statistics, and writes one hash-addressed
acceptance report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=sorted(STAGE_WORLDS), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--seed-registry",
        type=Path,
        default=REPO / "data/lineopt/liveteacher_layer2_seeds_v1.json",
    )
    parser.add_argument(
        "--noise-arm", choices=("clean", "impulse"), default="clean"
    )
    parser.add_argument(
        "--audit-device", choices=("cpu", "cuda"), default="cpu",
        help=("Device shared by both rollout arms and the shadow probe. "
              "CPU and CUDA random tapes are not seed-equivalent."),
    )
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
    shadow = json.loads(args.shadow.read_text())
    if shadow.get("schema") != "vq2_controller_shadow_parity_v1":
        raise SystemExit("shadow artifact has the wrong schema")
    if not shadow.get("PASS", False):
        raise SystemExit("same-state shadow parity has not passed")
    if float(shadow.get("pct_worlds_diverged_1e-4", 1.0)) != 0.0:
        raise SystemExit("shadow probe has controller divergences above 1e-4")
    if float(shadow.get("max_diff_p95", float("inf"))) > 1e-5:
        raise SystemExit("shadow probe p95 action error exceeds 1e-5")
    if int(shadow.get("worlds", 0)) < 64:
        raise SystemExit("shadow probe requires at least 64 worlds")
    if str(shadow.get("device")) != args.audit_device:
        raise SystemExit("shadow probe device differs from audit device")
    expected_config_hash = layer1.get("config_sha256")
    if shadow.get("config_sha256") != expected_config_hash:
        raise SystemExit("shadow config hash differs from Layer 1")
    seed_registry = json.loads(args.seed_registry.read_text())
    stage_registry = seed_registry[args.stage]
    expected_seed = int(stage_registry[f"{args.noise_arm}_seed"])
    expected_worlds = int(stage_registry["worlds"])
    minimum_worlds = STAGE_WORLDS[args.stage]
    if expected_worlds < minimum_worlds:
        raise SystemExit(
            f"seed registry weakens {args.stage} below {minimum_worlds} worlds"
        )

    def metadata(path: Path) -> dict[str, object]:
        data = np.load(path, allow_pickle=False)
        required = {"seed", "controller", "config", "ensemble"}
        missing = required.difference(data.files)
        if missing:
            raise SystemExit(f"{path} lacks audit metadata {sorted(missing)}")
        row = {
            "seed": int(np.asarray(data["seed"]).item()),
            "controller": str(np.asarray(data["controller"]).item()),
            "config": Path(str(np.asarray(data["config"]).item())),
            "ensemble": np.asarray(data["ensemble"]).astype(str).tolist(),
        }
        row["device"] = (
            str(np.asarray(data["device"]).item())
            if "device" in data.files else None
        )
        return row

    baseline_meta = metadata(args.baseline)
    candidate_meta = metadata(args.candidate)
    if baseline_meta["controller"] != "scalar":
        raise SystemExit("baseline artifact is not labeled scalar")
    if candidate_meta["controller"] != "batched":
        raise SystemExit("candidate artifact is not labeled batched")
    if baseline_meta["seed"] != expected_seed or candidate_meta[
        "seed"
    ] != expected_seed:
        raise SystemExit(
            f"{args.stage}/{args.noise_arm} requires seed {expected_seed}"
        )
    if baseline_meta["ensemble"] != candidate_meta["ensemble"]:
        raise SystemExit("baseline and candidate ensemble lists differ")
    for label, row in (
        ("baseline", baseline_meta), ("candidate", candidate_meta)
    ):
        if row["device"] is not None and row["device"] != args.audit_device:
            raise SystemExit(f"{label} artifact device differs from audit device")
    shadow_ensemble = [str(Path(p).resolve()) for p in shadow["ensemble"]]
    audit_ensemble = [
        str(Path(p).resolve()) for p in baseline_meta["ensemble"]
    ]
    if shadow_ensemble != audit_ensemble:
        raise SystemExit("shadow and rollout ensemble lists differ")
    for label, row in (
        ("baseline", baseline_meta), ("candidate", candidate_meta)
    ):
        config_path = row["config"]
        if not config_path.exists():
            raise SystemExit(f"{label} config does not exist: {config_path}")
        if expected_config_hash and sha256(config_path) != expected_config_hash:
            raise SystemExit(f"{label} config hash differs from Layer 1")

    baseline = load_worlds(args.baseline)
    candidate = load_worlds(args.candidate)
    if len(baseline["world_id"]) != expected_worlds:
        raise SystemExit(
            f"{args.stage} audit requires exactly {expected_worlds} worlds; "
            f"got {len(baseline['world_id'])}"
        )
    expected_ids = (
        np.int64(expected_seed) * np.int64(1_000_000)
        + np.arange(expected_worlds, dtype=np.int64)
    )
    if not np.array_equal(baseline["world_id"], expected_ids):
        raise SystemExit("baseline world IDs do not match the frozen registry")
    if not np.array_equal(candidate["world_id"], expected_ids):
        raise SystemExit("candidate world IDs do not match the frozen registry")
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
        "schema": "vq2_controller_parity_acceptance_v2",
        "stage": args.stage,
        "noise_arm": args.noise_arm,
        "minimum_worlds": minimum_worlds,
        "expected_worlds": expected_worlds,
        "expected_seed": expected_seed,
        "audit_device": args.audit_device,
        "git_revision": git_revision(),
        "seed_registry": {
            "path": str(args.seed_registry.resolve()),
            "sha256": sha256(args.seed_registry),
        },
        "layer1": {
            "path": str(args.layer1.resolve()),
            "sha256": sha256(args.layer1),
            "config_sha256": layer1.get("config_sha256"),
            "action_max_abs_error": layer1.get("final", {}).get("max"),
        },
        "shadow": {
            "path": str(args.shadow.resolve()),
            "sha256": sha256(args.shadow),
            "worlds": shadow.get("worlds"),
            "seed": shadow.get("seed"),
            "max_diff_p95": shadow.get("max_diff_p95"),
            "pct_worlds_diverged_1e-4": shadow.get(
                "pct_worlds_diverged_1e-4"
            ),
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
