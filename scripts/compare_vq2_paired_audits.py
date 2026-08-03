"""Compare baseline/candidate per-world VQ2 audits with paired statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_worlds(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {"finished", "finish_time_s", "failure_gate"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    result = {key: np.asarray(data[key]) for key in required}
    result["world_id"] = (
        np.asarray(data["world_id"])
        if "world_id" in data.files
        else np.arange(len(result["finished"]), dtype=np.int64)
    )
    return result


def validate_worlds(worlds: dict[str, np.ndarray], label: str) -> None:
    lengths = {key: len(np.asarray(value)) for key, value in worlds.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{label} arrays have different lengths: {lengths}")
    if not lengths or next(iter(lengths.values())) == 0:
        raise ValueError(f"{label} contains no worlds")
    world_id = np.asarray(worlds["world_id"])
    if len(np.unique(world_id)) != len(world_id):
        raise ValueError(f"{label} world_id values are not unique")
    finished = np.asarray(worlds["finished"], bool)
    finish_time = np.asarray(worlds["finish_time_s"], float)
    if not np.all(np.isfinite(finish_time[finished])):
        raise ValueError(f"{label} has non-finite finish times for finishes")


def bootstrap_statistic(
    values: np.ndarray,
    statistic,
    *,
    draws: int,
    rng: np.random.Generator,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Bootstrap without allocating draws-by-worlds for the entire audit."""
    values = np.asarray(values)
    output = np.empty(draws, float)
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        indices = rng.integers(
            0, len(values), size=(stop - start, len(values))
        )
        output[start:stop] = statistic(values[indices], axis=1)
    return output


def percentile_interval(samples: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    values = np.union1d(a, b)
    pa = np.asarray([(a == value).mean() for value in values], float)
    pb = np.asarray([(b == value).mean() for value in values], float)
    midpoint = 0.5 * (pa + pb)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        keep = p > 0
        return float(np.sum(p[keep] * np.log2(p[keep] / q[keep])))

    return 0.5 * kl(pa, midpoint) + 0.5 * kl(pb, midpoint)


def compare(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    bootstrap: int = 20000,
    seed: int = 20260802,
    finish_rate_tolerance: float = 0.03,
    finish_time_tolerance_s: float = 0.15,
    terminal_js_tolerance_bits: float = 0.02,
) -> dict[str, object]:
    validate_worlds(baseline, "baseline")
    validate_worlds(candidate, "candidate")
    if not np.array_equal(baseline["world_id"], candidate["world_id"]):
        raise ValueError("world_id arrays differ; audit is not paired")
    base_finished = baseline["finished"].astype(bool)
    cand_finished = candidate["finished"].astype(bool)
    n = len(base_finished)
    rng = np.random.default_rng(seed)
    paired_finish = cand_finished.astype(float) - base_finished.astype(float)
    finish_boot = bootstrap_statistic(
        paired_finish, np.mean, draws=bootstrap, rng=rng
    )

    both = base_finished & cand_finished
    time_delta = (
        candidate["finish_time_s"][both] - baseline["finish_time_s"][both]
    )
    if len(time_delta):
        time_boot = bootstrap_statistic(
            time_delta, np.median, draws=bootstrap, rng=rng
        )
        time_ci = percentile_interval(time_boot)
        time_median = float(np.median(time_delta))
    else:
        time_ci = [float("nan"), float("nan")]
        time_median = float("nan")

    # Treat a finish as its own terminal category for histogram parity.
    base_terminal = np.where(base_finished, -1, baseline["failure_gate"])
    cand_terminal = np.where(cand_finished, -1, candidate["failure_gate"])
    js_bits = js_divergence(base_terminal, cand_terminal)
    finish_delta = float(paired_finish.mean())
    finish_pass = abs(finish_delta) <= finish_rate_tolerance
    time_pass = (
        bool(len(time_delta))
        and abs(time_median) <= finish_time_tolerance_s
    )
    histogram_pass = js_bits <= terminal_js_tolerance_bits

    def histogram(values: np.ndarray) -> dict[str, int]:
        unique, counts = np.unique(values, return_counts=True)
        return {str(int(key)): int(count) for key, count in zip(unique, counts)}

    return {
        "worlds": n,
        "baseline_finish_rate": float(base_finished.mean()),
        "candidate_finish_rate": float(cand_finished.mean()),
        "paired_finish_rate_delta": finish_delta,
        "paired_finish_rate_delta_ci95": percentile_interval(finish_boot),
        "candidate_only_finishes": int((cand_finished & ~base_finished).sum()),
        "baseline_only_finishes": int((base_finished & ~cand_finished).sum()),
        "both_finished": int(both.sum()),
        "paired_finish_time_delta_median_s": time_median,
        "paired_finish_time_delta_ci95_s": time_ci,
        "baseline_terminal_histogram": histogram(base_terminal),
        "candidate_terminal_histogram": histogram(cand_terminal),
        "terminal_histogram_js_bits": js_bits,
        "acceptance": {
            "finish_rate_tolerance": finish_rate_tolerance,
            "finish_rate_pass": finish_pass,
            "finish_time_tolerance_s": finish_time_tolerance_s,
            "finish_time_pass": time_pass,
            "terminal_js_tolerance_bits": terminal_js_tolerance_bits,
            "terminal_histogram_pass": histogram_pass,
            "pass": bool(finish_pass and time_pass and histogram_pass),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--finish-rate-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--finish-time-tolerance-s", type=float, default=0.15
    )
    parser.add_argument(
        "--terminal-js-tolerance-bits", type=float, default=0.02
    )
    parser.add_argument(
        "--require-pass", action="store_true",
        help="Exit non-zero when any Layer-2 acceptance criterion fails.",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(
        load_worlds(args.baseline),
        load_worlds(args.candidate),
        bootstrap=args.bootstrap,
        seed=args.seed,
        finish_rate_tolerance=args.finish_rate_tolerance,
        finish_time_tolerance_s=args.finish_time_tolerance_s,
        terminal_js_tolerance_bits=args.terminal_js_tolerance_bits,
    )
    text = json.dumps(report, indent=2, allow_nan=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")
    return 0 if not args.require_pass or report["acceptance"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
