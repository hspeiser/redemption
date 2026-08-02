"""Measure the stochastic EKF correction process across all live sessions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.worldmodel import decode_observations  # noqa: E402


def quantiles(value: np.ndarray) -> dict:
    value = np.asarray(value, float)
    return {
        "count": int(len(value)),
        **{f"p{int(q * 100):02d}": float(np.quantile(value, q))
           for q in (0.5, 0.9, 0.95, 0.99)},
        "mean": float(np.mean(value)),
    }


def era(config: dict) -> str:
    args = config.get("args", config)
    return "|".join([
        f"vision={args.get('vision_hz', 'unknown')}hz-{args.get('vision_device', 'unknown')}",
        f"crop={bool(args.get('crop_tracker', False))}",
        f"map={Path(str(args.get('map', 'unknown'))).name}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gates = json.loads(args.map.read_text())["gates"]
    gate_positions = np.asarray([gate["pos"] for gate in gates], float)
    groups = defaultdict(lambda: defaultdict(list))
    session_rows = []
    for log in sorted(args.training_root.glob("**/episodes.jsonl")):
        config_path = log.parent / "config.json"
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            config = {}
        signature = era(config)
        episode_count = correction_count = 0
        for path in sorted(log.parent.glob("episode_*.npz")):
            try:
                payload = np.load(path, allow_pickle=False)
                obs = np.asarray(payload["observation"], np.float32)
                next_obs = np.asarray(payload["next_observation"], np.float32)
                post = np.asarray(payload["position"], np.float32)
                done = np.asarray(payload["done"], bool)
                timing = np.asarray(
                    payload["timing_healthy"]
                    if "timing_healthy" in payload.files
                    else np.ones(len(obs), bool), bool,
                )
                gate = np.asarray(payload["gate_index"], int)
            except (OSError, KeyError, ValueError):
                continue
            if len(obs) < 2 or post.shape != (len(obs), 3):
                continue
            current = decode_observations(obs, gate_positions)
            following = decode_observations(next_obs, gate_positions)
            current_position = np.vstack([current.position[:1], post[:-1]])
            # Semi-implicit integration matches the fastsim/world-model step.
            correction = post - current_position - following.velocity / 30.0
            norm = np.linalg.norm(correction, axis=1)
            sigma = np.asarray(current.confidence[:, 0], float) * 0.5
            speed = np.linalg.norm(current.velocity, axis=1)
            valid = (
                timing & ~done & (gate >= 0) & (gate < len(gates))
                & np.isfinite(norm) & np.isfinite(sigma) & np.isfinite(speed)
            )
            if not np.any(valid):
                continue
            episode_count += 1
            correction_count += int(valid.sum())
            for key in (signature, "ALL"):
                groups[key]["correction_norm"].extend(norm[valid].tolist())
                groups[key]["sigma"].extend(sigma[valid].tolist())
                groups[key]["speed"].extend(speed[valid].tolist())
                groups[key]["gate"].extend(gate[valid].tolist())
                groups[key]["session"].extend(
                    [str(log.parent)] * int(valid.sum())
                )
        if correction_count:
            session_rows.append({
                "path": str(log.parent), "era": signature,
                "episodes": episode_count, "rows": correction_count,
            })

    report_groups = {}
    for key, raw in groups.items():
        norm = np.asarray(raw["correction_norm"], float)
        sigma = np.asarray(raw["sigma"], float)
        speed = np.asarray(raw["speed"], float)
        gate = np.asarray(raw["gate"], int)
        rows = {
            "correction_m": quantiles(norm),
            "reported_sigma_m": quantiles(sigma),
            "jump_rate_hz": {
                str(threshold): float(np.mean(norm > threshold) * 30.0)
                for threshold in (0.10, 0.15, 0.25, 0.50, 0.75)
            },
            "by_sigma": {},
            "by_speed": {},
            "by_gate": {},
        }
        for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 1.1)):
            selected = (sigma >= lo) & (sigma < hi)
            if selected.sum() >= 50:
                rows["by_sigma"][f"{lo:.2f}-{hi:.2f}"] = quantiles(norm[selected])
        for lo in range(0, 16, 2):
            selected = (speed >= lo) & (speed < lo + 2)
            if selected.sum() >= 50:
                rows["by_speed"][f"{lo}-{lo + 2}"] = quantiles(norm[selected])
        for value in np.unique(gate):
            selected = gate == value
            if selected.sum() >= 50:
                rows["by_gate"][str(int(value))] = quantiles(norm[selected])
        report_groups[key] = rows
    report = {
        "training_root": str(args.training_root.resolve()),
        "map": str(args.map.resolve()),
        "groups": report_groups,
        "sessions": session_rows,
        "interpretation": (
            "Correction is logged_position(t+1)-logged_position(t)-"
            "logged_velocity(t+1)/30; it mixes EKF measurement updates with "
            "residual velocity/position timing mismatch and must not be learned "
            "as deterministic acceleration."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        key: {"correction_m": value["correction_m"],
              "jump_rate_hz": value["jump_rate_hz"]}
        for key, value in report_groups.items()
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
