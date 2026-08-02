"""Fit a compact, deployable-state vision-fusion probability model.

The fast simulator previously sampled every visible 10 Hz frame from one
constant probability.  Real failures occur in correlated, viewpoint-specific
droughts.  This fitter uses only information available to the deployed actor
plus landmark age and the previous fusion outcome, and emits a small JSON
logistic model that the batched simulator can evaluate cheaply.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


FEATURE_NAMES = (
    *(f"distance_{index}" for index in range(3)),
    *(f"camera_depth_{index}" for index in range(3)),
    *(f"horizontal_ratio_{index}" for index in range(3)),
    *(f"vertical_ratio_{index}" for index in range(3)),
    "speed_mps",
    *(f"gate_{index}" for index in range(5)),
    "landmark_age_s",
    "position_sigma_m",
    "previous_fix",
)


def features(
    observation: np.ndarray,
    landmark_age_s: np.ndarray,
    previous_fix: np.ndarray,
) -> np.ndarray:
    rel = observation[:, :9].reshape(-1, 3, 3).astype(np.float64) * 10.0
    cam_pitch = np.deg2rad(20.0)
    cos_pitch, sin_pitch = np.cos(cam_pitch), np.sin(cam_pitch)
    depth = cos_pitch * rel[:, :, 0] - sin_pitch * rel[:, :, 2]
    camera_right = rel[:, :, 1]
    camera_down = sin_pitch * rel[:, :, 0] + cos_pitch * rel[:, :, 2]
    safe_depth = np.maximum(depth, 0.25)
    horizontal = np.abs(camera_right) / (
        safe_depth * np.tan(np.deg2rad(43.0))
    )
    vertical = np.abs(camera_down) / (
        safe_depth * np.tan(np.deg2rad(27.0))
    )
    distance = np.linalg.norm(rel, axis=2)
    speed = np.linalg.norm(observation[:, 18:21] * 10.0, axis=1)
    gate = observation[:, 34:39]
    sigma = observation[:, 51] * 0.5
    return np.column_stack([
        np.clip(distance / 30.0, 0.0, 3.0),
        np.clip(depth / 30.0, -1.0, 3.0),
        np.clip(horizontal, 0.0, 4.0),
        np.clip(vertical, 0.0, 4.0),
        np.clip(speed / 12.0, 0.0, 2.0),
        gate,
        np.clip(landmark_age_s / 2.0, 0.0, 3.0),
        np.clip(sigma / 0.5, 0.0, 4.0),
        previous_fix,
    ]).astype(np.float32)


def load_run(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
    by_episode: dict[int, list[dict]] = {}
    for row in step_rows:
        by_episode.setdefault(int(row["episode"]), []).append(row)
    all_x, all_y, all_episode = [], [], []
    for episode, rows in sorted(by_episode.items()):
        archive = run / f"episode_{episode:04d}.npz"
        if not archive.is_file():
            continue
        payload = np.load(archive)
        obs = np.asarray(payload["observation"], np.float32)
        rows = sorted(rows, key=lambda row: int(row["step"]))
        count = min(len(obs), len(rows))
        obs, rows = obs[:count], rows[:count]
        previous = 0.0
        for start in range(0, count, 3):
            window = rows[start:min(start + 3, count)]
            label = float(any(int(row.get("corners_fused", 0)) > 0 for row in window))
            age = float(rows[start].get("visual_age_s", 0.0))
            all_x.append(features(
                obs[start:start + 1],
                np.asarray([age], np.float32),
                np.asarray([previous], np.float32),
            )[0])
            all_y.append(label)
            all_episode.append(episode)
            previous = label
    return (
        np.asarray(all_x, np.float32),
        np.asarray(all_y, np.float32),
        np.asarray(all_episode, np.int32),
    )


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    steps: int = 2500,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-3)
    xn = torch.as_tensor((x - mean) / std, dtype=torch.float32)
    target = torch.as_tensor(y[:, None], dtype=torch.float32)
    torch.manual_seed(seed)
    weight = torch.zeros((1, x.shape[1]), requires_grad=True)
    bias = torch.tensor([
        float(np.log((y.mean() + 1e-3) / (1.0 - y.mean() + 1e-3)))
    ], requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.03)
    for _ in range(steps):
        logits = xn @ weight.t() + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target
        ) + 2e-3 * weight.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return (
        weight.detach().numpy()[0], float(bias.detach()), mean, std
    )


def probabilities(
    x: np.ndarray,
    weight: np.ndarray,
    bias: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    logits = ((x - mean) / std) @ weight + bias
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))


def auc(y: np.ndarray, probability: np.ndarray) -> float:
    positive = probability[y > 0.5]
    negative = probability[y <= 0.5]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(np.mean(
        (positive[:, None] > negative[None]).astype(float)
        + 0.5 * (positive[:, None] == negative[None])
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260898)
    args = parser.parse_args()
    loaded = [load_run(run) for run in args.run]
    x = np.concatenate([row[0] for row in loaded])
    y = np.concatenate([row[1] for row in loaded])
    # Keep episode identities unique across sessions for leave-one-episode-out
    # reporting.
    episode = np.concatenate([
        row[2] + index * 10000 for index, row in enumerate(loaded)
    ])
    folds = []
    for held_out in np.unique(episode):
        train = episode != held_out
        test = ~train
        weight, bias, mean, std = fit_logistic(
            x[train], y[train], seed=args.seed + int(held_out)
        )
        p = probabilities(x[test], weight, bias, mean, std)
        folds.append({
            "episode": int(held_out),
            "rows": int(test.sum()),
            "positive_rate": float(y[test].mean()),
            "brier": float(np.mean((p - y[test]) ** 2)),
            "auc": auc(y[test], p),
        })
    weight, bias, mean, std = fit_logistic(x, y, seed=args.seed)
    p = probabilities(x, weight, bias, mean, std)
    payload = {
        "type": "vq2_vision_fusion_logistic_v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weight": weight.tolist(),
        "bias": bias,
        "training_rows": int(len(y)),
        "training_positive_rate": float(y.mean()),
        "training_brier": float(np.mean((p - y) ** 2)),
        "training_auc": auc(y, p),
        "constant_brier": float(np.mean((y.mean() - y) ** 2)),
        "leave_one_episode_out": folds,
        "source_runs": [str(run.resolve()) for run in args.run],
        "seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
