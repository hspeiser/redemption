"""Train a truth-state dynamics ensemble from VQ1 odometry captures.

This deliberately does not consume VQ2 EKF positions.  The ensemble corrects
only physical velocity, attitude, and body rates after one analytic step; the
fast simulator's localization model remains a separate observation process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.data import load_episode  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import (  # noqa: E402
    ResidualEnsemble,
    analytic_step,
    residual_features,
    rotation_log,
)
from aigp.rl.vq2_features import wire_command_to_action  # noqa: E402


def episode_rows(path: Path, hz: float) -> dict[str, np.ndarray] | None:
    episode = load_episode(path, hz=hz, require_odometry=True)
    if episode is None or len(episode.t) < 32:
        return None
    speed = np.linalg.norm(episode.vel_world, axis=1)
    if float(speed.max()) < 3.0:
        return None
    rotation = Rotation.from_quat(episode.quat_wb).as_matrix()
    action = np.stack(
        [wire_command_to_action(row) for row in episode.cmd], axis=0
    )
    # State k plus command k predicts state k+1.  Reject reset/impact jumps;
    # those are terminal/outcome-model data, not smooth plant dynamics.
    position_step_error = np.linalg.norm(
        episode.pos[1:] - episode.pos[:-1]
        - episode.vel_world[:-1] / hz,
        axis=1,
    )
    valid = (
        (position_step_error < 0.25)
        & (speed[:-1] < 40.0)
        & (speed[1:] < 40.0)
        & (np.linalg.norm(episode.rates_body[:-1], axis=1) < 12.0)
        & np.isfinite(episode.pos[:-1]).all(1)
        & np.isfinite(episode.vel_world[:-1]).all(1)
        & np.isfinite(rotation[:-1]).all((1, 2))
        & np.isfinite(action[:-1]).all(1)
    )
    row = np.nonzero(valid)[0]
    if len(row) < 30:
        return None
    previous = np.vstack([action[:1], action[:-1]])
    return {
        "position": episode.pos[row].astype(np.float32),
        "velocity": episode.vel_world[row].astype(np.float32),
        "rotation": rotation[row].astype(np.float32),
        "rates": episode.rates_body[row].astype(np.float32),
        "action": action[row].astype(np.float32),
        "previous_action": previous[row].astype(np.float32),
        "next_position": episode.pos[row + 1].astype(np.float32),
        "next_velocity": episode.vel_world[row + 1].astype(np.float32),
        "next_rotation": rotation[row + 1].astype(np.float32),
        "next_rates": episode.rates_body[row + 1].astype(np.float32),
        "step": row.astype(np.int32),
    }


def concatenate(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([item[key] for item in items]) for key in items[0]}


@torch.no_grad()
def make_xy(data: dict[str, np.ndarray], model: SurrogateModel, device: str):
    tensor = {
        key: torch.as_tensor(value, device=device)
        for key, value in data.items()
        if key != "step"
    }
    p1, v1, r1, w1 = analytic_step(
        tensor["position"], tensor["velocity"], tensor["rotation"],
        tensor["rates"], tensor["action"], model,
    )
    x = residual_features(
        tensor["velocity"], tensor["rotation"], tensor["rates"],
        tensor["action"], tensor["previous_action"],
    )
    dv = torch.einsum(
        "nij,nj->ni", tensor["rotation"].transpose(1, 2),
        tensor["next_velocity"] - v1,
    )
    dr = rotation_log(r1.transpose(1, 2) @ tensor["next_rotation"])
    dw = tensor["next_rates"] - w1
    y = torch.cat([dv, dr, dw], dim=1)
    keep = (
        torch.isfinite(y).all(1)
        & (torch.linalg.norm(dv, dim=1) < 3.0)
        & (torch.linalg.norm(dr, dim=1) < 0.5)
        & (torch.linalg.norm(dw, dim=1) < 6.0)
    )
    return x[keep], y[keep], keep.cpu().numpy(), tensor


def train_member(
    member: torch.nn.Module,
    ensemble: ResidualEnsemble,
    x: torch.Tensor,
    y: torch.Tensor,
    episode: torch.Tensor,
    xv: torch.Tensor,
    yv: torch.Tensor,
    *, seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    unique = torch.unique(episode)
    sampled = unique[torch.randint(
        len(unique), (len(unique),), generator=generator, device=x.device
    )]
    rows = torch.nonzero(torch.isin(episode, sampled)).squeeze(1)
    xn = (x - ensemble.x_mean) / ensemble.x_std
    yn = (y - ensemble.y_mean) / ensemble.y_std
    xvn = (xv - ensemble.x_mean) / ensemble.x_std
    yvn = (yv - ensemble.y_mean) / ensemble.y_std
    optimizer = torch.optim.AdamW(member.parameters(), lr=lr, weight_decay=1e-5)
    best = None
    stale = 0
    for epoch in range(epochs):
        permutation = rows[torch.randperm(
            len(rows), generator=generator, device=x.device
        )]
        member.train()
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            mean, log_std = member(xn[batch])
            error = yn[batch] - mean
            loss = (
                0.5 * error.square() * torch.exp(-2.0 * log_std) + log_std
                + 0.05 * error.square()
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), 10.0)
            optimizer.step()
        member.eval()
        with torch.no_grad():
            predicted, _ = member(xvn)
            score = float((predicted - yvn).square().mean())
        if best is None or score < best[0] - 1e-6:
            best = (score, {k: v.detach().cpu().clone()
                            for k, v in member.state_dict().items()}, epoch)
            stale = 0
        else:
            stale += 1
            if stale >= 15:
                break
    member.load_state_dict(best[1])
    return {"best_epoch": best[2], "validation_normalized_mse": best[0]}


@torch.no_grad()
def one_step_report(
    ensemble: ResidualEnsemble,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    means, _ = ensemble(x)
    prediction = means.mean(0)
    error = prediction - y
    groups = {"velocity_mps": error[:, :3], "attitude_rad": error[:, 3:6],
              "rates_rps": error[:, 6:9]}
    report = {}
    for name, value in groups.items():
        norm = torch.linalg.norm(value, dim=1)
        report[name] = {
            "median": float(norm.median()),
            "p90": float(norm.quantile(0.9)),
            "p99": float(norm.quantile(0.99)),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    episodes = []
    for path in sorted(args.captures.glob("rc_*")):
        try:
            rows = episode_rows(path, 30.0)
        except Exception as error:
            print(f"skip {path.name}: {error!r}", flush=True)
            continue
        if rows is not None:
            episodes.append((path.name, rows))
    order = rng.permutation(len(episodes))
    n_test = max(5, round(0.15 * len(order)))
    n_validation = max(5, round(0.15 * len(order)))
    split_index = {
        "test": order[:n_test],
        "validation": order[n_test:n_test + n_validation],
        "train": order[n_test + n_validation:],
    }
    split = {
        name: concatenate([episodes[i][1] for i in indices])
        for name, indices in split_index.items()
    }
    split_names = {
        name: [episodes[i][0] for i in indices]
        for name, indices in split_index.items()
    }
    model = SurrogateModel.load(args.model)
    x, y, keep, _ = make_xy(split["train"], model, args.device)
    xv, yv, _keepv, _ = make_xy(split["validation"], model, args.device)
    xt, yt, _keept, _ = make_xy(split["test"], model, args.device)
    episode_id = np.concatenate([
        np.full(len(item["step"]), local, np.int64)
        for local, i in enumerate(split_index["train"])
        for item in [episodes[i][1]]
    ])[keep]
    episode_id = torch.as_tensor(episode_id, device=args.device)
    ensemble = ResidualEnsemble(
        members=args.members, input_dim=x.shape[1], output_dim=y.shape[1]
    ).to(args.device)
    ensemble.set_normalization(x, y)
    training = []
    for index, member in enumerate(ensemble.members):
        row = train_member(
            member, ensemble, x, y, episode_id, xv, yv,
            seed=args.seed + 1009 * index, epochs=args.epochs,
            batch_size=args.batch_size, lr=3e-4,
        )
        training.append(row)
        print(f"member {index}: {row}", flush=True)
    report = {
        "captures": str(args.captures),
        "base_model": str(args.model),
        "episodes": split_names,
        "rows": {name: len(data["step"]) for name, data in split.items()},
        "training": training,
        "validation_one_step": one_step_report(ensemble, xv, yv),
        "test_one_step": one_step_report(ensemble, xt, yt),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(args.out, report)
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test_one_step"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
