"""Sequence-distill live VQ2 POC flights into a recurrent actor.

The default target remains the historical direct/full-action teacher.  With
``--target residual`` the network instead learns the normalized correction
that was applied around the protected geometric controller.  Failed DAgger
rollouts receive a zero-residual target, preserving the controller's recovery
behavior instead of imitating the failed policy action.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.sac import RecurrentActor  # noqa: E402
from aigp.rl.vq2_features import ACT_DIM, OBS_DIM  # noqa: E402
from scripts.fit_vq2_poc_success_actor import discover  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--through-gate", type=int, default=4)
    parser.add_argument("--fast-episode", type=Path, required=True)
    parser.add_argument("--fast-boost", type=float, default=100.0)
    parser.add_argument("--dagger-boost", type=float, default=8.0)
    parser.add_argument("--focus-session", type=Path, default=None)
    parser.add_argument("--focus-boost", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("direct", "residual"), default="direct"
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-flights", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def padded_batch(flights, mean, std, device):
    lengths = [len(f.observation) for f in flights]
    maximum = max(lengths)
    observation = np.zeros((len(flights), maximum, OBS_DIM), np.float32)
    action = np.zeros((len(flights), maximum, ACT_DIM), np.float32)
    mask = np.zeros((len(flights), maximum), np.float32)
    for index, flight in enumerate(flights):
        count = lengths[index]
        observation[index, :count] = (flight.observation - mean) / std
        action[index, :count] = flight.action
        mask[index, :count] = 1.0
    return (
        torch.from_numpy(observation).to(device),
        torch.from_numpy(action).to(device),
        torch.from_numpy(mask).to(device),
    )


@torch.no_grad()
def metrics(actor, flights, mean, std, device):
    squared = []
    absolute = []
    per_axis = []
    for flight in flights:
        observation = torch.from_numpy(
            ((flight.observation - mean) / std).astype(np.float32)
        ).to(device)[None]
        target = torch.from_numpy(flight.action).to(device)[None]
        predicted, _ = actor.sequence(observation)
        error = predicted - target
        squared.append(error.square().reshape(-1).cpu())
        absolute.append(error.abs().reshape(-1).cpu())
        per_axis.append(error.square()[0].cpu())
    axis = torch.cat(per_axis)
    return {
        "mse": float(torch.cat(squared).mean()),
        "mae": float(torch.cat(absolute).mean()),
        "axis_mse": [float(value) for value in axis.mean(dim=0)],
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    mean = np.asarray(payload["observation_mean"], np.float32)
    std = np.asarray(payload["observation_std"], np.float32)
    discovery_args = SimpleNamespace(
        runs_root=args.runs_root,
        through_gate=args.through_gate,
        fast_episode=args.fast_episode,
        fast_boost=args.fast_boost,
        dagger_boost=args.dagger_boost,
        focus_session=args.focus_session,
        focus_boost=args.focus_boost,
    )
    flights = discover(discovery_args)
    if args.target == "residual":
        for flight in flights:
            episode = np.load(flight.path, allow_pickle=False)
            gate = np.asarray(episode["gate_index"], np.int16)
            selected = gate <= args.through_gate
            if flight.dagger:
                flight.action = np.zeros(
                    (int(selected.sum()), ACT_DIM), np.float32
                )
            else:
                flight.action = np.asarray(
                    episode["action"], np.float32
                )[selected]
    sessions = sorted({f.session for f in flights})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sessions)
    n_validation = max(
        1, min(len(sessions) - 1, round(
            len(sessions) * args.validation_fraction
        ))
    )
    validation_sessions = set(sessions[:n_validation])
    validation_sessions.discard(args.fast_episode.resolve().parent)
    validation_sessions.difference_update(
        f.session for f in flights if f.dagger
    )
    train = [f for f in flights if f.session not in validation_sessions]
    validation = [f for f in flights if f.session in validation_sessions]
    if not validation:
        validation = train[-1:]
        train = train[:-1]
    fast = next(
        f for f in flights
        if f.path.resolve() == args.fast_episode.resolve()
    )

    actor = RecurrentActor(
        OBS_DIM, ACT_DIM, hidden_dim=args.hidden_dim
    ).to(device)
    actor_key = (
        "recurrent_residual_actor"
        if args.target == "residual" else "recurrent_teacher_actor"
    )
    metadata_key = (
        "recurrent_residual"
        if args.target == "residual" else "recurrent_teacher"
    )
    if actor_key in payload:
        metadata = payload.get(metadata_key, {})
        if int(metadata.get("hidden_dim", args.hidden_dim)) == args.hidden_dim:
            actor.load_state_dict(payload[actor_key])
    before = {
        "validation": metrics(actor, validation, mean, std, device),
        "fast": metrics(actor, [fast], mean, std, device),
    }
    actor.train()
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    probability = np.asarray([
        max(f.weight, 1e-3) * np.sqrt(len(f.observation)) for f in train
    ], np.float64)
    probability /= probability.sum()
    batch_count = min(max(1, args.batch_flights), len(train))
    for _ in range(max(0, args.steps)):
        indices = rng.choice(
            len(train), size=batch_count, replace=True, p=probability
        )
        batch = [train[int(index)] for index in indices]
        observation, target, mask = padded_batch(
            batch, mean, std, device
        )
        predicted, _ = actor.sequence(observation)
        row_error = (predicted - target).square().mean(dim=-1)
        loss = (row_error * mask).sum() / mask.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()
    actor.eval()
    after = {
        "train": metrics(actor, train, mean, std, device),
        "validation": metrics(actor, validation, mean, std, device),
        "fast": metrics(actor, [fast], mean, std, device),
    }
    payload[actor_key] = {
        key: value.detach().cpu() for key, value in actor.state_dict().items()
    }
    payload[metadata_key] = {
        "target": args.target,
        "hidden_dim": int(args.hidden_dim),
        "through_gate": int(args.through_gate),
        "flights": len(flights),
        "train_flights": len(train),
        "validation_flights": len(validation),
        "fast_episode": str(args.fast_episode.resolve()),
        "fast_boost": float(args.fast_boost),
        "dagger_boost": float(args.dagger_boost),
        "dagger_flights": sum(f.dagger for f in flights),
        "focus_session": (
            str(args.focus_session.resolve())
            if args.focus_session is not None else None
        ),
        "focus_boost": float(args.focus_boost),
        "steps": int(args.steps),
        "before": before,
        "after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(payload[metadata_key], indent=2))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
