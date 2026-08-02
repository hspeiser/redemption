"""Distill successful live VQ2 POC flights into the direct teacher actor.

The trainer normally blends a geometric/demo reference with a frozen teacher
actor.  This script replaces that teacher with a policy fitted to *live final
wire actions* from successful gates-0..N episodes.  Entire sessions, not
individual frames, are held out for validation.  A designated fastest episode
can be up-weighted while the remaining successes provide robustness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.sac import GaussianActor  # noqa: E402
from aigp.rl.vq2_features import ACT_DIM, OBS_DIM  # noqa: E402


@dataclass
class Flight:
    path: Path
    session: Path
    observation: np.ndarray
    action: np.ndarray
    lap_s: float
    weight: float
    dagger: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--through-gate", type=int, default=4)
    parser.add_argument("--fast-episode", type=Path, required=True)
    parser.add_argument("--fast-boost", type=float, default=12.0)
    parser.add_argument("--dagger-boost", type=float, default=4.0)
    parser.add_argument(
        "--focus-session", type=Path, default=None,
        help="Optional newly collected DAgger session to up-weight.",
    )
    parser.add_argument("--focus-boost", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args()


def discover(args: argparse.Namespace) -> list[Flight]:
    fast = args.fast_episode.resolve()
    focus_session = (
        args.focus_session.resolve()
        if getattr(args, "focus_session", None) is not None else None
    )
    flights: list[Flight] = []
    for journal in args.runs_root.rglob("episodes.jsonl"):
        config_path = journal.parent / "config.json"
        if not config_path.exists():
            continue
        try:
            config = json.loads(config_path.read_text()).get("args", {})
        except (OSError, json.JSONDecodeError):
            continue
        if int(config.get("poc_stop_after_gate", -1)) != args.through_gate:
            continue
        for line in journal.read_text().splitlines():
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not summary.get("timing_healthy", True):
                continue
            episode_path = journal.parent / (
                f"episode_{int(summary['episode']):04d}.npz"
            )
            if not episode_path.exists():
                continue
            try:
                episode = np.load(episode_path, allow_pickle=False)
                observation = np.asarray(
                    episode["observation"], np.float32
                )
                has_teacher = "teacher_action" in episode.files
                completed = bool(summary.get("poc_completed", False))
                teacher_source = (
                    str(np.asarray(episode["teacher_action_source"]).item())
                    if "teacher_action_source" in episode.files else ""
                )
                legacy_impulse_teacher = bool(
                    has_teacher
                    and float(config.get(
                        "domain_impulse_probability", 0.0
                    )) > 0.0
                    and float(config.get("teacher_blend", 1.0)) == 1.0
                )
                dagger = bool(
                    has_teacher
                    and (
                        legacy_impulse_teacher
                        or (
                            teacher_source == "protected_reference"
                            and not completed
                        )
                    )
                )
                if not summary.get("poc_completed", False) and not dagger:
                    continue
                action = np.asarray(
                    episode["teacher_action"]
                    if dagger else episode["wire_action"],
                    np.float32,
                )
                gate = np.asarray(episode["gate_index"], np.int16)
            except (OSError, KeyError, ValueError):
                continue
            selected = gate <= args.through_gate
            observation = observation[selected]
            action = action[selected]
            if (
                len(observation) < 30
                or observation.shape[1] != OBS_DIM
                or action.shape[1] != ACT_DIM
                or (
                    not dagger
                    and int(gate.max(initial=-1)) < args.through_gate
                )
            ):
                continue
            lap_s = (
                float(summary.get("steps", len(observation))) / 30.0
                if completed else 9.55
            )
            # Mildly favor faster successes without allowing the many robust
            # baseline laps to disappear. The explicit fast episode receives
            # the stronger preference below.
            weight = float(np.clip(math.exp(2.0 * (9.55 - lap_s)), 0.5, 3.0))
            if episode_path.resolve() == fast:
                weight *= max(float(args.fast_boost), 1.0)
            if dagger:
                weight *= max(float(args.dagger_boost), 1.0)
            if focus_session is not None and journal.parent.resolve() == focus_session:
                weight *= max(float(args.focus_boost), 1.0)
            flights.append(Flight(
                episode_path, journal.parent, observation, action,
                lap_s, weight, dagger,
            ))
    if not any(f.path.resolve() == fast for f in flights):
        raise RuntimeError(f"fast episode was not discovered as a success: {fast}")
    return flights


def stack(flights: list[Flight], mean: np.ndarray, std: np.ndarray):
    observation = np.concatenate([f.observation for f in flights])
    action = np.concatenate([f.action for f in flights])
    weight = np.concatenate([
        np.full(len(f.observation), f.weight, np.float32) for f in flights
    ])
    normalized = (observation - mean) / std
    return (
        torch.from_numpy(normalized.astype(np.float32)),
        torch.from_numpy(action.astype(np.float32)),
        torch.from_numpy(weight.astype(np.float32)),
    )


@torch.no_grad()
def metrics(actor, observation, action) -> dict[str, float]:
    predicted = actor.deterministic(observation)
    error = (predicted - action).square()
    return {
        "mse": float(error.mean()),
        "mae": float((predicted - action).abs().mean()),
        "lateral_mse": float(error[:, 0].mean()),
        "pitch_mse": float(error[:, 1].mean()),
        "yaw_mse": float(error[:, 2].mean()),
        "thrust_mse": float(error[:, 3].mean()),
    }


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    mean = np.asarray(payload["observation_mean"], np.float32)
    std = np.asarray(payload["observation_std"], np.float32)
    flights = discover(args)
    sessions = sorted({f.session for f in flights})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(sessions)
    n_validation = max(
        1, min(len(sessions) - 1, round(
            len(sessions) * args.validation_fraction
        ))
    )
    validation_sessions = set(sessions[:n_validation])
    fast = args.fast_episode.resolve()
    # The designated speed target must train the actor, never leak into val.
    validation_sessions.discard(fast.parent)
    validation_sessions.difference_update(
        f.session for f in flights if f.dagger
    )
    train_flights = [f for f in flights if f.session not in validation_sessions]
    validation_flights = [f for f in flights if f.session in validation_sessions]
    if not validation_flights:
        validation_flights = train_flights[-1:]
        train_flights = train_flights[:-1]

    train_observation, train_action, train_weight = stack(
        train_flights, mean, std
    )
    val_observation, val_action, _ = stack(validation_flights, mean, std)
    fast_flight = next(f for f in flights if f.path.resolve() == fast)
    fast_observation, fast_action, _ = stack([fast_flight], mean, std)

    actor = GaussianActor(OBS_DIM, ACT_DIM)
    actor.load_state_dict(payload.get("teacher_actor", payload["actor"]))
    actor.train()
    before = {
        "train": metrics(actor, train_observation, train_action),
        "validation": metrics(actor, val_observation, val_action),
        "fast": metrics(actor, fast_observation, fast_action),
    }
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    count = len(train_observation)
    batch_size = min(max(1, args.batch_size), count)
    probability = train_weight / train_weight.sum()
    for _ in range(max(0, args.steps)):
        indices = torch.multinomial(
            probability, batch_size, replacement=True
        )
        predicted = actor.deterministic(train_observation[indices])
        per_row = F.mse_loss(
            predicted, train_action[indices], reduction="none"
        ).mean(dim=1)
        loss = per_row.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()
    actor.eval()
    after = {
        "train": metrics(actor, train_observation, train_action),
        "validation": metrics(actor, val_observation, val_action),
        "fast": metrics(actor, fast_observation, fast_action),
    }
    payload["teacher_actor"] = actor.state_dict()
    payload["poc_success_actor"] = {
        "through_gate": int(args.through_gate),
        "flights": len(flights),
        "train_flights": len(train_flights),
        "validation_flights": len(validation_flights),
        "train_rows": len(train_observation),
        "validation_rows": len(val_observation),
        "fast_episode": str(fast),
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
    # Optimizer state belongs to the residual actor and remains valid. The
    # frozen direct teacher has no optimizer in the live trainer.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(payload["poc_success_actor"], indent=2))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
