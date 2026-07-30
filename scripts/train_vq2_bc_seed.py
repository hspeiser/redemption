"""Behavior-clone the clean VQ2 lap and fit critics to its returns."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.sac import GaussianActor, TwinCritic  # noqa: E402
from aigp.rl.vq2_features import ACT_DIM, OBS_DIM  # noqa: E402


def discounted_returns(
    reward: np.ndarray, done: np.ndarray, gamma: float
) -> np.ndarray:
    returns = np.zeros_like(reward, dtype=np.float32)
    carry = 0.0
    for index in range(len(reward) - 1, -1, -1):
        carry = float(reward[index]) + gamma * carry * (
            1.0 - float(done[index])
        )
        returns[index] = carry
    return returns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO / "data" / "vq2_sac_clean_demo.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data" / "models" / "vq2_bc_seed.pt",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument(
        "--state-noise",
        type=float,
        default=0.0,
        help=(
            "Optional normalized observation noise. Keep at zero for DAgger "
            "data because unlabeled perturbations teach incorrect recovery."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.dataset, allow_pickle=False)
    observation = np.asarray(data["observation"], np.float32)
    action = np.asarray(data["action"], np.float32)
    reward = np.asarray(data["reward"], np.float32)
    done = np.asarray(data["done"], np.float32)
    gate = np.asarray(data["gate_index"], np.int64)
    if observation.shape[1] != OBS_DIM or action.shape[1] != ACT_DIM:
        raise ValueError(
            f"dataset shape {observation.shape}/{action.shape}, "
            f"expected (*,{OBS_DIM})/(*,{ACT_DIM})"
        )

    # Deterministic interleaved validation leaves examples from every gate in
    # both sets, unlike a tail split that would validate only gates 14-16.
    # The first grounded/ramp samples are unique episode-start states.  The
    # old interleaved split held out index 0, so the actor matched the rest of
    # gate 0 but emitted 0.47 thrust instead of the demonstrated 0.03 on the
    # very first live command.  Startup is safety-critical training data.
    validation = (
        ((np.arange(len(observation)) % 8) == 0)
        & (np.arange(len(observation)) >= 32)
    )
    training = ~validation
    observation_mean = observation[training].mean(axis=0)
    observation_std = observation[training].std(axis=0)
    observation_std = np.maximum(observation_std, 0.05)
    normalized = (observation - observation_mean) / observation_std
    returns = discounted_returns(reward, done, args.gamma)
    return_scale = max(float(np.std(returns[training])), 1.0)
    normalized_returns = returns / return_scale

    device = torch.device(args.device)
    obs_t = torch.from_numpy(normalized).to(device)
    action_t = torch.from_numpy(action).to(device)
    return_t = torch.from_numpy(normalized_returns[:, None]).to(device)
    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    critic = TwinCritic(OBS_DIM, ACT_DIM).to(device)
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.actor_lr, weight_decay=1e-5
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr, weight_decay=1e-5
    )

    train_indices = np.flatnonzero(training)
    validation_indices = np.flatnonzero(validation)
    gate_count = np.bincount(gate[training], minlength=17)
    sample_weight = 1.0 / np.maximum(gate_count[gate[train_indices]], 1)
    sample_weight[np.searchsorted(train_indices, np.arange(20))] *= 20.0
    sample_weight /= sample_weight.sum()
    action_weight = torch.tensor(
        [1.5, 1.2, 1.0, 1.4], device=device
    )

    best_score = np.inf
    best_actor = None
    best_critic = None
    for step in range(1, args.steps + 1):
        batch_np = np.random.choice(
            train_indices,
            size=min(args.batch, len(train_indices)),
            replace=True,
            p=sample_weight,
        )
        batch = torch.from_numpy(batch_np).to(device)
        batch_observation = obs_t[batch]
        noisy_observation = batch_observation.clone()
        if args.state_noise > 0.0:
            # Never corrupt the 17-way gate identity or previous-action slots.
            noisy_observation[:, :30] += (
                args.state_noise
                * torch.randn_like(noisy_observation[:, :30])
            )
        prediction = actor.deterministic(noisy_observation)
        actor_loss = (
            F.smooth_l1_loss(
                prediction, action_t[batch], reduction="none", beta=0.04
            )
            * action_weight
        ).mean()
        _, log_std = actor.distribution(noisy_observation)
        uncertainty_loss = 0.002 * (log_std + 3.5).square().mean()
        actor_optimizer.zero_grad(set_to_none=True)
        (actor_loss + uncertainty_loss).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        actor_optimizer.step()

        q1, q2 = critic(batch_observation, action_t[batch])
        critic_loss = F.mse_loss(q1, return_t[batch]) + \
            F.mse_loss(q2, return_t[batch])
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        critic_optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            with torch.no_grad():
                val_index = torch.from_numpy(validation_indices).to(device)
                val_prediction = actor.deterministic(obs_t[val_index])
                error = (
                    val_prediction - action_t[val_index]
                ).abs().cpu().numpy()
                q1v, q2v = critic(
                    obs_t[val_index], action_t[val_index]
                )
                q_error = 0.5 * (
                    (q1v - return_t[val_index]).abs().mean()
                    + (q2v - return_t[val_index]).abs().mean()
                )
                score = float(np.mean(error * np.array(
                    [1.5, 1.2, 1.0, 1.4]
                )))
            if score < best_score:
                best_score = score
                best_actor = copy.deepcopy(actor.state_dict())
                best_critic = copy.deepcopy(critic.state_dict())
            print(
                f"step {step:5d} actor {actor_loss.item():.5f} "
                f"critic {critic_loss.item():.5f} val_mae "
                f"{np.round(error.mean(axis=0), 4).tolist()} "
                f"q_mae {float(q_error) * return_scale:.2f} "
                f"best {best_score:.5f}",
                flush=True,
            )

    actor.load_state_dict(best_actor)
    critic.load_state_dict(best_critic)
    with torch.no_grad():
        all_prediction = actor.deterministic(obs_t).cpu().numpy()
    startup_mae = np.mean(
        np.abs(all_prediction[:20] - action[:20]), axis=0
    )
    per_gate = {}
    for gate_index in range(17):
        mask = validation & (gate == gate_index)
        if not mask.any():
            continue
        per_gate[str(gate_index)] = {
            "samples": int(mask.sum()),
            "mae": np.mean(
                np.abs(all_prediction[mask] - action[mask]), axis=0
            ).round(6).tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "kind": "vq2_bc_sac_seed",
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": copy.deepcopy(critic.state_dict()),
        "observation_dim": OBS_DIM,
        "action_dim": ACT_DIM,
        "observation_mean": observation_mean,
        "observation_std": observation_std,
        "return_scale": return_scale,
        "gamma": args.gamma,
        "dataset": str(args.dataset),
        "validation_score": best_score,
        "per_gate_validation": per_gate,
        "startup_mae": startup_mae.tolist(),
    }
    torch.save(checkpoint, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps({
        "checkpoint": str(args.output),
        "dataset": str(args.dataset),
        "transitions": int(len(observation)),
        "validation_score": best_score,
        "startup_mae": startup_mae.tolist(),
        "per_gate_validation": per_gate,
    }, indent=2))
    print(f"saved {args.output}")
    print(f"report {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
