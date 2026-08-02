"""Train a gate-scoped recurrent residual actor with real-data BC + IQL.

The critic consumes every healthy real transition.  Actor regression consumes
only successful focus-gate actions, weighted by IQL advantage.  The resulting
GRU is stored inside a copy of the live seed checkpoint and is explicitly
routed to the requested gate, leaving all existing PPO actors untouched.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.sac import RecurrentActor, TwinCritic, mlp  # noqa: E402
from aigp.rl.vq2_features import ACT_DIM, OBS_DIM  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ValueNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = mlp(OBS_DIM, (256, 256, 128), 1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def expectile_loss(error: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(error > 0, expectile, 1.0 - expectile)
    return (weight * error.square()).mean()


def load_split(path: Path, mean: np.ndarray, std: np.ndarray) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    result = {name: np.asarray(data[name]) for name in data.files}
    result["normalized_observation"] = np.clip(
        (np.asarray(result["observation"], np.float32) - mean) / std,
        -8.0, 8.0,
    ).astype(np.float32)
    result["normalized_next_observation"] = np.clip(
        (np.asarray(result["next_observation"], np.float32) - mean) / std,
        -8.0, 8.0,
    ).astype(np.float32)
    return result


def actor_metrics(
    actor: RecurrentActor,
    data: dict[str, np.ndarray],
    device: torch.device,
) -> dict:
    errors = []
    axis = []
    actor.eval()
    for sequence in np.unique(data["sequence_id"]):
        row = np.flatnonzero(data["sequence_id"] == sequence)
        selected = data["actor_weight"][row] > 0
        if not np.any(selected):
            continue
        observation = torch.from_numpy(
            data["normalized_observation"][row]
        ).to(device)[None]
        with torch.no_grad():
            prediction, _ = actor.sequence(observation)
        target = torch.from_numpy(
            np.asarray(data["action"][row], np.float32)
        ).to(device)[None]
        error = (prediction[0, selected] - target[0, selected]).abs().cpu()
        errors.append(error.reshape(-1))
        axis.append(error)
    if not errors:
        return {"rows": 0, "mae": None, "axis_mae": None}
    per_axis = torch.cat(axis)
    return {
        "rows": int(len(per_axis)),
        "mae": float(torch.cat(errors).mean()),
        "axis_mae": [float(value) for value in per_axis.mean(dim=0)],
    }


def sequence_batch(
    data: dict[str, np.ndarray],
    sequence_ids: np.ndarray,
    row_weight: np.ndarray,
    device: torch.device,
):
    groups = [np.flatnonzero(data["sequence_id"] == value) for value in sequence_ids]
    maximum = max(len(row) for row in groups)
    count = len(groups)
    observation = np.zeros((count, maximum, OBS_DIM), np.float32)
    action = np.zeros((count, maximum, ACT_DIM), np.float32)
    weight = np.zeros((count, maximum), np.float32)
    for index, row in enumerate(groups):
        length = len(row)
        observation[index, :length] = data["normalized_observation"][row]
        action[index, :length] = data["action"][row]
        weight[index, :length] = row_weight[row]
    return (
        torch.from_numpy(observation).to(device),
        torch.from_numpy(action).to(device),
        torch.from_numpy(weight).to(device),
    )


def train_actor(
    actor: RecurrentActor,
    data: dict[str, np.ndarray],
    row_weight: np.ndarray,
    *,
    steps: int,
    batch_sequences: int,
    lr: float,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    eligible_sequences = np.asarray([
        value for value in np.unique(data["sequence_id"])
        if np.any(row_weight[data["sequence_id"] == value] > 0)
    ], np.int32)
    if not len(eligible_sequences):
        raise RuntimeError("no actor-eligible sequences")
    optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    actor.train()
    for _ in range(max(0, steps)):
        selected = rng.choice(
            eligible_sequences,
            size=min(batch_sequences, len(eligible_sequences)),
            replace=True,
        )
        observation, action, weight = sequence_batch(
            data, selected, row_weight, device,
        )
        prediction, _ = actor.sequence(observation)
        error = F.smooth_l1_loss(
            prediction, action, reduction="none", beta=0.03
        ).mean(dim=-1)
        loss = (error * weight).sum() / weight.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()
    actor.eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed-checkpoint", type=Path, required=True)
    parser.add_argument("--normalization-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-gate", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--critic-steps", type=int, default=20000)
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--awr-steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--batch-sequences", type=int, default=16)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=2e-4)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--max-awr-weight", type=float, default=20.0)
    parser.add_argument("--min-nonzero-actor-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=5005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    normalization = torch.load(
        args.normalization_checkpoint, map_location="cpu", weights_only=False
    )
    mean = np.asarray(normalization["obs_mean"], np.float32)
    variance = np.asarray(normalization["obs_var"], np.float32)
    std = np.sqrt(variance + 1e-6).astype(np.float32)
    train = load_split(args.dataset / "train.npz", mean, std)
    validation_path = args.dataset / "validation.npz"
    validation = load_split(
        validation_path if validation_path.exists() else args.dataset / "train.npz",
        mean, std,
    )

    critic = TwinCritic(OBS_DIM, ACT_DIM).to(device)
    value = ValueNetwork().to(device)
    target_critic = copy.deepcopy(critic).to(device).eval()
    for parameter in target_critic.parameters():
        parameter.requires_grad_(False)
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr, weight_decay=1e-6
    )
    value_optimizer = torch.optim.AdamW(
        value.parameters(), lr=args.critic_lr, weight_decay=1e-6
    )
    index = np.flatnonzero(train["critic_weight"] > 0)
    if not len(index):
        raise RuntimeError("no critic-eligible rows")
    reward_scale = 100.0
    last_losses = {}
    for step in range(max(0, args.critic_steps)):
        row = rng.choice(index, size=min(args.batch, len(index)), replace=True)
        observation = torch.from_numpy(train["normalized_observation"][row]).to(device)
        next_observation = torch.from_numpy(train["normalized_next_observation"][row]).to(device)
        action = torch.from_numpy(np.asarray(train["action"][row], np.float32)).to(device)
        reward = torch.from_numpy(np.asarray(train["reward"][row], np.float32)).to(device)[:, None] / reward_scale
        done = torch.from_numpy(np.asarray(train["done"][row], np.float32)).to(device)[:, None]
        discount = torch.from_numpy(np.asarray(train["discount"][row], np.float32)).to(device)[:, None]
        with torch.no_grad():
            target = reward + discount * (1.0 - done) * value(next_observation)
        q1, q2 = critic(observation, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        critic_optimizer.step()

        with torch.no_grad():
            tq1, tq2 = target_critic(observation, action)
            q = torch.minimum(tq1, tq2)
        estimate = value(observation)
        value_loss = expectile_loss(q - estimate, args.expectile)
        value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
        value_optimizer.step()
        with torch.no_grad():
            tau = 0.005
            for target_parameter, parameter in zip(
                target_critic.parameters(), critic.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
        if step % 500 == 0 or step + 1 == args.critic_steps:
            last_losses = {
                "step": step + 1,
                "critic_loss": float(critic_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
            }
            print(json.dumps(last_losses), flush=True)

    actor = RecurrentActor(OBS_DIM, ACT_DIM, hidden_dim=args.hidden_dim).to(device)
    seed_payload = torch.load(
        args.seed_checkpoint, map_location="cpu", weights_only=False
    )
    if "recurrent_residual_actor" in seed_payload:
        metadata = seed_payload.get("recurrent_residual", {})
        if int(metadata.get("hidden_dim", args.hidden_dim)) == args.hidden_dim:
            actor.load_state_dict(seed_payload["recurrent_residual_actor"])
    before = actor_metrics(actor, validation, device)
    actor_candidates = [("initial", copy.deepcopy(actor.state_dict()), before)]
    bc_weight = np.asarray(train["actor_weight"], np.float32)
    actor_rows = bc_weight > 0
    nonzero_actor_fraction = float(np.mean(
        np.linalg.norm(np.asarray(train["action"])[actor_rows], axis=1) > 1e-4
    ))
    if nonzero_actor_fraction < args.min_nonzero_actor_fraction:
        raise RuntimeError(
            "actor supervision has insufficient action diversity: "
            f"nonzero fraction {nonzero_actor_fraction:.4f} < "
            f"{args.min_nonzero_actor_fraction:.4f}"
        )
    train_actor(
        actor, train, bc_weight,
        steps=args.bc_steps, batch_sequences=args.batch_sequences,
        lr=args.actor_lr, device=device, rng=rng,
    )
    after_bc = actor_metrics(actor, validation, device)
    actor_candidates.append(("bc", copy.deepcopy(actor.state_dict()), after_bc))

    all_advantage = np.zeros(len(train["action"]), np.float32)
    batch = 8192
    target_critic.eval()
    value.eval()
    with torch.no_grad():
        for start in range(0, len(all_advantage), batch):
            end = min(len(all_advantage), start + batch)
            observation = torch.from_numpy(
                train["normalized_observation"][start:end]
            ).to(device)
            action = torch.from_numpy(
                np.asarray(train["action"][start:end], np.float32)
            ).to(device)
            q1, q2 = target_critic(observation, action)
            advantage = torch.minimum(q1, q2) - value(observation)
            all_advantage[start:end] = advantage[:, 0].cpu().numpy()
    awr_multiplier = np.clip(
        np.exp(args.temperature * all_advantage), 0.0, args.max_awr_weight
    ).astype(np.float32)
    awr_weight = bc_weight * awr_multiplier
    train_actor(
        actor, train, awr_weight,
        steps=args.awr_steps, batch_sequences=args.batch_sequences,
        lr=0.5 * args.actor_lr, device=device, rng=rng,
    )
    after_awr = actor_metrics(actor, validation, device)
    actor_candidates.append(("awr", copy.deepcopy(actor.state_dict()), after_awr))
    selected_stage, selected_state, selected_metrics = min(
        actor_candidates,
        key=lambda row: float("inf") if row[2]["mae"] is None else row[2]["mae"],
    )
    actor.load_state_dict(selected_state)

    actor.eval()
    seed_payload["recurrent_residual_actor"] = {
        name: value.detach().cpu() for name, value in actor.state_dict().items()
    }
    metadata = {
        "target": "residual",
        "algorithm": "recurrent_bc_iql_awr_real_data",
        "gates": [int(args.focus_gate)],
        "hidden_dim": int(args.hidden_dim),
        "dataset": str(args.dataset.resolve()),
        "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
        "normalization_checkpoint": str(args.normalization_checkpoint.resolve()),
        "normalization_checkpoint_sha256": sha256(args.normalization_checkpoint),
        "seed_checkpoint_sha256": sha256(args.seed_checkpoint),
        "critic_steps": int(args.critic_steps),
        "bc_steps": int(args.bc_steps),
        "awr_steps": int(args.awr_steps),
        "expectile": float(args.expectile),
        "temperature": float(args.temperature),
        "nonzero_actor_fraction": nonzero_actor_fraction,
        "selected_stage": selected_stage,
        "selected_validation": selected_metrics,
        "validation_improvement_fraction": (
            0.0 if before["mae"] in (None, 0.0) else
            float((before["mae"] - selected_metrics["mae"]) / before["mae"])
        ),
        "critic": last_losses,
        "advantage": {
            "mean": float(np.mean(all_advantage[bc_weight > 0])),
            "positive_fraction": float(np.mean(all_advantage[bc_weight > 0] > 0)),
            "awr_weight_mean": float(np.mean(awr_multiplier[bc_weight > 0])),
            "awr_weight_max": float(np.max(awr_multiplier[bc_weight > 0])),
        },
        "before": before,
        "after_bc": after_bc,
        "after_awr": after_awr,
    }
    seed_payload["recurrent_residual"] = metadata
    seed_payload["offline_iql"] = {
        "critic": {name: value.cpu() for name, value in target_critic.state_dict().items()},
        "value": {name: value.cpu() for name, value in value.state_dict().items()},
        "reward_scale": reward_scale,
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(seed_payload, args.output)
    report = args.output.with_suffix(".json")
    report.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output.resolve()), **metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
