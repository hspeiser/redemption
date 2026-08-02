"""Gate-local behavior cloning for a residual VQ2 actor checkpoint.

Only episodes that demonstrably clear ``--gate`` are accepted.  The actor's
feature extractor is frozen and the mean head is fitted to the successful
residual actions.  Non-lateral outputs are distilled back to their original
values, so a lateral line correction cannot silently rewrite thrust or rates.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.sac import GaussianActor  # noqa: E402
from aigp.rl.vq2_features import ACT_DIM, OBS_DIM  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--gate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lateral-weight", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(17)
    np.random.seed(17)
    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    actor = GaussianActor(OBS_DIM, ACT_DIM)
    actor.load_state_dict(payload["actor"])
    actor.train()
    original = copy.deepcopy(actor).eval()
    for parameter in actor.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in actor.log_std.parameters():
        parameter.requires_grad_(False)

    observations = []
    actions = []
    accepted = []
    for episode_path in args.episode:
        episode = np.load(episode_path, allow_pickle=False)
        gate = np.asarray(episode["gate_index"], np.int16)
        passed = np.asarray(episode["gates_passed"], np.int16)
        selected = gate == args.gate
        if not np.any(selected):
            raise ValueError(f"{episode_path} has no gate {args.gate} rows")
        if not np.any(selected & (passed > 0)):
            raise ValueError(
                f"{episode_path} did not officially clear gate {args.gate}"
            )
        observations.append(
            np.asarray(episode["observation"][selected], np.float32)
        )
        actions.append(np.asarray(episode["action"][selected], np.float32))
        accepted.append(str(episode_path))

    observation = np.concatenate(observations)
    target = np.concatenate(actions)
    mean = np.asarray(payload["observation_mean"], np.float32)
    std = np.asarray(payload["observation_std"], np.float32)
    normalized = torch.from_numpy((observation - mean) / std)
    target_tensor = torch.from_numpy(target)
    with torch.no_grad():
        original_action = original.deterministic(normalized)
        before = actor.deterministic(normalized)
        before_lateral = F.mse_loss(
            before[:, 0], target_tensor[:, 0]
        ).item()

    optimizer = torch.optim.Adam(actor.mean.parameters(), lr=args.lr)
    count = len(normalized)
    for _ in range(max(0, args.steps)):
        indices = torch.randint(
            count, (min(args.batch_size, count),)
        )
        predicted = actor.deterministic(normalized[indices])
        desired = target_tensor[indices]
        loss = (
            args.lateral_weight
            * F.mse_loss(predicted[:, 0], desired[:, 0])
            + args.distill_weight
            * F.mse_loss(
                predicted[:, 1:], original_action[indices, 1:]
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    actor.eval()
    with torch.no_grad():
        after = actor.deterministic(normalized)
        after_lateral = F.mse_loss(
            after[:, 0], target_tensor[:, 0]
        ).item()
        non_lateral_drift = F.mse_loss(
            after[:, 1:], original_action[:, 1:]
        ).item()
    payload["actor"] = actor.state_dict()
    # Old Adam moments point in the pre-cloning direction.
    payload.pop("actor_optimizer", None)
    payload["gate_local_bc"] = {
        "gate": int(args.gate),
        "episodes": accepted,
        "rows": int(count),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "lateral_mse_before": float(before_lateral),
        "lateral_mse_after": float(after_lateral),
        "non_lateral_drift_mse": float(non_lateral_drift),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(payload["gate_local_bc"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
