"""Fit the residual actor directly to a successful live episode.

This preserves the full reference controller and teaches only the bounded
closed-loop correction that was active in the winning gate segments.
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates", default="3")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--observation-noise", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(37)
    np.random.seed(37)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    if payload.get("kind") != "vq2_residual_sac":
        raise ValueError("checkpoint is not a VQ2 residual SAC checkpoint")

    episode = np.load(args.episode, allow_pickle=False)
    gates = {
        int(value.strip()) for value in args.gates.split(",")
        if value.strip()
    }
    gate_index = np.asarray(episode["gate_index"], np.int16)
    selected = np.isin(gate_index, sorted(gates))
    if not np.any(selected):
        raise ValueError(f"episode has no rows for gates {sorted(gates)}")

    mean = np.asarray(payload["observation_mean"], np.float32)
    std = np.asarray(payload["observation_std"], np.float32)
    observation = (
        np.asarray(episode["observation"], np.float32)[selected] - mean
    ) / std
    target = np.asarray(episode["action"], np.float32)[selected]
    observation_tensor = torch.from_numpy(observation).to(device)
    target_tensor = torch.from_numpy(target).to(device)

    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    actor.load_state_dict(payload["actor"])
    initial_actor = copy.deepcopy(actor).eval()
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.lr, weight_decay=1e-6
    )

    with torch.no_grad():
        initial_mse = F.mse_loss(
            actor.deterministic(observation_tensor), target_tensor
        ).item()
    for epoch in range(args.epochs):
        noisy = observation_tensor + (
            args.observation_noise
            * torch.randn_like(observation_tensor)
        )
        prediction = actor.deterministic(noisy)
        imitation = F.mse_loss(prediction, target_tensor)
        # Keep the fit local to the demonstrated neighborhood.
        with torch.no_grad():
            initial = initial_actor.deterministic(noisy)
        trust = F.mse_loss(prediction, initial)
        progress = epoch / max(args.epochs - 1, 1)
        loss = imitation + (0.02 * (1.0 - progress)) * trust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        optimizer.step()

    with torch.no_grad():
        final_prediction = actor.deterministic(observation_tensor)
        final_mse = F.mse_loss(
            final_prediction, target_tensor
        ).item()
        max_error = (
            final_prediction - target_tensor
        ).abs().max().item()

    payload["actor"] = actor.state_dict()
    payload.pop("actor_optimizer", None)
    payload["winning_residual_fit"] = {
        "source_episode": str(args.episode),
        "gates": sorted(gates),
        "rows": int(selected.sum()),
        "epochs": int(args.epochs),
        "initial_mse": float(initial_mse),
        "final_mse": float(final_mse),
        "max_abs_error": float(max_error),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        f"fit {int(selected.sum())} rows: mse "
        f"{initial_mse:.8f} -> {final_mse:.8f}, "
        f"max_abs_error={max_error:.6f}"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
