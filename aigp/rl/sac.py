"""Small SAC networks shared by offline seeding and live VQ2 training."""

from __future__ import annotations

import math

import torch
from torch import nn


def mlp(input_dim: int, hidden: tuple[int, ...], output_dim: int) \
        -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend([
            nn.Linear(previous, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        ])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (256, 256, 128),
    ) -> None:
        super().__init__()
        self.backbone = mlp(observation_dim, hidden, hidden[-1])
        # Remove the final linear output from mlp and use that width as a
        # feature layer for separate mean/uncertainty heads.
        feature_layer = self.backbone[-1]
        if not isinstance(feature_layer, nn.Linear):
            raise TypeError("actor backbone construction failed")
        self.backbone[-1] = nn.Identity()
        self.mean = nn.Linear(hidden[-1], action_dim)
        self.log_std = nn.Linear(hidden[-1], action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -3.5)

    def distribution(self, observation: torch.Tensor):
        feature = self.backbone(observation)
        mean = self.mean(feature)
        log_std = torch.clamp(self.log_std(feature), -5.0, -1.5)
        return mean, log_std

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution(observation)
        return torch.tanh(mean)

    def sample(self, observation: torch.Tensor):
        mean, log_std = self.distribution(observation)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        action = torch.tanh(pre_tanh)
        log_probability = (
            -0.5 * (
                noise.square() + 2.0 * log_std + math.log(2.0 * math.pi)
            )
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return action, log_probability, torch.tanh(mean)


class RecurrentActor(nn.Module):
    """Deterministic recurrent policy used for sequence distillation.

    The residual SAC actor remains feed-forward. This actor replaces only the
    frozen direct teacher, allowing gate-local curriculum blends to use an
    explicitly phase-aware policy without changing critic/replay formats.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def sequence(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        feature, hidden = self.gru(encoded, hidden)
        return torch.tanh(self.mean(feature)), hidden

    def step(
        self,
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action, hidden = self.sequence(observation.unsqueeze(1), hidden)
        return action[:, 0], hidden


class Critic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (256, 256, 128),
    ) -> None:
        super().__init__()
        self.network = mlp(observation_dim + action_dim, hidden, 1)

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([observation, action], dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.q1 = Critic(observation_dim, action_dim)
        self.q2 = Critic(observation_dim, action_dim)

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)
