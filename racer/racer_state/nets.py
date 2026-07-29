"""MLP actor + twin-Q critic for the state-based SAC (no conv — obs is a small state vector)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(inp, hidden, out, n=2):
    layers, d = [], inp
    for _ in range(n):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


def droq_mlp(inp, hidden, out, n=2, dropout=0.01):
    """DroQ-style critic MLP: Dropout + LayerNorm before each activation. Damps Q overestimation
    (the phantom peaks the actor exploits) and keeps high update-to-data ratios stable."""
    layers, d = [], inp
    for _ in range(n):
        layers += [nn.Linear(d, hidden), nn.Dropout(dropout), nn.LayerNorm(hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.body = mlp(obs_dim, hidden, hidden, 1)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, o):
        h = F.relu(self.body(o))
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, o):
        mean, log_std = self.forward(o)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        u = dist.rsample()
        a = torch.tanh(u)
        logp = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return a, logp

    @torch.no_grad()
    def act(self, o, deterministic=True):
        mean, log_std = self.forward(o)
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(torch.distributions.Normal(mean, log_std.exp()).rsample())


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, n=2, dropout=0.0):
        super().__init__()
        make = (lambda: droq_mlp(obs_dim + act_dim, hidden, 1, 2, dropout)) if dropout > 0 \
            else (lambda: mlp(obs_dim + act_dim, hidden, 1, 2))
        self.heads = nn.ModuleList([make() for _ in range(n)])

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return torch.stack([h(x) for h in self.heads], 0)   # (n, B, 1)


def build_actor(cfg):
    return Actor(cfg.obs_dim, cfg.act_dim, cfg.hidden)


def build_critic(cfg):
    return Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden, cfg.num_critics,
                  getattr(cfg, "critic_dropout", 0.0))
