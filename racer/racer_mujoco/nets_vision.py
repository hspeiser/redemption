"""Conv actor (and twin critic, for the optional stage-2 RL) for the pixel policy.

Input: 3 x 84x84 gate-filtered frames (spaced stack — the 3 channels ARE the time axis) + a 6-dim
proprio vector [body_rates/3, gravity_body]. Nature-CNN-ish trunk, tanh-Gaussian head, same action
semantics as the state policy (4 raw actions in (-1,1))."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def conv_trunk():
    return nn.Sequential(
        nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 256), nn.LayerNorm(256), nn.Tanh(),
    )


class VisionActor(nn.Module):
    def __init__(self, act_dim=4, proprio_dim=9):
        super().__init__()
        self.trunk = conv_trunk()
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 64), nn.ReLU())
        self.body = nn.Sequential(nn.Linear(256 + 64, 256), nn.ReLU())
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, img, prop):
        h = torch.cat([self.trunk(img), self.prop(prop)], -1)
        h = self.body(h)
        return self.mean(h), torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)

    @torch.no_grad()
    def act(self, img, prop, deterministic=True):
        mean, log_std = self.forward(img, prop)
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(torch.distributions.Normal(mean, log_std.exp()).rsample())


class VisionCritic(nn.Module):
    """Twin Q over (frames, proprio, action) — for the stage-2 SAC fine-tune if needed."""

    def __init__(self, act_dim=4, proprio_dim=9, n=2):
        super().__init__()
        self.trunk = conv_trunk()          # shared trunk (standard for pixel SAC)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256 + proprio_dim + act_dim, 256), nn.ReLU(),
                          nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n)])

    def forward(self, img, prop, act):
        h = self.trunk(img)
        x = torch.cat([h, prop, act], -1)
        return torch.stack([hd(x) for hd in self.heads], 0)
