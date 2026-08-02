"""Hybrid residual world model used by the bounded VQ2 POC.

The learned component does not replace rigid-body integration.  It predicts
small corrections to velocity, attitude, and body rates after one analytic
control step.  Keeping position as an integrated physical state prevents the
network from turning EKF relocalization jumps into fictitious acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from aigp.fastsim.sysid import SurrogateModel
from aigp.rl.vq2_features import MAX_RATE, WIRE_RATE_LIMIT


@dataclass
class DecodedState:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    rates: np.ndarray
    previous_action: np.ndarray
    gate_index: np.ndarray
    confidence: np.ndarray


def decode_observations(
    observation: np.ndarray,
    gate_positions: np.ndarray,
) -> DecodedState:
    """Recover the localizer state encoded by the shared 53-D observation."""
    obs = np.asarray(observation, np.float64)
    one = obs.ndim == 1
    if one:
        obs = obs[None]
    first = obs[:, 21:24].copy()
    second = obs[:, 24:27].copy()
    first /= np.linalg.norm(first, axis=1, keepdims=True) + 1e-12
    second -= first * np.sum(first * second, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True) + 1e-12
    third = np.cross(first, second)
    rotation = np.stack([first, second, third], axis=2)
    gate = np.argmax(obs[:, 34:51], axis=1).astype(np.int64)
    gate = np.clip(gate, 0, len(gate_positions) - 1)
    relative_body = obs[:, :3] * 10.0
    relative_world = np.einsum("nij,nj->ni", rotation, relative_body)
    position = np.asarray(gate_positions, float)[gate] - relative_world
    velocity_body = obs[:, 18:21] * 10.0
    velocity = np.einsum("nij,nj->ni", rotation, velocity_body)
    state = DecodedState(
        position=position,
        velocity=velocity,
        rotation=rotation,
        rates=obs[:, 27:30] * MAX_RATE,
        previous_action=obs[:, 30:34].copy(),
        gate_index=gate,
        confidence=obs[:, 51:53].copy(),
    )
    if not one:
        return state
    return DecodedState(**{
        key: value[0] for key, value in state.__dict__.items()
    })


def rotation_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) exponential map."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    axis = rotvec / torch.clamp(theta, min=1e-9)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ], dim=-1).reshape(*rotvec.shape[:-1], 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    eye = eye.expand(*rotvec.shape[:-1], 3, 3)
    s = torch.sin(theta)[..., None]
    c = (1.0 - torch.cos(theta))[..., None]
    result = eye + s * skew + c * (skew @ skew)
    tiny = (theta[..., 0] < 1e-7)[..., None, None]
    return torch.where(tiny, eye + skew * theta[..., None], result)


def rotation_log(rotation: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) logarithm, stable for the small one-step rotations here."""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0))
    vee = torch.stack([
        rotation[..., 2, 1] - rotation[..., 1, 2],
        rotation[..., 0, 2] - rotation[..., 2, 0],
        rotation[..., 1, 0] - rotation[..., 0, 1],
    ], dim=-1)
    scale = angle / torch.clamp(2.0 * torch.sin(angle), min=1e-7)
    result = vee * scale[..., None]
    small = angle < 1e-5
    return torch.where(small[..., None], 0.5 * vee, result)


def analytic_step(
    position: torch.Tensor,
    velocity: torch.Tensor,
    rotation: torch.Tensor,
    rates: torch.Tensor,
    action: torch.Tensor,
    model: SurrogateModel,
    *,
    control_hz: float = 30.0,
    substeps: int = 4,
    thrust_cap: float = 0.52,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One nominal rigid-body step in the conventions used by FastVQ2Env."""
    dtype, device = position.dtype, position.device
    rate_limit = torch.as_tensor(WIRE_RATE_LIMIT, dtype=dtype, device=device)
    target_rates = action[:, :3] * rate_limit
    gain = torch.as_tensor(
        np.abs(model.rate_gain), dtype=dtype, device=device
    )
    tau = torch.as_tensor(model.rate_tau, dtype=dtype, device=device)
    drag = torch.as_tensor(
        np.abs(model.drag_lin), dtype=dtype, device=device
    )
    drag_quad = torch.as_tensor(
        model.drag_quad or [0.0, 0.0, 0.0], dtype=dtype, device=device
    )
    gravity = torch.tensor([0.0, 0.0, 9.81], dtype=dtype, device=device)
    wire_thrust = torch.clamp(0.5 * (action[:, 3] + 1.0), 0.0, thrust_cap)
    thrust = (
        model.thrust_gain * wire_thrust
        + model.thrust_quad * wire_thrust.square()
    )
    dt = 1.0 / (control_hz * substeps)
    p, v, r, w = position, velocity, rotation, rates
    for _ in range(substeps):
        w = w + dt * (gain * target_rates - w) / tau
        r = r @ rotation_exp(w * dt)
        v_body = torch.einsum("nij,nj->ni", r.transpose(1, 2), v)
        force_body = -drag * v_body
        force_body = force_body - drag_quad * torch.linalg.norm(
            v, dim=1, keepdim=True
        ) * v_body
        force_body = force_body.clone()
        force_body[:, 2] -= thrust
        acceleration = gravity + torch.einsum("nij,nj->ni", r, force_body)
        v = v + acceleration * dt
        p = p + v * dt
    return p, v, r, w


def residual_features(
    velocity: torch.Tensor,
    rotation: torch.Tensor,
    rates: torch.Tensor,
    action: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Course-independent features for the physical residual."""
    velocity_body = torch.einsum(
        "nij,nj->ni", rotation.transpose(1, 2), velocity
    )
    gravity_body = rotation[:, 2, :]  # R^T @ world +z
    return torch.cat([
        velocity_body,
        gravity_body,
        rates,
        action,
        previous_action,
    ], dim=1)


class ResidualMember(nn.Module):
    """Probabilistic residual member; output is mean and aleatoric log sigma."""

    def __init__(self, input_dim: int = 17, hidden: int = 192,
                 output_dim: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, output_dim * 2),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = self.net(features).chunk(2, dim=-1)
        log_std = torch.clamp(raw, -5.0, 2.0)
        return mean, log_std


class ResidualEnsemble(nn.Module):
    def __init__(self, members: int = 5, input_dim: int = 17,
                 hidden: int = 192, output_dim: int = 9) -> None:
        super().__init__()
        self.members = nn.ModuleList([
            ResidualMember(input_dim, hidden, output_dim)
            for _ in range(members)
        ])
        self.register_buffer("x_mean", torch.zeros(input_dim))
        self.register_buffer("x_std", torch.ones(input_dim))
        self.register_buffer("y_mean", torch.zeros(output_dim))
        self.register_buffer("y_std", torch.ones(output_dim))

    def set_normalization(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(torch.clamp(x.std(0), min=1e-5))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(torch.clamp(y.std(0), min=1e-5))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.x_mean) / self.x_std
        means, log_stds = [], []
        for member in self.members:
            mean, log_std = member(normalized)
            means.append(mean * self.y_std + self.y_mean)
            log_stds.append(log_std + torch.log(self.y_std))
        return torch.stack(means), torch.stack(log_stds)

    def save(self, path: str | Path, metadata: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "members": len(self.members),
            "input_dim": int(self.x_mean.numel()),
            "output_dim": int(self.y_mean.numel()),
            "metadata": metadata,
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") \
            -> tuple["ResidualEnsemble", dict]:
        payload = torch.load(path, map_location=device, weights_only=False)
        ensemble = cls(
            members=payload["members"],
            input_dim=payload["input_dim"],
            output_dim=payload["output_dim"],
        ).to(device)
        ensemble.load_state_dict(payload["state_dict"])
        return ensemble, payload.get("metadata", {})


class ResidualEnsemblePool(nn.Module):
    """Treat independently trained residual ensembles as one model pool.

    Each constituent ensemble retains its own input/output normalization.
    Concatenating checkpoints by copying only their member networks would be
    incorrect because the checkpoints were fitted on different datasets.
    ``FastVQ2Env`` assigns every simulated world one virtual member from this
    pool, so a policy is optimized across both within-model epistemic
    uncertainty and between-model dataset/version disagreement.
    """

    def __init__(self, ensembles: list[ResidualEnsemble]) -> None:
        super().__init__()
        if not ensembles:
            raise ValueError("ResidualEnsemblePool requires at least one model")
        input_dims = {int(model.x_mean.numel()) for model in ensembles}
        output_dims = {int(model.y_mean.numel()) for model in ensembles}
        if len(input_dims) != 1 or len(output_dims) != 1:
            raise ValueError(
                "all pooled residual ensembles must share input/output dims"
            )
        self.ensembles = nn.ModuleList(ensembles)
        self.member_counts = tuple(len(model.members) for model in ensembles)
        self.member_count = int(sum(self.member_counts))

    def forward(self, features: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [model(features) for model in self.ensembles]
        means = torch.cat([result[0] for result in outputs], dim=0)
        log_stds = torch.cat([result[1] for result in outputs], dim=0)
        return means, log_stds

    def support_z_by_member(self, features: torch.Tensor) -> torch.Tensor:
        """Return each virtual member's own-normalization support distance."""
        rows = []
        for model, count in zip(self.ensembles, self.member_counts):
            normalized = (features - model.x_mean) / model.x_std
            support = torch.sqrt(normalized.square().mean(1))
            rows.append(support.unsqueeze(0).expand(count, -1))
        return torch.cat(rows, dim=0)
