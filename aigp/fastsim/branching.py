"""State restoration helpers for offline counterfactual VQ2 branches."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def calibrated_position_sigma(
    raw_sigma_m: float,
    *,
    scale: float = 1.0,
    floor_m: float = 0.0,
) -> float:
    """Combine EKF covariance with measured unmodelled belief error."""
    raw = float(raw_sigma_m)
    if not np.isfinite(raw):
        raw = 0.10
    return max(float(floor_m), max(0.0, float(scale)) * raw)


def repair_race_gates(target_gate: int) -> int:
    """Return the minimum race horizon that keeps a repair target active."""
    gate = int(target_gate)
    if not 0 <= gate < 17:
        raise ValueError(f"target_gate must be in [0, 16], got {gate}")
    return max(5, gate + 1)


def matrices_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(rotation, float)).as_quat()
    return np.column_stack([xyzw[:, 3], xyzw[:, :3]]).astype(np.float32)


@torch.no_grad()
def restore_branch_cloud(
    env,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    rotation: np.ndarray,
    rates: np.ndarray,
    previous_action: np.ndarray,
    target_gate: int,
    position_sigma_m: float,
    landmark_age_s: float,
    seed: int,
    velocity_sigma_mps: float = 0.10,
    attitude_sigma_deg: float = 0.75,
    rate_sigma_radps: float = 0.05,
) -> None:
    """Restore one state into every world with a measured uncertainty cloud."""
    generator = torch.Generator(device=env.device)
    generator.manual_seed(int(seed))
    n = env.cfg.n_envs
    dev = env.device
    pos = torch.as_tensor(position, dtype=torch.float32, device=dev)
    vel = torch.as_tensor(velocity, dtype=torch.float32, device=dev)
    q = torch.as_tensor(
        matrices_to_wxyz(np.asarray(rotation)[None])[0],
        dtype=torch.float32,
        device=dev,
    )
    body_rates = torch.as_tensor(rates, dtype=torch.float32, device=dev)
    previous = torch.as_tensor(
        previous_action, dtype=torch.float32, device=dev
    )
    pos_sigma = float(np.clip(position_sigma_m, 0.02, 0.50))
    # ``position`` is the recorded EKF/controller belief, not privileged true
    # position.  Branching must perturb the latent physical state while
    # preserving that belief; otherwise every counterfactual controller sees
    # the sampled truth and unrealistically corrects the very localization
    # error we are trying to test.  FastVQ2Env defines belief as p + noise_pos.
    belief_error = pos_sigma * torch.randn(
        n, 3, generator=generator, device=dev
    )
    env.p[:] = pos - belief_error
    env.noise_pos[:] = belief_error
    env.noise_amp[:] = pos_sigma
    env.v[:] = vel + float(velocity_sigma_mps) * torch.randn(
        n, 3, generator=generator, device=dev
    )
    attitude_delta = (
        np.deg2rad(float(attitude_sigma_deg))
        * torch.randn(n, 3, generator=generator, device=dev)
    )
    delta_q = env._qrotvec(attitude_delta, 1.0)
    env.q[:] = env._qmul(q.expand(n, -1), delta_q)
    env.q[:] /= torch.linalg.norm(env.q, dim=-1, keepdim=True)
    env.w[:] = body_rates + float(rate_sigma_radps) * torch.randn(
        n, 3, generator=generator, device=dev
    )
    env.target[:] = int(target_gate)
    env.t_ep[:] = 0.0
    env.t_gate[:] = 0.0
    env.prev_action[:] = previous
    env.act_buf[:] = previous
    env.vis_age[:] = max(0.0, float(landmark_age_s))
    env.progress[:] = env._course_progress(env.p, env.target)
    env.best_prog[:] = env.progress
    env.t_best[:] = 0.0
    env.spawn_flag[:] = False
    env.reloc_next_t[:] = 1e6
    env.reloc_end_t[:] = 0.0
    env.reloc_rate[:] = 0.0
    if env.backbone is not None:
        ids = torch.arange(n, device=dev)
        if hasattr(env.backbone, "reset_nearest"):
            env.backbone.reset_nearest(ids, env.p + env.noise_pos)
        if hasattr(env.backbone, "set_target"):
            env.backbone.set_target(env.target)
        if hasattr(env.backbone, "set_previous_action"):
            env.backbone.set_previous_action(env.prev_action)
