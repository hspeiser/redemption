"""Scalar Layer-2 baseline: the ACTUAL deployed learner in fastsim.

The trusted baseline arm for controller-parity rollouts must be the
deployed implementation itself, not another port.  This adapter wraps a
real train_vq2_sac_live.VQ2SACLearner (constructed with the verbatim
kwarg mapping the trainer uses, from a frozen session config) behind
the fastsim backbone interface, looping it scalar-fashion over every
world each step with per-world controller state swapped in and out
(reference cursor/gate, trajectory z-integrator, predictive-handoff
latch).

Residual-actor observations are built with the SAME shared builder the
batched port uses, so a paired Layer-2 run differs from the port arm in
exactly one dimension: controller implementation.

Self-check: replaying the golden 35.37 fixture through this adapter
must match final actions at <= the Layer-1 bar; that validates the
kwarg mapping against the deployed constructor.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


def _learner_from_config(config_path: Path):
    """Construct VQ2SACLearner with the trainer's own kwarg mapping."""
    from scripts.train_vq2_sac_live import (
        VQ2SACLearner,
        parse_gate_value_pairs,
        parse_gate_phase_windows,
    )

    raw = json.loads(Path(config_path).read_text())["args"]
    args = Namespace(**raw)
    args.seed_checkpoint = Path(args.seed_checkpoint)
    args.demo = Path(args.demo)
    learner = VQ2SACLearner(
        args.seed_checkpoint,
        args.demo,
        line_path=args.line,
        line_map_path=args.map,
        line_model_path=args.line_model,
        line_speed_cap=args.line_speed_cap,
        line_clearance=args.line_clearance,
        device=args.device,
        actor_lr=args.actor_lr,
        warmup_actor_lr=args.warmup_actor_lr,
        critic_lr=args.critic_lr,
        alpha=args.alpha,
        bc_weight=args.bc_weight,
        sac_actor_weight=args.sac_actor_weight,
        trust_weight=args.trust_weight,
        teacher_blend=args.teacher_blend,
        teacher_blends=parse_gate_value_pairs(args.teacher_blends),
        actor_speed_limits=parse_gate_value_pairs(args.actor_speed_limits),
        actor_speed_governor_margin=args.actor_speed_governor_margin,
        residual_scale=args.residual_scale,
        residual_scale_gates=parse_gate_value_pairs(
            args.residual_scale_gates
        ),
        residual_output_clip=args.residual_output_clip,
        macro_exploration_gates=tuple(
            int(value.strip())
            for value in args.macro_exploration_gates.split(",")
            if value.strip()
        ),
        macro_lateral_residual_scale=args.macro_lateral_residual_scale,
        zero_actor_output=args.zero_actor_output,
        actor_warmup_updates=args.actor_warmup_updates,
        longitudinal_position_gain=args.longitudinal_position_gain,
        longitudinal_velocity_gain=args.longitudinal_velocity_gain,
        longitudinal_position_gains=parse_gate_value_pairs(
            args.longitudinal_position_gains, value_type=float
        ),
        longitudinal_velocity_gains=parse_gate_value_pairs(
            args.longitudinal_velocity_gains, value_type=float
        ),
        lateral_position_gain=args.lateral_position_gain,
        lateral_velocity_gain=args.lateral_velocity_gain,
        special_lateral_gate=args.special_lateral_gate,
        special_lateral_gain_scale=args.special_lateral_gain_scale,
        lateral_gain_scales=parse_gate_value_pairs(
            args.lateral_gain_scales
        ),
        lateral_feedback_limits=parse_gate_value_pairs(
            args.lateral_feedback_limits
        ),
        lateral_bias_gates=tuple(
            int(value.strip())
            for value in args.lateral_bias_gates.split(",")
            if value.strip()
        ),
        lateral_action_bias=args.lateral_action_bias,
        right_lateral_bias_gates=tuple(
            int(value.strip())
            for value in args.right_lateral_bias_gates.split(",")
            if value.strip()
        ),
        right_lateral_action_bias=args.right_lateral_action_bias,
        extra_lateral_biases=tuple(
            (
                int(item.split(":", 1)[0].strip()),
                float(item.split(":", 1)[1].strip()),
            )
            for item in args.extra_lateral_biases.split(",")
            if item.strip()
        ),
        vertical_position_gain=args.vertical_position_gain,
        vertical_velocity_gain=args.vertical_velocity_gain,
        special_vertical_gate=args.special_vertical_gate,
        special_vertical_gain_scale=args.special_vertical_gain_scale,
        vertical_bias_gates=tuple(
            int(value.strip())
            for value in args.vertical_bias_gates.split(",")
            if value.strip()
        ),
        vertical_action_bias=args.vertical_action_bias,
        extra_vertical_biases=parse_gate_value_pairs(
            args.extra_vertical_biases, value_type=float
        ),
        trajectory_blend=args.trajectory_blend,
        trajectory_blend_gates=tuple(
            int(value.strip())
            for value in args.trajectory_blend_gates.split(",")
            if value.strip()
        ),
        trajectory_blends=parse_gate_value_pairs(
            args.trajectory_blends, value_type=float
        ),
        trajectory_kp_scale=args.trajectory_kp_scale,
        trajectory_kv_scale=args.trajectory_kv_scale,
        trajectory_attitude_gain=args.trajectory_attitude_gain,
        reference_mode=args.reference_mode,
        reference_feedback_scale=args.reference_feedback_scale,
        reference_thrust_scale=args.reference_thrust_scale,
        reference_thrust_scales=parse_gate_value_pairs(
            args.reference_thrust_scales, value_type=float
        ),
        reference_rate_scale=args.reference_rate_scale,
        reference_rate_scales=parse_gate_value_pairs(
            args.reference_rate_scales, value_type=float
        ),
        reference_velocity_scale=args.reference_velocity_scale,
        reference_velocity_scales=parse_gate_value_pairs(
            args.reference_velocity_scales, value_type=float
        ),
        reference_lateral_offsets=parse_gate_value_pairs(
            args.reference_lateral_offsets, value_type=float
        ),
        reference_vertical_offsets=parse_gate_value_pairs(
            args.reference_vertical_offsets, value_type=float
        ),
        reference_sequential_speed=args.reference_sequential_speed,
        reference_sequential_speeds=parse_gate_value_pairs(
            args.reference_sequential_speeds, value_type=float
        ),
        predictive_handoff_distances=parse_gate_value_pairs(
            args.predictive_handoff_distances, value_type=float
        ),
        gate_center_funnel_gates=tuple(
            int(value.strip())
            for value in args.gate_center_funnel_gates.split(",")
            if value.strip()
        ),
        gate_center_funnel_distance=args.gate_center_funnel_distance,
        gate_center_funnel_full_distance=(
            args.gate_center_funnel_full_distance
        ),
        gate_center_funnel_strength=args.gate_center_funnel_strength,
        reference_action_lead=args.reference_action_lead,
        gate4_action_lead=args.gate4_action_lead,
        reference_action_leads=parse_gate_value_pairs(
            args.reference_action_leads, value_type=int
        ),
        reference_max_advance=args.reference_max_advance,
        reference_max_retreat=args.reference_max_retreat,
        train_gate=args.train_gate,
        residual_gates=tuple(
            int(value.strip())
            for value in args.residual_gates.split(",")
            if value.strip()
        ),
        residual_phase_windows=parse_gate_phase_windows(
            args.residual_phase_windows
        ),
        frozen_residual_episode=args.frozen_residual_episode,
        frozen_residual_gates=tuple(
            int(value.strip())
            for value in args.frozen_residual_gates.split(",")
            if value.strip()
        ),
        frozen_action_episode=args.frozen_action_episode,
        frozen_action_gates=tuple(
            int(value.strip())
            for value in args.frozen_action_gates.split(",")
            if value.strip()
        ),
        actor_objective=args.actor_objective,
        awr_weight=args.awr_weight,
        awr_temperature=args.awr_temperature,
        awr_max_weight=args.awr_max_weight,
        positive_td_priority=args.positive_td_priority,
        finish_replay_boost=args.finish_replay_boost,
        n_step=args.n_step,
        ppo_residual_checkpoint=args.ppo_residual_checkpoint,
        secondary_ppo_residual_checkpoint=(
            args.secondary_ppo_residual_checkpoint
        ),
        secondary_ppo_residual_gates=tuple(
            int(value.strip())
            for value in args.secondary_ppo_residual_gates.split(",")
            if value.strip()
        ),
        ppo_residual_schedule=args.ppo_residual_schedule,
        champion_demo_path=args.interleave_champion_demo,
        champion_config_path=args.interleave_champion_config,
        champion_uses_residual=getattr(
            args, "interleave_champion_residual", True
        ),
    )
    return learner


class ScalarLiveTeacherAdapter:
    """Backbone adapter running the deployed scalar learner per world."""

    needs_extras = True

    def __init__(self, config_path, n_envs, device="cpu",
                 map_path=None, demo_path=None):
        from aigp.fastsim.liveteacher import BatchedLiveTeacher

        self.dev = torch.device(device)
        self.n_envs = n_envs
        self.learner = _learner_from_config(Path(config_path))
        # shared observation construction (identical to the port arm)
        self._obs_builder = BatchedLiveTeacher(
            Path(config_path), n_envs=n_envs, device=device,
            map_path=map_path, demo_path=demo_path,
        )
        self.n_pts = getattr(self._obs_builder, "n_pts", 0)
        self._fresh = (None, None, 0.0, None)
        self._state = [self._fresh] * n_envs
        self.idx = torch.zeros(n_envs, dtype=torch.long)
        self.steps = torch.zeros(n_envs, dtype=torch.long)

    def _capture(self):
        lr = self.learner
        return (
            lr.reference_cursor,
            lr.reference_gate,
            float(lr.trajectory_controller.zi),
            lr.predictive_handoff_latched_gate,
        )

    def _restore(self, state):
        lr = self.learner
        (lr.reference_cursor, lr.reference_gate,
         lr.trajectory_controller.zi,
         lr.predictive_handoff_latched_gate) = state

    def reset(self, env_ids):
        self.learner.begin_episode(explore=False)
        fresh = self._capture()
        for env_index in np.atleast_1d(
                env_ids.cpu().numpy() if torch.is_tensor(env_ids)
                else np.asarray(env_ids)):
            self._state[int(env_index)] = fresh

    def reset_nearest(self, env_ids, p):
        self.reset(env_ids)

    @torch.no_grad()
    def action(self, p, v, R, prev_action=None, target=None):
        n = p.shape[0]
        if prev_action is None:
            prev_action = torch.zeros(n, 4, device=p.device)
        if target is None:
            target = torch.zeros(n, dtype=torch.long, device=p.device)
        obs = self._obs_builder._build_obs(
            p.to(self.dev), v.to(self.dev), R.to(self.dev),
            prev_action.to(self.dev),
            torch.clamp(target.to(self.dev), 0, 16),
        ).cpu().numpy().astype(np.float32)
        out = np.zeros((n, 4), np.float32)
        for env_index in range(n):
            self._restore(self._state[env_index])
            selected, _residual, _mean = self.learner.action(
                obs[env_index], deterministic=True,
                exploration_clip=0.0,
            )
            out[env_index] = selected
            self._state[env_index] = self._capture()
        return torch.tensor(out, device=p.device)
