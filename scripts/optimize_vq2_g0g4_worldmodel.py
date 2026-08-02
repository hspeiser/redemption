"""CEM optimization of the exact live teacher in the learned world ensemble."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import ACT_DIM, HOLE_HALF, FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.lineopt import load_oriented_gates  # noqa: E402
from aigp.fastsim.live_teacher import LiveTeacherController  # noqa: E402
from aigp.rl.sac import GaussianActor  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble, decode_observations  # noqa: E402
from scripts.fastsim_train_ppo import (  # noqa: E402
    load_live_teacher_config,
    load_schedule_gate_positions,
)


def decoded_demo(demo_path: Path, map_path: Path) -> dict:
    data = np.load(demo_path)
    gates = json.loads(map_path.read_text())["gates"]
    positions = np.asarray([g["pos"] for g in gates], float)
    state = decode_observations(data["observation"], positions)
    quat = np.roll(Rotation.from_matrix(state.rotation).as_quat(), 1, axis=1)
    return {
        "pos": state.position.astype(np.float32),
        "vel": state.velocity.astype(np.float32),
        "quat": quat.astype(np.float32),
        "gate": state.gate_index.astype(np.float32),
    }


def release_states(dataset: Path) -> dict:
    payloads = [np.load(dataset / f"{name}.npz")
                for name in ("train", "validation", "test")]
    rows = []
    for payload in payloads:
        episode = payload["episode"]
        step = payload["step"]
        first = np.r_[True, episode[1:] != episode[:-1]] & (step == 0)
        rows.append({
            "pos": payload["position"][first],
            "vel": payload["velocity"][first],
            "rotation": payload["rotation"][first],
        })
    pos = np.concatenate([r["pos"] for r in rows]).astype(np.float32)
    vel = np.concatenate([r["vel"] for r in rows]).astype(np.float32)
    rotation = np.concatenate([r["rotation"] for r in rows])
    quat = np.roll(Rotation.from_matrix(rotation).as_quat(), 1, axis=1)
    return {"pos": pos, "vel": vel, "quat": quat.astype(np.float32)}


def unpack(
    theta: np.ndarray,
    geometry_limit: float = 0.35,
    lead_limit: int = 14,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray,
]:
    if theta.ndim != 2 or theta.shape[1] not in (20, 30, 35):
        raise ValueError(
            "theta must have shape (N, 20), (N, 30), or (N, 35), "
            f"got {theta.shape}"
        )
    leads = np.rint(
        np.clip(theta[:, :5], -2.0, float(lead_limit))
    ).astype(np.int64)
    thrust = np.clip(theta[:, 5:10], 0.72, 1.35).astype(np.float32)
    velocity_scale = np.clip(theta[:, 10:15], 0.75, 2.20).astype(np.float32)
    blend = np.clip(theta[:, 15:20], 0.0, 0.60).astype(np.float32)
    if theta.shape[1] == 20:
        lateral_offset = np.zeros((len(theta), 5), np.float32)
        vertical_offset = np.zeros((len(theta), 5), np.float32)
    else:
        lateral_offset = np.clip(
            theta[:, 20:25], -geometry_limit, geometry_limit
        ).astype(np.float32)
        vertical_offset = np.clip(
            theta[:, 25:30], -geometry_limit, geometry_limit
        ).astype(np.float32)
    rate_scale = (
        np.clip(theta[:, 30:35], 0.65, 2.20).astype(np.float32)
        if theta.shape[1] == 35
        else np.ones((len(theta), 5), np.float32)
    )
    return (
        leads,
        thrust,
        velocity_scale,
        blend,
        lateral_offset,
        vertical_offset,
        rate_scale,
    )


@torch.no_grad()
def evaluate(
    theta: np.ndarray,
    *,
    worlds: int,
    model: SurrogateModel,
    ensemble: ResidualEnsemble,
    demo_path: Path,
    map_path: Path,
    obstacles: Path,
    teacher_config: Path,
    demo_states: dict,
    spawn_states: dict,
    device: str,
    robust: bool = True,
    controller_rate_gain: np.ndarray | None = None,
    aleatoric_scale: float = 0.0,
    live_estimator_realism: bool = False,
    impulse_rate_hz: float = 0.0,
    controller_fixed_overrides: dict | None = None,
    actor: GaussianActor | None = None,
    actor_obs_mean: torch.Tensor | None = None,
    actor_obs_var: torch.Tensor | None = None,
    residual_scale: float = 0.20,
    multigate_vision: bool = False,
    vision_outcome_model: Path | None = None,
    geometry_limit: float = 0.35,
    lead_limit: int = 14,
    tier_target: float = 0.0,
) -> list[dict]:
    candidates = len(theta)
    n = candidates * worlds
    (
        leads,
        thrust,
        velocity_scale,
        blend,
        lateral_offset,
        vertical_offset,
        rate_scale,
    ) = unpack(theta, geometry_limit, lead_limit)
    leads_env = np.repeat(leads, worlds, axis=0)
    thrust_env = np.repeat(thrust, worlds, axis=0)
    velocity_env = np.repeat(velocity_scale, worlds, axis=0)
    blend_env = np.repeat(blend, worlds, axis=0)
    rate_env = np.repeat(rate_scale, worlds, axis=0)
    _gate_positions, gate_frames = load_oriented_gates(map_path)
    reference_offsets_world = (
        lateral_offset[:, :, None] * gate_frames[None, :5, :, 0]
        + vertical_offset[:, :, None] * gate_frames[None, :5, :, 2]
    ).astype(np.float32)
    reference_offsets_env = np.repeat(
        reference_offsets_world, worlds, axis=0
    )
    cfg = FastEnvConfig(
        n_envs=n,
        race_gates=5,
        random_start_frac=0.0,
        spawn_at_rest=True,
        max_episode_s=14.0,
        speed_cap_mps=12.0,
        act_delay_steps_min=0,
        # Command latency is already present in the learned closed-loop
        # residual through current+previous action. Adding another full frame
        # here double-counted it and drove the protected baseline below its
        # measured 100% four-run calibration.
        act_delay_steps_max=0,
        reloc_events=False,
        world_model_use_mean=not robust,
        world_model_aleatoric_scale=(
            float(aleatoric_scale) if robust else 0.0
        ),
        impulse_rate_hz=float(impulse_rate_hz) if robust else 0.0,
        impulse_velocity_mps=(0.10, 0.55),
        impulse_vertical_scale=0.35,
        residual_scale=float(residual_scale),
        vision_outcome_model=(
            str(vision_outcome_model) if vision_outcome_model else ""
        ),
        common_random_worlds=int(worlds),
    )
    cfg.apply_vision10hz()
    if multigate_vision:
        cfg.apply_multigate10hz()
    if live_estimator_realism and robust:
        # Preserve the measured GPU-10 Hz estimator distribution.  The old
        # optimizer disabled FOV droughts/relocalization and then collapsed
        # position noise to 0--3 cm.  That made a controller which failed
        # 0/4 live look 95% reliable in the surrogate.  These effects perturb
        # the controller's observed position only; they are deliberately
        # separate from the learned vehicle dynamics.
        cfg.fov_vision = True
        cfg.reloc_events = True
    else:
        cfg.fov_vision = False
        cfg.reloc_events = False
        cfg.pos_noise_lo = 0.0
        cfg.pos_noise_hi = 0.03 if robust else 0.0
    cfg.dr_thrust = (0.99, 1.01) if robust else (1.0, 1.0)
    cfg.dr_rate_gain = (0.99, 1.01) if robust else (1.0, 1.0)
    cfg.dr_rate_tau = (0.97, 1.03) if robust else (1.0, 1.0)
    cfg.dr_drag = (0.23, 0.27) if robust else (0.25, 0.25)
    controller_arrays, controller_fixed = load_live_teacher_config(
        teacher_config, n, map_path=map_path
    )
    controller_fixed.setdefault(
        "schedule_gate_positions", load_schedule_gate_positions(map_path)
    )
    if controller_fixed_overrides:
        controller_fixed.update(controller_fixed_overrides)
    # Rate scales are normally a fixed five-gate vector loaded from the live
    # config.  CEM promotes them to per-environment candidate parameters, so
    # remove the fixed copy before splatting both dictionaries into the exact
    # controller.
    controller_fixed.pop("reference_rate_scales", None)
    # Preserve every fixed option from the exact live campaign (predictive
    # handoff, feedback gains, gate trims, gain scales, and funneling).  CEM
    # changes only the schedule and smooth gate-local reference geometry
    # represented by theta.
    controller_arrays.update({
        "action_leads": leads_env,
        "thrust_scales": thrust_env,
        "trajectory_velocity_scales": velocity_env,
        "trajectory_blends": blend_env,
        "reference_rate_scales": rate_env,
        "reference_gate_offsets_world": reference_offsets_env,
    })
    controller = LiveTeacherController(
        demo_path, n, device=device,
        # The live teacher converts desired rates with --line-model.  That
        # calibration is intentionally separate from the plant/world model.
        # Keeping the two coupled reproduced blend=0 actions but silently
        # changed every trajectory-blend action.
        rate_gain=np.asarray(
            model.rate_gain
            if controller_rate_gain is None else controller_rate_gain
        ),
        **controller_arrays,
        **controller_fixed,
    )
    env = FastVQ2Env(
        model, map_path, demo_states=demo_states, config=cfg, device=device,
        obstacles_path=obstacles, backbone=controller,
        residual_ensemble=ensemble, spawn_states=spawn_states,
    )
    # Every candidate must see the same randomized plants, estimator events,
    # action delay, and residual member.  Independent random worlds turned
    # small-batch CEM winners into lucky seeds rather than better controls.
    for name in (
        "dr_thrust", "dr_K", "dr_tau", "dr_drag", "noise_amp",
        "reloc_next_t", "reloc_end_t", "reloc_dir", "reloc_rate",
        "act_delay", "residual_member",
    ):
        value = getattr(env, name)
        base_value = value[:worlds].clone()
        repeats = (candidates,) + (1,) * (value.ndim - 1)
        value.copy_(base_value.repeat(*repeats))
    obs = env.observations()
    action = torch.zeros(n, ACT_DIM, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    failed = torch.zeros_like(finished)
    elapsed = torch.zeros(n, device=device)
    steps = torch.zeros(n, device=device)
    minimum_clearance = torch.full((n,), 10.0, device=device)
    gate_clearance = torch.full((n, 5), np.nan, device=device)
    gate_cross_lateral = torch.full((n, 5), np.nan, device=device)
    gate_cross_vertical = torch.full((n, 5), np.nan, device=device)
    gate_times = torch.zeros(n, 5, device=device)
    failure_gate = torch.full((n,), -1, dtype=torch.long, device=device)
    support_sum = torch.zeros(n, device=device)
    disagreement_sum = torch.zeros(n, device=device)
    active_count = torch.zeros(n, device=device)
    all_rows = torch.arange(n, device=device)
    for _ in range(420):
        if actor is not None:
            if actor_obs_mean is None or actor_obs_var is None:
                raise ValueError("actor normalization tensors are required")
            normalized = torch.clamp(
                (obs - actor_obs_mean) / torch.sqrt(actor_obs_var + 1e-6),
                -8.0,
                8.0,
            )
            action = actor.deterministic(normalized)
        obs, _reward, done, info = env.step(action)
        live = ~(finished | failed)
        steps += live.float()
        support_sum += info["world_model_support_z"] * live.float()
        disagreement_sum += info["world_model_disagreement"] * live.float()
        active_count += live.float()
        crossed = info["cross_r"] >= 0.0
        margin = HOLE_HALF - info["cross_r"]
        minimum_clearance = torch.where(
            crossed & live, torch.minimum(minimum_clearance, margin),
            minimum_clearance,
        )
        passed = info["passed"] & live
        if passed.any():
            passed_gate = torch.clamp(info["target"] - 1, 0, 4)
            gate_clearance[all_rows, passed_gate] = torch.where(
                passed, margin, gate_clearance[all_rows, passed_gate]
            )
            gate_cross_lateral[all_rows, passed_gate] = torch.where(
                passed,
                info["cross_lateral_m"],
                gate_cross_lateral[all_rows, passed_gate],
            )
            gate_cross_vertical[all_rows, passed_gate] = torch.where(
                passed,
                info["cross_vertical_m"],
                gate_cross_vertical[all_rows, passed_gate],
            )
            gate_times[all_rows, passed_gate] = torch.where(
                passed, steps / cfg.control_hz,
                gate_times[all_rows, passed_gate]
            )
        new_finish = info["finished"] & live
        elapsed = torch.where(new_finish, steps / cfg.control_hz, elapsed)
        finished |= new_finish
        new_failure = done & live & ~new_finish
        failure_gate = torch.where(
            new_failure, info["target"], failure_gate
        )
        failed |= new_failure
        if bool((finished | failed).all()):
            break
    finish = finished.view(candidates, worlds)
    elapsed = elapsed.view(candidates, worlds)
    clearance = minimum_clearance.view(candidates, worlds)
    support = (support_sum / active_count.clamp(min=1)).view(candidates, worlds)
    disagreement = (
        disagreement_sum / active_count.clamp(min=1)
    ).view(candidates, worlds)
    gate_clearance = gate_clearance.view(candidates, worlds, 5)
    gate_cross_lateral = gate_cross_lateral.view(candidates, worlds, 5)
    gate_cross_vertical = gate_cross_vertical.view(candidates, worlds, 5)
    gate_times = gate_times.view(candidates, worlds, 5)
    failure_gate = failure_gate.view(candidates, worlds)
    reports = []
    for index in range(candidates):
        mask = finish[index]
        rate = float(mask.float().mean())
        if tier_target > 0.0:
            tier_mask = mask & (elapsed[index] < float(tier_target))
            tier_rate = float(tier_mask.float().mean())
            # Smooth curriculum signal around the hard deadline.  Failures
            # remain zero; successful laps transition continuously from
            # nearly zero to nearly one over roughly 0.4 s.  The strict rate
            # above remains the qualification metric and dominant score.
            soft_tier = torch.sigmoid(
                (float(tier_target) - elapsed[index]) / 0.20
            ) * mask.float()
            tier_soft_rate = float(soft_tier.mean())
        else:
            tier_mask = mask
            tier_rate = rate
            tier_soft_rate = rate
        reports.append({
            "finish_rate": rate,
            "tier_target_s": float(tier_target) if tier_target > 0.0 else None,
            "tier_success_rate": tier_rate,
            "tier_success_count": int(tier_mask.sum()),
            "tier_soft_rate": tier_soft_rate,
            "median_s": float(elapsed[index][mask].median()) if mask.any() else None,
            "p90_s": float(elapsed[index][mask].quantile(0.9)) if mask.any() else None,
            "clearance_p10_m": float(clearance[index][mask].quantile(0.1))
                if mask.any() else -1.0,
            "support_z_mean": float(support[index].mean()),
            "support_z_p90": float(support[index].quantile(0.9)),
            "disagreement_mean": float(disagreement[index].mean()),
            "gate_time_median_s": [
                float(gate_times[index, gate_times[index, :, gate] > 0, gate].median())
                if bool((gate_times[index, :, gate] > 0).any()) else None
                for gate in range(5)
            ],
            "gate_clearance_p10_m": [
                float(torch.nanquantile(gate_clearance[index, :, gate], 0.1))
                if bool(torch.isfinite(gate_clearance[index, :, gate]).any()) else None
                for gate in range(5)
            ],
            "gate_cross_lateral_quantiles_m": [
                [
                    float(torch.nanquantile(
                        gate_cross_lateral[index, :, gate], quantile
                    ))
                    for quantile in (0.1, 0.5, 0.9)
                ]
                if bool(torch.isfinite(
                    gate_cross_lateral[index, :, gate]
                ).any()) else None
                for gate in range(5)
            ],
            "gate_cross_vertical_quantiles_m": [
                [
                    float(torch.nanquantile(
                        gate_cross_vertical[index, :, gate], quantile
                    ))
                    for quantile in (0.1, 0.5, 0.9)
                ]
                if bool(torch.isfinite(
                    gate_cross_vertical[index, :, gate]
                ).any()) else None
                for gate in range(5)
            ],
            "failure_histogram": {
                str(gate): int((failure_gate[index] == gate).sum())
                for gate in range(5)
                if bool((failure_gate[index] == gate).any())
            },
        })
    return reports


def score(
    report: dict,
    *,
    reliability_floor: float = 0.90,
    time_weight: float = 35.0,
    tier_target: float = 0.0,
    tier_bonus: float = 0.0,
    clearance_floor: float = 0.12,
) -> float:
    finish_rate = report["finish_rate"]
    # A timed tier is a per-world terminal outcome: the vehicle must both
    # finish and beat the strict cutoff.  Applying the reliability floor to
    # finish rate plus successful-run median allowed candidates with many
    # slow laps to masquerade as 9/10-qualified policies.
    rate = (
        report.get("tier_success_rate", finish_rate)
        if tier_target > 0.0 else finish_rate
    )
    soft_rate = (
        report.get("tier_soft_rate", rate)
        if tier_target > 0.0 else rate
    )
    duration = report["median_s"] if report["median_s"] is not None else 20.0
    clearance = report["clearance_p10_m"]
    support_penalty = max(0.0, report["support_z_p90"] - 2.5)
    disagreement_penalty = max(0.0, report["disagreement_mean"] - 0.08)
    clearance_penalty = max(0.0, float(clearance_floor) - clearance)
    floor = float(np.clip(reliability_floor, 0.0, 1.0))
    # Below the requested reliability, each point of success probability is
    # worth much more than any plausible lap-time gain.  Once the candidate
    # clears the floor, the slope drops and CEM follows the speed/reliability
    # frontier instead of drifting ever deeper into a slow safe basin.
    reliability_value = (
        10_000.0 * rate
        if rate < floor
        else 10_000.0 * floor + 200.0 * (rate - floor)
    )
    tier_value = (
        float(tier_bonus)
        if tier_target > 0.0 and rate >= floor
        else 0.0
    )
    return (
        reliability_value - float(time_weight) * duration
        + 25.0 * finish_rate
        + (500.0 * soft_rate if tier_target > 0.0 else 0.0)
        - 80.0 * support_penalty - 500.0 * disagreement_penalty
        # Clearance is a safety constraint, not a minor style preference.
        # At the old 3,000x slope a 10 cm floor violation cost only 300
        # points and was routinely overwhelmed by reliability/timing terms.
        - 50_000.0 * clearance_penalty
        + tier_value
    )


def promotion_eligible(
    reports: list[dict],
    *,
    reliability_floor: float,
    clearance_floor: float,
    tier_target: float,
) -> bool:
    """Require every plausible dynamics model to clear every hard floor."""
    for report in reports:
        rate = (
            float(report.get("tier_success_rate", report["finish_rate"]))
            if tier_target > 0.0
            else float(report["finish_rate"])
        )
        if rate < float(reliability_floor):
            return False
        if float(report["finish_rate"]) < float(reliability_floor):
            return False
        if float(report["clearance_p10_m"]) < float(clearance_floor):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--ensemble", type=Path, required=True, action="append",
        help=(
            "world-model ensemble; repeat to optimize the worst-case score "
            "across multiple plausible dynamics models"
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--controller-model", type=Path,
        help="Rate-command calibration used by the live trajectory tracker.",
    )
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument(
        "--teacher-config", type=Path, required=True,
        help="Exact live campaign config whose fixed controller options are preserved.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elite", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=14)
    parser.add_argument("--worlds", type=int, default=48)
    parser.add_argument(
        "--selection-worlds", type=int, default=512,
        help="paired worlds used to re-rank all generation winners",
    )
    parser.add_argument(
        "--final-worlds", type=int, default=2048,
        help="worlds in the locked robust audit of the selected candidate",
    )
    parser.add_argument(
        "--clean-worlds", type=int, default=512,
        help="worlds in the locked mean-model audit of the selected candidate",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aleatoric-scale", type=float, default=0.0)
    parser.add_argument("--reliability-floor", type=float, default=0.90)
    parser.add_argument("--time-weight", type=float, default=35.0)
    parser.add_argument("--tier-target", type=float, default=0.0)
    parser.add_argument("--tier-bonus", type=float, default=0.0)
    parser.add_argument(
        "--clearance-floor", type=float, default=0.12,
        help="minimum p10 gate clearance rewarded without a heavy penalty",
    )
    parser.add_argument("--impulse-rate-hz", type=float, default=0.08)
    parser.add_argument(
        "--actor", type=Path,
        help="Optional PPO/SAC checkpoint evaluated as residual control during CEM.",
    )
    parser.add_argument("--residual-scale", type=float, default=0.20)
    parser.add_argument(
        "--active-gates", nargs="+", type=int,
        help="Only optimize these gate slots; freeze all other controller values.",
    )
    parser.add_argument(
        "--geometry-gates", nargs="+", type=int,
        help=(
            "Gate slots whose local crossing geometry may move. Defaults to "
            "--active-gates; use e.g. 1 2 3 4 to protect gate 0."
        ),
    )
    parser.add_argument(
        "--freeze-geometry",
        action="store_true",
        help=(
            "Keep every inherited lateral/vertical crossing offset fixed "
            "while optimizing the other parameters of --active-gates."
        ),
    )
    parser.add_argument(
        "--geometry-limit", type=float, default=0.35,
        help="Maximum absolute gate-local lateral/vertical offset in metres.",
    )
    parser.add_argument(
        "--lead-limit", type=int, default=14,
        help="Maximum per-gate reference action lead in controller steps.",
    )
    parser.add_argument(
        "--multigate-vision", action="store_true",
        help="Use the current multigate 10 Hz estimator error model.",
    )
    parser.add_argument(
        "--vision-outcome-model", type=Path,
        help="Learned state-dependent 10 Hz vision-fusion probability JSON.",
    )
    parser.add_argument(
        "--live-estimator-realism", action="store_true",
        help=(
            "retain measured 10 Hz estimator noise, FOV droughts, and "
            "relocalization events during robust evaluation"
        ),
    )
    parser.add_argument(
        "--initial", type=Path,
        help="Live-validated or incrementally optimized controller to center CEM on.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.population, args.elite, args.iterations, args.worlds = 6, 2, 2, 4
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = SurrogateModel.load(args.model)
    controller_rate_gain = (
        SurrogateModel.load(args.controller_model).rate_gain
        if args.controller_model is not None else model.rate_gain
    )
    ensembles = []
    ensemble_metadata = []
    for ensemble_path in args.ensemble:
        ensemble, metadata = ResidualEnsemble.load(
            ensemble_path, args.device
        )
        ensemble.eval()
        ensembles.append(ensemble)
        ensemble_metadata.append(metadata)
    demo_states = decoded_demo(args.demo, args.map)
    spawn_states = release_states(args.dataset)
    actor = actor_obs_mean = actor_obs_var = None
    if args.actor is not None:
        checkpoint = torch.load(
            args.actor, map_location=args.device, weights_only=False,
        )
        actor = GaussianActor(53, ACT_DIM).to(args.device)
        actor.load_state_dict(checkpoint["actor"])
        actor.eval()
        actor_obs_mean = checkpoint["obs_mean"].to(args.device)
        actor_obs_var = checkpoint["obs_var"].to(args.device)
        print(f"actor residual: {args.actor} scale={args.residual_scale}", flush=True)

    def evaluate_models(
        theta: np.ndarray,
        *,
        worlds: int,
        robust: bool,
        seed: int,
    ) -> list[list[dict]]:
        reports = []
        for ensemble in ensembles:
            torch.manual_seed(seed)
            reports.append(evaluate(
                theta, worlds=worlds, model=model, ensemble=ensemble,
                demo_path=args.demo, map_path=args.map,
                obstacles=args.obstacles, teacher_config=args.teacher_config,
                demo_states=demo_states,
                spawn_states=spawn_states, device=args.device,
                robust=robust, controller_rate_gain=controller_rate_gain,
                aleatoric_scale=args.aleatoric_scale,
                live_estimator_realism=args.live_estimator_realism,
                impulse_rate_hz=args.impulse_rate_hz,
                actor=actor,
                actor_obs_mean=actor_obs_mean,
                actor_obs_var=actor_obs_var,
                residual_scale=args.residual_scale,
                multigate_vision=args.multigate_vision,
                vision_outcome_model=args.vision_outcome_model,
                geometry_limit=args.geometry_limit,
                lead_limit=args.lead_limit,
                tier_target=args.tier_target,
            ))
        return reports

    def worst_case_scores(reports_by_model: list[list[dict]]) -> np.ndarray:
        return np.asarray([
            [
                score(
                    report,
                    reliability_floor=args.reliability_floor,
                    time_weight=args.time_weight,
                    tier_target=args.tier_target,
                    tier_bonus=args.tier_bonus,
                    clearance_floor=args.clearance_floor,
                )
                for report in reports
            ]
            for reports in reports_by_model
        ]).min(axis=0)
    if args.initial is not None:
        initial_text = args.initial.read_text()
        try:
            initial = json.loads(initial_text)
        except json.JSONDecodeError:
            # Optimizer stdout is JSONL. This makes a finished generation
            # search recoverable if the later large-world audit is killed.
            initial = None
            for line in reversed(initial_text.splitlines()):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "iteration" in row and "leads" in row:
                    initial = row
                    break
            if initial is None:
                raise ValueError(
                    f"{args.initial} is neither a candidate JSON nor "
                    "optimizer JSONL containing an iteration candidate"
                )
        mean = np.asarray(
            initial["leads"] + initial["thrust_scales"]
            + initial["velocity_scales"] + initial["trajectory_blends"]
            + initial.get("lateral_offsets_m", [0.0] * 5)
            + initial.get("vertical_offsets_m", [0.0] * 5)
            + initial.get("rate_scales", [1.0] * 5),
            dtype=np.float64,
        )
    else:
        # With no explicit initial, center CEM on the exact live campaign,
        # never on generic defaults that may describe a different controller.
        base_arrays, _base_fixed = load_live_teacher_config(
            args.teacher_config, 1, map_path=args.map
        )
        mean = np.r_[
            base_arrays["action_leads"][0],
            base_arrays["thrust_scales"][0],
            base_arrays["trajectory_velocity_scales"][0],
            base_arrays["trajectory_blends"][0],
            np.zeros(5, np.float64),
            np.zeros(5, np.float64),
            base_arrays["reference_rate_scales"][0],
        ].astype(np.float64)
    # Canonicalize an inherited candidate to the bounds of this search.  If a
    # prior run used a larger geometry/lead envelope, leaving the raw mean
    # outside the new bounds would collapse many CEM samples onto the clipping
    # boundary and prevent the optimizer from exploring inward.
    bounded = unpack(mean[None], args.geometry_limit, args.lead_limit)
    mean = np.concatenate([value[0] for value in bounded]).astype(np.float64)
    protected = mean.copy()
    active_gates = set(range(5)) if args.active_gates is None else set(args.active_gates)
    invalid_gates = sorted(gate for gate in active_gates if gate < 0 or gate >= 5)
    if invalid_gates:
        parser.error(f"--active-gates values must be in 0..4, got {invalid_gates}")
    geometry_gates = (
        set()
        if args.freeze_geometry else
        set(active_gates)
        if args.geometry_gates is None else set(args.geometry_gates)
    )
    invalid_geometry_gates = sorted(
        gate for gate in geometry_gates if gate < 0 or gate >= 5
    )
    if invalid_geometry_gates:
        parser.error(
            "--geometry-gates values must be in 0..4, got "
            f"{invalid_geometry_gates}"
        )
    if args.geometry_limit < 0.0 or args.geometry_limit > HOLE_HALF:
        parser.error(
            f"--geometry-limit must be in [0, {HOLE_HALF}], "
            f"got {args.geometry_limit}"
        )

    def freeze_inactive(values: np.ndarray) -> None:
        for gate in range(5):
            if gate in active_gates:
                continue
            values[..., gate] = protected[gate]
            values[..., 5 + gate] = protected[5 + gate]
            values[..., 10 + gate] = protected[10 + gate]
            values[..., 15 + gate] = protected[15 + gate]
            values[..., 30 + gate] = protected[30 + gate]
        for gate in range(5):
            if gate in geometry_gates:
                continue
            values[..., 20 + gate] = protected[20 + gate]
            values[..., 25 + gate] = protected[25 + gate]

    std = np.r_[
        np.full(5, 3.0), np.full(5, 0.10),
        np.full(5, 0.35), np.full(5, 0.08),
        np.full(5, 0.10), np.full(5, 0.10),
        np.full(5, 0.20),
    ]
    best = None
    history = []
    candidate_pool = [("initial", mean.copy())]
    started = time.time()
    for iteration in range(args.iterations):
        theta = mean + std * rng.standard_normal((args.population, 35))
        freeze_inactive(theta)
        theta[0] = protected
        if best is not None:
            theta[1] = best["theta"]
        reports_by_model = evaluate_models(
            theta,
            worlds=args.worlds,
            robust=True,
            seed=args.seed + 1009 * iteration,
        )
        scores = worst_case_scores(reports_by_model)
        order = np.argsort(-scores)
        winner = int(order[0])
        winner_reports = [reports[winner] for reports in reports_by_model]
        if best is None or scores[winner] > best["score"]:
            best = {
                "score": float(scores[winner]),
                "theta": theta[winner].copy(),
                "reports": winner_reports,
            }
        elite = theta[order[:args.elite]]
        mean = 0.35 * mean + 0.65 * elite.mean(0)
        freeze_inactive(mean)
        std = 0.45 * std + 0.55 * (elite.std(0) + np.r_[
            np.full(5, 0.35), np.full(5, 0.012),
            np.full(5, 0.025), np.full(5, 0.012),
            np.full(5, 0.010), np.full(5, 0.010),
            np.full(5, 0.015),
        ])
        unpacked = unpack(
            best["theta"][None], args.geometry_limit, args.lead_limit
        )
        row = {
            "iteration": iteration,
            "iteration_best_by_model": winner_reports,
            "champion_by_model": best["reports"],
            "leads": unpacked[0][0].tolist(),
            "thrust_scales": unpacked[1][0].tolist(),
            "velocity_scales": unpacked[2][0].tolist(),
            "trajectory_blends": unpacked[3][0].tolist(),
            "lateral_offsets_m": unpacked[4][0].tolist(),
            "vertical_offsets_m": unpacked[5][0].tolist(),
            "rate_scales": unpacked[6][0].tolist(),
        }
        history.append(row)
        candidate_pool.append((f"iteration_{iteration}", theta[winner].copy()))
        checkpoint = {
            "leads": row["leads"],
            "thrust_scales": row["thrust_scales"],
            "velocity_scales": row["velocity_scales"],
            "trajectory_blends": row["trajectory_blends"],
            "lateral_offsets_m": row["lateral_offsets_m"],
            "vertical_offsets_m": row["vertical_offsets_m"],
            "rate_scales": row["rate_scales"],
            "source_iteration": iteration,
            "champion_by_model": row["champion_by_model"],
        }
        checkpoint_path = args.out.with_name(args.out.stem + "_checkpoint.json")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(checkpoint, indent=2))
        print(json.dumps(row), flush=True)
    # Re-rank every distinct generation winner on a much larger paired world
    # set.  The protected candidate often wins several early generations
    # bit-for-bit; evaluating those duplicates again wastes most of the locked
    # audit without adding evidence.
    generated_candidate_count = len(candidate_pool)
    unique_pool = []
    duplicate_labels: dict[str, list[str]] = {}
    exact_candidates: dict[bytes, int] = {}
    for label, value in candidate_pool:
        key = np.ascontiguousarray(value).tobytes()
        existing = exact_candidates.get(key)
        if existing is None:
            exact_candidates[key] = len(unique_pool)
            unique_pool.append((label, value))
            duplicate_labels[label] = []
        else:
            duplicate_labels[unique_pool[existing][0]].append(label)
    candidate_pool = unique_pool
    selection_theta = np.stack([value for _, value in candidate_pool])
    # Re-rank one candidate at a time with the same RNG seed.  Besides keeping
    # memory bounded (21 winners x 1,024 worlds cannot fit on a laptop GPU),
    # this gives candidates genuinely identical stochastic worlds, including
    # runtime aleatoric noise, vision dropouts, relocations, and impulses.
    selection_reports_by_model = [[] for _ in ensembles]
    for candidate_index in range(len(selection_theta)):
        candidate_reports = evaluate_models(
            selection_theta[candidate_index:candidate_index + 1],
            worlds=max(1, int(args.selection_worlds)),
            robust=True,
            seed=args.seed + 77191,
        )
        for model_index, reports in enumerate(candidate_reports):
            selection_reports_by_model[model_index].append(reports[0])
        print(json.dumps({
            "selection_candidate": candidate_index + 1,
            "selection_candidates": len(selection_theta),
            "reports_by_model": [reports[0] for reports in candidate_reports],
        }), flush=True)
    selection_scores = worst_case_scores(selection_reports_by_model)
    selection_eligibility = np.asarray([
        promotion_eligible(
            [reports[index] for reports in selection_reports_by_model],
            reliability_floor=args.reliability_floor,
            clearance_floor=args.clearance_floor,
            tier_target=args.tier_target,
        )
        for index in range(len(selection_theta))
    ], dtype=bool)
    eligible_indices = np.flatnonzero(selection_eligibility)
    selected = int(
        eligible_indices[np.argmax(selection_scores[eligible_indices])]
        if len(eligible_indices)
        else np.argmax(selection_scores)
    )
    best = {
        "theta": selection_theta[selected].copy(),
        "score": float(selection_scores[selected]),
        "label": candidate_pool[selected][0],
        "reports": [
            reports[selected] for reports in selection_reports_by_model
        ],
    }
    # High-statistics locked surrogate evaluations of the re-ranked winner.
    final_by_model = evaluate_models(
        best["theta"][None],
        worlds=max(1, int(args.final_worlds)) if not args.smoke else 32,
        robust=True,
        seed=args.seed + 99173,
    )
    final_by_model = [reports[0] for reports in final_by_model]
    clean_by_model = evaluate_models(
        best["theta"][None],
        worlds=max(1, int(args.clean_worlds)) if not args.smoke else 16,
        robust=False,
        seed=args.seed + 221137,
    )
    clean_by_model = [reports[0] for reports in clean_by_model]
    final_promotion_eligible = promotion_eligible(
        final_by_model,
        reliability_floor=args.reliability_floor,
        clearance_floor=args.clearance_floor,
        tier_target=args.tier_target,
    )
    (
        leads,
        thrust,
        velocity_scale,
        blend,
        lateral_offset,
        vertical_offset,
        rate_scale,
    ) = unpack(best["theta"][None], args.geometry_limit, args.lead_limit)
    gate_values = lambda values: ",".join(
        f"{gate}:{float(value):.9g}"
        for gate, value in enumerate(values)
    )
    output = {
        "leads": leads[0].tolist(),
        "thrust_scales": thrust[0].tolist(),
        "velocity_scales": velocity_scale[0].tolist(),
        "trajectory_blends": blend[0].tolist(),
        "lateral_offsets_m": lateral_offset[0].tolist(),
        "vertical_offsets_m": vertical_offset[0].tolist(),
        "rate_scales": rate_scale[0].tolist(),
        "geometry_limit_m": float(args.geometry_limit),
        "lead_limit": int(args.lead_limit),
        "live_overrides": {
            "reference_action_leads": ",".join(
                f"{gate}:{int(value)}"
                for gate, value in enumerate(leads[0])
            ),
            "reference_thrust_scales": gate_values(thrust[0]),
            "reference_velocity_scales": gate_values(velocity_scale[0]),
            "reference_rate_scales": gate_values(rate_scale[0]),
            "trajectory_blends": gate_values(blend[0]),
            "reference_lateral_offsets": gate_values(lateral_offset[0]),
            "reference_vertical_offsets": gate_values(vertical_offset[0]),
        },
        "robust_eval_by_model": final_by_model,
        "mean_model_eval_by_model": clean_by_model,
        "selection": {
            "worlds": int(args.selection_worlds),
            "label": best["label"],
            "reports_by_model": best["reports"],
            "candidates": len(candidate_pool),
            "generated_candidates": generated_candidate_count,
            "exact_duplicate_labels": duplicate_labels,
            "eligible_candidates": int(selection_eligibility.sum()),
            "selected_was_eligible": bool(selection_eligibility[selected]),
        },
        "promotion_eligible": bool(final_promotion_eligible),
        "promotion_rejection_reasons": [
            reason
            for reason, failed in (
                (
                    "reliability_floor",
                    any(
                        float(report.get(
                            "tier_success_rate" if args.tier_target > 0.0
                            else "finish_rate",
                            report["finish_rate"],
                        )) < args.reliability_floor
                        for report in final_by_model
                    ),
                ),
                (
                    "finish_rate_floor",
                    any(
                        float(report["finish_rate"])
                        < args.reliability_floor
                        for report in final_by_model
                    ),
                ),
                (
                    "clearance_floor",
                    any(
                        float(report["clearance_p10_m"])
                        < args.clearance_floor
                        for report in final_by_model
                    ),
                ),
            )
            if failed
        ],
        "final_worlds": int(args.final_worlds),
        "clean_worlds": int(args.clean_worlds),
        "clearance_floor_m": float(args.clearance_floor),
        "history": history,
        "elapsed_s": time.time() - started,
        "world_model_metadata": [
            {
                "dataset": metadata.get("dataset"),
                "base_model": metadata.get("base_model"),
            }
            for metadata in ensemble_metadata
        ],
        "vision_outcome_model": (
            str(args.vision_outcome_model.resolve())
            if args.vision_outcome_model else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
