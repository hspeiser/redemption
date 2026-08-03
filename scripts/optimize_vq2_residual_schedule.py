"""CEM-optimize a gate/phase residual schedule on top of a protected actor.

The learned actor supplies state feedback.  This optimizer searches only a
small feed-forward correction table (gate x phase knot x action) under the
counterexample-corrected world ensemble.  It gives long-horizon course time
and clearance direct credit without asking PPO to discover it through TD.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import ACT_DIM, HOLE_HALF, FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.live_teacher import LiveTeacherController  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from aigp.rl.sac import GaussianActor  # noqa: E402
from aigp.rl.counterfactual_gate import load_state_gated_actor  # noqa: E402
from scripts.fastsim_train_ppo import (  # noqa: E402
    load_demo_states,
    load_live_teacher_config,
    load_schedule_gate_positions,
)


def load_actor(path: Path, device: str):
    payload = torch.load(path, map_location=device, weights_only=False)
    actor = GaussianActor(53, ACT_DIM).to(device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    gated_actor = load_state_gated_actor(
        payload,
        actor,
        observation_dim=53,
        actor_factory=lambda: GaussianActor(53, ACT_DIM),
        device=device,
    )
    if gated_actor is not None:
        actor = gated_actor
    if "obs_mean" in payload and "obs_var" in payload:
        obs_mean = torch.as_tensor(payload["obs_mean"], device=device)
        obs_var = torch.as_tensor(payload["obs_var"], device=device)
    elif "observation_mean" in payload and "observation_std" in payload:
        # Live SAC checkpoints store standard deviation instead of variance.
        # Accepting both formats lets the offline optimizer evaluate the exact
        # actor used by train_vq2_sac_live.py rather than a converted proxy.
        obs_mean = torch.as_tensor(
            payload["observation_mean"], dtype=torch.float32, device=device
        )
        obs_std = torch.as_tensor(
            payload["observation_std"], dtype=torch.float32, device=device
        )
        obs_var = obs_std.square()
    else:
        raise KeyError(
            "actor checkpoint has neither obs_mean/obs_var nor "
            "observation_mean/observation_std normalization statistics"
        )
    return (
        actor,
        obs_mean,
        obs_var,
    )


def schedule_action(
    table: torch.Tensor,
    env: FastVQ2Env,
    worlds: int,
) -> torch.Tensor:
    """Linear interpolation over course phase within the current gate."""
    n, _, knots, _ = table.shape[0] * worlds, *table.shape[1:]
    expanded = table.repeat_interleave(worlds, dim=0)
    gate = torch.clamp(env.target, 0, table.shape[1] - 1)
    start = env.cum_len[gate]
    length = env.seg_len[gate]
    phase = torch.clamp((env.progress - start) / (length + 1e-9), 0.0, 1.0)
    coordinate = phase * (knots - 1)
    left = torch.floor(coordinate).long()
    right = torch.clamp(left + 1, max=knots - 1)
    fraction = coordinate - left.float()
    row = torch.arange(n, device=env.device)
    a = expanded[row, gate, left]
    b = expanded[row, gate, right]
    return a + fraction[:, None] * (b - a)


@torch.no_grad()
def evaluate(
    theta: np.ndarray,
    *,
    worlds: int,
    knots: int,
    actor: GaussianActor,
    obs_mean: torch.Tensor,
    obs_var: torch.Tensor,
    model: SurrogateModel,
    controller_model: SurrogateModel,
    ensemble: ResidualEnsemble,
    demo_path: Path,
    demo_states: dict,
    map_path: Path,
    obstacles: Path,
    teacher_config: Path,
    device: str,
    aleatoric_scale: float,
    impulse_rate_hz: float,
    seed: int,
    race_gates: int = 5,
    max_episode_s: float = 14.0,
    start_gate: int | None = None,
    multigate_estimator: bool = False,
    residual_scale: float = 0.2,
    residual_gates: tuple[int, ...] | None = None,
    residual_phase_windows: dict[int, tuple[float, float]] | None = None,
) -> list[dict]:
    torch.manual_seed(seed)
    candidates = len(theta)
    n = candidates * worlds
    table = torch.as_tensor(
        theta.reshape(candidates, race_gates, knots, ACT_DIM),
        dtype=torch.float32,
        device=device,
    )
    start_gate_weights: tuple[float, ...] = ()
    random_start_frac = 0.0
    if start_gate is not None:
        weights = np.zeros(race_gates, np.float32)
        weights[start_gate] = 1.0
        start_gate_weights = tuple(float(value) for value in weights)
        random_start_frac = 1.0
    cfg = FastEnvConfig(
        n_envs=n,
        race_gates=race_gates,
        random_start_frac=random_start_frac,
        demo_gate_weights=start_gate_weights,
        spawn_at_rest=True,
        max_episode_s=max_episode_s,
        speed_cap_mps=16.0,
        act_delay_steps_min=0,
        act_delay_steps_max=0,
        residual_scale=float(residual_scale),
        reloc_events=True,
        fov_vision=True,
        world_model_aleatoric_scale=aleatoric_scale,
        impulse_rate_hz=impulse_rate_hz,
        impulse_velocity_mps=(0.10, 0.55),
        impulse_vertical_scale=0.35,
        demo_corridor_m=2.0,
    )
    if multigate_estimator:
        cfg.apply_multigate10hz()
    else:
        cfg.apply_vision10hz()
    cfg.fov_vision = True
    cfg.reloc_events = True
    cfg.dr_thrust = (0.97, 1.03)
    cfg.dr_rate_gain = (0.95, 1.05)
    cfg.dr_rate_tau = (0.90, 1.10)
    cfg.dr_drag = (0.20, 0.35)
    arrays, fixed = load_live_teacher_config(
        teacher_config, n, gate_count=race_gates, map_path=map_path
    )
    fixed.setdefault(
        "schedule_gate_positions", load_schedule_gate_positions(map_path)
    )
    backbone = LiveTeacherController(
        demo_path,
        n,
        device=device,
        rate_gain=np.asarray(controller_model.rate_gain),
        **arrays,
        **fixed,
    )
    env = FastVQ2Env(
        model,
        map_path,
        demo_states=demo_states,
        config=cfg,
        device=device,
        obstacles_path=obstacles,
        backbone=backbone,
        residual_ensemble=ensemble,
    )
    # Common random numbers: every candidate sees the same randomized plant
    # worlds.  Without pairing, 32-world CEM selected lucky candidates that
    # collapsed from 100% in-search to 85% on the locked 2,048-world audit.
    for name in (
        "dr_thrust", "dr_K", "dr_tau", "dr_drag", "noise_amp",
        "reloc_next_t", "reloc_end_t", "reloc_dir", "reloc_rate",
        "act_delay", "residual_member",
    ):
        value = getattr(env, name)
        base = value[:worlds].clone()
        repeats = (candidates,) + (1,) * (value.ndim - 1)
        value.copy_(base.repeat(*repeats))
    active = torch.ones(n, dtype=torch.bool, device=device)
    finished = torch.zeros_like(active)
    elapsed = torch.full((n,), float("nan"), device=device)
    failure_gate = torch.full((n,), -1, dtype=torch.long, device=device)
    clearance = torch.full((n,), 10.0, device=device)
    demo_sum = torch.zeros(n, device=device)
    active_steps = torch.zeros(n, device=device)
    correction_sq = torch.zeros(n, device=device)
    support_max = torch.zeros(n, device=device)
    disagreement_max = torch.zeros(n, device=device)
    obs = env.observations()
    for _ in range(int(cfg.max_episode_s * cfg.control_hz) + 1):
        normalized = torch.clamp(
            (obs - obs_mean) / torch.sqrt(obs_var + 1e-6), -8.0, 8.0
        )
        actor_action = actor.deterministic(normalized)
        if residual_gates is not None:
            actor_active = torch.zeros_like(env.target, dtype=torch.bool)
            for gate_index in residual_gates:
                actor_active |= env.target == int(gate_index)
            if residual_phase_windows:
                phase_active = torch.ones_like(actor_active)
                for gate_index, (start, end) in residual_phase_windows.items():
                    on_gate = env.target == int(gate_index)
                    lookup_gate = env.target.clamp(0, race_gates - 1)
                    segment_start = env.cum_len[lookup_gate]
                    segment_length = env.seg_len[lookup_gate]
                    phase = torch.clamp(
                        (env.progress - segment_start)
                        / (segment_length + 1e-9),
                        0.0,
                        1.0,
                    )
                    phase_active &= ~on_gate | (
                        (phase >= float(start)) & (phase <= float(end))
                    )
                actor_active &= phase_active
            actor_action = actor_action * actor_active[:, None]
        correction = schedule_action(table, env, worlds)
        action = torch.clamp(actor_action + correction, -1.0, 1.0)
        correction_sq += correction.square().mean(1) * active.float()
        obs, _reward, done, info = env.step(action)
        support_max = torch.maximum(
            support_max,
            info["world_model_support_z"] * active.float(),
        )
        disagreement_max = torch.maximum(
            disagreement_max,
            info["world_model_disagreement"] * active.float(),
        )
        demo_sum += info["demo_dist"] * active.float()
        active_steps += active.float()
        crossed = active & (info["cross_r"] >= 0.0)
        clearance = torch.where(
            crossed,
            torch.minimum(clearance, HOLE_HALF - info["cross_r"]),
            clearance,
        )
        new_finish = active & info["finished"]
        elapsed = torch.where(new_finish, info["t_ep"], elapsed)
        finished |= new_finish
        failed = active & done & ~new_finish
        failure_gate = torch.where(failed, info["target"].long(), failure_gate)
        active &= ~done
        if not bool(active.any()):
            break
    finish = finished.view(candidates, worlds)
    elapsed = elapsed.view(candidates, worlds)
    clearance = clearance.view(candidates, worlds)
    failure_gate = failure_gate.view(candidates, worlds)
    demo_mean = (demo_sum / active_steps.clamp(min=1)).view(candidates, worlds)
    correction_rms = torch.sqrt(
        correction_sq / active_steps.clamp(min=1)
    ).view(candidates, worlds)
    support_max = support_max.view(candidates, worlds)
    disagreement_max = disagreement_max.view(candidates, worlds)
    reports = []
    for index in range(candidates):
        success = finish[index]
        times = elapsed[index][success]
        failures = failure_gate[index][~success]
        reports.append({
            "finish_rate": float(success.float().mean()),
            "median_s": float(times.median()) if len(times) else None,
            "p90_s": float(torch.quantile(times, 0.9)) if len(times) else None,
            "clearance_p10_m": (
                float(torch.quantile(clearance[index][success], 0.1))
                if bool(success.any()) else -1.0
            ),
            "demo_dist_mean_m": float(demo_mean[index].mean()),
            "correction_rms": float(correction_rms[index].mean()),
            "world_model_support_p90": float(torch.quantile(
                support_max[index], 0.9
            )),
            "world_model_disagreement_p90": float(torch.quantile(
                disagreement_max[index], 0.9
            )),
            "failure_histogram": {
                str(gate): int((failures == gate).sum())
                for gate in range(race_gates)
                if bool((failures == gate).any())
            },
        })
    return reports


def objective(
    report: dict,
    reliability_floor: float,
    time_weight: float,
    tier_target: float,
    tier_bonus: float,
) -> float:
    rate = report["finish_rate"]
    duration = report["median_s"] if report["median_s"] is not None else 20.0
    clearance = report["clearance_p10_m"]
    if rate < reliability_floor:
        reliability = 12_000.0 * rate
    else:
        reliability = (
            12_000.0 * reliability_floor
            + 300.0 * (rate - reliability_floor)
        )
    tier_value = (
        tier_bonus
        if tier_target > 0.0
        and rate >= reliability_floor
        and duration < tier_target
        else 0.0
    )
    return (
        reliability
        - time_weight * duration
        - 2500.0 * max(0.0, 0.08 - clearance)
        - 20.0 * report["correction_rms"]
        + tier_value
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument(
        "--ensemble", type=Path, required=True, action="append",
        help=(
            "world-model ensemble; repeat this argument to optimize the "
            "worst-case score across multiple plausible dynamics models"
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elite", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument(
        "--selection-worlds", type=int, default=512,
        help=(
            "Common-random worlds used to re-rank every generation winner "
            "before the expensive locked audits. This prevents a lucky "
            "small-batch champion from being promoted."
        ),
    )
    parser.add_argument("--robust-worlds", type=int, default=2048)
    parser.add_argument("--impulse-worlds", type=int, default=1024)
    parser.add_argument("--knots", type=int, default=3)
    parser.add_argument("--race-gates", type=int, default=5)
    parser.add_argument("--max-episode-s", type=float, default=14.0)
    parser.add_argument(
        "--start-gate", type=int, default=-1,
        help=(
            "Optional target gate whose demonstrated states seed every "
            "search world. Use this to optimize a protected late section "
            "without resimulating immutable prefix gates."
        ),
    )
    parser.add_argument(
        "--active-residual-gates", default="",
        help=(
            "Optional comma-separated gates on which the protected actor "
            "remains active. The searched schedule is controlled separately "
            "by --optimize-gates."
        ),
    )
    parser.add_argument(
        "--optimize-gates", default="",
        help=(
            "Optional comma-separated gate indices whose schedule knots may "
            "change. Empty optimizes all gates."
        ),
    )
    parser.add_argument("--aleatoric-scale", type=float, default=1.25)
    parser.add_argument(
        "--multigate-estimator",
        action="store_true",
        help=(
            "Use the fastsim multigate localization model. Enable this when "
            "the live harness runs with AIGP_MULTIGATE=1."
        ),
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.2,
        help=(
            "Residual authority applied by the live teacher backbone. This "
            "must match train_vq2_sac_live.py --residual-scale."
        ),
    )
    parser.add_argument("--impulse-rate-hz", type=float, default=0.06)
    parser.add_argument("--reliability-floor", type=float, default=0.90)
    parser.add_argument("--time-weight", type=float, default=65.0)
    parser.add_argument("--tier-target", type=float, default=0.0)
    parser.add_argument("--tier-bonus", type=float, default=0.0)
    parser.add_argument(
        "--initial", type=Path,
        help="optional prior result.json whose residual_schedule seeds CEM",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.start_gate >= args.race_gates:
        parser.error("--start-gate must be below --race-gates")
    if args.smoke:
        args.population, args.elite, args.iterations, args.worlds = 6, 2, 2, 4
    actor, obs_mean, obs_var = load_actor(args.actor, args.device)
    ensembles = []
    ensemble_metadata = []
    for ensemble_path in args.ensemble:
        ensemble, metadata = ResidualEnsemble.load(ensemble_path, args.device)
        ensemble.eval()
        ensembles.append(ensemble)
        ensemble_metadata.append(metadata)
    model = SurrogateModel.load(args.model)
    controller_model = SurrogateModel.load(args.controller_model)
    demo = load_demo_states(args.demo, args.map)
    keep = np.asarray(demo["gate"]) < args.race_gates
    demo = {key: np.asarray(value)[keep] for key, value in demo.items()}

    def evaluate_models(
        values: np.ndarray,
        *,
        worlds: int,
        impulse_rate_hz: float,
        seed: int,
    ) -> list[list[dict]]:
        """Evaluate identical candidates/world seeds under every model."""
        return [
            evaluate(
                values,
                worlds=worlds,
                knots=args.knots,
                actor=actor,
                obs_mean=obs_mean,
                obs_var=obs_var,
                model=model,
                controller_model=controller_model,
                ensemble=ensemble,
                demo_path=args.demo,
                demo_states=demo,
                map_path=args.map,
                obstacles=args.obstacles,
                teacher_config=args.teacher_config,
                device=args.device,
                aleatoric_scale=args.aleatoric_scale,
                impulse_rate_hz=impulse_rate_hz,
                seed=seed,
                race_gates=args.race_gates,
                max_episode_s=args.max_episode_s,
                start_gate=(
                    args.start_gate if args.start_gate >= 0 else None
                ),
                multigate_estimator=args.multigate_estimator,
                residual_scale=args.residual_scale,
                residual_gates=residual_gates,
            )
            for ensemble in ensembles
        ]

    def worst_case_scores(reports_by_model: list[list[dict]]) -> np.ndarray:
        per_model = np.asarray([
            [
                objective(
                    row, args.reliability_floor, args.time_weight,
                    args.tier_target, args.tier_bonus,
                )
                for row in reports
            ]
            for reports in reports_by_model
        ])
        return per_model.min(axis=0)
    rng = np.random.default_rng(args.seed)
    dimensions = args.race_gates * args.knots * ACT_DIM
    mean = np.zeros(dimensions, np.float64)
    if args.initial is not None:
        initial = json.loads(args.initial.read_text())
        mean = np.asarray(initial["residual_schedule"], np.float64).reshape(-1)
        if len(mean) != dimensions:
            raise ValueError("initial schedule shape does not match --knots")
    protected = mean.copy()
    std = np.full(dimensions, 0.10, np.float64)
    optimize_gates = {
        int(value.strip()) for value in args.optimize_gates.split(",")
        if value.strip()
    }
    if optimize_gates and not all(
        0 <= gate < args.race_gates for gate in optimize_gates
    ):
        raise ValueError(
            "--optimize-gates entries must be in "
            f"0..{args.race_gates - 1}"
        )
    residual_gates = tuple(
        int(value.strip())
        for value in args.active_residual_gates.split(",")
        if value.strip()
    ) or None
    if residual_gates is not None and not all(
        0 <= gate < args.race_gates for gate in residual_gates
    ):
        raise ValueError(
            "--active-residual-gates entries must be in "
            f"0..{args.race_gates - 1}"
        )
    active = np.ones(
        (args.race_gates, args.knots, ACT_DIM), bool
    )
    if optimize_gates:
        active[:] = False
        for gate in optimize_gates:
            active[gate] = True
    active = active.reshape(-1)
    std[~active] = 0.0
    best = None
    history = []
    candidate_pool = [("initial", mean.copy())]
    started = time.time()
    for iteration in range(args.iterations):
        theta = mean + std * rng.standard_normal(
            (args.population, dimensions)
        )
        theta = np.clip(theta, -0.45, 0.45)
        theta[0] = protected
        if best is not None:
            theta[1] = best["theta"]
        reports_by_model = evaluate_models(
            theta,
            worlds=args.worlds,
            impulse_rate_hz=args.impulse_rate_hz,
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
            protected = theta[winner].copy()
        elite = theta[order[:args.elite]]
        mean = 0.30 * mean + 0.70 * elite.mean(0)
        std = 0.45 * std + 0.55 * (elite.std(0) + 0.008)
        row = {
            "iteration": iteration,
            "iteration_best_by_model": winner_reports,
            "champion_by_model": best["reports"],
            "iteration_schedule": theta[winner].reshape(
                args.race_gates, args.knots, ACT_DIM
            ).tolist(),
            "champion_schedule": best["theta"].reshape(
                args.race_gates, args.knots, ACT_DIM
            ).tolist(),
        }
        history.append(row)
        candidate_pool.append((f"iteration_{iteration}", theta[winner].copy()))
        print(json.dumps(row), flush=True)
    assert best is not None
    # The in-generation champion is selected from a deliberately small batch
    # and is often a statistical winner rather than a robust controller.
    # Re-rank all generation winners on one larger common-random audit before
    # spending the locked 2,048/1,024-world evaluations on a finalist.
    selection_theta = np.stack([value for _, value in candidate_pool])
    # Bound memory and make the re-rank truly common-random: each generation
    # winner is evaluated separately with the identical seed.  A single
    # (iterations + 1) x selection_worlds tensor both wastes VRAM and gives
    # each candidate different runtime dropout/impulse draws.
    selection_reports_by_model = [[] for _ in ensembles]
    for candidate_index in range(len(selection_theta)):
        candidate_reports = evaluate_models(
            selection_theta[candidate_index:candidate_index + 1],
            worlds=max(1, int(args.selection_worlds)),
            impulse_rate_hz=args.impulse_rate_hz,
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
    selected = int(np.argmax(selection_scores))
    search_best = best
    best = {
        "score": float(selection_scores[selected]),
        "theta": selection_theta[selected].copy(),
        "reports": [
            reports[selected] for reports in selection_reports_by_model
        ],
        "label": candidate_pool[selected][0],
    }
    robust_by_model = evaluate_models(
        best["theta"][None],
        worlds=args.robust_worlds if not args.smoke else 32,
        impulse_rate_hz=0.0,
        seed=args.seed + 99173,
    )
    robust_by_model = [reports[0] for reports in robust_by_model]
    impulse_by_model = evaluate_models(
        best["theta"][None],
        worlds=args.impulse_worlds if not args.smoke else 16,
        impulse_rate_hz=args.impulse_rate_hz,
        seed=args.seed + 221137,
    )
    impulse_by_model = [reports[0] for reports in impulse_by_model]
    output = {
        "actor": str(args.actor),
        "knots": args.knots,
        "residual_scale": float(args.residual_scale),
        "multigate_estimator": bool(args.multigate_estimator),
        "residual_schedule": best["theta"].reshape(
            args.race_gates, args.knots, ACT_DIM
        ).tolist(),
        "robust_eval_by_model": robust_by_model,
        "impulse_eval_by_model": impulse_by_model,
        "search_champion_by_model": search_best["reports"],
        "selection": {
            "worlds": int(args.selection_worlds),
            "label": best["label"],
            "report_by_model": best["reports"],
            "candidates": len(candidate_pool),
        },
        "optimize_gates": sorted(optimize_gates),
        "history": history,
        "elapsed_s": time.time() - started,
        "world_model_datasets": [
            metadata.get("dataset") for metadata in ensemble_metadata
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
