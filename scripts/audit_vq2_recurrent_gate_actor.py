"""Fresh-seed fastsim audit for a gate-scoped recurrent residual actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import HOLE_HALF, FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.live_teacher import LiveTeacherController  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from aigp.rl.sac import RecurrentActor  # noqa: E402
from scripts.fastsim_train_ppo import (  # noqa: E402
    load_demo_states,
    load_live_teacher_config,
    load_schedule_gate_positions,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_arm(args, ensemble, actor, obs_mean, obs_var, *, enabled, impulse_rate, seed):
    torch.manual_seed(seed)
    n = int(args.worlds)
    cfg = FastEnvConfig(
        n_envs=n,
        race_gates=args.focus_gate + 1,
        random_start_frac=0.0,
        spawn_at_rest=True,
        max_episode_s=args.max_episode_s,
        speed_cap_mps=16.0,
        act_delay_steps_min=0,
        act_delay_steps_max=0,
        residual_scale=args.residual_scale,
        reloc_events=True,
        fov_vision=True,
        world_model_aleatoric_scale=args.aleatoric_scale,
        impulse_rate_hz=impulse_rate,
        impulse_velocity_mps=(0.10, 0.55),
        impulse_vertical_scale=0.35,
        demo_corridor_m=2.0,
    )
    if args.multigate_estimator:
        cfg.apply_multigate10hz()
    else:
        cfg.apply_vision10hz()
    arrays, fixed = load_live_teacher_config(
        args.teacher_config, n, map_path=args.map
    )
    fixed.setdefault("schedule_gate_positions", load_schedule_gate_positions(args.map))
    controller_model = SurrogateModel.load(args.controller_model)
    backbone = LiveTeacherController(
        args.demo, n, device=args.device,
        rate_gain=np.asarray(controller_model.rate_gain),
        **arrays, **fixed,
    )
    demo = load_demo_states(args.demo, args.map)
    keep = np.asarray(demo["gate"]) <= args.focus_gate
    demo = {name: np.asarray(value)[keep] for name, value in demo.items()}
    env = FastVQ2Env(
        SurrogateModel.load(args.model), args.map,
        demo_states=demo, config=cfg, device=args.device,
        obstacles_path=args.obstacles, backbone=backbone,
        residual_ensemble=ensemble,
    )
    active = torch.ones(n, dtype=torch.bool, device=args.device)
    finished = torch.zeros_like(active)
    elapsed = torch.full((n,), float("nan"), device=args.device)
    failure_gate = torch.full((n,), -1, dtype=torch.long, device=args.device)
    clearance = torch.full((n,), 10.0, device=args.device)
    support_max = torch.zeros(n, device=args.device)
    disagreement_max = torch.zeros(n, device=args.device)
    action_sq = torch.zeros(n, device=args.device)
    active_steps = torch.zeros(n, device=args.device)
    hidden = None
    observation = env.observations()
    for _ in range(int(cfg.max_episode_s * cfg.control_hz) + 1):
        normalized = torch.clamp(
            (observation - obs_mean) / torch.sqrt(obs_var + 1e-6), -8.0, 8.0
        )
        actor_action, hidden = actor.step(normalized, hidden)
        gate_active = env.target == args.focus_gate
        action = actor_action * gate_active[:, None] if enabled else torch.zeros_like(actor_action)
        action_sq += action.square().mean(1) * active.float()
        observation, _reward, done, info = env.step(action)
        support_max = torch.maximum(support_max, info["world_model_support_z"] * active.float())
        disagreement_max = torch.maximum(disagreement_max, info["world_model_disagreement"] * active.float())
        active_steps += active.float()
        crossed = active & (info["cross_r"] >= 0.0)
        clearance = torch.where(
            crossed, torch.minimum(clearance, HOLE_HALF - info["cross_r"]), clearance
        )
        new_finish = active & info["finished"]
        elapsed = torch.where(new_finish, info["t_ep"], elapsed)
        finished |= new_finish
        failed = active & done & ~new_finish
        failure_gate = torch.where(failed, info["target"].long(), failure_gate)
        active &= ~done
        if not bool(active.any()):
            break
    times = elapsed[finished]
    failures = failure_gate[~finished]
    return {
        "finish_rate": float(finished.float().mean()),
        "median_s": float(times.median()) if len(times) else None,
        "p90_s": float(torch.quantile(times, 0.9)) if len(times) else None,
        "clearance_p10_m": (
            float(torch.quantile(clearance[finished], 0.1)) if bool(finished.any()) else -1.0
        ),
        "action_rms": float(torch.sqrt(action_sq / active_steps.clamp_min(1)).mean()),
        "world_model_support_p90": float(torch.quantile(support_max, 0.9)),
        "world_model_disagreement_p90": float(torch.quantile(disagreement_max, 0.9)),
        "failure_histogram": {
            str(gate): int((failures == gate).sum())
            for gate in range(args.focus_gate + 1)
            if bool((failures == gate).any())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--normalization-checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controller-model", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--focus-gate", type=int, default=5)
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026080205)
    parser.add_argument("--max-episode-s", type=float, default=18.0)
    parser.add_argument("--aleatoric-scale", type=float, default=1.5)
    parser.add_argument("--impulse-rate-hz", type=float, default=0.08)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--multigate-estimator", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    payload = torch.load(args.candidate, map_location=args.device, weights_only=False)
    metadata = payload["recurrent_residual"]
    if args.focus_gate not in {int(gate) for gate in metadata.get("gates", [])}:
        raise ValueError("candidate is not routed to the requested focus gate")
    actor = RecurrentActor(53, 4, hidden_dim=int(metadata["hidden_dim"])).to(args.device).eval()
    actor.load_state_dict(payload["recurrent_residual_actor"])
    normalization = torch.load(
        args.normalization_checkpoint, map_location=args.device, weights_only=False
    )
    obs_mean = torch.as_tensor(normalization["obs_mean"], device=args.device)
    obs_var = torch.as_tensor(normalization["obs_var"], device=args.device)
    reports = []
    for path in args.ensemble:
        ensemble, ensemble_metadata = ResidualEnsemble.load(path, args.device)
        ensemble.eval()
        arms = {}
        for condition, impulse in (("clean", 0.0), ("impulse", args.impulse_rate_hz)):
            condition_seed = args.seed + (1 if condition == "impulse" else 0)
            arms[condition] = {
                "protected": run_arm(
                    args, ensemble, actor, obs_mean, obs_var,
                    enabled=False, impulse_rate=impulse, seed=condition_seed,
                ),
                "candidate": run_arm(
                    args, ensemble, actor, obs_mean, obs_var,
                    enabled=True, impulse_rate=impulse, seed=condition_seed,
                ),
            }
        reports.append({
            "ensemble": str(path.resolve()),
            "ensemble_dataset": ensemble_metadata.get("dataset"),
            "arms": arms,
        })
    result = {
        "schema": 1,
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "candidate_metadata": metadata,
        "focus_gate": args.focus_gate,
        "worlds_per_arm": args.worlds,
        "seed": args.seed,
        "aleatoric_scale": args.aleatoric_scale,
        "impulse_rate_hz": args.impulse_rate_hz,
        "multigate_estimator": bool(args.multigate_estimator),
        "reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
