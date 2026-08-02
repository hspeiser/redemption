"""Audit a geometric line controller under a learned world ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import ACT_DIM, HOLE_HALF, FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.lineopt import (  # noqa: E402
    FlatRefController,
    LineConfig,
    build_reference,
    demo_states_from_ref,
    feedforward_actions,
    load_oriented_gates,
)
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402


def build_line(path: Path, map_path: Path, model: SurrogateModel) -> tuple:
    payload = np.load(path, allow_pickle=False)
    gate_pos, gate_rotation = load_oriented_gates(map_path)
    config = LineConfig(
        speed_cap=float(payload["speed_cap"]),
        clearance=float(payload["clearance"]),
        cap_margin=float(payload["cap_margin"]),
        a_lat_max=float(payload["a_lat_max"]),
        a_fwd=float(payload["a_fwd"]),
        a_brk=float(payload["a_brk"]),
        yaw_margin=float(payload["yaw_margin"]),
        normal_lead_m=float(payload["normal_lead_m"]),
        launch_speed=float(payload["launch_speed"]),
    )
    reference = build_reference(
        gate_pos,
        gate_rotation,
        payload["offsets"],
        payload["seg_scale"],
        config,
    )
    return reference, feedforward_actions(reference, model), config


@torch.no_grad()
def evaluate(args, model, ensemble, reference, feedforward, impulse_rate):
    torch.manual_seed(args.seed + (1 if impulse_rate else 0))
    cfg = FastEnvConfig(
        n_envs=args.worlds,
        race_gates=5,
        random_start_frac=0.0,
        spawn_at_rest=True,
        max_episode_s=14.0,
        speed_cap_mps=args.speed_cap,
        residual_scale=0.0,
        reloc_events=True,
        fov_vision=True,
        act_delay_steps_min=0,
        act_delay_steps_max=0,
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
    cfg.fov_vision = True
    cfg.dr_thrust = (0.97, 1.03)
    cfg.dr_rate_gain = (0.95, 1.05)
    cfg.dr_rate_tau = (0.90, 1.10)
    cfg.dr_drag = (0.20, 0.35)
    backbone = FlatRefController(
        reference,
        feedforward,
        args.worlds,
        device=args.device,
        speed_cap=args.speed_cap,
        model=model,
        trim0=args.trim,
        lead=args.lead,
        kp=args.kp,
        kv=args.kv,
        katt=args.katt,
    )
    env = FastVQ2Env(
        model,
        args.map,
        demo_states=demo_states_from_ref(reference),
        config=cfg,
        device=args.device,
        obstacles_path=args.obstacles,
        backbone=backbone,
        residual_ensemble=ensemble,
    )
    action = torch.zeros(args.worlds, ACT_DIM, device=args.device)
    active = torch.ones(args.worlds, dtype=torch.bool, device=args.device)
    finished = torch.zeros_like(active)
    elapsed = torch.full(
        (args.worlds,), float("nan"), dtype=torch.float32, device=args.device
    )
    failure_gate = torch.full(
        (args.worlds,), -1, dtype=torch.long, device=args.device
    )
    clearance = torch.full(
        (args.worlds,), 10.0, dtype=torch.float32, device=args.device
    )
    support_max = torch.zeros(args.worlds, device=args.device)
    disagreement_max = torch.zeros(args.worlds, device=args.device)
    for _ in range(int(cfg.max_episode_s * cfg.control_hz) + 1):
        _obs, _reward, done, info = env.step(action)
        support_max = torch.maximum(
            support_max, info["world_model_support_z"] * active.float()
        )
        disagreement_max = torch.maximum(
            disagreement_max,
            info["world_model_disagreement"] * active.float(),
        )
        crossed = active & (info["cross_r"] >= 0.0)
        clearance = torch.where(
            crossed,
            torch.minimum(clearance, HOLE_HALF - info["cross_r"]),
            clearance,
        )
        new_finish = active & info["finished"]
        finished |= new_finish
        elapsed = torch.where(new_finish, info["t_ep"], elapsed)
        failed = active & done & ~new_finish
        failure_gate = torch.where(
            failed, info["target"].long(), failure_gate
        )
        active &= ~done
        if not bool(active.any()):
            break
    times = elapsed[finished]
    failures = failure_gate[~finished]
    return {
        "finish_rate": float(finished.float().mean()),
        "median_s": float(times.median()) if len(times) else None,
        "p90_s": float(torch.quantile(times, 0.9)) if len(times) else None,
        "clearance_p10_m": float(torch.quantile(
            clearance[finished], 0.1
        )) if bool(finished.any()) else None,
        "support_p90": float(torch.quantile(support_max, 0.9)),
        "disagreement_p90": float(torch.quantile(disagreement_max, 0.9)),
        "failure_histogram": {
            str(gate): int((failures == gate).sum())
            for gate in range(5)
            if bool((failures == gate).any())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--obstacles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20268301)
    parser.add_argument("--aleatoric-scale", type=float, default=1.5)
    parser.add_argument("--impulse-rate-hz", type=float, default=0.08)
    parser.add_argument("--trim", type=float, default=-0.045)
    parser.add_argument("--lead", type=int, default=6)
    parser.add_argument("--kp", type=float, default=2.0)
    parser.add_argument("--kv", type=float, default=2.8)
    parser.add_argument("--katt", type=float, default=5.0)
    parser.add_argument("--speed-cap", type=float, default=15.5)
    parser.add_argument("--multigate-estimator", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = SurrogateModel.load(args.model)
    ensemble, metadata = ResidualEnsemble.load(args.ensemble, args.device)
    ensemble.eval()
    reference, feedforward, line_config = build_line(
        args.line, args.map, model
    )
    result = {
        "line": str(args.line),
        "world_model_dataset": metadata.get("dataset"),
        "worlds_per_arm": args.worlds,
        "planned_lap_s": float(reference["planned_lap_s"]),
        "line_speed_cap": float(line_config.speed_cap),
        "controller": {
            "trim": args.trim,
            "lead": args.lead,
            "kp": args.kp,
            "kv": args.kv,
            "katt": args.katt,
        },
        "multigate_estimator": bool(args.multigate_estimator),
        "clean": evaluate(args, model, ensemble, reference, feedforward, 0.0),
        "impulse": evaluate(
            args, model, ensemble, reference, feedforward,
            args.impulse_rate_hz,
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
