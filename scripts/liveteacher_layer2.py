"""Layer-2 rollout parity: BatchedLiveTeacher under the audit stack.

Runs the frozen-config deployment controller (batched port, residuals
composed INTERNALLY -- env residual path disabled) through the same
environment recipe as the campaign's world-model audits (fastsim v3
plant + learned residual ensemble, audit DR bands, 10Hz noise era, FOV
vision, reloc events, optional impulses), and writes the per-world
comparator format Codex's paired-audit tooling consumes:

    world_id / finished / finish_time_s / failure_gate

Acceptance (vs the frozen baseline audit under paired seeds): finish
rate +-3 pts, median +-0.15 s, and small failure-histogram JS distance.

    python scripts/liveteacher_layer2.py \
        --config <frozen champion config> --ensemble <residual .pt> \
        --worlds 256 --seed 20270101 --out data/lineopt/l2_port.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import ACT_DIM, FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from aigp.fastsim.liveteacher import BatchedLiveTeacher  # noqa: E402
from scripts.fastsim_train_ppo import load_demo_states  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="frozen session config (authority)")
    ap.add_argument("--model", default=str(
        REPO / "data/fastsim_model_v3_live.json"))
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--speed-cap", type=float, default=12.5)
    ap.add_argument("--aleatoric-scale", type=float, default=1.5)
    ap.add_argument("--impulse-rate-hz", type=float, default=0.0)
    ap.add_argument("--max-episode-s", type=float, default=45.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    cfg_json = json.loads(Path(args.config).read_text())["args"]
    map_path = Path(cfg_json["map"])
    demo_path = Path(cfg_json["demo"])
    obstacles = REPO / "data/vq2_obstacles_inflated.json"

    n = args.worlds
    cfg = FastEnvConfig(
        n_envs=n,
        race_gates=17,
        random_start_frac=0.0,
        spawn_at_rest=True,
        max_episode_s=args.max_episode_s,
        speed_cap_mps=args.speed_cap,
        act_delay_steps_min=0,
        act_delay_steps_max=0,
        residual_scale=0.0,          # port composes residuals internally
        reloc_events=True,
        fov_vision=True,
        world_model_aleatoric_scale=args.aleatoric_scale,
        impulse_rate_hz=args.impulse_rate_hz,
        impulse_velocity_mps=(0.10, 0.55),
        impulse_vertical_scale=0.35,
        demo_corridor_m=2.0,
    )
    cfg.apply_vision10hz()
    cfg.fov_vision = True
    cfg.reloc_events = True
    cfg.dr_thrust = (0.97, 1.03)
    cfg.dr_rate_gain = (0.95, 1.05)
    cfg.dr_rate_tau = (0.90, 1.10)
    cfg.dr_drag = (0.20, 0.35)

    model = SurrogateModel.load(args.model)
    ensemble, _meta = ResidualEnsemble.load(args.ensemble, str(device))
    ensemble.eval()
    demo_states = load_demo_states(demo_path, map_path)
    backbone = BatchedLiveTeacher(args.config, n_envs=n,
                                  device=str(device))
    env = FastVQ2Env(model, map_path, demo_states=demo_states,
                     config=cfg, device=str(device),
                     obstacles_path=obstacles, backbone=backbone,
                     residual_ensemble=ensemble)

    zeros = torch.zeros(n, ACT_DIM, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    fail_gate = torch.full((n,), -1, dtype=torch.long, device=device)
    lap = torch.full((n,), float("nan"), device=device)
    alive = torch.zeros(n, device=device)
    dt = 1.0 / cfg.control_hz
    with torch.no_grad():
        for _ in range(int(cfg.max_episode_s * cfg.control_hz) + 1):
            _o, _r, done, info = env.step(zeros)
            live = ~(finished | failed)
            alive += live.float()
            newf = info["finished"] & live
            lap = torch.where(newf, alive * dt, lap)
            finished |= newf
            newfail = done & live & ~info["finished"]
            fail_gate = torch.where(newfail, info["target"], fail_gate)
            failed |= newfail
            if bool((finished | failed).all()):
                break

    fr = float(finished.float().mean())
    ok = finished
    summary = {
        "worlds": n,
        "seed": args.seed,
        "finish_rate": round(fr, 4),
        "median_s": round(float(lap[ok].median()), 3) if fr else None,
        "p90_s": round(float(lap[ok].quantile(0.9)), 3) if fr else None,
        "failure_gate_hist": {},
    }
    fg = fail_gate[failed].cpu().numpy()
    for gate in sorted(set(fg.tolist())):
        summary["failure_gate_hist"][str(int(gate))] = int(
            (fg == gate).sum())
    np.savez(
        args.out,
        world_id=np.arange(n, dtype=np.int64),
        finished=finished.cpu().numpy(),
        finish_time_s=lap.cpu().numpy(),
        failure_gate=fail_gate.cpu().numpy(),
        seed=np.int64(args.seed),
        config=str(args.config),
    )
    Path(str(args.out) + ".summary.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
