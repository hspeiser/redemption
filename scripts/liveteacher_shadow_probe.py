"""Shadow probe: where do the scalar and batched controllers diverge?

Drives paired worlds with the TRUSTED scalar deployed learner while the
batched port shadow-computes actions on the identical input stream
(belief position, velocity, attitude, prev action, target), each arm
evolving its own cursor state.  Records, per world, the first step
where the action difference exceeds small/large thresholds and the gate
where it happened -- separating a residual semantic difference (first
divergence large, systematic gate pattern) from chaotic decorrelation
(first divergence tiny, uniform).

    python scripts/liveteacher_shadow_probe.py --worlds 64 \
        --seed 20261217 --out data/lineopt/shadow_probe.json
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
from aigp.fastsim.worldmodel import (  # noqa: E402
    ResidualEnsemble, ResidualEnsemblePool,
)
from aigp.fastsim.liveteacher import BatchedLiveTeacher  # noqa: E402
from aigp.fastsim.liveteacher_scalar import (  # noqa: E402
    ScalarLiveTeacherAdapter,
)
from scripts.fastsim_train_ppo import load_demo_states  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(
        Path(r"D:\ai-gp\training\vq2_full17_fastprefix_abba_v1")
        / "20260802_191537" / "config.json"))
    ap.add_argument("--model", default=str(
        REPO / "data/fastsim_model_v3_live.json"))
    ap.add_argument("--ensemble", nargs="+", default=[
        r"D:\ai-gp\worldmodel\v25_allgate_v8\residual_ensemble_v25.pt",
        r"D:\ai-gp\worldmodel\v28_allgate_registry_flywheel"
        r"\residual_ensemble_v28.pt",
        r"D:\ai-gp\worldmodel\v30_allgate_registry_flywheel"
        r"\residual_ensemble_v30.pt",
    ])
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261217)
    ap.add_argument("--speed-cap", type=float, default=12.5)
    ap.add_argument("--aleatoric-scale", type=float, default=1.5)
    ap.add_argument("--max-episode-s", type=float, default=45.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    cfg_json = json.loads(Path(args.config).read_text())["args"]
    map_path = Path(cfg_json["map"])
    demo_path = Path(cfg_json["demo"])

    n = args.worlds
    cfg = FastEnvConfig(
        n_envs=n, race_gates=17, random_start_frac=0.0,
        spawn_at_rest=True, max_episode_s=args.max_episode_s,
        auto_reset=False, speed_cap_mps=args.speed_cap,
        act_delay_steps_min=0, act_delay_steps_max=0,
        residual_scale=0.0, reloc_events=True, fov_vision=True,
        world_model_aleatoric_scale=args.aleatoric_scale,
        demo_corridor_m=2.0,
    )
    cfg.apply_multigate10hz()
    cfg.dr_thrust = (0.97, 1.03)
    cfg.dr_rate_gain = (0.95, 1.05)
    cfg.dr_rate_tau = (0.90, 1.10)
    cfg.dr_drag = (0.20, 0.35)

    model = SurrogateModel.load(args.model)
    loaded = [ResidualEnsemble.load(p, "cpu") for p in args.ensemble]
    models = [item[0] for item in loaded]
    ensemble = (models[0] if len(models) == 1
                else ResidualEnsemblePool(models))
    ensemble.eval()
    demo_states = load_demo_states(demo_path, map_path)
    scalar = ScalarLiveTeacherAdapter(args.config, n_envs=n,
                                      device="cpu")
    port = BatchedLiveTeacher(args.config, n_envs=n, device="cpu")
    torch.manual_seed(args.seed)
    env = FastVQ2Env(model, map_path, demo_states=demo_states,
                     config=cfg, device="cpu",
                     obstacles_path=REPO / "data"
                     / "vq2_obstacles_inflated.json",
                     backbone=scalar,
                     residual_ensemble=ensemble)

    zeros = torch.zeros(n, ACT_DIM)
    first_small = np.full(n, -1)
    first_big = np.full(n, -1)
    gate_small = np.full(n, -1)
    diff_at_small = np.zeros(n)
    max_diff = np.zeros(n)
    done_mask = torch.zeros(n, dtype=torch.bool)
    step = 0
    with torch.no_grad():
        while step < int(cfg.max_episode_s * 30) and not bool(
                done_mask.all()):
            p_b = env.p + env.noise_pos
            v = env.v.clone()
            R = env._qmat(env.q)
            pa = env.prev_action.clone()
            tg = env.target.clone()
            shadow = port.action(p_b, v, R, prev_action=pa, target=tg)
            base = scalar.action(p_b, v, R, prev_action=pa, target=tg)
            # scalar.action advanced its state; env.step will call it
            # again -- give the env the precomputed action by restoring
            # scalar state? Instead: rewind is impossible; so compare
            # shadow vs base here, then let env.step recompute (state
            # advanced twice would corrupt).  To avoid double-advance,
            # drive the env manually below with `base` through the
            # residual==backbone trick: temporarily swap backbone out.
            diff = (shadow - base).abs().max(dim=1).values.numpy()
            live = (~done_mask).numpy()
            upd = live & (first_small < 0) & (diff > 1e-4)
            first_small[upd] = step
            gate_small[upd] = tg.numpy()[upd]
            diff_at_small[upd] = diff[upd]
            updb = live & (first_big < 0) & (diff > 1e-2)
            first_big[updb] = step
            max_diff = np.maximum(max_diff, np.where(live, diff, 0.0))
            # step env with the baseline action directly
            env.backbone = None
            _o, _r, done, info = env.step(base)
            env.backbone = scalar
            done_mask |= done
            step += 1

    out = {
        "worlds": n,
        "steps": step,
        "pct_worlds_diverged_1e-4": float((first_small >= 0).mean()),
        "pct_worlds_diverged_1e-2": float((first_big >= 0).mean()),
        "first_small_step_p50": float(np.median(
            first_small[first_small >= 0])) if (first_small >= 0).any()
        else None,
        "first_big_step_p50": float(np.median(
            first_big[first_big >= 0])) if (first_big >= 0).any()
        else None,
        "first_divergence_gate_hist": {
            str(g): int((gate_small == g).sum())
            for g in sorted(set(gate_small.tolist())) if g >= 0
        },
        "diff_at_first_p50": float(np.median(
            diff_at_small[first_small >= 0])) if (first_small >= 0).any()
        else None,
        "max_diff_p50": float(np.median(max_diff)),
        "max_diff_p95": float(np.percentile(max_diff, 95)),
    }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
