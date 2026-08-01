"""Convert a fastsim_line_opt winner into the SAC campaign's demo format.

Rebuilds the optimized reference from the saved CEM parameters, re-flies
it deterministically (no noise, no DR) in the corrected surrogate while
recording full state, then synthesizes the 53-dim observations exactly as
the live stack would (aigp.rl.vq2_features.build_observation with the
corrected-map geometry) and the campaign's reward:

    2.0 * d_progress + 25 * gate_pass - 0.8 * dt
    - 0.02 * |d_action|^2 (+600 finish)

Output keys mirror data/vq2_sac_clean_demo.npz: observation, action,
reward, next_observation, done, wall, gate_index, position, velocity,
sigma, source_episode.

    python scripts/build_lineopt_demo.py \
        --best data/lineopt/r1_cap8_best.npz --speed-cap 8 \
        --clearance 0.25 --out data/vq2_lineopt_demo_r1cap8.npz
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
from aigp.fastsim.lineopt import (  # noqa: E402
    DT, LineConfig, FlatRefController, N_GATES, build_reference,
    demo_states_from_ref, feedforward_actions, load_oriented_gates,
)
from aigp.rl.vq2_features import (  # noqa: E402
    build_observation, course_progress, load_gate_geometry,
)

PROGRESS_SCALE = 2.0
GATE_BONUS = 25.0
TIME_PENALTY = 0.8
SMOOTHNESS = 0.02
FINISH_BONUS = 600.0
SIGMA_M = 0.03
WALL_BASE = 1785600000.0


def clean_rollout(ref, ff, model, map_path, obstacles, speed_cap,
                  max_steps=2700):
    cfg = FastEnvConfig(n_envs=1, random_start_frac=0.0,
                        rate_gain_sign=1.0, speed_cap_mps=speed_cap,
                        demo_corridor_m=2.5)
    cfg.spawn_at_rest = True
    cfg.residual_scale = 0.0
    cfg.pos_noise_lo = cfg.pos_noise_hi = 0.0
    cfg.dr_thrust = (1.0, 1.0)
    cfg.dr_rate_gain = (1.0, 1.0)
    cfg.dr_rate_tau = (1.0, 1.0)
    cfg.dr_drag = (0.3, 0.3)
    cfg.act_delay_steps_min = 1
    cfg.act_delay_steps_max = 1
    cfg.max_episode_s = 89.0
    bb = FlatRefController(ref, ff, 1, device="cpu",
                           speed_cap=speed_cap, model=model)
    env = FastVQ2Env(model, map_path, demo_states=demo_states_from_ref(ref),
                     config=cfg, device="cpu",
                     obstacles_path=obstacles or None, backbone=bb)
    zeros = torch.zeros(1, ACT_DIM)
    rows = []
    for _ in range(max_steps):
        state = {
            "p": env.p[0].numpy().copy(),
            "v": env.v[0].numpy().copy(),
            "q": env.q[0].numpy().copy(),      # wxyz
            "w": env.w[0].numpy().copy(),      # body rates rad/s
            "target": int(env.target[0]),
        }
        _obs, _r, done, info = env.step(zeros)
        state["action"] = env.prev_action[0].numpy().copy()
        state["passed"] = int(info["passed"][0])
        state["finished"] = bool(info["finished"][0])
        rows.append(state)
        if bool(done[0]):
            if not state["finished"]:
                raise RuntimeError(
                    f"clean rollout FAILED at gate {state['target']}: "
                    f"{ {k: bool(info[k][0]) for k in ('hit', 'off', 'overspeed', 'timeout')} }"
                )
            break
    else:
        raise RuntimeError("clean rollout never terminated")
    return rows


def ideal_reference_rows(ref, ff):
    """Return the optimized line itself with feedforward-only actions.

    ``clean_rollout`` records the actions emitted by ``FlatRefController``.
    Those actions already contain that controller's position/velocity and
    attitude feedback.  They are appropriate demonstrations for an
    end-to-end policy, but not as the feedforward table for the live SAC
    reference controller, which adds its own tracking feedback.  This mode
    preserves the optimized geometry while avoiding double feedback.
    """
    rows = []
    target_raw = np.asarray(ref["gate"], int)
    target = np.clip(target_raw, 0, N_GATES - 1)
    for index in range(len(ref["pos"])):
        rows.append({
            "p": np.asarray(ref["pos"][index], np.float32),
            "v": np.asarray(ref["vel"][index], np.float32),
            "q": np.asarray(ref["quat_wxyz"][index], np.float32),
            "w": np.asarray(ref["rates"][index], np.float32),
            "target": int(target[index]),
            "action": np.asarray(ff[index], np.float32),
            "passed": int(
                index > 0 and target_raw[index] > target_raw[index - 1]
            ),
            "finished": index == len(ref["pos"]) - 1,
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True,
                    help="*_best.npz from fastsim_line_opt")
    ap.add_argument("--speed-cap", type=float, required=True)
    ap.add_argument("--clearance", type=float, required=True)
    ap.add_argument("--map", default=str(
        REPO / "data/vq2_runtime_map_g9g15fix.json"))
    ap.add_argument("--model", default=str(
        REPO / "data/fastsim_model_v2.json"))
    ap.add_argument("--obstacles", default=str(
        REPO / "data/vq2_obstacles_inflated.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--ideal-reference",
        action="store_true",
        help=(
            "write optimized reference states with feedforward-only actions "
            "for the live SAC reference controller; default records the "
            "surrogate FlatRefController's closed-loop actions"
        ),
    )
    args = ap.parse_args()

    best = np.load(args.best)
    theta = best["theta"]
    gate_pos, gate_R = load_oriented_gates(args.map)
    lcfg = LineConfig(speed_cap=args.speed_cap, clearance=args.clearance)
    off = theta[:N_GATES * 2]
    sc = theta[N_GATES * 2:]
    ref = build_reference(gate_pos, gate_R, off, sc, lcfg)
    model = SurrogateModel.load(args.model)
    ff = feedforward_actions(ref, model)
    rows = (
        ideal_reference_rows(ref, ff)
        if args.ideal_reference
        else clean_rollout(ref, ff, model, args.map, args.obstacles,
                           args.speed_cap)
    )
    n = len(rows)
    lap_s = n * DT
    print(f"clean lap FINISHED: {lap_s:.2f}s over {n} rows")

    gates_json = json.loads(Path(args.map).read_text())["gates"]
    geometry = load_gate_geometry(gates_json)
    spawn = rows[0]["p"]

    obs = np.zeros((n, 53), np.float32)
    prev_action = np.zeros(4, np.float32)
    for i, r in enumerate(rows):
        obs[i] = build_observation(
            position_world=r["p"],
            quat_wxyz=r["q"],
            velocity_world=r["v"],
            gyro_raw=-r["w"],          # build_observation negates gyro
            previous_action=prev_action,
            gate_index=r["target"],
            geometry=geometry,
            position_sigma_m=SIGMA_M,
        )
        prev_action = rows[i]["action"].astype(np.float32)

    action = np.stack([r["action"] for r in rows]).astype(np.float32)
    position = np.stack([r["p"] for r in rows]).astype(np.float64)
    velocity = np.stack([r["v"] for r in rows]).astype(np.float64)
    gate_index = np.asarray([r["target"] for r in rows], np.int64)
    done = np.zeros(n, np.float32)
    done[-1] = 1.0

    reward = np.zeros(n, np.float32)
    prev_prog = course_progress(spawn, 0, geometry, spawn)
    prev_a = np.zeros(4)
    for i, r in enumerate(rows):
        tgt = min(r["target"], N_GATES - 1)
        prog = course_progress(r["p"], tgt, geometry, spawn)
        d_prog = float(np.clip(prog - prev_prog, -0.5, 1.0))
        rw = (PROGRESS_SCALE * d_prog + GATE_BONUS * r["passed"]
              - TIME_PENALTY * DT
              - SMOOTHNESS * float(np.sum((r["action"] - prev_a) ** 2)))
        if r["finished"]:
            rw += FINISH_BONUS
        reward[i] = rw
        prev_prog = prog
        prev_a = r["action"]

    next_obs = np.vstack([obs[1:], obs[-1:]])
    wall = WALL_BASE + np.arange(n) * DT
    tag = args.tag or Path(args.best).stem
    source = (f"lineopt/{tag} surrogate-optimized line "
              f"(fastsim v2, map g9g15fix, cap {args.speed_cap})")

    np.savez(
        args.out,
        observation=obs, action=action, reward=reward,
        next_observation=next_obs, done=done, wall=wall,
        gate_index=gate_index, position=position, velocity=velocity,
        sigma=np.full(n, SIGMA_M, np.float64),
        source_episode=np.str_(source[:120]),
    )
    stats = {
        "rows": n,
        "lap_s": round(lap_s, 2),
        "reward_sum": round(float(reward.sum()), 1),
        "gates_passed": int(sum(r["passed"] for r in rows)),
        "peak_speed": round(float(np.linalg.norm(velocity, axis=1).max()), 2),
        "peak_rate": [round(float(np.abs(
            np.stack([r["w"] for r in rows])[:, k]).max()), 3)
            for k in range(3)],
        "peak_wire_thrust": round(float(
            (0.5 * (action[:, 3] + 1.0)).max()), 3),
        "out": str(args.out),
    }
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
