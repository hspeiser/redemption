"""PPO on the fast surrogate VQ2 track. Runs on one GPU; the policy is
the live stack's GaussianActor so checkpoints drop straight into the
deployment path.

    python scripts/fastsim_train_ppo.py --iters 3000 --n-envs 4096
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import (  # noqa: E402
    ACT_DIM,
    OBS_DIM,
    FastEnvConfig,
    FastVQ2Env,
    N_GATES,
)
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import (  # noqa: E402
    ResidualEnsemble,
    ResidualEnsemblePool,
    decode_observations,
)
from aigp.rl.sac import GaussianActor, mlp  # noqa: E402


def build_demo_states(trace_path: Path, map_path: Path) -> dict:
    """(pos, vel, quat, gate) rows along the clean lap for random starts."""
    from scipy.signal import savgol_filter

    tr = np.load(trace_path, allow_pickle=True)
    pos = np.asarray(tr["pos"], float)
    q = np.asarray(tr["quat"], float)          # wxyz
    t = np.asarray(tr["t"], float)
    hz = 1.0 / np.median(np.diff(t))
    vel = savgol_filter(pos, 15, 3, deriv=1, delta=1.0 / hz, axis=0)
    gates = json.loads(Path(map_path).read_text())["gates"]
    gpos = np.array([g["pos"] for g in gates[:N_GATES]])
    # per-row target gate: first gate whose center is still ahead along
    # the course (nearest-upcoming by arc order)
    target = np.zeros(len(pos), int)
    gi = 0
    for i in range(len(pos)):
        while gi < N_GATES - 1 and np.linalg.norm(
            pos[i] - gpos[gi]
        ) < 2.0:
            gi += 1
        target[i] = gi
        # advance when passing near a gate
        if gi < N_GATES - 1 and np.dot(
            pos[i] - gpos[gi],
            gpos[min(gi + 1, N_GATES - 1)] - gpos[gi],
        ) > 0 and np.linalg.norm(pos[i] - gpos[gi]) < 6.0:
            gi += 1
    speed = np.linalg.norm(vel, axis=1)
    keep = speed > 2.0
    return {
        "pos": pos[keep].astype(np.float32),
        "vel": vel[keep].astype(np.float32),
        "quat": q[keep].astype(np.float32),
        "gate": target[keep].astype(np.float32),
    }


def load_demo_states(path: Path, map_path: Path) -> dict:
    """Accept both old state-only NPZs and current 53-D live demos."""
    data = np.load(path)
    pos_key = "pos" if "pos" in data.files else "position"
    vel_key = "vel" if "vel" in data.files else "velocity"
    gate_key = "gate" if "gate" in data.files else "gate_index"
    pos = np.asarray(data[pos_key], np.float32)
    vel = np.asarray(data[vel_key], np.float32)
    gate = np.asarray(data[gate_key], np.float32)
    if "quat" in data.files:
        quat = np.asarray(data["quat"], np.float32)
    elif "observation" in data.files:
        from scipy.spatial.transform import Rotation

        gates = json.loads(map_path.read_text())["gates"]
        gate_pos = np.asarray([g["pos"] for g in gates], float)
        decoded = decode_observations(data["observation"], gate_pos)
        quat = np.roll(
            Rotation.from_matrix(decoded.rotation).as_quat(), 1, axis=1
        ).astype(np.float32)
    else:
        raise ValueError(
            f"{path} has no quat and no observation from which to decode it"
        )
    if not (len(pos) == len(vel) == len(quat) == len(gate)):
        raise ValueError("demo state arrays are not row-aligned")
    return {"pos": pos, "vel": vel, "quat": quat, "gate": gate}


def _gate_values(text: str, default: float, *, limit: int = 5) -> np.ndarray:
    values = np.full(limit, default, np.float32)
    for item in str(text or "").split(","):
        if not item.strip():
            continue
        gate, value = item.split(":", 1)
        gate = int(gate)
        if 0 <= gate < limit:
            values[gate] = float(value)
    return values


def _apply_gate_values(values: np.ndarray, text: str) -> np.ndarray:
    """Apply live ``gate:value`` overrides to an existing legacy baseline."""
    result = np.asarray(values, np.float32).copy()
    for item in str(text or "").split(","):
        if not item.strip():
            continue
        gate, value = item.split(":", 1)
        gate = int(gate)
        if 0 <= gate < len(result):
            result[gate] = float(value)
    return result


def _gate_set(text: str, *, limit: int = 5) -> tuple[int, ...]:
    return tuple(
        gate for gate in (
            int(value) for value in str(text or "").split(",")
            if value.strip()
        )
        if 0 <= gate < limit
    )


def _gate_value_keys(text: str, *, limit: int = 5) -> tuple[int, ...]:
    result = []
    for item in str(text or "").split(","):
        if not item.strip():
            continue
        gate = int(item.split(":", 1)[0])
        if 0 <= gate < limit:
            result.append(gate)
    return tuple(result)


def load_live_teacher_config(
    path: Path,
    n_envs: int,
    *,
    gate_count: int = 5,
    map_path: str | Path | None = None,
) -> tuple[dict, dict]:
    """Return per-world line knobs and fixed feedback options."""
    if not 1 <= gate_count <= 17:
        raise ValueError(f"gate_count must be in [1, 17], got {gate_count}")
    payload = json.loads(path.read_text())
    cfg = payload.get("args", payload)
    if "reference_action_leads" in cfg:
        leads = np.full(
            gate_count, float(cfg.get("reference_action_lead", 0)), np.float32
        )
        if gate_count > 4:
            leads[4] += float(cfg.get("gate4_action_lead", 0))
        leads = _apply_gate_values(
            leads, cfg.get("reference_action_leads", "")
        )
        thrust = _gate_values(
            cfg.get("reference_thrust_scales", ""),
            float(cfg.get("reference_thrust_scale", 1.0)), limit=gate_count,
        )
        velocity = _gate_values(
            cfg.get("reference_velocity_scales", ""),
            float(cfg.get("reference_velocity_scale", 1.0)), limit=gate_count,
        )
        blends = _gate_values(
            cfg.get("trajectory_blends", ""),
            float(cfg.get("trajectory_blend", 0.0)), limit=gate_count,
        )
        lateral_offsets = _gate_values(
            cfg.get("reference_lateral_offsets", ""), 0.0,
            limit=gate_count,
        )
        vertical_offsets = _gate_values(
            cfg.get("reference_vertical_offsets", ""), 0.0,
            limit=gate_count,
        )
        handoffs = _gate_values(
            cfg.get("predictive_handoff_distances", ""), 0.0,
            limit=gate_count,
        )
        rate_scales = _gate_values(
            cfg.get("reference_rate_scales", ""),
            float(cfg.get("reference_rate_scale", 1.0)),
            limit=gate_count,
        )
        sequential_speeds = _gate_values(
            cfg.get("reference_sequential_speeds", ""),
            float(cfg.get("reference_sequential_speed", 1.0)),
            limit=gate_count,
        )
        sequential_enabled = np.zeros(gate_count, bool)
        if str(cfg.get("reference_mode", "nearest")) == "sequential":
            sequential_enabled[:] = True
        for gate in _gate_value_keys(
            cfg.get("reference_sequential_speeds", ""), limit=gate_count
        ):
            sequential_enabled[gate] = True
        lateral_scales = np.ones(gate_count, np.float32)
        special_lateral_gate = int(cfg.get("special_lateral_gate", -1))
        if 0 <= special_lateral_gate < gate_count:
            lateral_scales[special_lateral_gate] = float(
                cfg.get("special_lateral_gain_scale", 1.0)
            )
        lateral_scales = _apply_gate_values(
            lateral_scales, cfg.get("lateral_gain_scales", "")
        )
        vertical_scales = np.ones(gate_count, np.float32)
        special_vertical_gate = int(cfg.get("special_vertical_gate", -1))
        if 0 <= special_vertical_gate < gate_count:
            vertical_scales[special_vertical_gate] = float(
                cfg.get("special_vertical_gain_scale", 1.0)
            )
        longitudinal_position = _gate_values(
            cfg.get("longitudinal_position_gains", ""),
            float(cfg.get("longitudinal_position_gain", 0.0)),
            limit=gate_count,
        )
        longitudinal_velocity = _gate_values(
            cfg.get("longitudinal_velocity_gains", ""),
            float(cfg.get("longitudinal_velocity_gain", 0.0)),
            limit=gate_count,
        )
        lateral_limits = _gate_values(
            cfg.get("lateral_feedback_limits", ""), 0.20,
            limit=gate_count,
        )
        lateral_biases = _gate_values(
            cfg.get("extra_lateral_biases", ""), 0.0, limit=gate_count,
        )
        vertical_biases = _gate_values(
            cfg.get("extra_vertical_biases", ""), 0.0, limit=gate_count
        )
        funnel_gates = _gate_set(cfg.get("gate_center_funnel_gates", ""))
        fixed = {
            "reference_rate_scales": rate_scales,
            "reference_sequential_speeds": sequential_speeds,
            "reference_sequential_enabled": sequential_enabled,
            "reference_max_advance": int(cfg.get(
                "reference_max_advance", 8)),
            "reference_max_retreat": int(cfg.get(
                "reference_max_retreat", 2)),
            "lateral_position_gain": float(cfg.get(
                "lateral_position_gain", 0.08)),
            "lateral_velocity_gain": float(cfg.get(
                "lateral_velocity_gain", 0.04)),
            "longitudinal_position_gains": longitudinal_position,
            "longitudinal_velocity_gains": longitudinal_velocity,
            "lateral_feedback_limits": lateral_limits,
            "vertical_position_gain": float(cfg.get(
                "vertical_position_gain", 0.30)),
            "vertical_velocity_gain": float(cfg.get(
                "vertical_velocity_gain", 0.10)),
            "vertical_gain_scales": vertical_scales,
            "lateral_bias_gates": _gate_set(
                cfg.get("lateral_bias_gates", ""), limit=17),
            "lateral_action_bias": float(cfg.get(
                "lateral_action_bias", 0.0)),
            "right_lateral_bias_gates": _gate_set(
                cfg.get("right_lateral_bias_gates", ""), limit=17),
            "right_lateral_action_bias": float(cfg.get(
                "right_lateral_action_bias", 0.0)),
            "extra_lateral_biases": lateral_biases,
            "vertical_bias_gates": _gate_set(
                cfg.get("vertical_bias_gates", ""), limit=17),
            "vertical_action_bias": float(cfg.get(
                "vertical_action_bias", 0.05)),
            "lateral_gain_scales": lateral_scales,
            "extra_vertical_biases": vertical_biases,
            "gate_center_funnel_gates": funnel_gates,
            "gate_center_funnel_distance": float(cfg.get(
                "gate_center_funnel_distance", 0.0)),
            "gate_center_funnel_full_distance": float(cfg.get(
                "gate_center_funnel_full_distance", 0.0)),
            "gate_center_funnel_strength": float(cfg.get(
                "gate_center_funnel_strength", 0.0)),
            "feedback_scale": float(cfg.get(
                "reference_feedback_scale", 1.0)),
        }
    else:
        leads = np.asarray(cfg.get("leads", [0] * gate_count), np.float32)
        thrust = np.asarray(
            cfg.get("thrust_scales", [1] * gate_count), np.float32
        )
        velocity = np.asarray(
            cfg.get("velocity_scales", [1] * gate_count), np.float32
        )
        blends = np.asarray(
            cfg.get("trajectory_blends", [0] * gate_count), np.float32
        )
        lateral_offsets = np.asarray(
            cfg.get("lateral_offsets_m", [0] * gate_count), np.float32
        )
        vertical_offsets = np.asarray(
            cfg.get("vertical_offsets_m", [0] * gate_count), np.float32
        )
        handoffs = np.asarray(
            cfg.get("predictive_handoff_distances", [0] * gate_count),
            np.float32,
        )
        fixed = {}
    arrays = {
        "action_leads": np.repeat(leads[None], n_envs, axis=0),
        "thrust_scales": np.repeat(thrust[None], n_envs, axis=0),
        "trajectory_velocity_scales": np.repeat(
            velocity[None], n_envs, axis=0
        ),
        "trajectory_blends": np.repeat(blends[None], n_envs, axis=0),
        "predictive_handoff_distances": np.repeat(
            handoffs[None], n_envs, axis=0
        ),
    }
    if np.any(lateral_offsets != 0.0) or np.any(vertical_offsets != 0.0):
        if map_path is None:
            raise ValueError(
                "reference geometry offsets require map_path so local "
                "gate axes can be converted to world coordinates"
            )
        from aigp.fastsim.lineopt import load_oriented_gates

        _gate_positions, gate_rotation = load_oriented_gates(map_path)
        offsets_world = (
            lateral_offsets[:, None] * gate_rotation[:gate_count, :, 0]
            + vertical_offsets[:, None] * gate_rotation[:gate_count, :, 2]
        ).astype(np.float32)
        arrays["reference_gate_offsets_world"] = np.repeat(
            offsets_world[None], n_envs, axis=0
        )
    return arrays, fixed


def load_schedule_gate_positions(map_path: str | Path) -> np.ndarray:
    """Load the 17 runtime-map gate centers used by live observations."""
    gates = json.loads(Path(map_path).read_text())["gates"]
    positions = np.asarray([gate["pos"] for gate in gates[:17]], np.float32)
    if positions.shape != (17, 3):
        raise ValueError(
            f"runtime map must provide 17 gate positions, got {positions.shape}"
        )
    return positions


@torch.no_grad()
def build_residual_demo_pairs(
    args: argparse.Namespace,
    model: SurrogateModel,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Expert observations and expert-minus-reference residual targets.

    Residual PPO previously used the demo only to normalize observations, so
    it never actually imitated the successful flight.  Re-evaluate the exact
    live reference controller on each recorded expert state and clone the
    bounded correction needed to reproduce the recorded command.
    """
    if not args.live_teacher_config:
        raise ValueError(
            "residual behavior cloning requires --live-teacher-config"
        )
    from aigp.fastsim.live_teacher import LiveTeacherController

    payload = np.load(args.bc_init)
    gate = np.asarray(payload["gate_index"], np.int64)
    keep = gate < args.race_gates
    source_rows = np.flatnonzero(keep)
    obs = torch.as_tensor(
        payload["observation"][keep], dtype=torch.float32, device=device
    )
    position = torch.as_tensor(
        payload["position"][keep], dtype=torch.float32, device=device
    )
    velocity = torch.as_tensor(
        payload["velocity"][keep], dtype=torch.float32, device=device
    )
    target_gate = torch.as_tensor(
        gate[keep], dtype=torch.long, device=device
    )
    expert = torch.as_tensor(
        payload["action"][keep], dtype=torch.float32, device=device
    )
    first = torch.nn.functional.normalize(obs[:, 21:24], dim=1)
    second = obs[:, 24:27]
    second = second - first * (first * second).sum(1, keepdim=True)
    second = torch.nn.functional.normalize(second, dim=1)
    third = torch.linalg.cross(first, second)
    rotation = torch.stack([first, second, third], dim=2)

    arrays, fixed = load_live_teacher_config(
        Path(args.live_teacher_config), len(obs), gate_count=args.race_gates,
        map_path=args.map,
    )
    controller_model = (
        SurrogateModel.load(args.controller_model)
        if args.controller_model else model
    )
    reference = LiveTeacherController(
        args.demo_npz, len(obs), device=str(device),
        schedule_gate_positions=load_schedule_gate_positions(args.map),
        rate_gain=np.asarray(controller_model.rate_gain),
        **arrays, **fixed,
    )
    reference.target.copy_(target_gate)
    reference.reference_gate.copy_(target_gate)
    reference.idx.copy_(torch.as_tensor(
        source_rows, dtype=torch.long, device=device
    ))
    reference.previous_action.copy_(obs[:, 30:34])
    base_action = reference.action(position, velocity, rotation)
    unbounded = (expert - base_action) / max(args.residual_scale, 1e-6)
    demo_schedule = torch.zeros_like(unbounded)
    if args.fixed_residual_schedule:
        schedule_payload = json.loads(Path(
            args.fixed_residual_schedule
        ).read_text())
        knots = int(schedule_payload["knots"])
        schedule = torch.as_tensor(
            schedule_payload["residual_schedule"],
            dtype=torch.float32,
            device=device,
        )
        expected = (args.race_gates, knots, ACT_DIM)
        if schedule.shape != expected:
            raise ValueError(
                f"fixed residual schedule shape {tuple(schedule.shape)}, "
                f"expected {expected}"
            )
        row_index = torch.as_tensor(
            source_rows, dtype=torch.long, device=device
        )
        lookup_gate = torch.clamp(target_gate, 0, args.race_gates - 1)
        start = reference.gate_start[lookup_gate]
        end = reference.gate_end[lookup_gate]
        phase = torch.clamp(
            (row_index - start).float()
            / torch.clamp((end - start).float(), min=1.0),
            0.0,
            1.0,
        )
        coordinate = phase * (knots - 1)
        left = torch.floor(coordinate).long()
        right = torch.clamp(left + 1, max=knots - 1)
        fraction = coordinate - left.float()
        demo_schedule = (
            schedule[lookup_gate, left]
            + fraction[:, None] * (
                schedule[lookup_gate, right]
                - schedule[lookup_gate, left]
            )
        )
        # The environment executes actor + fixed_schedule as the residual.
        # Therefore BC must teach only the remaining actor contribution.
        unbounded = unbounded - demo_schedule
    target = torch.clamp(unbounded, -0.999, 0.999)
    reconstructed_residual = torch.clamp(
        target + demo_schedule, -1.0, 1.0
    )
    stats = {
        "rows": int(len(obs)),
        "all_actions_in_support": float(
            (unbounded.abs() <= 1.0).all(1).float().mean()
        ),
        "base_action_mse": float(F.mse_loss(base_action, expert)),
        "clipped_action_mse": float(F.mse_loss(
            base_action + args.residual_scale * reconstructed_residual,
            expert,
        )),
        "fixed_schedule_accounted": bool(args.fixed_residual_schedule),
    }
    return obs, torch.atanh(target), stats


def pin_rate_sign(model: SurrogateModel, trace_path: Path,
                  episode_dir: Path) -> float:
    """Integrate the rate model open-loop under both gain signs against
    the lap's vision-corrected attitude; return the winning sign."""
    from aigp.fastsim.data import load_episode
    from scipy.spatial.transform import Rotation, Slerp

    ep = load_episode(episode_dir, hz=100.0, require_odometry=False)
    tr = np.load(trace_path, allow_pickle=True)
    q = np.asarray(tr["quat"], float)
    t_tr = np.asarray(tr["t"], float)
    # align: episode grid starts at imu t0; trace t is relative to same t0
    t0 = ep.t[0]
    quat_xyzw = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)
    inc = np.zeros(len(t_tr), bool)
    last = -np.inf
    for i, ti in enumerate(t_tr):
        if ti > last + 1e-6:
            inc[i] = True
            last = ti
    slerp = Slerp(t_tr[inc] + t0, Rotation.from_quat(quat_xyzw[inc]))
    K = np.abs(np.asarray(model.rate_gain))
    tau = np.asarray(model.rate_tau)
    errs = {}
    for sign in (+1.0, -1.0):
        # short segments: start from trace attitude, integrate 1.5 s
        seg_errs = []
        for t_start in np.arange(t_tr[inc][0] + t0 + 2,
                                 t_tr[inc][-1] + t0 - 3, 4.0):
            i0 = int(np.searchsorted(ep.t, t_start))
            i1 = i0 + 150
            if i1 >= len(ep.t):
                break
            R = slerp([ep.t[i0]])[0]
            w = ep.gyro[i0] * sign
            dt = 0.01
            for i in range(i0, i1):
                w = w + dt * (
                    sign * K * ep.cmd[i, :3]
                    * np.asarray([1.35, 1.34, 0.887]) - w
                ) / tau
                R = R * Rotation.from_rotvec(w * dt)
            true = slerp([ep.t[i1]])[0]
            seg_errs.append(np.degrees(
                (R.inv() * true).magnitude()
            ))
        errs[sign] = float(np.median(seg_errs))
    best = min(errs, key=errs.get)
    print(f"rate sign pin: +1 -> {errs[+1.0]:.1f} deg, "
          f"-1 -> {errs[-1.0]:.1f} deg  => {best:+.0f}")
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--n-envs", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=16384)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", default=str(REPO / "data" /
                                                 "fastsim_runs" / "ppo_v1"))
    parser.add_argument("--model", default=str(REPO / "data" /
                                               "fastsim_model.json"))
    parser.add_argument("--map", default=str(REPO / "data" /
                                             "vq2_map_final.json"))
    parser.add_argument("--trace", default=str(REPO / "data" /
                                               "vq2_trace_101_v5.npz"))
    parser.add_argument(
        "--episode-dir",
        default=r"C:\Users\henry\Downloads\AI-GP Simulator v1.0.3379"
                r"\ai-grand-prix\outputs\captures\rc_20260724_003101",
    )
    parser.add_argument("--rate-sign", type=float, default=0.0,
                        help="+1/-1 to skip the pin check")
    parser.add_argument("--demo-npz", default="",
                        help="precomputed demo-state npz (pos/vel/quat/"
                             "gate); skips trace+episode loading")
    parser.add_argument("--bc-init", default="",
                        help="live demo npz (observation/action) to "
                             "behavior-clone the actor before PPO")
    parser.add_argument("--bc-steps", type=int, default=3000)
    parser.add_argument("--bc-lr", type=float, default=1e-3)
    parser.add_argument(
        "--bc-on-resume", action="store_true",
        help="apply supervised demo fitting to a resumed actor before eval/PPO",
    )
    parser.add_argument("--reloc-events", action="store_true")
    parser.add_argument("--noise-era", choices=["3hz", "10hz"],
                        default="3hz",
                        help="estimator-noise calibration: 3hz = legacy "
                             "CPU-vision era, 10hz = GPU vision (v77 "
                             "measured)")
    parser.add_argument("--fov-vision", action="store_true",
                        help="vision fixes require a lookahead gate in "
                             "the camera frustum (flight-6/7 root cause)")
    parser.add_argument(
        "--multigate-vision", action="store_true",
        help=(
            "model the validated course-wide gate association path: any "
            "plausibly visible mapped gate can refresh the localizer"
        ),
    )
    parser.add_argument("--action-smoothness", type=float, default=None,
                        help="override jerk penalty (violent-flight fix: "
                             "0.12)")
    parser.add_argument("--act-delay-min", type=int, default=None,
                        help="minimum actuation delay steps (measured "
                             "plant lag ~1 step at 30Hz)")
    parser.add_argument("--act-delay-max", type=int, default=None,
                        help="maximum transport delay; use 0 when the learned "
                             "world residual already models command latency")
    parser.add_argument("--bc-anchor", type=float, default=0.0,
                        help="standing BC pull toward the demo during "
                             "PPO updates (0.03-0.10 typical)")
    parser.add_argument("--spawn-at-rest", action="store_true",
                        help="spawn starts at rest on the pitched pad "
                             "(matches live episode start)")
    parser.add_argument("--residual", action="store_true",
                        help="policy is a bounded residual on the "
                             "reference-line backbone (RefController); "
                             "backbone built from --demo-npz positions "
                             "+ --bc-init actions")
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--fixed-residual-schedule", default="",
        help=(
            "Optional residual-schedule result.json added to the actor output "
            "during rollout and evaluation. PPO then improves around the "
            "already validated scheduled controller instead of relearning it."
        ),
    )
    parser.add_argument("--active-residual-gates", type=int, nargs="*",
                        default=None,
                        help="empty/omitted means residual authority everywhere")
    parser.add_argument("--residual-gate-scales", type=float, nargs="*",
                        default=None,
                        help="optional per-gate multipliers on residual-scale")
    parser.add_argument("--live-teacher-config", default="",
                        help="use exact vectorized live teacher as backbone")
    parser.add_argument(
        "--batched-live-teacher-composition",
        action="store_true",
        help=(
            "Eval-only: use BatchedLiveTeacher as the complete deployed "
            "controller, including its routed actors, and disable the "
            "outer PPO residual. Intended for Layer-2 parity artifacts."
        ),
    )
    parser.add_argument("--controller-model", default="",
                        help="rate calibration used by live trajectory tracker")
    parser.add_argument("--residual-log-std", type=float, default=-1.5,
                        help="initial exploration log-std for residual PPO")
    parser.add_argument("--race-gates", type=int, default=N_GATES)
    parser.add_argument("--max-episode-s", type=float, default=45.0)
    parser.add_argument("--random-start-frac", type=float, default=0.6)
    parser.add_argument("--start-gate-weights", type=float, nargs="*",
                        default=None)
    parser.add_argument(
        "--world-model", nargs="+", default=None,
        help=(
            "one or more learned ResidualEnsemble checkpoints; multiple "
            "checkpoints are pooled with model-specific normalization"
        ),
    )
    parser.add_argument("--world-model-residual-scale", type=float,
                        default=1.0)
    parser.add_argument("--world-model-aleatoric-scale", type=float,
                        default=0.0)
    parser.add_argument("--world-model-mean", action="store_true",
                        help="use ensemble mean instead of one member/world")
    parser.add_argument("--impulse-rate-hz", type=float, default=0.0)
    parser.add_argument("--impulse-min-mps", type=float, default=0.10)
    parser.add_argument("--impulse-max-mps", type=float, default=0.55)
    parser.add_argument("--impulse-vertical-scale", type=float, default=0.35)
    parser.add_argument("--dr-thrust", type=float, nargs=2, default=None)
    parser.add_argument("--dr-rate-gain", type=float, nargs=2, default=None)
    parser.add_argument("--dr-rate-tau", type=float, nargs=2, default=None)
    parser.add_argument("--dr-drag", type=float, nargs=2, default=None)
    parser.add_argument("--time-penalty-per-s", type=float, default=None)
    parser.add_argument("--finish-time-target-s", type=float, default=0.0)
    parser.add_argument("--finish-time-bonus-per-s", type=float, default=0.0)
    parser.add_argument("--gate-time-targets", type=float, nargs="*",
                        default=None)
    parser.add_argument("--gate-time-bonus-per-s", type=float, default=0.0)
    parser.add_argument("--collision-penalty", type=float, default=None)
    parser.add_argument("--offtrack-penalty", type=float, default=None)
    parser.add_argument("--clearance-bonus", type=float, default=None)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-envs", type=int, default=0,
                        help="0 evaluates all training worlds")
    parser.add_argument("--eval-seed", type=int, default=20260801)
    parser.add_argument(
        "--eval-world-dump",
        default="",
        help=(
            "Optional NPZ path for per-world deterministic-eval outcomes. "
            "Used by the paired Layer-2 controller-parity oracle."
        ),
    )
    parser.add_argument(
        "--eval-impulse-rate-hz",
        type=float,
        default=0.0,
        help=(
            "disturbance rate used for deterministic checkpoint selection; "
            "default 0 preserves clean legacy evaluation"
        ),
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-gate-masks", default="",
        help=(
            "Semicolon-separated residual gate masks for eval-only, e.g. "
            "'all;none;0,1;2,3,4'. All uses every gate; none uses no actor."
        ),
    )
    parser.add_argument(
        "--eval-scale-profiles", default="",
        help=(
            "Semicolon-separated comma lists of per-gate residual "
            "multipliers for eval-only."
        ),
    )
    parser.add_argument("--demo-corridor", type=float, default=2.0)
    parser.add_argument(
        "--demo-tracking-penalty-per-s", type=float, default=0.0,
        help="dense reward penalty per squared metre-second outside the "
             "demo tracking free radius",
    )
    parser.add_argument("--demo-tracking-free-m", type=float, default=0.15)
    parser.add_argument(
        "--demo-crossing-bonus", type=float, default=0.0,
        help="bonus for crossing near the demonstrated gate-plane point",
    )
    parser.add_argument("--demo-crossing-radius-m", type=float, default=0.50)
    parser.add_argument("--speed-cap", type=float, default=16.0)
    parser.add_argument("--obstacles", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-reset-optimizer", action="store_true")
    parser.add_argument("--resume-log-std", type=float, default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(0)

    model = SurrogateModel.load(args.model)
    sign = args.rate_sign or pin_rate_sign(
        model, Path(args.trace), Path(args.episode_dir)
    )
    if args.demo_npz:
        demo = load_demo_states(Path(args.demo_npz), Path(args.map))
    else:
        demo = build_demo_states(Path(args.trace), Path(args.map))
        np.savez(run_dir / "demo_states.npz", **demo)
    # A short-course POC must not randomly initialize beyond its terminal
    # gate; those rows would create one-step fake finishes and poison PPO.
    keep = np.asarray(demo["gate"]) < args.race_gates
    demo = {key: np.asarray(value)[keep] for key, value in demo.items()}
    print(f"demo states for random starts: {len(demo['pos'])}")

    cfg = FastEnvConfig(n_envs=args.n_envs, rate_gain_sign=float(sign),
                        race_gates=args.race_gates,
                        max_episode_s=args.max_episode_s,
                        random_start_frac=args.random_start_frac,
                        residual_active_gates=tuple(
                            args.active_residual_gates or ()
                        ),
                        residual_gate_scales=tuple(
                            args.residual_gate_scales or ()
                        ),
                        demo_gate_weights=tuple(
                            args.start_gate_weights or ()
                        ),
                        reloc_events=args.reloc_events,
                        demo_corridor_m=args.demo_corridor,
                        demo_tracking_penalty_per_s=(
                            args.demo_tracking_penalty_per_s
                        ),
                        demo_tracking_free_m=args.demo_tracking_free_m,
                        demo_crossing_bonus=args.demo_crossing_bonus,
                        demo_crossing_radius_m=args.demo_crossing_radius_m,
                        speed_cap_mps=args.speed_cap,
                        world_model_residual_scale=(
                            args.world_model_residual_scale
                        ),
                        world_model_aleatoric_scale=(
                            args.world_model_aleatoric_scale
                        ),
                        world_model_use_mean=args.world_model_mean,
                        impulse_rate_hz=args.impulse_rate_hz,
                        impulse_velocity_mps=(
                            args.impulse_min_mps, args.impulse_max_mps
                        ),
                        impulse_vertical_scale=args.impulse_vertical_scale,
                        finish_time_target_s=args.finish_time_target_s,
                        finish_time_bonus_per_s=(
                            args.finish_time_bonus_per_s
                        ),
                        gate_time_targets_s=tuple(
                            args.gate_time_targets or ()
                        ),
                        gate_time_bonus_per_s=args.gate_time_bonus_per_s)
    if args.time_penalty_per_s is not None:
        cfg.time_penalty_per_s = args.time_penalty_per_s
    if args.collision_penalty is not None:
        cfg.collision_penalty = args.collision_penalty
    if args.offtrack_penalty is not None:
        cfg.offtrack_penalty = args.offtrack_penalty
    if args.clearance_bonus is not None:
        cfg.clearance_bonus = args.clearance_bonus
    if args.noise_era == "10hz":
        cfg.apply_vision10hz()
    if args.multigate_vision:
        cfg.apply_multigate10hz()
        cfg.fov_vision = True
    if args.fov_vision:
        cfg.fov_vision = True
        # true blind-drift rate (the era presets are time-averaged over
        # mostly-sighted flight; these apply only while coasting)
        cfg.coast_speed_diffuse = 0.005
        cfg.coast_speed_bias = 0.015
    if args.action_smoothness is not None:
        cfg.action_smoothness = args.action_smoothness
    if args.act_delay_min is not None:
        cfg.act_delay_steps_min = args.act_delay_min
    if args.act_delay_max is not None:
        cfg.act_delay_steps_max = args.act_delay_max
    if cfg.act_delay_steps_min > cfg.act_delay_steps_max:
        raise ValueError("act-delay-min cannot exceed act-delay-max")
    for name in ("dr_thrust", "dr_rate_gain", "dr_rate_tau", "dr_drag"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, tuple(float(v) for v in value))
    if args.spawn_at_rest:
        cfg.spawn_at_rest = True
    backbone = None
    if args.residual:
        cfg.residual_scale = args.residual_scale
        if args.batched_live_teacher_composition:
            if not args.eval_only:
                raise ValueError(
                    "--batched-live-teacher-composition is eval-only"
                )
            if not args.live_teacher_config:
                raise ValueError(
                    "--batched-live-teacher-composition requires "
                    "--live-teacher-config"
                )
            from aigp.fastsim.liveteacher import BatchedLiveTeacher

            backbone = BatchedLiveTeacher(
                args.live_teacher_config,
                cfg.n_envs,
                device=str(device),
                map_path=args.map,
                demo_path=args.demo_npz,
            )
            # This backbone already includes the primary/secondary actor
            # routing and residual blend from the frozen live config.
            cfg.residual_scale = 0.0
            print(
                "batched live-teacher composition: "
                f"{args.live_teacher_config}"
            )
        elif args.live_teacher_config:
            from aigp.fastsim.live_teacher import LiveTeacherController

            arrays, fixed = load_live_teacher_config(
                Path(args.live_teacher_config), cfg.n_envs,
                gate_count=args.race_gates,
                map_path=args.map,
            )
            controller_model = (
                SurrogateModel.load(args.controller_model)
                if args.controller_model else model
            )
            backbone = LiveTeacherController(
                args.demo_npz, cfg.n_envs, device=str(device),
                schedule_gate_positions=load_schedule_gate_positions(args.map),
                rate_gain=np.asarray(controller_model.rate_gain),
                **arrays, **fixed,
            )
            print(f"live-teacher backbone: {args.live_teacher_config}")
        else:
            from aigp.fastsim.refctl import load_winner_backbone

            backbone = load_winner_backbone(
                args.demo_npz, args.bc_init, cfg.n_envs, device=str(device)
            )
        print(f"residual mode: backbone over {backbone.n_pts} ref rows, "
              f"scale {cfg.residual_scale}")
    ensemble = None
    if args.world_model:
        loaded = [
            ResidualEnsemble.load(path, device=str(device))
            for path in args.world_model
        ]
        models = [item[0] for item in loaded]
        metadata = [item[1] for item in loaded]
        ensemble = (
            models[0] if len(models) == 1
            else ResidualEnsemblePool(models).to(device)
        )
        ensemble.eval()
        member_count = getattr(ensemble, "member_count", None)
        if member_count is None:
            member_count = len(ensemble.members)
        print(
            "learned world model pool: "
            f"{len(models)} checkpoint(s), {member_count} members, "
            f"aleatoric={cfg.world_model_aleatoric_scale}"
        )
        for path, row in zip(args.world_model, metadata):
            print(
                f"  {path}: dataset={row.get('dataset')}, "
                f"train_rows={row.get('train_rows')}"
            )
    env = FastVQ2Env(model, args.map, demo_states=demo, config=cfg,
                     device=str(device),
                     obstacles_path=args.obstacles or None,
                     backbone=backbone,
                     residual_ensemble=ensemble)

    fixed_schedule = None
    fixed_schedule_knots = 0
    if args.fixed_residual_schedule:
        schedule_payload = json.loads(Path(
            args.fixed_residual_schedule
        ).read_text())
        fixed_schedule_knots = int(schedule_payload["knots"])
        schedule_np = np.asarray(
            schedule_payload["residual_schedule"], np.float32
        )
        expected = (args.race_gates, fixed_schedule_knots, ACT_DIM)
        if schedule_np.shape != expected:
            raise ValueError(
                f"fixed residual schedule shape {schedule_np.shape}, "
                f"expected {expected}"
            )
        fixed_schedule = torch.as_tensor(
            schedule_np, dtype=torch.float32, device=device
        )
        print(
            f"fixed residual schedule: {args.fixed_residual_schedule} "
            f"({fixed_schedule_knots} knots)"
        )

    def add_fixed_schedule(action: torch.Tensor) -> torch.Tensor:
        if fixed_schedule is None:
            return action
        gate = torch.clamp(env.target, 0, args.race_gates - 1)
        start = env.cum_len[gate]
        length = env.seg_len[gate]
        phase = torch.clamp(
            (env.progress - start) / (length + 1e-9), 0.0, 1.0
        )
        coordinate = phase * (fixed_schedule_knots - 1)
        left = torch.floor(coordinate).long()
        right = torch.clamp(left + 1, max=fixed_schedule_knots - 1)
        fraction = coordinate - left.float()
        correction = (
            fixed_schedule[gate, left]
            + fraction[:, None] * (
                fixed_schedule[gate, right] - fixed_schedule[gate, left]
            )
        )
        return torch.clamp(action + correction, -1.0, 1.0)

    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    critic = mlp(OBS_DIM, (512, 512, 256), 1).to(device)
    log_std = torch.nn.Parameter(
        torch.full((ACT_DIM,), args.residual_log_std, device=device)
    )
    params = (
        list(actor.parameters()) + list(critic.parameters()) + [log_std]
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)
    obs_mean = torch.zeros(OBS_DIM, device=device)
    obs_var = torch.ones(OBS_DIM, device=device)
    obs_count = 1e-4
    start_iter = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device,
                        weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
        log_std.data = ck["log_std"].to(device)
        if not args.resume_reset_optimizer:
            optimizer.load_state_dict(ck["optimizer"])
        if args.resume_log_std is not None:
            log_std.data.fill_(args.resume_log_std)
        obs_mean = ck["obs_mean"].to(device)
        obs_var = ck["obs_var"].to(device)
        obs_count = ck["obs_count"]
        start_iter = ck.get("iter", 0)
        print(f"resumed from {args.resume} @ iter {start_iter}")
    elif args.residual and args.bc_init:
        # Residual actors begin at exactly zero, but they still need a stable
        # feature scale from iteration zero.  Letting the first rollout replace
        # an identity normalizer caused a policy-coordinate jump after every
        # cold start.
        seed = np.load(args.bc_init)
        seed_obs = torch.as_tensor(
            seed["observation"], dtype=torch.float32, device=device
        )
        obs_mean = seed_obs.mean(0)
        obs_var = seed_obs.var(0, unbiased=False) + 1e-3
        obs_count = float(len(seed_obs))
        print(f"residual normalizer seeded from {len(seed_obs)} live rows")

    def normalize(o):
        return torch.clamp(
            (o - obs_mean) / torch.sqrt(obs_var + 1e-6), -8.0, 8.0
        )

    residual_demo_obs = residual_demo_raw = None
    if args.residual and args.bc_init:
        residual_demo_obs, residual_demo_raw, residual_demo_stats = (
            build_residual_demo_pairs(args, model, device)
        )
        print("residual demo targets: " + json.dumps(residual_demo_stats))

    if args.bc_init and (
        (not args.resume and start_iter == 0) or args.bc_on_resume
    ):
        if args.residual:
            bc_obs = residual_demo_obs
            target_raw = residual_demo_raw
        else:
            bc = np.load(args.bc_init)
            bc_obs = torch.tensor(bc["observation"], dtype=torch.float32,
                                  device=device)
            bc_act = torch.tensor(bc["action"], dtype=torch.float32,
                                  device=device)
            target_raw = torch.atanh(torch.clamp(
                bc_act, -0.999, 0.999
            ))
        assert bc_obs is not None and target_raw is not None
        # Seed the normalizer from the same demonstration distribution used
        # by the supervised target.
        obs_mean = bc_obs.mean(0)
        obs_var = bc_obs.var(0, unbiased=False) + 1e-3
        obs_count = float(len(bc_obs))
        bc_opt = torch.optim.Adam(actor.parameters(), lr=args.bc_lr)
        for step in range(args.bc_steps):
            k = torch.randint(0, len(bc_obs), (256,), device=device)
            mean, _ = actor.distribution(normalize(bc_obs[k]))
            loss = F.mse_loss(mean, target_raw[k])
            bc_opt.zero_grad(set_to_none=True)
            loss.backward()
            bc_opt.step()
            if step % 1000 == 0:
                print(f"bc-init step {step}: loss {float(loss):.4f}")
        print(f"bc-init done ({len(bc_obs)} demo pairs)")

    # demo anchor (review find): PPO is free to forget the demonstrated
    # corridor and exploit surrogate quirks; keep a standing BC pull
    # toward the completed demonstration during every update.
    anchor_obs = anchor_raw = None
    if args.bc_anchor > 0 and args.bc_init:
        if args.residual:
            anchor_obs = residual_demo_obs
            anchor_raw = residual_demo_raw
        else:
            bc = np.load(args.bc_init)
            anchor_obs = torch.tensor(
                bc["observation"], dtype=torch.float32, device=device
            )
            anchor_raw = torch.atanh(torch.clamp(
                torch.tensor(bc["action"], dtype=torch.float32,
                             device=device), -0.999, 0.999,
            ))

    @torch.no_grad()
    def policy_sample(o):
        mean, _ = actor.distribution(normalize(o))
        std = log_std.exp()
        raw = mean + std * torch.randn_like(mean)
        act = torch.tanh(raw)
        logp = (
            -0.5 * (((raw - mean) / std) ** 2
                    + 2 * log_std + np.log(2 * np.pi))
            - torch.log(1 - act ** 2 + 1e-6)
        ).sum(-1)
        val = critic(normalize(o)).squeeze(-1)
        return act, raw, logp, val

    obs = env.observations()
    ep_return = torch.zeros(args.n_envs, device=device)
    stats = {"finish": 0, "pass": 0, "hit": 0, "episodes": 0,
             "best_gate": 0}
    t_start = time.time()
    log_path = run_dir / "train_log.jsonl"
    best_eval_score = -np.inf

    @torch.no_grad()
    def deterministic_eval() -> dict:
        """Full-start, randomized-world audit of the current actor."""
        nonlocal obs
        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda" else None
        )
        torch.manual_seed(args.eval_seed)
        old_start = cfg.random_start_frac
        old_impulse = cfg.impulse_rate_hz
        cfg.random_start_frac = 0.0
        cfg.impulse_rate_hz = max(0.0, args.eval_impulse_rate_hz)
        ids = torch.arange(args.n_envs, device=device)
        env.reset(ids)
        eo = env.observations()
        active = torch.ones(args.n_envs, dtype=torch.bool, device=device)
        finished = torch.zeros_like(active)
        elapsed = torch.full(
            (args.n_envs,), float("nan"), dtype=torch.float32, device=device
        )
        min_clearance = torch.full(
            (args.n_envs,), 10.0, dtype=torch.float32, device=device
        )
        failure_gate = torch.full(
            (args.n_envs,), -1, dtype=torch.long, device=device
        )
        demo_dist_sum = torch.zeros((), dtype=torch.float32, device=device)
        demo_dist_count = torch.zeros((), dtype=torch.float32, device=device)
        crossing_errors = [[] for _ in range(args.race_gates)]
        crossing_times = [[] for _ in range(args.race_gates)]
        for _ in range(int(cfg.max_episode_s * cfg.control_hz) + 1):
            action = actor.deterministic(normalize(eo))
            eo, _reward, done, info = env.step(add_fixed_schedule(action))
            demo_dist_sum += info["demo_dist"][active].sum()
            demo_dist_count += active.float().sum()
            crossed = info["cross_r"] >= 0.0
            for gate_idx in range(args.race_gates):
                selected = active & crossed & (info["cross_gate"] == gate_idx)
                if bool(selected.any()):
                    crossing_errors[gate_idx].append(
                        info["demo_cross_error"][selected].detach().cpu()
                    )
                    crossing_times[gate_idx].append(
                        info["t_ep"][selected].detach().cpu()
                    )
            min_clearance = torch.where(
                active & crossed,
                torch.minimum(min_clearance, 0.75 - info["cross_r"]),
                min_clearance,
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
        cfg.random_start_frac = old_start
        cfg.impulse_rate_hz = old_impulse
        env.reset(ids)
        obs = env.observations()
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        count = args.eval_envs if args.eval_envs > 0 else args.n_envs
        count = min(count, args.n_envs)
        f = finished[:count]
        times = elapsed[:count][f]
        failures = failure_gate[:count][~f]
        histogram = {
            str(g): int((failures == g).sum())
            for g in range(args.race_gates)
            if bool((failures == g).any())
        }
        crossing_summary = {}
        crossing_time_summary = {}
        for gate_idx, chunks in enumerate(crossing_errors):
            if chunks:
                values = torch.cat(chunks)
                crossing_summary[str(gate_idx)] = {
                    "median_m": float(values.median()),
                    "p90_m": float(torch.quantile(values, 0.9)),
                    "count": int(len(values)),
                }
                times_for_gate = torch.cat(crossing_times[gate_idx])
                crossing_time_summary[str(gate_idx)] = {
                    "median_s": float(times_for_gate.median()),
                    "p90_s": float(torch.quantile(times_for_gate, 0.9)),
                    "count": int(len(times_for_gate)),
                }
        if args.eval_world_dump:
            dump_path = Path(args.eval_world_dump)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dump_path,
                world_id=(
                    np.int64(args.eval_seed) * np.int64(1_000_000)
                    + np.arange(count, dtype=np.int64)
                ),
                finished=f.detach().cpu().numpy().astype(bool),
                finish_time_s=elapsed[:count].detach().cpu().numpy(),
                failure_gate=failure_gate[:count].detach().cpu().numpy(),
                min_clearance_m=(
                    min_clearance[:count].detach().cpu().numpy()
                ),
            )
        return {
            "eval_worlds": count,
            "eval_finish_rate": float(f.float().mean()),
            "eval_median_s": (
                float(times.median()) if len(times) else None
            ),
            "eval_p90_s": (
                float(torch.quantile(times, 0.9)) if len(times) else None
            ),
            "eval_clearance_p10_m": float(torch.quantile(
                min_clearance[:count][min_clearance[:count] < 9.0], 0.1
            )) if bool((min_clearance[:count] < 9.0).any()) else None,
            "eval_failure_gate": histogram,
            "eval_demo_dist_mean_m": float(
                demo_dist_sum / torch.clamp(demo_dist_count, min=1.0)
            ),
            "eval_demo_cross_error": crossing_summary,
            "eval_gate_time": crossing_time_summary,
        }

    if args.eval_only:
        masks = [args.eval_gate_masks] if not args.eval_gate_masks else (
            args.eval_gate_masks.split(";")
        )
        profiles = [""] if not args.eval_scale_profiles else (
            args.eval_scale_profiles.split(";")
        )
        evaluations = []
        original_mask = cfg.residual_active_gates
        original_scales = cfg.residual_gate_scales
        for profile in profiles:
            profile = profile.strip()
            cfg.residual_gate_scales = (
                tuple(float(v) for v in profile.split(",") if v.strip())
                if profile else original_scales
            )
            for label in masks:
                label = label.strip() or "configured"
                if label == "all":
                    cfg.residual_active_gates = ()
                elif label == "none":
                    cfg.residual_active_gates = (-1,)
                elif label != "configured":
                    cfg.residual_active_gates = tuple(
                        int(v) for v in label.split(",") if v.strip()
                    )
                evaluation = deterministic_eval()
                evaluation.update({
                    "iter": start_iter,
                    "kind": "deterministic_eval",
                    "residual_gate_mask": label,
                    "residual_gate_scales": list(cfg.residual_gate_scales),
                })
                evaluations.append(evaluation)
                print(json.dumps(evaluation), flush=True)
        cfg.residual_active_gates = original_mask
        cfg.residual_gate_scales = original_scales
        output = evaluations[0] if len(evaluations) == 1 else evaluations
        (run_dir / "eval_only.json").write_text(json.dumps(
            output, indent=2
        ))
        if args.bc_on_resume and args.bc_steps > 0:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "log_std": log_std.data,
                "optimizer": optimizer.state_dict(),
                "obs_mean": obs_mean,
                "obs_var": obs_var,
                "obs_count": obs_count,
                "iter": start_iter,
                "config": vars(args),
                "evaluation": output,
                "supervised_adaptation": {
                    "steps": int(args.bc_steps),
                    "learning_rate": float(args.bc_lr),
                    "demo": str(args.bc_init),
                },
            }, run_dir / "eval_actor.pt")
        return 0
    for it in range(start_iter, args.iters):
        O = torch.zeros(args.horizon, args.n_envs, OBS_DIM, device=device)
        A_raw = torch.zeros(args.horizon, args.n_envs, ACT_DIM,
                            device=device)
        LP = torch.zeros(args.horizon, args.n_envs, device=device)
        RW = torch.zeros(args.horizon, args.n_envs, device=device)
        DN = torch.zeros(args.horizon, args.n_envs, device=device)
        VL = torch.zeros(args.horizon + 1, args.n_envs, device=device)
        pass_ct = hit_ct = fin_ct = ep_ct = 0
        spawn_done_ct = spawn_launch_ct = 0
        impulse_ct = 0
        gate_max = 0
        for h in range(args.horizon):
            act, raw, logp, val = policy_sample(obs)
            O[h] = obs
            A_raw[h] = raw
            LP[h] = logp
            VL[h] = val
            obs, reward, done, info = env.step(add_fixed_schedule(act))
            RW[h] = reward
            DN[h] = done.float()
            pass_ct += int(info["passed"].sum())
            hit_ct += int(info["hit"].sum())
            fin_ct += int(info["finished"].sum())
            ep_ct += int(done.sum())
            spawn_done_ct += int(info["spawn_done"].sum())
            spawn_launch_ct += int(info["spawn_launched"].sum())
            impulse_ct += int(info.get("impulse", torch.zeros(
                (), device=device, dtype=torch.bool
            )).sum())
            gate_max = max(gate_max, int(info["target"].max()))
        with torch.no_grad():
            VL[args.horizon] = critic(normalize(obs)).squeeze(-1)

        adv = torch.zeros_like(RW)
        gae = torch.zeros(args.n_envs, device=device)
        for h in reversed(range(args.horizon)):
            delta_t = (
                RW[h] + args.gamma * VL[h + 1] * (1 - DN[h]) - VL[h]
            )
            gae = delta_t + args.gamma * args.lam * (1 - DN[h]) * gae
            adv[h] = gae
        ret = adv + VL[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        b_obs = O.reshape(-1, OBS_DIM)
        b_raw = A_raw.reshape(-1, ACT_DIM)
        b_lp = LP.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        n_batch = b_obs.shape[0]
        idx = torch.randperm(n_batch, device=device)
        pi_losses, v_losses = [], []
        for _ in range(args.epochs):
            idx = torch.randperm(n_batch, device=device)
            for s in range(0, n_batch, args.minibatch):
                mb = idx[s:s + args.minibatch]
                no = normalize(b_obs[mb])
                mean, _ = actor.distribution(no)
                std = log_std.exp()
                raw = b_raw[mb]
                act = torch.tanh(raw)
                logp = (
                    -0.5 * (((raw - mean) / std) ** 2
                            + 2 * log_std + np.log(2 * np.pi))
                    - torch.log(1 - act ** 2 + 1e-6)
                ).sum(-1)
                ratio = torch.exp(logp - b_lp[mb])
                surr = torch.minimum(
                    ratio * b_adv[mb],
                    torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                    * b_adv[mb],
                )
                entropy = (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum()
                pi_loss = -surr.mean() - args.entropy * entropy
                if anchor_obs is not None:
                    ka = torch.randint(0, len(anchor_obs), (512,),
                                       device=device)
                    a_mean, _ = actor.distribution(
                        normalize(anchor_obs[ka])
                    )
                    pi_loss = pi_loss + args.bc_anchor * F.mse_loss(
                        a_mean, anchor_raw[ka]
                    )
                v = critic(no).squeeze(-1)
                v_loss = F.mse_loss(v, b_ret[mb])
                optimizer.zero_grad(set_to_none=True)
                (pi_loss + 0.5 * v_loss).backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                with torch.no_grad():
                    # unbounded global std previously blew up to
                    # exp(5.8)~340 (saturated bang-bang exploration)
                    log_std.clamp_(-4.0, 0.3)
                pi_losses.append(float(pi_loss))
                v_losses.append(float(v_loss))

        # obs normalization update (batched Welford-ish) -- AFTER the
        # PPO epochs: updating between rollout and update made the
        # old/new log-probs use different normalizations, so the ratio
        # was not 1 even before the first gradient step (review find)
        flatO = O.reshape(-1, OBS_DIM)
        bmean = flatO.mean(0)
        bvar = flatO.var(0, unbiased=False)
        bn = flatO.shape[0]
        delta = bmean - obs_mean
        tot = obs_count + bn
        obs_mean = obs_mean + delta * bn / tot
        obs_var = (
            obs_var * (obs_count / tot) + bvar * (bn / tot)
            + delta ** 2 * obs_count * bn / tot ** 2
        )
        obs_count = tot

        stats["pass"] += pass_ct
        stats["hit"] += hit_ct
        stats["finish"] += fin_ct
        stats["episodes"] += ep_ct
        stats["best_gate"] = max(stats["best_gate"], gate_max)
        if it % 10 == 0 or fin_ct:
            sps = (
                (it - start_iter + 1) * args.horizon * args.n_envs
                / (time.time() - t_start)
            )
            row = {
                "iter": it,
                "steps_per_s": int(sps),
                "reward_mean": float(RW.mean()),
                "pass_per_ep": pass_ct / max(ep_ct, 1),
                "hit_frac": hit_ct / max(ep_ct, 1),
                "finish": fin_ct,
                "episodes": ep_ct,
                "gate_max": gate_max,
                "spawn_launch_rate": round(
                    spawn_launch_ct / max(spawn_done_ct, 1), 3
                ),
                "spawn_eps": spawn_done_ct,
                "impulses": impulse_ct,
                "pi_loss": float(np.mean(pi_losses)),
                "v_loss": float(np.mean(v_losses)),
                "log_std": [round(float(v), 2) for v in log_std],
            }
            print(json.dumps(row))
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        if args.eval_interval > 0 and (
            it % args.eval_interval == 0 or it == args.iters - 1
        ):
            evaluation = deterministic_eval()
            evaluation["iter"] = it
            evaluation["kind"] = "deterministic_eval"
            print(json.dumps(evaluation), flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(evaluation) + "\n")
            rate = evaluation["eval_finish_rate"]
            duration = evaluation["eval_median_s"] or 99.0
            checkpoint = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "log_std": log_std.data,
                "optimizer": optimizer.state_dict(),
                "obs_mean": obs_mean,
                "obs_var": obs_var,
                "obs_count": obs_count,
                "iter": it,
                "config": vars(args),
                "evaluation": evaluation,
            }
            if rate >= 0.90:
                # Keep every threshold-qualified candidate.  Later independent
                # model and live audits may rank them differently, and speed
                # discoveries must not disappear behind a slightly higher
                # same-seed reliability estimate.
                torch.save(
                    checkpoint, run_dir / f"qualifying_{it:04d}.pt"
                )
            # Reliability is a hard gate.  Once it is met, minimum time wins;
            # a tiny rate term only breaks exact time ties.  The old 100*rate
            # term valued +1% reliability like a full second and silently
            # discarded faster 90%+ policies.
            score = (
                1000.0 * rate if rate < 0.90
                else 1000.0 - duration + 0.01 * rate
            )
            if score > best_eval_score:
                best_eval_score = score
                torch.save(checkpoint, run_dir / "best.pt")
        if it % 100 == 0 or it == args.iters - 1:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "log_std": log_std.data,
                "optimizer": optimizer.state_dict(),
                "obs_mean": obs_mean,
                "obs_var": obs_var,
                "obs_count": obs_count,
                "iter": it,
                "config": vars(args),
            }, run_dir / "latest.pt")
            if fin_ct:
                torch.save(
                    torch.load(run_dir / "latest.pt",
                               weights_only=False),
                    run_dir / f"finish_{it}.pt",
                )
    print("TRAINING DONE", json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
