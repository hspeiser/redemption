"""Fly the fastsim-trained PPO policy in the REAL simulator, eval-only.

Construction mirrors scripts/train_vq2_sac_live.py exactly (v29+ process
isolation defaults that produced 2 stale episodes in 500). DO NOT run
while the live SAC trainer or a manual session owns the sim.

Dry run (no sim, no sockets -- verifies the policy pipeline end to end
against logged observations):

    python scripts/eval_vq2_ppo_live.py --dry-run

Real flight (sim must be free):

    python scripts/eval_vq2_ppo_live.py --episodes 5 \
        --policy data/models/vq2_ppo_conservative.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.sac import GaussianActor  # noqa: E402


def load_policy(path: str):
    art = torch.load(path, map_location="cpu", weights_only=False)
    actor = GaussianActor(art["obs_dim"], art["act_dim"])
    actor.load_state_dict(art["actor"])
    actor.eval()
    mean = art["obs_mean"].numpy().astype(np.float32)
    std = np.sqrt(art["obs_var"].numpy().astype(np.float32) + 1e-6)

    def act(observation: np.ndarray) -> np.ndarray:
        o = np.clip((observation - mean) / std, -8.0, 8.0)
        with torch.no_grad():
            a = actor.deterministic(
                torch.from_numpy(o[None].astype(np.float32))
            )[0].numpy()
        return a

    return act, art


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--policy",
        default=str(REPO / "data" / "models" / "vq2_ppo_conservative.pt"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--map", type=Path,
        default=REPO / "data" / "vq2_runtime_map_g9g15fix.json",
    )
    parser.add_argument(
        "--primary", type=Path,
        default=REPO / "data/models/gatenet_v13drought_best.pt",
    )
    parser.add_argument(
        "--refiner", type=Path,
        default=REPO / "data/models/gatenet_v10strict_ep0.pt",
    )
    parser.add_argument(
        "--crop", type=Path,
        default=REPO / "data/models/crop_gatenet_v11crop_ep13.pt",
    )
    parser.add_argument(
        "--proposal", type=Path,
        default=REPO / "data/models/gatepose_v5vq2b_best.pt",
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=REPO / "data/calib/calib.json",
    )
    parser.add_argument("--mav-port", type=int, default=14550)
    parser.add_argument("--camera-port", type=int, default=5600)
    # defaults = the current proven live stack (GPU dense at 10 Hz,
    # tight staleness), not the legacy CPU-era configuration
    parser.add_argument("--vision-hz", type=float, default=10.0)
    parser.add_argument("--vision-device", default="cuda")
    parser.add_argument("--max-vision-result-age", type=float,
                        default=0.30)
    parser.add_argument("--vision-process-isolation", default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--vision-worker-threads", type=int, default=8)
    parser.add_argument("--vision-worker-affinity", default="0x03FC")
    # A/B settled (flights 9 vs 12): the direct pin's ~1m belief jumps
    # destabilize an end-to-end policy (gates [1,1,7,4,2] without pins,
    # [1,1,0,0,0,0,0,0] with). Pins suit the reference-following SAC
    # controller, not policies that map obs->action directly. Default
    # OFF for policy flights; flag retained for experiments.
    parser.add_argument("--direct-position-pins", default=False,
                        action=argparse.BooleanOptionalAction)
    # hybrid mode: policy output is a bounded residual on the
    # reference-line backbone (same combination as residual training)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.3)
    parser.add_argument("--backbone-demo", type=Path,
                        default=REPO / "data/fastsim_demo_winner.npz")
    parser.add_argument("--backbone-episode", type=Path,
                        default=REPO / "data/vq2_sac_runs/gate3_nstep_"
                        "v60b/20260730_161745/episode_0002.npz")
    # line mode: fly a CEM-optimized reference (data/lineopt/*_best.npz)
    # with the FlatRefController geometric tracker. Pure tracker by
    # default; --line-residual-scale > 0 adds the policy as a bounded
    # residual on top.
    parser.add_argument("--line", type=Path, default=None)
    parser.add_argument("--line-speed-cap", type=float, default=8.0)
    parser.add_argument("--line-clearance", type=float, default=0.25)
    parser.add_argument("--line-model", type=Path,
                        default=REPO / "data/fastsim_model_v2.json")
    parser.add_argument("--line-residual-scale", type=float, default=0.0)
    parser.add_argument("--cpu-affinity", default="0xC000")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument(
        "--log", default=str(REPO / "data" / "ppo_live_eval.jsonl")
    )
    args = parser.parse_args()

    act, art = load_policy(args.policy)
    print(f"policy: {args.policy} (source iter {art.get('source_iter')})")

    if args.dry_run:
        demo = np.load(REPO / "data" / "vq2_sac_clean_demo.npz")
        obs = demo["observation"]
        acts = demo["action"]
        pred = np.stack([act(o) for o in obs])
        for axis, name in enumerate(("roll", "pitch", "yaw", "thrust")):
            c = np.corrcoef(pred[:, axis], acts[:, axis])[0, 1]
            sat = float(np.mean(np.abs(pred[:, axis]) > 0.98))
            print(f"  {name:6s}: corr(policy, Henry) {c:+.3f}  "
                  f"|a|>0.98 frac {sat:.3f}  "
                  f"mean|a| {np.abs(pred[:, axis]).mean():.3f}")
        finite = np.isfinite(pred).all()
        print(f"  all outputs finite: {finite}")
        print("DRY RUN OK -- pipeline verified, no sim contact")
        return 0

    # ---- real flight ----
    from scripts.train_vq2_sac_live import (
        acquire_single_instance_guard,
        set_process_cpu_affinity,
    )
    from aigp.mavlink_io import MavIO
    from aigp.rl.vq2_env import VQ2EnvConfig, VQ2LiveEnv
    from aigp.vision_io import VisionRX
    from aigp.vq2_live_localizer import LiveVQ2Localizer

    guard = acquire_single_instance_guard()
    set_process_cpu_affinity(args.cpu_affinity)
    torch.set_num_threads(args.torch_threads)

    mavlink = MavIO(port=args.mav_port)
    vision = VisionRX(port=args.camera_port)
    localizer = LiveVQ2Localizer(
        mavlink=mavlink,
        vision=vision,
        map_path=args.map,
        primary_checkpoint=args.primary,
        refine_checkpoint=args.refiner,
        crop_checkpoint=args.crop,
        proposal_checkpoint=args.proposal,
        calibration_path=args.calibration,
        async_interval_s=1.0 / args.vision_hz,
        max_async_result_age_s=args.max_vision_result_age,
        dense_device=args.vision_device,
        dense_process_isolation=args.vision_process_isolation,
        dense_worker_threads=args.vision_worker_threads,
        dense_worker_affinity=args.vision_worker_affinity,
        direct_position_pins=args.direct_position_pins,
    )
    # match the surrogate's training envelope: the 12 m/s default is the
    # SAC trainer's safety guard, not a sim or competition rule
    environment = VQ2LiveEnv(
        mavlink, localizer, VQ2EnvConfig(speed_cap_mps=16.0)
    )

    backbone = None
    residual_scale = args.residual_scale
    if args.line is not None:
        from aigp.fastsim.lineopt import (
            LineConfig, FlatRefController, N_GATES as LO_GATES,
            build_reference, feedforward_actions, load_oriented_gates,
        )
        from aigp.fastsim.sysid import SurrogateModel
        best = np.load(args.line)
        theta = best["theta"]
        gate_pos, gate_R = load_oriented_gates(args.map)
        lcfg = LineConfig(speed_cap=args.line_speed_cap,
                          clearance=args.line_clearance)
        ref = build_reference(gate_pos, gate_R, theta[:LO_GATES * 2],
                              theta[LO_GATES * 2:], lcfg)
        line_model = SurrogateModel.load(str(args.line_model))
        ff = feedforward_actions(ref, line_model)
        # trim0: the live plant hovers at ~0.25 wire vs the fitted
        # curve's 0.295 (flight 31/32) -- seed the integrator there
        backbone = FlatRefController(
            ref, ff, n_envs=1, device="cpu",
            speed_cap=args.line_speed_cap, model=line_model,
            trim0=-0.045, lead=6,
        )
        residual_scale = args.line_residual_scale
        print(f"LINE mode: {args.line.name}, {backbone.n_pts} rows, "
              f"planned lap {ref['planned_lap_s']:.1f}s, "
              f"cap {args.line_speed_cap}, "
              f"residual scale {residual_scale}")
    elif args.residual:
        from aigp.fastsim.refctl import load_winner_backbone
        backbone = load_winner_backbone(
            str(args.backbone_demo), str(args.backbone_episode),
            n_envs=1, device="cpu",
        )
        print(f"residual mode: backbone {backbone.n_pts} rows, "
              f"scale {args.residual_scale}")

    if backbone is not None:
        from scipy.spatial.transform import Rotation

        def combined_act(obs):
            st = localizer.state()
            qw, qx, qy, qz = st.quat_wxyz
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            base = backbone.action(
                torch.tensor(st.position, dtype=torch.float32)[None],
                torch.tensor(st.velocity, dtype=torch.float32)[None],
                torch.tensor(R, dtype=torch.float32)[None],
            )[0].numpy()
            if residual_scale <= 0.0:
                return np.clip(base, -1.0, 1.0)
            return np.clip(
                base + residual_scale * act(obs), -1.0, 1.0
            )

    def reset_with_retry(max_attempts: int = 6):
        # Anchor diagnostics are noisy right at spawn (GPU contention with
        # the sim, transient translation spread).  The SAC trainer survives
        # these by re-running the countdown; do the same instead of dying.
        for attempt in range(max_attempts):
            try:
                return environment.reset()
            except RuntimeError as error:
                message = str(error)
                transient = (
                    "spawn visual anchor is inconsistent" in message
                    or "spawn attitude does not match" in message
                    or "spawn gate anchor failed" in message
                )
                if not transient or attempt == max_attempts - 1:
                    raise
                print(f"RESET RETRY {attempt + 1}/{max_attempts}: {message}")
                time.sleep(1.0)

    results = []
    try:
        for ep_i in range(args.episodes):
            observation, info = reset_with_retry()
            print("ANCHOR", json.dumps({
                k: v for k, v in localizer.anchor_diagnostics.items()
                if k in ("map_yaw_delta_deg", "gate0_visual_error_m",
                         "translation_spread_p90_m", "pitch_deg",
                         "gate0_visual", "frames")
            }, default=str), flush=True)
            done = False
            steps = 0
            ep_reward = 0.0
            step_info = {}
            step_log = []
            frame_dir = os.environ.get("AIGP_SAVE_DEBUG_FRAMES")
            last_frame_save = 0.0
            obs_log = [] if os.environ.get("AIGP_SAVE_OBS") else None
            if backbone is not None:
                backbone.reset(torch.tensor([0]))
            while not done:
                if obs_log is not None:
                    obs_log.append(np.asarray(observation, np.float32))
                action = (
                    combined_act(observation) if backbone is not None
                    else act(observation)
                )
                observation, reward, term, trunc, step_info = \
                    environment.step(action)
                if backbone is not None and hasattr(backbone, "cur_gate"):
                    # race status is authoritative: belief-based plane
                    # tests can advance (or fail to advance) the gate
                    # cursor when the estimate drifts at the reversals
                    backbone.cur_gate[0] = int(step_info["target"])
                ep_reward += float(reward)
                steps += 1
                done = term or trunc
                if frame_dir and time.time() - last_frame_save > 0.1:
                    with localizer._debug_lock:
                        image = (
                            None if localizer._debug_image is None
                            else localizer._debug_image.copy()
                        )
                    if image is not None:
                        os.makedirs(frame_dir, exist_ok=True)
                        cv2.imwrite(
                            f"{frame_dir}/ep{ep_i}_s{steps:04d}_"
                            f"g{step_info['target']}.jpg", image,
                        )
                        last_frame_save = time.time()
                step_log.append({
                    "p": [round(v, 3) for v in step_info["position"]],
                    "tgt": int(step_info["target"]),
                    "spd": round(float(step_info["speed"]), 2),
                    "sig": round(float(step_info["position_sigma_m"]), 3),
                    "va": round(float(step_info["visual_age_s"]), 2),
                    # lag forensics: separate policy blindness (va grows
                    # while ca stays fresh) from pipeline lag (ca/ia/ss
                    # grow) -- flight-7 confound
                    "ca": round(float(step_info.get("camera_age_s", -1)), 3),
                    "ia": round(float(step_info.get("imu_age_s", -1)), 3),
                    "ss": round(float(step_info.get("sim_step_s", -1)), 3),
                    "a": [round(float(v), 3) for v in action],
                    "tilt": round(float(step_info.get("tilt_deg", -1)), 1),
                })
            with open(str(args.log) + f".ep{ep_i}.steps.json", "w") as fh:
                json.dump(step_log, fh)
            if obs_log is not None:
                np.savez_compressed(
                    str(args.log) + f".ep{ep_i}.obs.npz",
                    observation=np.asarray(obs_log, np.float32),
                )
            row = {
                "episode": ep_i,
                "reward": round(ep_reward, 1),
                "steps": steps,
                "gate_reached": int(step_info.get("target", -1)),
                "finished": bool(step_info.get("finished")),
                "failure": step_info.get("failure"),
                "sim_time_s": step_info.get("sim_time_s"),
                "update_counts": dict(localizer.update_counts),
            }
            results.append(row)
            print("EP", json.dumps(row))
            with open(args.log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            environment.park_after_episode()
            time.sleep(1.0)
    finally:
        environment.shutdown_to_spawn()
    n_finished = sum(1 for r in results if r["finished"])
    print(f"\n{n_finished}/{len(results)} finished; gates: "
          f"{[r['gate_reached'] for r in results]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
