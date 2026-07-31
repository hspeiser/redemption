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
    parser.add_argument("--reloc-events", action="store_true")
    parser.add_argument("--noise-era", choices=["3hz", "10hz"],
                        default="3hz",
                        help="estimator-noise calibration: 3hz = legacy "
                             "CPU-vision era, 10hz = GPU vision (v77 "
                             "measured)")
    parser.add_argument("--fov-vision", action="store_true",
                        help="vision fixes require a lookahead gate in "
                             "the camera frustum (flight-6/7 root cause)")
    parser.add_argument("--action-smoothness", type=float, default=None,
                        help="override jerk penalty (violent-flight fix: "
                             "0.12)")
    parser.add_argument("--act-delay-min", type=int, default=None,
                        help="minimum actuation delay steps (measured "
                             "plant lag ~1 step at 30Hz)")
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
    parser.add_argument("--demo-corridor", type=float, default=2.0)
    parser.add_argument("--speed-cap", type=float, default=16.0)
    parser.add_argument("--obstacles", default="")
    parser.add_argument("--resume", default="")
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
        demo = dict(np.load(args.demo_npz))
    else:
        demo = build_demo_states(Path(args.trace), Path(args.map))
        np.savez(run_dir / "demo_states.npz", **demo)
    print(f"demo states for random starts: {len(demo['pos'])}")

    cfg = FastEnvConfig(n_envs=args.n_envs, rate_gain_sign=float(sign),
                        reloc_events=args.reloc_events,
                        demo_corridor_m=args.demo_corridor,
                        speed_cap_mps=args.speed_cap)
    if args.noise_era == "10hz":
        cfg.apply_vision10hz()
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
    if args.spawn_at_rest:
        cfg.spawn_at_rest = True
    backbone = None
    if args.residual:
        from aigp.fastsim.refctl import load_winner_backbone
        cfg.residual_scale = args.residual_scale
        backbone = load_winner_backbone(
            args.demo_npz, args.bc_init, cfg.n_envs, device=str(device)
        )
        print(f"residual mode: backbone over {backbone.n_pts} ref rows, "
              f"scale {cfg.residual_scale}")
    env = FastVQ2Env(model, args.map, demo_states=demo, config=cfg,
                     device=str(device),
                     obstacles_path=args.obstacles or None,
                     backbone=backbone)

    actor = GaussianActor(OBS_DIM, ACT_DIM).to(device)
    critic = mlp(OBS_DIM, (512, 512, 256), 1).to(device)
    log_std = torch.nn.Parameter(
        torch.full((ACT_DIM,), -0.7, device=device)
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
        optimizer.load_state_dict(ck["optimizer"])
        obs_mean = ck["obs_mean"].to(device)
        obs_var = ck["obs_var"].to(device)
        obs_count = ck["obs_count"]
        start_iter = ck.get("iter", 0)
        print(f"resumed from {args.resume} @ iter {start_iter}")

    def normalize(o):
        return torch.clamp(
            (o - obs_mean) / torch.sqrt(obs_var + 1e-6), -8.0, 8.0
        )

    if args.bc_init and not args.resume and start_iter == 0 \
            and not args.residual:
        bc = np.load(args.bc_init)
        bc_obs = torch.tensor(bc["observation"], dtype=torch.float32,
                              device=device)
        bc_act = torch.tensor(bc["action"], dtype=torch.float32,
                              device=device)
        # seed the normalizer from the demonstration distribution
        obs_mean = bc_obs.mean(0)
        obs_var = bc_obs.var(0) + 1e-3
        obs_count = float(len(bc_obs))
        bc_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
        target_raw = torch.atanh(torch.clamp(bc_act, -0.999, 0.999))
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
    if args.bc_anchor > 0 and args.bc_init and not args.residual:
        bc = np.load(args.bc_init)
        anchor_obs = torch.tensor(bc["observation"], dtype=torch.float32,
                                  device=device)
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
        gate_max = 0
        for h in range(args.horizon):
            act, raw, logp, val = policy_sample(obs)
            O[h] = obs
            A_raw[h] = raw
            LP[h] = logp
            VL[h] = val
            obs, reward, done, info = env.step(act)
            RW[h] = reward
            DN[h] = done.float()
            pass_ct += int(info["passed"].sum())
            hit_ct += int(info["hit"].sum())
            fin_ct += int(info["finished"].sum())
            ep_ct += int(done.sum())
            spawn_done_ct += int(info["spawn_done"].sum())
            spawn_launch_ct += int(info["spawn_launched"].sum())
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
                "pi_loss": float(np.mean(pi_losses)),
                "v_loss": float(np.mean(v_losses)),
                "log_std": [round(float(v), 2) for v in log_std],
            }
            print(json.dumps(row))
            with open(log_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
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
