"""CEM search for a faster VQ2 reference line in the corrected surrogate.

Each candidate = per-gate hole-plane crossing offsets + per-segment speed
scales.  Candidates become 30 Hz flatness references (lineopt.py) flown by
FlatRefController inside FastVQ2Env under the full measured noise stack
(10 Hz vision era, FOV-coupled fixes, reloc events, actuation delay,
at-rest spawn, obstacle cylinders, domain randomization).  Score = finish
rate first, lap time second.

    python scripts/fastsim_line_opt.py --speed-cap 8 --clearance 0.25 \
        --out-prefix data/lineopt/r1_cap8
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

from aigp.fastsim.env import FastEnvConfig, FastVQ2Env, ACT_DIM, HOLE_HALF  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.lineopt import (  # noqa: E402
    BatchedFlatRefController, LineConfig, FlatRefController, N_GATES,
    build_reference, demo_states_from_pop, demo_states_from_ref,
    feasibility, feedforward_actions, load_oriented_gates,
)


def make_env_cfg(args) -> FastEnvConfig:
    cfg = FastEnvConfig(
        n_envs=args.n_envs,
        race_gates=args.race_gates,
        random_start_frac=0.0,
        rate_gain_sign=1.0,
        reloc_events=True,
        demo_corridor_m=args.demo_corridor,
        speed_cap_mps=args.speed_cap,
    )
    cfg.apply_vision10hz()
    cfg.fov_vision = True
    cfg.coast_speed_diffuse = 0.005
    cfg.coast_speed_bias = 0.015
    cfg.act_delay_steps_min = 1
    cfg.spawn_at_rest = True
    cfg.residual_scale = 0.0        # pure backbone, no policy on top
    cfg.max_episode_s = 80.0        # don't truncate slow-but-alive laps
    return cfg


def evaluate(ref, ff, model, args, device, max_steps=2400,
             clean: bool = False):
    """Fly the reference; return metrics dict (and rollout when clean)."""
    cfg = make_env_cfg(args)
    if clean:
        cfg.n_envs = 1
        cfg.pos_noise_lo = cfg.pos_noise_hi = 0.0
        cfg.reloc_events = False
        cfg.fov_vision = False
        cfg.dr_thrust = (1.0, 1.0)
        cfg.dr_rate_gain = (1.0, 1.0)
        cfg.dr_rate_tau = (1.0, 1.0)
        cfg.dr_drag = (0.3, 0.3)
        cfg.act_delay_steps_max = 1
    n = cfg.n_envs
    backbone = FlatRefController(ref, ff, n, device=str(device),
                                 speed_cap=args.speed_cap, model=model)
    env = FastVQ2Env(model, args.map, demo_states=demo_states_from_ref(ref),
                     config=cfg, device=str(device),
                     obstacles_path=args.obstacles or None,
                     backbone=backbone)
    zeros = torch.zeros(n, ACT_DIM, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    fail_gate = torch.full((n,), -1, dtype=torch.long, device=device)
    fail_kind = torch.zeros(n, dtype=torch.long, device=device)
    lap = torch.zeros(n, device=device)
    alive_steps = torch.zeros(n, device=device)
    clear_min = torch.full((n, N_GATES), np.nan, device=device)
    peak_speed = torch.zeros(n, device=device)
    trace = {k: [] for k in
             ("p", "v", "q", "act", "target", "t")} if clean else None
    dt = 1.0 / cfg.control_hz
    with torch.no_grad():
        for _ in range(max_steps):
            if clean:
                trace["p"].append(env.p[0].cpu().numpy().copy())
                trace["v"].append(env.v[0].cpu().numpy().copy())
                trace["q"].append(env.q[0].cpu().numpy().copy())
                trace["target"].append(int(env.target[0]))
                trace["t"].append(float(env.t_ep[0]))
            obs, _r, done, info = env.step(zeros)
            if clean:
                trace["act"].append(
                    env.prev_action[0].cpu().numpy().copy())
            live = ~(finished | failed)
            alive_steps += live.float()
            peak_speed = torch.maximum(
                peak_speed, info["speed"] * live.float())
            ok = info["passed"] & live
            if ok.any():
                gid = torch.clamp(info["target"] - 1, 0, N_GATES - 1)
                margin = HOLE_HALF - info["cross_r"]
                cur = clear_min[torch.arange(n, device=device), gid]
                upd = torch.where(torch.isnan(cur), margin,
                                  torch.minimum(cur, margin))
                clear_min[torch.arange(n, device=device), gid] = \
                    torch.where(ok, upd, cur)
            newly_fin = info["finished"] & live
            lap = torch.where(newly_fin, alive_steps * dt, lap)
            finished |= newly_fin
            newly_fail = done & live & ~info["finished"]
            fail_gate = torch.where(newly_fail, info["target"], fail_gate)
            for code, key in ((1, "hit"), (2, "off"), (3, "overspeed"),
                              (4, "timeout")):
                fail_kind = torch.where(
                    newly_fail & info[key],
                    torch.full_like(fail_kind, code), fail_kind)
            failed |= newly_fail
            if bool((finished | failed).all()):
                break

    fin_rate = float(finished.float().mean())
    out = {
        "finish_rate": fin_rate,
        "n_envs": n,
        "lap_median": float(lap[finished].median()) if fin_rate else None,
        "lap_best": float(lap[finished].min()) if fin_rate else None,
        "lap_p90": float(lap[finished].quantile(0.9)) if fin_rate else None,
        "peak_speed": float(peak_speed.max()),
    }
    cm = clear_min.cpu().numpy()
    with np.errstate(all="ignore"):
        out["clearance_min_by_gate"] = [
            round(float(np.nanmin(cm[:, g])), 3)
            if np.isfinite(cm[:, g]).any() else None
            for g in range(N_GATES)
        ]
        out["clearance_p05_by_gate"] = [
            round(float(np.nanquantile(cm[:, g], 0.05)), 3)
            if np.isfinite(cm[:, g]).any() else None
            for g in range(N_GATES)
        ]
    if int(failed.sum()):
        hist = {}
        names = {0: "?", 1: "hit", 2: "off", 3: "overspeed", 4: "timeout"}
        for g, k in zip(fail_gate[failed].cpu().numpy(),
                        fail_kind[failed].cpu().numpy()):
            key = f"g{g}/{names.get(int(k), '?')}"
            hist[key] = hist.get(key, 0) + 1
        out["failures"] = dict(
            sorted(hist.items(), key=lambda kv: -kv[1])[:10])
    if clean:
        out["trace"] = {k: np.asarray(v) for k, v in trace.items()}
    return out


def evaluate_population(refs, model, args, device, max_steps=2400):
    """Fly ALL candidates in one batched env; per-candidate metrics."""
    C = len(refs)
    per = args.n_envs
    cfg = make_env_cfg(args)
    cfg.n_envs = C * per
    n = cfg.n_envs
    backbone = BatchedFlatRefController(
        refs, per, device=str(device), speed_cap=args.speed_cap,
        model=model)
    env = FastVQ2Env(model, args.map, demo_states=demo_states_from_pop(refs),
                     config=cfg, device=str(device),
                     obstacles_path=args.obstacles or None,
                     backbone=backbone)
    zeros = torch.zeros(n, ACT_DIM, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    failed = torch.zeros(n, dtype=torch.bool, device=device)
    lap = torch.zeros(n, device=device)
    alive_steps = torch.zeros(n, device=device)
    clear_min = torch.full((n, N_GATES), np.nan, device=device)
    dt = 1.0 / cfg.control_hz
    ar = torch.arange(n, device=device)
    with torch.no_grad():
        for _ in range(max_steps):
            _obs, _r, done, info = env.step(zeros)
            live = ~(finished | failed)
            alive_steps += live.float()
            ok = info["passed"] & live
            if ok.any():
                gid = torch.clamp(info["target"] - 1, 0, N_GATES - 1)
                margin = 0.75 - info["cross_r"]
                cur = clear_min[ar, gid]
                upd = torch.where(torch.isnan(cur), margin,
                                  torch.minimum(cur, margin))
                clear_min[ar, gid] = torch.where(ok, upd, cur)
            newly_fin = info["finished"] & live
            lap = torch.where(newly_fin, alive_steps * dt, lap)
            finished |= newly_fin
            failed |= done & live & ~info["finished"]
            if float((finished | failed).float().mean()) > 0.999:
                break
    finC = finished.view(C, per)
    lapC = lap.view(C, per)
    out = []
    cmin = clear_min.view(C, per, N_GATES).cpu().numpy()
    for ci in range(C):
        fr = float(finC[ci].float().mean())
        m = {"finish_rate": fr, "n_envs": per,
             "lap_median": float(lapC[ci][finC[ci]].median())
             if fr else None,
             "lap_best": float(lapC[ci][finC[ci]].min()) if fr else None}
        with np.errstate(all="ignore"):
            m["clearance_p05_by_gate"] = [
                round(float(np.nanquantile(cmin[ci, :, g], 0.05)), 3)
                if np.isfinite(cmin[ci, :, g]).any() else None
                for g in range(N_GATES)]
        out.append(m)
    return out


def score(metrics, ref) -> float:
    fr = metrics["finish_rate"]
    lap = metrics["lap_median"] if metrics["lap_median"] else 200.0
    feas = feasibility(ref)
    pen = 30.0 * max(0.0, feas["rate_sat_frac"] - 0.05)
    return 1000.0 * fr - lap - pen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=str(
        REPO / "data/vq2_runtime_map_g9g15fix.json"))
    ap.add_argument("--model", default=str(
        REPO / "data/fastsim_model_v2.json"))
    ap.add_argument("--obstacles", default=str(
        REPO / "data/vq2_obstacles_inflated.json"))
    ap.add_argument("--speed-cap", type=float, default=8.0)
    ap.add_argument(
        "--race-gates", type=int, default=N_GATES,
        help="Finish and score after this many gates; geometry after it is frozen.",
    )
    ap.add_argument("--clearance", type=float, default=0.25)
    ap.add_argument("--cap-margin", type=float, default=0.90)
    ap.add_argument("--a-lat-max", type=float, default=9.0)
    ap.add_argument("--a-fwd", type=float, default=5.0)
    ap.add_argument("--a-brk", type=float, default=6.0)
    ap.add_argument("--yaw-margin", type=float, default=0.70)
    ap.add_argument("--normal-lead-m", type=float, default=1.3)
    ap.add_argument("--launch-speed", type=float, default=0.5)
    ap.add_argument("--initial-speed-scale", type=float, default=0.80)
    ap.add_argument("--demo-corridor", type=float, default=2.0)
    ap.add_argument("--n-envs", type=int, default=256)
    ap.add_argument("--pop", type=int, default=28)
    ap.add_argument("--elite", type=int, default=7)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument(
        "--init-best",
        default=None,
        help=(
            "Optional prior *_best.npz whose theta seeds the CEM mean. "
            "Useful after a measured dynamics update."
        ),
    )
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to validate the pipeline")
    args = ap.parse_args()
    if not 1 <= args.race_gates <= N_GATES:
        raise ValueError(f"--race-gates must be in 1..{N_GATES}")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.smoke:
        args.n_envs, args.pop, args.iters = 16, 4, 2

    model = SurrogateModel.load(args.model)
    gate_pos, gate_R = load_oriented_gates(args.map)
    lcfg = LineConfig(
        speed_cap=args.speed_cap,
        clearance=args.clearance,
        cap_margin=args.cap_margin,
        a_lat_max=args.a_lat_max,
        a_fwd=args.a_fwd,
        a_brk=args.a_brk,
        yaw_margin=args.yaw_margin,
        normal_lead_m=args.normal_lead_m,
        launch_speed=args.launch_speed,
    )
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    n_par = N_GATES * 2 + N_GATES + 1
    mean = np.concatenate([np.zeros(N_GATES * 2),
                           np.full(N_GATES + 1, args.initial_speed_scale)])
    sd = np.concatenate([np.full(N_GATES * 2, 0.22),
                         np.full(N_GATES + 1, 0.10)])
    active = np.zeros(n_par, bool)
    active[:args.race_gates * 2] = True
    active[N_GATES * 2:N_GATES * 2 + args.race_gates] = True
    sd[~active] = 0.0
    if args.init_best:
        prior = np.load(args.init_best, allow_pickle=False)
        prior_theta = np.asarray(prior["theta"], np.float64)
        if prior_theta.shape != mean.shape:
            raise ValueError(
                f"init theta shape {prior_theta.shape}, expected {mean.shape}"
            )
        mean = prior_theta.copy()
        # Search locally around the previously reliable geometry/schedule;
        # the changed dynamics should not throw away a good map solution.
        sd = np.concatenate([np.full(N_GATES * 2, 0.10),
                             np.full(N_GATES + 1, 0.06)])
        sd[~active] = 0.0
        print(f"initialized CEM from {args.init_best}", flush=True)
    off_lim = HOLE_HALF - args.clearance

    def unpack(theta):
        off = np.clip(theta[:N_GATES * 2], -off_lim, off_lim)
        sc = np.clip(theta[N_GATES * 2:], 0.35, 1.0)
        return off, sc

    history = []
    best = {"score": -1e9}
    t0 = time.time()
    for it in range(args.iters):
        thetas = mean[None] + sd[None] * rng.standard_normal(
            (args.pop, n_par))
        thetas[0] = mean                     # always test the mean
        if best.get("theta") is not None:
            thetas[1] = best["theta"]        # elitism: keep the champion
        t_it = time.time()
        refs = []
        for th in thetas:
            off, sc = unpack(th)
            refs.append(build_reference(gate_pos, gate_R, off, sc, lcfg))
        t_build = time.time() - t_it
        mets = evaluate_population(refs, model, args, device)
        results = []
        for j, (th, ref_j, m_j) in enumerate(zip(thetas, refs, mets)):
            sc_j = score(m_j, ref_j)
            results.append((sc_j, th))
            if sc_j > best["score"]:
                best = {"score": sc_j, "theta": th.copy(),
                        "metrics": m_j,
                        "planned_lap": ref_j["planned_lap_s"],
                        "feas": feasibility(ref_j)}
        top = max(results, key=lambda r: r[0])
        fin_top = max(m["finish_rate"] for m in mets)
        print(f"[it {it}] best_score {top[0]:8.2f} "
              f"best_fin {fin_top:.2f} "
              f"champion {best['score']:.2f}/"
              f"lap {best['metrics']['lap_median']} "
              f"({time.time() - t0:.0f}s, build {t_build:.0f}s)",
              flush=True)
        results.sort(key=lambda r: -r[0])
        elite = np.stack([th for _s, th in results[:args.elite]])
        mean = 0.4 * mean + 0.6 * elite.mean(0)
        sd = 0.5 * sd + 0.5 * (elite.std(0) + 0.02)
        sd[~active] = 0.0
        history.append({
            "iter": it,
            "best_score": results[0][0],
            "elite_mean_score": float(np.mean([s for s, _ in
                                               results[:args.elite]])),
        })
        json.dump(
            {"history": history, "best_metrics": best["metrics"],
             "best_planned_lap": best.get("planned_lap"),
             "best_feas": best.get("feas"),
             "args": {k: str(v) for k, v in vars(args).items()}},
            open(f"{out_prefix}_progress.json", "w"), indent=1)

    # final: re-eval best at scale + clean deterministic trace
    off, sc = unpack(best["theta"])
    ref = build_reference(gate_pos, gate_R, off, sc, lcfg)
    ff = feedforward_actions(ref, model)
    args.n_envs = max(args.n_envs, 512 if not args.smoke else 16)
    final = evaluate(ref, ff, model, args, device)
    clean = evaluate(ref, ff, model, args, device, clean=True)
    trace = clean.pop("trace")
    np.savez(
        f"{out_prefix}_best.npz",
        theta=best["theta"], offsets=ref["offsets"],
        seg_scale=ref["seg_scale"],
        speed_cap=np.float64(args.speed_cap),
        clearance=np.float64(args.clearance),
        cap_margin=np.float64(args.cap_margin),
        a_lat_max=np.float64(args.a_lat_max),
        a_fwd=np.float64(args.a_fwd),
        a_brk=np.float64(args.a_brk),
        yaw_margin=np.float64(args.yaw_margin),
        normal_lead_m=np.float64(args.normal_lead_m),
        launch_speed=np.float64(args.launch_speed),
        ref_pos=ref["pos"], ref_vel=ref["vel"],
        ref_quat=ref["quat_wxyz"], ref_act=ff, ref_gate=ref["gate"],
        trace_p=trace["p"], trace_v=trace["v"], trace_q=trace["q"],
        trace_act=trace["act"], trace_target=trace["target"],
        trace_t=trace["t"],
    )
    report = {
        "config": {"speed_cap": args.speed_cap,
                   "race_gates": args.race_gates,
                   "clearance": args.clearance,
                   "cap_margin": args.cap_margin,
                   "a_lat_max": args.a_lat_max,
                   "a_fwd": args.a_fwd,
                   "a_brk": args.a_brk,
                   "yaw_margin": args.yaw_margin,
                   "normal_lead_m": args.normal_lead_m,
                   "launch_speed": args.launch_speed,
                   "map": args.map, "model": args.model},
        "planned_lap_s": ref["planned_lap_s"],
        "feasibility": feasibility(ref),
        "noisy_eval": final,
        "clean_eval": clean,
    }
    json.dump(report, open(f"{out_prefix}_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items()
                      if k != "config"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
