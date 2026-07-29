"""Fast VECTORIZED SAC in the MuJoCo twin. N parallel envs amortize the (fixed ~13ms) SAC update
over many transitions, so wall-clock throughput is high despite this machine's slow torch dispatch.
Reuses racer_state's SAC/config/reward/geom. Gate 0 pass = terminal success. Logs to
runs_mj/metrics.jsonl for the dashboard.

  python -m racer_mujoco.train_mj --envs 64 --updates 8
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import mujoco

from racer_state.config import CFG
from racer_state.sac import SAC
from racer_state.reward import step_reward
from racer_state.geom import rot_world_to_body
from racer_mujoco.env import QuadEnv

_DOWN = np.array([0.0, 0.0, -1.0])
AWAY = 10
RUN_DIR = os.path.join(os.path.dirname(__file__), "runs_mj")


def obs_from(pos, quat, vel, rates, gate_pos, gate_normal):
    return np.concatenate([
        rot_world_to_body(quat, gate_pos - pos) / CFG.pos_scale,
        rot_world_to_body(quat, vel) / CFG.vel_scale,
        rates / CFG.rate_scale,
        rot_world_to_body(quat, _DOWN),
        rot_world_to_body(quat, gate_normal),
    ]).astype(np.float32)


def obs_of(env):
    return obs_from(env.pos, env.quat, env.vel, env.rates, env.gate_pos, env.gate_normal)


# y-mirror symmetry: the quad + gate task is left/right symmetric in the gate frame, so every
# transition has a valid mirrored twin (free 2x data). True vectors flip their y-component
# (rel_gate, vel, gravity, normal); body rates are a pseudovector so roll & yaw flip instead;
# actions flip roll & yaw commands. obs = [rel(3), vel(3), rates(3), grav(3), normal(3)].
MIR_O = np.array([1, -1, 1,  1, -1, 1,  -1, 1, -1,  1, -1, 1,  1, -1, 1], np.float32)
MIR_A = np.array([-1, 1, -1, 1], np.float32)


def mirror_tr(tr):
    o, a, r, no, d, g, ret = tr
    return (o * MIR_O, a * MIR_A, r, no * MIR_O, d, g, ret)


def pack_cols(pack):
    """Transition list -> stacked column arrays for ReplayBuffer.add_batch."""
    o, a, r, no, d, g, ret = zip(*pack)
    return (np.stack(o), np.stack(a), np.asarray(r, np.float32), np.stack(no),
            np.asarray(d, np.float32), np.asarray(g, np.float32), np.asarray(ret, np.float32))


def add_pack(buf, cols, mirror):
    """Bulk-insert a packed episode (and optionally its y-mirrored twin) into a buffer."""
    O, A, R, NO, D, G, RT = cols
    buf.add_batch(O, A, R, NO, D, G, RT)
    if mirror:
        buf.add_batch(O * MIR_O, A * MIR_A, R, NO * MIR_O, D, G, RT)


def nstep_pack(trans, n, gamma):
    """Convert an episode-ordered list of 1-step transitions (o, a, r, no, term) into n-step
    transitions (o, a, R_n, o_{t+k}, term_k, gamma^k, ret) where R_n is the k-step discounted
    reward sum (k<=n, truncated at terminals) and ret is the full discounted return-to-go
    (feeds the advantage-filtered self-imitation)."""
    L = len(trans)
    rets = [0.0] * L
    run = 0.0
    for t in range(L - 1, -1, -1):
        run = trans[t][2] + gamma * run
        rets[t] = run
    out = []
    for t in range(L):
        R = 0.0
        k = 0
        while k < n and t + k < L:
            R += (gamma ** k) * trans[t + k][2]
            k += 1
            if trans[t + k - 1][4]:      # hit a true terminal -> stop the window
                break
        last = trans[t + k - 1]
        out.append((trans[t][0], trans[t][1], R, last[3], last[4], gamma ** k, rets[t]))
    return out


def her_relabel(raw, cfg, rng):
    """Hindsight virtual-gate relabeling: for a FAILED episode, place a virtual gate ON the flown
    path (center = a point the drone actually crossed, normal = its velocity direction there) and
    rebuild obs/rewards wrt that gate — the episode becomes a successful pass of the virtual gate.
    Because the obs is purely gate-relative, 'thread the gate wherever it is' transfers directly to
    the real gate. This solves discovery with a FIXED spawn (no sim control needed -> VQ1-portable).

    raw: list of (pos, quat, vel, rates, action) per step + a final entry for the terminal state
    (action=None). Returns a 1-step transition list ending in a +gate_bonus terminal, or None."""
    S = len(raw) - 1                     # number of action steps
    if S < 12:
        return None
    lo = max(1, int(S * 0.4))            # relabel a point in the later part of the flight
    cands = [t for t in range(lo, S)
             if np.linalg.norm(raw[t + 1][0] - raw[t][0]) > 0.02]   # moving, not parked
    if not cands:
        return None
    ts = int(rng.choice(cands))
    p0, p1 = raw[ts][0], raw[ts + 1][0]
    normal = p1 - p0
    normal = normal / np.linalg.norm(normal)
    center = 0.5 * (p0 + p1)             # the step ts -> ts+1 crosses this plane by construction
    trans = []
    prev_d = float(np.linalg.norm(center - raw[0][0]))
    for t in range(ts + 1):
        pos, quat, vel, rates, act = raw[t]
        npos, nquat, nvel, nrates, _ = raw[t + 1]
        o = obs_from(pos, quat, vel, rates, center, normal)
        no = obs_from(npos, nquat, nvel, nrates, center, normal)
        d = float(np.linalg.norm(center - npos))
        r = cfg.w_prog * (prev_d - d) - cfg.step_penalty
        term = 1.0 if t == ts else 0.0
        if term:
            r += cfg.gate_bonus          # the virtual pass pays exactly like a real one
        trans.append((o, act, r, no, term))
        prev_d = d
    return trans


def _euler_pr(q):
    w, x, y, z = q
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return pitch, roll


def hand_action(env, tilt):
    """Expert (verified 100% crossing): hold a forward pitch angle, keep roll level, hold altitude
    at the gate height. Gate opening (1.35 m half) contains the 0.4 m y-offset, so no y-steering."""
    pitch, roll = _euler_pr(env.quat)
    a1 = np.clip(3.0 * (tilt - pitch), -1, 1)                 # forward tilt
    a0 = np.clip(-3.0 * roll, -1, 1)                          # keep level
    a3 = np.clip(1.5 * (env.gate_pos[2] - env.pos[2]) - 0.3 * env.vel[2], -1, 1)  # hold altitude
    return np.array([a0, a1, 0.0, a3], np.float32)


def seed_demos(agent, n_eps=60):
    """Run the expert for n_eps episodes; push transitions into the buffer (successful ones also
    into elite). Return the (obs, action) pairs from SUCCESSFUL runs for BC warm-start."""
    e = QuadEnv(episode_s=8.0, seed=7)
    added = succ = 0; dobs = []; dact = []
    for k in range(n_eps):
        e.reset(); prev = e.dist_to_gate(); away = 0; trans = []; sa = []
        tilt = 0.2 + 0.25 * (k % 5) / 4.0
        for _ in range(e.max_steps):
            obs = obs_of(e)
            a = np.clip(hand_action(e, tilt) + np.random.normal(0, 0.03, 4).astype(np.float32), -1, 1)
            passed, crashed = e.step(a); d = e.dist_to_gate()
            away = 0 if d < prev else away + 1
            r, done, reason = step_reward(prev, d, passed, crashed, d > CFG.stray_dist, CFG)
            if passed: done, reason = True, "success"
            elif not done and away >= AWAY: r += CFG.fail_reward(d); done, reason = True, "away"
            term = reason in ("success", "crash", "stray", "away")
            no = obs_of(e); agent.buf.add(obs, a, r, no, float(term))
            trans.append((obs, a, r, no, float(term))); sa.append((obs, a)); prev = d; added += 1
            if done: break
        if reason == "success":
            agent.add_elite(trans); succ += 1
            for o, ac in sa:
                dobs.append(o); dact.append(ac)
    print(f"seeded {added} demo transitions from {succ}/{n_eps} successful expert runs", flush=True)
    return np.array(dobs, np.float32), np.array(dact, np.float32)


def bc_warmstart(agent, dobs, dact, epochs=400, batch=256):
    """Behavior-clone the actor onto the expert (obs -> action) so it STARTS able to cross."""
    import torch.nn.functional as F
    O = torch.as_tensor(dobs, device=agent.device); A = torch.as_tensor(dact, device=agent.device)
    n = len(O)
    for ep in range(epochs):
        idx = torch.randint(0, n, (batch,), device=agent.device)
        mean, _ = agent.actor(O[idx])
        loss = F.mse_loss(torch.tanh(mean), A[idx])
        agent.a_opt.zero_grad(set_to_none=True); loss.backward(); agent.a_opt.step()
    print(f"BC warm-start: {epochs} epochs on {n} expert pairs, final MSE={float(loss):.4f}", flush=True)


_RENDERER = None
_VIDEO_OK = True


def record_greedy(agent, env, path, width=384, height=256, fps=30, gates=1):
    """Roll out ONE greedy episode (up to `gates` gate passes) with a chase camera -> mp4.
    Best-effort: if the GL renderer ever fails, video is disabled for the rest of the run."""
    global _RENDERER, _VIDEO_OK
    if not _VIDEO_OK:
        return False
    try:
        if _RENDERER is None:
            _RENDERER = mujoco.Renderer(env.model, height, width)
        import cv2
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = 90.0; cam.elevation = -18.0; cam.distance = 9.0
        frames = []
        env.reset(); prev = env.dist_to_gate(); away = 0; g = 0; result = "timeout"
        steps = env.max_steps
        for _ in range(steps):
            cam.lookat[:] = 0.5 * (env.pos + env.gate_pos)
            _RENDERER.update_scene(env.data, cam)
            frames.append(_RENDERER.render().copy())
            passed, crashed = env.step(agent.act(obs_of(env), deterministic=True, ema=True))
            d = env.dist_to_gate()
            if passed:
                g += 1; away = 0; prev = d
                if g < gates and not crashed:
                    continue
            else:
                away = 0 if d < prev else away + 1; prev = d
            if (passed and g >= gates) or crashed or d > CFG.stray_dist or away >= AWAY:
                result = f"PASSx{g}" if g >= gates else (f"crash(g{g})" if crashed else f"away(g{g})")
                for _ in range(8):        # trailing frames so the outcome is visible
                    cam.lookat[:] = 0.5 * (env.pos + env.gate_pos)
                    _RENDERER.update_scene(env.data, cam); frames.append(_RENDERER.render().copy())
                break
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"  video saved: {os.path.basename(path)} [{result}, {len(frames)} frames]", flush=True)
        return True
    except Exception as ex:
        _VIDEO_OK = False
        print(f"  video disabled (renderer error: {ex})", flush=True)
        return False


def greedy_eval(env, agent, n=20, gates=1):
    """n greedy episodes, each flown until `gates` passes / failure. Returns (counts, mean_min_dist)
    where counts[k] = episodes that passed at least k+1 gates."""
    reached = []; mds = []
    for _ in range(n):
        env.reset(); prev = env.dist_to_gate(); away = 0; g = 0; md = prev
        for _ in range(env.max_steps):
            passed, crashed = env.step(agent.act(obs_of(env), deterministic=True, ema=True))
            d = env.dist_to_gate(); md = min(md, d)
            if passed:
                g += 1; away = 0; prev = d
                if g >= gates: break
                continue
            away = 0 if d < prev else away + 1; prev = d
            if crashed or d > CFG.stray_dist or away >= AWAY: break
        reached.append(g); mds.append(md)
    counts = [sum(1 for g in reached if g > k) for k in range(gates)]
    return counts, float(np.mean(mds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--updates", type=int, default=8, help="SAC updates per vector tick")
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--ent", type=float, default=None, help="override target_entropy (exploration)")
    ap.add_argument("--away", type=int, default=10, help="max consecutive receding frames")
    ap.add_argument("--gbonus", type=float, default=None, help="override gate pass bonus")
    ap.add_argument("--demos", type=int, default=0, help="seed N expert demo episodes into the buffer")
    ap.add_argument("--bcw", type=float, default=2.0, help="BC-anchor weight during SAC (0=off)")
    ap.add_argument("--cwarm", type=int, default=4000, help="critic-only warm-up updates after BC")
    ap.add_argument("--alpha0", type=float, default=None, help="initial entropy temperature (low keeps the BC policy deterministic)")
    ap.add_argument("--curr", action="store_true", help="distance curriculum: spawn close, move back on success")
    ap.add_argument("--curr_start", type=float, default=4.0)
    ap.add_argument("--curr_step", type=float, default=2.5)
    ap.add_argument("--curr_thresh", type=int, default=16, help="greedy passes/20 to advance the curriculum")
    ap.add_argument("--video_every", type=int, default=1000, help="save a greedy chase-cam mp4 every N episodes (multiple of 200)")
    ap.add_argument("--resume", default=None, help="load actor+critic+alpha from a rung checkpoint (.pt)")
    ap.add_argument("--batch", type=int, default=1024, help="SAC batch size (this box is dispatch-bound: 1024 costs ~1.7x a 256 batch for 4x the samples)")
    ap.add_argument("--nstep", type=int, default=5, help="n-step returns (fast sparse-bonus propagation)")
    ap.add_argument("--her", type=int, default=1, help="hindsight virtual-gate relabels per FAILED episode (0=off)")
    ap.add_argument("--silw", type=float, default=1.0, help="advantage-filtered self-imitation weight (0=off)")
    ap.add_argument("--gates", type=int, default=1, help="gates per episode: episode 'finish' after passing this many")
    ap.add_argument("--episode_s", type=float, default=8.0, help="episode cap in seconds")
    ap.add_argument("--mirror", type=int, default=1, help="y-mirror symmetry augmentation (free 2x data); 0=off")
    ap.add_argument("--learner", type=int, default=0, help="1 = run SAC updates in a background thread (continuous gradient descent while envs collect)")
    ap.add_argument("--utd_cap", type=float, default=2.0, help="learner-thread cap: max updates per collected env step")
    ap.add_argument("--min_utd", type=float, default=0.0, help="learner-thread pacing floor: env side sleeps when updates/steps falls below this (frees CPU for the learner; self-balancing)")
    ap.add_argument("--cdrop", type=float, default=None, help="critic dropout override (0 = plain critics; DroQ dropout doubles update cost)")
    ap.add_argument("--sil_every", type=int, default=1, help="run the SIL term only every k-th update (cheaper updates)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    global AWAY
    AWAY = args.away
    if args.ent is not None:
        CFG.target_entropy = args.ent
    if args.gbonus is not None:
        CFG.gate_bonus = args.gbonus
    CFG.batch_size = args.batch
    CFG.sil_weight = args.silw
    if args.cdrop is not None:
        CFG.critic_dropout = args.cdrop
    print(f"knobs: target_entropy={CFG.target_entropy} away={AWAY} gate_bonus={CFG.gate_bonus} "
          f"batch={CFG.batch_size} nstep={args.nstep} her={args.her} silw={CFG.sil_weight}", flush=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    mpath = os.path.join(RUN_DIR, "metrics.jsonl"); open(mpath, "w").close()
    dev = args.device
    N = args.envs
    envs = [QuadEnv(episode_s=args.episode_s, seed=i) for i in range(N)]
    for e in envs:
        e.reset()
    prev = np.array([e.dist_to_gate() for e in envs])
    away = np.zeros(N, int)
    ep_r = np.zeros(N); ep_steps = np.zeros(N, int); ep_mind = prev.copy()
    ep_gates = np.zeros(N, int)      # gates passed this episode (per env)
    last_pass = np.zeros(N, int)     # raw_list index right after the last real pass (HER segment)
    agent = SAC(CFG, dev)
    warm_left = 0
    if args.resume is not None:
        agent.load_full(args.resume)
        warm_left = args.cwarm       # actor-only ckpt -> warm the fresh critic before actor updates
        print(f"resumed from {os.path.basename(args.resume)} (critic warm-up: {warm_left} updates)", flush=True)
    if args.alpha0 is not None:
        with torch.no_grad():
            agent.log_alpha.copy_(torch.log(torch.tensor(float(args.alpha0))))
    eval_env = QuadEnv(episode_s=args.episode_s, seed=999)
    FULL_DIST = float(eval_env.spawn_dist)
    if args.curr:
        for e in envs + [eval_env]:
            e.spawn_dist = args.curr_start
            e.reset()
        prev = np.array([e.dist_to_gate() for e in envs]); ep_mind = prev.copy()
        print(f"CURRICULUM on: spawn_dist {args.curr_start} -> {FULL_DIST}", flush=True)
    print(f"MuJoCo VEC-SAC | {N} envs | {args.updates} upd/tick | device={dev}", flush=True)

    DOBS = DACT = None
    if args.demos > 0 and not args.curr:
        dobs, dact = seed_demos(agent, args.demos)
        if len(dobs) > 0:
            bc_warmstart(agent, dobs, dact)
            DOBS = torch.as_tensor(dobs, device=dev); DACT = torch.as_tensor(dact, device=dev)
            c, gmd = greedy_eval(eval_env, agent, 20)
            print(f"after BC warm-start: GREEDY {c[0]}/20 (mind {gmd:.1f})", flush=True)
            for _ in range(args.cwarm):    # warm the critic (actor frozen) so Q values the demos
                agent.update(metrics=False, critic_only=True)
            c, gmd = greedy_eval(eval_env, agent, 20)
            print(f"after critic warm-up ({args.cwarm}): GREEDY {c[0]}/20 (mind {gmd:.1f})", flush=True)
    ep = 0; total = 0; passes = []; m_last = {}; solved = set()
    ep_list = [[] for _ in range(N)]    # per-env 1-step transitions (flushed n-step at episode end)
    raw_list = [[] for _ in range(N)]   # per-env raw kinematics (for hindsight relabeling)
    rng = np.random.default_rng(123)
    shared = {"steps": 0, "updates": 0, "m": {}, "stop": False, "warm": warm_left}
    if args.learner:
        import threading

        def _learner():
            # Continuous gradient descent while the main thread collects: torch CPU kernels release
            # the GIL, so updates overlap env stepping. Benign races (actor weights mid-act, ring
            # rows mid-sample) are accepted — rare and tiny vs lr. UTD is capped against collected
            # steps so the learner can't over-train a small early buffer.
            while not shared["stop"]:
                if agent.buf.size < CFG.update_after or \
                        shared["updates"] > shared["steps"] * args.utd_cap + 1000:
                    time.sleep(0.002); continue
                if shared["warm"] > 0:
                    agent.update(metrics=False, critic_only=True)
                    shared["warm"] -= 1
                    if shared["warm"] == 0:
                        print("critic warm-up done (learner thread)", flush=True)
                elif shared["updates"] % 200 == 0:
                    shared["m"] = agent.update(metrics=True)
                else:
                    if args.sil_every > 1 and shared["updates"] % args.sil_every != 0:
                        sw = CFG.sil_weight; CFG.sil_weight = 0.0   # cheap update: skip SIL forwards
                        agent.update(metrics=False)
                        CFG.sil_weight = sw
                    else:
                        agent.update(metrics=False)
                shared["updates"] += 1
        threading.Thread(target=_learner, daemon=True).start()
        print(f"LEARNER THREAD on (utd_cap {args.utd_cap} upd/env-step)", flush=True)
    t0 = time.time(); last_t = t0; last_step = 0
    random_ticks = 0 if (args.demos > 0 or args.resume is not None) else CFG.start_random_steps // N + 1
    tick = 0
    while total < args.steps:
        obs_b = np.stack([obs_of(e) for e in envs])
        if tick < random_ticks:
            acts = np.random.uniform(-1, 1, (N, 4)).astype(np.float32)
        else:
            with torch.no_grad():
                o = torch.as_tensor(obs_b, dtype=torch.float32, device=dev)
                acts = agent.actor.act(o, deterministic=False).cpu().numpy()
        for i, e in enumerate(envs):
            raw_list[i].append((e.pos.copy(), e.quat.copy(), e.vel.copy(), e.rates.copy(),
                                acts[i].copy()))
            passed, crashed = e.step(acts[i])
            d = e.dist_to_gate()          # to the ACTIVE gate (already retargeted on a pass)
            ep_mind[i] = min(ep_mind[i], d)
            away[i] = 0 if (passed or d < prev[i]) else away[i] + 1
            # on a pass the target switched: zero the progress term that step (prev := d)
            r, done, reason = step_reward(prev[i] if not passed else d, d, passed, crashed,
                                          d > CFG.stray_dist, CFG)
            if passed:
                ep_gates[i] += 1
                last_pass[i] = len(raw_list[i])   # HER segment starts after the last real pass
                if ep_gates[i] >= args.gates and not done:
                    r += CFG.finish_bonus; done, reason = True, "finish"
            if not done and away[i] >= AWAY:
                # away is a CHOICE, not an accident: never pay the near-gate commitment bonus for
                # retreating ("approach then hover" was a comfy positive-return local optimum)
                r += min(0.0, CFG.fail_reward(d)); done, reason = True, "away"
            term = reason in ("finish", "crash", "stray", "away")
            ep_list[i].append((obs_b[i], acts[i], r, obs_of(e), float(term)))
            ep_r[i] += r; ep_steps[i] += 1; prev[i] = d
            if done or ep_steps[i] >= e.max_steps:
                # flush the episode: n-step pack into the main buffer; any episode with a real pass
                # feeds the elite buffer (-> oversampling + self-imitation); the failed tail segment
                # (after the last real pass) gets hindsight virtual-gate relabels so EVERY episode
                # teaches gate-crossing — including the gate_k -> gate_{k+1} leg.
                raw_list[i].append((e.pos.copy(), e.quat.copy(), e.vel.copy(), e.rates.copy(), None))
                cols = pack_cols(nstep_pack(ep_list[i], args.nstep, CFG.gamma))
                add_pack(agent.buf, cols, args.mirror)
                if ep_gates[i] >= 1:
                    add_pack(agent.elite, cols, args.mirror)
                if reason != "finish" and args.her > 0:
                    seg = raw_list[i][last_pass[i]:]
                    for _ in range(args.her):
                        v = her_relabel(seg, CFG, rng)
                        if v:
                            add_pack(agent.buf, pack_cols(nstep_pack(v, args.nstep, CFG.gamma)),
                                     args.mirror)
                ep_list[i] = []; raw_list[i] = []
                passes.append(int(ep_gates[i] >= args.gates))
                pr = float(np.mean(passes[-200:]))
                now = time.time(); sps = (total - last_step) / max(1e-6, now - last_t)
                rec = {"ep": ep, "t": now, "step": total, "reward": float(ep_r[i]),
                       "min_dist": float(ep_mind[i]), "reason": reason,
                       "gates": int(ep_gates[i]),
                       "passed": int(ep_gates[i] >= args.gates), "pass_rate100": pr,
                       "steps_per_s": sps, "buf": agent.buf.size, **m_last}
                if ep % 200 == 0 and agent.buf.size >= CFG.update_after:
                    counts, gmd = greedy_eval(eval_env, agent, 20, args.gates)
                    sd = eval_env.spawn_dist
                    rec["eval_g0"] = counts[0]; rec["eval_mind"] = gmd; rec["spawn_dist"] = sd
                    if len(counts) > 1:
                        rec["eval_g1"] = counts[1]
                    rec["eval_gates"] = counts
                    agent.save(os.path.join(RUN_DIR, "autosave.pt"))   # always have a resume point
                    cstr = " ".join(f"g{k}:{c}/20" for k, c in enumerate(counts))
                    rec["updates"] = shared["updates"]
                    utd_s = f" utd={shared['updates'] / max(1, total):.2f}" if args.learner else ""
                    print(f"ep {ep:6d} step {total:8d} | train finish%={100*pr:3.0f} | GREEDY {cstr} "
                          f"(mind {gmd:.1f}) | {sps:5.0f} st/s{utd_s} buf={agent.buf.size}", flush=True)
                    last_t = now; last_step = total
                    if ep % args.video_every == 0:
                        record_greedy(agent, eval_env, os.path.join(RUN_DIR, f"greedy_ep{ep}.mp4"),
                                      gates=args.gates)
                    if args.curr and counts[0] >= args.curr_thresh and sd < FULL_DIST - 0.01:
                        nd = min(FULL_DIST, sd + args.curr_step)
                        for e in envs + [eval_env]:
                            e.spawn_dist = nd
                        print(f"  >> curriculum advance: spawn_dist {sd:.1f} -> {nd:.1f}", flush=True)
                    for k in range(args.gates):     # per-gate 20/20 milestones + marginal timing
                        if counts[k] >= 20 and k not in solved:
                            solved.add(k)
                            mins = (now - t0) / 60.0
                            agent.save(os.path.join(RUN_DIR, f"mj_gate{k}.pt"))
                            record_greedy(agent, eval_env,
                                          os.path.join(RUN_DIR, f"gate{k}_solved.mp4"), gates=k + 1)
                            print(f"  *** 20/20 through gate {k} at +{mins:.1f} min "
                                  f"(ep {ep}, step {total}) ***", flush=True)
                with open(mpath, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                e.reset(); prev[i] = e.dist_to_gate(); away[i] = 0
                ep_r[i] = 0.0; ep_steps[i] = 0; ep_mind[i] = prev[i]
                ep_gates[i] = 0; last_pass[i] = 0; ep += 1
        total += N; tick += 1
        shared["steps"] = total
        if args.learner:
            m_last = shared["m"]
            if args.min_utd > 0 and agent.buf.size >= CFG.update_after:
                # pacing floor: don't outrun the learner — sleeping here frees the GIL and CPU
                # cores so the learner's update rate rises; self-balancing on this shared box
                while (shared["updates"] + 30) / max(1, total) < args.min_utd:
                    time.sleep(0.003)
        elif agent.buf.size >= CFG.update_after:
            bc = None
            for _ in range(args.updates):
                if warm_left > 0:      # resume path: warm the fresh critic before touching the actor
                    agent.update(metrics=False, critic_only=True); warm_left -= 1
                    if warm_left == 0:
                        print(f"critic warm-up done (step {total})", flush=True)
                    continue
                if DOBS is not None and args.bcw > 0:
                    bi = torch.randint(0, len(DOBS), (CFG.batch_size,), device=dev)
                    bc = (DOBS[bi], DACT[bi])
                m = agent.update(metrics=False, bc=bc, bc_weight=args.bcw)
            if tick % 20 == 0 and warm_left <= 0:
                m_last = agent.update(metrics=True, bc=bc, bc_weight=args.bcw)


if __name__ == "__main__":
    main()
