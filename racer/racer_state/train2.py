"""Stack-D trainer for the REAL VQ1 sim, warm-started from the MuJoCo twin policy.

Everything that solved the full track in the twin, ported to the 30 Hz real-time loop:
  - twin<->VQ1 ADAPTER (probed 2026-07-23): obs handedness flip FRD->FLU, and the measured
    sign-inverted ~2.5x command-response folded into the action map — so the twin actor flies
    VQ1 natively and all training happens in the twin's convention (checkpoints stay compatible).
  - GPU LEARNER THREAD: continuous SAC updates while the control loop flies (utd-capped).
  - episode-end n-step flush (add_batch), HER virtual-gate relabels on the failed segment,
    y-mirror augmentation, elite + AWR-soft self-imitation, away-exploit fix.
  - per-gate milestone evals (EMA actor) with checkpoints + camera videos.

  python -m racer_state.train2 --resume <twin ckpt> --cwarm 1500      # warm-started training
  python -m racer_state.train2 --resume <twin ckpt> --eval 3          # adapter probe / eval only
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
os.environ.setdefault("SCIPY_ARRAY_API", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from racer_state.config import CFG
from racer_state.mav_io import MavLink
from racer_state.geom import rot_world_to_body, rot_body_to_world, gate_normal_world
from racer_state.reward import step_reward
from racer_state.sac import SAC
from racer_state.train import wait_reset, write_video

# ---------- twin <-> VQ1 adapter: PHYSICAL-UNIT action map ----------
# obs: VQ1 body frame is FRD (NED), twin is FLU -> flip y,z of every 3-vector block.
OBS_SIGN = np.tile(np.array([1.0, -1.0, -1.0], np.float32), 5)
# The policy's action has fixed PHYSICAL semantics (the twin's): a[:3] = desired FLU body rates
# (|a|=1 -> 4 rad/s), a[3] = desired thrust-to-weight, T/W = (0.28 + a3*0.35)/0.28 (the twin's
# hover-centered map — 0.28 is the TWIN's hover constant, part of the action semantics, NOT
# VQ1's hover). Those physical targets are mapped to VQ1 commands through runs/calib.json
# (racer_state.calibrate). The old map reused the twin's 0.35 cmd-span around VQ1's 0.19 hover:
# right hover point, ~1.5x too much accel per unit a3 — the policy had to re-learn its own
# thrust authority. Rates were silently capped ~3 rad/s by the +/-1.2 cmd clip.
RATE_SCALE_TWIN = 4.0
THRUST_SPAN = 0.35
TWIN_HOVER = 0.28

_CAL_PATH = os.path.join(CFG.run_dir, "calib.json")
if os.path.exists(_CAL_PATH):
    with open(_CAL_PATH, encoding="utf-8") as _f:
        _CAL = json.load(_f)
    GAIN = np.array(_CAL["rates"]["gain"])
    RATE_MAX = np.array(_CAL["rates"]["rate_max"])
    CMD_MAX = np.array(_CAL["rates"]["cmd_max"])
    T_INT = float(_CAL["thrust"]["intercept"])
    T_SLOPE = float(_CAL["thrust"]["slope"])
    HOVER_CMD = float(_CAL["thrust"]["hover_cmd"])
    print(f"calib.json loaded: hover_cmd={HOVER_CMD:.3f} t_slope={T_SLOPE:.1f} "
          f"rate_gain={np.round(GAIN, 2)} rate_max={np.round(RATE_MAX, 2)}", flush=True)
else:
    # fallback = 2026-07-28 probes + proportional-thrust assumption through the 0.19 hover
    # (pitch cmd gain is POSITIVE — the odometry report was inverted, not the plant)
    GAIN = np.array([-2.48, 2.52, -2.31])
    RATE_MAX = np.abs(GAIN) * 1.2
    CMD_MAX = np.array([1.2, 1.2, 1.2])
    T_INT, T_SLOPE, HOVER_CMD = 0.0, 9.81 / CFG.thrust_hover, CFG.thrust_hover
    print("WARNING: no calib.json (run racer_state.calibrate) — using fallback constants",
          flush=True)


def thrust_cmd(tw):
    """Desired thrust-to-weight -> VQ1 thrust cmd via the measured curve."""
    return float(np.clip((max(0.0, tw) * 9.81 - T_INT) / T_SLOPE, 0.0, 1.0))


TOP_CMD = thrust_cmd((TWIN_HOVER + THRUST_SPAN) / TWIN_HOVER)   # cmd at a3=+1


def act_desired(a, tcap=1.0, rcaps=None):
    """Twin action -> desired FRD body rates (rad/s) + calibrated VQ1 thrust cmd. The rate command
    is closed inside stream_rates (feedforward + P on live odometry rates), mimicking the crisp KP
    inner loop the policy trained on in the twin — raw open-loop cmds turned its aggression into
    PIO. tcap/rcaps: optional authority ceilings (default = full twin authority, no re-learning)."""
    des_frd = np.array([a[0], -a[1], -a[2]]) * RATE_SCALE_TWIN
    des_frd = np.clip(des_frd, -RATE_MAX, RATE_MAX)
    if rcaps is not None:
        des_frd = np.clip(des_frd, -rcaps, rcaps)
    tw = (TWIN_HOVER + a[3] * THRUST_SPAN) / TWIN_HOVER
    thrust = float(min(thrust_cmd(tw), tcap))
    return des_frd, thrust


def thrust_cap(ep, start=1.0, full_ep=1000):
    """Optional authority curriculum (cmd units). start >= TOP_CMD disables it — with the
    calibrated map the policy already knows its authority; default is no ramp."""
    if start >= TOP_CMD:
        return 1.0
    f = min(1.0, ep / max(1, full_ep))
    return start + f * (TOP_CMD - start)


def rate_cap(ep, start, full_ep):
    """Per-axis rate-authority ramp: `start` rad/s at ep 0 -> full twin authority (4 rad/s) at
    full_ep. Pitch is the fly-forward axis so it starts stronger and ramps fast; roll/yaw mostly
    destabilize early (gates are nearly straight ahead) so they start small and ramp slow."""
    f = min(1.0, ep / max(1, full_ep))
    return start + f * (RATE_SCALE_TWIN - start)


_DOWN_NED = np.array([0.0, 0.0, 1.0])
NGATES_OH = 6                      # one-hot gate-index dims appended when --gate_onehot
GATE_ONEHOT = False                # set from args in main(); module-level so helpers see it


def frd_obs(pos, quat, vel, rates, gate_pos, gate_normal, gid=0):
    """VQ1 state -> twin-convention 15-dim obs (identical semantics to racer_mujoco.obs_from),
    optionally + a one-hot of the ACTIVE gate index (--gate_onehot). The one-hot mainly fixes
    CRITIC value aliasing (2 m before gate 0 is worth 5 more gates than 2 m before gate 5, but
    the gate-relative obs alone can't tell them apart). Zero-padded first-layer columns keep
    15-dim twin checkpoints byte-identical in behavior at load.
    NOTE: VQ1's ODOMETRY velocity is ALREADY body-frame (probed: corr 0.90 body vs 0.79 world) —
    do NOT rotate it. (The old from-scratch trainer double-rotated it and simply learned around
    the corrupted-but-consistent feature; a transferred policy can't.)"""
    o = np.concatenate([
        rot_world_to_body(quat, gate_pos - pos) / CFG.pos_scale,
        np.asarray(vel, np.float64) / CFG.vel_scale,
        np.asarray(rates, np.float64) / CFG.rate_scale,
        rot_world_to_body(quat, _DOWN_NED),
        rot_world_to_body(quat, gate_normal),
    ]).astype(np.float32) * OBS_SIGN
    if not GATE_ONEHOT:
        return o
    oh = np.zeros(NGATES_OH, np.float32)
    oh[min(int(gid), NGATES_OH - 1)] = 1.0
    return np.concatenate([o, oh])


# ---------- stack-D buffer machinery (twin-convention space) ----------
MIR_O = np.array([1, -1, 1,  1, -1, 1,  -1, 1, -1,  1, -1, 1,  1, -1, 1], np.float32)
MIR_A = np.array([-1, 1, -1, 1], np.float32)
# y-mirror leaves the gate index unchanged -> ones over the one-hot tail (set in main()).


def _pad_obs_cols(w, at, extra):
    z = torch.zeros(w.shape[0], extra, dtype=w.dtype)
    return torch.cat([w[:, :at], z, w[:, at:]], 1)


def _buf_state(b):
    n = b.size
    return {"o": b.o[:n].cpu(), "a": b.a[:n].cpu(), "r": b.r[:n].cpu(), "no": b.no[:n].cpu(),
            "d": b.d[:n].cpu(), "g": b.g[:n].cpu(), "ret": b.ret[:n].cpu()}


def save_buffers(agent, path):
    """Persist replay + elite so restarts don't discard experience (at ~1 pass / 8 episodes on
    the real-time sim, a lost elite buffer sets consolidation back by hours)."""
    torch.save({"obs_dim": agent.buf.o.shape[1], "main": _buf_state(agent.buf),
                "elite": _buf_state(agent.elite)}, path)


def load_buffers(agent, path):
    d = torch.load(path, map_location="cpu")
    extra = agent.buf.o.shape[1] - int(d["obs_dim"])
    for name, buf in (("main", agent.buf), ("elite", agent.elite)):
        s = d[name]
        n = min(len(s["o"]), buf.cap)
        o, no = s["o"][:n], s["no"][:n]
        if extra > 0:      # pre-onehot data: zero one-hot tail = "no ID" (zero-init cols -> inert)
            z = torch.zeros(n, extra)
            o = torch.cat([o, z], 1); no = torch.cat([no, z], 1)
        for t, key in ((o, "o"), (s["a"][:n], "a"), (s["r"][:n], "r"), (no, "no"),
                       (s["d"][:n], "d"), (s["g"][:n], "g"), (s["ret"][:n], "ret")):
            getattr(buf, key)[:n] = t.to(buf.device)
        buf.pos = n % buf.cap
        buf.size = n
    print(f"buffers restored: main {agent.buf.size}, elite {agent.elite.size}", flush=True)


def load_resume(agent, path, act_dim=4):
    """agent.load_full that also accepts SMALLER-obs checkpoints (15-dim twin/pre-onehot):
    zero-pad the new obs columns into every first layer — appended inputs start with zero
    weight, so the loaded policy/critic behave EXACTLY as before until training uses them."""
    d = torch.load(path, map_location=agent.device)
    want = agent.actor.body[0].weight.shape[1]
    have = d["actor"]["body.0.weight"].shape[1]
    if have < want:
        extra = want - have
        for key in ("actor", "actor_ema"):
            if key in d:
                d[key]["body.0.weight"] = _pad_obs_cols(d[key]["body.0.weight"], have, extra)
        if "critic" in d:
            for k in list(d["critic"]):
                if k.endswith(".0.weight") and d["critic"][k].shape[1] == have + act_dim:
                    d["critic"][k] = _pad_obs_cols(d["critic"][k], have, extra)
        print(f"resume: padded obs {have} -> {want} (zero-init new columns)", flush=True)
    agent.actor.load_state_dict(d["actor"])
    agent.actor_ema.load_state_dict(d.get("actor_ema", d["actor"]))
    if "critic" in d:
        agent.critic.load_state_dict(d["critic"])
        agent.critic_tgt.load_state_dict(d["critic"])
    if "log_alpha" in d:
        with torch.no_grad():
            agent.log_alpha.copy_(torch.as_tensor(d["log_alpha"], device=agent.device))


def nstep_pack(trans, n, gamma):
    L = len(trans); rets = [0.0] * L; run = 0.0
    for t in range(L - 1, -1, -1):
        run = trans[t][2] + gamma * run; rets[t] = run
    out = []
    for t in range(L):
        R = 0.0; k = 0
        while k < n and t + k < L:
            R += (gamma ** k) * trans[t + k][2]; k += 1
            if trans[t + k - 1][4]:
                break
        last = trans[t + k - 1]
        out.append((trans[t][0], trans[t][1], R, last[3], last[4], gamma ** k, rets[t]))
    return out


def pack_cols(pack):
    o, a, r, no, d, g, ret = zip(*pack)
    return (np.stack(o), np.stack(a), np.asarray(r, np.float32), np.stack(no),
            np.asarray(d, np.float32), np.asarray(g, np.float32), np.asarray(ret, np.float32))


def add_pack(buf, cols, mirror):
    O, A, R, NO, D, G, RT = cols
    buf.add_batch(O, A, R, NO, D, G, RT)
    if mirror:
        buf.add_batch(O * MIR_O, A * MIR_A, R, NO * MIR_O, D, G, RT)


def her_relabel(raw, rng, gid=0):
    """raw: [(pos, quat, vel, rates, action|None)] (NED). Plant a virtual gate on the flown path.
    gid = index of the gate the segment was chasing (one-hot convention for virtual gates)."""
    S = len(raw) - 1
    if S < 12:
        return None
    lo = max(1, int(S * 0.4))
    cands = [t for t in range(lo, S) if np.linalg.norm(raw[t + 1][0] - raw[t][0]) > 0.02]
    if not cands:
        return None
    ts = int(rng.choice(cands))
    p0, p1 = raw[ts][0], raw[ts + 1][0]
    normal = p1 - p0; normal = normal / np.linalg.norm(normal)
    center = 0.5 * (p0 + p1)
    trans = []
    prev_d = float(np.linalg.norm(center - raw[0][0]))
    for t in range(ts + 1):
        pos, quat, vel, rates, act = raw[t]
        npos, nquat, nvel, nrates, _ = raw[t + 1]
        o = frd_obs(pos, quat, vel, rates, center, normal, gid)
        no = frd_obs(npos, nquat, nvel, nrates, center, normal, gid)
        d = float(np.linalg.norm(center - npos))
        r = CFG.w_prog * (prev_d - d) - CFG.step_penalty
        term = 1.0 if t == ts else 0.0
        if term:
            r += CFG.gate_bonus
        trans.append((o, act, r, no, term))
        prev_d = d
    return trans


# ---------- real-time helpers ----------
KC_RATE = 1.5      # inner-loop P gain on rate error (on top of exact feedforward)


def stream_rates(mav, des_frd, thrust, until, hb_state):
    """Inner rate loop @ ~250 Hz: cmd = (des + Kc*err)/GAIN — exact feedforward plus feedback on
    live odometry rates. Recreates the twin's crisp rate tracking on VQ1's sloppy command channel.
    Cmd clip = the measured linear range per axis (the old fixed +/-1.2 capped rates ~3 rad/s)."""
    while time.time() < until:
        err = des_frd - np.asarray(mav.rates, np.float64)
        cmd = np.clip((des_frd + KC_RATE * err) / GAIN, -CMD_MAX, CMD_MAX)
        mav.att(float(cmd[0]), float(cmd[1]), float(cmd[2]), thrust)
        if time.time() - hb_state[0] > 0.3:
            mav.hb(); hb_state[0] = time.time()
        time.sleep(0.004)


def stream_cmd(mav, cmd, until, hb_state):
    while time.time() < until:
        mav.att(*cmd)
        if time.time() - hb_state[0] > 0.3:
            mav.hb(); hb_state[0] = time.time()
        time.sleep(0.004)


def takeoff(mav, hb_state, alt=1.3, timeout=4.0):
    """Closed-loop takeoff: P-D altitude law inverted through the CALIBRATED thrust curve +
    attitude leveler; exits level at ~alt with ~zero vertical speed — the twin spawns hovering
    at gate height, so hand the policy that state. (The old open-loop boost was tuned for the
    stale thrust map and rocketed to ~11 m / 8 m/s climb with the real curve — the policy got
    an out-of-distribution handoff and tumbled from step 0.)"""
    z_t = float(mav.pos[2]) - alt               # NED: up = z decreasing
    t0 = time.time()
    while time.time() - t0 < timeout:
        g = rot_world_to_body(mav.quat, _DOWN_NED)      # level -> (0,0,1) in FRD
        # nose-up -> g0<0 -> need nose-down (negative FRD pitch rate) => +3*g0 (post rate-sign fix)
        des = np.array([-3.0 * float(g[1]), 3.0 * float(g[0]), 0.0])    # level the attitude
        vz = float(rot_body_to_world(mav.quat, np.asarray(mav.vel, np.float64))[2])
        err_up = float(mav.pos[2]) - z_t                # >0: below target
        a_up = float(np.clip(1.6 * err_up, -2.5, 2.5) + np.clip(1.2 * vz, -2.5, 2.5))
        spec = (9.81 + a_up) / max(0.6, float(g[2]))    # tilt-compensated specific force
        stream_rates(mav, des, thrust_cmd(spec / 9.81), time.time() + 0.03, hb_state)
        if abs(err_up) < 0.25 and abs(vz) < 0.4 \
                and abs(float(g[0])) < 0.1 and abs(float(g[1])) < 0.1:
            break
    stream_rates(mav, np.zeros(3), thrust_cmd(1.0), time.time() + 0.1, hb_state)


def fly_episode(mav, agent, gates, args, deterministic, ema, vis=None, capture=False, tcap=1.0,
                rcaps=None):
    """One episode on the real sim. Returns (transitions, raw, gates_passed, min_dist, reason,
    steps, frames). transitions/raw are None-free only when collecting (deterministic=False)."""
    ngates = len(gates)
    dt = 1.0 / CFG.control_hz
    max_steps = int(args.episode_s * CFG.control_hz)
    wait_reset(mav); time.sleep(CFG.reset_settle_s)
    t0 = time.time()
    while mav.t_us == 0 and time.time() - t0 < 3:
        mav.hb(); time.sleep(0.03)
    mav.arm(True)
    hb_state = [0.0]
    if args.takeoff:
        takeoff(mav, hb_state)
    else:
        # no scripted boost: hand the policy control on the ground (hover-centered thrust makes
        # liftoff a3>0 — learnable; the 1.5 s away-grace covers the launch)
        stream_rates(mav, np.zeros(3), HOVER_CMD, time.time() + 0.1, hb_state)
    active = max(0, min(int(mav.active_gate), ngates - 1))
    gate = gates[active]; gnorm = gate_normal_world(gate["quat"])
    side_prev = float(np.dot(mav.pos - gate["pos"], gnorm))
    prev_d = float(np.linalg.norm(gate["pos"] - mav.pos))
    min_d = prev_d
    obs = frd_obs(mav.pos, mav.quat, mav.vel, mav.rates, gate["pos"], gnorm, active)
    col0 = mav.collision_epoch
    trans = []; raw = []; frames = []; last_ftns = None
    away = 0; g_passed = 0; last_pass = 0; reason = "timeout"
    for step in range(max_steps):
        tick = time.time()
        a = agent.act(obs, deterministic=deterministic, ema=ema)
        raw.append((mav.pos.copy(), mav.quat.copy(), np.array(mav.vel, float),
                    np.array(mav.rates, float), a.copy()))
        des, thr = act_desired(a, tcap, rcaps)
        stream_rates(mav, des, thr, tick + dt, hb_state)
        if capture and vis is not None:
            fr = vis.get()
            if fr is not None and fr[0] != last_ftns:
                frames.append(fr[1]); last_ftns = fr[0]
        # pass detection: GEOMETRIC plane-crossing at this control step, like the twin env.
        # The sim's RACE_STATUS is ~4 Hz -> up to 0.25 s late = ~3 m at race speed, so the +50
        # and the retarget landed well past the gate, smearing credit exactly where the greedy
        # mean must learn to commit. Sim's active_gate stays as a fallback signal.
        rel = mav.pos - gate["pos"]
        side = float(np.dot(rel, gnorm))
        radial = float(np.linalg.norm(rel - side * gnorm))
        passed = (side_prev < 0.0 <= side and radial < 0.5 * float(gate.get("w", 2.72))) \
            or int(mav.active_gate) > active
        if passed:
            active = min(active + 1, ngates - 1); gate = gates[active]
            gnorm = gate_normal_world(gate["quat"])
            g_passed += 1; last_pass = len(raw)
            side_prev = float(np.dot(mav.pos - gate["pos"], gnorm))
        else:
            side_prev = side
        d = float(np.linalg.norm(gate["pos"] - mav.pos))
        min_d = min(min_d, d)
        # grace window: don't arm the away-terminal in the first ~1.5 s — residual takeoff drift
        # would kill episodes before the policy has a chance to turn the nose around
        grace = int(1.5 * CFG.control_hz)
        away = 0 if (passed or d <= prev_d or step < grace) else away + 1
        collided = mav.collision_epoch > col0
        strayed = d > CFG.stray_dist
        r, done, reason_s = step_reward(prev_d if not passed else d, d, passed, collided,
                                        strayed, CFG)
        if passed and g_passed >= args.gates and not done:
            r += CFG.finish_bonus; done, reason_s = True, "finish"
        if not done and away >= int(args.away * CFG.control_hz / 30.0):
            r += min(0.0, CFG.fail_reward(d))          # away-exploit fix: retreat never pays
            done, reason_s = True, "away"
        next_obs = frd_obs(mav.pos, mav.quat, mav.vel, mav.rates, gate["pos"], gnorm, active)
        term = reason_s in ("finish", "crash", "stray", "away")
        trans.append((obs, a, r, next_obs, float(term)))
        obs = next_obs; prev_d = d
        if done:
            reason = reason_s; break
    mav.arm(False)
    for _ in range(6):
        mav.att(0, 0, 0, 0); mav.hb(); time.sleep(0.02)
    raw.append((mav.pos.copy(), mav.quat.copy(), np.array(mav.vel, float),
                np.array(mav.rates, float), None))
    return trans, raw, g_passed, min_d, reason, len(trans), frames, last_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=r"C:/Users/satas/Downloads/AI-GP Simulator v1.0.3385-VQ1/PyAIPilotExample-v1/racer_mujoco/runs_mj/mj_gate5.pt")
    ap.add_argument("--eval", type=int, default=0, help="N greedy episodes only (adapter probe)")
    ap.add_argument("--hz", type=float, default=30.0, help="control rate (twin trained at 30)")
    ap.add_argument("--gate_onehot", type=int, default=1, help="append one-hot active-gate index to the obs (fixes critic value aliasing across gates)")
    ap.add_argument("--takeoff", type=int, default=0, help="1 = scripted closed-loop takeoff before handoff (default: policy flies from the ground)")
    ap.add_argument("--gates", type=int, default=6)
    ap.add_argument("--episode_s", type=float, default=30.0)
    ap.add_argument("--away", type=int, default=15)
    ap.add_argument("--ent", type=float, default=-2.0)
    ap.add_argument("--nstep", type=int, default=5)
    ap.add_argument("--her", type=int, default=2)
    ap.add_argument("--silw", type=float, default=1.0)
    ap.add_argument("--sil_every", type=int, default=1, help="GPU updates are cheap; SIL every update")
    ap.add_argument("--mirror", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--utd_cap", type=float, default=1.0, help="max learner updates per env step")
    ap.add_argument("--cwarm", type=int, default=1500, help="critic-only warm updates (twin critic -> real dynamics)")
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--eval_batch", type=int, default=10)
    # Authority ramps default OFF: the calibrated physical-unit map gives the policy the same
    # authority it trained with in the twin — ramps would make it re-learn its own limits.
    ap.add_argument("--tcap_start", type=float, default=1.0, help="thrust-authority cap at ep 0 (>=TOP_CMD disables)")
    ap.add_argument("--tcap_ep", type=int, default=1000, help="episode at which thrust reaches full authority")
    ap.add_argument("--ep0", type=int, default=0, help="episode offset for the authority ramps (set on resume so restarts don't reset them)")
    ap.add_argument("--rcap_start", type=float, default=4.0, help="roll-rate cap (rad/s) at ep 0")
    ap.add_argument("--rcap_ep", type=int, default=1000, help="roll reaches full authority here")
    ap.add_argument("--pcap_start", type=float, default=4.0, help="pitch-rate cap (rad/s) at ep 0")
    ap.add_argument("--pcap_ep", type=int, default=300, help="pitch reaches full authority here (fast — it's the fly-forward axis)")
    ap.add_argument("--ycap_start", type=float, default=4.0, help="yaw-rate cap (rad/s) at ep 0")
    ap.add_argument("--ycap_ep", type=int, default=1000, help="yaw reaches full authority here")
    args = ap.parse_args()

    global GATE_ONEHOT, MIR_O
    GATE_ONEHOT = bool(args.gate_onehot)
    if GATE_ONEHOT:
        CFG.obs_dim = 15 + NGATES_OH
        MIR_O = np.concatenate([MIR_O, np.ones(NGATES_OH, np.float32)])
    CFG.critic_dropout = 0.0          # twin checkpoints use plain critics
    CFG.control_hz = args.hz
    CFG.target_entropy = args.ent
    CFG.batch_size = args.batch
    CFG.sil_weight = args.silw
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = CFG.run_dir; os.makedirs(run_dir, exist_ok=True)
    mpath = os.path.join(run_dir, "metrics.jsonl")

    agent = SAC(CFG, dev)
    if args.resume and os.path.exists(args.resume):
        load_resume(agent, args.resume, CFG.act_dim)
        print(f"warm-started from {os.path.basename(args.resume)}", flush=True)

    mav = MavLink(CFG.mav_addr)
    print(f"connected sys={mav.sys} device={dev}", flush=True)
    for attempt in range(5):
        if mav.capture_gates():
            break
        print(f"gate capture inconclusive (attempt {attempt + 1}); retrying", flush=True)
    gates = mav.gates
    print(f"gates: {len(gates)}; gate0 @ {np.round(gates[0]['pos'], 2)}", flush=True)

    # ---------------- eval-only mode (adapter probe) ----------------
    if args.eval > 0:
        for k in range(args.eval):
            _, _, g, md, reason, steps, _, _ = fly_episode(mav, agent, gates, args,
                                                           deterministic=True, ema=True)
            print(f"[EVAL {k}] gates+{g} min_d={md:.1f} end={reason} steps={steps}", flush=True)
        mav.stop(); return

    # ---------------- training ----------------
    open(mpath, "w").close()
    vis = None
    try:
        from racer_state.vision_rx import VisionRX
        vis = VisionRX(CFG.vision_port)
        print("vision rx up (milestone videos)", flush=True)
    except Exception as e:
        print(f"vision rx unavailable ({e}); videos disabled", flush=True)

    bpath = os.path.join(run_dir, "buffers_vq1.pt")
    if os.path.exists(bpath):
        try:
            load_buffers(agent, bpath)
        except Exception as e:
            print(f"buffer restore failed ({e}); starting with empty buffers", flush=True)

    shared = {"steps": 0, "updates": 0, "m": {}, "warm": args.cwarm}

    def _learner():
        while True:
            if agent.buf.size < CFG.update_after or \
                    shared["updates"] > shared["steps"] * args.utd_cap + 500:
                time.sleep(0.005); continue
            if shared["warm"] > 0:
                agent.update(metrics=False, critic_only=True)
                shared["warm"] -= 1
                if shared["warm"] == 0:
                    print("critic warm-up done (learner)", flush=True)
            elif shared["updates"] % 200 == 0:
                shared["m"] = agent.update(metrics=True)
            else:
                if args.sil_every > 1 and shared["updates"] % args.sil_every != 0:
                    sw = CFG.sil_weight; CFG.sil_weight = 0.0
                    agent.update(metrics=False); CFG.sil_weight = sw
                else:
                    agent.update(metrics=False)
            shared["updates"] += 1
    threading.Thread(target=_learner, daemon=True).start()
    print(f"LEARNER THREAD on ({dev}, utd_cap {args.utd_cap})", flush=True)

    rng = np.random.default_rng(7)
    ep = 0; total = 0; fin100 = []
    solved = set()
    t0 = time.time()
    while True:
        E = ep + args.ep0
        tc = thrust_cap(E, args.tcap_start, args.tcap_ep)
        rc = np.array([rate_cap(E, args.rcap_start, args.rcap_ep),
                       rate_cap(E, args.pcap_start, args.pcap_ep),
                       rate_cap(E, args.ycap_start, args.ycap_ep)])
        trans, raw, g, md, reason, steps, _, last_pass = fly_episode(
            mav, agent, gates, args, deterministic=False, ema=False, tcap=tc, rcaps=rc)
        total += steps; shared["steps"] = total
        fin100.append(int(g >= args.gates))
        cols = pack_cols(nstep_pack(trans, args.nstep, CFG.gamma))
        add_pack(agent.buf, cols, args.mirror)
        if g >= 1:
            add_pack(agent.elite, cols, args.mirror)
        if reason != "finish" and args.her > 0:
            seg = raw[last_pass:]
            gid_fail = min(g, len(gates) - 1)          # gate the failed segment was chasing
            for _ in range(args.her):
                v = her_relabel(seg, rng, gid_fail)
                if v:
                    add_pack(agent.buf, pack_cols(nstep_pack(v, args.nstep, CFG.gamma)),
                             args.mirror)
        pr = float(np.mean(fin100[-100:]))
        rec = {"ep": ep, "t": time.time(), "step": total, "reward": 0.0,
               "min_dist": md, "reason": reason, "gates": int(g),
               "passed": int(g >= args.gates), "pass_rate100": pr,
               "steps_per_s": steps / max(1e-6, steps / CFG.control_hz),
               "buf": agent.buf.size, "updates": shared["updates"], **shared["m"]}
        rec["reward"] = float(sum(t[2] for t in trans))
        print(f"ep {ep:5d} gates+{g} end={reason:7s} min_d={md:5.1f} rew={rec['reward']:+7.1f} "
              f"| fin%={100 * pr:3.0f} buf={agent.buf.size} upd={shared['updates']} "
              f"utd={shared['updates'] / max(1, total):.2f}", flush=True)
        ep += 1

        if ep % 20 == 0:
            agent.save(os.path.join(run_dir, "autosave_vq1.pt"))
            save_buffers(agent, os.path.join(run_dir, "buffers_vq1.pt"))
        if ep % args.eval_every == 0 and agent.buf.size >= CFG.update_after \
                and shared["warm"] <= 0:
            counts = [0] * args.gates; mds = []
            frames_best = []; g_best = -1
            for k in range(args.eval_batch):
                cap = vis is not None and k == 0
                _, _, ge, mde, re, _, fr, _ = fly_episode(mav, agent, gates, args,
                                                          deterministic=True, ema=True,
                                                          vis=vis, capture=cap, tcap=tc, rcaps=rc)
                for j in range(min(ge, args.gates)):
                    counts[j] += 1
                mds.append(mde)
                if ge > g_best and fr:
                    g_best = ge; frames_best = fr
            cstr = " ".join(f"g{k}:{c}/{args.eval_batch}" for k, c in enumerate(counts))
            rec2 = {"ep": ep, "t": time.time(), "eval_g0": counts[0],
                    "eval_gates": counts, "eval_mind": float(np.mean(mds))}
            if len(counts) > 1:
                rec2["eval_g1"] = counts[1]
            with open(mpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec2) + "\n")
            print(f"[EVAL ep{ep}] GREEDY {cstr} (mind {np.mean(mds):.1f}) tcap={tc:.2f} "
                  f"rpy_cap={np.round(rc, 1)} +{(time.time() - t0) / 60:.1f} min", flush=True)
            for k in range(args.gates):
                if counts[k] >= args.eval_batch and k not in solved:
                    solved.add(k)
                    agent.save(os.path.join(run_dir, f"vq1_gate{k}.pt"))
                    if frames_best:
                        write_video(frames_best, os.path.join(run_dir, f"vq1_gate{k}.mp4"),
                                    CFG.video_fps)
                    print(f"  *** VQ1 {args.eval_batch}/{args.eval_batch} through gate {k} "
                          f"at +{(time.time() - t0) / 60:.1f} min (ep {ep}) ***", flush=True)
        with open(mpath, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
