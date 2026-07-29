"""Single-process online SAC on ground-truth state (NEW sim). Connect -> capture gates -> loop
episodes: reset, arm, fly at control_hz building obs from odometry, exact-distance reward, add
transition, run SAC updates. Logs per-episode metrics to runs/metrics.jsonl for the dashboard.

  python -m racer_state.train                 # train
  python -m racer_state.train --eval 20       # 20 greedy eval episodes, report gate-0 pass rate
"""
from __future__ import annotations
import argparse, json, os, sys, time
os.environ.setdefault("SCIPY_ARRAY_API", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from racer_state.config import CFG
from racer_state.mav_io import MavLink
from racer_state.state_obs import build_obs, dist_to_gate
from racer_state.action_map import action_to_cmd
from racer_state.reward import step_reward
from racer_state.sac import SAC


def wait_reset(mav, timeout=6.0):
    """Reset the sim AND wait until the race state is back at gate 0. Waiting only for the odometry
    reset_counter isn't enough: RACE_STATUS is ~4 Hz, so active_gate can still read a STALE value
    (e.g. 1, carried over from a prior gate-0 pass) when the episode starts — which would make the
    whole episode target gate 1 before the drone has even reached gate 0."""
    rc0 = mav.reset_count
    mav.active_gate = -1          # invalidate; the next post-reset RACE_STATUS will set it to 0
    mav.reset()
    t0 = time.time()
    reset_seen = False
    while time.time() - t0 < timeout:
        if mav.reset_count != rc0:
            reset_seen = True
        if reset_seen and mav.active_gate == 0:   # race genuinely reset to the first gate
            return True
        mav.hb(); time.sleep(0.02)
    return True   # proceed anyway (fail-open)


def greedy_episode(mav, agent, gates, cfg, cep, vis=None, capture=False):
    """Fly one deterministic (mean-action) episode; no learning. Returns (gates_passed, min_dist,
    frames). If capture and vis given, grabs the latest camera frame each control step."""
    ngates = len(gates)
    dt = 1.0 / cfg.control_hz
    max_steps = int(cfg.episode_cap_s * cfg.control_hz)
    wait_reset(mav); time.sleep(cfg.reset_settle_s)
    t0 = time.time()
    while mav.t_us == 0 and time.time() - t0 < 3:
        mav.hb(); time.sleep(0.03)
    mav.arm(True)
    active = max(0, min(int(mav.active_gate), ngates - 1))   # every episode starts at gate 0
    gate = gates[active]
    prev_d = dist_to_gate(mav.pos, gate)
    min_d = prev_d
    obs = build_obs(mav, gate, cfg)
    col0 = mav.collision_epoch
    ep_gates = 0; away = 0; last_hb = 0.0; frames = []; last_ftns = None
    for step in range(max_steps):
        tick = time.time()
        a = agent.act(obs, deterministic=True)
        cmd = action_to_cmd(a, cfg, cep)
        while time.time() - tick < dt:
            mav.att(*cmd)
            if time.time() - last_hb > 0.3:
                mav.hb(); last_hb = time.time()
            time.sleep(0.004)
        if capture and vis is not None:
            fr = vis.get()
            if fr is not None and fr[0] != last_ftns:
                frames.append(fr[1]); last_ftns = fr[0]
        cur = int(mav.active_gate)
        passed = cur > active
        finished = cur >= ngates
        if passed:
            active = min(cur, ngates - 1); gate = gates[active]; ep_gates += 1
        d = dist_to_gate(mav.pos, gate); min_d = min(min_d, d)
        away = 0 if (passed or d <= prev_d) else away + 1
        collided = mav.collision_epoch > col0
        done = collided or finished or d > cfg.stray_dist or away >= cfg.max_away_frames
        obs = build_obs(mav, gate, cfg); prev_d = d
        if done:
            break
    mav.arm(False)
    for _ in range(6):
        mav.att(0, 0, 0, 0); mav.hb(); time.sleep(0.02)
    return ep_gates, min_d, frames


def write_video(frames, path, fps=30):
    if not frames:
        return False
    import cv2
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    return os.path.exists(path)


def run(eval_n=0, resume=False):
    cfg = CFG
    os.makedirs(cfg.run_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mav = MavLink(cfg.mav_addr)
    print(f"connected sys={mav.sys} device={device}", flush=True)
    for attempt in range(5):
        if mav.capture_gates():
            break
        print(f"WARNING: gate capture inconclusive (attempt {attempt+1}); retrying", flush=True)
    gates = mav.gates
    ngates = len(gates) if gates else 0
    g0 = float(np.linalg.norm(gates[0]["pos"] - mav.pos)) if ngates else -1
    print(f"gates: {ngates}; gate0 @ {np.round(gates[0]['pos'],2)} (dist {g0:.1f}m from spawn)", flush=True)

    agent = SAC(cfg, device)
    metrics_path = os.path.join(cfg.run_dir, "metrics.jsonl")
    ckpt_path = os.path.join(cfg.run_dir, "actor.pt")
    is_eval = eval_n > 0
    if is_eval and os.path.exists(ckpt_path):
        agent.load_actor(ckpt_path); print(f"loaded {ckpt_path} for eval", flush=True)

    dt = 1.0 / cfg.control_hz
    max_steps = int(cfg.episode_cap_s * cfg.control_hz)
    total_steps = 0
    ep = 0
    cep = 0            # curriculum episode: frozen during the random-exploration phase
    reward_ema = 0.0   # EMA of episode task-reward; drives the adaptive pitch penalty
    pen_thrust_on = False   # fallback (auto-enabled if not converging): also penalize thrust
    state_path = os.path.join(cfg.run_dir, "train_state.json")
    if resume and os.path.exists(ckpt_path):
        agent.load_full(ckpt_path)
        if os.path.exists(state_path):
            st = json.load(open(state_path))
            cep = st.get("cep", 100); reward_ema = st.get("reward_ema", 0.0)
            total_steps = st.get("total_steps", cfg.start_random_steps); ep = st.get("ep", 0)
        else:                       # no sidecar (older ckpt): warm defaults, skip random phase
            cep = 100; total_steps = cfg.start_random_steps
        print(f"RESUMED from {ckpt_path}: ep={ep} cep={cep} reward_ema={reward_ema:.1f} "
              f"(skipping random phase; buffer refills from the trained policy)", flush=True)
    n_eps = eval_n if is_eval else 10_000_000
    passes = 0

    # vision receiver for milestone videos (state control doesn't use vision, so port 5600 is free)
    vis = None
    if not is_eval:
        try:
            from racer_state.vision_rx import VisionRX
            vis = VisionRX(cfg.vision_port)
            print("vision rx up (for milestone videos)", flush=True)
        except Exception as e:
            print(f"vision rx unavailable ({e}); videos disabled", flush=True)
    gate0_done = os.path.exists(os.path.join(cfg.run_dir, "gate0_reached.pt"))
    gate1_done = os.path.exists(os.path.join(cfg.run_dir, "gate1_reached.pt"))

    def milestone(tag, frames):
        """Save a labelled checkpoint + write the flight video for a reached milestone."""
        try:
            agent.save(os.path.join(cfg.run_dir, f"{tag}.pt"))
            vp = os.path.join(cfg.run_dir, f"{tag}.mp4")
            ok = write_video(frames, vp, cfg.video_fps)
            print(f"*** MILESTONE {tag}: saved {tag}.pt" +
                  (f" + video {vp} ({len(frames)} frames)" if ok else " (no video frames)"), flush=True)
        except Exception as e:
            print(f"milestone {tag} save/video error: {e}", flush=True)

    while ep < n_eps:
        wait_reset(mav)
        time.sleep(cfg.reset_settle_s)
        # fresh state
        t0 = time.time()
        while mav.t_us == 0 and time.time() - t0 < 3:
            mav.hb(); time.sleep(0.03)
        mav.arm(True)
        active = max(0, min(int(mav.active_gate), ngates - 1))   # every episode starts at gate 0
        gate = gates[active]
        prev_dist = dist_to_gate(mav.pos, gate)
        start_dist = prev_dist
        obs = build_obs(mav, gate, cfg)
        col0 = mav.collision_epoch
        ep_reward = 0.0; ep_task = 0.0; ep_gates = 0; last_hb = 0.0; pitch_pen_total = 0.0
        min_dist = start_dist         # closest approach to the active gate this episode
        reason = "timeout"; m_last = {}
        ep_trans = []                 # this episode's transitions (for the elite buffer)
        away_count = 0                # consecutive frames moving away from the active gate
        is_random_ep = (not is_eval) and total_steps < cfg.start_random_steps
        w_pitch = cfg.pitch_pen_weight(reward_ema)   # adaptive pitch-rate penalty weight this episode
        ep_t0 = time.time(); upd_t = 0.0; upd_n = 0   # timing
        for step in range(max_steps):
            tick = time.time()
            if is_random_ep:
                a = np.random.uniform(-1, 1, cfg.act_dim).astype(np.float32)
            else:
                a = agent.act(obs, deterministic=is_eval)
            cmd = action_to_cmd(a, cfg, cep)   # caps use the curriculum ep (frozen during random)
            # hold the command across the control step (sim wants a continuous stream)
            while time.time() - tick < dt:
                mav.att(*cmd)
                if time.time() - last_hb > 0.3:
                    mav.hb(); last_hb = time.time()
                time.sleep(0.004)
            # observe next state
            cur_active = int(mav.active_gate)
            passed = cur_active > active
            finished = cur_active >= ngates          # passed the final gate -> completed the track
            if passed:
                active = min(cur_active, ngates - 1)
                gate = gates[active]
                ep_gates += 1
            curr_dist = dist_to_gate(mav.pos, gate)
            min_dist = min(min_dist, curr_dist)      # closest approach to whatever gate it chases
            # consecutive frames moving away from the active gate (reset on a pass — target changed)
            away_count = 0 if passed or curr_dist <= prev_dist else away_count + 1
            collided = mav.collision_epoch > col0
            strayed = curr_dist > cfg.stray_dist
            r, done, reason_s = step_reward(prev_dist if not passed else curr_dist,
                                            curr_dist, passed, collided, strayed, cfg)
            if finished:                             # completed the track -> terminal success
                r += cfg.finish_bonus
                done, reason_s = True, "finish"
            elif not done and away_count >= cfg.max_away_frames:   # regressing -> cut it short
                r += cfg.fail_reward(curr_dist)   # distance-scaled: gentle if near the gate, harsh if far
                done, reason_s = True, "away"
            # adaptive pitch+roll rate penalty: exp in |rate request|, weight scales with performance
            k = cfg.pitch_pen_sharpness
            pen_terms = (np.exp(k * abs(a[1])) - 1.0) + (np.exp(k * abs(a[0])) - 1.0)
            if pen_thrust_on:                        # fallback: also damp thrust deviation from hover
                pen_terms += np.exp(k * abs(a[3])) - 1.0
            pp = w_pitch * float(pen_terms)
            ra = r - pp                              # reward the agent actually receives
            next_obs = build_obs(mav, gate, cfg)
            # failures are TRUE terminals (value = fail reward, no bootstrap) — bootstrapping off a
            # receding/failing state spirals the critic downward. Only timeout truncates (bootstrap).
            term = reason_s in ("crash", "stray", "away", "finish")
            if not is_eval:
                ep_trans.append((obs, a, ra, next_obs, float(term)))
                agent.buf.add(obs, a, ra, next_obs, float(term))
                if agent.buf.size >= cfg.update_after:
                    _u = time.time()
                    for _ in range(cfg.updates_per_step):
                        m_last = agent.update()
                    upd_t += time.time() - _u; upd_n += cfg.updates_per_step
            obs = next_obs; prev_dist = curr_dist
            ep_reward += ra; ep_task += r; pitch_pen_total += pp; total_steps += 1
            if done:
                reason = reason_s; break
        # episode end
        mav.arm(False)
        for _ in range(6):
            mav.att(0, 0, 0, 0); mav.hb(); time.sleep(0.02)
        reached_g0 = ep_gates >= 1
        passes += int(reached_g0)
        # elite: keep this episode's transitions if it passed a gate OR got close (oversampled later)
        is_elite = (ep_gates >= 1) or (min_dist < cfg.elite_dist)
        if not is_eval and is_elite:
            agent.add_elite(ep_trans)
        # update the reward EMA (task reward) that drives the adaptive pitch penalty
        if not is_eval:
            reward_ema = cfg.reward_ema_beta * reward_ema + (1 - cfg.reward_ema_beta) * ep_task
        ep_dur = time.time() - ep_t0
        ms_per_step = 1000.0 * ep_dur / max(1, step + 1)      # env step wall time (control loop)
        upd_ms = 1000.0 * upd_t / max(1, upd_n)               # avg SAC update time
        steps_per_s = (step + 1) / max(1e-6, ep_dur)
        rec = {"ep": ep, "cep": cep, "t": time.time(), "steps": step + 1, "reward": ep_reward,
               "ms_per_step": ms_per_step, "upd_ms": upd_ms, "steps_per_s": steps_per_s,
               "ep_dur": ep_dur,
               "task_reward": ep_task, "gates": ep_gates, "reason": reason, "reached_g0": reached_g0,
               "start_dist": start_dist, "min_dist": min_dist, "elite": is_elite,
               "elite_buf": agent.elite.size, "total_steps": total_steps, "eval": is_eval,
               "random_ep": is_random_ep, "reward_ema": reward_ema, "w_pitch": w_pitch,
               "pitch_pen": pitch_pen_total, "thrust_cap": cfg.thrust_cap(cep),
               "rollyaw_cap": cfg.rollyaw_cap(cep), **m_last}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        tag = "EVAL" if is_eval else f"ep {ep}"
        print(f"[{tag}] steps={step+1} rew={ep_reward:+.1f} gates+{ep_gates} end={reason} "
              f"min_d={min_dist:.1f}/{start_dist:.0f} | ema={reward_ema:+.1f} wpitch={w_pitch:.3f} "
              f"buf={agent.buf.size} elite={agent.elite.size} " +
              (f"c={m_last.get('c_loss',0):.2f} alpha={m_last.get('alpha',0):.3f} "
               f"Q={m_last.get('q',0):.1f}" if m_last else "random"),
              flush=True)
        if not is_eval and ep % 10 == 0:
            agent.save(ckpt_path)
            with open(state_path, "w") as f:   # sidecar so --resume restores curriculum + EMA
                json.dump({"ep": ep, "cep": cep, "reward_ema": reward_ema,
                           "total_steps": total_steps}, f)
        ep += 1
        if not is_random_ep:
            cep += 1        # advance the curriculum only once we're past random exploration

        # periodic greedy self-eval + milestone recording (past the random phase)
        if (not is_eval) and ep % cfg.eval_every == 0 and total_steps >= cfg.start_random_steps:
            try:
                g0 = g1 = 0
                for _ in range(cfg.eval_batch):
                    eg, _, _ = greedy_episode(mav, agent, gates, cfg, cep)
                    g0 += int(eg >= 1); g1 += int(eg >= 2)
                print(f"[EVAL ep{ep}] gate0 {g0}/{cfg.eval_batch}  gate1 {g1}/{cfg.eval_batch}", flush=True)
                if (not pen_thrust_on) and ep >= cfg.pen_thrust_auto_ep and g0 < cfg.pen_thrust_auto_g0:
                    pen_thrust_on = True
                    print(f"[FALLBACK] not converging by ep{ep} (gate0 {g0}/{cfg.eval_batch}) -> "
                          f"adding thrust to the adaptive penalty", flush=True)
                with open(metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"ep": ep, "eval": True, "t": time.time(),
                                        "eval_g0": g0, "eval_g1": g1, "n": cfg.eval_batch}) + "\n")
                if (not gate0_done) and g0 >= cfg.eval_pass_threshold:
                    gate0_done = True
                    eg, _, frames = greedy_episode(mav, agent, gates, cfg, cep, vis, capture=True)
                    milestone("gate0_reached", frames)
                if (not gate1_done) and g1 >= cfg.eval_g1_threshold:
                    gate1_done = True
                    eg, _, frames = greedy_episode(mav, agent, gates, cfg, cep, vis, capture=True)
                    milestone("gate1_reached", frames)
            except Exception as e:
                print(f"[EVAL] error (continuing training): {e}", flush=True)

        # progress reel: a greedy flight video every video_every episodes
        if (not is_eval) and vis is not None and ep % cfg.video_every == 0 \
                and total_steps >= cfg.start_random_steps:
            try:
                eg, md, frames = greedy_episode(mav, agent, gates, cfg, cep, vis, capture=True)
                vdir = os.path.join(cfg.run_dir, "videos"); os.makedirs(vdir, exist_ok=True)
                vp = os.path.join(vdir, f"ep{ep:05d}_g{eg}_d{md:.1f}.mp4")
                if write_video(frames, vp, cfg.video_fps):
                    print(f"[VIDEO] ep{ep}: {vp} ({len(frames)} frames, gates+{eg}, min_d {md:.1f})", flush=True)
            except Exception as e:
                print(f"[VIDEO] error (continuing training): {e}", flush=True)

    if is_eval:
        print(f"\nRESULT: {passes}/{eval_n} passed gate 0 ({passes/max(1,eval_n):.0%}) "
              f"-> {'PASS' if passes >= eval_n else 'NOT-YET'}", flush=True)
    mav.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=0, help="run N greedy eval episodes")
    ap.add_argument("--resume", action="store_true", help="continue from runs/actor.pt")
    ap.add_argument("--ent", type=float, default=None,
                    help="override target_entropy. MuJoCo showed -2 sharpens the greedy mean so it "
                         "COMMITS through the gate (default -1 parks it just short). Try --resume --ent -2.")
    args = ap.parse_args()
    if args.ent is not None:
        CFG.target_entropy = args.ent
        print(f"target_entropy override: {CFG.target_entropy}", flush=True)
    run(eval_n=args.eval, resume=args.resume)


if __name__ == "__main__":
    main()
