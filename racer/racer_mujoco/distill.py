"""DAgger distillation: the solved state policy (mj_gate5.pt) teaches a conv net to fly the full
track from 3 spaced, gate-color-filtered frames + IMU proprio.

The teacher sees ground-truth state and labels EVERY step; the student flies from pixels (after a
short teacher-flown warm phase) so the dataset covers the student's own visitation distribution —
that's what makes DAgger converge where plain behavior cloning drifts.

  python -m racer_mujoco.distill --envs 8
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
import mujoco

from racer_state.config import CFG
from racer_state.sac import SAC
from racer_state.geom import rot_world_to_body
from racer_mujoco.env import QuadEnv
from racer_mujoco.train_mj import obs_of
from racer_mujoco.vision import gate_filter, FrameRing, VisionBuffer, rand_shift, IMG
from racer_mujoco.nets_vision import VisionActor

RUN_DIR = os.path.join(os.path.dirname(__file__), "runs_mj")
AWAY = 15
_DOWN = np.array([0.0, 0.0, -1.0])


def proprio_of(env):
    """Body rates + gravity direction + body-frame velocity (what a real drone's IMU+VIO gives).
    Velocity was added after the 6-dim variant plateaued at g0 ~14/20: the teacher's actions are
    strongly velocity-dependent and 3 spaced frames under-determine speed -> regression ceiling."""
    return np.concatenate([env.rates / CFG.rate_scale,
                           rot_world_to_body(env.quat, _DOWN),
                           rot_world_to_body(env.quat, env.vel) / CFG.vel_scale]).astype(np.float32)


class VisionEnv:
    """QuadEnv + fpv renderer + frame ring, with buffer-slot tracking."""

    def __init__(self, seed, clutter_seed, episode_s, spacing, buf):
        self.env = QuadEnv(episode_s=episode_s, seed=seed, clutter_seed=clutter_seed)
        self.r = mujoco.Renderer(self.env.model, IMG, IMG)
        self.ring = FrameRing(spacing=spacing)
        self.buf = buf

    def _frame(self):
        self.r.update_scene(self.env.data, camera="fpv")
        return gate_filter(self.r.render())

    def reset(self):
        self.env.reset()
        f = self._frame()
        self.ring.reset(f, self.buf.add_frame(f))
        return self

    def step(self, a):
        passed, crashed = self.env.step(a)
        f = self._frame()
        self.ring.push(f, self.buf.add_frame(f))
        return passed, crashed


def student_act_batch(student, stacks, props, device, deterministic):
    imgs = torch.as_tensor(np.stack(stacks), device=device).float().div_(255.0)
    prop = torch.as_tensor(np.stack(props), device=device)
    return student.act(imgs, prop, deterministic=deterministic).cpu().numpy()


def eval_student(student, ve, gates_target, device, n=20):
    """Student flies greedy from pixels on the (unseen-clutter) eval env."""
    reached = []; mds = []
    for _ in range(n):
        ve.reset()
        e = ve.env
        prev = e.dist_to_gate(); away = 0; g = 0; md = prev
        for _ in range(e.max_steps):
            a = student_act_batch(student, [ve.ring.stack()], [proprio_of(e)], device, True)[0]
            passed, crashed = ve.step(a)
            d = e.dist_to_gate(); md = min(md, d)
            if passed:
                g += 1; away = 0; prev = d
                if g >= gates_target:
                    break
                continue
            away = 0 if d < prev else away + 1; prev = d
            if crashed or d > CFG.stray_dist or away >= AWAY:
                break
        reached.append(g); mds.append(md)
    counts = [sum(1 for x in reached if x > k) for k in range(gates_target)]
    return counts, float(np.mean(mds))


def record_student(student, ve, path, device, gates_target, width=384, height=256, fps=30):
    """Chase-cam mp4 + a side-by-side of what the student SEES (filtered stack, newest frame)."""
    try:
        import cv2
        # cache the chase renderer on the env: creating/GC-ing GL contexts mid-run silently
        # blacks out other live renderers (Windows WGL quirk — probed)
        if not hasattr(ve, "chase"):
            ve.chase = mujoco.Renderer(ve.env.model, height, width)
        chase = ve.chase
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = 90.0; cam.elevation = -18.0; cam.distance = 10.0
        ve.reset(); e = ve.env
        prev = e.dist_to_gate(); away = 0; g = 0
        frames = []
        for _ in range(e.max_steps):
            cam.lookat[:] = 0.5 * (e.pos + e.gate_pos)
            chase.update_scene(e.data, cam)
            rgb = chase.render().copy()
            see = cv2.resize(ve.ring.stack()[0], (height, height),
                             interpolation=cv2.INTER_NEAREST)
            see = cv2.cvtColor(see, cv2.COLOR_GRAY2RGB)
            frames.append(np.concatenate([rgb, see], axis=1))
            a = student_act_batch(student, [ve.ring.stack()], [proprio_of(e)], device, True)[0]
            passed, crashed = ve.step(a)
            d = e.dist_to_gate()
            if passed:
                g += 1; away = 0; prev = d
                if g >= gates_target:
                    break
                continue
            away = 0 if d < prev else away + 1; prev = d
            if crashed or d > CFG.stray_dist or away >= AWAY:
                break
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width + height, height))
        for f in frames:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        print(f"  video saved: {os.path.basename(path)} [{g} gates, {len(frames)} frames]",
              flush=True)
    except Exception as ex:
        print(f"  video failed: {ex}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--spacing", type=int, default=5)
    ap.add_argument("--gates", type=int, default=6)
    ap.add_argument("--episode_s", type=float, default=40.0)
    ap.add_argument("--teacher", default=os.path.join(RUN_DIR, "mj_gate5.pt"))
    ap.add_argument("--beta_min", type=float, default=0.1, help="floor of the teacher-mixture probability")
    ap.add_argument("--beta_anneal_eps", type=int, default=1500, help="episodes to anneal the per-STEP teacher mixture from 1.0 to beta_min")
    ap.add_argument("--tnoise", type=float, default=0.05, help="action noise on teacher-driven steps (state diversity)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aug", type=int, default=4, help="DrQ random-shift pad (0=off)")
    ap.add_argument("--cap", type=int, default=250_000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--video_every", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RUN_DIR, exist_ok=True)
    mpath = os.path.join(RUN_DIR, "metrics.jsonl"); open(mpath, "w").close()

    CFG.critic_dropout = 0.0
    teacher = SAC(CFG, "cpu")
    teacher.load_full(args.teacher)
    student = VisionActor().to(dev)
    opt = torch.optim.Adam(student.parameters(), args.lr)
    buf = VisionBuffer(cap=args.cap, device=dev)

    N = args.envs
    envs = [VisionEnv(seed=i, clutter_seed=100 + i, episode_s=args.episode_s,
                      spacing=args.spacing, buf=buf).reset() for i in range(N)]
    eval_ve = VisionEnv(seed=999, clutter_seed=9999, episode_s=args.episode_s,
                        spacing=args.spacing, buf=VisionBuffer(cap=64, device=dev))
    print(f"DAgger distill | {N} envs | teacher={os.path.basename(args.teacher)} | device={dev} "
          f"| spacing={args.spacing} aug={args.aug}", flush=True)

    shared = {"updates": 0, "loss": 0.0, "stop": False}

    def _learner():
        ema = None
        while not shared["stop"]:
            if buf.size < 2000:
                time.sleep(0.01); continue
            imgs, prop, act = buf.sample(args.batch)
            if args.aug > 0:
                imgs = rand_shift(imgs, args.aug)
            mean, _ = student(imgs, prop)
            loss = F.mse_loss(torch.tanh(mean), act)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            l = float(loss.detach())
            ema = l if ema is None else 0.99 * ema + 0.01 * l
            shared["loss"] = ema
            shared["updates"] += 1
    threading.Thread(target=_learner, daemon=True).start()

    prev = np.array([ve.env.dist_to_gate() for ve in envs])
    away = np.zeros(N, int); ep_gates = np.zeros(N, int)
    ep_steps = np.zeros(N, int); ep_mind = prev.copy()
    passes = []; solved = set()
    ep = 0; total = 0
    t0 = time.time(); last_t = t0; last_step = 0
    while total < args.steps:
        # observations + teacher labels (batched)
        stacks = [ve.ring.stack() for ve in envs]
        props = [proprio_of(ve.env) for ve in envs]
        sobs = np.stack([obs_of(ve.env) for ve in envs])
        with torch.no_grad():
            to = torch.as_tensor(sobs, dtype=torch.float32)
            labels = teacher.actor_ema.act(to, deterministic=True).numpy()
        # per-STEP mixture (DAgger): teacher w.p. beta, student otherwise. Keeps trajectories near
        # the expert corridor while the student earns control — a hard per-episode handoff floods
        # the buffer with corrections in unrecoverable states and the corridor behavior degrades.
        beta = max(args.beta_min, 1.0 - ep / max(1, args.beta_anneal_eps))
        tacts = np.clip(labels + np.random.normal(0, args.tnoise, labels.shape), -1, 1)
        if beta < 1.0:
            sacts = student_act_batch(student, stacks, props, dev, deterministic=False)
            pick = np.random.random(N) < beta
            acts = np.where(pick[:, None], tacts, sacts)
        else:
            acts = tacts
        for i, ve in enumerate(envs):
            buf.add_entry(ve.ring.stack_slots(), props[i], labels[i])
            passed, crashed = ve.step(acts[i].astype(np.float32))
            e = ve.env
            d = e.dist_to_gate(); ep_mind[i] = min(ep_mind[i], d)
            away[i] = 0 if (passed or d < prev[i]) else away[i] + 1
            if passed:
                ep_gates[i] += 1
            done = crashed or d > CFG.stray_dist or away[i] >= AWAY \
                or ep_gates[i] >= args.gates
            reason = ("finish" if ep_gates[i] >= args.gates else
                      "crash" if crashed else "stray" if d > CFG.stray_dist else
                      "away" if away[i] >= AWAY else "timeout")
            prev[i] = d; ep_steps[i] += 1
            if done or ep_steps[i] >= e.max_steps:
                passes.append(int(ep_gates[i] >= args.gates))
                pr = float(np.mean(passes[-200:]))
                now = time.time()
                sps = (total - last_step) / max(1e-6, now - last_t)
                rec = {"ep": ep, "t": now, "step": total, "reward": 0.0,
                       "min_dist": float(ep_mind[i]), "reason": reason,
                       "gates": int(ep_gates[i]), "passed": int(ep_gates[i] >= args.gates),
                       "pass_rate100": pr, "steps_per_s": sps, "buf": buf.size,
                       "updates": shared["updates"], "c_loss": shared["loss"]}
                if ep % args.eval_every == 0 and buf.size >= 2000 and ep > 0:
                    counts, gmd = eval_student(student, eval_ve, args.gates, dev)
                    rec["eval_g0"] = counts[0]; rec["eval_gates"] = counts
                    rec["eval_mind"] = gmd
                    if len(counts) > 1:
                        rec["eval_g1"] = counts[1]
                    cstr = " ".join(f"g{k}:{c}/20" for k, c in enumerate(counts))
                    print(f"ep {ep:6d} step {total:8d} | STUDENT {cstr} (mind {gmd:.1f}) | "
                          f"loss={shared['loss']:.4f} upd={shared['updates']} buf={buf.size} "
                          f"| {sps:5.0f} st/s +{(now - t0) / 60:.1f} min", flush=True)
                    last_t = time.time(); last_step = total
                    torch.save({"student": student.state_dict(),
                                "spacing": args.spacing}, os.path.join(RUN_DIR, "autosave_v.pt"))
                    if ep % args.video_every == 0:
                        record_student(student, eval_ve, os.path.join(RUN_DIR,
                                       f"student_ep{ep}.mp4"), dev, args.gates)
                    for k in range(args.gates):
                        if counts[k] >= 20 and k not in solved:
                            solved.add(k)
                            torch.save({"student": student.state_dict(),
                                        "spacing": args.spacing},
                                       os.path.join(RUN_DIR, f"mjv_gate{k}.pt"))
                            record_student(student, eval_ve,
                                           os.path.join(RUN_DIR, f"vision_gate{k}.mp4"),
                                           dev, args.gates)
                            print(f"  *** STUDENT 20/20 through gate {k} FROM PIXELS at "
                                  f"+{(now - t0) / 60:.1f} min (ep {ep}) ***", flush=True)
                with open(mpath, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                ve.reset()
                prev[i] = ve.env.dist_to_gate(); away[i] = 0
                ep_gates[i] = 0; ep_steps[i] = 0; ep_mind[i] = prev[i]
                ep += 1
        total += N


if __name__ == "__main__":
    main()
