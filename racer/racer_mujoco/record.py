"""Standalone flight recorder: load a checkpoint, fly greedy, save a polished mp4.

  python -m racer_mujoco.record --view follow            # smooth chase cam
  python -m racer_mujoco.record --view iso               # isometric, brighter
  python -m racer_mujoco.record --ckpt runs_mj/mj_gate5.pt --out my.mp4 --gates 6

Camera FOLLOWS the drone with exponential smoothing (no snap when the active gate switches);
scene colors/lighting are brightened on the loaded model (env untouched)."""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import mujoco
import cv2

from racer_state.config import CFG
from racer_state.sac import SAC
from racer_mujoco.env import QuadEnv
from racer_mujoco.train_mj import obs_of

RUN_DIR = os.path.join(os.path.dirname(__file__), "runs_mj")

VIEWS = {
    "follow": dict(azimuth=90.0, elevation=-18.0, distance=10.0),
    "iso": dict(azimuth=135.0, elevation=-32.0, distance=13.0),
}


def brighten(model):
    """Lift the scene out of the murk: brighter ground/gates/drone + stronger headlight."""
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.85, 0.85, 0.85]
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        rgba = model.geom_rgba[gid]
        if name == "ground":
            model.geom_rgba[gid] = [0.42, 0.48, 0.55, 1.0]
        elif name == "core":
            model.geom_rgba[gid] = [1.0, 0.32, 0.22, 1.0]
        elif rgba[0] > 0.8 and rgba[1] > 0.5:          # gate posts (yellow-ish)
            model.geom_rgba[gid] = [1.0, 0.82, 0.12, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(RUN_DIR, "autosave.pt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--view", choices=list(VIEWS), default="follow")
    ap.add_argument("--gates", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=12, help="try up to N seeds, keep the best flight")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=400)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cdrop", type=float, default=0.0, help="critic dropout of the ckpt's arch")
    args = ap.parse_args()
    out = args.out or os.path.join(RUN_DIR, f"flight_{args.view}.mp4")
    CFG.critic_dropout = args.cdrop
    torch.set_num_threads(2)
    agent = SAC(CFG, "cpu")
    agent.load_full(args.ckpt)
    v = VIEWS[args.view]

    best = None   # (gates, frames)
    for seed in range(2000, 2000 + args.seeds):
        env = QuadEnv(episode_s=40, seed=seed); env.reset()
        brighten(env.model)
        r = mujoco.Renderer(env.model, args.h, args.w)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = v["azimuth"]; cam.elevation = v["elevation"]; cam.distance = v["distance"]
        look = env.pos.copy()
        frames = []; prev = env.dist_to_gate(); away = 0; g = 0
        for _ in range(env.max_steps):
            # exponential follow with a touch of velocity lead — smooth through gate handoffs
            look += 0.12 * (env.pos + 0.35 * env.vel - look)
            cam.lookat[:] = look
            r.update_scene(env.data, cam)
            frames.append(r.render().copy())
            passed, crashed = env.step(agent.act(obs_of(env), deterministic=True, ema=True))
            d = env.dist_to_gate()
            if passed:
                g += 1; away = 0; prev = d
                if g >= args.gates:
                    break
                continue
            away = 0 if d < prev else away + 1; prev = d
            if crashed or d > CFG.stray_dist or away >= 15:
                break
        for _ in range(14):                    # settle shot at the end
            look += 0.12 * (env.pos - look)
            cam.lookat[:] = look
            r.update_scene(env.data, cam); frames.append(r.render().copy())
        print(f"seed {seed}: {g}/{args.gates} gates, {len(frames)} frames", flush=True)
        if best is None or g > best[0]:
            best = (g, frames)
        r.close()
        if g >= args.gates:
            break

    g, frames = best
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.w, args.h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"SAVED {out}: {g}/{args.gates} gates, {len(frames)} frames = {len(frames)/args.fps:.1f}s", flush=True)


if __name__ == "__main__":
    main()
