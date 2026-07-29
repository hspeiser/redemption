"""One instrumented greedy episode: print obs blocks + action + world motion every few steps,
so twin->VQ1 transfer failures show WHERE the policy's world model diverges from reality.

  python -m racer_state.debug_eval
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from racer_state.config import CFG
from racer_state.mav_io import MavLink
from racer_state.geom import rot_world_to_body, rot_body_to_world, gate_normal_world
from racer_state.sac import SAC
from racer_state.train import wait_reset
from racer_state.train2 import (act_desired, frd_obs, stream_rates, takeoff, _DOWN_NED)

CKPT = (r"C:\Users\satas\Downloads\AI-GP Simulator v1.0.3385-VQ1\PyAIPilotExample-v1"
        r"\racer_mujoco\runs_mj\mj_gate5.pt")


def main():
    CFG.critic_dropout = 0.0
    CFG.control_hz = 30.0
    agent = SAC(CFG, "cuda" if torch.cuda.is_available() else "cpu")
    agent.load_full(CKPT)
    mav = MavLink(CFG.mav_addr)
    for _ in range(5):
        if mav.capture_gates():
            break
    gates = mav.gates
    gate = gates[0]; gnorm = gate_normal_world(gate["quat"])
    print(f"gate0 pos={np.round(gate['pos'], 2)} normal={np.round(gnorm, 3)}", flush=True)

    wait_reset(mav); time.sleep(CFG.reset_settle_s)
    t0 = time.time()
    while mav.t_us == 0 and time.time() - t0 < 3:
        mav.hb(); time.sleep(0.03)
    mav.arm(True)
    hb = [0.0]
    print(f"pre-takeoff: pos={np.round(mav.pos, 2)} quat={np.round(mav.quat, 3)}", flush=True)
    takeoff(mav, hb)
    print(f"post-takeoff: pos={np.round(mav.pos, 2)} vel_body={np.round(mav.vel, 2)} "
          f"rates={np.round(mav.rates, 2)}", flush=True)

    dt = 1.0 / CFG.control_hz
    for step in range(90):
        obs = frd_obs(mav.pos, mav.quat, mav.vel, mav.rates, gate["pos"], gnorm)
        a = agent.act(obs, deterministic=True, ema=True)
        des, thr = act_desired(a)
        tick = time.time()
        stream_rates(mav, des, thr, tick + dt, hb)
        if step % 3 == 0:
            d = float(np.linalg.norm(gate["pos"] - mav.pos))
            vworld = rot_body_to_world(mav.quat, np.asarray(mav.vel, np.float64))
            print(f"s{step:3d} d={d:5.2f} | obs rel={np.round(obs[0:3], 2)} "
                  f"vel={np.round(obs[3:6], 2)} rates={np.round(obs[6:9], 2)} "
                  f"grav={np.round(obs[9:12], 2)} gn={np.round(obs[12:15], 2)} | "
                  f"a={np.round(a, 2)} -> des={np.round(des, 2)} thr={thr:.2f} | "
                  f"vworld={np.round(vworld, 2)} rates_now={np.round(mav.rates, 2)}", flush=True)
    mav.arm(False)
    for _ in range(6):
        mav.att(0, 0, 0, 0); mav.hb(); time.sleep(0.02)
    mav.stop()


if __name__ == "__main__":
    main()
