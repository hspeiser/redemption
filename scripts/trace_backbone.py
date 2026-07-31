"""Single-env backbone trajectory dump: where does tracking break?"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.refctl import load_winner_backbone  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402

cfg = FastEnvConfig(
    n_envs=2, random_start_frac=0.0, rate_gain_sign=1.0,
    reloc_events=False, demo_corridor_m=4.0, speed_cap_mps=8.0,
    spawn_at_rest=True, pos_noise_lo=0.0, pos_noise_hi=0.0,
    dr_thrust=(1.0, 1.0), dr_rate_gain=(1.0, 1.0),
    dr_rate_tau=(1.0, 1.0), dr_drag=(0.3, 0.3),
)
cfg.act_delay_steps_min = 0
cfg.act_delay_steps_max = 0
backbone = load_winner_backbone(
    str(REPO / "data/fastsim_demo_winner.npz"),
    str(REPO / "data/vq2_sac_runs/gate3_nstep_v60b/20260730_161745"
               "/episode_0002.npz"),
    2, device="cpu",
)
model = SurrogateModel.load(REPO / "data/fastsim_model_v2.json")
demo = dict(np.load(REPO / "data/fastsim_demo_winner.npz"))
env = FastVQ2Env(model, str(REPO / "data/vq2_map_winner_train.json"),
                 demo_states=demo, config=cfg, device="cpu",
                 backbone=backbone)
zeros = torch.zeros(2, 4)
obs = env.observations()
for k in range(400):
    obs, r, d, info = env.step(zeros)
    if k % 20 == 0:
        i = int(backbone.idx[0])
        ref = backbone.P[i].numpy()
        p = env.p[0].numpy()
        err = float(np.linalg.norm(p - ref))
        a = backbone.action(
            env.p + env.noise_pos, env.v, env._qmat(env.q))[0].numpy()
        print(
            f"k={k:3d} tgt={int(env.target[0])} idx={i:4d} "
            f"p=({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f}) "
            f"spd={float(torch.linalg.norm(env.v[0])):.2f} "
            f"referr={err:.2f} "
            f"a=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f},{a[3]:+.2f}]"
        )
    if bool(d[0]):
        print(f"env0 done at k={k}")
        break
