"""Rerank suffix candidates against the EMPIRICAL g10-handoff pool.

Codex's promotion checklist for the co-visibility candidates: the
search starts every rollout from the single 35.37 handoff state, which
risks overfitting that exact entry.  This reranker evaluates the frozen
baseline suffix and every candidate under identical machinery with
starts drawn from the OBSERVED distribution of gate-10 crossings (the
13-finish expert corpus: speeds 3.07-5.16 m/s, ~2 m position spread),
plus estimator noise, and reports per-candidate:

  finish rate, suffix time, g13 split, g13 crossing speed,
  tracking error at the g13 crossing, clearance p05 by gate,
  detection-aware dual coverage g12-14, sha256 of the artifacts.

    python scripts/rerank_suffix_candidates.py \
        --candidates data/lineopt/sfx_geometry_best.npz ... \
        --out data/lineopt/sfx_rerank_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import ACT_DIM, FastEnvConfig, FastVQ2Env  # noqa: E402
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.lineopt import (  # noqa: E402
    DT, BatchedFlatRefController, LineConfig, build_reference,
    load_oriented_gates,
)
from aigp.vq2_map import gate_quads_world_vq2  # noqa: E402
from scripts.fastsim_suffix_covis import (  # noqa: E402
    N_SUFFIX, SUFFIX_FIRST, covis_metrics, load_detect,
)

N_GATES = 17


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="sfx_*_best.npz files (theta key)")
    ap.add_argument("--map", default=str(
        REPO / "data/vq2_runtime_map_g9g15fix.json"))
    ap.add_argument("--model", default=str(
        REPO / "data/fastsim_model_v2.json"))
    ap.add_argument("--obstacles", default=str(
        REPO / "data/vq2_obstacles_inflated.json"))
    ap.add_argument("--handoff", default=str(
        REPO / "data/lineopt/handoff_g10_35p37.json"))
    ap.add_argument("--pool", default=str(
        REPO / "data/lineopt/handoff_pool_13finish.json"))
    ap.add_argument("--seed-theta", default=str(
        REPO / "data/lineopt/a2_cap12_best.npz"))
    ap.add_argument("--detect", default=str(
        REPO / "data/gate_detectability_v1.json"))
    ap.add_argument("--calib", default=str(REPO / "data/calib/calib.json"))
    ap.add_argument("--n-envs", type=int, default=768)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    handoff = json.loads(Path(args.handoff).read_text())
    pool = json.loads(Path(args.pool).read_text())
    gate_pos, gate_R = load_oriented_gates(args.map)
    gp_s = gate_pos[SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX]
    gR_s = gate_R[SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX]
    model = SurrogateModel.load(args.model)

    gates_json = json.loads(Path(args.map).read_text())["gates"]
    all_corners = [np.concatenate(gate_quads_world_vq2(g))
                   for g in gates_json[:17]]
    all_centers = [np.mean(c, axis=0) for c in all_corners]
    all_normals = []
    for g in gates_json[:17]:
        qw, qx, qy, qz = g["quat_wxyz"]
        all_normals.append(Rotation.from_quat(
            [qx, qy, qz, qw]).as_matrix()[:, 1])
    calib = json.loads(Path(args.calib).read_text())
    K = np.array([[calib["fx"], 0, calib["cx"]],
                  [0, calib["fy"], calib["cy"]], [0, 0, 1.0]])
    R_cb = np.asarray(calib["R_cam_from_body"], float)
    wdet = load_detect(args.detect)

    lcfg = LineConfig(speed_cap=12.0, clearance=0.15,
                      launch_speed=float(handoff["speed"]))
    lcfg.spawn = tuple(handoff["position"])
    v0 = np.asarray(handoff["velocity"], float)
    lcfg.start_dir = tuple(v0 / (np.linalg.norm(v0) + 1e-9))

    seed = np.load(args.seed_theta)
    theta_full = seed["theta"]
    base_off = theta_full[:34].reshape(17, 2)[
        SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX]
    base_sc = theta_full[34:][SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX + 1]

    entries = [("baseline_cap12_suffix", None, base_off, base_sc,
                args.seed_theta)]
    for path in args.candidates:
        d = np.load(path)
        off = d["suffix_offsets"]
        sc = d["suffix_scales"]
        entries.append((Path(path).stem, None, off, sc, path))

    refs, names, hashes = [], [], []
    covs = []
    for name, _t, off, sc, path in entries:
        ref = build_reference(gp_s, gR_s, off, sc, lcfg)
        refs.append(ref)
        names.append(name)
        hashes.append(sha16(path))
        covs.append(covis_metrics(ref, all_corners, all_centers,
                                  all_normals, K, R_cb, wdet))

    # empirical handoff pool as start states
    demo_states = {
        "pos": np.asarray([r["position"] for r in pool], np.float32),
        "vel": np.asarray([r["velocity"] for r in pool], np.float32),
        "quat": np.asarray([r["quat_wxyz"] for r in pool], np.float32),
        "gate": np.full(len(pool), SUFFIX_FIRST, np.float32),
    }
    per = args.n_envs
    C = len(refs)
    cfg = FastEnvConfig(
        n_envs=C * per, random_start_frac=1.0, rate_gain_sign=1.0,
        reloc_events=True, speed_cap_mps=12.5, demo_corridor_m=999.0,
    )
    cfg.apply_vision10hz()
    cfg.fov_vision = True
    cfg.coast_speed_diffuse = 0.005
    cfg.coast_speed_bias = 0.015
    cfg.act_delay_steps_min = 1
    cfg.residual_scale = 0.0
    cfg.max_episode_s = 40.0
    cfg.start_noise_pos_m = 0.05
    cfg.start_noise_vel_mps = 0.15
    backbone = BatchedFlatRefController(
        refs, per, device=str(device), speed_cap=12.5, model=model)
    env = FastVQ2Env(model, args.map, demo_states=demo_states,
                     config=cfg, device=str(device),
                     obstacles_path=args.obstacles or None,
                     backbone=backbone)
    n = cfg.n_envs
    zeros = torch.zeros(n, ACT_DIM, device=device)
    fin = torch.zeros(n, dtype=torch.bool, device=device)
    fail = torch.zeros(n, dtype=torch.bool, device=device)
    t_fin = torch.zeros(n, device=device)
    alive = torch.zeros(n, device=device)
    t12 = torch.zeros(n, device=device)
    t13 = torch.zeros(n, device=device)
    v13 = torch.zeros(n, device=device)
    e13 = torch.zeros(n, device=device)
    clear_min = torch.full((n, N_GATES), np.nan, device=device)
    ar = torch.arange(n, device=device)
    with torch.no_grad():
        for _ in range(1400):
            _o, _r, done, info = env.step(zeros)
            live = ~(fin | fail)
            alive += live.float()
            pas = info["passed"] & live
            if pas.any():
                gid = torch.clamp(info["target"] - 1, 0, N_GATES - 1)
                margin = 0.75 - info["cross_r"]
                cur = clear_min[ar, gid]
                upd = torch.where(torch.isnan(cur), margin,
                                  torch.minimum(cur, margin))
                clear_min[ar, gid] = torch.where(pas, upd, cur)
            hit13 = pas & (info["target"] == 14)
            t12 = torch.where(pas & (info["target"] == 13),
                              alive * DT, t12)
            t13 = torch.where(hit13, alive * DT, t13)
            v13 = torch.where(hit13, info["speed"], v13)
            if hit13.any():
                track_err = torch.linalg.norm(
                    backbone.P[backbone.cand, backbone.idx] - env.p,
                    dim=-1)
                e13 = torch.where(hit13, track_err, e13)
            newf = info["finished"] & live
            t_fin = torch.where(newf, alive * DT, t_fin)
            fin |= newf
            fail |= done & live & ~info["finished"]
            if float((fin | fail).float().mean()) > 0.999:
                break

    finC = fin.view(C, per)
    tC = t_fin.view(C, per)
    splitC = (t13 - t12).view(C, per)
    v13C = v13.view(C, per)
    e13C = e13.view(C, per)
    cm = clear_min.view(C, per, N_GATES).cpu().numpy()
    report = {"pool": f"{len(pool)} empirical g10 handoff states "
                      f"(speeds {min(r['speed'] for r in pool):.2f}-"
                      f"{max(r['speed'] for r in pool):.2f} m/s)",
              "n_envs_per_candidate": per,
              "entries": []}
    for ci, name in enumerate(names):
        ok = finC[ci]
        fr = float(ok.float().mean())
        sp = splitC[ci][ok & (splitC[ci] > 0)]
        entry = {
            "name": name,
            "sha256_16": hashes[ci],
            "finish_rate": round(fr, 4),
            "suffix_t_med": round(float(tC[ci][ok].median()), 3)
            if fr else None,
            "suffix_t_p90": round(float(tC[ci][ok].quantile(0.9)), 3)
            if fr else None,
            "g13_split_med": round(float(sp.median()), 3)
            if len(sp) else None,
            "g13_cross_speed_med": round(float(
                v13C[ci][ok & (v13C[ci] > 0)].median()), 2)
            if fr else None,
            "g13_track_err_med": round(float(
                e13C[ci][ok & (e13C[ci] > 0)].median()), 3)
            if fr else None,
            "dual_cov_g12_14": round(covs[ci]["dual_cov"], 3),
            "single_cov_g12_14": round(covs[ci]["single_cov"], 3),
        }
        with np.errstate(all="ignore"):
            entry["clearance_p05_suffix"] = [
                round(float(np.nanquantile(cm[ci, :, g], 0.05)), 3)
                if np.isfinite(cm[ci, :, g]).any() else None
                for g in range(SUFFIX_FIRST, SUFFIX_FIRST + N_SUFFIX)
            ]
        report["entries"].append(entry)
        print(json.dumps(entry))
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
