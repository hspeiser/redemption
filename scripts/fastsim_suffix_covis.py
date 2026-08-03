"""Co-visibility-aware suffix line search for gates 11-16.

Per the campaign spec after the 35.374s record: the g12->g14 reversal
collapses localization to single-gate views (measured from the winning
lap's multigate debug), so the suffix search scores candidates on
DETECTION-AWARE dual-gate coverage through that window in addition to
finish rate and time.  The prefix through gate 10 is frozen: every
rollout starts from the 35.37 lap's measured g10-crossing state.

Detectability comes from data/gate_detectability_v1.json (logistic fit
on 56k measured association outcomes), not from raw frustum tests.

Modes (Codex's three candidates):
    geometry  -- offsets only, speed profile frozen at the cap-12 seed
    timing    -- offsets + per-segment speeds
    aggressive-- offsets + speeds, stronger lap-time weight
    speed     -- reliability-floor-constrained timing optimization

    python scripts/fastsim_suffix_covis.py --mode geometry \
        --out-prefix data/lineopt/sfx_geo
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.env import (  # noqa: E402
    ACT_DIM, FastEnvConfig, FastVQ2Env, HOLE_HALF,
)
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.liveteacher import (  # noqa: E402
    BatchedLiveTeacher,
    observation_rotation_batch,
)
from aigp.fastsim.worldmodel import (  # noqa: E402
    ResidualEnsemble,
    ResidualEnsemblePool,
)
from aigp.fastsim.lineopt import (  # noqa: E402
    DT, BatchedFlatRefController, LineConfig, build_reference,
    load_oriented_gates,
)
from aigp.vq2_map import gate_quads_world_vq2  # noqa: E402
from scripts.build_vq2_suffix_reference_demo import (  # noqa: E402
    build_suffix_demo,
)

SUFFIX_FIRST = 11
N_SUFFIX = 6            # gates 11..16
FRAME_W, FRAME_H = 640, 360


def load_detect(path):
    d = json.loads(Path(path).read_text())
    return np.asarray(d["weights"], float)


def detect_prob(w, span_px, margin_px, range_m, cos_view):
    x = np.stack([
        np.log(np.maximum(span_px, 2.0)),
        np.clip(margin_px, -60.0, 120.0) / 60.0,
        np.asarray(range_m) / 20.0,
        np.abs(cos_view),
        np.ones_like(span_px),
    ], axis=-1)
    return 1.0 / (1.0 + np.exp(-(x @ w)))


def covis_metrics(ref, all_corners, all_centers, all_normals,
                  K, R_cb, wdet):
    """Time-integrated detection-aware coverage over the g12->g14 leg."""
    pos = ref["pos"]
    R_body = ref["R"]                      # (n,3,3) body->world
    t_gate = ref["t_gate"]
    # window: crossing of local gate 1 (g12) to local gate 3 (g14)
    row_lo = int(t_gate[1] / DT)
    row_hi = min(int(t_gate[3] / DT) + 1, len(pos))
    if row_hi - row_lo < 2:
        return {"dual_cov": 0.0, "single_cov": 0.0}
    dual = single = 0
    fx = K[0, 0]
    for k in range(row_lo, row_hi):
        Rt = R_cb @ R_body[k].T
        p = pos[k]
        det_gates = []
        for gi in range(len(all_corners)):
            rel = all_centers[gi] - p
            rng = float(np.linalg.norm(rel))
            if not 1.5 <= rng <= 45.0:
                continue
            Xc = (Rt @ (all_corners[gi] - p).T).T
            if (Xc[:, 2] <= 0.3).any():
                continue
            uv = np.stack([
                fx * Xc[:, 0] / Xc[:, 2] + K[0, 2],
                K[1, 1] * Xc[:, 1] / Xc[:, 2] + K[1, 2],
            ], axis=1)
            span = float(np.max(np.linalg.norm(
                uv[:, None] - uv[None, :], axis=-1)))
            margin = float(min(uv[:, 0].min(), FRAME_W - uv[:, 0].max(),
                               uv[:, 1].min(), FRAME_H - uv[:, 1].max()))
            if margin < -40:
                continue
            cosv = float(np.dot(rel / rng, all_normals[gi]))
            prob = float(detect_prob(
                wdet, np.array(span), np.array(margin),
                np.array(rng), np.array(cosv)))
            if prob >= 0.5:
                det_gates.append((gi, rel / rng, prob))
        if det_gates:
            single += 1
        ok_dual = False
        for i in range(len(det_gates)):
            for j in range(i + 1, len(det_gates)):
                cosang = float(np.clip(np.dot(
                    det_gates[i][1], det_gates[j][1]), -1, 1))
                if np.degrees(np.arccos(cosang)) >= 12.0:
                    ok_dual = True
        if ok_dual:
            dual += 1
    n = row_hi - row_lo
    return {"dual_cov": dual / n, "single_cov": single / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=("geometry", "timing", "aggressive", "speed"),
                    required=True)
    ap.add_argument("--map", default=str(
        REPO / "data/vq2_runtime_map_g9g15fix.json"))
    ap.add_argument("--model", default=str(
        REPO / "data/fastsim_model_v2.json"))
    ap.add_argument("--obstacles", default=str(
        REPO / "data/vq2_obstacles_inflated.json"))
    ap.add_argument("--handoff", default=str(
        REPO / "data/lineopt/handoff_g10_35p37.json"))
    ap.add_argument("--seed-theta", default=str(
        REPO / "data/lineopt/a2_cap12_best.npz"))
    ap.add_argument("--detect", default=str(
        REPO / "data/gate_detectability_v1.json"))
    ap.add_argument("--calib", default=str(REPO / "data/calib/calib.json"))
    ap.add_argument("--n-envs", type=int, default=192)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=6)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--final-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--live-teacher-config", type=Path,
        help=("Use the parity-certified deployed controller instead of the "
              "legacy geometric tracker."),
    )
    ap.add_argument(
        "--exact-search-space", choices=("config", "line"),
        default="config",
        help=("config perturbs the frozen controller's per-gate offsets and "
              "velocity scales around the exact champion; line rebuilds a "
              "replacement suffix demo and is retained for diagnostics."),
    )
    ap.add_argument(
        "--ensemble", nargs="+",
        help="Residual ensembles required by --live-teacher-config.",
    )
    ap.add_argument(
        "--prefix-demo", type=Path,
        help="Frozen full-course champion demo to splice through gate 10.",
    )
    ap.add_argument(
        "--handoff-pool", type=Path, default=(
            REPO / "data/lineopt/handoff_pool_13finish.json"
        ),
    )
    ap.add_argument("--bridge-rows", type=int, default=30)
    ap.add_argument(
        "--reliability-floor", type=float, default=0.65,
        help="Minimum search finish rate before speed mode optimizes time.",
    )
    ap.add_argument(
        "--freeze-offsets", action="store_true",
        help=("Keep the config's suffix geometry fixed and search only "
              "velocity scales (timing/aggressive modes)."),
    )
    ap.add_argument(
        "--speed-gates", default="",
        help=("Comma-separated target gates whose velocity scales may change. "
              "Only supported by the exact config search; default is 11-16."),
    )
    ap.add_argument(
        "--lead-gates", default="",
        help=("Comma-separated target gates whose integer reference-action "
              "leads may change. Exact config search only."),
    )
    ap.add_argument(
        "--lead-max", type=int, default=8,
        help="Largest per-gate reference-action lead considered.",
    )
    ap.add_argument(
        "--finalists", type=int, default=1,
        help=("Number of search candidates to rerank together on the fresh "
              "final-world pool before selecting the winner."),
    )
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.smoke:
        args.n_envs, args.pop, args.iters, args.final_envs = 24, 4, 2, 64
    if args.live_teacher_config and not args.ensemble:
        ap.error("--live-teacher-config requires --ensemble")

    handoff = json.loads(Path(args.handoff).read_text())
    start_p = np.asarray(handoff["position"], float)
    start_v = np.asarray(handoff["velocity"], float)
    start_R = np.asarray(handoff["R"], float)
    q = Rotation.from_matrix(start_R).as_quat()
    start_quat = np.array([q[3], q[0], q[1], q[2]], np.float32)

    gate_pos, gate_R = load_oriented_gates(args.map)
    gp_s = gate_pos[SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX]
    gR_s = gate_R[SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX]

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

    seed = np.load(args.seed_theta)
    theta_full = seed["theta"]
    seed_off = theta_full[:17 * 2].reshape(17, 2)[
        SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX]
    seed_sc = theta_full[17 * 2:][SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX + 1]
    seed_line_sc = seed_sc.copy()
    seed_lead = np.zeros(N_SUFFIX, np.int64)

    lcfg = LineConfig(speed_cap=12.0, clearance=0.15,
                      launch_speed=float(handoff["speed"]))
    lcfg.spawn = tuple(start_p)
    lcfg.start_dir = tuple(start_v / (np.linalg.norm(start_v) + 1e-9))
    if args.mode in ("aggressive", "speed"):
        lcfg.a_fwd, lcfg.a_brk = 6.5, 7.0

    model = SurrogateModel.load(args.model)
    exact_ensemble = None
    exact_handoffs = None
    exact_prefix = None
    exact_config_doc = None
    exact_demo_covis = None
    exact_tmp = None
    if args.live_teacher_config:
        exact_config_doc = json.loads(args.live_teacher_config.read_text())
        live_cfg = exact_config_doc["args"]
        exact_prefix = args.prefix_demo or Path(live_cfg["demo"])
        demo = np.load(exact_prefix, allow_pickle=False)
        demo_gate = np.asarray(demo["gate_index"], np.int64)
        suffix_rows = np.flatnonzero(demo_gate >= SUFFIX_FIRST)
        suffix_gate = demo_gate[suffix_rows] - SUFFIX_FIRST
        suffix_pos = np.asarray(demo["position"], np.float64)[suffix_rows]
        suffix_R = observation_rotation_batch(
            np.asarray(demo["observation"], np.float32)[suffix_rows]
        )
        suffix_t_gate = np.zeros(N_SUFFIX, np.float64)
        for local_gate in range(N_SUFFIX):
            rows_for_gate = np.flatnonzero(suffix_gate == local_gate)
            if not len(rows_for_gate):
                raise SystemExit(
                    f"champion demo lacks suffix gate {local_gate + SUFFIX_FIRST}"
                )
            suffix_t_gate[local_gate] = rows_for_gate[0] * DT
        exact_demo_covis = {
            "pos": suffix_pos,
            "R": suffix_R,
            "gate": suffix_gate,
            "t_gate": suffix_t_gate,
        }
        rows = json.loads(args.handoff_pool.read_text())
        exact_handoffs = {
            "pos": np.asarray([row["position"] for row in rows], np.float32),
            "vel": np.asarray([row["velocity"] for row in rows], np.float32),
            "quat": np.asarray([row["quat_wxyz"] for row in rows], np.float32),
            "gate": np.full(len(rows), SUFFIX_FIRST, np.float32),
        }
        loaded = [
            ResidualEnsemble.load(path, str(device))
            for path in args.ensemble
        ]
        members = [item[0] for item in loaded]
        exact_ensemble = (
            members[0] if len(members) == 1
            else ResidualEnsemblePool(members).to(device)
        )
        exact_ensemble.eval()
        exact_tmp = Path(tempfile.mkdtemp(prefix="vq2_suffix_exact_"))

        def parsed_values(name, default):
            values = np.full(17, float(default), np.float64)
            for item in str(live_cfg.get(name, "") or "").split(","):
                if item.strip():
                    gate, value = item.split(":", 1)
                    values[int(gate)] = float(value)
            return values

        if args.exact_search_space == "config":
            lat = parsed_values("reference_lateral_offsets", 0.0)
            vert = parsed_values("reference_vertical_offsets", 0.0)
            seed_off = np.stack([lat, vert], axis=1)[
                SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX
            ]
            vel_default = float(live_cfg.get("reference_velocity_scale", 1.0))
            seed_sc = parsed_values(
                "reference_velocity_scales", vel_default
            )[SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX]
            lead_default = int(live_cfg.get("reference_action_lead", 0))
            seed_lead = parsed_values(
                "reference_action_leads", lead_default
            )[SUFFIX_FIRST:SUFFIX_FIRST + N_SUFFIX].astype(np.int64)
    off_lim = (
        0.35 if args.live_teacher_config
        and args.exact_search_space == "config"
        else HOLE_HALF - 0.15
    )
    g15_local = 15 - SUFFIX_FIRST
    g15_lo = seed_off[g15_local] - 0.15
    g15_hi = seed_off[g15_local] + 0.15

    speed_local_indices = np.arange(N_SUFFIX, dtype=np.int64)
    if args.speed_gates:
        if not (args.live_teacher_config
                and args.exact_search_space == "config"):
            ap.error("--speed-gates requires the exact config search")
        requested_gates = [
            int(item.strip()) for item in args.speed_gates.split(",")
            if item.strip()
        ]
        if not requested_gates or any(
            gate < SUFFIX_FIRST or gate >= SUFFIX_FIRST + N_SUFFIX
            for gate in requested_gates
        ):
            ap.error("--speed-gates must contain gates from 11 through 16")
        speed_local_indices = np.asarray(
            sorted(set(gate - SUFFIX_FIRST for gate in requested_gates)),
            dtype=np.int64,
        )
    lead_local_indices = np.empty(0, dtype=np.int64)
    if args.lead_gates:
        if args.mode == "geometry" or not (
            args.live_teacher_config
            and args.exact_search_space == "config"
        ):
            ap.error("--lead-gates requires a non-geometry exact config search")
        requested_lead_gates = [
            int(item.strip()) for item in args.lead_gates.split(",")
            if item.strip()
        ]
        if not requested_lead_gates or any(
            gate < SUFFIX_FIRST or gate >= SUFFIX_FIRST + N_SUFFIX
            for gate in requested_lead_gates
        ):
            ap.error("--lead-gates must contain gates from 11 through 16")
        lead_local_indices = np.asarray(
            sorted(set(gate - SUFFIX_FIRST
                       for gate in requested_lead_gates)),
            dtype=np.int64,
        )
    if args.lead_max < 0:
        ap.error("--lead-max must be non-negative")

    speed_params = (
        len(speed_local_indices) if args.live_teacher_config
        and args.exact_search_space == "config"
        else N_SUFFIX + 1
    )

    def unpack(theta):
        offset_params = 0 if args.freeze_offsets else N_SUFFIX * 2
        off = (
            seed_off.copy() if args.freeze_offsets
            else np.clip(
                theta[:offset_params].reshape(N_SUFFIX, 2),
                -off_lim,
                off_lim,
            )
        )
        off[g15_local] = np.clip(off[g15_local], g15_lo, g15_hi)
        if args.mode == "geometry":
            sc = seed_sc.copy()
        else:
            bounds = (
                (0.80, 1.30) if args.live_teacher_config
                and args.exact_search_space == "config"
                else (0.35, 1.0)
            )
            proposed = np.clip(
                theta[offset_params:offset_params + speed_params],
                *bounds,
            )
            if args.live_teacher_config and args.exact_search_space == "config":
                sc = seed_sc.copy()
                sc[speed_local_indices] = proposed
            else:
                sc = proposed
        return off, sc

    def unpack_leads(theta):
        leads = seed_lead.copy()
        if len(lead_local_indices):
            start = offset_params + speed_params
            proposed = np.rint(theta[start:start + len(lead_local_indices)])
            leads[lead_local_indices] = np.clip(
                proposed, 0, args.lead_max
            ).astype(np.int64)
        return leads

    offset_params = 0 if args.freeze_offsets else N_SUFFIX * 2
    if args.freeze_offsets and args.mode == "geometry":
        ap.error("--freeze-offsets requires timing, aggressive, or speed mode")
    lead_params = len(lead_local_indices)
    n_par = (
        offset_params
        + (0 if args.mode == "geometry" else speed_params)
        + lead_params
    )
    mean = (
        np.empty(0, np.float64)
        if args.freeze_offsets else seed_off.flatten().copy()
    )
    sd = np.full(offset_params, 0.16)
    if args.mode != "geometry":
        speed_mean = (
            seed_sc[speed_local_indices]
            if args.live_teacher_config
            and args.exact_search_space == "config"
            else seed_sc
        )
        mean = np.concatenate([mean, speed_mean])
        sd = np.concatenate([sd, np.full(speed_params, 0.08)])
    if lead_params:
        mean = np.concatenate([mean, seed_lead[lead_local_indices]])
        sd = np.concatenate([sd, np.full(lead_params, 2.0)])

    time_w = {
        "geometry": 8.0,
        "timing": 20.0,
        "aggressive": 45.0,
        "speed": 200.0,
    }[args.mode]
    cov_w = {
        "geometry": 250.0,
        "timing": 200.0,
        "aggressive": 100.0,
        "speed": 30.0,
    }[args.mode]

    demo_states = {
        "pos": np.repeat(start_p[None], 8, 0).astype(np.float32),
        "vel": np.repeat(start_v[None], 8, 0).astype(np.float32),
        "quat": np.repeat(start_quat[None], 8, 0),
        "gate": np.full(8, SUFFIX_FIRST, np.float32),
    }

    exact_eval_round = 0

    def exact_covis_reference(off):
        ref = {key: np.array(value, copy=True)
               for key, value in exact_demo_covis.items()}
        gate_rows = ref["gate"].astype(np.int64)
        for local_gate in range(N_SUFFIX):
            rows = np.flatnonzero(gate_rows == local_gate)
            phase = np.linspace(0.0, 1.0, len(rows), dtype=np.float64)
            prev = off[local_gate - 1] if local_gate else np.zeros(2)
            local = prev[None] + phase[:, None] * (
                off[local_gate] - prev
            )[None]
            world_offset = (
                local[:, :1] * gR_s[local_gate, :, 0][None]
                + local[:, 1:] * gR_s[local_gate, :, 2][None]
            )
            ref["pos"][rows] += world_offset
        return ref

    def materialize_exact_demo(ref, index):
        ref_path = exact_tmp / f"ref_{index}.npz"
        suffix_path = exact_tmp / f"suffix_{index}.npz"
        hybrid_path = exact_tmp / f"hybrid_{index}.npz"
        np.savez(
            ref_path,
            ref_pos=ref["pos"],
            ref_vel=ref["vel"],
            ref_quat=ref["quat_wxyz"],
            ref_gate=ref["gate"] + SUFFIX_FIRST,
        )
        np.savez(
            suffix_path,
            **build_suffix_demo(ref_path, Path(args.map), Path(args.model)),
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts/build_vq2_hybrid_suffix_demo.py"),
                "--prefix", str(exact_prefix),
                "--suffix", str(suffix_path),
                "--out", str(hybrid_path),
                "--map", str(args.map),
                "--suffix-start-gate", str(SUFFIX_FIRST),
                "--bridge-rows", str(args.bridge_rows),
            ],
            cwd=REPO,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return hybrid_path

    def format_gate_values(original, updates):
        values = {}
        for item in str(original or "").split(","):
            if item.strip():
                gate, value = item.split(":", 1)
                values[int(gate)] = float(value)
        values.update(updates)
        return ",".join(f"{gate}:{values[gate]:.9g}" for gate in sorted(values))

    def materialize_exact_config(theta, index):
        off, sc = unpack(theta)
        leads = unpack_leads(theta)
        document = json.loads(json.dumps(exact_config_doc))
        cfg_args = document["args"]
        gates = range(SUFFIX_FIRST, SUFFIX_FIRST + N_SUFFIX)
        cfg_args["reference_lateral_offsets"] = format_gate_values(
            cfg_args.get("reference_lateral_offsets", ""),
            {gate: float(off[j, 0]) for j, gate in enumerate(gates)},
        )
        cfg_args["reference_vertical_offsets"] = format_gate_values(
            cfg_args.get("reference_vertical_offsets", ""),
            {gate: float(off[j, 1]) for j, gate in enumerate(gates)},
        )
        if args.mode != "geometry":
            cfg_args["reference_velocity_scales"] = format_gate_values(
                cfg_args.get("reference_velocity_scales", ""),
                {gate: float(sc[j]) for j, gate in enumerate(gates)},
            )
        if lead_params:
            cfg_args["reference_action_leads"] = format_gate_values(
                cfg_args.get("reference_action_leads", ""),
                {gate: int(leads[j]) for j, gate in enumerate(gates)},
            )
        path = exact_tmp / f"config_{index}.json"
        path.write_text(json.dumps(document, indent=2))
        return path

    def evaluate_exact(refs, thetas):
        nonlocal exact_eval_round
        per = args.n_envs
        out = []
        world_seed = args.seed + 1009 * exact_eval_round
        exact_eval_round += 1
        for ci, (ref, theta) in enumerate(zip(refs, thetas)):
            if args.exact_search_space == "config":
                config_path = materialize_exact_config(theta, ci)
                demo_path = exact_prefix
            else:
                config_path = args.live_teacher_config
                demo_path = materialize_exact_demo(ref, ci)
            cfg = FastEnvConfig(
                n_envs=per,
                race_gates=17,
                random_start_frac=1.0,
                spawn_at_rest=False,
                max_episode_s=20.0,
                auto_reset=False,
                speed_cap_mps=12.5,
                act_delay_steps_min=0,
                act_delay_steps_max=0,
                residual_scale=0.0,
                reloc_events=True,
                fov_vision=True,
                world_model_aleatoric_scale=1.5,
                demo_corridor_m=999.0,
                start_noise_pos_m=0.08,
                start_noise_vel_mps=0.20,
            )
            cfg.apply_multigate10hz()
            cfg.dr_thrust = (0.97, 1.03)
            cfg.dr_rate_gain = (0.95, 1.05)
            cfg.dr_rate_tau = (0.90, 1.10)
            cfg.dr_drag = (0.20, 0.35)
            backbone = BatchedLiveTeacher(
                config_path,
                n_envs=per,
                device=str(device),
                demo_path=demo_path,
            )
            torch.manual_seed(world_seed)
            env = FastVQ2Env(
                model,
                args.map,
                demo_states=exact_handoffs,
                config=cfg,
                device=str(device),
                obstacles_path=args.obstacles or None,
                backbone=backbone,
                residual_ensemble=exact_ensemble,
            )
            zeros = torch.zeros(per, ACT_DIM, device=device)
            fin = torch.zeros(per, dtype=torch.bool, device=device)
            fail = torch.zeros_like(fin)
            fail_gate = torch.full(
                (per,), -1, dtype=torch.long, device=device
            )
            t_fin = torch.full((per,), float("nan"), device=device)
            t12 = torch.zeros(per, device=device)
            t13 = torch.zeros(per, device=device)
            with torch.no_grad():
                for _ in range(int(cfg.max_episode_s * cfg.control_hz) + 1):
                    _o, _r, done, info = env.step(zeros)
                    live = ~(fin | fail)
                    passed = info["passed"] & live
                    t12 = torch.where(
                        passed & (info["target"] == 13), info["t_ep"], t12
                    )
                    t13 = torch.where(
                        passed & (info["target"] == 14), info["t_ep"], t13
                    )
                    newf = info["finished"] & live
                    t_fin = torch.where(newf, info["t_ep"], t_fin)
                    fin |= newf
                    newfail = done & live & ~info["finished"]
                    fail_gate = torch.where(
                        newfail, info["target"], fail_gate
                    )
                    fail |= newfail
                    if bool((fin | fail).all()):
                        break
            split = t13 - t12
            valid_split = fin & (split > 0)
            hist = {
                str(g): int((fail_gate == g).sum())
                for g in torch.unique(fail_gate[fail]).cpu().tolist()
            }
            out.append({
                "finish_rate": float(fin.float().mean()),
                "t_med": float(t_fin[fin].median()) if bool(fin.any()) else None,
                "g13_split_med": (
                    float(split[valid_split].median())
                    if bool(valid_split.any()) else None
                ),
                "failure_gate_hist": hist,
            })
            del env, backbone
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return out

    def evaluate_pop(refs, thetas=None):
        if args.live_teacher_config:
            return evaluate_exact(refs, thetas)
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
        cfg.start_noise_pos_m = 0.08
        cfg.start_noise_vel_mps = 0.2
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
        with torch.no_grad():
            for _ in range(1400):
                _o, _r, done, info = env.step(zeros)
                live = ~(fin | fail)
                alive += live.float()
                pas = info["passed"] & live
                t12 = torch.where(pas & (info["target"] == 13),
                                  alive * DT, t12)
                t13 = torch.where(pas & (info["target"] == 14),
                                  alive * DT, t13)
                newf = info["finished"] & live
                t_fin = torch.where(newf, alive * DT, t_fin)
                fin |= newf
                fail |= done & live & ~info["finished"]
                if float((fin | fail).float().mean()) > 0.999:
                    break
        finC = fin.view(C, per)
        tC = t_fin.view(C, per)
        split = (t13 - t12).view(C, per)
        out = []
        for ci in range(C):
            fr = float(finC[ci].float().mean())
            ok = finC[ci]
            g13s = split[ci][ok & (split[ci] > 0)]
            out.append({
                "finish_rate": fr,
                "t_med": float(tC[ci][ok].median()) if fr else None,
                "g13_split_med": float(g13s.median()) if len(g13s) else None,
            })
        return out

    def candidate_score(metrics, coverage):
        t_med = metrics["t_med"] if metrics["t_med"] else 60.0
        if args.mode == "speed":
            shortfall = max(0.0, args.reliability_floor
                            - metrics["finish_rate"])
            if shortfall > 0.0:
                # Every feasible candidate outranks every infeasible one.
                return (
                    -1_000_000.0
                    - 10_000.0 * shortfall
                    - time_w * t_med
                )
            # Inside the feasible set, time is the objective. Reliability and
            # co-visibility only break sub-millisecond-equivalent ties.
            return (
                1_000_000.0
                - time_w * t_med
                + metrics["finish_rate"]
                + 0.01 * coverage["dual_cov"]
            )
        return (
            1000.0 * metrics["finish_rate"]
            + cov_w * coverage["dual_cov"]
            - time_w * t_med
        )

    history = []
    archive = []
    best = {"score": -1e9}
    t0 = time.time()
    for it in range(args.iters):
        thetas = mean[None] + sd[None] * rng.standard_normal(
            (args.pop, n_par))
        thetas[0] = mean
        if best.get("theta") is not None:
            thetas[1] = best["theta"]
        refs, covs = [], []
        for th in thetas:
            off, sc = unpack(th)
            line_sc = sc if len(sc) == N_SUFFIX + 1 else np.concatenate(
                [sc, seed_line_sc[-1:]]
            )
            ref = build_reference(gp_s, gR_s, off, line_sc, lcfg)
            refs.append(ref)
            cov_ref = (
                exact_covis_reference(off)
                if args.live_teacher_config
                and args.exact_search_space == "config"
                else ref
            )
            covs.append(covis_metrics(cov_ref, all_corners, all_centers,
                                      all_normals, K, R_cb, wdet))
        mets = evaluate_pop(refs, thetas)
        results = []
        for th, ref, cov, m in zip(thetas, refs, covs, mets):
            score = candidate_score(m, cov)
            results.append((score, th))
            archive.append((score, th.copy()))
            if score > best["score"]:
                best = {"score": score, "theta": th.copy(),
                        "metrics": {**m, **cov},
                        "planned": ref["planned_lap_s"]}
        results.sort(key=lambda r: -r[0])
        elite = np.stack([th for _s, th in results[:args.elite]])
        mean = 0.4 * mean + 0.6 * elite.mean(0)
        sd = 0.5 * sd + 0.5 * (elite.std(0) + 0.015)
        history.append({"iter": it, "best": results[0][0],
                        "champion": best["score"]})
        print(f"[{args.mode} it {it}] best {results[0][0]:.1f} "
              f"champ {best['score']:.1f} "
              f"{json.dumps(best['metrics'])} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Final: rerank several unique finalists together on one untouched,
    # common-random-number pool. This prevents the luckiest candidate from any
    # one CEM generation becoming champion merely through winner's curse.
    archive.sort(key=lambda item: -item[0])
    finalist_thetas = []
    seen = set()
    for _score, theta in archive:
        key = np.round(theta, 10).tobytes()
        if key in seen:
            continue
        seen.add(key)
        finalist_thetas.append(theta)
        if len(finalist_thetas) >= max(1, args.finalists):
            break
    finalist_refs, finalist_covs = [], []
    for theta in finalist_thetas:
        off_i, sc_i = unpack(theta)
        line_sc_i = (
            sc_i if len(sc_i) == N_SUFFIX + 1
            else np.concatenate([sc_i, seed_line_sc[-1:]])
        )
        ref_i = build_reference(gp_s, gR_s, off_i, line_sc_i, lcfg)
        cov_ref_i = (
            exact_covis_reference(off_i)
            if args.live_teacher_config and args.exact_search_space == "config"
            else ref_i
        )
        finalist_refs.append(ref_i)
        finalist_covs.append(covis_metrics(
            cov_ref_i, all_corners, all_centers, all_normals, K, R_cb, wdet
        ))
    args.n_envs = args.final_envs
    finalist_metrics = evaluate_pop(finalist_refs, finalist_thetas)
    finalist_scores = [
        candidate_score(metrics, coverage)
        for metrics, coverage in zip(finalist_metrics, finalist_covs)
    ]
    winner_i = int(np.argmax(finalist_scores))
    best["theta"] = finalist_thetas[winner_i]
    final = finalist_metrics[winner_i]
    cov = finalist_covs[winner_i]
    off, sc = unpack(best["theta"])
    ref = finalist_refs[winner_i]
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{out_prefix}_best.npz",
             theta=best["theta"], suffix_offsets=off, suffix_scales=sc,
             suffix_action_leads=unpack_leads(best["theta"]),
             ref_pos=ref["pos"], ref_vel=ref["vel"],
             ref_quat=ref["quat_wxyz"], ref_gate=ref["gate"] + SUFFIX_FIRST)
    report = {
        "mode": args.mode,
        "handoff": handoff["source"],
        "planned_suffix_s": ref["planned_lap_s"],
        "coverage": cov,
        "final_worlds": args.final_envs,
        "final": final,
        "finalists": [
            {
                "rank": index + 1,
                "score": float(score),
                "metrics": metrics,
                "suffix_scales": unpack(theta)[1].tolist(),
                "suffix_action_leads": unpack_leads(theta).tolist(),
            }
            for index, (score, metrics, theta) in enumerate(sorted(
                zip(finalist_scores, finalist_metrics, finalist_thetas),
                key=lambda item: -item[0],
            ))
        ],
        "history": history,
        "suffix_offsets": off.tolist(),
        "suffix_scales": sc.tolist(),
        "suffix_action_leads": unpack_leads(best["theta"]).tolist(),
    }
    Path(f"{out_prefix}_report.json").write_text(
        json.dumps(report, indent=1))
    if args.live_teacher_config and args.exact_search_space == "config":
        winner_config = materialize_exact_config(best["theta"], "winner")
        shutil.copy2(winner_config, f"{out_prefix}_config.json")
    print(json.dumps({k: v for k, v in report.items()
                      if k != "history"}, indent=1))
    if exact_tmp is not None:
        shutil.rmtree(exact_tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
