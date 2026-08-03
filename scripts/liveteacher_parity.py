"""Layer-1 parity: batched live-teacher port vs the golden 35.37 fixture.

Replays all 1,067 recorded control steps through BatchedLiveTeacher
(n_envs=1, recorded observations fed to the residual actors) and diffs
every component the fixture records.  Acceptance per the oracle doc:
exact reference rows / segment bounds / routing, final-action max abs
error <= 1e-5, with per-component first-divergence attribution.

    python scripts/liveteacher_parity.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.liveteacher import (  # noqa: E402
    BatchedLiveTeacher, observation_rotation_batch,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(
        REPO / "tests/fixtures/vq2_35p37_teacher_parity_v1.npz"))
    ap.add_argument(
        "--config",
        help=("Exact config used to generate the fixture. Defaults to the "
              "config_source recorded in the adjacent fixture manifest."),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(
        REPO / "data/lineopt/liveteacher_parity_layer1.json"))
    args = ap.parse_args()

    fixture_path = Path(args.fixture)
    manifest_path = fixture_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    config_path = Path(args.config or manifest["config_source"])
    expected_config_hash = manifest.get("config_sha256")
    actual_config_hash = sha256(config_path)
    if expected_config_hash and actual_config_hash != expected_config_hash:
        raise SystemExit(
            "fixture/config mismatch: parity is meaningless unless the "
            f"config hash is {expected_config_hash}; got "
            f"{actual_config_hash} from {config_path}"
        )

    fx = np.load(fixture_path, allow_pickle=True)
    obs = np.asarray(fx["observation"], np.float32)
    n_steps = len(obs)
    teacher = BatchedLiveTeacher(config_path, n_envs=1,
                                 device=args.device)
    gate_pos = teacher.demo_gate_position.cpu().numpy()

    rot = observation_rotation_batch(obs)
    gate = np.asarray(fx["gate_index"], np.int64)
    gate_vec = np.einsum("nij,nj->ni", rot, obs[:, :3]) * 10.0
    pos = gate_pos[np.clip(gate, 0, 16)] - gate_vec
    vel = np.einsum("nij,nj->ni", rot, obs[:, 18:21]) * 10.0

    rows_ok = seg_ok = 0
    diffs = {k: [] for k in
             ("final", "base_vs_teacher", "actor", "lat_fb",
              "lat_gain", "vel_scale", "thrust_scale", "lead",
              "funnel")}
    row_mism = []
    first_divergence = None
    teacher.reset(torch.tensor([0]))
    for k in range(n_steps):
        p = torch.tensor(pos[k], dtype=torch.float32)[None]
        v = torch.tensor(vel[k], dtype=torch.float32)[None]
        R = torch.tensor(rot[k], dtype=torch.float32)[None]
        pa = torch.tensor(obs[k, 30:34], dtype=torch.float32)[None]
        tg = torch.tensor([int(gate[k])])
        ov = torch.tensor(obs[k], dtype=torch.float32)[None]
        act, dbg = teacher.action(p, v, R, prev_action=pa, target=tg,
                                  debug=True, obs_override=ov)
        row = int(dbg["reference_row"][0])
        row_ref = int(fx["reference_row_int"][k])
        if row == row_ref:
            rows_ok += 1
        elif len(row_mism) < 12:
            row_mism.append((k, int(gate[k]), row, row_ref))
        if (int(dbg["seg_start"][0]) == int(fx["reference_segment_start"][k])
                and int(dbg["seg_end"][0]) == int(
                    fx["reference_segment_end"][k])):
            seg_ok += 1
        diffs["final"].append(float(np.abs(
            act[0].numpy() - fx["final_action"][k]).max()))
        diffs["base_vs_teacher"].append(float(np.abs(
            dbg["reference_pre_residual"][0].numpy()
            - fx["teacher_action"][k]).max()))
        diffs["actor"].append(float(np.abs(
            dbg["res_mean"][0].numpy() - fx["actor_mean"][k]).max()))
        diffs["lat_fb"].append(float(abs(
            float(dbg["raw_lateral_feedback"][0])
            - float(fx["raw_lateral_feedback"][k]))))
        diffs["lat_gain"].append(float(abs(
            float(dbg["lat_gain_scale"][0])
            - float(fx["effective_lateral_gain_scale"][k]))))
        diffs["vel_scale"].append(float(abs(
            float(dbg["vel_scale"][0])
            - float(fx["effective_reference_velocity_scale"][k]))))
        diffs["thrust_scale"].append(float(abs(
            float(dbg["thrust_scale"][0])
            - float(fx["effective_reference_thrust_scale"][k]))))
        diffs["lead"].append(float(abs(
            float(dbg["action_lead"][0])
            - float(fx["effective_reference_action_lead"][k]))))
        diffs["funnel"].append(float(abs(
            float(dbg["funnel_weight"][0])
            - float(fx["gate_center_funnel_weight"][k]))))
        if first_divergence is None and (
                row != row_ref or diffs["final"][-1] > 1e-5):
            worst = max(
                ((name, vals[-1]) for name, vals in diffs.items()),
                key=lambda kv: kv[1])
            first_divergence = {
                "step": k, "gate": int(gate[k]),
                "row_port": row, "row_fixture": row_ref,
                "worst_component": worst[0],
                "worst_err": worst[1],
                "component_errs": {name: vals[-1]
                                   for name, vals in diffs.items()},
            }

    summary = {
        "steps": n_steps,
        "config": str(config_path.resolve()),
        "config_sha256": actual_config_hash,
        "rows_exact": rows_ok,
        "segments_exact": seg_ok,
        "first_divergence": first_divergence,
    }
    for name, vals in diffs.items():
        arr = np.asarray(vals)
        summary[name] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }
    summary["row_mismatches_sample"] = row_mism
    summary["PASS"] = bool(
        rows_ok == n_steps and seg_ok == n_steps
        and summary["final"]["max"] <= 1e-5)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if summary["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
