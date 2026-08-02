"""Compare multiple residual models on identical frozen rollout starts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import (  # noqa: E402
    ResidualEnsemble,
    analytic_step,
    residual_features,
    rotation_exp,
    rotation_log,
)
from scripts.train_vq2_g0g4_worldmodel import (  # noqa: E402
    features_targets,
    tensors,
)


def valid_starts(data, transition_valid, horizon):
    episode, step, done = data["episode"], data["step"], data["done"]
    valid = torch.ones(len(step) - horizon, dtype=torch.bool, device=step.device)
    for offset in range(1, horizon + 1):
        n = len(valid)
        valid &= (
            (episode[:n] == episode[offset:offset + n])
            & (step[offset:offset + n] == step[:n] + offset)
            & (done[offset - 1:offset - 1 + n] == 0)
            & transition_valid[offset - 1:offset - 1 + n]
        )
    return torch.nonzero(valid).squeeze(1)


@torch.no_grad()
def rollout(ensemble, data, base, starts, horizon):
    p = data["position"][starts].clone()
    v = data["velocity"][starts].clone()
    rotation = data["rotation"][starts].clone()
    rates = data["rates"][starts].clone()
    previous = data["previous_action"][starts].clone()
    for offset in range(horizon):
        action = data["action"][starts + offset]
        features = residual_features(v, rotation, rates, action, previous)
        means, _ = ensemble(features)
        correction = means.mean(0)
        p, nominal_v, nominal_rotation, nominal_rates = analytic_step(
            p, v, rotation, rates, action, base
        )
        v = nominal_v + torch.einsum("nij,nj->ni", rotation, correction[:, :3])
        rotation = nominal_rotation @ rotation_exp(correction[:, 3:6])
        rates = nominal_rates + correction[:, 6:9]
        previous = action
    target = starts + horizon
    return {
        "position_m": torch.linalg.norm(
            p - data["position"][target], dim=1
        ).cpu().numpy(),
        "velocity_mps": torch.linalg.norm(
            v - data["velocity"][target], dim=1
        ).cpu().numpy(),
        "attitude_deg": torch.rad2deg(torch.linalg.norm(rotation_log(
            rotation.transpose(1, 2) @ data["rotation"][target]
        ), dim=1)).cpu().numpy(),
        "rate_radps": torch.linalg.norm(
            rates - data["rates"][target], dim=1
        ).cpu().numpy(),
    }


def stats(values):
    values = np.asarray(values, float)
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "mean": float(np.mean(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="NAME=PATH (repeatable)")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()

    base = SurrogateModel.load(args.base_model)
    gates = json.loads(args.map.read_text())["gates"]
    gate_positions = torch.tensor(
        [gate["pos"] for gate in gates], dtype=torch.float32, device=args.device
    )
    data = tensors(np.load(args.dataset / "test.npz"), args.device)
    transition_valid = features_targets(data, base, gate_positions)[2]
    starts = valid_starts(data, transition_valid, args.horizon)
    models = {}
    for item in args.model:
        name, raw_path = item.split("=", 1)
        models[name] = ResidualEnsemble.load(raw_path, args.device)[0].eval()
    metrics = {
        name: rollout(model, data, base, starts, args.horizon)
        for name, model in models.items()
    }

    # Oracle integration isolates the irreducible disagreement between the
    # logged EKF position and its own logged velocity sequence.
    oracle_p = data["position"][starts].clone()
    for offset in range(args.horizon):
        oracle_p += data["next_velocity"][starts + offset] / 30.0
    oracle_position = torch.linalg.norm(
        oracle_p - data["position"][starts + args.horizon], dim=1
    ).cpu().numpy()
    position_errors = {
        name: value["position_m"] for name, value in metrics.items()
    }
    position_errors["oracle_velocity"] = oracle_position

    manifest = json.loads((args.dataset / "manifest.json").read_text())
    id_to_path = {
        int(row.get("session_id", index)): row["path"]
        for index, row in enumerate(manifest["sessions"])
    }
    session = data["session"][starts].cpu().numpy()
    gate = data["gate_index"][starts].cpu().numpy()
    speed = torch.linalg.norm(data["velocity"][starts], dim=1).cpu().numpy()
    groups = {}
    selectors = {
        **{f"gate_{value}": gate == value for value in np.unique(gate)},
        **{f"speed_{lo}_{lo + 2}": (speed >= lo) & (speed < lo + 2)
           for lo in range(0, 16, 2)},
        **{f"session_{value}": session == value for value in np.unique(session)},
    }
    for key, selector in selectors.items():
        if not np.any(selector):
            continue
        groups[key] = {
            "session_path": id_to_path.get(int(key.split("_")[1]), "")
            if key.startswith("session_") else "",
            **{name: stats(value[selector])
               for name, value in position_errors.items()},
        }
    report = {
        "horizon": args.horizon,
        "seconds": args.horizon / 30.0,
        "starts": int(len(starts)),
        "overall": {
            name: {metric: stats(value) for metric, value in rows.items()}
            for name, rows in metrics.items()
        } | {"oracle_velocity": {"position_m": stats(oracle_position)}},
        "groups": groups,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
