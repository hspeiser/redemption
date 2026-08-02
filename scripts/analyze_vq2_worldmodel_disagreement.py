"""Locate control-relevant disagreement between two VQ2 world models.

The report separates passive prediction disagreement from disagreement in the
action Jacobian.  The latter answers the active-identification question: at
which gate/range and on which control axis will a small, bounded live probe be
most informative about the real vehicle dynamics?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.fastsim.worldmodel import (  # noqa: E402
    ResidualEnsemble,
    residual_features,
)
from scripts.train_vq2_g0g4_worldmodel import (  # noqa: E402
    features_targets,
    tensors,
)


DISTANCE_EDGES = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0, 12.0, np.inf])
AXIS_NAMES = ("roll_rate", "pitch_rate", "yaw_rate", "thrust")


@torch.no_grad()
def model_mean(model: ResidualEnsemble, features: torch.Tensor) -> torch.Tensor:
    mean, _log_std = model(features)
    return mean.mean(0)


def quantiles(values: np.ndarray) -> dict:
    if not len(values):
        return {"p50": None, "p90": None, "mean": None}
    return {
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "mean": float(np.mean(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True, action="append")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--action-epsilon", type=float, default=0.08)
    args = parser.parse_args()
    if len(args.ensemble) != 2:
        parser.error("exactly two --ensemble arguments are required")

    payload = np.load(args.dataset / f"{args.split}.npz")
    data = tensors(payload, args.device)
    gate_payload = json.loads(args.map.read_text())
    gate_positions_np = np.asarray(
        [gate["pos"] for gate in gate_payload["gates"]], np.float32
    )
    gate_positions = torch.as_tensor(
        gate_positions_np, dtype=torch.float32, device=args.device
    )
    base = SurrogateModel.load(args.base_model)
    features, target, keep = features_targets(data, base, gate_positions)
    indices = torch.nonzero(keep, as_tuple=False)[:, 0]

    models = []
    metadata = []
    for path in args.ensemble:
        model, meta = ResidualEnsemble.load(path, args.device)
        model.eval()
        models.append(model)
        metadata.append(meta)

    count = len(indices)
    passive = np.empty((count, 3), np.float32)
    errors = np.empty((2, count, 3), np.float32)
    jacobian_gap = np.empty((count, 4, 3), np.float32)
    gates = data["gate_index"][indices].detach().cpu().numpy()
    positions = data["position"][indices].detach().cpu().numpy()
    gate_targets = gate_positions_np[np.clip(gates, 0, len(gate_positions_np) - 1)]
    distances = np.linalg.norm(gate_targets - positions, axis=1)

    for begin in range(0, count, args.batch_size):
        end = min(count, begin + args.batch_size)
        row = indices[begin:end]
        feat = features[row]
        truth = target[row]
        means = [model_mean(model, feat) for model in models]
        delta = means[0] - means[1]
        passive[begin:end, 0] = torch.linalg.norm(delta[:, :3], dim=1).cpu()
        passive[begin:end, 1] = torch.linalg.norm(delta[:, 3:6], dim=1).cpu()
        passive[begin:end, 2] = torch.linalg.norm(delta[:, 6:9], dim=1).cpu()
        for model_index, mean in enumerate(means):
            error = mean - truth
            errors[model_index, begin:end, 0] = torch.linalg.norm(
                error[:, :3], dim=1
            ).cpu()
            errors[model_index, begin:end, 1] = torch.linalg.norm(
                error[:, 3:6], dim=1
            ).cpu()
            errors[model_index, begin:end, 2] = torch.linalg.norm(
                error[:, 6:9], dim=1
            ).cpu()

        state_velocity = data["velocity"][row]
        state_rotation = data["rotation"][row]
        state_rates = data["rates"][row]
        previous_action = data["previous_action"][row]
        action = data["action"][row]
        for axis in range(4):
            plus = action.clone()
            minus = action.clone()
            plus[:, axis] = torch.clamp(
                plus[:, axis] + args.action_epsilon, -1.0, 1.0
            )
            minus[:, axis] = torch.clamp(
                minus[:, axis] - args.action_epsilon, -1.0, 1.0
            )
            plus_features = residual_features(
                state_velocity, state_rotation, state_rates, plus,
                previous_action,
            )
            minus_features = residual_features(
                state_velocity, state_rotation, state_rates, minus,
                previous_action,
            )
            derivatives = []
            denominator = torch.clamp(
                plus[:, axis] - minus[:, axis], min=1e-4
            )[:, None]
            for model in models:
                derivatives.append(
                    (model_mean(model, plus_features)
                     - model_mean(model, minus_features)) / denominator
                )
            gap = derivatives[0] - derivatives[1]
            jacobian_gap[begin:end, axis, 0] = torch.linalg.norm(
                gap[:, :3], dim=1
            ).cpu()
            jacobian_gap[begin:end, axis, 1] = torch.linalg.norm(
                gap[:, 3:6], dim=1
            ).cpu()
            jacobian_gap[begin:end, axis, 2] = torch.linalg.norm(
                gap[:, 6:9], dim=1
            ).cpu()

    rows = []
    for gate in range(5):
        for distance_index in range(len(DISTANCE_EDGES) - 1):
            lo, hi = DISTANCE_EDGES[distance_index:distance_index + 2]
            selected = (
                (gates == gate) & (distances >= lo) & (distances < hi)
            )
            if selected.sum() < 25:
                continue
            action_rows = []
            for axis, name in enumerate(AXIS_NAMES):
                # Velocity response is most important for path identification;
                # rate response breaks ties for attitude-loop uncertainty.
                score = (
                    jacobian_gap[selected, axis, 0]
                    + 0.25 * jacobian_gap[selected, axis, 2]
                )
                action_rows.append({
                    "axis": axis,
                    "name": name,
                    "identification_score": quantiles(score),
                    "velocity_jacobian_gap": quantiles(
                        jacobian_gap[selected, axis, 0]
                    ),
                    "attitude_jacobian_gap": quantiles(
                        jacobian_gap[selected, axis, 1]
                    ),
                    "rate_jacobian_gap": quantiles(
                        jacobian_gap[selected, axis, 2]
                    ),
                })
            action_rows.sort(
                key=lambda row: row["identification_score"]["mean"],
                reverse=True,
            )
            rows.append({
                "gate": gate,
                "distance_m": [float(lo), None if np.isinf(hi) else float(hi)],
                "samples": int(selected.sum()),
                "passive_disagreement": {
                    "velocity_mps": quantiles(passive[selected, 0]),
                    "attitude_rad": quantiles(passive[selected, 1]),
                    "rate_radps": quantiles(passive[selected, 2]),
                },
                "model_error": [
                    {
                        "velocity_mps": quantiles(errors[i, selected, 0]),
                        "attitude_rad": quantiles(errors[i, selected, 1]),
                        "rate_radps": quantiles(errors[i, selected, 2]),
                    }
                    for i in range(2)
                ],
                "action_axes": action_rows,
            })

    ranking = sorted(
        rows,
        key=lambda row: row["action_axes"][0]["identification_score"]["mean"],
        reverse=True,
    )
    result = {
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "models": [
            {"path": str(path.resolve()), "dataset": meta.get("dataset")}
            for path, meta in zip(args.ensemble, metadata)
        ],
        "valid_samples": count,
        "action_epsilon": args.action_epsilon,
        "ranking": ranking,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"valid_samples": count, "top": ranking[:12]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
