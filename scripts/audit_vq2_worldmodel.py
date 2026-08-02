"""Audit an existing residual world model on a frozen dataset split."""

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
from aigp.fastsim.worldmodel import ResidualEnsemble  # noqa: E402
from scripts.train_vq2_g0g4_worldmodel import (  # noqa: E402
    features_targets,
    one_step_report,
    rollout_report,
    tensors,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    ensemble, source_metadata = ResidualEnsemble.load(args.model, device)
    ensemble.eval()
    base = SurrogateModel.load(args.base_model)
    gates = json.loads(args.map.read_text())["gates"]
    gate_positions = torch.as_tensor(
        [gate["pos"] for gate in gates], dtype=torch.float32, device=device
    )
    report = {
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "source_metadata": source_metadata,
    }
    for split in ("validation", "test"):
        data = tensors(np.load(args.dataset / f"{split}.npz"), device)
        valid = features_targets(data, base, gate_positions)[2]
        report[split] = {
            "rows": int(len(data["action"])),
            "physically_valid_rows": int(valid.sum()),
            "one_step": one_step_report(ensemble, data, base, gate_positions),
            "rollouts": rollout_report(
                ensemble, data, base, gate_positions, valid
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        split: report[split]["rollouts"] for split in ("validation", "test")
    }, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
