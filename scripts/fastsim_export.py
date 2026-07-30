"""Export a fastsim PPO checkpoint as a deployable policy artifact:
GaussianActor weights + observation normalization + metadata, one file.

    python scripts/fastsim_export.py --ckpt data/fastsim_runs/ppo_v1/latest.pt \
        --out data/models/vq2_ppo_policy.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--out", default=str(REPO / "data" / "models" / "vq2_ppo_policy.pt")
    )
    args = parser.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    artifact = {
        "actor": ck["actor"],
        "obs_mean": ck["obs_mean"].cpu(),
        "obs_var": ck["obs_var"].cpu(),
        "obs_dim": 53,
        "act_dim": 4,
        "source_iter": ck.get("iter"),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trained_on": "fastsim surrogate (vq2_map_final, DR+noise)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out)
    print(f"wrote {out} (iter {artifact['source_iter']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
