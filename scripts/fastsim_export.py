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
    config = ck.get("config", {})

    def file_hash(path):
        import hashlib
        try:
            return hashlib.sha256(
                Path(path).read_bytes()
            ).hexdigest()[:16]
        except (OSError, TypeError):
            return None

    artifact = {
        "actor": ck["actor"],
        "obs_mean": ck["obs_mean"].cpu(),
        "obs_var": ck["obs_var"].cpu(),
        "obs_dim": 53,
        "act_dim": 4,
        "source_iter": ck.get("iter"),
        "source_ckpt": str(args.ckpt),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # full provenance: the exact training configuration, plus
        # content hashes of the world files it trained against
        "train_config": config,
        "input_hashes": {
            key: file_hash(config.get(key))
            for key in ("map", "model", "demo_npz", "obstacles")
            if config.get(key)
        },
        "trained_on": (
            f"fastsim map={Path(str(config.get('map', '?'))).name} "
            f"model={Path(str(config.get('model', '?'))).name} "
            f"speed_cap={config.get('speed_cap')} "
            f"fov={config.get('fov_vision')} "
            f"noise_era={config.get('noise_era')} "
            f"corridor={config.get('demo_corridor')}"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out)
    print(f"wrote {out} (iter {artifact['source_iter']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
