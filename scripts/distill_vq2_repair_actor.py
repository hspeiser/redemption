"""Distill an accepted synthetic repair into a protected actor copy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.sac import GaussianActor  # noqa: E402
from aigp.rl.counterfactual_gate import build_state_gate  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repair-dataset", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--anchor-replay", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--repair-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--focus-anchor-gate", type=int, default=1)
    parser.add_argument("--focus-anchor-weight", type=float, default=4.0)
    parser.add_argument("--state-gate-hidden", default="32,16")
    parser.add_argument("--state-gate-steps", type=int, default=2000)
    parser.add_argument(
        "--state-gate-false-positive-rate", type=float, default=0.001
    )
    parser.add_argument(
        "--state-gate-min-repair-recall", type=float, default=0.95
    )
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"immutable distilled checkpoint exists: {args.out}")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    payload = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    if payload.get("kind") != "vq2_residual_sac":
        raise ValueError("distillation requires a vq2_residual_sac checkpoint")
    actor = GaussianActor(53, 4).to(args.device)
    actor.load_state_dict(payload["actor"])
    protected = copy.deepcopy(actor).eval()
    for parameter in protected.parameters():
        parameter.requires_grad_(False)
    repair_npz = np.load(args.repair_dataset, allow_pickle=False)
    repair_observation = np.asarray(
        repair_npz["observation"], np.float32
    )
    repair_action = np.asarray(repair_npz["action"], np.float32)
    repair_world = np.asarray(repair_npz["world_id"], np.int64)
    repair_family = np.asarray(repair_npz["model_family"], np.int64)
    if not np.all(np.asarray(repair_npz["synthetic_repair"], bool)):
        raise ValueError("repair dataset contains non-synthetic rows")
    if np.any(np.asarray(repair_npz["dynamics_eligible"], bool)):
        raise ValueError("synthetic rows must never be dynamics-eligible")
    anchor_npz = np.load(args.anchor_replay, allow_pickle=False)
    anchor_observation_all = np.asarray(
        anchor_npz["observation"], np.float32
    )
    is_demo = np.asarray(anchor_npz["is_demo"], bool)
    anchor_gate_all = np.asarray(anchor_npz["gate_index"], np.int16)
    anchor_observation = anchor_observation_all[is_demo]
    anchor_gate = anchor_gate_all[is_demo]
    observation_mean = np.asarray(payload["observation_mean"], np.float32)
    observation_std = np.asarray(payload["observation_std"], np.float32)

    def normalize(value: np.ndarray) -> np.ndarray:
        return (value - observation_mean) / observation_std

    repair_observation = normalize(repair_observation)
    anchor_observation = normalize(anchor_observation)
    # Hold out complete synthetic worlds, not correlated individual steps.
    repair_validation_mask = (
        (repair_world * 17 + repair_family * 101 + args.seed) % 5 == 0
    )
    repair_train_observation = repair_observation[~repair_validation_mask]
    repair_train_action = repair_action[~repair_validation_mask]
    repair_validation_observation = repair_observation[repair_validation_mask]
    repair_validation_action = repair_action[repair_validation_mask]
    anchor_order = rng.permutation(len(anchor_observation))
    anchor_split = max(1, int(0.90 * len(anchor_order)))
    anchor_train_observation = anchor_observation[anchor_order[:anchor_split]]
    anchor_validation_observation = anchor_observation[anchor_order[anchor_split:]]
    focus_observation = anchor_observation[
        anchor_gate == int(args.focus_anchor_gate)
    ]
    focus_order = rng.permutation(len(focus_observation))
    focus_split = max(1, int(0.90 * len(focus_order)))
    focus_train_observation = focus_observation[focus_order[:focus_split]]
    focus_validation_observation = focus_observation[focus_order[focus_split:]]
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=1e-6
    )
    history = []
    actor.train()
    for step in range(args.steps):
        repair_index = rng.integers(
            0, len(repair_train_observation), size=args.batch_size
        )
        anchor_index = rng.integers(
            0, len(anchor_train_observation), size=args.batch_size
        )
        focus_index = rng.integers(
            0, len(focus_train_observation), size=args.batch_size
        )
        repair_obs = torch.as_tensor(
            repair_train_observation[repair_index], device=args.device
        )
        repair_target = torch.as_tensor(
            repair_train_action[repair_index], device=args.device
        )
        anchor_obs = torch.as_tensor(
            anchor_train_observation[anchor_index], device=args.device
        )
        focus_obs = torch.as_tensor(
            focus_train_observation[focus_index], device=args.device
        )
        repair_prediction = actor.deterministic(repair_obs)
        anchor_prediction = actor.deterministic(anchor_obs)
        with torch.no_grad():
            anchor_target = protected.deterministic(anchor_obs)
            _old_mean, old_log_std = protected.distribution(anchor_obs)
            focus_target = protected.deterministic(focus_obs)
        _new_mean, new_log_std = actor.distribution(anchor_obs)
        repair_loss = F.mse_loss(repair_prediction, repair_target)
        anchor_loss = F.mse_loss(anchor_prediction, anchor_target)
        focus_loss = F.mse_loss(
            actor.deterministic(focus_obs), focus_target
        )
        uncertainty_loss = F.mse_loss(new_log_std, old_log_std)
        loss = (
            args.repair_weight * repair_loss
            + args.anchor_weight * anchor_loss
            + args.focus_anchor_weight * focus_loss
            + 0.10 * args.anchor_weight * uncertainty_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "repair_mse": float(repair_loss.detach()),
                "anchor_mse": float(anchor_loss.detach()),
                "focus_anchor_mse": float(focus_loss.detach()),
                "uncertainty_mse": float(uncertainty_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    actor.eval()
    # Frozen, broad validation samples are not reused by the optimizer.
    repair_validation = rng.choice(
        len(repair_validation_observation),
        size=min(8192, len(repair_validation_observation)),
        replace=False,
    )
    anchor_validation = rng.choice(
        len(anchor_validation_observation),
        size=min(8192, len(anchor_validation_observation)),
        replace=False,
    )
    focus_validation = rng.choice(
        len(focus_validation_observation),
        size=min(8192, len(focus_validation_observation)),
        replace=False,
    )
    with torch.no_grad():
        repair_obs = torch.as_tensor(
            repair_validation_observation[repair_validation], device=args.device
        )
        repair_target = torch.as_tensor(
            repair_validation_action[repair_validation], device=args.device
        )
        anchor_obs = torch.as_tensor(
            anchor_validation_observation[anchor_validation], device=args.device
        )
        focus_obs = torch.as_tensor(
            focus_validation_observation[focus_validation], device=args.device
        )
        repair_mse = float(F.mse_loss(
            actor.deterministic(repair_obs), repair_target
        ))
        anchor_mse = float(F.mse_loss(
            actor.deterministic(anchor_obs),
            protected.deterministic(anchor_obs),
        ))
        anchor_max_abs = float(torch.max(torch.abs(
            actor.deterministic(anchor_obs)
            - protected.deterministic(anchor_obs)
        )))
        focus_anchor_mse = float(F.mse_loss(
            actor.deterministic(focus_obs),
            protected.deterministic(focus_obs),
        ))
        focus_anchor_max_abs = float(torch.max(torch.abs(
            actor.deterministic(focus_obs)
            - protected.deterministic(focus_obs)
        )))
    # A repair actor must not be allowed to perturb every state in its gate.
    # Learn a deployable classifier that separates the synthetic failure
    # corridor from healthy real demonstration states.  Hold out complete
    # synthetic worlds so correlated rollout rows cannot leak across splits.
    hidden_dims = tuple(
        int(value.strip())
        for value in args.state_gate_hidden.split(",")
        if value.strip()
    )
    state_gate = build_state_gate(53, hidden_dims).to(args.device)
    state_gate_optimizer = torch.optim.AdamW(
        state_gate.parameters(), lr=2e-3, weight_decay=1e-4
    )
    positive_train = repair_observation[
        (~repair_validation_mask)
        & (np.asarray(repair_npz["gate_index"]) == args.focus_anchor_gate)
    ]
    positive_validation = repair_observation[
        repair_validation_mask
        & (np.asarray(repair_npz["gate_index"]) == args.focus_anchor_gate)
    ]
    negative_train = focus_train_observation
    negative_validation = focus_validation_observation
    if not all(map(len, (
        positive_train, positive_validation,
        negative_train, negative_validation,
    ))):
        raise ValueError("state-gate training or validation partition is empty")
    state_gate.train()
    for _ in range(args.state_gate_steps):
        half = max(1, args.batch_size // 2)
        positive_index = rng.integers(0, len(positive_train), size=half)
        negative_index = rng.integers(0, len(negative_train), size=half)
        gate_observation = np.concatenate([
            positive_train[positive_index],
            negative_train[negative_index],
        ])
        gate_target = np.concatenate([
            np.ones(half, np.float32),
            np.zeros(half, np.float32),
        ])
        order = rng.permutation(len(gate_observation))
        gate_tensor = torch.as_tensor(
            gate_observation[order], device=args.device
        )
        target_tensor = torch.as_tensor(
            gate_target[order, None], device=args.device
        )
        gate_loss = F.binary_cross_entropy_with_logits(
            state_gate(gate_tensor), target_tensor
        )
        state_gate_optimizer.zero_grad(set_to_none=True)
        gate_loss.backward()
        torch.nn.utils.clip_grad_norm_(state_gate.parameters(), 1.0)
        state_gate_optimizer.step()
    state_gate.eval()
    with torch.no_grad():
        negative_scores = torch.sigmoid(state_gate(torch.as_tensor(
            focus_observation, device=args.device
        ))).squeeze(-1).cpu().numpy()
        positive_scores = torch.sigmoid(state_gate(torch.as_tensor(
            positive_validation, device=args.device
        ))).squeeze(-1).cpu().numpy()
    false_positive_rate = float(np.clip(
        args.state_gate_false_positive_rate, 0.0, 0.25
    ))
    threshold = float(np.quantile(
        negative_scores, 1.0 - false_positive_rate
    ))
    measured_false_positive_rate = float(np.mean(
        negative_scores >= threshold
    ))
    repair_recall = float(np.mean(positive_scores >= threshold))
    if repair_recall < args.state_gate_min_repair_recall:
        raise RuntimeError(
            "counterfactual state gate rejected: held-out repair recall "
            f"{repair_recall:.4f} < {args.state_gate_min_repair_recall:.4f}"
        )
    output = copy.deepcopy(payload)
    output["actor"] = actor.state_dict()
    output["counterfactual_protected_actor"] = protected.state_dict()
    output["counterfactual_repair_gate"] = {
        "input": "normalized_observation",
        "target_gate": int(args.focus_anchor_gate),
        "hidden_dims": list(hidden_dims),
        "state_dict": state_gate.state_dict(),
        "threshold": threshold,
        "requested_false_positive_rate": false_positive_rate,
        "measured_anchor_false_positive_rate": measured_false_positive_rate,
        "heldout_repair_recall": repair_recall,
    }
    # Stale Adam moments from the protected policy would partially undo the
    # distillation on resume.  Force any later live run to start fresh actor
    # optimizer state while preserving critic/value state.
    output.pop("actor_optimizer", None)
    output["counterfactual_repair_distillation"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "repair_dataset_sha256": sha256(args.repair_dataset),
        "repair_manifest_sha256": sha256(args.repair_manifest),
        "repair_report_sha256": sha256(args.repair_report),
        "anchor_replay_sha256": sha256(args.anchor_replay),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "repair_weight": args.repair_weight,
        "anchor_weight": args.anchor_weight,
        "focus_anchor_gate": args.focus_anchor_gate,
        "focus_anchor_weight": args.focus_anchor_weight,
        "seed": args.seed,
        "repair_validation_mse": repair_mse,
        "anchor_validation_mse": anchor_mse,
        "anchor_validation_max_abs": anchor_max_abs,
        "focus_anchor_validation_mse": focus_anchor_mse,
        "focus_anchor_validation_max_abs": focus_anchor_max_abs,
        "state_gate_hidden_dims": list(hidden_dims),
        "state_gate_threshold": threshold,
        "state_gate_anchor_false_positive_rate": measured_false_positive_rate,
        "state_gate_heldout_repair_recall": repair_recall,
        "synthetic_critic_weight": 0.0,
        "synthetic_dynamics_eligible": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    report_path = args.out.with_suffix(".json")
    report = {
        **output["counterfactual_repair_distillation"],
        "output_checkpoint": str(args.out.resolve()),
        "output_checkpoint_sha256": sha256(args.out),
        "history": history,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "checkpoint_sha256": report["output_checkpoint_sha256"],
        "repair_validation_mse": repair_mse,
        "anchor_validation_mse": anchor_mse,
        "anchor_validation_max_abs": anchor_max_abs,
        "focus_anchor_validation_mse": focus_anchor_mse,
        "focus_anchor_validation_max_abs": focus_anchor_max_abs,
        "state_gate_anchor_false_positive_rate": measured_false_positive_rate,
        "state_gate_heldout_repair_recall": repair_recall,
        "report_sha256": sha256(report_path),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
