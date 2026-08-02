"""Train and audit the hybrid residual ensemble for the gates 0-4 POC."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def tensors(payload: np.lib.npyio.NpzFile, device: str) -> dict[str, torch.Tensor]:
    result = {}
    for key in (
        "position", "velocity", "rotation", "rates", "previous_action",
        "next_position", "next_velocity", "next_rotation", "next_rates",
        "action", "done", "episode", "step", "gate_index", "confidence",
    ):
        value = np.asarray(payload[key])
        dtype = torch.long if key in ("episode", "step", "gate_index") \
            else torch.float32
        result[key] = torch.as_tensor(value, dtype=dtype, device=device)
    result["sample_weight"] = torch.as_tensor(
        np.asarray(
            payload["sample_weight"]
            if "sample_weight" in payload.files
            else np.ones(len(payload["action"]), np.float32)
        ),
        dtype=torch.float32,
        device=device,
    )
    for key in ("session", "outcome", "stratum"):
        if key in payload.files:
            result[key] = torch.as_tensor(
                np.asarray(payload[key]), dtype=torch.long, device=device
            )
    return result


def course_context(
    position: torch.Tensor,
    rotation: torch.Tensor,
    gate_index: torch.Tensor,
    confidence: torch.Tensor,
    gate_positions: torch.Tensor,
) -> torch.Tensor:
    gate = gate_positions[torch.clamp(gate_index, 0, len(gate_positions) - 1)]
    relative_body = torch.einsum(
        "nij,nj->ni", rotation.transpose(1, 2), gate - position
    ) / 10.0
    return torch.cat([relative_body, confidence], dim=1)


@torch.no_grad()
def features_targets(
    data: dict[str, torch.Tensor],
    model: SurrogateModel,
    gate_positions: torch.Tensor,
) \
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p1, v1, r1, w1 = analytic_step(
        data["position"], data["velocity"], data["rotation"], data["rates"],
        data["action"], model,
    )
    features = residual_features(
        data["velocity"], data["rotation"], data["rates"], data["action"],
        data["previous_action"],
    )
    dp = torch.einsum(
        "nij,nj->ni", data["rotation"].transpose(1, 2),
        data["next_position"] - p1,
    )
    dv = torch.einsum(
        "nij,nj->ni", data["rotation"].transpose(1, 2),
        data["next_velocity"] - v1,
    )
    dr = rotation_log(r1.transpose(1, 2) @ data["next_rotation"])
    dw = data["next_rates"] - w1
    # Keep the learned residual physical and course-independent.  Position is
    # already the integral of velocity; learning decoded-EKF position jumps as
    # fictitious per-gate forces made older 22-D models look accurate offline
    # while failing when the estimator noise pattern changed live.
    target = torch.cat([dv, dr, dw], dim=1)
    keep = (
        (data["done"] == 0)
        & (torch.linalg.norm(dp, dim=1) < 0.75)
        & (torch.linalg.norm(dv, dim=1) < 1.5)
        & (torch.linalg.norm(dr, dim=1) < 0.35)
        & (torch.linalg.norm(dw, dim=1) < 3.0)
        & torch.isfinite(target).all(1)
    )
    return features, target, keep


def train_member(
    member: torch.nn.Module,
    ensemble: ResidualEnsemble,
    x: torch.Tensor,
    y: torch.Tensor,
    episodes: torch.Tensor,
    sample_weight: torch.Tensor,
    xv: torch.Tensor,
    yv: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    generator = torch.Generator(device=x.device).manual_seed(seed)
    unique = torch.unique(episodes)
    sampled = unique[torch.randint(
        len(unique), (len(unique),), generator=generator, device=x.device
    )]
    # Episode bootstrap preserves temporal dependence and gives meaningful
    # epistemic disagreement between members.
    selected = torch.isin(episodes, sampled)
    indices = torch.nonzero(selected).squeeze(1)
    selected_weight = torch.clamp(sample_weight[indices], min=1e-4)
    xn = (x - ensemble.x_mean) / ensemble.x_std
    yn = (y - ensemble.y_mean) / ensemble.y_std
    xvn = (xv - ensemble.x_mean) / ensemble.x_std
    yvn = (yv - ensemble.y_mean) / ensemble.y_std
    optimizer = torch.optim.AdamW(member.parameters(), lr=lr, weight_decay=1e-5)
    best = None
    history = []
    patience = 12
    stale = 0
    for epoch in range(epochs):
        permutation = indices[torch.multinomial(
            selected_weight,
            len(indices),
            replacement=True,
            generator=generator,
        )]
        member.train()
        losses = []
        for start in range(0, len(permutation), batch_size):
            row = permutation[start:start + batch_size]
            mean, log_std = member(xn[row])
            inverse_variance = torch.exp(-2.0 * log_std)
            nll = 0.5 * (yn[row] - mean).square() * inverse_variance + log_std
            # A small mean-fit term prevents uncertainty inflation from hiding
            # a poor dynamics mean.
            loss = nll.mean() + 0.05 * (yn[row] - mean).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), 10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        member.eval()
        with torch.no_grad():
            vm, vl = member(xvn)
            val_mse = float((vm - yvn).square().mean())
            val_nll = float((
                0.5 * (yvn - vm).square() * torch.exp(-2.0 * vl) + vl
            ).mean())
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "validation_mse": val_mse, "validation_nll": val_nll,
        })
        score = val_mse
        if best is None or score < best[0] - 1e-5:
            best = (score, copy.deepcopy(member.state_dict()), epoch)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best is not None
    member.load_state_dict(best[1])
    return {
        "best_epoch": best[2],
        "validation_normalized_mse": best[0],
        "epochs_run": len(history),
        "bootstrap_rows": int(len(indices)),
        "history": history,
    }


def rollout_finetune_member(
    member: torch.nn.Module,
    ensemble: ResidualEnsemble,
    data: dict[str, torch.Tensor],
    model: SurrogateModel,
    gate_positions: torch.Tensor,
    transition_valid: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    horizon: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    max_starts: int,
) -> list[dict]:
    """Fine-tune the residual mean on differentiable multi-step rollouts.

    One-step likelihood can look excellent while a 0.05 m/s directional bias
    compounds into a gate-panel miss.  This loss optimizes the quantity the
    planner actually consumes.  Huber losses keep EKF relocalization jumps
    from being learned as fictitious forces.
    """
    if epochs <= 0:
        return []
    episode, step, done = data["episode"], data["step"], data["done"]
    valid = torch.ones(
        len(step) - horizon, dtype=torch.bool, device=step.device
    )
    for offset in range(1, horizon + 1):
        valid &= (
            (episode[:len(valid)] == episode[offset:offset + len(valid)])
            & (step[offset:offset + len(valid)]
               == step[:len(valid)] + offset)
            & (done[offset - 1:offset - 1 + len(valid)] == 0)
            & transition_valid[offset - 1:offset - 1 + len(valid)]
        )
    starts = torch.nonzero(valid).squeeze(1)
    generator = torch.Generator(device=starts.device).manual_seed(seed)
    if max_starts > 0 and len(starts) > max_starts:
        starts = starts[torch.randperm(
            len(starts), generator=generator, device=starts.device
        )[:max_starts]]
    optimizer = torch.optim.AdamW(member.parameters(), lr=lr, weight_decay=1e-6)
    history = []
    member.train()
    for epoch in range(epochs):
        start_weight = torch.clamp(sample_weight[starts], min=1e-4)
        permutation = starts[torch.multinomial(
            start_weight, len(starts), replacement=True, generator=generator
        )]
        losses = []
        for begin in range(0, len(permutation), batch_size):
            row = permutation[begin:begin + batch_size]
            p = data["position"][row]
            v = data["velocity"][row]
            rotation = data["rotation"][row]
            rates = data["rates"][row]
            previous = data["previous_action"][row]
            loss = torch.zeros((), device=row.device)
            for offset in range(horizon):
                action = data["action"][row + offset]
                features = residual_features(
                    v, rotation, rates, action, previous,
                )
                normalized = (features - ensemble.x_mean) / ensemble.x_std
                mean, _ = member(normalized)
                correction = mean * ensemble.y_std + ensemble.y_mean
                nominal_p, nominal_v, nominal_rotation, nominal_rates = analytic_step(
                    p, v, rotation, rates, action, model,
                )
                p = nominal_p
                v = nominal_v + torch.einsum(
                    "nij,nj->ni", rotation, correction[:, :3]
                )
                rotation = nominal_rotation @ rotation_exp(
                    correction[:, 3:6]
                )
                rates = nominal_rates + correction[:, 6:9]
                target = row + offset + 1
                # Scale each state group by a materially meaningful error.
                # Rotation-matrix loss avoids the singular derivative of
                # acos/log maps at an almost-perfect one-step prediction.
                loss = loss + (
                    0.50 * F.smooth_l1_loss(
                        (p - data["position"][target]) / 0.25,
                        torch.zeros_like(p), beta=1.0,
                    )
                    + F.smooth_l1_loss(
                        (v - data["velocity"][target]) / 0.50,
                        torch.zeros_like(v), beta=1.0,
                    )
                    + 0.50 * F.smooth_l1_loss(
                        (rotation - data["rotation"][target]) / 0.10,
                        torch.zeros_like(rotation), beta=1.0,
                    )
                    + 0.25 * F.smooth_l1_loss(
                        (rates - data["rates"][target]) / 0.50,
                        torch.zeros_like(rates), beta=1.0,
                    )
                ) / horizon
                previous = action
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(member.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        history.append(row)
        print(f"  rollout epoch {epoch + 1}/{epochs}: {row['loss']:.6f}",
              flush=True)
    member.eval()
    return history


@torch.no_grad()
def one_step_report(
    ensemble: ResidualEnsemble,
    data: dict[str, torch.Tensor],
    model: SurrogateModel,
    gate_positions: torch.Tensor,
) -> dict:
    base_p, base_v, base_r, base_w = analytic_step(
        data["position"], data["velocity"], data["rotation"], data["rates"],
        data["action"], model,
    )
    features = residual_features(
        data["velocity"], data["rotation"], data["rates"], data["action"],
        data["previous_action"],
    )
    means, _ = ensemble(features)
    correction = means.mean(0)
    corrected_p = base_p
    corrected_v = base_v + torch.einsum(
        "nij,nj->ni", data["rotation"], correction[:, :3]
    )
    corrected_r = base_r @ rotation_exp(correction[:, 3:6])
    corrected_w = base_w + correction[:, 6:9]

    def stats(value: torch.Tensor) -> dict:
        value = value.detach().cpu().numpy()
        return {
            "median": float(np.median(value)),
            "p90": float(np.quantile(value, 0.9)),
            "rmse": float(np.sqrt(np.mean(value ** 2))),
        }
    base_att = torch.rad2deg(torch.linalg.norm(rotation_log(
        base_r.transpose(1, 2) @ data["next_rotation"]
    ), dim=1))
    corrected_att = torch.rad2deg(torch.linalg.norm(rotation_log(
        corrected_r.transpose(1, 2) @ data["next_rotation"]
    ), dim=1))
    return {
        "base_velocity_mps": stats(torch.linalg.norm(
            base_v - data["next_velocity"], dim=1
        )),
        "corrected_velocity_mps": stats(torch.linalg.norm(
            corrected_v - data["next_velocity"], dim=1
        )),
        "base_attitude_deg": stats(base_att),
        "corrected_attitude_deg": stats(corrected_att),
        "base_rate_radps": stats(torch.linalg.norm(
            base_w - data["next_rates"], dim=1
        )),
        "corrected_rate_radps": stats(torch.linalg.norm(
            corrected_w - data["next_rates"], dim=1
        )),
        "position_integrator_m": stats(torch.linalg.norm(
            base_p - data["next_position"], dim=1
        )),
        "corrected_position_m": stats(torch.linalg.norm(
            corrected_p - data["next_position"], dim=1
        )),
    }


@torch.no_grad()
def rollout_report(
    ensemble: ResidualEnsemble,
    data: dict[str, torch.Tensor],
    model: SurrogateModel,
    gate_positions: torch.Tensor,
    transition_valid: torch.Tensor | None = None,
    horizons: tuple[int, ...] = (4, 8, 16, 32),
    max_starts: int = 0,
) -> dict:
    episode = data["episode"]
    step = data["step"]
    result = {}
    rng = torch.Generator(device=episode.device).manual_seed(8172)
    for horizon in horizons:
        valid = torch.ones(len(step) - horizon, dtype=torch.bool, device=step.device)
        for offset in range(1, horizon + 1):
            valid &= (
                (episode[:len(valid)] == episode[offset:offset + len(valid)])
                & (step[offset:offset + len(valid)] == step[:len(valid)] + offset)
                & (data["done"][offset - 1:offset - 1 + len(valid)] == 0)
            )
            if transition_valid is not None:
                valid &= transition_valid[
                    offset - 1:offset - 1 + len(valid)
                ]
        starts = torch.nonzero(valid).squeeze(1)
        if max_starts > 0 and len(starts) > max_starts:
            starts = starts[torch.randperm(
                len(starts), generator=rng, device=starts.device
            )[:max_starts]]
        corr_p = data["position"][starts].clone()
        corr_v = data["velocity"][starts].clone()
        corr_r = data["rotation"][starts].clone()
        corr_w = data["rates"][starts].clone()
        ana_p, ana_v, ana_r, ana_w = (
            corr_p.clone(), corr_v.clone(), corr_r.clone(), corr_w.clone()
        )
        previous = data["previous_action"][starts].clone()
        disagreement = torch.zeros(len(starts), device=starts.device)
        for offset in range(horizon):
            action = data["action"][starts + offset]
            features = residual_features(
                corr_v, corr_r, corr_w, action, previous,
            )
            means, _ = ensemble(features)
            mean = means.mean(0)
            disagreement += means.std(0).square().mean(1).sqrt()
            nominal_p, nominal_v, nominal_r, nominal_w = analytic_step(
                corr_p, corr_v, corr_r, corr_w, action, model,
            )
            corr_p = nominal_p
            corr_v = nominal_v + torch.einsum(
                "nij,nj->ni", corr_r, mean[:, :3]
            )
            corr_r = nominal_r @ rotation_exp(mean[:, 3:6])
            corr_w = nominal_w + mean[:, 6:9]
            # Independent analytic-only rollout for the same starts/actions.
            ana_p, ana_v, ana_r, ana_w = analytic_step(
                ana_p, ana_v, ana_r, ana_w, action, model,
            )
            previous = action
        target = starts + horizon
        pos_error = torch.linalg.norm(
            corr_p - data["position"][target], dim=1
        )
        vel_error = torch.linalg.norm(
            corr_v - data["velocity"][target], dim=1
        )
        att_error = torch.rad2deg(torch.linalg.norm(rotation_log(
            corr_r.transpose(1, 2) @ data["rotation"][target]
        ), dim=1))
        error = pos_error + 0.25 * vel_error + 0.02 * att_error
        base_pos_error = torch.linalg.norm(
            ana_p - data["position"][target], dim=1
        )
        base_vel_error = torch.linalg.norm(
            ana_v - data["velocity"][target], dim=1
        )
        base_att_error = torch.rad2deg(torch.linalg.norm(rotation_log(
            ana_r.transpose(1, 2) @ data["rotation"][target]
        ), dim=1))
        # Spearman without a scipy dependency in the training process.
        rank_e = torch.argsort(torch.argsort(error))
        rank_u = torch.argsort(torch.argsort(disagreement))
        corr = torch.corrcoef(torch.stack([
            rank_e.float(), rank_u.float()
        ]))[0, 1]
        def s(x: torch.Tensor) -> dict:
            a = x.cpu().numpy()
            return {
                "median": float(np.median(a)),
                "p90": float(np.quantile(a, 0.9)),
            }
        result[str(horizon)] = {
            "seconds": horizon / 30.0,
            "starts": int(len(starts)),
            "position_m": s(pos_error),
            "velocity_mps": s(vel_error),
            "attitude_deg": s(att_error),
            "analytic_position_m": s(base_pos_error),
            "analytic_velocity_mps": s(base_vel_error),
            "analytic_attitude_deg": s(base_att_error),
            "uncertainty_error_spearman": float(corr),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--map", type=Path,
        help="Override the map path stored in a dataset built on another host.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed", type=int, default=20260801,
        help="base seed for episode bootstraps and rollout-window sampling",
    )
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--rollout-finetune-epochs", type=int, default=0)
    parser.add_argument("--rollout-horizon", type=int, default=12)
    parser.add_argument(
        "--rollout-horizons",
        help="comma-separated horizons; overrides --rollout-horizon",
    )
    parser.add_argument("--rollout-lr", type=float, default=8e-5)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument(
        "--rollout-max-starts", type=int, default=32768,
        help="maximum training windows sampled per horizon and epoch",
    )
    parser.add_argument(
        "--audit-dataset", type=Path,
        help="optional untouched counterexample split reported but never trained",
    )
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="resume from an incremental checkpoint written after a member",
    )
    parser.add_argument(
        "--init-ensemble", type=Path,
        help=(
            "initialize every member (including normalization buffers) from "
            "an existing completed ensemble, then fine-tune all members on "
            "this dataset"
        ),
    )
    parser.add_argument(
        "--skip-one-step-finetune",
        action="store_true",
        help=(
            "With --init-ensemble, preserve the fitted one-step model and "
            "run only the requested differentiable rollout fine-tuning."
        ),
    )
    args = parser.parse_args()
    if args.resume_checkpoint is not None and args.init_ensemble is not None:
        parser.error("--resume-checkpoint and --init-ensemble are mutually exclusive")
    if args.skip_one_step_finetune and args.init_ensemble is None:
        parser.error("--skip-one-step-finetune requires --init-ensemble")
    device = args.device
    base = SurrogateModel.load(args.base_model)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    gate_map = args.map or Path(manifest["map"])
    gates = json.loads(gate_map.read_text())["gates"]
    gate_positions = torch.as_tensor(
        [gate["pos"] for gate in gates], dtype=torch.float32, device=device
    )
    train = tensors(np.load(args.dataset / "train.npz"), device)
    validation = tensors(np.load(args.dataset / "validation.npz"), device)
    test = tensors(np.load(args.dataset / "test.npz"), device)
    x, y, keep = features_targets(train, base, gate_positions)
    xv, yv, keepv = features_targets(validation, base, gate_positions)
    _, _, keept = features_targets(test, base, gate_positions)
    x, y = x[keep], y[keep]
    episodes = train["episode"][keep]
    sample_weight = train["sample_weight"][keep]
    xv, yv = xv[keepv], yv[keepv]
    ensemble = ResidualEnsemble(
        members=args.members, input_dim=int(x.shape[1]), output_dim=int(y.shape[1])
    ).to(device)
    ensemble.set_normalization(x, y)
    training = []
    start_member = 0
    if args.init_ensemble is not None:
        initialized, init_metadata = ResidualEnsemble.load(
            args.init_ensemble, device
        )
        if len(initialized.members) != args.members:
            parser.error("initial ensemble member count does not match --members")
        ensemble.load_state_dict(initialized.state_dict())
        print(
            f"initialized {args.members} members from {args.init_ensemble} "
            f"(dataset={init_metadata.get('dataset')})",
            flush=True,
        )
    elif args.resume_checkpoint is not None:
        resumed, resume_metadata = ResidualEnsemble.load(
            args.resume_checkpoint, device
        )
        if len(resumed.members) != args.members:
            parser.error("resume checkpoint member count does not match --members")
        ensemble.load_state_dict(resumed.state_dict())
        start_member = int(resume_metadata.get("completed_members", 0))
        training = list(resume_metadata.get("training", []))
        print(f"resuming after {start_member}/{args.members} members", flush=True)
    partial_path = args.out.with_suffix(".partial.pt")
    for index in range(start_member, len(ensemble.members)):
        member = ensemble.members[index]
        print(f"training ensemble member {index + 1}/{args.members}", flush=True)
        if args.skip_one_step_finetune:
            report = {
                "initialized_without_one_step_finetune": True,
                "source_ensemble": str(args.init_ensemble.resolve()),
            }
        else:
            report = train_member(
                member, ensemble, x, y, episodes, sample_weight, xv, yv,
                seed=args.seed + index * 101,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
        training.append(report)
        printable = dict(report)
        if "history" in report:
            printable["history"] = f"{len(report['history'])} rows"
        print(printable, flush=True)
        rollout_horizons = (
            tuple(int(value) for value in args.rollout_horizons.split(","))
            if args.rollout_horizons else (args.rollout_horizon,)
        )
        rollout_history = {}
        for horizon in rollout_horizons:
            print(f"  rollout horizon {horizon}", flush=True)
            rollout_history[str(horizon)] = rollout_finetune_member(
                member, ensemble, train, base, gate_positions, keep,
                train["sample_weight"],
                horizon=horizon,
                epochs=args.rollout_finetune_epochs,
                batch_size=args.rollout_batch_size,
                lr=args.rollout_lr,
                seed=args.seed + 1000 + index * 101 + horizon,
                max_starts=args.rollout_max_starts,
            )
        report["rollout_finetune"] = rollout_history
        ensemble.save(partial_path, {
            "completed_members": index + 1,
            "training": training,
            "dataset": str(args.dataset.resolve()),
            "base_model": str(args.base_model.resolve()),
        })
        print(f"  checkpointed {index + 1}/{args.members}: {partial_path}",
              flush=True)
    ensemble.eval()
    report = {
        "dataset": str(args.dataset.resolve()),
        "base_model": str(args.base_model.resolve()),
        "train_rows": int(len(x)),
        "validation_rows": int(len(xv)),
        "training": training,
        "validation_one_step": one_step_report(
            ensemble, validation, base, gate_positions
        ),
        "test_one_step": one_step_report(ensemble, test, base, gate_positions),
        "test_rollouts": rollout_report(
            ensemble, test, base, gate_positions, keept
        ),
    }
    if args.audit_dataset is not None:
        audit_validation = tensors(
            np.load(args.audit_dataset / "validation.npz"), device
        )
        audit_test = tensors(
            np.load(args.audit_dataset / "test.npz"), device
        )
        report["counterexample_audit"] = {
            "dataset": str(args.audit_dataset.resolve()),
            "validation_one_step": one_step_report(
                ensemble, audit_validation, base, gate_positions
            ),
            "test_one_step": one_step_report(
                ensemble, audit_test, base, gate_positions
            ),
            "test_rollouts": rollout_report(
                ensemble, audit_test, base, gate_positions,
                features_targets(audit_test, base, gate_positions)[2],
                max_starts=512,
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save(args.out, report)
    report_path = args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "validation_one_step": report["validation_one_step"],
        "test_one_step": report["test_one_step"],
        "test_rollouts": report["test_rollouts"],
    }, indent=2))
    print(f"wrote {args.out} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
