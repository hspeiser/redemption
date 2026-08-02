"""Deployable state gating for counterfactual repair actors.

Counterfactual repairs are deliberately local.  A distilled actor may fit the
repair rows while still moving healthy trajectories elsewhere in the same
gate segment.  This module keeps the protected actor exact outside a learned
failure-state corridor and selects the repaired actor only inside it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn


COUNTERFACTUAL_CHECKPOINT_KEYS = (
    "counterfactual_protected_actor",
    "counterfactual_repair_gate",
)


def preserved_counterfactual_payload(payload: dict) -> dict:
    """Deep-copy the immutable repair selector fields for the next checkpoint.

    The current actor is saved through the normal ``actor`` field.  These two
    fields are the protected fallback and the classifier that decides when the
    repaired actor is allowed to run; dropping either on save/resume silently
    turns a local repair into an ordinary global actor.
    """
    present = [key in payload for key in COUNTERFACTUAL_CHECKPOINT_KEYS]
    if not any(present):
        return {}
    if not all(present):
        missing = [
            key for key in COUNTERFACTUAL_CHECKPOINT_KEYS
            if key not in payload
        ]
        raise ValueError(
            "incomplete counterfactual checkpoint metadata; missing "
            + ", ".join(missing)
        )
    return {
        key: copy.deepcopy(payload[key])
        for key in COUNTERFACTUAL_CHECKPOINT_KEYS
    }


def build_state_gate(
    observation_dim: int,
    hidden_dims: Sequence[int],
) -> nn.Sequential:
    """Build the tiny classifier whose input is a normalized observation."""
    layers: list[nn.Module] = []
    width = int(observation_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(width, int(hidden)), nn.ReLU()])
        width = int(hidden)
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class CounterfactualStateGatedActor(nn.Module):
    """Select a repair actor only for states certified by a gate classifier."""

    def __init__(
        self,
        *,
        protected_actor: nn.Module,
        repair_actor: nn.Module,
        classifier: nn.Module,
        threshold: float,
    ) -> None:
        super().__init__()
        self.protected_actor = protected_actor
        self.repair_actor = repair_actor
        self.classifier = classifier
        self.threshold = float(threshold)

    @torch.inference_mode()
    def gate_score(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(observation)).squeeze(-1)

    @torch.inference_mode()
    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        protected = self.protected_actor.deterministic(observation)
        repaired = self.repair_actor.deterministic(observation)
        active = self.gate_score(observation) >= self.threshold
        return torch.where(active[:, None], repaired, protected)


def load_state_gated_actor(
    payload: dict,
    repair_actor: nn.Module,
    *,
    observation_dim: int,
    actor_factory,
    device: str | torch.device,
) -> CounterfactualStateGatedActor | None:
    """Construct the optional immutable protected/repair actor selector."""
    metadata = payload.get("counterfactual_repair_gate")
    protected_state = payload.get("counterfactual_protected_actor")
    if metadata is None or protected_state is None:
        return None
    protected = actor_factory().to(device)
    protected.load_state_dict(protected_state)
    protected.eval()
    classifier = build_state_gate(
        observation_dim,
        metadata["hidden_dims"],
    ).to(device)
    classifier.load_state_dict(metadata["state_dict"])
    classifier.eval()
    repair_actor.eval()
    return CounterfactualStateGatedActor(
        protected_actor=protected,
        repair_actor=repair_actor,
        classifier=classifier,
        threshold=float(metadata["threshold"]),
    ).eval()
