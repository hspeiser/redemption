import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from aigp.fastsim.branching import (
    calibrated_position_sigma,
    restore_branch_cloud,
)
from aigp.rl.counterfactual_gate import (
    CounterfactualStateGatedActor,
    preserved_counterfactual_payload,
)
from scripts.build_vq2_repair_candidate_config import (
    format_gate_phase_windows,
    gate_phase_windows,
)
from scripts.derive_vq2_teacher_config import _replace_prefix


class ConstantActor(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (len(observation), 4), self.value, device=observation.device
        )


class FakeBranchEnv:
    def __init__(self, n: int = 8) -> None:
        self.cfg = SimpleNamespace(n_envs=n)
        self.device = torch.device("cpu")
        self.p = torch.zeros(n, 3)
        self.v = torch.zeros(n, 3)
        self.q = torch.zeros(n, 4)
        self.q[:, 0] = 1.0
        self.w = torch.zeros(n, 3)
        self.target = torch.zeros(n, dtype=torch.long)
        self.t_ep = torch.zeros(n)
        self.t_gate = torch.zeros(n)
        self.prev_action = torch.zeros(n, 4)
        self.act_buf = torch.zeros(1, n, 4)
        self.noise_pos = torch.zeros(n, 3)
        self.noise_amp = torch.zeros(n, 1)
        self.vis_age = torch.zeros(n)
        self.progress = torch.zeros(n)
        self.best_prog = torch.zeros(n)
        self.t_best = torch.zeros(n)
        self.spawn_flag = torch.zeros(n, dtype=torch.bool)
        self.reloc_next_t = torch.zeros(n)
        self.reloc_end_t = torch.zeros(n)
        self.reloc_rate = torch.zeros(n)
        self.backbone = None

    def _qrotvec(self, value: torch.Tensor, _: float) -> torch.Tensor:
        result = torch.zeros(len(value), 4)
        result[:, 0] = 1.0
        return result

    def _qmul(self, left: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return left.clone()

    def _course_progress(
        self, position: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros(len(position))


class CounterfactualRepairIntegrationTests(unittest.TestCase):
    def test_position_sigma_calibration_uses_measured_floor(self) -> None:
        self.assertAlmostEqual(
            calibrated_position_sigma(0.03, scale=2.0, floor_m=0.15),
            0.15,
        )
        self.assertAlmostEqual(
            calibrated_position_sigma(0.10, scale=2.0, floor_m=0.15),
            0.20,
        )

    def test_branch_cloud_separates_physical_state_from_fixed_belief(self) -> None:
        env = FakeBranchEnv()
        belief = np.asarray([4.0, -2.0, 1.0], np.float32)

        restore_branch_cloud(
            env,
            position=belief,
            velocity=np.zeros(3, np.float32),
            rotation=np.eye(3, dtype=np.float32),
            rates=np.zeros(3, np.float32),
            previous_action=np.zeros(4, np.float32),
            target_gate=3,
            position_sigma_m=0.2,
            landmark_age_s=0.15,
            seed=7,
            velocity_sigma_mps=0.0,
            attitude_sigma_deg=0.0,
            rate_sigma_radps=0.0,
        )

        expected = torch.as_tensor(belief).expand_as(env.p)
        torch.testing.assert_close(env.p + env.noise_pos, expected)
        self.assertGreater(float(env.noise_pos.std()), 0.05)
        torch.testing.assert_close(env.noise_amp, torch.full((8, 1), 0.2))

    def test_state_gate_keeps_protected_fallback_exact(self) -> None:
        classifier = nn.Linear(2, 1)
        with torch.no_grad():
            classifier.weight.copy_(torch.tensor([[10.0, 0.0]]))
            classifier.bias.zero_()
        actor = CounterfactualStateGatedActor(
            protected_actor=ConstantActor(1.0),
            repair_actor=ConstantActor(2.0),
            classifier=classifier,
            threshold=0.5,
        )

        action = actor.deterministic(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))

        torch.testing.assert_close(action[0], torch.ones(4))
        torch.testing.assert_close(action[1], torch.full((4,), 2.0))

    def test_checkpoint_selector_fields_are_preserved_and_detached(self) -> None:
        source = {
            "actor": {"ignored": True},
            "counterfactual_protected_actor": {
                "weight": torch.tensor([1.0])
            },
            "counterfactual_repair_gate": {
                "threshold": 0.75,
                "hidden_dims": [32, 16],
            },
        }

        saved = preserved_counterfactual_payload(source)
        source["counterfactual_repair_gate"]["threshold"] = 0.1
        source["counterfactual_protected_actor"]["weight"][0] = 9.0

        self.assertEqual(saved["counterfactual_repair_gate"]["threshold"], 0.75)
        torch.testing.assert_close(
            saved["counterfactual_protected_actor"]["weight"],
            torch.tensor([1.0]),
        )
        self.assertNotIn("actor", saved)

    def test_partial_checkpoint_selector_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete counterfactual"):
            preserved_counterfactual_payload({
                "counterfactual_repair_gate": {"threshold": 0.5}
            })

    def test_repair_window_replaces_only_its_gate(self) -> None:
        windows = gate_phase_windows("1:0.1:0.9,3:0.2:0.8")
        windows[2] = (0.75, 1.0)

        encoded = format_gate_phase_windows(windows)

        self.assertEqual(
            gate_phase_windows(encoded),
            {1: (0.1, 0.9), 2: (0.75, 1.0), 3: (0.2, 0.8)},
        )

    def test_derived_teacher_prefix_preserves_downstream_values(self) -> None:
        self.assertEqual(
            _replace_prefix("0:1.0,4:0.8,9:0.5", [1.2, 1.3], 2),
            "0:1.2,1:1.3,4:0.8,9:0.5",
        )


if __name__ == "__main__":
    unittest.main()
