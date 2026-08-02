import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aigp.flight import RateController
from scripts.train_vq2_sac_live import VQ2SACLearner


class LiveProbeControllerProfileTests(unittest.TestCase):
    def _bare_learner(self) -> VQ2SACLearner:
        learner = VQ2SACLearner.__new__(VQ2SACLearner)
        for name in learner._CONTROLLER_PROFILE_FIELDS:
            setattr(learner, name, {"candidate": name})
        learner.reference_gate_offsets_world = np.full((17, 3), 7.0)
        learner.trajectory_controller = RateController(thrust_limit=0.52)
        learner.reference_demo_tables = {
            "candidate": {"demo_marker": "candidate"},
            "protected_champion": {"demo_marker": "champion"},
        }
        learner.probe_arm = "candidate"
        return learner

    def test_full_profile_switches_and_restores_every_field(self) -> None:
        learner = self._bare_learner()
        candidate = learner._capture_controller_profile()
        champion = {
            key: (np.full_like(value, 3) if isinstance(value, np.ndarray)
                  else {"champion": key})
            for key, value in candidate.items()
        }
        champion["trajectory_controller_k_att"] = 9.0
        learner.controller_profiles = {
            "candidate": candidate,
            "protected_champion": champion,
        }

        learner.set_probe_arm("protected_champion")
        self.assertEqual(learner.demo_marker, "champion")
        self.assertEqual(learner.trajectory_controller.k_att, 9.0)
        for name in learner._CONTROLLER_PROFILE_FIELDS:
            expected = champion[name]
            actual = getattr(learner, name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                self.assertEqual(actual, expected)

        learner.set_probe_arm("candidate")
        self.assertEqual(learner.demo_marker, "candidate")
        self.assertEqual(learner.trajectory_controller.k_att, 4.0)
        for name in learner._CONTROLLER_PROFILE_FIELDS:
            expected = candidate[name]
            actual = getattr(learner, name)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                self.assertEqual(actual, expected)

    def test_config_profile_parses_action_affecting_overrides(self) -> None:
        learner = self._bare_learner()
        payload = {
            "args": {
                "reference_action_lead": 2,
                "gate4_action_lead": 3,
                "reference_action_leads": "1:-4",
                "reference_thrust_scales": "2:1.25",
                "reference_velocity_scales": "3:1.4",
                "reference_rate_scales": "4:1.6",
                "reference_lateral_offsets": "1:0.2",
                "reference_vertical_offsets": "2:-0.3",
                "lateral_gain_scales": "3:2.0",
                "lateral_feedback_limits": "3:0.31",
                "extra_lateral_biases": "4:-0.07",
                "vertical_bias_gates": "2,5",
                "vertical_action_bias": 0.04,
                "trajectory_blend": 0.2,
                "trajectory_blend_gates": "1,4",
                "gate_center_funnel_gates": "4,10",
                "gate_center_funnel_distance": 3.0,
                "trajectory_kp_scale": 1.5,
                "trajectory_kv_scale": 0.5,
                "trajectory_attitude_gain": 6.0,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload))
            rotation = np.repeat(np.eye(3)[None], 17, axis=0)
            profile = learner._controller_profile_from_config(path, rotation)

        self.assertEqual(profile["reference_action_lead"], 2)
        self.assertEqual(profile["gate4_action_lead"], 3)
        self.assertEqual(profile["reference_action_leads"], {1: -4})
        self.assertEqual(profile["reference_thrust_scales"], {2: 1.25})
        self.assertEqual(profile["reference_velocity_scales"], {3: 1.4})
        self.assertEqual(profile["reference_rate_scales"], {4: 1.6})
        self.assertEqual(profile["lateral_gain_scales"], {3: 2.0})
        self.assertEqual(profile["lateral_feedback_limits"], {3: 0.31})
        self.assertEqual(profile["extra_lateral_biases"], {4: -0.07})
        self.assertEqual(profile["vertical_bias_gates"], frozenset({2, 5}))
        self.assertEqual(profile["trajectory_blend_gates"], frozenset({1, 4}))
        self.assertEqual(
            profile["gate_center_funnel_gates"], frozenset({4, 10})
        )
        np.testing.assert_allclose(
            profile["reference_gate_offsets_world"][1], [0.2, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            profile["reference_gate_offsets_world"][2], [0.0, 0.0, -0.3]
        )
        np.testing.assert_allclose(
            profile["trajectory_controller_kp"], [1.8, 1.8, 3.0]
        )
        np.testing.assert_allclose(
            profile["trajectory_controller_kv"], [1.0, 1.0, 1.4]
        )
        self.assertEqual(profile["trajectory_controller_k_att"], 6.0)

    def test_protected_arm_without_profile_fails_closed(self) -> None:
        learner = self._bare_learner()
        learner.controller_profiles = {
            "candidate": learner._capture_controller_profile()
        }
        with self.assertRaisesRegex(RuntimeError, "champion-config"):
            learner.set_probe_arm("protected_champion")


if __name__ == "__main__":
    unittest.main()
