import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from aigp.fastsim.live_teacher import LiveTeacherController
from scripts.fastsim_train_ppo import load_live_teacher_config


def synthetic_demo(path: Path) -> None:
    rows = 34
    gate = np.repeat(np.arange(17), 2)
    observation = np.zeros((rows, 51), np.float32)
    observation[:, 21] = 1.0
    observation[:, 25] = 1.0
    observation[np.arange(rows), 34 + gate] = 1.0
    position = np.stack([
        gate.astype(np.float32) * 10.0,
        np.zeros(rows, np.float32),
        np.zeros(rows, np.float32),
    ], axis=1)
    # Every gate is one metre ahead in the demonstrated body/world frame.
    observation[:, 0] = 0.1
    velocity = np.zeros((rows, 3), np.float32)
    action = np.zeros((rows, 4), np.float32)
    np.savez(
        path,
        observation=observation,
        action=action,
        position=position,
        velocity=velocity,
        wall=np.arange(rows, dtype=np.float64) / 30.0,
        gate_index=gate,
    )


class LiveTeacherParityTest(unittest.TestCase):
    def test_config_loader_converts_local_reference_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(json.dumps({"args": {
                "reference_action_leads": "",
                "reference_lateral_offsets": "1:0.25",
                "reference_vertical_offsets": "1:-0.10",
            }}))
            gate_map = root / "map.json"
            gate_map.write_text(json.dumps({"gates": [
                {
                    "pos": [float(index), 0.0, 0.0],
                    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
                for index in range(17)
            ]}))
            arrays, _fixed = load_live_teacher_config(
                config, 2, map_path=gate_map
            )
            self.assertEqual(
                arrays["reference_gate_offsets_world"].shape,
                (2, 5, 3),
            )
            np.testing.assert_allclose(
                arrays["reference_gate_offsets_world"][:, 1],
                [[0.25, 0.0, -0.10], [0.25, 0.0, -0.10]],
            )

    def test_config_loader_preserves_live_gate_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"args": {
                "reference_action_lead": 1,
                "gate4_action_lead": 2,
                "reference_action_leads": "1:-2",
                "reference_rate_scale": 1.1,
                "reference_rate_scales": "2:1.3",
                "reference_mode": "nearest",
                "reference_sequential_speed": 1.0,
                "reference_sequential_speeds": "2:0.95",
                "special_lateral_gate": 3,
                "special_lateral_gain_scale": 1.5,
                "lateral_gain_scales": "2:1.25",
                "lateral_feedback_limits": "2:0.12",
                "longitudinal_position_gain": 0.01,
                "longitudinal_position_gains": "2:0.03",
                "longitudinal_velocity_gain": 0.02,
                "longitudinal_velocity_gains": "2:0.04",
                "lateral_bias_gates": "1,7",
                "lateral_action_bias": -0.02,
                "right_lateral_bias_gates": "2,9",
                "right_lateral_action_bias": 0.03,
                "extra_lateral_biases": "2:-0.025,3:0.01",
                "special_vertical_gate": 2,
                "special_vertical_gain_scale": 1.8,
                "extra_vertical_biases": "2:0.02",
                "vertical_bias_gates": "3,5",
            }}))
            arrays, fixed = load_live_teacher_config(config, 2)
            np.testing.assert_array_equal(
                arrays["action_leads"][0], [1, -2, 1, 1, 3]
            )
            np.testing.assert_allclose(
                fixed["reference_rate_scales"], [1.1, 1.1, 1.3, 1.1, 1.1]
            )
            np.testing.assert_array_equal(
                fixed["reference_sequential_enabled"],
                [False, False, True, False, False],
            )
            self.assertAlmostEqual(
                float(fixed["reference_sequential_speeds"][2]), 0.95
            )
            self.assertAlmostEqual(float(fixed["lateral_gain_scales"][2]), 1.25)
            self.assertAlmostEqual(float(fixed["lateral_gain_scales"][3]), 1.5)
            self.assertAlmostEqual(float(fixed["lateral_feedback_limits"][2]), 0.12)
            self.assertAlmostEqual(float(fixed["extra_lateral_biases"][2]), -0.025)
            self.assertAlmostEqual(float(fixed["vertical_gain_scales"][2]), 1.8)
            self.assertEqual(fixed["lateral_bias_gates"], (1, 7))
            self.assertEqual(fixed["right_lateral_bias_gates"], (2, 9))

            arrays17, fixed17 = load_live_teacher_config(
                config, 1, gate_count=17
            )
            self.assertEqual(arrays17["action_leads"].shape, (1, 17))
            self.assertEqual(len(fixed17["lateral_gain_scales"]), 17)
            self.assertAlmostEqual(
                float(fixed17["extra_lateral_biases"][3]), 0.01
            )

    def test_extra_lateral_bias_and_fractional_cursor_execute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo = Path(directory) / "demo.npz"
            synthetic_demo(demo)
            common = dict(
                device="cpu",
                reference_sequential_enabled=np.array(
                    [False, False, True, False, False]
                ),
                reference_sequential_speeds=np.array(
                    [1.0, 1.0, 0.95, 1.0, 1.0], np.float32
                ),
            )
            plain = LiveTeacherController(demo, 1, **common)
            biased = LiveTeacherController(
                demo, 1,
                extra_lateral_biases=np.array(
                    [0.0, 0.0, 0.05, 0.0, 0.0], np.float32
                ),
                **common,
            )
            target = torch.tensor([2])
            plain.set_target(target)
            biased.set_target(target)
            p = plain.position[plain.gate_start[2]][None]
            v = torch.zeros((1, 3))
            rotation = torch.eye(3)[None]
            plain_action = plain.action(p, v, rotation)
            biased_action = biased.action(p, v, rotation)
            self.assertAlmostEqual(
                float(biased_action[0, 0] - plain_action[0, 0]),
                0.05,
                places=5,
            )
            self.assertAlmostEqual(
                float(biased.reference_cursor[0]),
                float(biased.gate_start[2]) + 0.95,
                places=4,
            )
            biased.action(p, v, rotation)
            biased.action(p, v, rotation)
            self.assertAlmostEqual(
                float(biased.reference_cursor[0]),
                float(biased.gate_end[2]),
                places=4,
            )

    def test_reference_rate_scale_changes_feed_forward_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo = Path(directory) / "demo.npz"
            synthetic_demo(demo)
            payload = dict(np.load(demo, allow_pickle=False))
            payload["action"][:, 0] = 0.20
            np.savez(demo, **payload)
            baseline = LiveTeacherController(demo, 1, device="cpu")
            accelerated = LiveTeacherController(
                demo,
                1,
                device="cpu",
                reference_rate_scales=np.array(
                    [1.0, 1.0, 1.5, 1.0, 1.0], np.float32
                ),
            )
            target = torch.tensor([2])
            baseline.set_target(target)
            accelerated.set_target(target)
            p = baseline.position[baseline.gate_start[2]][None]
            v = torch.zeros((1, 3))
            rotation = torch.eye(3)[None]
            baseline_action = baseline.action(p, v, rotation)
            accelerated_action = accelerated.action(p, v, rotation)
            self.assertAlmostEqual(float(baseline_action[0, 0]), 0.20, places=5)
            self.assertAlmostEqual(float(accelerated_action[0, 0]), 0.30, places=5)

    def test_per_environment_reference_rate_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo = Path(directory) / "demo.npz"
            synthetic_demo(demo)
            payload = dict(np.load(demo, allow_pickle=False))
            payload["action"][:, 0] = 0.20
            np.savez(demo, **payload)
            scales = np.ones((2, 5), np.float32)
            scales[1, 2] = 1.5
            controller = LiveTeacherController(
                demo, 2, device="cpu", reference_rate_scales=scales
            )
            controller.set_target(torch.tensor([2, 2]))
            p = controller.position[controller.gate_start[2]].repeat(2, 1)
            action = controller.action(
                p, torch.zeros((2, 3)), torch.eye(3).repeat(2, 1, 1)
            )
            self.assertAlmostEqual(float(action[0, 0]), 0.20, places=5)
            self.assertAlmostEqual(float(action[1, 0]), 0.30, places=5)

    def test_runtime_map_gate_offset_is_translated_into_demo_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo = Path(directory) / "demo.npz"
            synthetic_demo(demo)
            baseline = LiveTeacherController(demo, 1, device="cpu")
            schedule = baseline.gate_position.cpu().numpy().copy()
            schedule[0, 1] += 0.30
            shifted = LiveTeacherController(
                demo,
                1,
                device="cpu",
                schedule_gate_positions=schedule,
            )
            target = torch.tensor([0])
            baseline.set_target(target)
            shifted.set_target(target)
            # Same physical gate-relative observation: moving the surveyed
            # gate +30 cm and the runtime-frame drone +30 cm must yield the
            # exact same demonstrated-frame controller action and row.
            p_demo = baseline.position[baseline.gate_start[0]][None]
            p_runtime = p_demo + torch.tensor([[0.0, 0.30, 0.0]])
            velocity = torch.zeros((1, 3))
            rotation = torch.eye(3)[None]
            action_baseline = baseline.action(p_demo, velocity, rotation)
            action_shifted = shifted.action(p_runtime, velocity, rotation)
            torch.testing.assert_close(action_shifted, action_baseline)
            self.assertEqual(
                int(shifted.idx[0]), int(baseline.idx[0])
            )

    def test_zero_reference_geometry_offset_preserves_exact_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo = Path(directory) / "demo.npz"
            synthetic_demo(demo)
            baseline = LiveTeacherController(demo, 2, device="cpu")
            zero_offset = LiveTeacherController(
                demo,
                2,
                device="cpu",
                reference_gate_offsets_world=np.zeros(
                    (2, 5, 3), np.float32
                ),
            )
            target = torch.tensor([0, 3])
            baseline.set_target(target)
            zero_offset.set_target(target)
            p = torch.stack([
                baseline.position[baseline.gate_start[0]],
                baseline.position[baseline.gate_start[3]],
            ])
            v = torch.zeros((2, 3))
            rotation = torch.eye(3).repeat(2, 1, 1)
            for _ in range(3):
                action_baseline = baseline.action(p, v, rotation)
                action_offset = zero_offset.action(p, v, rotation)
                torch.testing.assert_close(
                    action_offset, action_baseline, rtol=0.0, atol=0.0
                )
                torch.testing.assert_close(
                    zero_offset.reference_cursor,
                    baseline.reference_cursor,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_reference_geometry_offset_changes_position_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo = Path(directory) / "demo.npz"
            synthetic_demo(demo)
            sequential = np.ones(5, bool)
            baseline = LiveTeacherController(
                demo,
                1,
                device="cpu",
                reference_sequential_enabled=sequential,
            )
            offsets = np.zeros((1, 5, 3), np.float32)
            offsets[0, 0, 1] = 0.20
            shifted = LiveTeacherController(
                demo,
                1,
                device="cpu",
                reference_sequential_enabled=sequential,
                reference_gate_offsets_world=offsets,
            )
            target = torch.tensor([0])
            baseline.set_target(target)
            shifted.set_target(target)
            p = baseline.position[baseline.gate_end[0]][None]
            v = torch.zeros((1, 3))
            rotation = torch.eye(3)[None]
            # First call advances the two-row synthetic cursor to the gate
            # crossing, where the full requested offset applies.
            baseline.action(p, v, rotation)
            shifted.action(p, v, rotation)
            baseline_action = baseline.action(p, v, rotation)
            shifted_action = shifted.action(p, v, rotation)
            self.assertGreater(
                float(shifted_action[0, 0] - baseline_action[0, 0]),
                0.01,
            )

    def test_reference_geometry_offset_shape_is_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo = Path(directory) / "demo.npz"
            synthetic_demo(demo)
            with self.assertRaisesRegex(
                ValueError, "reference_gate_offsets_world"
            ):
                LiveTeacherController(
                    demo,
                    2,
                    device="cpu",
                    reference_gate_offsets_world=np.zeros(
                        (1, 5, 3), np.float32
                    ),
                )


if __name__ == "__main__":
    unittest.main()
