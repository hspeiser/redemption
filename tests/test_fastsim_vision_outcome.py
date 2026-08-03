import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from aigp.fastsim.env import FastEnvConfig, FastVQ2Env
from aigp.fastsim.sysid import SurrogateModel


REPO = Path(__file__).resolve().parents[1]


class VisionOutcomeModelTests(unittest.TestCase):
    def test_audit_random_tape_stays_aligned_after_state_divergence(self):
        class ConstantBackbone:
            def __init__(self, action):
                self.command = torch.tensor(action, dtype=torch.float32)

            def reset_nearest(self, _indices, _position):
                return None

            def action(self, p, _v, _rotation):
                return self.command.to(p.device).repeat(len(p), 1)

        def random_tail(command):
            torch.manual_seed(918273)
            config = FastEnvConfig(
                n_envs=3,
                auto_reset=False,
                spawn_at_rest=True,
                max_episode_s=0.25,
                residual_scale=0.0,
                reloc_events=True,
                fov_vision=True,
                impulse_rate_hz=5.0,
            )
            config.apply_multigate10hz()
            env = FastVQ2Env(
                SurrogateModel.load(
                    REPO / "data" / "fastsim_model_v2.json"
                ),
                REPO / "data" / "vq2_runtime_map_g9g15fix.json",
                config=config,
                device="cpu",
                backbone=ConstantBackbone(command),
            )
            for _ in range(20):
                env.step(torch.zeros((3, 4)))
            return torch.rand(8)

        calm_tail = random_tail([0.0, 0.0, 0.0, -0.5])
        turning_tail = random_tail([1.0, -1.0, 1.0, 0.5])
        torch.testing.assert_close(calm_tail, turning_tail)

    def test_backbone_receives_declared_previous_action_and_target(self):
        class ExtrasBackbone:
            needs_extras = True

            def __init__(self):
                self.previous_action = None
                self.target = None

            def action(self, p, v, rotation, *, prev_action, target):
                self.previous_action = prev_action.clone()
                self.target = target.clone()
                return torch.zeros((len(p), 4), device=p.device)

            def reset_nearest(self, _indices, _position):
                return None

        backbone = ExtrasBackbone()
        config = FastEnvConfig(n_envs=2)
        env = FastVQ2Env(
            SurrogateModel.load(REPO / "data" / "fastsim_model_v2.json"),
            REPO / "data" / "vq2_runtime_map_g9g15fix.json",
            config=config,
            device="cpu",
            backbone=backbone,
        )
        expected_previous = torch.tensor([
            [0.1, 0.2, 0.3, 0.4],
            [-0.1, -0.2, -0.3, -0.4],
        ])
        expected_target = torch.tensor([2, 7])
        env.prev_action.copy_(expected_previous)
        env.target.copy_(expected_target)
        env.step(torch.zeros((2, 4)))
        torch.testing.assert_close(
            backbone.previous_action, expected_previous
        )
        torch.testing.assert_close(backbone.target, expected_target)

    def test_model_probability_and_common_random_stream(self):
        probability = 0.7
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "vision.json"
            model_path.write_text(json.dumps({
                "type": "vq2_vision_fusion_logistic_v1",
                "feature_mean": [0.0] * 21,
                "feature_std": [1.0] * 21,
                "weight": [0.0] * 21,
                "bias": math.log(probability / (1.0 - probability)),
            }))
            config = FastEnvConfig(
                n_envs=6,
                vision_outcome_model=str(model_path),
                common_random_worlds=2,
            )
            env = FastVQ2Env(
                SurrogateModel.load(REPO / "data" / "fastsim_model_v2.json"),
                REPO / "data" / "vq2_runtime_map_g9g15fix.json",
                config=config,
                device="cpu",
            )
            predicted = env._vision_fusion_probability()
            self.assertTrue(torch.allclose(
                predicted,
                torch.full_like(predicted, probability),
                atol=1e-6,
            ))
            torch.manual_seed(7)
            draw = env._vision_uniform()
            self.assertTrue(torch.equal(draw[:2], draw[2:4]))
            self.assertTrue(torch.equal(draw[:2], draw[4:6]))


if __name__ == "__main__":
    unittest.main()
