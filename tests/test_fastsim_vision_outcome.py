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
