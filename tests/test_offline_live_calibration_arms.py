import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_vq2_offline_live_calibration import gate4_outcomes


class Gate4OutcomeArmTests(unittest.TestCase):
    def test_interleaved_arms_are_counted_independently(self):
        rows = [
            {
                "schedule_arm": "protected_champion",
                "timing_healthy": True,
                "crossing_offsets": [{"gate": 4, "step": 299}],
                "multigate": True,
            },
            {
                "schedule_arm": "candidate",
                "timing_healthy": True,
                "crossing_offsets": [],
                "multigate": True,
            },
            {
                "schedule_arm": "candidate",
                "timing_healthy": True,
                "crossing_offsets": [{"gate": 4, "step": 260}],
                "multigate": True,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "episodes.jsonl"
            log.write_text("\n".join(json.dumps(row) for row in rows))
            champion = gate4_outcomes(
                log, schedule_arm="protected_champion"
            )
            candidate = gate4_outcomes(log, schedule_arm="candidate")
        self.assertEqual(champion["attempts"], 1)
        self.assertEqual(champion["successes"], 1)
        self.assertEqual(candidate["attempts"], 2)
        self.assertEqual(candidate["successes"], 1)
        self.assertAlmostEqual(candidate["times"][0], 261 / 30.0)

    def test_legacy_rows_default_to_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "episodes.jsonl"
            log.write_text(json.dumps({
                "timing_healthy": True,
                "crossing_offsets": [{"gate": 4, "step": 300}],
                "multigate": False,
            }))
            candidate = gate4_outcomes(log, schedule_arm="candidate")
            champion = gate4_outcomes(
                log, schedule_arm="protected_champion"
            )
        self.assertEqual(candidate["attempts"], 1)
        self.assertEqual(champion["attempts"], 0)


if __name__ == "__main__":
    unittest.main()
