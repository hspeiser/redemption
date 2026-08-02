import unittest

import numpy as np

from scripts.optimize_vq2_g0g4_worldmodel import (
    promotion_eligible,
    score,
    unpack,
)


def report(
    *, finish_rate: float, tier_success_rate: float, median_s: float,
    tier_soft_rate: float | None = None, clearance_m: float = 0.25,
) -> dict:
    return {
        "finish_rate": finish_rate,
        "tier_success_rate": tier_success_rate,
        "tier_soft_rate": (
            tier_success_rate if tier_soft_rate is None else tier_soft_rate
        ),
        "median_s": median_s,
        "clearance_p10_m": clearance_m,
        "support_z_p90": 1.0,
        "disagreement_mean": 0.01,
    }


class TimedTierScoreTests(unittest.TestCase):
    def test_promotion_requires_every_hard_floor_on_every_model(self) -> None:
        qualified = report(
            finish_rate=0.95, tier_success_rate=0.92, median_s=8.5,
            clearance_m=0.22,
        )
        low_clearance = report(
            finish_rate=0.98, tier_success_rate=0.96, median_s=8.4,
            clearance_m=0.17,
        )
        self.assertTrue(promotion_eligible(
            [qualified], reliability_floor=0.90,
            clearance_floor=0.18, tier_target=8.7,
        ))
        self.assertFalse(promotion_eligible(
            [qualified, low_clearance], reliability_floor=0.90,
            clearance_floor=0.18, tier_target=8.7,
        ))

    def test_measured_clearance_floor_rejects_model_error_exposure(self) -> None:
        exposed = report(
            finish_rate=0.99, tier_success_rate=0.99, median_s=8.4,
            clearance_m=0.15,
        )
        buffered = report(
            finish_rate=0.99, tier_success_rate=0.99, median_s=8.5,
            clearance_m=0.32,
        )

        self.assertGreater(
            score(buffered, tier_target=8.7, clearance_floor=0.28),
            score(exposed, tier_target=8.7, clearance_floor=0.28),
        )

    def test_lead_limit_clips_ood_action_lookahead(self) -> None:
        theta = np.zeros((1, 35), np.float64)
        theta[:, :5] = [1, 4, 13, 2, 9]
        theta[:, 5:10] = 1.0
        theta[:, 10:15] = 1.0
        theta[:, 30:35] = 1.0

        leads = unpack(theta, lead_limit=8)[0]

        np.testing.assert_array_equal(leads[0], [1, 4, 8, 2, 8])

    def test_timed_floor_uses_finish_and_time_joint_outcome(self) -> None:
        lucky_fast_median = report(
            finish_rate=0.99, tier_success_rate=0.50, median_s=8.0,
        )
        qualified = report(
            finish_rate=0.98, tier_success_rate=0.98, median_s=8.6,
        )

        self.assertGreater(
            score(
                qualified, reliability_floor=0.97,
                tier_target=8.7, tier_bonus=300.0,
            ),
            score(
                lucky_fast_median, reliability_floor=0.97,
                tier_target=8.7, tier_bonus=300.0,
            ),
        )

    def test_untimed_score_still_uses_finish_rate(self) -> None:
        reliable = report(
            finish_rate=0.98, tier_success_rate=0.10, median_s=9.5,
        )
        unreliable = report(
            finish_rate=0.50, tier_success_rate=0.50, median_s=9.0,
        )

        self.assertGreater(score(reliable), score(unreliable))

    def test_soft_deadline_breaks_equal_strict_rate_ties(self) -> None:
        near = report(
            finish_rate=0.95, tier_success_rate=0.20,
            tier_soft_rate=0.60, median_s=8.75,
        )
        far = report(
            finish_rate=0.95, tier_success_rate=0.20,
            tier_soft_rate=0.25, median_s=8.75,
        )
        self.assertGreater(
            score(near, tier_target=8.7),
            score(far, tier_target=8.7),
        )


if __name__ == "__main__":
    unittest.main()
