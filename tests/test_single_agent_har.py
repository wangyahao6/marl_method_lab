"""Focused checks for offline MinAtar historical action metrics."""

import unittest

import numpy as np

from scripts.analyze_minatar_har import action_metrics, summarize_metrics


class HistoricalActionMetricsTest(unittest.TestCase):
    def test_old_model_values_both_actions(self):
        old_q = np.array([[3., 1., 0.], [0., 4., 2.], [1., 1., 0.]], dtype=np.float32)
        current_q = np.array([[0., 100., 0.], [0., 100., 0.], [0., 2., 0.]], dtype=np.float32)
        values = action_metrics(old_q, current_q)
        np.testing.assert_array_equal(values["old_actions"], [0, 1, 0])
        np.testing.assert_array_equal(values["current_actions"], [1, 1, 1])
        np.testing.assert_array_equal(values["old_q_reference_action"], [3., 4., 1.])
        np.testing.assert_array_equal(values["old_q_current_action"], [1., 4., 1.])
        np.testing.assert_array_equal(values["old_q_regret"], [2., 0., 0.])
        summary = summarize_metrics("DQN-FT", "SpaceInvaders", 0, 1, values, "states.npz")
        self.assertAlmostEqual(summary["har"], 1 / 3)
        self.assertAlmostEqual(summary["adr"], 2 / 3)
        self.assertAlmostEqual(summary["mean_old_q_regret"], 2 / 3)
        self.assertEqual(summary["median_old_q_regret"], 0.)


if __name__ == "__main__":
    unittest.main()
