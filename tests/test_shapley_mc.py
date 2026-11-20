from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from drivers.continual.analysis.shapley_frpr import coalition_rows, exact_shapley
from drivers.continual.analysis.shapley_mc import batched_shapley, fit_transfer, transferred_scores

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_shapley_validation import bootstrap
from validate_shapley_frpr import validate_context


SETTINGS = dict(batch=64, minimum=256, maximum=1024, abs_tol=.002, rel_tol=.10,
                min_top_overlap=2/3, stable_batches=2, stop_early=False)


class ShapleyPermutationTest(unittest.TestCase):
    def test_sampled_matches_exact_with_interaction(self):
        pairs = [(2, 1), (3, 1), (3, 2)]

        def game(mask):
            x, y, z = (bool(mask & (1 << i)) for i in range(3))
            return np.array([2 * x - y + 5 * z + 3 * (x and y), .2 * (x and z)])

        expected, _ = exact_shapley(np.array([game(mask) for mask in range(8)]), 3)
        actual, se, history, audit = batched_shapley(game, pairs, 2, 2026, SETTINGS)
        np.testing.assert_allclose(actual, expected, atol=.09)
        np.testing.assert_allclose(actual.sum(axis=0), game(7) - game(0), atol=1e-12)
        np.testing.assert_array_equal(actual, batched_shapley(game, pairs, 2, 2026, SETTINGS)[0])
        self.assertEqual(audit["permutations"], 1024)
        self.assertGreater(se[0, 0], 0.)
        self.assertTrue(history)

    def test_high_m_visits_only_sampled_prefixes(self):
        pairs = [(i, i + 1) for i in range(38)]
        calls = set()

        def game(mask):
            calls.add(mask)
            return np.array([bin(mask).count("1"), 0.])

        actual, _, _, audit = batched_shapley(game, pairs, 3, 10, dict(SETTINGS, stop_early=True))
        np.testing.assert_array_equal(actual[:, 0], np.ones(38))
        self.assertTrue(audit["converged"])
        self.assertEqual(audit["permutations"], 256)
        self.assertLessEqual(len(calls), 1 + 38 * 256)

    def test_masked_coalitions_equal_full_enumeration(self):
        base = [1., 2., 3.]
        pairs = [(0, 1), (1, 2)]
        outcomes = np.array([[[0., 0.], [1., 1.], [2., 1.]]])
        all_rows = list(coalition_rows(base, [0, 1, 2], pairs, outcomes, 0., 1.))
        small = list(coalition_rows(base, [0, 1, 2], pairs, outcomes, 0., 1., masks=[0, 3]))
        self.assertEqual([row[0] for row in small], [0, 3])
        np.testing.assert_array_equal(all_rows[3][3], small[1][3])

    def test_transfer_uses_only_discovery_and_retains_direction(self):
        a, b, c = (1, 2), (2, 1), (1, 3)
        fit = fit_transfer([(("early", "low"), [a, b], [3., -1.])], 3)
        scores, seen = transferred_scores(("late", "high"), [a, b, c], fit)
        self.assertEqual(scores, [1., 0., .5])
        self.assertEqual(seen, [True, True, False])


    def test_validation_masks_are_frozen_before_outcome_collection(self):
        pairs = [(1, 2), (2, 3)]
        fit = fit_transfer([(("early", "low"), pairs, [1., -1.])], 3)
        cfg = dict(top_k=1, random_seeds=[101, 102], rollout_repeats=1,
                   projection_margin=0., softmax_temperature=1.)
        row = dict(task="stalker", episode_id="held_out", timestep=1, agent_id=2,
                   battle_phase="early", regret_stratum="low", legal_actions="[1, 2, 3]")
        item = (row, None, pairs, [0., -2., 1., 3.])
        selections, values = [], []
        def audited_collector(*args):
            self.assertEqual(len(selections), 5)
            self.assertEqual(values, [])
            return np.asarray([[[0., 0.], [1., 0.], [2., 1.]]]), []
        audit = dict(coalitions=0, projection_residual=0., stationarity_residual=0.)
        with patch("validate_shapley_frpr.collect_action_branches", audited_collector):
            validate_context(item, None, cfg, audit, fit, selections, values)
        self.assertEqual(next(r["mask"] for r in selections if r["arm"] == "shapley_topk"), 1)
        self.assertEqual(row["status"], "complete")
        self.assertEqual(len(values), 5)

    def test_bootstrap_pairs_repeated_contexts_with_same_initial_seed(self):
        rows = [{"task": "stalker", "env_seed": seed} for seed in (1, 1, 2, 2)]
        self.assertEqual(bootstrap(rows, [0., 2., 2., 4.], 5000, 5), (2, 1., 3.))
        self.assertEqual(bootstrap(rows[:2], [0., 2.], 100, 5), (1, None, None))


if __name__ == "__main__":
    unittest.main()
