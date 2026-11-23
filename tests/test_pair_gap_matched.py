"""Exact sorted-gap matching and unchanged pair preference training loss."""
import itertools
import json
from pathlib import Path
import unittest

import torch as th

from action_set_protocol import PAIR_DEFAULTS, PAIR_GAP_MATCHED, validate
from learners.pair_ablation import ablation_loss, select_pairs
from learners.pair_gap_matched import optimal_gap_subset, select_gap_matched_pairs
from learners.ranking_preference import reference_order


ROOT = Path(__file__).resolve().parents[1]


class GapMatchedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def test_dp_matches_brute_force_and_breaks_ties_by_original_pair_id(self):
        targets = [0.7, 1.4, 2.1]
        candidates = [(0.3, 7), (0.8, 5), (1.2, 4), (1.5, 2), (2.2, 9), (2.5, 1)]
        chosen, w1 = optimal_gap_subset(targets, candidates)
        exact = min((sum(abs(x - y) for x, y in zip(sorted(targets), sorted(v for v, _ in subset)))
                     / len(targets), tuple(sorted(pair for _, pair in subset)))
                    for subset in itertools.combinations(candidates, len(targets)))
        self.assertEqual((w1, chosen), exact)
        self.assertEqual(optimal_gap_subset([1., 2.], [(1., 5), (1., 3), (2., 7)])[0], (3, 7))

    def test_selection_matches_archived_k1_ratio0_valid_rule_and_budget(self):
        size = 11
        counts = th.arange(4, size + 1)
        actions = th.arange(size)
        legal = (actions - counts[:, None]) % size < counts[:, None]
        old = actions.float().expand_as(legal).clone()
        # A strict old-Q tie is excluded by the archived structure-K1 rule.
        tied = legal[1].nonzero().flatten()
        old[1, tied[0]] = old[1, tied[1]]
        decisions = th.ones(len(counts), dtype=th.bool)
        mask, valid, selected_counts, audit = select_gap_matched_pairs(
            old, legal, decisions, "gap_matched_non_top1", 3535)
        k1, k1_valid, _, _ = select_pairs(old, legal, decisions, "k1", 3535)
        ratio0, ratio0_valid, _, _ = select_pairs(old, legal, decisions, 0, 3535)
        th.testing.assert_close(valid, k1_valid & ratio0_valid)
        th.testing.assert_close(mask, select_gap_matched_pairs(
            old, legal, decisions, "gap_matched_non_top1", 2222)[0])
        self.assertEqual(audit["budget_mismatch_states"], 0)
        self.assertFalse(mask[1].any())
        left, right = th.triu_indices(size, size, 1)
        standardized, _, _, _ = reference_order(old, legal, .1)
        gap = (standardized[..., left] - standardized[..., right]).abs()
        top = old.masked_fill(~legal, -th.inf).argmax(-1)
        for row in valid.nonzero().flatten():
            n = int(legal[row].sum())
            chosen = mask[row]
            self.assertEqual(int(chosen.sum()), n - 1)
            self.assertEqual(int(selected_counts[row, 2]), 0)
            self.assertFalse((chosen & ((left == top[row]) | (right == top[row]))).any())
            pairs = [tuple(pair) for pair in th.stack((left[chosen], right[chosen]), -1).tolist()]
            self.assertEqual(len(pairs), len(set(pairs)))
            target = sorted(gap[row, k1[row]].tolist())
            matched = sorted(gap[row, chosen].tolist())
            random = sorted(gap[row, ratio0[row]].tolist())
            self.assertLessEqual(sum(abs(a - b) for a, b in zip(target, matched)),
                                 sum(abs(a - b) for a, b in zip(target, random)) + 1e-6)

    def test_loss_has_normal_gradient_and_old_q_is_frozen(self):
        old = th.tensor([[8., 7., 6., 5., 4., 3.]], requires_grad=True)
        current = th.tensor([[3., 5., 2., 7., 1., 0.]], requires_grad=True)
        legal = th.ones_like(old, dtype=th.bool)
        decisions = th.ones(1, dtype=th.bool)
        mask, valid, counts, _ = select_gap_matched_pairs(old, legal, decisions,
                                                           "gap_matched_non_top1", 3535)
        loss, active, _ = ablation_loss(current, old, legal, decisions,
                                         dict(temperature=1., q_scale_floor=.1), mask, counts,
                                         pair_audit=True)
        self.assertEqual(int(active), 1)
        gradient, old_gradient = th.autograd.grad(loss, (current, old), allow_unused=True)
        self.assertIsNone(old_gradient)
        self.assertTrue(gradient.isfinite().all())
        self.assertGreater(float(gradient.norm()), 0)

    def test_protocol_is_independent_of_existing_methods(self):
        root = ROOT / "configs/continual/action_preference"
        new = json.loads((root / "pair_gap_matched/szc_vdn64_pair_gap_matched_1m.json").read_text())
        old = json.loads((root / "pair_structure/szc_vdn64_pair_structure_1m.json").read_text())
        self.assertEqual(new["pair_gap_matched_non_top1"], old["pair_structure_k1"])
        self.assertEqual(new["pair_gap_matched_non_top1"], PAIR_DEFAULTS)
        self.assertEqual(PAIR_GAP_MATCHED, {"pair_gap_matched_non_top1": "gap_matched_non_top1"})
        self.assertEqual(validate(new, "vdn", "pair_gap_matched_non_top1")["method"],
                         "pair_gap_matched_non_top1")


if __name__ == "__main__":
    unittest.main()
