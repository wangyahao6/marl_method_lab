"""Fixed pair budgets, gap rules, and unchanged preference loss."""
import json
from pathlib import Path
import unittest

import torch as th

from action_set_protocol import PAIR_DISTANCE, PAIR_GAP_FAR_MID, validate
from learners.pair_ablation import PairAblationPreference, ablation_loss
from learners.pair_distance import PairDistancePreference, select_distance_pairs, distance_statistics


ROOT = Path(__file__).resolve().parents[1]
SEED = 3535


def budget(n):
    return min(n - 2, n - 3, (n - 4) * (n - 3) // 2) if n >= 5 else 0


def pair_ids(mask, size):
    return {tuple(pair) for pair in th.triu_indices(size, size, 1)[:, mask].T.tolist()}


class PairDistanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def test_all_structures_preserve_top1_and_exact_extra_budget(self):
        size = 11
        counts = th.arange(4, size + 1)
        actions = th.arange(size)
        legal = (actions - counts[:, None]) % size < counts[:, None]
        old = actions.float().expand_as(legal)
        decisions = th.ones(len(counts), dtype=th.bool)
        all_masks = {}
        for method in PAIR_DISTANCE.values():
            mask, valid, _, audit = select_distance_pairs(old, legal, decisions, method, SEED)
            self.assertTrue(valid.all())
            th.testing.assert_close(mask, select_distance_pairs(old, legal, decisions, method, SEED)[0])
            all_masks[method] = mask
            for state, (row, n) in enumerate(zip(mask, counts.tolist())):
                chosen = pair_ids(row, size)
                legal_ids = legal[state].nonzero().flatten().tolist()
                top = max(legal_ids)
                self.assertEqual({p for p in chosen if top in p},
                                 {tuple(sorted((top, a))) for a in legal_ids if a != top})
                extra = {p for p in chosen if top not in p}
                self.assertEqual(len(extra), 0 if method == "k1" else budget(n))
                self.assertEqual(len(chosen), n - 1 + len(extra))
                ranks = {a: i for i, a in enumerate(sorted(legal_ids, reverse=True))}
                distances = [abs(ranks[a] - ranks[b]) for a, b in extra]
                if method == "near":
                    self.assertTrue(all(d == 1 for d in distances))
                elif method == "mid":
                    self.assertTrue(all(d == 2 for d in distances))
                elif method == "far":
                    self.assertTrue(all(d >= 3 for d in distances))
                elif method == "far_mid":
                    self.assertEqual(sum(d >= 3 for d in distances), (2 * budget(n) + 1) // 3)
                    self.assertEqual(sum(d == 2 for d in distances), budget(n) - (2 * budget(n) + 1) // 3)
                elif method == "maxgap":
                    available = {tuple(sorted(p)) for p in __import__("itertools").combinations(legal_ids, 2)
                                 if top not in p}
                    expected = sorted(available, key=lambda p: (-abs(p[1] - p[0]), p))[:budget(n)]
                    self.assertEqual(extra, set(expected))
            self.assertEqual(audit["extra_selected_pairs"],
                             0 if method == "k1" else sum(budget(int(n)) for n in counts))

    def test_gap_arms_share_states_and_top1_while_filtering_only_extras(self):
        old = th.tensor([[5., 4., 3., 2., 1., 0., -1.],
                         [100., 4., 3., 2., 1., 0., -1.],
                         [5., 4., 3., 2., 1., 0., -1.],
                         [5., 4., 3., 2., 1., 0., -1.]])
        legal = th.ones_like(old, dtype=th.bool)
        legal[2, 4:] = False  # Four legal actions: no additional pair.
        old[3, 4] = old[3, 5]  # Old-Q tie excludes the state.
        decisions = th.ones(4, dtype=th.bool)
        masks = {}
        for method in PAIR_GAP_FAR_MID.values():
            threshold = .5 if method == "far_mid_filtered" else 0
            mask, valid, _, audit = select_distance_pairs(old, legal, decisions, method, SEED,
                                                           extra_min_gap=threshold)
            self.assertEqual(valid.tolist(), [True, False, True, False])
            th.testing.assert_close(mask, select_distance_pairs(old, legal, decisions, method, SEED,
                                                               extra_min_gap=threshold)[0])
            self.assertEqual(audit["excluded_by_extra_gap_states"], 1)
            self.assertFalse(mask[1].any() or mask[3].any())
            masks[method] = mask
            for state in (0, 2):
                n = int(legal[state].sum())
                self.assertEqual(int(mask[state].sum()), n - 1 + budget(n))
                if state == 0:
                    top_pairs = {p for p in pair_ids(mask[state], 7) if 0 in p}
                    self.assertEqual(len(top_pairs), n - 1)
                if method == "far_mid_filtered":
                    scale = old[state, legal[state]].std(unbiased=False).clamp_min(.1)
                    for a, b in pair_ids(mask[state], 7):
                        if a != 0 and b != 0:
                            self.assertGreaterEqual(abs(float(old[state, a] - old[state, b])) / float(scale), .5)
        for state in (0, 2):
            top = int(old[state].masked_fill(~legal[state], -th.inf).argmax())
            a = {p for p in pair_ids(masks["far_mid_unfiltered"][state], 7) if top in p}
            b = {p for p in pair_ids(masks["far_mid_filtered"][state], 7) if top in p}
            self.assertEqual(a, b)

    def test_pair_loss_and_equal_history_task_reduction(self):
        old = th.tensor([[8., 7., 6., 5., 4., 3.]], requires_grad=True)
        current = th.tensor([[3., 5., 2., 7., 1., 0.]], requires_grad=True)
        legal = th.ones_like(old, dtype=th.bool)
        decisions = th.ones(1, dtype=th.bool)
        selected, _, counts, _ = select_distance_pairs(old, legal, decisions, "far_mid", SEED)
        settings = dict(temperature=1., q_scale_floor=.1)
        loss, valid, summary = ablation_loss(current, old, legal, decisions, settings,
                                              selected, counts, pair_audit=True)
        self.assertEqual(int(valid), 1)
        stats = distance_statistics(summary.tolist())
        self.assertEqual(stats["valid_pairs"], 5 + budget(6))
        self.assertTrue(0 <= stats["pair_violation_rate"] <= 1)
        gradient, old_gradient = th.autograd.grad(loss, (current, old), allow_unused=True)
        self.assertIsNone(old_gradient)
        self.assertTrue(gradient.isfinite().all())
        self.assertGreater(float(gradient.norm()), 0)
        losses = th.tensor([1., 5., 99.], requires_grad=True)
        result = PairDistancePreference.reduce_task_losses(losses.unbind(), th.tensor([2, 10, 0]))
        th.testing.assert_close(result, th.tensor(3.))
        th.testing.assert_close(th.autograd.grad(result, losses)[0], th.tensor([.5, .5, 0.]))
        self.assertIs(PairDistancePreference.reduce_task_losses, PairAblationPreference.reduce_task_losses)

    def test_protocols_keep_original_pair_methods(self):
        base = ROOT / "configs/continual/action_preference"
        a = json.loads((base / "pair_distance/szc_vdn64_pair_distance_1m.json").read_text())
        b = json.loads((base / "pair_gap_far_mid/szc_vdn64_pair_gap_far_mid_1m.json").read_text())
        original = json.loads((base / "pair_structure/szc_vdn64_pair_structure_1m.json").read_text())
        self.assertEqual(original["pair_structure_k1"]["min_gap"], 0)
        for protocol, method in ((a, "pair_distance_k1"), (a, "pair_distance_maxgap"),
                                 (b, "pair_gap_far_mid_unfiltered"), (b, "pair_gap_far_mid_filtered")):
            checked = validate(protocol, "vdn", method)
            self.assertEqual(checked[method]["min_gap"], 0)
            self.assertEqual(checked[method]["pair_seed"], SEED)
        self.assertEqual(b["pair_gap_far_mid_filtered"]["extra_min_gap"], .5)


if __name__ == "__main__":
    unittest.main()
