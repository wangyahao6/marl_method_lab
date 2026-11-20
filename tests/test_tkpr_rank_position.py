import json
import unittest

import torch as th
from torch.nn.functional import softplus

from test_action_set_preference import driver
from learners.ranking_preference import rank_position_preference, rank_position_statistics, topk_preference


class RankPositionTest(unittest.TestCase):
    def test_only_single_anchor_incident_pairs_and_stable_ties(self):
        # Old ranks are action 3, action 0, action 2, action 1. The tie is stable.
        old = th.tensor([[4., 0., 4., 8., 999.]]).repeat(3, 1)
        legal = th.tensor([[True, True, True, True, False],
                           [True, False, False, True, False],
                           [False, False, False, True, False]])
        current = th.tensor([[2., 3., 1., 0., -100.]]).repeat(3, 1).requires_grad_()
        decisions = th.ones(3, dtype=th.bool)
        settings = dict(min_gap=.5, temperature=1., q_scale_floor=.1, k=1)
        for rank, anchor, valid in ((2, 0, 2), (3, 2, 1)):
            loss, count, summary = rank_position_preference(current, old, legal, decisions, settings, rank=rank)
            rows = []
            for index in range(valid):
                ids = legal[index].nonzero().flatten().tolist()
                scale = old[index, ids].std(unbiased=False).clamp_min(.1)
                losses = []
                for j in ids:
                    gap = old[index, anchor] - old[index, j]
                    if gap != 0 and gap.abs() / scale >= settings["min_gap"]:
                        losses.append(softplus(-gap.sign() * (current[index, anchor] - current[index, j]) / scale))
                rows.append(th.stack(losses).mean())
            expected = th.stack(rows).mean()
            th.testing.assert_close(loss, expected)
            actual_grad, = th.autograd.grad(loss, current, retain_graph=True)
            expected_grad, = th.autograd.grad(expected, current, retain_graph=True)
            th.testing.assert_close(actual_grad, expected_grad)
            self.assertEqual(count.item(), valid)
            stats = rank_position_statistics(summary.tolist(), rank=rank, action_count=5)
            self.assertEqual(stats["anchor_available_states"], valid)
            self.assertEqual(stats["candidate_pairs"], 4 if rank == 2 else 3)
            self.assertEqual(stats["tie_pairs"], 1)
            self.assertEqual(stats["valid_pairs"], 3 if rank == 2 else 2)
            self.assertEqual(stats["pair_violations"], 3 if rank == 2 else 2)
            self.assertEqual(stats["anchor_new_rank_histogram"][str(rank)], 1)
            self.assertEqual(stats["anchor_new_rank_histogram"]["1"], int(rank == 2))

    def test_audited_k1_matches_original_and_gap_filter_applies(self):
        old = th.tensor([[6., 5.99, 5.98, 0.]])
        current = th.tensor([[0., 1., 2., 3.]], requires_grad=True)
        legal = th.ones_like(old, dtype=th.bool)
        decisions = th.ones(1, dtype=th.bool)
        settings = dict(min_gap=.5, temperature=1., q_scale_floor=.1, k=1)
        standard = topk_preference(current, old, legal, decisions, settings)
        audited = rank_position_preference(current, old, legal, decisions, settings, rank=1)
        for actual, expected in zip(audited[:2], standard[:2]):
            th.testing.assert_close(actual, expected)
        th.testing.assert_close(audited[2][:len(standard[2])], standard[2])
        for rank in (1, 2, 3):
            _, _, summary = rank_position_preference(current, old, legal, decisions, settings, rank=rank)
            stats = rank_position_statistics(summary.tolist(), rank=rank, action_count=4)
            self.assertEqual(stats["eligible_pairs"], 3)
            self.assertEqual(stats["filtered_pairs"], 2)
            self.assertEqual(stats["valid_pairs"], 1)

    def test_ranks_inherit_k1_protocol_and_match_existing_parent(self):
        root = driver.ROOT / "configs/continual/action_preference/szc"
        parent_config = json.loads((root / "szc_vdn64_tkpr_k1_shared.json").read_text())
        new_config = json.loads((root / "szc_vdn64_tkpr_rank_position.json").read_text())
        parent = driver.validate(parent_config, "vdn", "topk_preference")
        expected = {k: v for k, v in parent["topk_preference"].items() if k != "k"}
        for method in ("topk_preference", "tkpr_r2", "tkpr_r3"):
            child = driver.validate(new_config, "vdn", method)
            self.assertTrue(driver.matching_shared_protocol(parent, child))
            if method != "topk_preference":
                self.assertEqual(child[method], expected)
                with self.assertRaisesRegex(ValueError, "same loss settings"):
                    driver.validate(dict(new_config, **{method: {"min_gap": 0.}}), "vdn", method)
