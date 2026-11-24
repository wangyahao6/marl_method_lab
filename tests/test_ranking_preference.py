"""Ranking loss checks and CPU sequence tests with real VDN/GRU."""
import copy
import json
import subprocess
import sys
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch as th
from torch.nn.functional import softplus

from test_action_set_preference import PROTOCOL_PATH, ScriptedProtoss, driver, jsonlines
from action_set_protocol import (
    ASPR_GROUP_DEFAULTS,
    FRPR_RAW_DEFAULTS, FRPR_RAW_LOSS_DEFAULTS, FRPR_UNFILTERED_DEFAULTS,
    RANK_DEFAULTS, RANK_METHODS, RANK_RELU_DEFAULTS,
    TOP1_MAX_DEFAULTS, TOP1_UNFILTERED_DEFAULTS, TOPK_DEFAULTS,
    TOPK_HIGH_REGRET_DEFAULTS, TOPK_WEIGHTED_DEFAULTS,
    CPR_DEFAULTS,
)
from protection import digest
from learners.action_set_preference import batch_episodes, decision_masks, make_good_mask
from drivers.continual.memory import PreferenceHistory
from learners.q_learner import agent_gradient_metrics
from learners.ranking_preference import (
    RankingPreference, aggregate_pair_filter_statistics, aspr_preference, aspr_statistics,
    aspr_group_preference, aspr_group_statistics, group_old_q,
    full_rank_preference, full_rank_preference_raw,
    full_rank_preference_raw_loss, full_rank_preference_relu,
    full_rank_preference_unfiltered, full_rank_statistics, pair_filter_statistics, top1_max_preference,
    top1_max_statistics, top1_preference, top1_unfiltered_preference, top1_unfiltered_statistics,
    topk_preference, topk_statistics,
)


METHOD_DEFAULTS = {
    **ASPR_GROUP_DEFAULTS,
    "full_rank_preference_relu": RANK_RELU_DEFAULTS,
    "full_rank_preference_unfiltered": FRPR_UNFILTERED_DEFAULTS,
    "full_rank_preference_raw": FRPR_RAW_DEFAULTS,
    "full_rank_preference_raw_loss": FRPR_RAW_LOSS_DEFAULTS,
    "top1_max_preference": TOP1_MAX_DEFAULTS,
    "top1_unfiltered_preference": TOP1_UNFILTERED_DEFAULTS,
    "topk_preference": TOPK_DEFAULTS,
    "tkpr_refresh": TOPK_WEIGHTED_DEFAULTS,
    "tkpr_cpr": CPR_DEFAULTS,
    "tkpr_k1_gap0": dict(TOPK_WEIGHTED_DEFAULTS, min_gap=0.0),
    "topk_preference_high_regret": TOPK_HIGH_REGRET_DEFAULTS,
    "topk_preference_weighted": TOPK_WEIGHTED_DEFAULTS,
}


class RankingTest(unittest.TestCase):
    def test_aspr_grouping_exact_ties_and_cross_group_loss(self):
        old = th.tensor([[10., 10., 9.9, 5., 5., 0., -100.]])
        legal = th.tensor([[True] * 6 + [False]])
        decision = th.tensor([True])
        groups = group_old_q(old, legal, decision, 4, .1)
        self.assertEqual(groups.tolist(), [[0, 0, 1, 2, 2, 3, -1]])
        th.testing.assert_close(groups, group_old_q(old, legal, decision, 4, .1))
        self.assertEqual(group_old_q(old, legal, decision, 3, .1)[0, 0].item(), 0)
        self.assertEqual(group_old_q(old, legal, decision, 3, .1)[0, 1].item(), 0)
        for k in (3, 4):
            ids = group_old_q(old, legal, decision, k, .1)
            current = th.tensor([[1., 2., 3., 4., 5., 6., 100.]], requires_grad=True)
            settings = dict(RANK_DEFAULTS, group_count=k, min_gap=0)
            loss, count, summary = aspr_group_preference(current, old, legal, decision,
                                                         settings, group_ids=ids)
            scale = old[0, :6].std(unbiased=False).clamp_min(.1)
            pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)
                     if ids[0, a] != ids[0, b] and old[0, a] != old[0, b]]
            expected = th.stack([softplus((current[0, b] - current[0, a]) / scale)
                                 for a, b in pairs]).mean()
            th.testing.assert_close(loss, expected)
            th.testing.assert_close(th.autograd.grad(loss, current)[0],
                                    th.autograd.grad(expected, current)[0])
            self.assertEqual(count.item(), 1)
            stats = aspr_group_statistics(summary.tolist())
            self.assertEqual(stats["candidate_pairs"], len(pairs))
            self.assertEqual(stats["valid_pairs"], len(pairs))
            self.assertFalse(current.grad)

    def test_aspr_grouping_small_and_invalid_states(self):
        old = th.tensor([[3., 3., 3., 99.], [1., 0., 0., 0.], [3., 2., 1., 0.]])
        legal = th.tensor([[True, True, True, False], [True, False, False, False], [True] * 4])
        decisions = th.tensor([True, True, False])
        ids = group_old_q(old, legal, decisions, 4, .1)
        self.assertEqual(ids.tolist(), [[0, 0, 0, -1], [0, -1, -1, -1], [-1] * 4])
        current = old.clone().requires_grad_()
        loss, count, summary = aspr_group_preference(current, old, legal, decisions,
                                                     dict(RANK_DEFAULTS, group_count=4), group_ids=ids)
        self.assertEqual(count.item(), 0)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(aspr_group_statistics(summary.tolist())["candidate_pairs"], 0)

    def test_aspr_cross_group_softplus_and_filtering(self):
        # Interleave Good and Bad indices so both index orientations are exercised.
        old = th.tensor([[5., 0., 4., 1., 3., 100.]]).repeat(3, 1)
        current = th.tensor([[2., 4., 5., 2., 0., -100.]]).repeat(3, 1).requires_grad_()
        legal = th.tensor([[True, True, True, True, True, False]]).repeat(3, 1)
        good = th.tensor([[True, False, True, False, True, True]]).repeat(3, 1)
        good[1] = legal[1]  # All legal actions are Good.
        decisions = th.tensor([True, True, False])
        for min_gap, floor, temperature, expected_count in ((0., .1, 1., 6), (.75, 4., 2., 5)):
            settings = dict(RANK_DEFAULTS, min_gap=min_gap, q_scale_floor=floor, temperature=temperature)
            loss, count, summary = aspr_preference(current, old, legal, decisions, settings, good=good)
            scale = old[0, :5].std(unbiased=False).clamp_min(floor)
            pairs = [(a, b) for a in (0, 2, 4) for b in (1, 3) if (old[0, a] - old[0, b]) / scale >= min_gap]
            expected = th.stack([softplus((current[0, b] - current[0, a]) / (scale * temperature))
                                 for a, b in pairs]).mean()
            th.testing.assert_close(loss, expected)
            self.assertEqual(count.item(), 1)
            stats = aspr_statistics(summary.tolist())
            self.assertEqual(stats["candidate_pairs"], 6)
            self.assertEqual(stats["valid_pairs"], expected_count)
            self.assertEqual(stats["all_action_ratio"], .5)
            self.assertEqual(stats["average_good_set_size"], 4.)
            self.assertEqual(stats["pair_keep_rate"], expected_count / 6)
            violations = sum(current[0, a] <= current[0, b] for a, b in pairs).item()
            self.assertEqual(stats["pair_violation_rate"], violations / expected_count)
            gradient = th.autograd.grad(loss, current)[0]
            th.testing.assert_close(gradient, th.autograd.grad(expected, current)[0])
            self.assertFalse(gradient[:, 5].any())
            self.assertFalse(gradient[1:].any())

    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def test_full_rank_also_protects_non_top_pair(self):
        reference = th.tensor([[1.0, 3.0, 0.0, 5000.0]], requires_grad=True)
        current = th.tensor([[0.0, 3.0, 1.0, -5000.0]], requires_grad=True)
        legal = th.tensor([[True, True, True, False]])
        decisions = th.tensor([True])
        scale = reference.detach()[0, :3].std(unbiased=False)

        loss, count, stats = full_rank_preference(current, reference, legal, decisions, RANK_DEFAULTS)
        expected = softplus(th.tensor([-3.0, 1.0, -2.0]) / scale).mean()
        th.testing.assert_close(loss, expected)
        self.assertEqual(count.item(), 1)
        self.assertEqual(stats[2:].tolist(), [3, 1, 0, 1, 1])
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(current.grad[0, 3].item(), 0)

        top1, count, stats = top1_preference(current, reference, legal, decisions, RANK_DEFAULTS)
        expected = softplus(th.tensor([-3.0, -2.0]) / scale).mean() * (2 / scale)
        th.testing.assert_close(top1, expected)
        self.assertEqual(stats[2:].tolist(), [2, 1, 0, 1, 1])

    def test_full_rank_relu_gradients_and_zero_loss(self):
        reference = th.tensor([[1.0, 3.0, 0.0, 5000.0]], requires_grad=True)
        legal = th.tensor([[True, True, True, False]])
        decisions = th.tensor([True])
        scale = reference.detach()[0, :3].std(unbiased=False)
        for values in ([1.0, 3.0, 0.0, 5000.0], [2.0, 5.0, -2.0, -5000.0],
                       [2.0, 2.0, 2.0, -5000.0], [0.0, 3.0, 1.0, -5000.0]):
            with self.subTest(current_q=values):
                current = th.tensor([values], requires_grad=True)
                loss, count, stats = full_rank_preference_relu(
                    current, reference, legal, decisions, RANK_RELU_DEFAULTS)
                self.assertEqual((count.item(), stats[2].item()), (1, 3))
                loss.backward()
                self.assertIsNone(reference.grad)
                if values[0] < values[2]:
                    expected = 1 / (3 * scale)
                    th.testing.assert_close(loss, expected)
                    th.testing.assert_close(current.grad, th.tensor([[-1., 0., 1., 0.]]) * expected)
                else:
                    self.assertEqual(loss.item(), 0)
                    th.testing.assert_close(current.grad, th.zeros_like(current))

    def test_full_rank_relu_masks_invalid_decisions_and_small_gaps(self):
        # Each row has two agents. Step 0 is valid, step 1 is terminal,
        # step 2 is bootstrap, and step 3 is padding.
        batch = {
            "filled": th.tensor([[[1], [1], [1], [0], [0]]]),
            "terminated": th.tensor([[[0], [1], [0], [0], [0]]]),
            "avail_actions": th.tensor([[[[0, 1, 1], [0, 1, 1]],
                                         [[0, 1, 1], [1, 0, 0]],
                                         [[0, 1, 1], [0, 1, 1]],
                                         [[0, 1, 1], [0, 1, 1]],
                                         [[0, 0, 0], [0, 0, 0]]]]),
        }
        legal, decisions = decision_masks(batch)
        decisions = decisions & (legal.sum(dim=-1) >= 2)
        current = th.tensor([[[[5000., 0., 2.], [5000., 0., 2.]]] * 4], requires_grad=True)
        reference = th.tensor([[[[-5000., 2., 0.], [-5000., .01, 0.]]] * 4])
        loss, count, stats = full_rank_preference_relu(
            current, reference, legal, decisions, RANK_RELU_DEFAULTS)
        self.assertEqual((count.item(), stats[2].item()), (2, 2))
        th.testing.assert_close(loss, th.tensor(2.))
        loss.backward()
        expected = th.zeros_like(current)
        expected[0, :2, 0, 1] = -.5
        expected[0, :2, 0, 2] = .5
        th.testing.assert_close(current.grad, expected)

    def test_frpr_unfiltered_keeps_small_gaps_and_ignores_ties(self):
        reference = th.tensor([[2., 2., 1.95, 0., 500.], [1., 1., 1., 500., 500.],
                               [3., 2., 1., 0., 500.]], requires_grad=True)
        current = th.tensor([[0., 1., 2., 3., 500.], [1., 2., 3., 500., 500.],
                             [0., 1., 2., 3., 500.]], requires_grad=True)
        legal = th.tensor([[True, True, True, True, False], [True, True, True, False, False],
                           [True, True, True, True, False]])
        decisions = th.tensor([True, True, False])
        original = full_rank_preference(current, reference, legal, decisions, RANK_DEFAULTS, pair_audit=True)
        new = full_rank_preference_unfiltered(current, reference, legal, decisions, FRPR_UNFILTERED_DEFAULTS)
        zero_gap = full_rank_preference(
            current, reference, legal, decisions, dict(RANK_DEFAULTS, min_gap=0), pair_audit=True)
        self.assertEqual(new[1].item(), 1)
        for result, valid_pairs in ((original, 3), (new, 5)):
            stats = full_rank_statistics(result[2].tolist())
            self.assertEqual(stats["candidate_pairs"], 9)
            self.assertEqual(stats["tie_pairs"], 4)
            self.assertEqual(stats["eligible_pairs"], 5)
            self.assertEqual(stats["valid_pairs"], valid_pairs)
            self.assertEqual(stats["filtered_pairs"], 5 - valid_pairs)
            self.assertEqual(stats["pair_filter_rate"], (5 - valid_pairs) / 5)
            self.assertEqual(stats["pair_keep_rate"], valid_pairs / 5)
        self.assertTrue(all(th.equal(a, b) for a, b in zip(new, zero_gap)))
        gradient, reference_gradient = th.autograd.grad(new[0], (current, reference), allow_unused=True)
        self.assertTrue(th.equal(gradient, th.autograd.grad(zero_gap[0], current)[0]))
        self.assertIsNone(reference_gradient)
        self.assertFalse(gradient[1:].any())
        self.assertFalse(gradient[:, 4].any())

    def test_frpr_raw_uses_raw_threshold_and_softplus_without_reference_normalization(self):
        reference = th.tensor([[4., 3., 1., 1., 500.], [0., 3., 500., 500., 500.],
                               [1., 1., 1., 1., 500.], [4., 3., 1., 1., 500.]], requires_grad=True)
        current = th.tensor([[0., 1., 2., 3., 500.]] * 4, requires_grad=True)
        legal = th.tensor([[True, True, True, True, False], [True, True, False, False, False],
                           [True, True, True, True, False], [True, True, True, True, False]])
        decisions = th.tensor([True, True, True, False])
        settings = dict(FRPR_RAW_DEFAULTS, min_gap_raw=2., temperature=0.5)
        first_state = (current[0, 2:4] - current[0, :2].unsqueeze(-1)).reshape(-1)
        second_state = current[1, 0] - current[1, 1]
        expected = (softplus(first_state / 0.5).mean() + softplus(second_state / 0.5)) / 2

        with patch("learners.ranking_preference.reference_order",
                   side_effect=AssertionError("Raw must not normalize old Q")):
            loss, count, summary = full_rank_preference_raw(current, reference, legal, decisions, settings)
            _, _, zero_summary = full_rank_preference_raw(
                current, reference, legal, decisions, dict(settings, min_gap_raw=0))
        th.testing.assert_close(loss, expected)
        self.assertEqual(count.item(), 2)
        stats = full_rank_statistics(summary.tolist())
        self.assertEqual((stats["candidate_pairs"], stats["tie_pairs"], stats["valid_pairs"]), (13, 7, 5))
        self.assertEqual(full_rank_statistics(zero_summary.tolist())["valid_pairs"], 6)
        gradient, reference_gradient = th.autograd.grad(loss, (current, reference), allow_unused=True)
        th.testing.assert_close(gradient, th.autograd.grad(expected, current)[0])
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertIsNone(reference_gradient)

    def test_frpr_raw_loss_keeps_normalized_pair_mask_but_uses_raw_softplus(self):
        reference = th.tensor([[4., 3.5, 0., 0., 500.]], requires_grad=True)
        current = th.tensor([[0., 1., 2., -3., 500.]], requires_grad=True)
        legal = th.tensor([[True, True, True, True, False]])
        decisions = th.tensor([True])
        settings = dict(FRPR_RAW_LOSS_DEFAULTS, temperature=0.5)
        logits = (current[0, 2:4] - current[0, :2].unsqueeze(-1)).reshape(-1) / 0.5
        scale = reference.detach()[0, :4].std(unbiased=False)

        original, original_count, original_summary = full_rank_preference(
            current, reference, legal, decisions, settings, pair_audit=True)
        loss, count, summary = full_rank_preference_raw_loss(current, reference, legal, decisions, settings)
        self.assertTrue(th.equal(count, original_count))
        self.assertTrue(th.equal(summary[1:], original_summary[1:]))
        stats = full_rank_statistics(summary.tolist())
        self.assertEqual((stats["tie_pairs"], stats["filtered_pairs"], stats["valid_pairs"]), (1, 1, 4))

        # A distinct power of two for each pair identifies the exact retained set.
        pair_codes = (2. ** th.arange(25)).reshape(1, 5, 5)
        with patch("learners.ranking_preference.softplus", return_value=pair_codes):
            original_mask, _, _ = full_rank_preference(current, reference, legal, decisions, settings)
            raw_loss_mask, _, _ = full_rank_preference_raw_loss(current, reference, legal, decisions, settings)
        th.testing.assert_close(original_mask, pair_codes[0, :2, 2:4].mean())
        self.assertTrue(th.equal(raw_loss_mask, original_mask))

        expected = softplus(logits).mean()
        th.testing.assert_close(original, softplus(logits / scale).mean())
        th.testing.assert_close(loss, expected)
        self.assertFalse(th.isclose(loss, original))
        gradient, reference_gradient = th.autograd.grad(loss, (current, reference), allow_unused=True)
        th.testing.assert_close(gradient, th.autograd.grad(expected, current)[0])
        self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertIsNone(reference_gradient)

    def test_frpr_pair_rates_use_total_counts(self):
        tasks = [pair_filter_statistics(3, 2, 1), pair_filter_statistics(10, 8, 2)]
        self.assertEqual(aggregate_pair_filter_statistics(tasks), dict(
            candidate_pairs=13, tie_pairs=3, eligible_pairs=10, valid_pairs=3,
            filtered_pairs=7, pair_filter_rate=.7, pair_keep_rate=.3))
        self.assertEqual(aggregate_pair_filter_statistics([])["pair_filter_rate"], 0.)
        self.assertIsNone(aggregate_pair_filter_statistics([])["pair_keep_rate"])

    def test_top1_filters_ties_near_ties_and_masked_decisions(self):
        reference = th.tensor([[1.0, 0.99, -1.0], [1.0, 1.0, -1.0], [1.0, 0.99, 5000.0],
                               [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
        current = th.tensor([[0.0, 1.0, 0.0]] * 5, requires_grad=True)
        legal = th.tensor([[True, True, True]] * 5)
        legal[2, 2] = False
        decisions = th.tensor([True, True, True, True, False])

        loss, count, stats = top1_preference(current, reference, legal, decisions, RANK_DEFAULTS)
        scale = reference[3].std(unbiased=False)
        expected = softplus(th.tensor([1.0, 0.0]) / scale).mean() / scale
        th.testing.assert_close(loss, expected)
        self.assertEqual(count.item(), 1)
        self.assertEqual(stats[2:].tolist(), [2, 4, 3, 3, 1])
        loss.backward()
        self.assertTrue(th.equal(current.grad[:3], th.zeros_like(current.grad[:3])))
        self.assertTrue(th.equal(current.grad[4], th.zeros_like(current.grad[4])))
        self.assertLess(current.grad[3, 0].item(), 0)

        for rank_loss in (full_rank_preference, top1_preference):
            current.grad = None
            disabled = dict(RANK_DEFAULTS, min_gap=10.0)
            loss, count, stats = rank_loss(current, reference, legal, decisions, disabled)
            self.assertEqual((loss.item(), count.item(), stats[2].item()), (0, 0, 0))
            loss.backward()
            self.assertTrue(th.equal(current.grad, th.zeros_like(current.grad)))

    def test_top1_max_loss_margin_and_only_strongest_gradient(self):
        reference = th.tensor([[3., 1., 0., 200.]], requires_grad=True)
        legal = th.tensor([[True, True, True, False]])
        decisions = th.tensor([True])
        scale = reference.detach()[0, :3].std(unbiased=False)
        for values, margin, difference in (
            ([4., 2., 1., 300.], 0., -2.),
            ([1., 3., 2., 300.], 0., 2.),
            ([2., 2., 1., 300.], .25, 0.),
        ):
            with self.subTest(values=values, margin=margin):
                current = th.tensor([values], requires_grad=True)
                loss, count, _ = top1_max_preference(
                    current, reference, legal, decisions, dict(TOP1_MAX_DEFAULTS, margin=margin))
                slack = (difference / scale + margin).clamp_min(0)
                th.testing.assert_close(loss, slack.square())
                self.assertEqual(count.item(), 1)
                loss.backward()
                th.testing.assert_close(current.grad, th.tensor([[-1., 1., 0., 0.]]) * (2 * slack / scale))
                self.assertIsNone(reference.grad)

    def test_top1_max_masks_and_metric_denominators(self):
        # At t=1 the episode terminates. Later filled rows must still be ignored.
        batch = dict(
            filled=th.tensor([[[1], [1], [1], [1], [0], [0]]]),
            terminated=th.tensor([[[0], [1], [0], [0], [0], [0]]]),
            avail_actions=th.tensor([[[[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 0]]] * 6]),
        )
        legal, decisions = decision_masks(batch)
        reference = th.tensor([[[[100., 2., 0.], [100., .01, 0.],
                                 [100., 1., 1.], [100., 2., 0.]]] * 5])
        current = th.tensor([[[[1000., 0., 2.], [1000., 2., 0.],
                               [1000., 0., 2.], [1000., 0., 2.]]] * 5], requires_grad=True)
        loss, count, summary = top1_max_preference(current, reference, legal, decisions, TOP1_MAX_DEFAULTS)
        self.assertEqual(count.item(), 2)
        th.testing.assert_close(loss, th.tensor(4.))
        stats = top1_max_statistics(summary.tolist())
        self.assertEqual(stats['decision_states'], 6)
        self.assertEqual(stats['top1_states'], 4)
        self.assertEqual(stats['coverage'], 2 / 6)
        self.assertEqual(stats['top1_flip_rate'], .5)
        self.assertEqual(stats['constrained_top1_flip_rate'], 1.)
        self.assertEqual(stats['margin_violation_rate'], 1.)
        self.assertEqual(stats["valid_pairs"], stats["valid_states"])
        loss.backward()
        expected = th.zeros_like(current)
        expected[0, :2, 0, 1] = -2.
        expected[0, :2, 0, 2] = 2.
        th.testing.assert_close(current.grad, expected)
        loss, count, summary = top1_max_preference(current, reference, legal, decisions & False,
                                                   TOP1_MAX_DEFAULTS)
        self.assertEqual((loss.item(), count.item()), (0, 0))
        self.assertIsNone(top1_max_statistics(summary.tolist())['margin_violation_rate'])

    def test_top1_unfiltered_keeps_low_confidence_and_exact_high_confidence_loss(self):
        legal = th.tensor([[True, True, False]])
        decisions = th.tensor([True])
        current = th.tensor([[0., .1, 100.]], requires_grad=True)
        for old_gap in (.01, .2, 0.):
            with self.subTest(old_gap=old_gap):
                reference = th.tensor([[old_gap, 0., 200.]], requires_grad=True)
                old_loss, old_count, _ = top1_preference(
                    current, reference, legal, decisions, RANK_DEFAULTS)
                loss, count, summary = top1_unfiltered_preference(
                    current, reference, legal, decisions, TOP1_UNFILTERED_DEFAULTS)
                stats = top1_unfiltered_statistics(summary.tolist())
                if old_gap == .01:
                    self.assertEqual((old_count.item(), old_loss.item(), count.item()), (0, 0., 1))
                    th.testing.assert_close(loss, softplus(th.tensor(1.)) * .1)
                    self.assertAlmostEqual(stats["mean_constrained_confidence"], .1)
                    gradient = th.autograd.grad(loss, current)[0]
                    self.assertLess(gradient[0, 0].item(), 0)
                    self.assertGreater(gradient[0, 1].item(), 0)
                    self.assertEqual(gradient[0, 2].item(), 0)
                    self.assertIsNone(reference.grad)
                elif old_gap == .2:
                    self.assertEqual((old_count.item(), count.item()), (1, 1))
                    self.assertTrue(th.equal(loss, old_loss))
                else:
                    self.assertEqual((count.item(), loss.item()), (0, 0.))
                    self.assertIsNone(stats["mean_constrained_confidence"])

    def test_top1_unfiltered_excludes_invalid_decisions(self):
        # Step 1 is the final real decision, step 2 is bootstrap, and steps 3–4 are padding.
        batch = dict(
            filled=th.tensor([[[1], [1], [1], [0], [0]]]),
            terminated=th.tensor([[[0], [1], [0], [0], [0]]]),
            avail_actions=th.tensor([[[[0, 1, 1], [0, 1, 0], [1, 0, 0]]] * 5]),
        )
        legal, decisions = decision_masks(batch)
        reference = th.tensor([[[[100., .01, 0.]] * 3] * 4])
        current = th.tensor([[[[1000., 0., .1]] * 3] * 4], requires_grad=True)
        loss, count, summary = top1_unfiltered_preference(
            current, reference, legal, decisions, TOP1_UNFILTERED_DEFAULTS)
        self.assertEqual(count.item(), 2)
        stats = top1_unfiltered_statistics(summary.tolist())
        self.assertEqual((stats["decision_states"], stats["valid_pairs"], stats["coverage"]), (2, 2, 1.))
        th.testing.assert_close(loss, softplus(th.tensor(1.)) * .1)
        loss.backward()
        self.assertTrue((current.grad[0, :2, 0, 1] < 0).all())
        self.assertTrue((current.grad[0, :2, 0, 2] > 0).all())
        self.assertFalse(current.grad[..., 0].any())
        self.assertFalse(current.grad[0, 2:].any())
        self.assertFalse(current.grad[0, :, 1:].any())
        # Even a filled row following a terminal transition is excluded.
        batch["filled"][0, 3] = 1
        legal, decisions = decision_masks(batch)
        _, count_after_terminal, _ = top1_unfiltered_preference(
            current, reference, legal, decisions, TOP1_UNFILTERED_DEFAULTS)
        self.assertEqual(count_after_terminal.item(), 2)

    def test_topk_correct_pair_sets_expand_with_k_and_match_frpr(self):
        reference = th.tensor([[4., 3., 2., 1., 500.]], requires_grad=True)
        current = th.tensor([[2., 4., 1., 3., 900.]], requires_grad=True)
        legal = th.tensor([[True, True, True, True, False]])
        decisions = th.tensor([True])
        scale = reference.detach()[0, :4].std(unbiased=False)
        expected_pairs = (
            ((0, 1), (0, 2), (0, 3)),
            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3)),
            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        )
        for k, pairs in enumerate(expected_pairs, 1):
            with self.subTest(k=k):
                loss, count, summary = topk_preference(
                    current, reference, legal, decisions, dict(TOPK_DEFAULTS, k=k))
                expected = th.stack([softplus((current[0, j] - current[0, i]) / scale)
                                     for i, j in pairs]).mean()
                self.assertEqual(count.item(), 1)
                self.assertEqual(topk_statistics(summary.tolist())["valid_pairs"], len(pairs))
                th.testing.assert_close(loss, expected)
                actual_grad, old_grad = th.autograd.grad(loss, (current, reference), allow_unused=True)
                self.assertIsNone(old_grad)
                self.assertEqual(actual_grad[0, 4].item(), 0.)
                self.assertGreater(actual_grad.abs().sum().item(), 0.)
                th.testing.assert_close(actual_grad, th.autograd.grad(expected, current)[0])

        old_loss, old_count, old_summary = full_rank_preference(
            current, reference, legal, decisions, RANK_DEFAULTS)
        old_grad = th.autograd.grad(old_loss, current)[0]
        for k in (3, 4):  # Full local and full environment K.
            loss, count, summary = topk_preference(
                current, reference, legal, decisions, dict(TOPK_DEFAULTS, k=k))
            self.assertTrue(th.equal(loss, old_loss))
            self.assertTrue(th.equal(count, old_count))
            self.assertTrue(th.equal(summary[:old_summary.numel()], old_summary))
            self.assertTrue(th.equal(th.autograd.grad(loss, current)[0], old_grad))

    def test_topk_ties_clipping_and_pair_threshold(self):
        reference = th.tensor([[3., 3., 2., 2.9, 500.], [9., 1., 0., 8., 7.],
                               [9., 2., 0., 8., 7.], [9., 2., 0., 8., 7.]])
        legal = th.tensor([[True, True, True, True, False], [False, True, True, False, False],
                           [False, True, False, False, False], [False, True, True, False, False]])
        decisions = th.tensor([True, True, True, False])
        current = th.tensor([[0., 7., 1., 9., 900.], [10., 1., 1., 8., 7.],
                             [9., 2., 0., 8., 7.], [9., 2., 0., 8., 7.]], requires_grad=True)
        for k, competitor_count, mean_k, clip_rate in ((1, 1, 1., 0.), (2, 2, 1.5, .5)):
            with self.subTest(k=k):
                loss, count, summary = topk_preference(
                    current, reference, legal, decisions, dict(TOPK_DEFAULTS, k=k, q_scale_floor=1.))
                stats = topk_statistics(summary.tolist())
                self.assertEqual(count.item(), 2)
                self.assertEqual(stats["valid_pairs"], competitor_count + 1)
                self.assertEqual(stats["mean_effective_k"], mean_k)
                self.assertEqual(stats["k_clip_rate"], clip_rate)
                self.assertEqual(stats["effective_k_histogram"],
                                 {"1": 2} if k == 1 else {"1": 1, "2": 1})
                # Action 0 wins the old-Q tie for K=1. The close pair (0, 3)
                # and the tied pair (0, 1) are excluded. A current tie violates strict order.
                expected = softplus(current[0, 2] - current[0, :k]).mean()
                th.testing.assert_close(loss, (expected + softplus(th.tensor(0.))) / 2)
                self.assertEqual(stats["pair_violation_rate"], 2 / (competitor_count + 1))
                self.assertEqual(stats["top1_flip_rate"], 0.)
                grad = th.autograd.grad(loss, current)[0]
                self.assertFalse(grad[2:].any())
                self.assertFalse(grad[0, 3:].any())
                self.assertEqual(grad[0, 1].item() == 0, k == 1)

    def test_effective_k_histogram_excludes_decisions_without_a_valid_pair(self):
        reference = th.tensor([[4., 0., 1., 2.], [1., 1., 1., 1.], [2., 0., 1., 2.]])
        legal = th.tensor([[True, True, True, True], [True, True, True, True],
                           [True, True, False, False]])
        _, _, summary = topk_preference(reference, reference, legal, th.ones(3, dtype=th.bool),
                                        dict(TOPK_DEFAULTS, k=5))
        stats = topk_statistics(summary.tolist())
        self.assertEqual(stats["decision_states"], 3)
        self.assertEqual(stats["valid_states"], 2)
        self.assertEqual(stats["effective_k_histogram"], {"1": 1, "2": 0, "3": 1})

    def test_action_set_exploration_defaults_and_history(self):
        raw = json.loads(PROTOCOL_PATH.read_text())
        raw.pop("epsilon_on_switch", None)
        for requested in (None, "per_task", "cumulative_no_restart"):
            protocol = dict(raw)
            if requested is not None:
                protocol["epsilon_on_switch"] = requested
            expected = requested or "per_task"
            with self.subTest(mode=requested), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "protocol.json"
                path.write_text(json.dumps(protocol))
                preview = json.loads(subprocess.check_output([
                    sys.executable, "scripts/run.py", "--suite", "vdn_action_set",
                    "--algorithm", "vdn", "--method", "topk_preference", "--seed", "1",
                    "--protocol", str(path), "--run-id", "epsilon_config_test", "--dry-run",
                ], cwd=driver.ROOT, text=True))["details"]["protocol"]
                validated = driver.validate(protocol, "vdn", "topk_preference")
                self.assertEqual(preview["epsilon_on_switch"], expected)
                self.assertEqual(validated["epsilon_on_switch"], expected)
                self.assertEqual(driver.config(preview, 1, Path(tmp))["epsilon_on_switch"], expected)
                validated.pop("epsilon_on_switch")
                self.assertEqual(driver.config(validated, 1, Path(tmp))["epsilon_on_switch"], "per_task")

    def test_agent_gradient_metrics_use_weighted_gradients_without_accumulation(self):
        agent = th.nn.Linear(2, 1, bias=True)
        mixer = th.nn.Parameter(th.tensor(2.))
        agent.weight.grad = th.ones_like(agent.weight)
        td = (agent.weight * th.tensor([[3., 4.]])).sum() * mixer
        pref = 2 * (agent.weight * th.tensor([[-3., -4.]])).sum()
        expected = dict(td_grad_norm_agent=10., preference_grad_norm_agent=10.,
                        preference_td_grad_ratio=1., td_preference_grad_cosine=-1.,
                        preference_grad_nonzero_agent=True)
        self.assertEqual(agent_gradient_metrics(agent, td, pref), expected)
        self.assertTrue(th.equal(agent.weight.grad, th.ones_like(agent.weight)))
        self.assertIsNone(agent.bias.grad)
        self.assertIsNone(mixer.grad)
        missing = agent_gradient_metrics(agent, td, None)
        self.assertEqual(missing.pop("td_grad_norm_agent"), 10.)
        self.assertTrue(all(value is None for value in missing.values()))
        self.assertIsNone(agent_gradient_metrics(agent, td, mixer)["preference_grad_norm_agent"])
        self.assertIsNone(agent_gradient_metrics(agent, td, pref * th.inf)["preference_grad_norm_agent"])
        contrast = agent.weight.sum()
        nonzero_loss = 1. + contrast * 0.
        zero_loss = contrast - contrast.detach()
        self.assertEqual(nonzero_loss.item(), 1.)
        self.assertFalse(agent_gradient_metrics(agent, td, nonzero_loss)["preference_grad_nonzero_agent"])
        self.assertEqual(zero_loss.item(), 0.)
        self.assertTrue(agent_gradient_metrics(agent, td, zero_loss)["preference_grad_nonzero_agent"])
        (td + pref).backward()
        self.assertIsNotNone(mixer.grad)

    def test_k_ablation_configs_start_identically_and_monitoring_preserves_updates(self):
        configs = driver.ROOT / "configs/continual/action_preference/szc"
        original_config = driver.config

        def tiny_config(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": ScriptedProtoss}
        ), patch.object(driver, "config", tiny_config), patch.object(
            th.cuda, "is_available", return_value=False
        ), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            first_stage = None
            for k in (1, 2, 3, 5, None):
                tag = f"tkpr_k{k}" if k is not None else "frpr"
                method = "topk_preference" if k is not None else "full_rank_preference"
                protocol = json.loads((configs / f"szc_vdn64_{tag}.json").read_text())
                self.assertEqual(protocol[method].get("k"), k)
                self.assertEqual(protocol["gradient_monitor_interval"], 100)
                for stage in protocol["stages"]:
                    stage["steps"] = 306
                protocol["evaluation"].update(seeds=[3, 4], episodes_per_seed=1, interval=0,
                                              focus_seeds=None, focus_episodes_per_seed=None)
                protocol["action_set"].update(success_episodes=2, seed_offsets=[101, 102],
                                              attempts_per_seed=2, batch_episodes=4)
                checkpoints = []
                for interval in (0, 100):
                    run = Path(tmp) / f"{tag}_{interval}"
                    run.mkdir()
                    configured = driver.validate(dict(protocol, gradient_monitor_interval=interval), "vdn", method)
                    driver.train_sequence(configured, 1111, run)
                    checkpoints.append(sorted((run / "models").glob("phase*")))
                    updates = jsonlines(run / f"{method}_updates.jsonl")
                    phase0 = [row for row in updates if row["phase"] == 0]
                    self.assertTrue(phase0)
                    self.assertTrue(all(row["td_loss"] == row["total_loss"] and not row["historical_tasks"]
                                        for row in phase0))
                    metrics = jsonlines(run / "experiment_metrics.jsonl")
                    monitor = [row for row in metrics if row["context"].get("event") == "gradient_monitor"]
                    self.assertEqual(len(monitor), len(updates) // 100 * 5 if interval else 0)
                    flags = {row["context"]["learner_update"]: row["value"] for row in metrics
                             if row["metric"] == "preference_loss_nonzero"}
                    self.assertEqual(flags, {i: float(row["preference_loss"] != 0)
                                            for i, row in enumerate(updates, 1)})
                    if interval:
                        auxiliary = [row for row in monitor if row["metric"] == "preference_grad_norm_agent"]
                        self.assertEqual([row["context"]["learner_update"] for row in auxiliary],
                                         list(range(100, len(updates) + 1, 100)))
                        observed = {row["context"]["learner_update"]: row["value"] for row in monitor
                                    if row["metric"] == "preference_grad_nonzero_agent"}
                        self.assertEqual(observed, {
                            row["context"]["learner_update"]: (None if row["value"] is None
                                                              else float(row["value"] > 0))
                            for row in auxiliary})
                        self.assertIsNone(auxiliary[0]["value"])
                        self.assertTrue(any(row["value"] is not None for row in auxiliary[1:]))
                    if k is not None:
                        counts = [row["value"] for row in metrics
                                  if row["metric"] == "effective_k_histogram"
                                  and row["context"].get("scope") == "update"]
                        self.assertEqual(len(counts), len(updates))
                        self.assertEqual(counts[0], {})
                        self.assertTrue(any(sum(hist.values()) for hist in counts))
                self.assertEqual(len(checkpoints[0]), 3)
                self.assertEqual(len(checkpoints[1]), 3)
                for left, right in zip(*checkpoints):
                    self.assertEqual(left.name, right.name)
                    for name in ("agent.th", "opt.th"):
                        th.testing.assert_close(th.load(left / name), th.load(right / name), rtol=0, atol=0)
                agent = th.load(checkpoints[1][0] / "agent.th")
                if first_stage is None:
                    first_stage = agent
                th.testing.assert_close(agent, first_stage, rtol=0, atol=0)

    def test_topk_configuration_rejects_invalid_k(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        for k in (True, False, 1., 2.5, 0, -1, "2"):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "topk_preference.k"):
                driver.validate(dict(protocol, topk_preference={"k": k}), "vdn", "topk_preference")
        protocol = driver.validate(dict(protocol, topk_preference={"k": 11}), "vdn", "topk_preference")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": ScriptedProtoss}
        ):
            out = Path(tmp)
            with self.assertRaisesRegex(ValueError, "n_actions - 1"):
                driver.train_sequence(protocol, 8, out)
            self.assertEqual(json.loads((out / "status.json").read_text())["status"], "failed")

    def test_temperature_and_config_validation(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        protocol["action_set"].pop("delta_by_task")
        with self.assertRaisesRegex(ValueError, "action_set.delta_by_task"):
            driver.validate(protocol, "vdn", "aspr_preference")
        protocol["action_set"]["second_gap_quantile"] = 0.5
        for method in RANK_METHODS:
            effective = driver.validate(protocol, "vdn", method)
            defaults = METHOD_DEFAULTS.get(method, RANK_DEFAULTS)
            self.assertEqual(effective[method], defaults)
            for key in {"temperature", "margin", "min_gap", "min_gap_raw", "q_scale_floor"} - defaults.keys():
                with self.assertRaisesRegex(ValueError, "unknown or malformed"):
                    driver.validate(dict(protocol, **{method: {key: 1.0}}), "vdn", method)
            invalid = {"temperature": 0, "min_gap": -0.1, "min_gap_raw": -0.1,
                       "q_scale_floor": 0, "weight": True, "margin": -.1}
            for key, value in invalid.items():
                if key not in defaults:
                    continue
                with self.subTest(method=method, key=key):
                    with self.assertRaisesRegex(ValueError, f"{method}.{key}"):
                        driver.validate(dict(protocol, **{method: {key: value}}), "vdn", method)
        legal = th.ones((1, 2), dtype=th.bool)
        decisions = th.tensor([True])
        q = th.tensor([[1.0, -1.0]])
        for rank_loss in (full_rank_preference, top1_preference):
            sharp = rank_loss(q, q, legal, decisions, dict(RANK_DEFAULTS, temperature=0.5))[0]
            soft = rank_loss(q, q, legal, decisions, dict(RANK_DEFAULTS, temperature=2.0))[0]
            self.assertLess(sharp.item(), soft.item())

    def test_shared_replay_and_joint_update(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        for stage in protocol["stages"]:
            stage["steps"] = 16
        protocol["evaluation"].update(seeds=[3, 4], episodes_per_seed=1, interval=0)
        protocol["action_set"].update(batch_episodes=4, success_episodes=2,
                                      seed_offsets=[101, 102], attempts_per_seed=2, probe_episodes_per_seed=1)
        original_config = driver.config
        def tiny_config(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": ScriptedProtoss}
        ), patch.object(driver, "config", tiny_config), patch.object(
            th.cuda, "is_available", return_value=False
        ), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            out = Path(tmp)
            for method in ("baseline", "action_set_preference", "aspr_max_preference", *RANK_METHODS, "zero"):
                run = out / method
                run.mkdir()
                active = "top1_preference" if method == "zero" else method
                configured = copy.deepcopy(protocol)
                if active == "tkpr_refresh":
                    configured["action_set"]["refresh_success_episodes"] = 1
                if active in RANK_METHODS:
                    configured[active] = dict(weight=0 if method == "zero" else 0.5)
                    defaults = METHOD_DEFAULTS.get(active, RANK_DEFAULTS)
                    for gap_key in {"min_gap", "min_gap_raw"} & defaults.keys():
                        configured[active][gap_key] = 0
                    if active == "top1_max_preference":
                        configured[active]["margin"] = 1.
                driver.train_sequence(driver.validate(configured, "vdn", active), 8, run)
                self.assertEqual(json.loads((run / "status.json").read_text())["status"], "completed")

            baseline = sorted((out / "baseline" / "models").glob("*/agent.th"))
            zero = sorted((out / "zero" / "models").glob("*/agent.th"))
            self.assertEqual(len(baseline), len(zero))
            self.assertFalse((out / "zero" / "action_set_memory").exists())
            for a, b in zip(baseline, zero):
                first, second = th.load(a), th.load(b)
                self.assertTrue(all(th.equal(first[k], second[k]) for k in first))

            aspr = out / "action_set_preference"
            original = th.load(aspr / "action_set_memory/phase0000/success.th")
            original_samples = jsonlines(aspr / "action_set_updates.jsonl")
            max_run = out / "aspr_max_preference"
            for old_path in sorted((aspr / "models").glob("*/agent.th")):
                first, second = th.load(old_path), th.load(max_run / old_path.relative_to(aspr))
                self.assertTrue(all(th.equal(first[k], second[k]) for k in first))
            for run in (aspr, max_run):
                self.assertEqual([row["preference_loss"] for row in original_samples],
                                 [row["preference_loss"] for row in jsonlines(run / "aspr_max_preference_updates.jsonl")])
            refreshed = th.load(out / "tkpr_refresh/action_set_memory/phase0000/success.th")
            self.assertEqual(len(refreshed["episodes"]), 2)
            self.assertEqual(sum(ep["reference_phase"] in (0, 1) for ep in refreshed["episodes"]), 1)
            self.assertEqual(sum(ep["reference_phase"] == 2 for ep in refreshed["episodes"]), 1)
            for method in RANK_METHODS:
                if method == "tkpr_refresh":
                    continue
                run = out / method
                pool = th.load(run / "action_set_memory/phase0000/success.th")
                self.assertEqual(len(original["episodes"]), len(pool["episodes"]))
                for old, new in zip(original["episodes"], pool["episodes"]):
                    self.assertEqual((old["seed"], old["index"]), (new["seed"], new["index"]))
                    self.assertTrue(all(th.equal(v, new["data"][k]) for k, v in old["data"].items()))
                    if method == "aspr_preference":
                        self.assertTrue(th.equal(old["good_mask"], new["good_mask"]))
                        self.assertEqual(pool["delta"], original["delta"])
                    elif method in ASPR_GROUP_DEFAULTS:
                        self.assertEqual(new["group_ids"].shape, new["data"]["avail_actions"].shape)
                        self.assertTrue((new["group_ids"][-1] == -1).all())
                    else:
                        self.assertNotIn("good_mask", new)
                    self.assertNotIn("old_q_values", old["data"])
                    self.assertEqual(new["data"]["old_q_values"].shape, new["data"]["avail_actions"].shape)
                log_name = "tkpr_cpr_preference" if method == "tkpr_cpr" else method
                samples = jsonlines(run / f"{log_name}_updates.jsonl")
                if method in ASPR_GROUP_DEFAULTS:
                    restored = PreferenceHistory()
                    restored.restore_success(run, [0], require_old_q=True, require_group=True)
                    th.testing.assert_close(restored[0]["episodes"][0]["group_ids"],
                                            pool["episodes"][0]["group_ids"])
                    for name in ("aspr_group_summary.csv", "aspr_group_distribution.csv",
                                 "aspr_group_training_sampled.csv"):
                        self.assertTrue((run / name).is_file())
                    self.assertTrue(all("group_size_distribution" in historical
                                        for row in samples for historical in row["historical_tasks"]))
                if method == "aspr_preference":
                    fields = {"candidate_pairs", "valid_pairs", "pair_keep_rate", "pair_violation_rate",
                              "average_good_set_size", "singleton_ratio", "all_action_ratio"}
                    self.assertTrue(all(fields <= row["pair_filter_stats"].keys() for row in samples))
                    metrics = jsonlines(run / "experiment_metrics.jsonl")
                    self.assertTrue(fields <= {row["metric"] for row in metrics})
                    self.assertTrue(all(fields <= probes.keys()
                                        for probes in jsonlines(run / "aspr_preference_probes.jsonl")))
                    self.assertFalse((run / "action_set_probe_errors.jsonl").exists())
                    restored = PreferenceHistory()
                    restored.restore_success(run, [0], require_old_q=True, require_good=True)
                    self.assertTrue(th.equal(restored[0]["episodes"][0]["good_mask"],
                                             pool["episodes"][0]["good_mask"]))
                    invalid = copy.deepcopy(pool)
                    invalid["episodes"][0]["data"].pop("old_q_values")
                    with patch.object(PreferenceHistory, "load", return_value=invalid):
                        with self.assertRaisesRegex(ValueError, "old_q_values"):
                            restored.restore_success(run, [0], require_old_q=True, require_good=True)
                    invalid = copy.deepcopy(pool)
                    invalid["episodes"][0].pop("good_mask")
                    with patch.object(PreferenceHistory, "load", return_value=invalid):
                        with self.assertRaisesRegex(ValueError, "good_mask"):
                            restored.restore_success(run, [0], require_old_q=True, require_good=True)
                for phase, tasks in enumerate((set(), {"stalker"}, {"stalker", "zealot"})):
                    phase_rows = [row for row in samples if row["phase"] == phase]
                    aspr_rows = [row for row in original_samples if row["phase"] == phase]
                    self.assertTrue(phase_rows)
                    self.assertEqual(len(phase_rows), len(aspr_rows))
                    for row, aspr_row in zip(phase_rows, aspr_rows):
                        history = row["historical_tasks"]
                        self.assertEqual({item["task"] for item in history}, tasks)
                        self.assertEqual([(item["task"], item["episode_ids"]) for item in history],
                                         [(item["task"], item["episode_ids"]) for item in aspr_row["historical_tasks"]])
                        for item in history:
                            if method == "top1_max_preference":
                                self.assertEqual(item["valid_pairs"], item["valid_states"])
                                self.assertTrue(0 <= item["constrained_top1_flip_rate"] <= 1)
                                self.assertTrue(0 <= item["margin_violation_rate"] <= 1)
                            else:
                                self.assertGreater(item["valid_pairs"], 0)
                            self.assertEqual(item["valid_states"], (5 + 4) * item["sampled_episodes"])
                            self.assertEqual(item["coverage"], 1)
                            self.assertTrue(0 <= item["top1_flip_rate"] <= 1)
                        mean = sum(item["loss"] for item in history) / len(tasks) if tasks else 0
                        self.assertAlmostEqual(row["preference_loss"], mean, places=6)
                        self.assertEqual(row["preference_participated_update"], bool(tasks))
                        self.assertEqual(row["historical_episode_count"],
                                         sum(item["sampled_episodes"] for item in history))
                        self.assertEqual(row["decision_states"], sum(item["decision_states"] for item in history))
                        self.assertEqual(row["nonzero_loss_states"],
                                         sum(item["nonzero_loss_states"] for item in history))
                        self.assertEqual(row["valid_pairs"], sum(item["valid_pairs"] for item in history))
                        self.assertAlmostEqual(row["preference_participation_rate"],
                                               row["preference_participating_updates"] /
                                               row["preference_recorded_updates"])
                        self.assertAlmostEqual(row["total_loss"], row["td_loss"] + 0.5 * mean, places=6)
                if method == "top1_max_preference":
                    self.assertTrue(any(row["preference_loss"] > 0 for row in samples))
                    normalized = jsonlines(run / "experiment_metrics.jsonl")
                    self.assertTrue({"preference_loss", "valid_states", "coverage", "top1_flip_rate",
                                     "constrained_top1_flip_rate", "margin_violation_rate"}
                                    <= {row["metric"] for row in normalized})
                if method == "top1_unfiltered_preference":
                    self.assertTrue(any(row["preference_loss"] > 0 for row in samples))
                    self.assertTrue(all(item["mean_constrained_confidence"] > 0
                                        for row in samples for item in row["historical_tasks"]))
                    normalized = jsonlines(run / "experiment_metrics.jsonl")
                    self.assertTrue(any(row["metric"] == "mean_constrained_confidence" for row in normalized))
                if method == "topk_preference":
                    self.assertTrue(any(row["preference_loss"] > 0 for row in samples))
                    normalized = jsonlines(run / "experiment_metrics.jsonl")
                    self.assertTrue({"valid_states", "valid_pairs", "coverage", "pair_violation_rate",
                                     "top1_flip_rate", "mean_effective_k", "k_clip_rate"}
                                    <= {row["metric"] for row in normalized})
                if method in ("full_rank_preference", "full_rank_preference_unfiltered",
                              "full_rank_preference_raw", "full_rank_preference_raw_loss"):
                    normalized = jsonlines(run / "experiment_metrics.jsonl")
                    for row in samples:
                        historical = row["historical_tasks"]
                        totals = row["pair_filter_stats"]
                        for name, value in totals.items():
                            update_values = [metric["value"] for metric in normalized
                                             if metric["metric"] == name
                                             and metric["context"].get("scope") == "update"
                                             and metric["t_env"] == row["t_env"]]
                            self.assertEqual(update_values, [value])
                        for item in historical:
                            self.assertGreater(item["candidate_pairs"], 0)
                            self.assertEqual(item["tie_pairs"] + item["eligible_pairs"], item["candidate_pairs"])
                            self.assertEqual(item["filtered_pairs"] + item["valid_pairs"], item["eligible_pairs"])
                            if method == "full_rank_preference_unfiltered":
                                self.assertEqual(item["filtered_pairs"], 0)
                                self.assertEqual(item["pair_filter_rate"], 0.)
                            values = {metric["metric"]: metric["value"] for metric in normalized
                                      if metric["context"].get("scope") == "historical_task"
                                      and metric["context"]["source_task"] == item["task"]
                                      and metric["t_env"] == row["t_env"]}
                            self.assertEqual(values, {name: item[name] for name in totals})
                rank_checkpoints = sorted((run / "models").glob("*/agent.th"))
                initial, same = th.load(rank_checkpoints[0]), th.load(baseline[0])
                self.assertTrue(all(th.equal(initial[k], same[k]) for k in initial))
                trained, control = th.load(rank_checkpoints[1]), th.load(baseline[1])
                self.assertFalse(th.equal(trained["rnn.weight_hh"], control["rnn.weight_hh"]))

    def test_saved_q_matches_rollout_and_pads_for_backward(self):
        class VariableWins(ScriptedProtoss):
            def step(self, actions):
                self.t += 1
                done = self.t >= self.episodes + 1
                return float(done), done, dict(battle_won=done, episode_limit=False)

        class TracedRunner(driver.base.EpisodeRunner):
            def setup(self, scheme, groups, preprocess, mac):
                super().setup(scheme, groups, preprocess, mac)
                forward = mac.forward

                def trace(batch, t, test_mode=False):
                    q = forward(batch, t, test_mode=test_mode)
                    observed.append((t, q[0].detach().clone()))
                    return q

                mac.forward = trace

        observed = []
        protocol = driver.validate(json.loads(PROTOCOL_PATH.read_text()), "vdn", "top1_preference")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": VariableWins}
        ), patch.object(driver.base, "EpisodeRunner", TracedRunner), patch.object(
            th.cuda, "is_available", return_value=False
        ), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            out = Path(tmp)
            args = SimpleNamespace(**driver.config(protocol, 8, out), n_agents=5, n_actions=11)
            probe = TracedRunner(args, driver.base.Logger(logging.getLogger("test")))
            spec = driver.episode_spec(probe.get_env_info())
            expected_layout = driver.layout(probe)
            probe.close_env()
            scheme, groups = driver.augmented_scheme(spec), spec[1]
            mac = driver.base.controllers[args.mac](scheme, groups, args)
            settings = dict(protocol["action_set"], seed_offsets=[101], attempts_per_seed=2,
                            success_episodes=2)
            checkpoint = out / "source_checkpoint"
            driver.collect_memory(args, spec, expected_layout, mac.agent.state_dict(),
                                  checkpoint, "stalker", 0, settings, out)
            pool = th.load(out / "action_set_memory/phase0000/success.th", map_location="cpu")
            self.assertEqual(pool["reference_checkpoint"], "source_checkpoint/agent.th")
            episodes = pool["episodes"]
            self.assertEqual([ep["length"] for ep in episodes], [2, 3])
            cursor = 0
            for ep in episodes:
                size = ep["length"] + 1
                captured = observed[cursor:cursor + size]
                self.assertEqual([t for t, _ in captured], list(range(size)))
                stored = ep["data"]["old_q_values"]
                self.assertEqual(stored.dtype, th.float32)
                self.assertFalse(stored.requires_grad)
                self.assertTrue(th.equal(stored, th.stack([q for _, q in captured])))
                cursor += size
            self.assertEqual(cursor, len(observed))

            observed.clear()
            args.method = "tkpr_refresh"
            refreshed = driver.collect_memory(args, spec, expected_layout, mac.agent.state_dict(),
                                              checkpoint, "stalker", 1, settings, out,
                                              target=2, persist=False)
            self.assertNotIn("reference_checkpoint", refreshed)
            self.assertEqual(len(refreshed["episodes"]), 2)
            cursor = 0
            for ep in refreshed["episodes"]:
                self.assertEqual((ep["reference_phase"], ep["reference_checkpoint"]),
                                 (1, "source_checkpoint/agent.th"))
                size = ep["length"] + 1
                self.assertTrue(th.equal(ep["data"]["old_q_values"],
                                         th.stack([q for _, q in observed[cursor:cursor + size]])))
                cursor += size
            self.assertEqual(cursor, len(observed))

            batch, _ = batch_episodes(episodes, scheme, groups, "cpu",
                                      include_good=False, include_old_q=True)
            self.assertEqual(batch["old_q_values"].shape, (2, 4, 5, 11))
            for i, ep in enumerate(episodes):
                length = ep["length"]
                self.assertTrue(th.equal(batch["old_q_values"][i, :length + 1],
                                         ep["data"]["old_q_values"]))
                self.assertFalse(batch["old_q_values"][i, length + 1:].any())
            legal, decisions = decision_masks(batch)
            self.assertFalse(decisions[0, 2].any())
            self.assertFalse(decisions[1, 1, 4])
            self.assertTrue(decisions[0, 1, 0])
            self.assertFalse(legal[0, 0, 0, 0])
            good, _ = make_good_mask(batch, batch["old_q_values"][:, :-1], 0)
            for ep, mask in zip(episodes, good):
                ep["good_mask"] = mask[:ep["length"] + 1]
            for ep in episodes:
                ep["group_ids"] = driver.label_group_memory([ep], scheme, groups, 4, .1)[0]["group_ids"]
            for method in RANK_METHODS:
                defaults = METHOD_DEFAULTS.get(method, RANK_DEFAULTS)
                settings = dict(defaults)
                for gap_key in {"min_gap", "min_gap_raw"} & settings.keys():
                    settings[gap_key] = 0
                preference = RankingPreference(method, settings, None, scheme, groups, None)
                used_episodes = episodes
                if method == "tkpr_cpr":
                    from drivers.continual.memory.cpr import initialize_pool
                    used_episodes = copy.deepcopy(episodes)
                    initialize_pool(dict(pool, episodes=used_episodes))
                mac.agent.zero_grad()
                with patch.object(th, "load", side_effect=AssertionError("preference loaded a checkpoint")):
                    loss, count, _ = preference.task_loss(mac, pool, used_episodes)
                    self.assertTrue(th.isfinite(loss))
                    self.assertGreater(count.item(), 0)
                    self.assertIsNone(mac.hidden_states)
                    loss.backward()
                gradient = mac.agent.rnn.weight_hh.grad
                self.assertTrue(th.isfinite(gradient).all())
                if method not in ("full_rank_preference_relu", "top1_max_preference"):
                    self.assertGreater(gradient.abs().sum().item(), 0)

    def test_keep_branch_matches_uninterrupted_sequence(self):
        self.check_keep_branch_matches_uninterrupted_sequence("full_rank_preference")

    def test_topk_keep_branch_restores_history_after_two_stages(self):
        self.check_keep_branch_matches_uninterrupted_sequence("topk_preference")

    def check_keep_branch_matches_uninterrupted_sequence(self, method):
        raw = json.loads(PROTOCOL_PATH.read_text())
        revisit = dict(task="stalker", steps=16)
        if method == "topk_preference":
            raw["stages"][-1] = revisit
        else:
            raw["stages"].append(revisit)
        last_phase = len(raw["stages"]) - 1
        historical_tasks = {stage["task"] for stage in raw["stages"][:-1]} - {revisit["task"]}
        for stage in raw["stages"]:
            stage["steps"] = 16
        raw.update(phase_boundary="cumulative", epsilon_on_switch="cumulative_no_restart")
        raw["evaluation"].update(seeds=[3, 4], episodes_per_seed=1, interval=0)
        raw["action_set"].update(batch_episodes=4, success_episodes=2,
                                 seed_offsets=[101, 102], attempts_per_seed=2)
        raw[method] = dict(min_gap=0)
        protocol = driver.validate(raw, "vdn", method)

        class ForkableScriptedProtoss(ScriptedProtoss):
            def distribution_state(self):
                return self.episodes

            def restore_distribution(self, episodes):
                self.episodes = episodes

        original_config = driver.config

        def tiny_config(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2, target_update_interval=3)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": ForkableScriptedProtoss}
        ), patch.object(driver, "config", tiny_config), patch.object(
            th.cuda, "is_available", return_value=False
        ), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            full, prefix, keep, release = [Path(tmp) / name for name in ("full", "prefix", "keep", "release")]
            for run in (full, prefix, keep, release):
                run.mkdir()
            driver.train_sequence(protocol, 8, full)
            driver.train_sequence(protocol, 8, prefix, role="prefix")
            for branch in (keep, release):
                driver.train_sequence(protocol, 8, branch, role=branch.name, parent=prefix)
                restored = json.loads((branch / "branch_restore.json").read_text())
                self.assertEqual(restored["model_hash"], restored["parent_model_hash"])
                transitions = jsonlines(branch / "transitions.jsonl")
                self.assertEqual(len(transitions), 1)
                self.assertEqual(transitions[0]["phase"], last_phase)
                self.assertEqual(transitions[0]["start"], restored["t_env"])
                self.assertEqual(transitions[0]["target"], 16 * (last_phase + 1))
            for name in ("agent.th", "opt.th"):
                continued = th.load(next((full / "models").glob(f"phase{last_phase:04d}_*/{name}")))
                branched = th.load(next((keep / "models").glob(f"phase{last_phase:04d}_*/{name}")))
                self.assertEqual(digest(continued), digest(branched))
            keep_rows = jsonlines(keep / f"{method}_updates.jsonl")
            self.assertTrue(keep_rows)
            self.assertTrue(all({entry["task"] for entry in row["historical_tasks"]} ==
                                historical_tasks for row in keep_rows))
            self.assertFalse((release / f"{method}_updates.jsonl").exists())
            self.assertFalse((release / "action_set_memory").exists())


if __name__ == "__main__":
    unittest.main()
