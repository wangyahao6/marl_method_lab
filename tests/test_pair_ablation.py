"""Fixed ranking masks and shared replay through the VDN learner."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch as th
from torch.nn.functional import softplus

from test_action_set_preference import driver, jsonlines
from test_shared_tkpr_k1 import ForkableProtoss
from action_set_protocol import PAIR_DEFAULTS, PAIR_RATIOS
from learners.pair_ablation import STRUCTURES, ablation_loss, ablation_statistics, ratio_counts, select_pairs
from learners.ranking_preference import full_rank_preference, topk_preference


CONFIGS = Path(__file__).resolve().parents[1] / "configs/continual/action_preference"
SEED = 3535


def protocol(experiment):
    return json.loads((CONFIGS / experiment / f"szc_vdn64_{experiment}_1m.json").read_text())


def pair_ids(mask, action_count):
    return {tuple(indices) for indices in th.triu_indices(action_count, action_count, 1)[:, mask].T.tolist()}


def strict_closure(pairs, old):
    directed = {(a, b) if old[a] > old[b] else (b, a) for a, b in pairs}
    reach = set(directed)
    for mid in range(len(old)):
        reach |= {(a, b) for a, step in reach for dest, b in reach if step == mid == dest}
    return reach


class FourActionProtoss(ForkableProtoss):
    def get_avail_actions(self):
        availability = super().get_avail_actions()
        for agent in availability:
            if not agent[0]:
                agent[7] = 1
        return availability


class PairAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def test_structure_sets_and_actual_transitive_closures(self):
        old = th.tensor([[9., 1., 5., 7., 3., 100.]]).repeat(4, 1)
        legal = th.tensor([[True, True, True, True, True, False]]).repeat(4, 1)
        old[1, 1] = old[1, 2]
        legal[2, 3:5] = False
        decisions = th.tensor([True, True, True, False])
        adjacent = {(0, 3), (2, 3), (2, 4), (1, 4)}
        k1 = {(0, i) for i in range(1, 5)}
        full = {(i, j) for i in range(5) for j in range(i + 1, 5)}
        expected = dict(adjacent=adjacent, k1=k1, k1_adjacent=k1 | adjacent, frpr=full)

        for method in STRUCTURES:
            with self.subTest(method=method):
                mask, valid, counts, audit = select_pairs(old, legal, decisions, method, SEED)
                self.assertEqual(valid.tolist(), [True, False, False, False])
                self.assertFalse(mask[1:].any())
                self.assertEqual(counts[0, :2].tolist(), [10, 10])
                pairs = pair_ids(mask[0], 6)
                if method in expected:
                    self.assertEqual(pairs, expected[method])
                else:
                    self.assertEqual(len(pairs), 4)
                    self.assertTrue(pairs <= full)
                    th.testing.assert_close(mask, select_pairs(old, legal, decisions, method, SEED)[0])
                closure = strict_closure(pairs, old[0])
                self.assertEqual(audit["complete_order_states"], int(len(closure) == len(full)))
                if method in ("adjacent", "k1_adjacent"):
                    self.assertEqual(closure, strict_closure(full, old[0]))
                if method == "k1":
                    self.assertEqual(len(closure), 4)

    def test_ratios_same_states_exact_budget_and_rounding(self):
        old = th.tensor([[5., 0., 2., 2., 3., 4.],
                         [5., 0., 2., 3., 4., 1.],
                         [5., 5., 2., 3., 4., 1.],
                         [5., 0., 2., 2., 4., 1.],
                         [5., 0., 2., 3., 4., 1.],
                         [5., 0., 2., 3., 4., 1.]])
        legal = th.ones_like(old, dtype=th.bool)
        legal[0, 4:] = False  # Four actions with a lower-rank tie cannot fit three non-Top-1 pairs.
        legal[1, 4:] = False
        legal[4, 3:] = False
        decisions = th.tensor([True, True, True, True, True, False])
        budgets = th.tensor([0, 3, 0, 5, 0, 0])
        for percent in PAIR_RATIOS.values():
            with self.subTest(percent=percent):
                mask, valid, counts, audit = select_pairs(old, legal, decisions, percent, SEED)
                th.testing.assert_close(valid, budgets > 0)
                th.testing.assert_close(mask.sum(-1), budgets)
                self.assertEqual(audit["budget_mismatch_states"], 0)
                self.assertEqual(int(counts[:, 2].sum()), (percent * int(budgets.sum()) + 50) // 100)
                self.assertTrue(((counts[:, 2] - budgets * percent / 100).abs() < 1).all())
                for state in (1, 3):
                    pairs = pair_ids(mask[state], 6)
                    self.assertTrue(all(legal[state, i] and legal[state, j] and old[state, i] != old[state, j]
                                        for i, j in pairs))
                    self.assertEqual(int(counts[state, 2]), sum(0 in pair for pair in pairs))
                    if percent == 100:
                        self.assertEqual(pairs, {(0, i) for i in range(1, int(budgets[state]) + 1)})
                    if percent == 0:
                        self.assertTrue(all(0 not in pair for pair in pairs))
                th.testing.assert_close(mask, select_pairs(old, legal, decisions, percent, SEED)[0])

    def test_rounding_carries_across_history_pools(self):
        budget = th.tensor([3, 5, 0, 0, 0])
        generator = th.Generator().manual_seed(SEED)
        first = ratio_counts(budget, 20, generator, prior_budget=0)
        second = ratio_counts(budget, 20, generator, prior_budget=int(budget.sum()))
        self.assertEqual(int(first.sum()), 2)
        self.assertEqual(int((first + second).sum()), 3)
        self.assertTrue(((second - budget * .2).abs() < 1).all())

    def test_softplus_and_equal_state_weight_with_finite_gradients(self):
        old = th.tensor([[0., .001, .002, .003, .004], [1., 4., 3., 0., 2.]], requires_grad=True)
        current = th.tensor([[4., 2., -1., 5., 9.], [5., 4., 3., 2., 1.]], requires_grad=True)
        legal = th.ones_like(old, dtype=th.bool)
        legal[1, 4] = False
        decisions = th.tensor([True, True])
        for method, legacy in (("k1", topk_preference), ("frpr", full_rank_preference), (100, topk_preference)):
            mask, _, counts, _ = select_pairs(old, legal, decisions, method, SEED)
            loss, count, summary = ablation_loss(current, old, legal, decisions, PAIR_DEFAULTS, mask, counts)
            expected = []
            for i in range(2):
                std = old[i, legal[i]].detach().std(unbiased=False).clamp_min(.1)
                penalties = [softplus((current[i, b] - current[i, a]) * (old[i, a] - old[i, b]).detach().sign() / std)
                             for a, b in pair_ids(mask[i], 5)]
                expected.append(th.stack(penalties).mean())
            th.testing.assert_close(loss, th.stack(expected).mean())
            th.testing.assert_close(loss, legacy(current, old, legal, decisions, dict(PAIR_DEFAULTS, k=1))[0])
            self.assertEqual(int(count), 2)
            gradient, old_grad = th.autograd.grad(loss, (current, old), allow_unused=True)
            self.assertIsNone(old_grad)
            self.assertTrue(gradient.isfinite().all())
            self.assertGreater(float(gradient.norm()), 0)
            self.assertEqual(float(gradient[1, 4]), 0)
            self.assertEqual(ablation_statistics(summary.tolist())["coverage"], 1.)

    def test_reference_and_shared_branch_use_prepared_masks(self):
        def tiny_protocol(experiment, method):
            configured = protocol(experiment)
            for stage in configured["stages"]:
                stage["steps"] = 16
            configured["evaluation"].update(seeds=[3, 4], episodes_per_seed=1,
                                             focus_seeds=None, focus_episodes_per_seed=None, interval=0)
            configured["action_set"].update(batch_episodes=4, success_episodes=2,
                                           seed_offsets=[101, 102], attempts_per_seed=2)
            configured["gradient_monitor_interval"] = 1
            return driver.validate(configured, "vdn", method)

        original_config = driver.config

        def tiny_config(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": FourActionProtoss}
        ), patch.object(driver, "config", tiny_config), patch.object(th.cuda, "is_available", return_value=False):
            root = Path(tmp)
            parent = root / "reference"
            parent.mkdir()
            driver.train_sequence(tiny_protocol("pair_structure", "full_rank_preference"), SEED, parent,
                                  role="pair-reference")
            parent_rows = json.loads((parent / "retention.json").read_text())["stages"]
            self.assertEqual([row["task"] for row in parent_rows], ["stalker", "zealot"])
            self.assertFalse((parent / "full_rank_preference_updates.jsonl").exists())
            state = th.load(parent / "return_state.th")
            self.assertEqual(state["t_env"], parent_rows[0]["t_env"])
            summaries = {}
            for experiment, method in (("pair_structure", "pair_structure_adjacent"),
                                       ("pair_ratio", "pair_ratio_0"),
                                       ("pair_ratio", "pair_ratio_100")):
                run = root / method
                run.mkdir()
                with patch("learners.pair_ablation.select_pairs", wraps=select_pairs) as selector:
                    driver.train_sequence(tiny_protocol(experiment, method), SEED, run,
                                          role="shared-branch", parent=parent)
                self.assertEqual(selector.call_count, 2)
                restore = json.loads((run / "branch_restore.json").read_text())
                self.assertEqual(restore["model_hash"], state["model_hash"])
                rows = jsonlines(run / f"{method}_updates.jsonl")
                self.assertGreater(len(rows), 2)
                self.assertEqual({row["phase"] for row in rows}, {1, 2})
                self.assertTrue(all(row["preference_loss"] > 0 for row in rows))
                self.assertEqual({tuple(sorted(item["source_phase"] for item in row["historical_tasks"]))
                                  for row in rows if row["phase"] == 2}, {(0, 1)})
                gradients = [row["value"] for row in jsonlines(run / "experiment_metrics.jsonl")
                             if row["metric"] == "preference_grad_norm_agent"]
                self.assertTrue(gradients and all(value is not None and value > 0 for value in gradients))
                summaries[method] = []
                for phase in range(2):
                    path = run / "pair_ablation_memory" / f"phase{phase:04d}"
                    audit = json.loads((path / "summary.json").read_text())
                    self.assertEqual(Path(audit["source_pool"]),
                                     parent / "action_set_memory" / f"phase{phase:04d}" / "success.th")
                    self.assertEqual(sum(int(mask.sum()) for mask in th.load(path / "selection.th")),
                                     audit["valid_pairs"])
                    summaries[method].append(audit)
            for zero, full in zip(summaries["pair_ratio_0"], summaries["pair_ratio_100"]):
                self.assertEqual(zero["valid_state_sha256"], full["valid_state_sha256"])
                self.assertNotEqual(zero["pair_mask_sha256"], full["pair_mask_sha256"])
                self.assertEqual((zero["top1_ratio"], full["top1_ratio"]), (0., 1.))
