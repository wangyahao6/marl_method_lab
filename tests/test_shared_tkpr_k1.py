"""Controlled SZC forks, weighting, high-regret loss and paired timing."""
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch as th
from torch.nn.functional import softplus

from test_action_set_preference import ScriptedProtoss, driver, jsonlines
from action_set_protocol import TOPK_HIGH_REGRET_DEFAULTS, TOPK_WEIGHTED_DEFAULTS
from common.preference_timing import paired_updates
from learners.ranking_preference import (
    RankingPreference, topk_high_regret_statistics, topk_preference, topk_preference_high_regret,
    topk_preference_weighted, topk_weighted_statistics,
)
from protection import digest


class ForkableProtoss(ScriptedProtoss):
    def distribution_state(self):
        return self.episodes

    def restore_distribution(self, episodes):
        self.episodes = episodes


class SharedTKPRTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def test_confidence_weights_are_normalized_over_constrained_states(self):
        old = th.tensor([[3., 2., 0.], [3., 0., 0.], [3., 3., 0.], [3., 0., 0.]], requires_grad=True)
        new = th.tensor([[0., 2., 3.], [3., 2., 0.], [0., 1., 2.], [0., 1., 2.]], requires_grad=True)
        legal = th.ones_like(old, dtype=th.bool)
        valid = th.tensor([True, True, True, False])
        settings = TOPK_WEIGHTED_DEFAULTS
        loss, count, summary = topk_preference_weighted(new, old, legal, valid, settings)
        scale = old.detach().std(dim=-1, unbiased=False)
        confidence = th.tensor([1., 3., 0.]) / scale[:3]
        loss_per_state = softplus((new[:3, 1:] - new[:3, :1]) / scale[:3, None]).mean(dim=-1)
        expected = (confidence * loss_per_state).sum() / confidence.sum()
        th.testing.assert_close(loss, expected)
        self.assertEqual(count.item(), 3)
        stats = topk_weighted_statistics(summary.tolist())
        self.assertAlmostEqual(stats["mean_state_confidence"], confidence.mean().item())
        self.assertEqual(stats["zero_confidence_states"], 1)
        self.assertEqual(stats["nonzero_weight_states"], 2)
        self.assertAlmostEqual(stats["nonzero_weight_coverage"], 2 / 3)
        self.assertAlmostEqual(stats["nonzero_weight_fraction_of_valid"], 2 / 3)
        self.assertEqual(stats["valid_pairs"], 5)
        self.assertEqual(stats["nonzero_loss_states"], 2)
        self.assertEqual(stats["weight_histogram"],
                         {"zero": 1, "(0,0.5)": 0, "[0.5,1)": 1, "[1,2)": 0, "[2,+inf)": 1})
        expected_grad = th.autograd.grad(expected, new)[0]
        loss.backward()
        th.testing.assert_close(new.grad, expected_grad)
        self.assertIsNone(old.grad)

    def test_weighted_all_zero_confidence_falls_back_to_k1(self):
        old = th.tensor([[3., 3., 0.]])
        new = th.tensor([[0., 1., 2.]])
        legal = th.ones_like(old, dtype=th.bool)
        decisions = th.tensor([True])
        a = topk_preference_weighted(new, old, legal, decisions, TOPK_WEIGHTED_DEFAULTS)
        b = topk_preference(new, old, legal, decisions, TOPK_WEIGHTED_DEFAULTS)
        th.testing.assert_close(a[0], b[0])
        self.assertEqual(a[1].item(), 1)
        stats = topk_weighted_statistics(a[2].tolist())
        self.assertEqual(stats["zero_confidence_states"], 1)
        self.assertEqual(stats["nonzero_weight_states"], 1)
        self.assertEqual(stats["nonzero_weight_coverage"], 1.)
        self.assertEqual(stats["weight_histogram"]["[1,2)"], 1)
        _, _, summary = topk_preference_weighted(new, old, legal, ~decisions, TOPK_WEIGHTED_DEFAULTS)
        self.assertIsNone(topk_weighted_statistics(summary.tolist())["nonzero_weight_coverage"])

    def test_high_regret_only_adds_loss_for_legal_changed_actions_at_threshold(self):
        old = th.tensor([[1., 0., 0.], [1., .1, 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.]],
                        requires_grad=True)
        new = th.tensor([[0., 2., 0.], [0., 2., 0.], [2., 1., 0.], [0., 2., 0.], [0., 2., 0.]],
                        requires_grad=True)
        legal = th.ones_like(old, dtype=th.bool)
        legal[3, 1] = False
        decisions = th.tensor([True, True, True, True, False])
        settings = dict(TOPK_HIGH_REGRET_DEFAULTS, q_scale_floor=2.)
        base = topk_preference(new, old, legal, decisions, settings)[0]
        loss, _, summary = topk_preference_high_regret(new, old, legal, decisions, settings)
        extra = softplus((new[0, 1] - new[0, 0]) / 2)
        th.testing.assert_close(loss, base + extra)
        stats = topk_high_regret_statistics(summary.tolist())
        self.assertEqual(stats["high_regret_states"], 1)
        self.assertEqual(stats["mean_high_regret"], 0.5)
        self.assertEqual(stats["high_regret_fraction"], 1 / 4)
        self.assertAlmostEqual(stats["base_preference_loss"], base.item())
        self.assertAlmostEqual(stats["weighted_high_regret_loss"], extra.item())
        self.assertAlmostEqual(stats["high_regret_loss_share"], extra.item() / loss.item())
        disabled_loss, _, disabled_summary = topk_preference_high_regret(
            new, old, legal, decisions, dict(settings, hr_weight=0.))
        th.testing.assert_close(disabled_loss, base)
        disabled = topk_high_regret_statistics(disabled_summary.tolist())
        self.assertEqual(disabled["high_regret_states"], 1)
        self.assertEqual(disabled["weighted_high_regret_loss"], 0.)
        extra_only, count, summary = topk_preference_high_regret(
            new, old, legal, decisions, dict(settings, min_gap=2.))
        extra_only_stats = topk_high_regret_statistics(summary.tolist())
        self.assertEqual(extra_only_stats["valid_pairs"], 0)
        self.assertEqual(extra_only_stats["nonzero_loss_states"], 1)
        self.assertEqual(count.item(), 1)
        self.assertEqual(extra_only_stats["high_regret_loss_share"], 1.)
        th.testing.assert_close(extra_only, extra)
        th.testing.assert_close(th.autograd.grad(loss - base, new)[0], th.autograd.grad(extra, new)[0])
        self.assertIsNone(old.grad)

    def test_empty_pairs_ties_and_single_legal_action(self):
        old = th.tensor([[1., 1., 1.], [3., 2., 1.]])
        new = th.tensor([[0., 1., 2.], [1., 0., 2.]], requires_grad=True)
        legal = th.tensor([[True, True, True], [False, True, False]])
        decisions = th.ones(2, dtype=th.bool)
        cases = (
            (topk_preference_weighted, topk_weighted_statistics, TOPK_WEIGHTED_DEFAULTS),
            (topk_preference_high_regret, topk_high_regret_statistics, TOPK_HIGH_REGRET_DEFAULTS),
        )
        for method, statistics, options in cases:
            with self.subTest(method=method.__name__):
                loss, count, summary = method(new, old, legal, decisions, options)
                result = statistics(summary.tolist())
                self.assertEqual((loss.item(), count.item()), (0., 0))
                self.assertEqual(result["decision_states"], 1)
                for name in ("valid_states", "valid_pairs", "nonzero_loss_states"):
                    self.assertEqual(result[name], 0)
                self.assertTrue(th.equal(th.autograd.grad(loss, new)[0], th.zeros_like(new)))
                if "high_regret_states" in result:
                    self.assertEqual(result["high_regret_fraction"], 0.)
                    self.assertEqual(result["weighted_high_regret_loss"], 0.)
                    self.assertIsNone(result["high_regret_loss_share"])
                else:
                    self.assertEqual(sum(result["weight_histogram"].values()), 0)
        loss, _, summary = topk_preference_high_regret(old, old, th.ones_like(legal), decisions,
                                                       TOPK_HIGH_REGRET_DEFAULTS)
        result = topk_high_regret_statistics(summary.tolist())
        self.assertGreater(loss.item(), 0.)
        self.assertEqual(result["high_regret_states"], 0)
        self.assertEqual(result["high_regret_loss_share"], 0.)

    def test_empty_history_records_update_with_no_preference_participation(self):
        rows = []
        pref = RankingPreference("topk_preference_weighted",
                                 dict(TOPK_WEIGHTED_DEFAULTS, batch_episodes=4),
                                 np.random.default_rng(5), {}, {}, rows.append)
        pref.task = "stalker"
        pref.begin_update()
        self.assertIsNone(pref(None))
        td = th.tensor(1.)
        pref.record(td, td, None, 4)
        self.assertEqual(rows[0]["historical_tasks"], [])
        for name in ("historical_episode_count", "decision_states", "constrained_states",
                     "valid_pairs", "nonzero_loss_states", "preference_participating_updates"):
            self.assertEqual(rows[0][name], 0)
        self.assertEqual(rows[0]["preference_recorded_updates"], 1)
        self.assertFalse(rows[0]["preference_participated_update"])
        self.assertEqual(rows[0]["preference_participation_rate"], 0.)

    def test_all_branches_inherit_shared_s_and_timing_replays_matched_batch(self):
        path = driver.ROOT / "configs/continual/szc_vdn64_tkpr_k1_shared.json"
        raw = json.loads(path.read_text())
        for stage in raw["stages"]:
            stage["steps"] = 32
        raw["evaluation"].update(seeds=[3, 4], episodes_per_seed=1, interval=0,
                                 focus_seeds=None, focus_episodes_per_seed=None)
        raw["action_set"].update(batch_episodes=4, success_episodes=2,
                                 seed_offsets=[101, 102], attempts_per_seed=2)
        methods = ("baseline", "topk_preference", "top1_preference",
                   "topk_preference_weighted", "topk_preference_high_regret")
        for method in methods[1:]:
            raw[method]["min_gap"] = 0
        original_config = driver.config

        def configure(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2, target_update_interval=3)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": ForkableProtoss}
        ), patch.object(driver, "config", configure), patch.object(th.cuda, "is_available", return_value=False):
            out = Path(tmp)
            prefix = out / "s"
            prefix.mkdir()
            k1 = driver.validate(raw, "vdn", "topk_preference")
            driver.train_sequence(k1, 8, prefix, role="shared-prefix")
            driver_state = th.load(prefix / "return_state.th", map_location="cpu")
            self.assertEqual([r["phase"] for r in jsonlines(prefix / "transitions.jsonl")], [0])
            self.assertEqual(len(list((prefix / "models").iterdir())), 1)
            original_checkpoint = next((prefix / "models").iterdir())
            history = prefix / "action_set_memory/phase0000/success.th"
            self.assertTrue(all("old_q_values" in ep["data"] for ep in th.load(history)["episodes"]))
            self.assertFalse((prefix / "topk_preference_updates.jsonl").exists())

            # For each branch inspect the restored learner immediately before its first update.
            real_train = driver.base.learners["q_learner"].train
            for method in methods:
                run = out / method
                run.mkdir()
                effective = driver.validate(raw, "vdn", method)
                starts, td_buffers = [], []
                start_replay = driver.start_replay_task

                def verify_ft(buffer, task):
                    td_buffers.append((task, buffer.episodes_in_buffer))
                    return start_replay(buffer, task)

                def verify(learner, batch, t_env, episode, *a, **kw):
                    if not starts:
                        starts.append(True)
                        self.assertEqual(digest(learner.optimiser.state_dict()),
                                         digest(th.load(original_checkpoint / "opt.th", map_location="cpu")))
                        self.assertEqual(driver.base.parameter_hash(learner), driver_state["model_hash"])
                        self.assertEqual(learner.last_target_update_episode, driver_state["last_target_update_episode"])
                    return real_train(learner, batch, t_env, episode, *a, **kw)

                with patch.object(driver.base.learners["q_learner"], "train", verify), patch.object(
                    driver, "start_replay_task", verify_ft
                ):
                    driver.train_sequence(effective, 8, run, role="shared-branch", parent=prefix,
                                          capture_timing=method == "topk_preference")
                restored = json.loads((run / "branch_restore.json").read_text())
                self.assertEqual(restored["model_hash"], restored["parent_model_hash"])
                self.assertEqual(restored["model_hash"], driver_state["model_hash"])
                self.assertEqual([r["phase"] for r in jsonlines(run / "transitions.jsonl")], [1, 2])
                self.assertEqual(td_buffers, [("zealot", 0), ("colossus", 0)])
                self.assertEqual(len(list((run / "models").iterdir())), 2)
                if method != "baseline":
                    updates = jsonlines(run / f"{method}_updates.jsonl")
                    self.assertTrue(all(r["historical_tasks"] for r in updates))
                    self.assertEqual(updates[0]["historical_tasks"][0]["source_phase"], 0)

            run = out / "topk_preference"
            snapshots = sorted((run / "timing_snapshots").glob("*.th"))
            self.assertEqual(len(snapshots), 6)
            def timing_rows(path):
                snap = th.load(path, map_location="cpu")
                args = SimpleNamespace(**json.loads((run / "config.json").read_text()))
                spec = driver.episode_spec(json.loads((run / "env_info.json").read_text()))
                scheme, groups = driver.augmented_scheme(spec), spec[1]
                mac = driver.base.controllers[args.mac](scheme, groups, args)
                learner = driver.base.learners[args.learner](mac, scheme, driver.base.Logger(
                    logging.getLogger("test_timing")), args)
                pref = RankingPreference("topk_preference", dict(k1["topk_preference"], batch_episodes=4),
                                         np.random.default_rng(8), scheme, groups, lambda row: None)
                pref.history.restore_success(prefix, (0,), require_old_q=True)
                if snap["phase"] == 2:
                    pref.history.restore_success(run, (1,), require_old_q=True)
                initial_digest = digest(snap["optimizer"])
                measured = list(paired_updates(learner, snap, pref, scheme, groups, warmup=1, repeats=2))
                self.assertEqual(digest(snap["optimizer"]), initial_digest)
                return measured

            for snap in (snapshots[0], snapshots[-1]):
                measured = timing_rows(snap)
                self.assertEqual([r["first_arm"] for r in measured], ["TD+TKPR-K1", "TD-only"])
                self.assertTrue(all(r["td_ms"] > 0 and r["td_tkpr_ms"] > 0 for r in measured))


if __name__ == "__main__":
    unittest.main()
