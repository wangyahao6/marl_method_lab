"""Ranking loss checks and CPU sequence tests with real VDN/GRU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch as th
from torch.nn.functional import softplus

from test_action_set_preference import PROTOCOL_PATH, ScriptedProtoss, driver, jsonlines
from action_set_protocol import RANK_DEFAULTS, RANK_METHODS
from learners.ranking_preference import RankingPreference, full_rank_preference, top1_preference


class RankingTest(unittest.TestCase):
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
        self.assertEqual(stats[2:].tolist(), [3, 1, 0, 1])
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertEqual(current.grad[0, 3].item(), 0)

        top1, count, stats = top1_preference(current, reference, legal, decisions, RANK_DEFAULTS)
        expected = softplus(th.tensor([-3.0, -2.0]) / scale).mean() * (2 / scale)
        th.testing.assert_close(top1, expected)
        self.assertEqual(stats[2:].tolist(), [2, 1, 0, 1])

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
        self.assertEqual(stats[2:].tolist(), [2, 4, 3, 3])
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

    def test_temperature_and_config_validation(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        protocol["action_set"].pop("delta_by_task")
        for method in RANK_METHODS:
            effective = driver.validate(protocol, "vdn", method)
            self.assertEqual(effective[method], RANK_DEFAULTS)
            for key, value in (("temperature", 0), ("min_gap", -0.1), ("q_scale_floor", 0), ("weight", True)):
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
                                      seed_offsets=[101, 102], attempts_per_seed=2)
        original_config = driver.config
        original_reference = RankingPreference.reference_mac
        frozen = {}

        def tiny_config(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2)

        def track_reference(preference, mac, pool):
            reference = original_reference(preference, mac, pool)
            checkpoint = preference.run_dir / pool["reference_checkpoint"]
            if checkpoint not in frozen:
                frozen[checkpoint] = reference
            self.assertIsNot(reference.agent, mac.agent)
            self.assertIsNone(reference.hidden_states)
            self.assertFalse(reference.agent.training)
            return reference

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": ScriptedProtoss}
        ), patch.object(driver, "config", tiny_config), patch.object(
            RankingPreference, "reference_mac", track_reference
        ), patch.object(th.cuda, "is_available", return_value=False), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            out = Path(tmp)
            for method in ("baseline", "action_set_preference", *RANK_METHODS, "zero"):
                run = out / method
                run.mkdir()
                active = "top1_preference" if method == "zero" else method
                configured = copy.deepcopy(protocol)
                if active in RANK_METHODS:
                    configured[active] = dict(weight=0 if method == "zero" else 0.5, min_gap=0)
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
            for method in RANK_METHODS:
                run = out / method
                pool = th.load(run / "action_set_memory/phase0000/success.th")
                self.assertEqual(len(original["episodes"]), len(pool["episodes"]))
                for old, new in zip(original["episodes"], pool["episodes"]):
                    self.assertEqual((old["seed"], old["index"]), (new["seed"], new["index"]))
                    self.assertTrue(all(th.equal(v, new["data"][k]) for k, v in old["data"].items()))
                    self.assertNotIn("good_mask", new)
                samples = jsonlines(run / f"{method}_updates.jsonl")
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
                            self.assertGreater(item["valid_pairs"], 0)
                            self.assertEqual(item["valid_states"], (5 + 4) * item["sampled_episodes"])
                            self.assertEqual(item["coverage"], 1)
                            self.assertTrue(0 <= item["top1_flip_rate"] <= 1)
                        mean = sum(item["loss"] for item in history) / len(tasks) if tasks else 0
                        self.assertAlmostEqual(row["preference_loss"], mean, places=6)
                        self.assertAlmostEqual(row["total_loss"], row["td_loss"] + 0.5 * mean, places=6)
                rank_checkpoints = sorted((run / "models").glob("*/agent.th"))
                initial, same = th.load(rank_checkpoints[0]), th.load(baseline[0])
                self.assertTrue(all(th.equal(initial[k], same[k]) for k in initial))
                trained, control = th.load(rank_checkpoints[1]), th.load(baseline[1])
                self.assertFalse(th.equal(trained["rnn.weight_hh"], control["rnn.weight_hh"]))

            self.assertEqual(len(frozen), 4)
            for checkpoint, reference in frozen.items():
                saved = th.load(checkpoint)
                self.assertTrue(all(th.equal(saved[k], v) for k, v in reference.agent.state_dict().items()))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in reference.agent.parameters()))
                self.assertIsNone(reference.hidden_states)


if __name__ == "__main__":
    unittest.main()
