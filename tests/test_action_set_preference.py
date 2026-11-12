"""CPU integration tests with real VDN/GRU and scripted SMAC-style episodes."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/continual/vdn_action_set_example.json"
sys.path.insert(0, str(ROOT / "drivers/continual"))
with patch.dict(os.environ, {"MARL_PROTOCOL": str(PROTOCOL_PATH)}):
    import action_set_train as driver
from learners.action_set_preference import (
    ActionSetPreference, batch_episodes, decision_masks, make_good_mask,
    pack_episode, preference_loss, unroll,
)
from learners.q_learner import QLearner


class ScriptedProtoss:
    episode_limit = 4

    def __init__(self, **kwargs):
        self.wrapped = SimpleNamespace(env=SimpleNamespace(
            n_enemies=5, map_type="protoss_gen", n_actions_no_attack=6, unit_type_bits=3))
        self.episodes = 0

    def get_env_info(self):
        return dict(n_agents=5, n_actions=11, state_shape=3, obs_shape=4,
                    episode_limit=self.episode_limit)

    def set_task(self, task):
        self.task = ("stalker", "zealot", "colossus").index(task) + 1

    def reset(self):
        self.episodes += 1
        self.t = 0

    def get_obs(self):
        return np.tile([self.task, self.t, -self.t, 0.5], (5, 1)).astype(np.float32)

    def get_state(self):
        return np.array([self.task, self.t, 0], dtype=np.float32)

    def get_avail_actions(self):
        live = [0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        dead = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        return [live] * 4 + [dead if self.t else live]

    def step(self, actions):
        if any(not avail[a] for a, avail in zip(actions, self.get_avail_actions())):
            raise AssertionError("runner chose an illegal action")
        self.t += 1
        won = self.episodes % 2 == 0
        done = self.t >= (2 if won else self.episode_limit)
        return float(won and done), done, dict(battle_won=won and done, episode_limit=done and not won)

    def close(self):
        pass


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class ActionSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def test_three_stages_and_disabled_control(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        for stage in protocol["stages"]:
            stage["steps"] = 16
        protocol["evaluation"].update(seeds=[3, 4], episodes_per_seed=1, interval=0)
        protocol["action_set"].update(weight=0.5, margin=0.25, success_episodes=2,
                                      seed_offsets=[101, 102], attempts_per_seed=2)
        original_config = driver.config
        original_train = QLearner.train
        td_tasks = []

        def tiny_config(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2)

        def trace_train(learner, batch, *args, **kwargs):
            td_tasks.append(set(batch["obs"][:, 0, 0, 0].tolist()))
            return original_train(learner, batch, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {"action_set_protoss": ScriptedProtoss}
        ), patch.object(driver, "config", tiny_config), patch.object(
            QLearner, "train", trace_train
        ), patch.object(th.cuda, "is_available", return_value=False):
            out, baseline, zero = [Path(tmp) / name for name in ("preference", "baseline", "zero")]
            for path in (out, baseline, zero):
                path.mkdir()
            with np.testing.suppress_warnings() as quiet:
                quiet.filter(UserWarning)
                driver.train_sequence(driver.validate(protocol, "vdn", "action_set_preference"), 8, out)
                rows = jsonlines(out / "action_set_updates.jsonl")
                self.assertEqual(len(rows), len(td_tasks))
                for row, tasks in zip(rows, td_tasks):
                    self.assertEqual(tasks, {row["phase"] + 1})
                driver.train_sequence(driver.validate(protocol, "vdn", "baseline"), 8, baseline)
                disabled = copy.deepcopy(protocol)
                disabled["action_set"]["weight"] = 0
                driver.train_sequence(driver.validate(disabled, "vdn", "action_set_preference"), 8, zero)
                offline = Path(tmp) / "evaluation"
                offline.mkdir()
                driver.evaluate_final(out, offline, 3, protocol)
            for path in (out, baseline, zero):
                self.assertEqual(json.loads((path / "status.json").read_text())["status"], "completed")
            for path in (baseline, zero):
                self.assertFalse((path / "action_set_memory").exists())
                self.assertFalse((path / "action_set_updates.jsonl").exists())

            expected = [set(), {"stalker"}, {"stalker", "zealot"}]
            for phase, old in enumerate(expected):
                phase_rows = [row for row in rows if row["phase"] == phase]
                self.assertTrue(phase_rows)
                for row in phase_rows:
                    history = row["historical_tasks"]
                    self.assertEqual({item["task"] for item in history}, old)
                    self.assertEqual(row["historical_episode_count"], 8 if old else 0)
                    self.assertEqual({item["sampled_episodes"] for item in history},
                                     {8 // len(old)} if old else set())
                    for item in history:
                        self.assertGreater(item["valid_states"], 0)
                        self.assertEqual(item["pool_size"], 2)
                    mean_loss = sum(item["loss"] for item in history) / len(old) if old else 0
                    self.assertAlmostEqual(row["preference_loss"], mean_loss, places=6)
                    self.assertAlmostEqual(row["total_loss"], row["td_loss"] + 0.5 * mean_loss, places=6)
                if old:
                    self.assertTrue(any(row["preference_loss"] > 0 for row in phase_rows))
                pool_dir = out / "action_set_memory" / f"phase{phase:04d}"
                successful = th.load(pool_dir / "success.th")
                failures = th.load(pool_dir / "failure.th")
                self.assertEqual({ep["seed"] for ep in successful["episodes"]}, {109, 110})
                self.assertEqual(len(failures["episodes"]), 2)
                self.assertTrue(all(not ep["won"] and "good_mask" not in ep for ep in failures["episodes"]))
                for ep in successful["episodes"]:
                    good = ep["good_mask"]
                    self.assertFalse((good & ~ep["data"]["avail_actions"].bool()).any())
                    self.assertFalse(good[-1].any())
                    self.assertFalse(good[1, -1].any())
                    self.assertTrue(good[1, 0].any())
                    chosen = ep["data"]["actions"][:-1]
                    covered = good[:-1].gather(-1, chosen).squeeze(-1)
                    self.assertTrue(covered[good[:-1].any(dim=-1)].all())
                cfg = json.loads((out / "config.json").read_text())
                args = SimpleNamespace(**cfg)
                spec = driver.episode_spec(json.loads((out / "env_info.json").read_text()))
                scheme = driver.augmented_scheme(spec)
                mac = driver.base.controllers[args.mac](scheme, spec[1], args)
                mac.agent.load_state_dict(th.load(out / successful["reference_checkpoint"]))
                old_batch, stored_mask = batch_episodes(successful["episodes"], scheme, spec[1], "cpu")
                with th.no_grad():
                    q = unroll(mac, old_batch, old_batch.max_seq_length - 1)
                    rebuilt_mask, _ = make_good_mask(old_batch, q, successful["delta"])
                self.assertTrue(th.equal(rebuilt_mask, stored_mask))

            checkpoints = [sorted((path / "models").glob("phase0000_*"))[0] / "agent.th"
                           for path in (out, baseline, zero)]
            first, second, third = [th.load(path) for path in checkpoints]
            self.assertTrue(all(th.equal(first[k], second[k]) and th.equal(first[k], third[k]) for k in first))
            for a, b in zip(sorted((baseline / "models").glob("*/agent.th")),
                            sorted((zero / "models").glob("*/agent.th"))):
                a_state, b_state = th.load(a), th.load(b)
                self.assertTrue(all(th.equal(a_state[k], b_state[k]) for k in a_state))
            retention = json.loads((out / "retention.json").read_text())
            self.assertEqual(set(retention["stages"][-1]["win_rates"]), set(protocol["tasks"]))
            offline_rows = json.loads((offline / "results.json").read_text())["tasks"]
            self.assertEqual({row["task"] for row in offline_rows}, set(protocol["tasks"]))
            self.assertTrue(all(row["episodes"] == 3 for row in offline_rows))

    def test_preference_backprop_and_decision_masks(self):
        protocol = driver.validate(json.loads(PROTOCOL_PATH.read_text()), "vdn", "action_set_preference")
        args = SimpleNamespace(**driver.config(protocol, 7, Path("/tmp")))
        args.device, args.n_agents, args.n_actions = "cpu", 5, 11
        th.manual_seed(7)
        with patch.dict(driver.base.environments, {"action_set_protoss": ScriptedProtoss}):
            runner = driver.base.EpisodeRunner(args, driver.base.Logger(driver.logging.getLogger("test")))
        runner.env.set_task("stalker")
        spec = driver.episode_spec(runner.get_env_info())
        scheme = driver.augmented_scheme(spec)
        mac = driver.base.controllers[args.mac](scheme, spec[1], args)
        runner.setup(*spec, mac)
        records = []
        with th.no_grad(), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            for _ in range(2):
                batch = runner.run(test_mode=True)
                record = pack_episode(batch, runner.t)
                record["good_mask"] = th.zeros_like(record["data"]["avail_actions"], dtype=th.bool)
                records.append(record)
        batch, _ = batch_episodes(records, scheme, spec[1], "cpu")
        legal, decisions = decision_masks(batch)
        self.assertEqual(decisions.sum(dim=(1, 2)).tolist(), [17, 9])
        self.assertFalse(decisions[0, 3, -1])
        self.assertTrue(decisions[0, 3, 0])  # final timeout transition
        self.assertTrue(decisions[1, 1, 0])  # final terminal transition
        self.assertFalse(decisions[1, 2:].any())  # bootstrap and padding
        batch["terminated"][0, 1, 0] = 1
        self.assertFalse(decision_masks(batch)[1][0, 2:].any())
        batch["terminated"][0, 1, 0] = 0
        with th.no_grad():
            q = unroll(mac, batch, 4)
            worst_legal = q.masked_fill(~legal, th.inf).argmin(dim=-1, keepdim=True)
            good = th.zeros_like(batch["avail_actions"], dtype=th.bool)
            good[:, :-1].scatter_(-1, worst_legal, True)
        loss, count, _ = preference_loss(mac, batch, good, 0.0)
        self.assertEqual(count.item(), 26)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        for weight in (mac.agent.rnn.weight_hh, mac.agent.fc2.weight):
            self.assertTrue(th.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.norm().item(), 0)
        mac.agent.zero_grad(set_to_none=True)
        all_good = batch["avail_actions"].bool()
        loss, count, _ = preference_loss(mac, batch, all_good, 1.0)
        self.assertEqual((loss.item(), count.item()), (0, 0))
        loss.backward()
        self.assertTrue(th.equal(mac.agent.fc2.weight.grad, th.zeros_like(mac.agent.fc2.weight)))

    def test_fixed_budget_beyond_eight_tasks_and_revisit(self):
        sampler = ActionSetPreference(dict(batch_episodes=8), np.random.default_rng(7), {}, {}, None)
        for phase in range(12):
            sampler.history[phase] = dict(task=str(phase), phase=phase, episodes=[None])
        sampler.history[12] = dict(task="0", phase=12, episodes=[None])
        sampler.history[13] = dict(task="empty", phase=13, episodes=[])
        sampler.task = "1"
        seen = set()
        for _ in range(50):
            sample = sampler.sample()
            self.assertEqual(len(sample), 8)
            self.assertEqual(len({pool["task"] for pool, _ in sample}), 8)
            self.assertTrue(all(len(ids) == 1 for _, ids in sample))
            self.assertTrue(all(pool["phase"] != 0 for pool, _ in sample))
            seen.update(pool["task"] for pool, _ in sample)
        self.assertEqual(seen, {str(i) for i in range(12)} - {"1"})

    def test_incompatible_rosters_are_rejected(self):
        protocol = json.loads(PROTOCOL_PATH.read_text())
        protocol["tasks"]["zealot"]["ally"][1] = 4
        with self.assertRaisesRegex(ValueError, "shared Agent inputs"):
            driver.validate(protocol, "vdn", "action_set_preference")


if __name__ == "__main__":
    unittest.main()
