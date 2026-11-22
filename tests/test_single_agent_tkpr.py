"""Focused MinAtar DQN/TKPR and short continual-flow checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from learners.ranking_preference import full_rank_preference, topk_preference_weighted
from single_agent.dqn import DQNLearner, QNetwork
from single_agent.evaluate import evaluate_checkpoints, retention_metrics
from single_agent.memory import PreferenceMemory, collect_reference
from single_agent.minatar_env import GAMES, make_env
from single_agent.preference import preference_result, preference_statistics
from single_agent.train import DEFAULT_CONFIG, make_schedule, run


class MinAtarTKPRTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_official_games_have_shared_observation_and_six_actions(self):
        for game, channels in zip(GAMES, (4, 7, 6)):
            env = make_env(game)
            try:
                observation, _ = env.reset(seed=7)
                self.assertEqual(observation.shape, (7, 10, 10))
                self.assertEqual(observation.dtype, np.float32)
                self.assertEqual(env.action_space.n, 6)
                self.assertEqual(env.unwrapped.game.sticky_action_prob, .1)
                if game == "SpaceInvaders":
                    self.assertFalse(env.unwrapped.game.env.ramping)
                self.assertFalse(observation[channels:].any())
                next_observation, _, _, _, _ = env.step(0)
                self.assertEqual(next_observation.shape, (7, 10, 10))
            finally:
                env.close()

    def test_cnn_and_dqn_update(self):
        config = dict(self.config, batch_size=2)
        learner = DQNLearner(config, "DQN-FT", seed=1, device="cpu")
        learner.switch_game("Breakout")
        state = np.zeros((7, 10, 10), dtype=np.float32)
        self.assertEqual(QNetwork()(torch.zeros(4, 7, 10, 10)).shape, (4, 6))
        before = copy.deepcopy(learner.online.state_dict())
        for _ in range(2):
            learner.observe(state, 1, 1., state, False, False)
        metrics = learner.update()
        self.assertTrue(np.isfinite(metrics["total_loss"]))
        self.assertTrue(any(not torch.equal(before[k], value)
                            for k, value in learner.online.state_dict().items()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in learner.online.parameters()))

    def test_k1_k2_pair_counts_and_weighted_original(self):
        settings = self.config["preference"]
        old = torch.tensor([[6., 5., 4., 3., 2., 1.],
                            [7., 5., 4., 3., 2., 1.]])
        current = torch.randn(2, 6, requires_grad=True)
        k1 = preference_result("DQN-TKPR-K1", current, old, settings)
        k2 = preference_result("DQN-TKPR-K2", current, old, settings)
        weighted = preference_result("DQN-TKPR-K1-W", current, old, settings)
        self.assertEqual(preference_statistics("DQN-TKPR-K1", k1[2])["valid_pairs"], 10)
        self.assertEqual(preference_statistics("DQN-TKPR-K2", k2[2])["valid_pairs"], 18)
        self.assertEqual(preference_statistics("DQN-TKPR-K1-W", weighted[2])["valid_pairs"], 10)
        frpr = full_rank_preference(current, old, torch.ones_like(old, dtype=torch.bool),
                                    torch.ones(2, dtype=torch.bool), settings)
        self.assertEqual(int(frpr[2][2]), 30)
        direct = topk_preference_weighted(current, old, torch.ones_like(old, dtype=torch.bool),
                                           torch.ones(2, dtype=torch.bool), dict(settings, k=1))
        torch.testing.assert_close(weighted[0], direct[0])
        torch.testing.assert_close(torch.autograd.grad(weighted[0], current, retain_graph=True)[0],
                                   torch.autograd.grad(direct[0], current)[0])

    def test_frozen_reference_does_not_filter_by_success_or_overwrite(self):
        model = QNetwork()
        record = collect_reference("Breakout", model, 5, 73, 0)
        self.assertEqual(record["states"].shape, (5, 7, 10, 10))
        with torch.no_grad():
            expected = model(torch.as_tensor(record["states"]))
        torch.testing.assert_close(record["old_q_values"], expected)
        memory = PreferenceMemory()
        self.assertEqual(memory.add_reference(**record), 5)
        old_q = memory.tasks["Breakout"]["old_q_values"].clone()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.)
        self.assertEqual(memory.add_reference("Breakout", record["states"],
                     torch.zeros_like(old_q), collection_seed=99, model_stage=2), 0)
        torch.testing.assert_close(memory.tasks["Breakout"]["old_q_values"], old_q)
        self.assertEqual(memory.tasks["Breakout"]["collection_seed"], 73)
        self.assertEqual(memory.tasks["Breakout"]["model_stage"], 0)

    def test_balanced_history_excludes_current_game(self):
        memory = PreferenceMemory()
        for index, game in enumerate(GAMES):
            states = np.full((2, 7, 10, 10), index, dtype=np.float32)
            old_q = np.full((2, 6), index, dtype=np.float32)
            memory.add_reference(game, states, old_q, collection_seed=index, model_stage=index)
        states, old_q, sources = memory.sample(8, "SpaceInvaders")
        self.assertEqual(states.shape, (8, 7, 10, 10))
        self.assertEqual(old_q.shape, (8, 6))
        self.assertEqual(sources.count("Breakout"), 4)
        self.assertEqual(sources.count("Freeway"), 4)
        self.assertNotIn("SpaceInvaders", sources)

    def test_replay_switch_rules_and_schedule(self):
        state = np.zeros((7, 10, 10), dtype=np.float32)
        for method in DQNLearner.METHODS:
            learner = DQNLearner(self.config, method, 3, device="cpu")
            learner.switch_game("Breakout")
            learner.observe(state, 0, 0., state, False, False)
            self.assertFalse(learner.switch_game("Breakout"))
            self.assertEqual(len(learner.replay), 1)
            target = copy.deepcopy(learner.target.state_dict())
            optimizer = learner.optimizer
            self.assertTrue(learner.switch_game("Freeway"))
            self.assertEqual(len(learner.replay), 1 if method == "DQN-ER" else 0)
            self.assertIs(learner.optimizer, optimizer)
            self.assertTrue(all(torch.equal(target[k], value)
                                for k, value in learner.target.state_dict().items()))
            self.assertEqual(learner.steps, 1)
        self.assertEqual(make_schedule(3535), make_schedule(3535))
        self.assertEqual(set(make_schedule(3535)[:3]), set(GAMES))
        self.assertEqual(len(make_schedule(3535)), 7)

    def test_per_game_retention_uses_first_and_final_return(self):
        rows = []
        for stage, game in enumerate(GAMES):
            rows.append(dict(stage=stage, game=game, kind="stage_end", mean_return=stage + 2.))
            rows.append(dict(stage=2, game=game, kind="final", mean_return=stage + 1.))
        metrics = retention_metrics(GAMES, rows)
        self.assertEqual([item["bwt_return"] for item in metrics], [-1., -1., -1.])
        self.assertEqual([item["forgetting_return"] for item in metrics], [1., 1., 1.])

    def test_real_history_participates_in_tkpr_update(self):
        config = dict(self.config, batch_size=2, diagnostics_interval=1)
        learner = DQNLearner(config, "DQN-TKPR-K1", 5, device="cpu")
        reference = collect_reference("Breakout", learner.online, 4, 19, 0)
        memory = PreferenceMemory()
        memory.add_reference(**reference)
        learner.switch_game("Freeway")
        env = make_env("Freeway")
        try:
            state, _ = env.reset(seed=5)
            for _ in range(2):
                next_state, reward, terminated, truncated, _ = env.step(0)
                learner.observe(state, 0, reward, next_state, terminated, truncated)
                state = next_state
        finally:
            env.close()
        metrics = learner.update(memory, diagnostics=True)
        for key in ("preference_loss", "constrained_states", "valid_pairs",
                    "pair_violation_rate", "top1_flip_rate", "preference_grad_norm",
                    "td_grad_norm", "preference_td_grad_ratio",
                    "td_preference_grad_cosine"):
            self.assertIn(key, metrics)
        self.assertGreater(metrics["preference_loss"], 0)
        self.assertGreater(metrics["constrained_states"], 0)
        self.assertGreater(metrics["valid_pairs"], 0)
        self.assertEqual(metrics["historical_source_counts"], {"Breakout": 2})
        self.assertTrue(all(np.isfinite(value) for key, value in metrics.items()
                            if key in ("total_loss", "td_loss", "preference_loss",
                                       "td_grad_norm", "preference_grad_norm")))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in learner.online.parameters()))

    def test_five_method_three_game_smoke(self):
        config = dict(self.config, interval_steps=100, reference_steps=100,
                      evaluation_episodes=1, final_evaluation_episodes=1,
                      progress_interval=100, diagnostics_interval=50)
        schedule = list(GAMES)
        with tempfile.TemporaryDirectory() as temporary:
            observed = []
            for method in DQNLearner.METHODS:
                output = Path(temporary) / method
                summary = run(config, method, 17, output, schedule=schedule, device="cpu")
                self.assertEqual([entry["game"] for entry in summary], schedule)
                self.assertEqual(summary[-1]["environment_steps"], 300)
                self.assertEqual(json.loads((output / "task_schedule.json").read_text())["games"], schedule)
                self.assertTrue(all((output / f"stage_{i}_checkpoint.pt").exists() for i in range(3)))
                self.assertEqual(len((output / "evaluations.jsonl").read_text().splitlines()), 12)
                self.assertEqual(len(json.loads((output / "retention.json").read_text())), 3)
                self.assertEqual(len(summary[0]["memory_states"]), 1)
                saved = torch.load(output / "memory_Breakout.pt", map_location="cpu")
                self.assertEqual(saved["model_stage"], 0)
                self.assertEqual(saved["collection_seed"], 17 + config["collection_seed_base"])
                self.assertEqual(len(saved["states"]), 100)
                self.assertEqual(saved["old_q_values"].shape, (100, 6))
                if method == "DQN-FT":
                    rows, retention = evaluate_checkpoints(output)
                    self.assertEqual(len(rows), 12)
                    self.assertEqual(len(retention), 3)
                if method.startswith("DQN-TKPR"):
                    self.assertGreater(summary[1]["preference_updates"], 0)
                observed.append(json.loads((output / "task_schedule.json").read_text())["games"])
            self.assertTrue(all(item == observed[0] for item in observed))


if __name__ == "__main__":
    unittest.main()
