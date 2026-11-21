"""Focused single-agent DQN/TKPR and three-task smoke checks."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from learners.ranking_preference import topk_preference_weighted
from single_agent.dqn import DQNLearner
from single_agent.memory import PreferenceMemory
from single_agent.mountaincar import GOALS, GoalMountainCar
from single_agent.preference import frpr_loss, preference_loss
from single_agent.train import DEFAULT_CONFIG, run


class SingleAgentTKPRTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = json.loads(DEFAULT_CONFIG.read_text())

    def test_environment_and_goal_code(self):
        for task, goal in GOALS.items():
            env = GoalMountainCar(task)
            try:
                observation, _ = env.reset(seed=7)
                self.assertEqual(observation.shape, (3,))
                self.assertEqual(env.action_space.n, 3)
                self.assertAlmostEqual(float(observation[2]), (goal - .5) / .05)
                self.assertEqual(env.spec.max_episode_steps, 200)
                self.assertEqual(env.unwrapped.goal_position, goal)
                self.assertEqual(env.unwrapped.force, .001)
                self.assertEqual(env.unwrapped.gravity, .0025)
                next_observation, reward, *_ = env.step(1)
                self.assertEqual(next_observation.shape, (3,))
                self.assertEqual(reward, -1.)
            finally:
                env.close()

    def test_dqn_forward_update_and_switch(self):
        config = dict(self.config, batch_size=2, learning_starts=2)
        learner = DQNLearner(config, "DQN-FT", seed=1)
        state = np.array([-.5, 0., -1.], dtype=np.float32)
        self.assertEqual(learner.online(torch.from_numpy(state)).shape, (3,))
        before = copy.deepcopy(learner.online.state_dict())
        for _ in range(2):
            learner.observe(state, 1, -1., state, False, False)
        result = learner.update()
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in learner.online.state_dict().items()))
        optimizer_steps = sum(len(v) for v in learner.optimizer.state.values())
        self.assertGreater(optimizer_steps, 0)
        target = copy.deepcopy(learner.target.state_dict())
        learner.switch_task()
        self.assertEqual(len(learner.replay), 0)
        self.assertEqual(learner.task_steps, 0)
        self.assertTrue(all(torch.equal(target[k], v) for k, v in learner.target.state_dict().items()))
        self.assertGreater(sum(len(v) for v in learner.optimizer.state.values()), 0)

    def test_k1_selects_only_top_action_pairs(self):
        settings = self.config["preference"]
        old = torch.tensor([[3., 2., 0.]])
        current = torch.tensor([[0., 2., 4.]], requires_grad=True)
        k1 = preference_loss("DQN-TKPR-K1", current, old, settings)
        k2 = preference_loss("DQN-TKPR-K2", current, old, settings)
        scale = old.std(dim=-1, unbiased=False).clamp_min(settings["q_scale_floor"])
        expected = (torch.nn.functional.softplus((current[0, 1] - current[0, 0]) / scale) +
                    torch.nn.functional.softplus((current[0, 2] - current[0, 0]) / scale)) / 2
        torch.testing.assert_close(k1, expected.squeeze())
        self.assertNotAlmostEqual(k1.item(), k2.item())

    def test_k1_weighted_matches_repository_implementation(self):
        settings = self.config["preference"]
        old = torch.tensor([[3., 2., 0.], [4., 0., 0.]])
        current = torch.tensor([[0., 1., 2.], [1., 0., 2.]], requires_grad=True)
        adapted = preference_loss("DQN-TKPR-K1-W", current, old, settings)
        direct = topk_preference_weighted(current, old, torch.ones_like(old, dtype=torch.bool),
                                          torch.ones(2, dtype=torch.bool), dict(settings, k=1))[0]
        torch.testing.assert_close(adapted, direct)
        torch.testing.assert_close(torch.autograd.grad(adapted, current, retain_graph=True)[0],
                                   torch.autograd.grad(direct, current)[0])

    def test_k2_equals_frpr_loss_and_gradient(self):
        settings = self.config["preference"]
        old = torch.tensor([[3., 2., 0.], [0., 1., 4.]])
        current = torch.randn(2, 3, requires_grad=True)
        k2 = preference_loss("DQN-TKPR-K2", current, old, settings)
        frpr = frpr_loss(current, old, settings)
        torch.testing.assert_close(k2, frpr)
        torch.testing.assert_close(torch.autograd.grad(k2, current, retain_graph=True)[0],
                                   torch.autograd.grad(frpr, current)[0])

    def test_frozen_reference_and_preference_update(self):
        config = dict(self.config, batch_size=2, learning_starts=2)
        learner = DQNLearner(config, "DQN-TKPR-K1", seed=2)
        states = [np.array([-.5, 0., -1.], dtype=np.float32),
                  np.array([-.4, .01, -1.], dtype=np.float32)]
        reference = copy.deepcopy(learner.online)
        memory = PreferenceMemory()
        memory.add_successful_trajectories("T1", [states], reference)
        torch.testing.assert_close(memory.tasks["T1"][0][1],
                                   reference(torch.from_numpy(states[0])).detach())
        stored_q = memory.tasks["T1"][0][1].clone()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.add_(1.)
        torch.testing.assert_close(memory.tasks["T1"][0][1], stored_q)
        memory.add_successful_trajectories("T2", [states], reference)
        sample = memory.sample(4, learner.rng)
        self.assertEqual(sample[0].shape, (4, 3))
        self.assertEqual(sample[1].shape, (4, 3))
        memory.tasks["T2"] = [(np.array([-.5, 0., 1.], dtype=np.float32), q)
                              for _, q in memory.tasks["T2"]]
        balanced = memory.sample(6, learner.rng)[0][:, 2]
        self.assertEqual(int((balanced == -1).sum()), 3)
        self.assertEqual(int((balanced == 1).sum()), 3)
        for _ in range(2):
            learner.observe(states[0], 1, -1., states[1], False, False)
        result = learner.update(memory)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        self.assertGreater(result["preference_loss"], 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in learner.online.parameters()))
        learner.switch_task()
        self.assertEqual(len(learner.replay), 0)
        self.assertEqual(len(memory.tasks), 2)

    def test_three_task_smoke(self):
        config = dict(self.config, batch_size=2, learning_starts=2,
                      reference_collection_episodes=1)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "smoke"
            report = run(config, "DQN-TKPR-K1", 3, output, steps_per_task=3)
            self.assertEqual([item["task"] for item in report], ["T1", "T2", "T3"])
            self.assertTrue(all(item["steps"] == 3 for item in report))
            self.assertTrue(all((output / f"{task}_checkpoint.pt").exists() for task in GOALS))
            self.assertTrue(all((output / f"{task}_preference_memory.pt").exists() for task in GOALS))
            self.assertTrue((output / "source.tar.gz").exists())
            self.assertTrue((output / "git_revision.txt").exists())


if __name__ == "__main__":
    unittest.main()
