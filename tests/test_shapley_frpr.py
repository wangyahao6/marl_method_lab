import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

from test_action_set_preference import PROTOCOL_PATH, ScriptedProtoss, driver, jsonlines
from action_set_protocol import RANK_DEFAULTS, SELECTED_METHODS, validate
from drivers.continual.analysis.shapley_frpr import coalition_rows, exact_shapley, interactions
from drivers.continual.memory.episodes import preference_episode_id
from learners.ranking_preference import full_rank_preference, reference_order, reference_scale
from learners.selected_frpr import SelectedFRPR, choose_pairs, oriented_pairs, pair_inventory


class ShapleyMathTest(unittest.TestCase):
    def test_participants_scale_and_full_mask_equivalence(self):
        old = th.tensor([[0., .01, .01, .105, 900.], [2., 4., 1., 3., 100.], [1., 2., 3., 4., 5.]])
        q = th.tensor([[1., 2., 1., 0., 10.], [3., 1., 0., 2., 10.], [0., 0., 0., 0., 0.]],
                      requires_grad=True)
        legal = th.tensor([[True, True, True, True, False]]).expand_as(old)
        available = th.tensor([True, True, False])
        settings = dict(RANK_DEFAULTS, temperature=.7)
        mean, centered, scale = reference_scale(old, legal, settings["q_scale_floor"])
        normalized, original_scale, _, _ = reference_order(old, legal, settings["q_scale_floor"])
        th.testing.assert_close(mean[:, 0], old[:, :4].mean(dim=-1))
        th.testing.assert_close(scale[:, 0], old[:, :4].std(dim=-1, unbiased=False).clamp_min(.1))
        th.testing.assert_close(centered / scale, normalized)
        th.testing.assert_close(scale, original_scale)
        gaps, pairs, _ = pair_inventory(old, legal, available, settings)
        self.assertEqual(oriented_pairs(gaps[0], pairs[0]), [(3, 0), (3, 1), (3, 2)])
        self.assertEqual(len(oriented_pairs(gaps[1], pairs[1])), 6)
        reference = full_rank_preference(q, old, legal, available, settings)
        selected = full_rank_preference(q, old, legal, available, settings, pair_selection=pairs)
        for a, b in zip(reference, selected):
            th.testing.assert_close(a, b, atol=0, rtol=0)
        ref_grad = th.autograd.grad(reference[0], q)[0]
        selected_grad = th.autograd.grad(selected[0], q)[0]
        th.testing.assert_close(ref_grad, selected_grad, atol=0, rtol=0)

    def test_projection_and_probability(self):
        base = [0., 2., 4., 1e6]
        outcomes = np.array([[[0., 0.], [1., 0.], [3., 1.]]])
        rows = list(coalition_rows(base, [0, 1, 2], [(0, 1), (1, 2)], outcomes, .25, 1.))
        np.testing.assert_array_equal(rows[0][1], base[:3])
        np.testing.assert_allclose(rows[1][1], [1.125, .875, 4.])
        np.testing.assert_allclose(rows[-1][1], [2.25, 2., 1.75])
        for _, _, p, values, audit in rows:
            self.assertEqual(len(p), 3)
            self.assertAlmostEqual(float(p.sum()), 1.)
            self.assertAlmostEqual(values[0, 1], p[2])
            self.assertEqual(audit["after_violations"], 0)

    def test_exact_shapley_and_interactions(self):
        values = np.array([[0., 0.], [2., 1.], [1., 1.], [9., 1.]])
        phi, error = exact_shapley(values, 2)
        np.testing.assert_allclose(phi, [[5., .5], [4., .5]])
        np.testing.assert_allclose(error, 0)
        [(i, j, index)] = list(interactions(values, 2))
        self.assertEqual((i, j), (0, 1))
        np.testing.assert_allclose(index, [6., -1.])


class PilotOutputTest(unittest.TestCase):
    def test_all_legal_actions_and_exact_caps(self):
        import sys
        sys.path.insert(0, str(driver.ROOT / "scripts"))
        from analyze_shapley_frpr import evaluate

        class Writer:
            def __init__(self):
                self.rows = []

            def writerow(self, row):
                self.rows.append(row)

        context = dict(task="stalker", episode_id="trace", timestep=0, agent_id=0,
                       legal_actions="[1, 2, 3]", M=1)
        site = dict(greedy_action=3)
        spec = (None, None, {"env_seed": 11, "sc2_seed": 11}, {}, [], set())
        config = dict(max_exact_pairs=2, rollout_repeats=2, projection_margin=0.,
                      softmax_temperature=1., top_k=1, sampling_seed=2222)
        audit = dict(branch_rollouts=0, verified_branches=0, max_efficiency_error=0., coalitions=0,
                     projection_residual=0., stationarity_residual=0.)
        observed = []

        def verified_branch(*args, **kwargs):
            action = args[-2]
            observed.append(action)
            return dict(action=action, won=True, post_intervention_return=2. * action,
                        environment_replay_consistent=True, rnn_consistent=True, first_difference=None)

        writer = Writer()
        with patch("analyze_shapley_frpr.branch", verified_branch):
            _, rolls, values, _ = evaluate((context, site, [(1, 2)], [1e5, 0., 1., 2.]), spec,
                                           config, writer, audit)
            skip = dict(context, M=3)
            evaluate((skip, site, [(1, 2), (1, 3), (2, 3)], [1e5, 0., 1., 2.]), spec, config, writer, audit)
        self.assertEqual(observed, [1, 2, 3, 1, 2, 3])
        self.assertEqual(len(rolls), audit["verified_branches"])
        self.assertEqual(audit["coalitions"], 2)
        self.assertEqual(values[0]["phi_win"], 0.)
        self.assertEqual(skip["status"], "over_exact_cap")
        for row in writer.rows:
            self.assertEqual(json.loads(row["probabilities"])[0], 0.)
        self.assertNotEqual(context["empty_softmax_return"], context["vdn_ft_greedy_return"])


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.spec = driver.episode_spec(dict(n_agents=1, n_actions=4, state_shape=2, obs_shape=1))
        self.scheme = driver.augmented_scheme(self.spec)
        self.settings = dict(RANK_DEFAULTS, selection_k=2, selection_seed=3535,
                             shapley_results=[str(self.path)], batch_episodes=3)
        self.episodes = [self.episode(21), self.episode(47)]
        (self.path / "manifest.json").write_text(json.dumps(dict(frpr=RANK_DEFAULTS)))
        (self.path / "audit.json").write_text(json.dumps(dict(passed=True)))

    @staticmethod
    def episode(obs):
        return dict(length=2, data=dict(obs=th.full((3, 1, 1), float(obs)),
                    avail_actions=th.tensor([[[0, 1, 1, 1]], [[0, 1, 0, 1]], [[1, 0, 0, 0]]]),
                    actions_onehot=th.tensor([[[0., 1., 0., 0.]]] * 3),
                    old_q_values=th.tensor([[[100., 1., 2., 4.]], [[100., 4., 2., 1.]], [[0., 0., 0., 0.]]]),
                    terminated=th.tensor([[0], [1], [0]]), filled=th.ones((3, 1), dtype=th.long)))

    def controller(self):
        scheme, groups, _ = self.spec
        args = SimpleNamespace(device="cpu")
        mac = SimpleNamespace(args=args, hidden_states=None, init_hidden=lambda _: None)
        # Distinct current gaps per action and observation expose context mismatches.
        mac.forward = lambda batch, t: batch["obs"][:, t] * th.arange(4.) * .01
        return mac

    def write_scores(self, count=2):
        with (self.path / "shapley_values.csv").open("w", newline="") as stream:
            fields = ["task", "episode_id", "timestep", "agent_id",
                      "preferred_action", "dispreferred_action", "phi_return"]
            out = csv.writer(stream)
            out.writerow(fields)
            for index, episode in enumerate(self.episodes[:count]):
                identity = preference_episode_id("stalker", episode)
                relations = [(2, 1), (3, 1), (3, 2)]
                for i, pair in enumerate(relations):
                    out.writerow(["stalker", identity, 0, 0, *pair, i * (1 if index == 0 else -1)])
                out.writerow(["stalker", identity, 1, 0, 1, 3, .5])

    def method(self, name, **overrides):
        import numpy as np
        return SelectedFRPR(name, dict(self.settings, **overrides), np.random.default_rng(2222),
                            self.scheme, self.spec[1], lambda _: None)

    def test_context_mapping_after_reordering(self):
        self.write_scores()
        pool = dict(task="stalker", phase=0, episodes=list(reversed(self.episodes)))
        method = self.method("shapley_selected_frpr")
        prepared, audit = method.prepare_pool(pool)
        self.assertEqual((audit["covered"], audit["required"]), (4, 4))
        first, second = [ep["shapley_pair_mask"][0, 0].nonzero().tolist() for ep in prepared["episodes"]]
        self.assertEqual(first, [[1, 2], [1, 3]])
        self.assertEqual(second, [[1, 3], [2, 3]])
        self.assertNotEqual(preference_episode_id("stalker", self.episodes[0]),
                            preference_episode_id("zealot", self.episodes[0]))
        self.write_scores(count=1)
        _, audit = self.method("shapley_selected_frpr").prepare_pool(pool)
        self.assertEqual(audit["coverage"], .5)

    def test_full_budget_real_task_loss_and_random_reproducibility(self):
        from learners.ranking_preference import RankingPreference

        self.write_scores()
        pool = dict(task="stalker", phase=0, episodes=self.episodes)
        for name in SELECTED_METHODS:
            method = self.method(name, selection_k=100)
            prepared, audit = method.prepare_pool(pool)
            self.assertEqual(audit["coverage"], 1.)
            base = RankingPreference("full_rank_preference", self.settings, method.rng, self.scheme,
                                     self.spec[1], lambda _: None)
            expected = base.task_loss(self.controller(), pool, self.episodes)
            result = method.task_loss(self.controller(), prepared, prepared["episodes"])
            for a, b in zip(expected[:2], result[:2]):
                th.testing.assert_close(a, b, atol=0, rtol=0)
        scores = {(2, 1): 1., (3, 1): -1., (3, 2): 0.}
        context = ("stalker", "episode", 0, 0)
        original = choose_pairs(context, scores, self.settings, random_control=True)
        reversed_scores = dict(reversed(list(scores.items())))
        self.assertEqual(original, choose_pairs(context, reversed_scores, self.settings, random_control=True))
        self.assertEqual(len(original), 2)

    def test_driver_blocks_before_td_on_uncovered_historical_decisions(self):
        self.write_scores(count=0)
        protocol = json.loads(Path(PROTOCOL_PATH).read_text())
        for stage in protocol["stages"]:
            stage["steps"] = 16
        protocol["evaluation"].update(seeds=[3, 4], episodes_per_seed=1, interval=0)
        protocol["action_set"].update(success_episodes=1, seed_offsets=[101, 102], attempts_per_seed=2)
        protocol["full_rank_preference"] = dict(min_gap=0.)
        (self.path / "manifest.json").write_text(json.dumps(dict(frpr=dict(RANK_DEFAULTS, min_gap=0.))))
        method = "shapley_selected_frpr"
        protocol[method] = dict(selection_k=2, shapley_results=[str(self.path)])
        actual = validate(protocol, "vdn", method)
        out = self.path / "driver"
        out.mkdir()
        original_config = driver.config
        with patch.dict(driver.base.environments, {"action_set_protoss": ScriptedProtoss}), patch.object(
            driver, "config", lambda *args: dict(original_config(*args), buffer_size=8, batch_size=2)
        ), patch.object(th.cuda, "is_available", return_value=False), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            with self.assertRaisesRegex(ValueError, "Incomplete .* coverage before phase 1"):
                driver.train_sequence(actual, 2222, out)
        audit = jsonlines(out / "shapley_coverage.jsonl")[-1]
        self.assertEqual(audit["covered"], 0)
        self.assertGreater(audit["required"], 0)
        self.assertEqual({row["phase"] for row in jsonlines(out / f"{method}_updates.jsonl")}, {0})
        pool = th.load(out / "action_set_memory/phase0000/success.th", map_location="cpu")
        self.assertTrue(all(ep["episode_id"] == preference_episode_id(pool["task"], ep)
                            for ep in pool["episodes"]))

    def test_protocol_preserves_frpr_settings_and_requires_one_budget(self):
        protocol = json.loads(Path(PROTOCOL_PATH).read_text())
        protocol["full_rank_preference"] = dict(temperature=.6, min_gap=.9)
        for method in SELECTED_METHODS:
            protocol[method] = dict(shapley_results=["trusted"], selection_ratio=.25)
            actual = validate(protocol, "vdn", method)
            self.assertEqual(actual[method]["temperature"], .6)
            self.assertEqual(actual[method]["min_gap"], .9)
            protocol[method]["selection_k"] = 3
            with self.assertRaisesRegex(ValueError, "exactly one"):
                validate(protocol, "vdn", method)
            protocol.pop(method)


if __name__ == "__main__":
    unittest.main()
