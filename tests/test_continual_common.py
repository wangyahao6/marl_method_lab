"""Offline regressions for continual runtime and metric compatibility."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch as th

from drivers.continual.common.checkpoints import (
    continuation_state, parameter_hash, restore_networks_and_optimizer, save_stage_models,
)
from drivers.continual.common.evaluation import evaluate_block, fixed_seed_plan
from drivers.continual.common.metrics import Metrics, record_evaluation
from drivers.continual.common.runtime import episode_spec, log_task_epsilon, restore_rng, rng_state


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class CommonTest(unittest.TestCase):
    def test_original_evaluation_block_seeds_and_cleanup(self):
        self.assertEqual(fixed_seed_plan(2, True), ([91301, 91302, 91303, 91304], 32))
        self.assertEqual(fixed_seed_plan(1, False, stride=1000), ([92001, 92002], 16))
        args = SimpleNamespace(device='cuda:0', use_cuda=True, env_args={'seed': 7})
        info = dict(n_agents=1, n_actions=3, state_shape=2, obs_shape=3)
        spec = episode_spec(info)
        agent = th.nn.Linear(3, 3)
        mac = SimpleNamespace(agent=th.nn.Linear(3, 3))
        runner = Mock(t=4, batch={'reward': th.tensor([1.0, 2.0])})
        runner.env.last_info = {'battle_won': True}

        def rollout(*, test_mode):
            self.assertTrue(test_mode)
            self.assertFalse(th.is_grad_enabled())

        runner.run.side_effect = rollout
        make_runner, make_controller = Mock(return_value=runner), Mock(return_value=mac)
        record = Mock()
        result = evaluate_block(args, spec, spec[0], agent.state_dict(), 'A', 12, 2,
                                make_controller=make_controller, make_runner=make_runner,
                                make_logger=Mock(), record=record, include_return=True)
        self.assertEqual(result, [dict(env_seed=12, episode=i, won=True, length=4, return_value=3.)
                                  for i in range(2)])
        self.assertEqual([call.args[0] for call in record.call_args_list], result)
        runner.setup.assert_called_once_with(*spec, mac)
        runner.close_env.assert_called_once()
        self.assertFalse(mac.agent.training)
        self.assertTrue(th.equal(agent.weight, mac.agent.weight))
        self.assertEqual(make_runner.call_args.args[0].env_args['seed'], 12)
        self.assertEqual(args.env_args['seed'], 7)
        runner.run.side_effect = RuntimeError('example rollout failure')
        with self.assertRaises(RuntimeError):
            evaluate_block(args, spec, spec[0], agent.state_dict(), 'A', 12, 2,
                           make_controller=make_controller, make_runner=make_runner, make_logger=Mock())
        self.assertEqual(runner.close_env.call_count, 2)

    def test_legacy_continuation_and_model_artifacts(self):
        def learner():
            mac = SimpleNamespace(agent=th.nn.Linear(2, 2), action_selector=SimpleNamespace(epsilon=.4))
            result = SimpleNamespace(mac=mac, target_mac=SimpleNamespace(agent=th.nn.Linear(2, 2)),
                                     mixer=th.nn.Linear(2, 1), target_mixer=th.nn.Linear(2, 1),
                                     last_target_update_episode=13, log_stats_t=50)
            result.optimiser = th.optim.RMSprop(mac.agent.parameters(), lr=.1)
            result.save_models = lambda dest: th.save(mac.agent.state_dict(), Path(dest) / 'agent.th')
            return result

        source = learner()
        # Populate optimizer state directly, without a learner or environment step.
        source.optimiser.state[source.mac.agent.weight] = dict(step=th.tensor(1.), square_avg=th.ones(2, 2))
        runner = SimpleNamespace(t_env=234, env=SimpleNamespace(distribution_state=lambda: {'seed': 5}))
        record = continuation_state(source, runner, 'td-buffer', 16, 14)
        self.assertEqual(set(record), {'agent', 'mixer', 'target_agent', 'target_mixer', 'optimizer',
                                      'buffer', 't_env', 'episode', 'updates', 'last_target_update_episode',
                                      'log_stats_t', 'epsilon', 'rng', 'environment_distributions', 'model_hash'})
        dest_learner = learner()
        restore_networks_and_optimizer(dest_learner, record)
        self.assertEqual(parameter_hash(dest_learner), record['model_hash'])
        restored_optimizer = dest_learner.optimiser.state[dest_learner.mac.agent.weight]
        self.assertTrue(th.equal(restored_optimizer['square_avg'], th.ones(2, 2)))
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / 'models' / 'phase0_A_234'
            save_stage_models(source, dest, 234, 16, 14)
            self.assertTrue((dest / 'agent.th').is_file())
            targets = th.load(dest / 'targets.th', map_location='cpu')
            self.assertEqual(set(targets), {'target_agent', 'target_mixer', 't_env', 'episode', 'updates'})
            self.assertEqual((targets['t_env'], targets['episode'], targets['updates']), (234, 16, 14))

    def test_epsilon_events_preserve_reset_policy_and_rng(self):
        selector = Mock()
        selector.exploration_steps.return_value = 0
        selector.training_epsilon.return_value = 1.0
        runner = SimpleNamespace(mac=SimpleNamespace(action_selector=selector), t_env=52,
                                 args=SimpleNamespace(epsilon_on_switch='per_task'))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            log_task_epsilon(runner, out, 1, 'A', 'start')
            log_task_epsilon(runner, out, 1, 'A', 'end')
            runner.args.epsilon_on_switch = 'cumulative'
            log_task_epsilon(runner, out, 2, 'B', 'start')
            selector.reset_exploration.assert_called_once_with(52)
            self.assertEqual(rows(out/'task_epsilon.jsonl')[0], dict(
                event='start', phase=1, task='A', t_env=52, exploration_steps=0,
                epsilon=1., schedule='per_task'))
        saved = rng_state()
        try:
            expected = (random.random(), np.random.rand(), th.rand(()).item())
            restore_rng(saved)
            self.assertEqual((random.random(), np.random.rand(), th.rand(()).item()), expected)
        finally:
            restore_rng(saved)

    def test_metrics_keep_legacy_rows_and_scalar_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            writer = Metrics(out, 'vdn', 'softprotect_chi', 3535, group='A_chi')
            evaluation = dict(phase=2, unit='zealot', t_env=120, evaluation='periodic',
                              win_rate=.25, episodes=4)
            record_evaluation(out, writer, evaluation)
            writer.retention(2, 'zealot', 120, dict(acc_percent=25, bwt_pp=-4, fwt_pp=None))
            writer.neurons(2, 'zealot', 120, 'A', {'activation_rate': [0.2, 0.3, float('nan')]})
            writer.protection(2, 'zealot', 120, dict(protected={'A': [1]}, unit_scales={'A': [1., .1]}))
            self.assertEqual(rows(out/'evaluations.jsonl'), [evaluation])
            records = rows(out/'experiment_metrics.jsonl')
            self.assertEqual(records[0], dict(schema_version=1, run_id=out.name, algorithm='vdn',
                                             method='softprotect_chi', seed=3535, category='Performance',
                                             phase=2, task='zealot', t_env=120, metric='win_rate', value=.25,
                                             context=dict(event='periodic', episodes=4, group='A_chi')))
            self.assertEqual([(r['metric'], r['value']) for r in records[1:]],
                             [('acc_percent', 25.), ('bwt_pp', -4.), ('activation_rate', .2),
                              ('activation_rate', .3), ('protected_neuron_fraction', .5)])
            self.assertEqual([r['context']['neuron_index'] for r in records[3:5]], [0, 1])
            self.assertTrue(all(r['category'] == 'Mechanism' for r in records[3:]))


if __name__ == '__main__':
    unittest.main()
