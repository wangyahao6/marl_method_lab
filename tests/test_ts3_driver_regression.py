"""Exercise the TS3 prefix evaluation and save path with a stub environment."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch as th

from test_action_set_preference import ROOT, driver, jsonlines


class TS3DriverTest(unittest.TestCase):
    def test_prefix_evaluates_and_saves_with_original_layout(self):
        protocol = json.loads((ROOT / 'configs/continual/ts3_agent_3535.json').read_text())
        # Zero budget exercises both boundary evaluations without collecting TD data.
        protocol['thresholds'] = [0] * len(protocol['order'])
        from hetero_environment import TEAMS
        with patch.object(driver.base, 'PROTOCOL', protocol), patch.dict(TEAMS):
            ts3 = importlib.import_module('ts3_train')
        self.addCleanup(sys.modules.pop, 'ts3_train', None)

        class Runner:
            def __init__(self, args, logger):
                self.args = args
                self.t_env = 0
                self.t = 3
                self.env = SimpleNamespace(set_task=Mock(), last_info={'battle_won': True},
                                           distribution_state=lambda: {'rng': 1})

            def get_env_info(self):
                return dict(n_agents=1, n_actions=3, state_shape=2, obs_shape=2, episode_limit=3)

            def setup(self, scheme, groups, preprocess, mac):
                self.mac = mac
                if set(scheme) != {'state', 'obs', 'actions', 'avail_actions', 'reward', 'terminated'}:
                    raise AssertionError(scheme)

            def run(self, *, test_mode):
                if not test_mode:
                    raise AssertionError('test must only evaluate')

            def close_env(self):
                pass

        def make_controller(scheme, groups, args):
            selector = Mock(epsilon=.4)
            selector.exploration_steps.return_value = 0
            selector.training_epsilon.return_value = .4
            return SimpleNamespace(agent=th.nn.Linear(2, 3), action_selector=selector)

        def make_learner(mac, scheme, logger, args):
            result = SimpleNamespace(mac=mac, mixer=th.nn.Linear(1, 1), target_mixer=th.nn.Linear(1, 1),
                                     target_mac=make_controller(scheme, {}, args), cuda=Mock(),
                                     log_stats_t=-1, last_target_update_episode=0)
            result.optimiser = th.optim.RMSprop(mac.agent.parameters(), lr=.1)
            result.save_models = lambda dest: th.save(mac.agent.state_dict(), Path(dest) / 'agent.th')
            return result

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'MARL_RUN_DIR': tmp}), \
                patch.object(sys, 'argv', ['ts3_train.py', '--role', 'prefix', '--seed', '3535']), \
                patch.object(ts3.base, 'EpisodeRunner', Runner), \
                patch.dict(ts3.base.controllers, {'basic_mac': make_controller}), \
                patch.dict(ts3.base.learners, {'q_learner': make_learner}), \
                patch.object(ts3, 'qmix_select', return_value={}):
            ts3.main()
            out = Path(tmp)
            rows = jsonlines(out / 'evaluations.jsonl')
            self.assertEqual([(row['label'], row['episodes']) for row in rows],
                             [('phase_start', 32), ('stage_end', 128)])
            normalized = jsonlines(out / 'experiment_metrics.jsonl')
            self.assertEqual([row['value'] for row in normalized], [row['win_rate'] for row in rows])
            self.assertTrue((out / 'models/phase0_A_0/targets.th').is_file())
            self.assertEqual(th.load(out / 'prefix_state.th')['t_env'], 0)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
