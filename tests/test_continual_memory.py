"""History round trips using old payload shapes and CPU data."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch as th
from components.episode_buffer import EpisodeBatch

from test_action_set_preference import driver
from drivers.continual.memory import History, PreferenceHistory
from drivers.continual.memory.episodes import batch_on_device, combine, cpu_episode
from vdn_chi import reassess, save_probe_batch


class MemoryTest(unittest.TestCase):
    def test_restore_existing_preference_pools_and_sample_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fields = dict(old_q_values=th.arange(4))
            pools = [(0, 'A', [dict(data=fields)]), (1, 'B', [dict(data=fields)]),
                     (2, 'A', [dict(data=fields)]), (3, 'A', []), (4, 'C', [dict(data=fields)])]
            for phase, task, episodes in pools:
                path = PreferenceHistory.phase_path(root, phase)
                path.mkdir(parents=True)
                th.save(dict(task=task, phase=phase, episodes=episodes,
                             reference_checkpoint='models/phase/agent.th'), path / 'success.th')
            history = PreferenceHistory()
            history.restore_success(root, range(5), require_old_q=True)
            state = np.random.default_rng(3535).bit_generator.state
            rng = np.random.default_rng()
            rng.bit_generator.state = state
            expected = history.sample('C', 5, rng)
            rng.bit_generator.state = state
            self.assertEqual(history.sample('C', 5, rng), expected)
            self.assertEqual({pool['phase'] for pool, _ in expected}, {1, 2})
            self.assertEqual(sum(len(indices) for _, indices in expected), 5)
            self.assertTrue(any(pool is history[2] for pool, _ in expected))
            self.assertEqual(history[2]['reference_checkpoint'], str(root/'models/phase/agent.th'))
            invalid = dict(history[2], episodes=[{'data': {}}])
            path = PreferenceHistory.phase_path(root, 2) / 'success.th'
            History.save(path, invalid)
            with self.assertRaisesRegex(ValueError, 'predates stored old_q_values'):
                PreferenceHistory().restore_success(root, [2], require_old_q=True)
            self.assertEqual(History.load(path)['episodes'], invalid['episodes'])

    def test_chi_probe_legacy_payload_and_read_only_cpu_views(self):
        info = dict(n_agents=1, n_actions=3, state_shape=1, obs_shape=1)
        scheme, groups, preprocess = driver.episode_spec(info)
        batch = EpisodeBatch(scheme, groups, 1, 4, preprocess=preprocess)
        for step in range(3):
            batch.update(dict(avail_actions=[[[0, 1, 1]]], actions=[[[1]]]), ts=step)
        compact = cpu_episode(batch)
        self.assertEqual(compact.max_seq_length, 3)
        cpu = combine([compact, compact])
        view = batch_on_device(cpu, 'cpu')
        self.assertIsNot(view.data.transition_data, cpu.data.transition_data)
        self.assertEqual(view['avail_actions'].data_ptr(), cpu['avail_actions'].data_ptr())

        class Controller:
            args = SimpleNamespace(device='cpu')
            n_agents = 1

            def __init__(self):
                self.agent = th.nn.Module()
                self.agent.net = th.nn.Sequential(th.nn.Identity(), th.nn.Identity(), th.nn.Identity())

            def init_hidden(self, batch_size):
                self.hidden_states = th.zeros(batch_size)

            def forward(self, batch, t):
                activations = th.tensor([[8., 1., 1., 1., 1.]]).expand(batch.batch_size, -1)
                self.agent.net(activations)
                return th.tensor([0., 1., 3.]).expand(batch.batch_size, 1, 3)

        mac = Controller()
        with tempfile.TemporaryDirectory() as tmp:
            old, new = (Path(tmp) / name for name in ('old.th', 'new.th'))
            weights = th.ones((2, 2))
            th.save(dict(batch=cpu, weights=weights), old)
            save_probe_batch(mac, cpu, new)
            saved = History.load(new)
            self.assertEqual(set(saved), {'batch', 'weights'})
            self.assertTrue(th.equal(saved['weights'], weights))
            result = reassess(mac, {0: old, 1: new}, 'A', top_k=None, threshold=3.)['A']
            self.assertEqual(result['sets']['overall'], [0])
            self.assertEqual(result['tasks'][0]['scores'], result['tasks'][1]['scores'])
            self.assertEqual(result['tasks'][0]['states'], 4)
            self.assertFalse(hasattr(mac, 'hidden_states'))
            self.assertTrue(mac.agent.training)


if __name__ == '__main__':
    unittest.main()
