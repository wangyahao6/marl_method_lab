"""Full A-prefix round trip and independent ER/K1/CPR/Refresh branch ownership."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

from test_action_set_preference import driver, ScriptedProtoss, PROTOCOL_PATH, jsonlines
from drivers.continual.common.preference_prefix import file_sha256, reference_input_hash
from drivers.continual.memory import History
from test_tkpr_cpr import pool as cpr_pool, MAC, SCHEME, GROUPS
from drivers.continual.memory.cpr import update_pool
from drivers.continual.action_set_protocol import CPR_DEFAULTS


class ForkableScriptedProtoss(ScriptedProtoss):
    def distribution_state(self):
        return {'episodes': self.episodes}

    def restore_distribution(self, state):
        self.episodes = state['episodes']


class FullPrefixTest(unittest.TestCase):
    def test_complete_state_and_four_independent_branches(self):
        raw = json.loads(PROTOCOL_PATH.read_text())
        raw.update(full_shared_prefix=True, phase_boundary='cumulative', epsilon_on_switch='per_task')
        for stage in raw['stages']:
            stage['steps'] = 16
        raw['evaluation'].update(seeds=[3, 4], episodes_per_seed=1, interval=0)
        raw['action_set'].update(batch_episodes=8, success_episodes=2,
                                 seed_offsets=[101, 102], attempts_per_seed=2)
        raw['topk_preference'] = dict(k=1)
        original_config, original_sample = driver.config, driver.td_sample
        marker, first_batches = [None], {}

        def tiny_config(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2)

        def sample(*args):
            batch = original_sample(*args)
            if marker[0] != 'prefix' and marker[0] not in first_batches:
                first_batches[marker[0]] = set(batch['obs'][:, 0, 0, 0].tolist())
            return batch

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            driver.base.environments, {'action_set_protoss': ForkableScriptedProtoss}
        ), patch.object(driver, 'config', tiny_config), patch.object(
            th.cuda, 'is_available', return_value=False
        ), patch.object(driver, 'td_sample', sample), np.testing.suppress_warnings() as quiet:
            quiet.filter(UserWarning)
            root = Path(tmp)
            parent = root/'prefix';parent.mkdir();marker[0] = 'prefix'
            protocol = driver.validate(raw, 'vdn_1ff', 'topk_preference')
            driver.train_sequence(protocol, 7, parent, role='shared-prefix')
            state = th.load(parent/'return_state.th', map_location='cpu')
            for key in ('agent','target_agent','optimizer','replay_snapshot','rng','preference_rng',
                        't_env','episode','updates','epsilon','exploration_start','environment_distributions'):
                self.assertIn(key, state)
            saved_replay = th.load(parent/'prefix_replay.th', map_location='cpu')
            self.assertGreater(saved_replay.episodes_in_buffer, 0)
            self.assertEqual(len(json.loads((parent/'retention.json').read_text())['stages']), 1)
            original_parent = {str(p.relative_to(parent)): file_sha256(p)
                               for p in parent.rglob('*.th')}
            prefix_pool = th.load(parent/'action_set_memory/phase0000/success.th')
            prefix_input_hash = reference_input_hash(prefix_pool)
            for method in ('baseline', 'topk_preference', 'tkpr_cpr', 'tkpr_refresh'):
                branch_raw = copy.deepcopy(raw)
                branch_raw['id'] = 'different_method_id_' + method
                if method == 'baseline':
                    branch_raw['replay_on_switch'] = 'er'
                if method == 'tkpr_refresh':
                    branch_raw['action_set'].update(batch_episodes_per_task=8, refresh_success_episodes=1)
                branch = root/method;branch.mkdir();marker[0] = method
                effective = driver.validate(branch_raw, 'vdn_1ff', method)
                driver.train_sequence(effective, 7, branch, role='shared-branch', parent=parent)
                restore = json.loads((branch/'branch_restore.json').read_text())
                self.assertEqual(restore['model_hash'], restore['parent_model_hash'])
                self.assertEqual(restore['prefix_checkpoint_sha256'],
                                 json.loads((parent/'prefix_state_manifest.json').read_text())['checkpoint_sha256'])
                self.assertEqual(restore['parent_memory_input_hash'], prefix_input_hash)
                self.assertEqual(restore['branch_memory_input_hash'], prefix_input_hash)
                self.assertEqual(restore['restored_replay_episodes'], saved_replay.episodes_in_buffer)
                self.assertTrue((branch/'action_set_memory/phase0000/success.th').exists())
                self.assertTrue((branch/'final_replay.th').exists())
                final_state = th.load(branch/'return_state.th', map_location='cpu')
                self.assertIn('target_agent', final_state)
                self.assertIn('optimizer', final_state)
                self.assertEqual(final_state['replay_snapshot'], 'final_replay.th')
                self.assertFalse(any(r['phase']==0 for r in jsonlines(branch/'evaluations.jsonl')))
                if method == 'baseline':
                    replay = th.load(branch/'final_replay.th', map_location='cpu')
                    self.assertEqual(replay.mode, 'er')
                    self.assertEqual(first_batches[method], {1., 2.})
                else:
                    self.assertEqual(first_batches[method], {2.})
                for relative, digest in original_parent.items():
                    self.assertEqual(file_sha256(parent/relative), digest)
                self.assertEqual(json.loads((branch/'status.json').read_text())['status'], 'completed')

    def test_confidence_diagnostics_do_not_change_selection(self):
        memory, mac = cpr_pool(), MAC()
        result = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 1, .8, .8, 'phase1')
        self.assertEqual(result['updated_states'], 10)
        for key in ('old_reference_confidence', 'new_reference_confidence'):
            stats = result[key]
            self.assertEqual(stats['n'], 10)
            self.assertLessEqual(stats['p10'], stats['median'])
            self.assertLessEqual(stats['median'], stats['p90'])
        failed = update_pool(mac, memory, SCHEME, GROUPS, CPR_DEFAULTS, 2, .2, .8, 'phase2')
        self.assertIsNone(failed['new_reference_confidence'])
        self.assertEqual(failed['old_reference_confidence']['n'], 10)


if __name__ == '__main__':
    unittest.main()
