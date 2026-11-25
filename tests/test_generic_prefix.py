"""Generic Prefix contracts using the existing small CPU episode fixture."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch as th

from test_shared_tkpr_k1 import ForkableProtoss
from test_action_set_preference import driver, jsonlines
from common.prefix import prefix_spec, require_compatible, resolve_prefix, verify_manifest
from common.runtime import rng_state
from protection import digest


class GenericPrefixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def protocol(self):
        path = driver.ROOT / 'configs/continual/action_preference/szc/szc_vdn64_tkpr_k1_shared.json'
        raw = json.loads(path.read_text())
        for stage in raw['stages']:
            stage['steps'] = 20
        raw['evaluation'].update(seeds=[3, 4], episodes_per_seed=1, interval=0,
                                 focus_seeds=None, focus_episodes_per_seed=None)
        raw['action_set'].update(success_episodes=2, seed_offsets=[101, 102], attempts_per_seed=2)
        return raw

    def test_spec_accepts_suffix_and_method_changes_and_reports_training_differences(self):
        parent = driver.validate(self.protocol(), 'vdn', 'baseline')
        branch = copy.deepcopy(parent)
        branch['method'] = 'topk_preference'
        branch['topk_preference']['weight'] = 9
        branch['stages'][1:] = [dict(task='colossus', steps=77)]
        branch['tasks']['zealot']['ally'] = [2, 3, 0]
        branch['evaluation']['interval'] = 123
        branch['epsilon_on_switch'] = 'cumulative_no_restart'
        require_compatible(prefix_spec(parent, 8), prefix_spec(branch, 8))
        for field, change, error in (
                ('seed', lambda p: None, 'seed'),
                ('budget', lambda p: p['stages'][0].update(steps=21), 'stage.steps'),
                ('task', lambda p: p['stages'][0].update(task='zealot'), 'stage.task'),
                ('roster', lambda p: p['tasks']['stalker'].update(ally=[2, 3, 0]), 'roster.ally'),
                ('network', lambda p: p.update(algorithm='vdn_1ff_128'), 'algorithm'),
                ('optimizer', lambda p: p.update(optimizer='adam'), 'training.optimizer'),
                ('environment', lambda p: p.update(sc2_game_version='other'), 'game_version')):
            with self.subTest(field=field):
                changed = copy.deepcopy(branch)
                change(changed)
                with self.assertRaisesRegex(ValueError, error):
                    require_compatible(prefix_spec(parent, 8), prefix_spec(changed, 9 if field == 'seed' else 8))
        for key in ('lr', 'buffer_size', 'batch_size', 'epsilon_anneal_time', 'rnn_hidden_dim', 'gamma'):
            cfg = driver.config(parent, 8, Path('/tmp/unused'))
            changed = copy.deepcopy(cfg)
            changed[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'training.' + key):
                require_compatible(prefix_spec(parent, 8, cfg), prefix_spec(branch, 8, changed))

    def test_environment_restore_preserves_child_rosters_and_distribution_rng(self):
        from hetero_environment import ExactProtoss, ExactTeams
        parent_rosters = dict(stalker=dict(ally=[5, 0, 0], enemy=[5, 0, 0]))
        child_rosters = dict(parent_rosters, new_task=dict(ally=[0, 5, 0], enemy=[0, 0, 5]))
        env = ExactProtoss.__new__(ExactProtoss)
        env.wrapped = SimpleNamespace(env_key_to_distribution_map=dict(
            team_gen=ExactTeams(child_rosters), rng=np.random.default_rng(77)))
        env.task = 'stalker'
        saved_rng = np.random.default_rng(9)
        saved = dict(team_gen=ExactTeams(parent_rosters), rng=saved_rng)
        env.restore_distribution(saved)
        env.set_task('new_task')
        teams = env.wrapped.env_key_to_distribution_map['team_gen'].generate()['team_gen']
        self.assertEqual(teams['ally_team'], ['zealot'] * 5)
        self.assertEqual(teams['enemy_team'], ['colossus'] * 5)
        self.assertEqual(env.wrapped.env_key_to_distribution_map['rng'].random(), saved_rng.random())
        self.assertEqual(saved['team_gen'].rosters, parent_rosters)

    def test_baseline_prefix_cross_sequence_restore_and_collection_isolation(self):
        self.check_restore('ft', ('baseline', 'topk_preference', 'full_rank_preference', 'aspr_preference',
                                  'aspr_max_preference', 'aspr_3g_preference', 'tkpr_refresh', 'tkpr_cpr'))

    def test_er_retains_prefix_replay_and_uses_child_order(self):
        self.check_restore('er', ('baseline',))

    def test_legacy_parent_without_manifest_keeps_original_restore_path(self):
        raw = self.protocol()
        protocol = driver.validate(raw, 'vdn', 'topk_preference')
        original_config = driver.config

        def configure(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                driver.base.environments, {'action_set_protoss': ForkableProtoss}), patch.object(
                driver, 'config', configure), patch.object(th.cuda, 'is_available', return_value=False):
            parent, child = Path(tmp) / 'parent', Path(tmp) / 'child'
            parent.mkdir()
            child.mkdir()
            driver.train_sequence(protocol, 8, parent, role='shared-prefix')
            state = th.load(parent / 'return_state.th')
            state['protocol'] = protocol
            state['preference_rng'] = np.random.default_rng(8).bit_generator.state
            for key in ('state_version', 'online_hidden', 'target_hidden', 'runner_t', 'full_state_version'):
                state.pop(key)
            th.save(state, parent / 'return_state.th')
            (parent / 'prefix_manifest.json').unlink()
            report_path = parent / 'retention.json'
            report = json.loads(report_path.read_text())
            report['initial_reference'] = dict(zealot=0., colossus=0.)
            report['before_first_training'] = dict(zealot=0.)
            report_path.write_text(json.dumps(report))
            driver.train_sequence(protocol, 8, child, role='shared-branch', parent=parent)
            restored = json.loads((child / 'branch_restore.json').read_text())
            self.assertIsNone(restored.get('prefix_id'))
            self.assertEqual(restored['model_hash'], state['model_hash'])

    def check_restore(self, replay, methods):
        raw = self.protocol()
        raw['replay_on_switch'] = 'ft'
        raw['stages'] = raw['stages'][:1]
        original_config = driver.config

        def configure(*args):
            return dict(original_config(*args), buffer_size=8, batch_size=2, target_update_interval=3)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
                driver.base.environments, {'action_set_protoss': ForkableProtoss}), patch.object(
                driver, 'config', configure), patch.object(th.cuda, 'is_available', return_value=False):
            out = Path(tmp).resolve()
            parent = out / 'prefix'
            parent.mkdir()
            protocol = driver.validate(raw, 'vdn', 'baseline')
            collect = driver.collect_memory
            boundary = {}

            def observe_collection(*args, **kw):
                boundary['rng'] = digest(rng_state())
                result = collect(*args, **kw)
                # Deliberately perturb every global RNG after real collection: the saved boundary must survive.
                random.random()
                np.random.random()
                th.rand(3)
                return result

            with patch.object(driver, 'collect_memory', observe_collection):
                driver.train_sequence(protocol, 8, parent, role='shared-prefix')
            state = th.load(parent / 'return_state.th', map_location='cpu')
            self.assertEqual(boundary['rng'], digest(state['rng']))
            self.assertEqual(digest(rng_state()), digest(state['rng']))
            self.assertIsNone(state['preference_rng'])
            self.assertEqual(state['protocol']['method'], 'baseline')
            saved_buffer = th.load(parent / 'prefix_replay.th')
            self.assertGreater(saved_buffer.episodes_in_buffer, 0)
            pool = th.load(parent / 'action_set_memory/phase0000/success.th')
            self.assertTrue(pool['episodes'])
            self.assertTrue(all('old_q_values' in ep['data'] and 'good_mask' not in ep for ep in pool['episodes']))
            manifest = json.loads((parent / 'prefix_manifest.json').read_text())
            self.assertEqual(resolve_prefix(manifest['prefix_id'], out), parent)
            checkpoint = parent / json.loads((parent / 'retention.json').read_text())['stages'][0]['checkpoint']
            for method in methods:
                child_raw = copy.deepcopy(raw)
                child_raw['replay_on_switch'] = replay
                child_raw['stages'] += [dict(task='colossus', steps=14), dict(task='zealot', steps=10)]
                child_raw['action_set']['delta_by_task'] = dict(stalker=.5, zealot=.5, colossus=.5)
                if method == 'tkpr_refresh':
                    child_raw['action_set'].update(batch_episodes_per_task=2, refresh_success_episodes=1)
                child = driver.validate(child_raw, 'vdn', method)
                run = out / method
                run.mkdir()
                learners, runners, seen = [], [], []
                learner_factory = driver.base.learners['q_learner']
                runner_factory = driver.base.EpisodeRunner
                write = driver.base.write
                start_replay = driver.start_replay_task
                buffers = []

                def learner(*args, **kwargs):
                    value = learner_factory(*args, **kwargs)
                    learners.append(value)
                    return value

                def runner(*args, **kwargs):
                    value = runner_factory(*args, **kwargs)
                    runners.append(value)
                    return value

                def inspect_restore(path, value):
                    if path.name == 'branch_restore.json':
                        seen.append(True)
                        model, live = learners[0], runners[0]
                        self.assertEqual(driver.base.parameter_hash(model), state['model_hash'])
                        self.assertEqual(digest(model.optimiser.state_dict()), digest(th.load(checkpoint / 'opt.th')))
                        self.assertEqual(digest(model.mac.hidden_states), digest(state['online_hidden']))
                        self.assertEqual(digest(model.target_mac.hidden_states), digest(state['target_hidden']))
                        self.assertEqual(digest(rng_state()), digest(state['rng']))
                        self.assertEqual((live.t_env, live.t), (state['t_env'], state['runner_t']))
                        self.assertEqual((live.mac.action_selector.epsilon, live.mac.action_selector.exploration_start),
                                         (state['epsilon'], state['exploration_start']))
                        self.assertEqual(live.env.distribution_state(), state['environment_distributions'])
                        if model.args.method != 'baseline':
                            # The history is loaded by the driver before this record.
                            self.assertEqual(value['prefix_id'], manifest['prefix_id'])
                    write(path, value)

                def inspect_buffer(buffer, task):
                    before = buffer.episodes_in_buffer
                    result = start_replay(buffer, task)
                    buffers.append((before, result))
                    if replay == 'er':
                        self.assertEqual(buffer.order, ('stalker', 'colossus', 'zealot'))
                    return result

                with patch.dict(driver.base.learners, {'q_learner': learner}), patch.object(
                        driver.base, 'EpisodeRunner', runner), patch.object(driver.base, 'write', inspect_restore), patch.object(
                        driver, 'start_replay_task', inspect_buffer):
                    driver.train_sequence(child, 8, run, role='shared-branch', parent=parent)
                self.assertEqual(seen, [True])
                if replay == 'ft':
                    self.assertEqual([before for before, _ in buffers], [0, 0])
                else:
                    self.assertEqual(buffers[0][0], saved_buffer.episodes_in_buffer)
                    self.assertGreater(buffers[0][1]['replay_task_episodes']['stalker'], 0)
                self.assertEqual([row['task'] for row in jsonlines(run / 'transitions.jsonl')], ['colossus', 'zealot'])
                if method != 'baseline':
                    log_name = 'tkpr_cpr_preference' if method == 'tkpr_cpr' else method
                    updates = jsonlines(run / f'{log_name}_updates.jsonl')
                    self.assertTrue(updates[0]['historical_tasks'])
            (parent / 'initial_agent.th').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                verify_manifest(parent, manifest['prefix_spec'])


if __name__ == '__main__':
    unittest.main()
