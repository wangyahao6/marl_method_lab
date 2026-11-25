"""CPU-only provenance and hook ownership checks; no game process."""
import copy
import importlib.util
import json
import logging
from pathlib import Path
import platform
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch as th

from test_action_set_preference import driver, PROTOCOL_PATH
from common.prefix import prefix_spec, require_compatible, verify_manifest
from common.prefix_implementation import training_implementation
from utils.hook_lifecycle import HookHandles


class ReliabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.set_num_threads(1)

    def configuration(self):
        protocol = driver.validate(json.loads(PROTOCOL_PATH.read_text()), 'vdn', 'baseline')
        with patch.object(th.cuda, 'is_available', return_value=False):
            cfg = driver.config(protocol, 8, Path('/tmp/unused'))
        return protocol, cfg

    def test_runtime_python_and_cuda_metadata(self):
        protocol, cfg = self.configuration()
        spec = prefix_spec(protocol, 8, cfg)
        self.assertEqual(spec['runtime']['python'], platform.python_version())
        self.assertEqual(spec, prefix_spec(protocol, 8))
        self.assertIsNone(spec['runtime']['cuda'])
        cfg['use_cuda'] = True
        with patch.object(th.cuda, 'get_device_name', return_value='test-device'), patch.object(
                th.cuda, 'get_device_capability', return_value=(8, 0)), patch.object(
                th.backends.cudnn, 'version', return_value=9000):
            runtime = prefix_spec(protocol, 8, cfg)['runtime']
        self.assertEqual(runtime['cuda']['device'], 'test-device')
        self.assertEqual(runtime['cuda']['capability'], [8, 0])
        self.assertEqual(runtime['cuda']['cudnn'], 9000)

    def test_baseline_dependency_changes_reject_but_consumers_and_docs_do_not(self):
        _, cfg = self.configuration()
        initial = training_implementation(driver.ROOT, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in {key.split(':')[0] for key in initial['components']}:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(driver.ROOT / name, path)
            self.assertEqual(training_implementation(root, cfg), initial)
            (root / 'docs').mkdir()
            (root / 'docs/unrelated.md').write_text('different report')
            (root / 'src/learners/ranking_preference.py').write_text('changed method implementation')
            path = root / 'drivers/continual/action_set_train.py'
            source = path.read_text()
            path.write_text(source.replace('status("updating_references")', 'status("new CPR diagnostic")').replace(
                'row["learner_update"] = updates + 1', 'row["learner_update"] = updates + 2'))
            self.assertEqual(training_implementation(root, cfg), initial)
            # Budget and TD update orchestration are covered even in the mixed driver.
            path.write_text(source.replace('learner.train(sample, runner.t_env, episode,',
                                           'learner.train(sample, runner.t_env + 1, episode,'))
            changed = training_implementation(root, cfg)
            with self.assertRaisesRegex(ValueError, 'implementation'):
                require_compatible(dict(implementation=initial), dict(implementation=changed))
            path.write_text(source)
            learner = root / 'src/learners/q_learner.py'
            learner.write_text(learner.read_text().replace('loss.backward()', '(loss * 2).backward()'))
            self.assertNotEqual(training_implementation(root, cfg), initial)
            path.write_text(source.replace('runner.t_env < end:', 'runner.t_env <= end:'))
            with self.assertRaisesRegex(ValueError, 'Unrecognized Prefix TD loop'):
                training_implementation(root, cfg)

    def test_manifest_versions_are_explicit_and_fingerprint_is_checked(self):
        protocol, cfg = self.configuration()
        spec = prefix_spec(protocol, 8, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Legacy Prefix without'):
                verify_manifest(root, spec)
            with patch.object(driver.base, 'EpisodeRunner') as runner_factory:
                with self.assertRaisesRegex(ValueError, 'Legacy Prefix without'):
                    driver.train_sequence(protocol, 8, root / 'child', role='shared-branch', parent=root)
                runner_factory.assert_not_called()
            path = root / 'prefix_manifest.json'
            path.write_text(json.dumps(dict(state_version=1, prefix_spec=spec)))
            with self.assertRaisesRegex(ValueError, 'No automatic upgrade'):
                verify_manifest(root, spec)
            from common.prefix import spec_hash
            manifest = dict(manifest_version=2, state_version=1, prefix_spec=spec,
                            prefix_spec_sha256=spec_hash(spec), files={})
            path.write_text(json.dumps(manifest))
            changed = copy.deepcopy(spec)
            changed['implementation']['sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'implementation.sha256'):
                verify_manifest(root, changed)
            manifest['manifest_version'] = 999
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'unsupported Prefix Manifest'):
                verify_manifest(root, spec)

    def test_handles_preserve_external_hooks_on_normal_and_exception_exit(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                module = th.nn.Linear(2, 2)
                external, temporary = [], []
                external_handle = module.register_forward_hook(lambda *args: external.append(1))
                try:
                    with self.assertRaisesRegex(RuntimeError, 'failure') if failure else HookHandles() as unused:
                        if failure:
                            with HookHandles() as handles:
                                handles.add(module.register_forward_hook(lambda *args: temporary.append(1)))
                                module(th.zeros(1, 2))
                                raise RuntimeError('failure')
                        else:
                            unused.add(module.register_forward_hook(lambda *args: temporary.append(1)))
                            module(th.zeros(1, 2))
                    module(th.zeros(1, 2))
                    self.assertEqual((len(external), len(temporary)), (2, 1))
                finally:
                    external_handle.remove()

    def models(self):
        _, cfg = self.configuration()
        cfg.update(n_agents=5, n_actions=11, state_shape=3, use_cuda=False, device='cpu')
        args = SimpleNamespace(**cfg)
        spec = driver.episode_spec(dict(n_agents=5, n_actions=11, obs_shape=4, state_shape=3))
        scheme, groups = driver.augmented_scheme(spec), spec[1]
        batch = driver.base.ReplayBuffer(spec[0], groups, 1, 3, preprocess=spec[2])
        mac = driver.base.controllers[args.mac](scheme, groups, args)
        return args, scheme, groups, batch, mac

    def test_learner_removal_leaves_online_and_target_external_observers(self):
        args, scheme, groups, batch, mac = self.models()
        seen = []
        handle = mac.agent.net[2].register_forward_hook(lambda *unused: seen.append(1))
        learner = driver.base.learners[args.learner](mac, scheme, driver.base.Logger(logging.getLogger('hook-test')), args)
        try:
            learner.remove_activation_hooks()
            learner.remove_activation_hooks()
            x = th.zeros(1, mac.agent.net[0].in_features)
            h = th.zeros(1, args.rnn_hidden_dim)
            mac.agent(x, h)
            learner.target_mac.agent(x, h)
            self.assertEqual(seen, [1, 1])
            self.assertTrue(all(not value.any() for value in learner.neu_data.values()))
        finally:
            learner.remove_activation_hooks()
            handle.remove()

    def test_real_analysis_unroll_removes_temporary_handles_after_failure(self):
        path = driver.ROOT / 'scripts/network_similarity.py'
        module_spec = importlib.util.spec_from_file_location('network_hook_test', path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        for failure in (False, True):
            args, scheme, groups, batch, mac = self.models()
            external = Mock(return_value=None)
            external_handle = mac.agent.net[2].register_forward_hook(external)
            tracked = []
            for layer in (mac.agent.net[2], mac.agent.net[4]):
                register = layer.register_forward_hook
                def track(callback, register=register):
                    handle = register(callback)
                    handle.remove = Mock(wraps=handle.remove)
                    tracked.append(handle)
                    return handle
                layer.register_forward_hook = track
            try:
                with patch.dict(module.REGISTRY, {args.mac: lambda *unused: mac}):
                    if failure:
                        with patch.object(mac, 'forward', side_effect=RuntimeError('unroll failure')):
                            with self.assertRaisesRegex(RuntimeError, 'unroll failure'):
                                module.forward(vars(args), mac.agent.state_dict(), batch)
                    else:
                        module.forward(vars(args), mac.agent.state_dict(), batch)
                self.assertEqual(len(tracked), 2)
                self.assertTrue(all(handle.remove.called for handle in tracked))
                mac.agent(th.zeros(1, mac.agent.net[0].in_features), th.zeros(1, args.rnn_hidden_dim))
                self.assertTrue(external.called)
            finally:
                external_handle.remove()


if __name__ == '__main__':
    unittest.main()
