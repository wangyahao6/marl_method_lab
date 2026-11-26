"""CPU-only provenance and hook ownership checks; no game process."""
import copy
import importlib.util
import hashlib
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
from common.prefix import prefix_spec, require_compatible, verify_manifest, spec_hash, file_hash, REQUIRED_ARTIFACTS
from common.prefix_implementation import training_implementation, IMPLEMENTATION_VERSION, extraction_rules_hash
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

    def manifest_fixture(self, root, spec):
        checkpoint = 'models/phase0000_20'
        for name in [*REQUIRED_ARTIFACTS, *(checkpoint + '/' + item for item in ('agent.th', 'mixer.th', 'opt.th'))]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        report = dict(stages=[dict(phase=0, task=spec['stage']['task'], checkpoint=checkpoint, win_rates={spec['stage']['task']: 0.})],
                      initial_reference={}, before_first_training={})
        (root / 'retention.json').write_text(json.dumps(report))
        manifest = dict(manifest_version=2, state_version=1, prefix_id='prefix-' + 'a' * 32,
                        prefix_spec=spec, prefix_spec_sha256=spec_hash(spec),
                        code_version=dict(git_commit='a' * 40, git_status='',
                                          source_sha256={'src/source.py': hashlib.sha256(b'source').hexdigest()}),
                        history_collection=dict(seed_offsets=[101, 102], success_episodes=2, attempts_per_seed=2),
                        files={str(path.relative_to(root)): file_hash(path) for path in root.rglob('*') if path.is_file() and path.name != 'prefix_manifest.json'})
        (root / 'prefix_manifest.json').write_text(json.dumps(manifest))
        return manifest

    def copy_dependencies(self, root, cfg):
        coverage = dict(cfg, trhr={'window_updates': 1}, a2_trace=True, evaluate=False)
        for name in {key.split(':')[0] for key in training_implementation(driver.ROOT, coverage)['components']}:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(driver.ROOT / name, path)

    def test_extraction_rules_release_is_pinned(self):
        releases = {2: '922ec57601ee6fa42352b765dd514c14c4dacd6b5ba08592c7feb04e73df6a8e',
                    3: 'acc9c8335d0f2011d992ad5d3fda0f4c3e73c1d638b1605fca4e875362691318'}
        self.assertIn(IMPLEMENTATION_VERSION, releases,
                      'Audit the rule change and register a new extraction version')
        self.assertEqual(extraction_rules_hash(), releases[IMPLEMENTATION_VERSION],
                         'Boundary/algorithm changed: bump extraction version and register its new rules hash')

    def test_environment_module_initialization_and_registration_are_covered(self):
        _, cfg = self.configuration()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.copy_dependencies(root, cfg)
            initial = training_implementation(root, cfg)
            path = root / 'drivers/continual/train.py'
            source = path.read_text()
            for before, after in (
                    ("ModuleType('envs')", "ModuleType('changed_envs')"),
                    ("sys.modules['envs'] = environment_module", "sys.modules['other'] = environment_module"),
                    ("environment_module.REGISTRY = {'sequential_protoss': SequentialProtoss}", 'environment_module.REGISTRY = {}'),
                    ("from envs import REGISTRY as environments", "from another_envs import REGISTRY as environments"),
                    ("sys.path.insert(0, str(ROOT / 'vendor/smacv2'))", "sys.path.insert(0, str(ROOT / 'other_env'))")):
                with self.subTest(initializer=before):
                    self.assertIn(before, source)
                    path.write_text(source.replace(before, after))
                    self.assertNotEqual(training_implementation(root, cfg), initial)
            # Standalone QMIX training is not executed by the action-set driver.
            path.write_text(source.replace("random.seed(o.seed)", "random.seed(o.seed + 1)"))
            self.assertEqual(training_implementation(root, cfg), initial)
            # Executable initialization added after function declarations is covered too.
            path.write_text(source + "\nenvironment_module.REGISTRY.clear()\n")
            self.assertNotEqual(training_implementation(root, cfg), initial)

    def test_restore_implementations_and_transitive_helpers_change_fingerprint(self):
        protocol, cfg = self.configuration()
        spec = prefix_spec(protocol, 8, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.copy_dependencies(root, cfg)
            initial = training_implementation(root, cfg)
            self.assertEqual(initial, spec['implementation'])
            self.assertIn('src/components/optimizer_checkpoint.py', initial['components'])
            artifacts = root / 'prefix'
            artifacts.mkdir()
            manifest = self.manifest_fixture(artifacts, spec)
            cases = (
                ('src/learners/q_learner.py', 'self.mac.load_models(path)',
                 "self.mac.load_models(path + '/other')"),
                ('drivers/continual/common/checkpoints.py', 'network.load_state_dict(state[key])',
                 "network.load_state_dict(state['agent'])"),
                ('src/components/optimizer_checkpoint.py', 'optimizer.load_state_dict(state)',
                 "optimizer.load_state_dict(dict(state, state={}))"),
                ('src/components/optimizer_checkpoint.py', 'ids = saved_ids[2:]', 'ids = saved_ids[1:]'),
                # Already covered callees must remain covered, not only their call sites.
                ('src/controllers/basic_controller.py', 'map_location=lambda storage, loc: storage',
                 "map_location='cpu'"),
                ('src/modules/agents/rnn_agent.py', 'state_dict.pop(key)', 'state_dict.pop(key, None)'),
                ('src/modules/agents/rnn_agent.py', "LEGACY_FC1_KEYS = ('fc1.weight', 'fc1.bias')",
                 "LEGACY_FC1_KEYS = ('old_fc1.weight', 'old_fc1.bias')"),
            )
            for name, before, after in cases:
                with self.subTest(restore_dependency=name, changed=before):
                    path = root / name
                    source = path.read_text()
                    self.assertIn(before, source)
                    path.write_text(source.replace(before, after))
                    try:
                        current = training_implementation(root, cfg)
                        self.assertNotEqual(current, initial)
                        requested = copy.deepcopy(spec)
                        requested['implementation'] = current
                        with self.assertRaisesRegex(ValueError, 'implementation'):
                            verify_manifest(artifacts, requested)
                    finally:
                        path.write_text(source)
            # Adding restore dependencies does not pull unrelated learner algorithms in.
            path = root / 'src/learners/q_learner.py'
            source = path.read_text()
            self.assertIn('tau = 0.1', source)
            path.write_text(source.replace('tau = 0.1', 'tau = 0.2'))  # find() is not invoked.
            self.assertEqual(training_implementation(root, cfg), initial)
            path.write_text(source)
            manifest['prefix_spec']['implementation']['version'] = 2
            manifest['prefix_spec_sha256'] = spec_hash(manifest['prefix_spec'])
            (artifacts / 'prefix_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Unsupported Prefix implementation version: 2'):
                verify_manifest(artifacts, spec)

    def test_optional_trhr_and_trace_dependencies_follow_actual_configuration(self):
        _, cfg = self.configuration()
        variants = dict(off=cfg, trhr=dict(cfg, trhr={'window_updates': 1}),
                        trace=dict(cfg, a2_trace=True),
                        evaluated=dict(cfg, a2_trace=True, evaluate=True))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.copy_dependencies(root, cfg)
            initial = {name: training_implementation(root, value) for name, value in variants.items()}
            self.assertNotIn('src/modules/layers/trhr.py', initial['off']['components'])
            self.assertNotIn('src/utils/a2_trace.py', initial['evaluated']['components'])
            path = root / 'src/modules/layers/trhr.py'
            source = path.read_text()
            path.write_text(source.replace('inputs + self.offset', 'inputs + 2 * self.offset'))
            self.assertEqual(training_implementation(root, variants['off']), initial['off'])
            self.assertNotEqual(training_implementation(root, variants['trhr']), initial['trhr'])
            # Trace-only uses helpers, not the inactive TRHR controller class.
            self.assertEqual(training_implementation(root, variants['trace']), initial['trace'])
            path.write_text(source.replace('signs.append(output.detach() > 0)', 'signs.append(output.detach() > 1)'))
            self.assertNotEqual(training_implementation(root, variants['trace']), initial['trace'])
            path.write_text(source.replace('available = batch["avail_actions"][:, :-1].bool()',
                                           'available = ~batch["avail_actions"][:, :-1].bool()'))
            self.assertNotEqual(training_implementation(root, variants['trace']), initial['trace'])
            path.write_text(source)
            path = root / 'src/utils/a2_trace.py'
            path.write_text(path.read_text().replace('self.updates += 1', 'self.updates += 2'))
            self.assertEqual(training_implementation(root, variants['off']), initial['off'])
            self.assertEqual(training_implementation(root, variants['evaluated']), initial['evaluated'])
            self.assertNotEqual(training_implementation(root, variants['trace']), initial['trace'])
            # An actually called learner helper, not just its import statement.
            path = root / 'src/utils/logging.py'
            path.write_text(path.read_text().replace('self.stats[key].append((t, value))',
                                                   'self.stats[key].append((t + 1, value))'))
            self.assertNotEqual(training_implementation(root, variants['off']), initial['off'])

    def test_manifest_requires_fields_hash_list_and_complete_checkpoint(self):
        protocol, cfg = self.configuration()
        spec = prefix_spec(protocol, 8, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = self.manifest_fixture(root, spec)
            manifest_path = root / 'prefix_manifest.json'
            self.assertEqual(verify_manifest(root, spec), baseline)
            for key in baseline:
                with self.subTest(missing=key):
                    changed = copy.deepcopy(baseline)
                    changed.pop(key)
                    manifest_path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        verify_manifest(root, spec)
            for files in ({}, [], None):
                changed = dict(baseline, files=files)
                manifest_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, 'nonempty hash list'):
                    verify_manifest(root, spec)
            for name in baseline['files']:
                with self.subTest(unlisted=name):
                    changed = copy.deepcopy(baseline)
                    changed['files'].pop(name)
                    manifest_path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, 'missing from files'):
                        verify_manifest(root, spec)
            manifest_path.write_text(json.dumps(baseline).replace('"files": {', '"files": {}, "files": {', 1))
            with self.assertRaisesRegex(ValueError, 'Duplicate Prefix JSON key'):
                verify_manifest(root, spec)
            manifest_path.write_text(json.dumps(baseline))
            for name in baseline['files']:
                with self.subTest(absent=name):
                    path = root / name
                    content = path.read_bytes()
                    path.unlink()
                    with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                        verify_manifest(root, spec)
                    path.write_bytes(content)
            changed = copy.deepcopy(baseline)
            changed['prefix_spec']['implementation']['version'] = 1
            changed['prefix_spec_sha256'] = spec_hash(changed['prefix_spec'])
            manifest_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'Unsupported Prefix implementation version'):
                verify_manifest(root, spec)

    def test_manifest_paths_and_retention_checkpoint_are_confined(self):
        protocol, cfg = self.configuration()
        spec = prefix_spec(protocol, 8, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'parent'
            root.mkdir()
            baseline = self.manifest_fixture(root, spec)
            manifest_path = root / 'prefix_manifest.json'
            for name in ('/etc/passwd', '../outside', 'models/../outside', 'C:/outside',
                         'C:\\outside', 'models\\outside', './config.json'):
                for location in ('files', 'checkpoint', 'source_sha256'):
                    with self.subTest(path=name, location=location):
                        changed = copy.deepcopy(baseline)
                        if location == 'files':
                            changed['files'][name] = 'a' * 64
                        elif location == 'source_sha256':
                            changed['code_version']['source_sha256'][name] = 'a' * 64
                        else:
                            report = json.loads((root / 'retention.json').read_text())
                            report['stages'][0]['checkpoint'] = name
                            (root / 'retention.json').write_text(json.dumps(report))
                            changed['files']['retention.json'] = file_hash(root / 'retention.json')
                        manifest_path.write_text(json.dumps(changed))
                        with self.assertRaisesRegex(ValueError, 'Unsafe Prefix Manifest path'):
                            verify_manifest(root, spec)
                        report = dict(stages=[dict(phase=0, task=spec['stage']['task'], checkpoint='models/phase0000_20', win_rates={spec['stage']['task']: 0.})],
                                      initial_reference={}, before_first_training={})
                        (root / 'retention.json').write_text(json.dumps(report))
            external = Path(tmp) / 'external.th'
            external.write_bytes(b'external')
            (root / 'prefix_replay.th').unlink()
            (root / 'prefix_replay.th').symlink_to(external)
            changed = copy.deepcopy(baseline)
            changed['files']['prefix_replay.th'] = file_hash(external)
            manifest_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'escapes parent'):
                verify_manifest(root, spec)

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
            manifest = self.manifest_fixture(root, spec)
            changed = copy.deepcopy(spec)
            # The stored fingerprint is valid; a different live implementation is refused.
            changed['implementation']['sha256'] = 'a' * 64
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
