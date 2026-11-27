"""Index evidence and queries without importing an environment or training runtime."""
import importlib.util
import contextlib
import io
import json
import sqlite3
import hashlib
import argparse
import ast
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/experiment_index.py'
SPEC = importlib.util.spec_from_file_location('experiment_index', SCRIPT)
index = importlib.util.module_from_spec(SPEC)
with patch.object(sys, 'path', [str(SCRIPT.parent), *sys.path]):
    SPEC.loader.exec_module(index)


class ExperimentIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write('docs/results/catalog.json', {'experiments': []})
        self.write('docs/architectures/architectures.json', json.loads(
            (SCRIPT.parents[1] / 'docs/architectures/architectures.json').read_text()))
        self.write('docs/environments/environments.json', {
            'schema_version': 1, 'environments': [{'id': 'smacv2'}]})

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def builder(self):
        return index.Builder(self.root)

    def architecture_code(self):
        """Copy only existing source for a static Git fixture; never import its dependencies."""
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=self.root, text=True).strip()
        git('init', '-q')
        for path in ('src/modules/agents/rnn_agent.py', 'src/modules/agents/single_ff_rnn_agent.py',
                     'src/modules/agents/__init__.py', 'src/modules/mixers/vdn.py', 'src/modules/layers/act_layer.py'):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((SCRIPT.parents[1] / path).read_text())
        git('add', '.')
        git('-c', 'user.name=Index Test', '-c', 'user.email=index@example.invalid', 'commit', '-qm', 'architecture fixture')
        return git, git('rev-parse', 'HEAD')

    def candidate(self, builder, rid='actual_attempt001', **kwargs):
        config = dict(env='action_set_protoss', algorithm='vdn', method='baseline',
                      seed=3535, replay_on_switch='clear')
        config.update(kwargs.pop('config', {}))
        row = dict(status='completed', commit='abc123', _completed_phases=[0, 1])
        row.update(kwargs.pop('row', {}))
        builder.candidate(rid, row, config,
                          {'id': 'protocol', 'order': ['A', 'B'], 'tasks': {'A': [5, 0], 'B': [0, 5]}},
                          {}, kwargs.pop('source', 'runs.csv#row=2'),
                          kwargs.pop('experiment', 'experiment'), kwargs.pop('results', []), **kwargs)

    def test_configuration_is_not_execution_and_exit_zero_is_not_completion(self):
        self.assertFalse(index.started({'seed': 3535}, {}, {'started': True}))
        self.assertFalse(index.started({}, {'status.json': {'t_env': 10}},
                                       {'request': {'dry_run': True}}))
        self.assertEqual(index.status_from_runtime({'launcher_status.json': {'exit_code': 0}}), 'unknown')
        self.assertTrue(index.started({}, {'launcher_status.json': {'state': 'failed'}}, {'started': True}))

    def test_configuration_only_benchmark_has_no_runs(self):
        self.write('configs/continual/example.json', {'id': 'planned', 'order': ['A', 'B']})
        with patch.object(index.Builder, 'local_runtime'):
            b = self.builder().build()
        self.assertEqual(b.final, [])
        self.assertTrue(b.benchmarks)
        self.assertTrue(all(x['evidence_level'] == 'code_only' for x in b.benchmarks.values()))

    def test_local_worktrees_are_only_scanned_with_explicit_opt_in(self):
        with patch.object(index.Builder, 'local_runtime') as local:
            b = self.builder().build()
            local.assert_not_called()
            self.assertFalse(json.loads(index.artifacts(b)['docs/results/index_audit.json'])['local_runtime_included'])
        with patch.object(index.Builder, 'local_runtime') as local:
            index.Builder(self.root, include_local=True).build()
            local.assert_called_once()
        with patch.object(index.Builder, 'local_runtime') as local:
            with patch('sys.argv', ['experiment_index', '--root', str(self.root), 'build', '--include-local']):
                with contextlib.redirect_stdout(io.StringIO()):
                    index.main()
            local.assert_called_once()

    def test_dedup_memberships_and_empty_results_stay_runtime_only(self):
        b = self.builder()
        self.candidate(b, experiment='first')
        self.candidate(b, experiment='second', source='other/runs.csv#row=4')
        b.merge(); b.complete_results()
        self.assertEqual(len(b.final), 1)
        r = b.final[0]
        self.assertEqual(r['experiment_ids'], ['first', 'second'])
        self.assertEqual(r['result_paths'], [])
        self.assertEqual(r['result_status'], 'runtime_only')
        self.assertEqual(r['attempt_source'], 'run_id_label')
        self.assertEqual(r['retry_of'], 'unknown')

    def test_ft_er_variants_and_budget_independent_benchmarks(self):
        b = self.builder()
        self.candidate(b, 'ft')
        self.candidate(b, 'er', config={'replay_on_switch': 'keep'})
        b.merge()
        ft, er = b.final[1], b.final[0]
        self.assertEqual(ft['benchmark_id'], er['benchmark_id'])
        self.assertNotEqual(ft['method_variant_id'], er['method_variant_id'])
        a = index.task_spec({}, {'order': ['A'], 'tasks': {'A': [5]}, 'thresholds': [100]}, 'smacv2')
        z = index.task_spec({}, {'order': ['A'], 'tasks': {'A': [5]}, 'thresholds': [200]}, 'smacv2')
        self.assertEqual(a, z)

    def test_method_identity_excludes_backbone_and_configuration_alias(self):
        b = self.builder()
        self.candidate(b, 'two', config={'algorithm': 'vdn', 'agent': 'rnn', 'rnn_hidden_dim': 64, 'mixer': 'vdn'})
        self.candidate(b, 'one', config={'algorithm': 'vdn_1ff', 'agent': 'single_ff_rnn', 'rnn_hidden_dim': 64, 'mixer': 'vdn'})
        self.candidate(b, 'wide', config={'algorithm': 'vdn_1ff_128', 'agent': 'single_ff_rnn', 'rnn_hidden_dim': 128, 'mixer': 'vdn'})
        b.merge()
        self.assertEqual({r['algorithm'] for r in b.final}, {'vdn'})
        self.assertEqual(len({r['method_variant_id'] for r in b.final}), 1)
        self.assertEqual({r['algorithm_config'] for r in b.final}, set(index.VDN_CONFIGS))
        for r in b.final:
            self.assertFalse({'agent', 'rnn_hidden_dim', 'mixer'} & r['method_parameters'].keys())
        self.candidate(b, 'k1', config={'method': 'topk_preference', 'topk_preference': {'k': 1}})
        self.candidate(b, 'k2', config={'method': 'topk_preference', 'topk_preference': {'k': 2}})
        self.assertNotEqual(b.runs['k1'][0]['method_variant_id'], b.runs['k2'][0]['method_variant_id'])

    def test_architectures_match_actual_code_not_config_names(self):
        _, commit = self.architecture_code()
        resolver = self.builder().architecture_resolver
        for agent, width, expected in [('rnn', 64, 'vdn_2ff_h64'),
                                       ('single_ff_rnn', 64, 'vdn_1ff_h64'),
                                       ('single_ff_rnn', 128, 'vdn_1ff_h128')]:
            config = dict(agent=agent, rnn_hidden_dim=width, mixer='vdn', mac='basic_mac')
            for name in ('vdn.yaml', 'vdn_1ff_128.yaml', 'arbitrary.yaml'):
                r = resolver.resolve(config, commit, name)
                self.assertEqual(r['architecture_id'], expected)
                self.assertEqual(r['architecture_source']['code_commit'], commit)
                self.assertEqual(r['architecture_source']['source_verification'], 'git_commit_only')
        self.assertEqual(resolver.resolve({'algorithm': 'vdn_1ff_128'}, commit, 'config')['architecture_id'], 'unknown')
        self.assertEqual(resolver.resolve(dict(config, mixer='qmix'), commit, 'config')['architecture_id'], 'unknown')
        self.assertEqual(resolver.resolve(config, 'unknown', 'config')['architecture_id'], 'unknown')
        self.assertEqual(resolver.resolve(config, 'f' * 40, 'config')['architecture_source']['reason'], 'unavailable_code_version')

    def test_backbone_signature_ignores_format_and_empty_python_type_metadata(self):
        text = (SCRIPT.parents[1] / 'src/modules/agents/rnn_agent.py').read_text()
        expected = index.backbone_signature(text, 'RNNAgent', True)
        self.assertEqual(index.backbone_signature('# comment\n' + text, 'RNNAgent', True), expected)
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                node._fields = tuple(f for f in node._fields if f != 'type_params') + ('type_params',)
                node.type_params = []
        with patch.object(index.ast, 'parse', return_value=tree):
            self.assertEqual(index.backbone_signature(text, 'RNNAgent', True), expected)

    def test_unrecognized_code_and_launch_hashes_are_not_silently_accepted(self):
        git, commit = self.architecture_code()
        resolver = self.builder().architecture_resolver
        config = dict(agent='rnn', rnn_hidden_dim=64, mixer='vdn', mac='basic_mac')
        paths = ('src/modules/agents/rnn_agent.py', 'src/modules/agents/__init__.py', 'src/modules/mixers/vdn.py', 'src/modules/layers/act_layer.py')
        hashes = {p: hashlib.sha256((self.root / p).read_bytes()).hexdigest() for p in paths}
        r = resolver.resolve(config, commit, 'config', hashes)
        self.assertEqual(r['architecture_source']['source_verification'], 'launch_hashes_match_git')
        r = resolver.resolve(config, commit, 'config', dict(hashes, **{paths[0]: 'mismatch'}))
        self.assertEqual(r['architecture_id'], 'unknown')
        self.assertEqual(r['architecture_source']['reason'], 'launch_source_hash_mismatch')
        path = self.root / paths[0]
        path.write_text(path.read_text().replace('q = self.fc2(h)', 'q = self.fc2(h) + 1'))
        git('add', '.')
        git('-c', 'user.name=Index Test', '-c', 'user.email=index@example.invalid', 'commit', '-qm', 'unrecognized forward')
        r = resolver.resolve(config, git('rev-parse', 'HEAD'), 'config')
        self.assertEqual(r['architecture_id'], 'unknown')
        self.assertEqual(r['architecture_source']['reason'], 'unrecognized_backbone_implementation')

    def test_gate_is_required_for_both_hash_verification_and_backbone_matching(self):
        git, commit = self.architecture_code()
        resolver = self.builder().architecture_resolver
        config = dict(agent='rnn', rnn_hidden_dim=64, mixer='vdn', mac='basic_mac')
        files = ('src/modules/agents/rnn_agent.py','src/modules/agents/__init__.py','src/modules/mixers/vdn.py')
        hashes = {p: hashlib.sha256((self.root / p).read_bytes()).hexdigest() for p in files}
        r = resolver.resolve(config, commit, 'config', hashes)
        self.assertEqual(r['architecture_source']['source_verification'], 'git_commit_only')
        gate = 'src/modules/layers/act_layer.py'
        r = resolver.resolve(config, commit, 'config', dict(hashes, **{gate: 'mismatch'}))
        self.assertEqual(r['architecture_source']['reason'], 'launch_source_hash_mismatch')
        path = self.root / gate
        path.write_text(path.read_text().replace('th.ones(dim)', 'th.zeros(dim)'))
        git('add', '.')
        git('-c', 'user.name=Index Test', '-c', 'user.email=index@example.invalid', 'commit', '-qm', 'changed gate initialization')
        r = resolver.resolve(config, git('rev-parse', 'HEAD'), 'config')
        self.assertEqual(r['architecture_id'], 'unknown')
        self.assertEqual(r['architecture_source']['reason'], 'unrecognized_backbone_implementation')
        old = json.loads((self.root / 'docs/architectures/architectures.json').read_text())
        old['resolver_version'] = 1
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            index.ArchitectureResolver(self.root, old)

    def test_metadata_redaction_preserves_identity_and_is_idempotent(self):
        records = [dict(kind='collection',host='private-machine',roots=['/home/private/project']),
                   dict(kind='runtime',host='private-machine',root='/home/private/project',run_id='run-1',
                        run_dir='/home/private/project/runs/run-1',
                        config={'seed':3535,'agent':'rnn','rnn_hidden_dim':64},
                        launch={'request':{'parent':'/home/private/project/runs/prefix-1'},'git_commit':'a'*40},
                        status_documents={'status.json':{'state':'completed','t_env':100}},
                        error='File "/home/private/project/src/train.py"; PYTHONPATH=/home/private/lib:/home/private/deps',
                        command=['/usr/bin/env','C:\\Users\\Private Name\\python.exe'])]
        safe = index.redact(records)
        self.assertEqual(index.redact(safe), safe)
        text = json.dumps(safe)
        self.assertNotIn('/home/private', text)
        self.assertNotIn('private-machine', text)
        self.assertNotIn('Private Name', text)
        self.assertEqual(safe[1]['run_id'], 'run-1')
        self.assertTrue(safe[1]['run_dir'].endswith('/runs/run-1'))
        self.assertTrue(safe[1]['launch']['request']['parent'].endswith('/runs/prefix-1'))
        self.assertEqual(safe[1]['config'], records[1]['config'])
        self.assertEqual(safe[1]['status_documents'], records[1]['status_documents'])
        self.assertEqual(safe[1]['launch']['git_commit'], 'a'*40)
        self.assertEqual(index.redact({'parent':'/private/location/prefix-1'})['parent'].split('/')[-1], 'prefix-1')
        self.assertEqual(index.redact('https://example.com/source'), 'https://example.com/source')
        path = index.redact('/private/lib')
        self.assertEqual(index.redact('PYTHONPATH=/private/lib:/private/deps').split(':')[0], 'PYTHONPATH='+path)
        remote = index.redact('private-user@private-machine:/private/project/runs/run-1')
        self.assertNotIn('private-user', remote)
        self.assertNotIn('private-machine', remote)

    def test_architecture_conflicts_remain_unknown(self):
        _, commit = self.architecture_code()
        b = self.builder()
        self.candidate(b, row={'commit': commit}, config=dict(agent='rnn', rnn_hidden_dim=64, mixer='vdn', mac='basic_mac'))
        self.candidate(b, row={'commit': commit}, config=dict(agent='single_ff_rnn', rnn_hidden_dim=128, mixer='vdn', mac='basic_mac'))
        b.merge()
        self.assertEqual(b.final[0]['architecture_id'], 'unknown')
        self.assertEqual(b.final[0]['architecture_source']['reason'], 'conflicting_architecture_records')

    def test_known_git_candidate_does_not_hide_launch_source_mismatch(self):
        _, commit = self.architecture_code()
        b = self.builder()
        config = dict(agent='rnn', rnn_hidden_dim=64, mixer='vdn', mac='basic_mac')
        self.candidate(b, row={'commit': commit}, config=config)
        self.candidate(b, row={'commit': commit, '_source_sha256': {'src/modules/agents/rnn_agent.py': 'mismatch'}},
                       config=config, priority=200)
        b.merge()
        self.assertEqual(b.final[0]['architecture_id'], 'unknown')
        self.assertEqual(b.final[0]['architecture_source']['reason'], 'unverified_implementation_evidence')

    def test_architecture_filters_and_matrices_keep_seed_and_run_evidence(self):
        _, commit = self.architecture_code()
        b = self.builder()
        for rid, width in [('small', 64), ('wide', 128)]:
            self.candidate(b, rid, row={'commit': commit}, config=dict(agent='single_ff_rnn', rnn_hidden_dim=width, mixer='vdn', mac='basic_mac'))
        b.merge()
        args = argparse.Namespace(architecture='vdn_1ff_h128')
        self.assertEqual([r['run_id'] for r in index.filter_runs(b.final, args)], ['wide'])
        cells = index.matrix(b.final, 'architecture-method')
        self.assertEqual(len({c['column'] for c in cells}), 1)
        self.assertEqual({c['row'] for c in cells}, {'vdn_1ff_h64', 'vdn_1ff_h128'})
        cells = index.matrix(b.final, 'architecture-environment', environment_ids=['mpe'], architecture_ids=['vdn_2ff_h64'])
        self.assertTrue(any(c['row'] == 'vdn_2ff_h64' and c['column'] == 'mpe' and c['observed_runs'] == 0 for c in cells))

    def test_conflicting_seed_is_unknown_and_audited(self):
        b = self.builder()
        self.candidate(b)
        self.candidate(b, config={'seed': 2222})
        b.merge()
        self.assertEqual(b.final[0]['seed'], 'unknown')
        self.assertEqual(b.issues[0]['kind'], 'conflicting_run_records')

    def test_prefix_and_branch_endpoints_are_distinct(self):
        b = self.builder()
        self.candidate(b, 'prefix', row={'role': 'prefix', '_completed_phases': [0]}, results=['prefix/stage_metrics.csv'])
        self.candidate(b, 'branch', row={'parent_run_id': 'prefix', '_completed_phases': [1]}, results=['branch/stage_metrics.csv'])
        self.candidate(b, 'partial', row={'_completed_phases': [0]}, results=['partial/stage_metrics.csv'])
        b.merge(); b.complete_results()
        rows = {r['run_id']: r for r in b.final}
        self.assertEqual(rows['prefix']['result_status'], 'reported_prefix_result')
        self.assertEqual(rows['branch']['result_status'], 'reported_complete_result')
        self.assertEqual(rows['partial']['result_status'], 'published_partial_or_unverified')
        cells = index.matrix(b.final, 'seed-coverage', coverage='complete_result')
        self.assertEqual(sum(c['completed_runs'] for c in cells), 1)
        self.assertEqual(sum(c['observed_runs'] for c in cells), 2)
        environment_cells = index.matrix(b.final, 'environment-method', environment_ids=['mpe'])
        absent = [c for c in environment_cells if c['row'] == 'mpe']
        self.assertTrue(absent)
        self.assertTrue(all(c['observed_runs'] == 0 for c in absent))

    def test_parent_endpoints_with_different_tasks_do_not_complete_child(self):
        b = self.builder()
        self.candidate(b, 'parent', row={'_completed_phases': [0]}, results=['parent.csv'])
        self.candidate(b, 'child', config={'env_args': {'map_name': 'other'}},
                       row={'parent_run_id': 'parent', '_completed_phases': [1]}, results=['child.csv'])
        b.merge(); b.complete_results()
        child = next(r for r in b.final if r['run_id'] == 'child')
        self.assertEqual(child['result_status'], 'published_partial_or_unverified')

    def test_scheduler_does_not_create_runs_but_preserves_dependency_links(self):
        b = self.builder()
        self.candidate(b, 'actual')
        b.merge()
        b.scheduler_rows = [({'kind': 'scheduler_training', 'analysis_id': 'train', 'state': 'completed',
                             'spec': {'type': 'training', 'argv': ['python', '--run-id', 'actual']}}, 'export:1'),
                            ({'kind': 'scheduler_training', 'analysis_id': 'no-evidence', 'state': 'completed',
                              'spec': {'type': 'training', 'run_dir': '/runs/no-evidence'}}, 'export:2')]
        b.runtime_analyses = [({'kind': 'scheduler_analysis', 'analysis_id': 'eval',
                               'spec': {'type': 'evaluation', 'dependencies': ['train']}}, 'export:3')]
        b.scheduler_links()
        self.assertEqual(len(b.final), 1)
        self.assertEqual(b.final[0]['scheduler_task_ids'], ['train'])
        self.assertEqual(b.analysis_dependencies('eval'), ['actual'])
        self.assertEqual(b.issues[0]['kind'], 'scheduler_without_runtime_evidence')

    def test_analysis_keeps_unknown_inputs_and_exact_token_matching(self):
        b = self.builder()
        self.candidate(b, 'actual')
        b.merge(); results = {}
        b.add_analysis(results, 'analysis', 'exp', 'offline', 'completed', ['actual'], [], [],
                       {'actual': b.final[0]}, {'actual', 'missing'}, {'runs/missing/agent.th'})
        self.assertEqual(results['analysis']['unresolved_inputs'], ['missing'])
        self.assertEqual(index.refs_in({'path': '/runs/actual/agent.th'}, {'actual': {}}), ['actual'])
        self.assertEqual(index.refs_in({'path': '/runs/actual_extra/agent.th'}, {'actual': {}}), [])

    def test_analysis_does_not_consume_own_id_or_unused_protocol_references(self):
        payload = index.analysis_payload({
            'run_id': 'eval', 'run_dir': '/runs/eval',
            'launch': {'request': {'parent': '/runs/actual', 'run_id': 'eval'}},
            'protocol': {'unused_other_seed_baseline': '/runs/other'}})
        self.assertEqual(index.refs_in(payload, {'actual': {}, 'other': {}}), ['actual'])
        self.assertEqual(index.declared_inputs(payload)[0], {'actual'})

    def test_export_is_deterministic_and_csv_roundtrips(self):
        with patch.object(index.Builder, 'local_runtime'):
            first = index.artifacts(self.builder().build())
            second = index.artifacts(self.builder().build())
        self.assertEqual(first, second)
        b = self.builder(); self.candidate(b); b.merge()
        text = index.csv_text(index.RUN_FIELDS, b.final)
        (self.root / 'docs/results/run_index.csv').write_text(text)
        self.assertEqual(index.read_runs(self.root)[0]['method_parameters'], b.final[0]['method_parameters'])

    def test_check_rejects_stale_generated_view(self):
        with patch.object(index.Builder, 'local_runtime'):
            for path, text in index.artifacts(self.builder().build()).items():
                (self.root / path).parent.mkdir(parents=True, exist_ok=True)
                (self.root / path).write_text(text)
            with patch('sys.argv', ['experiment_index', '--root', str(self.root), 'check']):
                with contextlib.redirect_stdout(io.StringIO()):
                    index.main()
                (self.root / 'docs/results/run_index.csv').write_text('manually changed')
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    index.main()
                self.assertEqual(error.exception.code, 1)

    def test_runtime_evaluation_is_separate_and_failed_attempt_is_retained(self):
        base = dict(kind='runtime', host='fixture', run_id='actual', run_dir='/runs/actual',
                    config={'env': 'sc2', 'env_args': {'map_name': '2s3z'}, 'seed': 3535},
                    launch={'started': True, 'request': {}},
                    status_documents={'launcher_status.json': {'state': 'failed', 'exit_code': 1}})
        evaluation = dict(base, run_id='eval', run_dir='/runs/actual/evaluation/final')
        (self.root / 'docs/results/runtime_records.jsonl').write_text(
            '\n'.join(json.dumps(x) for x in (base, evaluation)) + '\n')
        with patch.object(index.Builder, 'local_runtime'):
            b = self.builder().build()
        self.assertEqual(len(b.final), 1)
        self.assertEqual(b.final[0]['status'], 'failed')
        self.assertEqual(b.final[0]['environment_id'], 'smac')
        self.assertEqual(len(b.analyses), 1)

    def test_branch_source_reads_pinned_tree_without_checkout(self):
        def git(*args):
            return subprocess.check_output(['git', *args], cwd=self.root, text=True).strip()
        git('init', '-q')
        self.write('configs/history.json', {'id': 'historical', 'order': ['A']})
        git('add', '.')
        git('-c', 'user.name=Index Test', '-c', 'user.email=index@example.invalid', 'commit', '-qm', 'fixture')
        commit = git('rev-parse', 'HEAD')
        self.write('configs/history.json', {'id': 'current'})
        src = index.Source(self.root, commit)
        self.assertEqual(src.json('configs/history.json')['id'], 'historical')
        self.assertEqual(git('rev-parse', 'HEAD'), commit)
        self.assertIn('configs/history.json', git('status', '--short'))

    def test_collector_reads_metadata_and_scheduler_without_mutation(self):
        spec = importlib.util.spec_from_file_location('collector', SCRIPT.with_name('collect_experiment_metadata.py'))
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        self.write('runs/actual/status.json', {'t_env': 10})
        self.write('runs/actual/config.json', {'seed': 3535})
        self.write('runs/actual/launch.json', {'source_sha256': {
            'src/modules/agents/rnn_agent.py': 'network-hash',
            'src/modules/layers/act_layer.py':'gate-hash', 'docs/unrelated.md': 'unrelated-hash'}})
        database = self.root / 'runtime/scheduler/state.sqlite3'
        database.parent.mkdir(parents=True)
        connection = sqlite3.connect(database)
        connection.execute('CREATE TABLE tasks (id TEXT,state TEXT,spec TEXT)')
        connection.execute('CREATE TABLE attempts (task TEXT,created REAL,ended REAL,exit_code INTEGER)')
        connection.execute('INSERT INTO tasks VALUES (?,?,?)', ('task', 'completed', '{"type":"training"}'))
        connection.execute('INSERT INTO attempts VALUES (?,?,?,?)', ('task', 1, 2, 0))
        connection.commit(); connection.close()
        before = database.read_bytes()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            collector.collect([self.root], 'fixture')
        records = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([r['kind'] for r in records], ['collection', 'runtime', 'scheduler_training'])
        self.assertEqual(records[1]['config']['seed'], 3535)
        self.assertEqual(records[1]['launch']['source_sha256'], {
            'src/modules/agents/rnn_agent.py': 'network-hash','src/modules/layers/act_layer.py':'gate-hash'})
        registered=set().union(*(set(a['implementation_paths']) for a in self.builder().architectures['architectures']))
        self.assertEqual(registered,set(collector.ARCHITECTURE_FILES))
        self.assertEqual(database.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
