"""Audited baseline source boundary, independent of preference implementations.

AST normalization ignores comments/formatting. Driver orchestration is specialized
for the actual shared-prefix contract (baseline, no parent, no preferences), ending
at the first phase's training loop. Unknown source layouts fail closed.
"""
import ast
import copy
import hashlib


class BaselinePath(ast.NodeTransformer):
    def __init__(self):
        from action_set_protocol import ASPR_MAX_METHODS, PREFERENCE_RANK_METHODS, HISTORY_RANK_METHODS
        self.known = dict(role='shared-prefix', parent=None, method='baseline', preference=None,
                          training_preference=None, gradient_monitor=None, monitor_interval=0,
                          selected_frpr=False, pair_ablation=False, aspr=False, aspr_group=False,
                          capture_timing=False, check=False, ASPR_MAX_METHODS=ASPR_MAX_METHODS,
                          PREFERENCE_RANK_METHODS=PREFERENCE_RANK_METHODS, HISTORY_RANK_METHODS=HISTORY_RANK_METHODS)

    def value(self, node):
        if ast.unparse(node) == 'isinstance(buffer, TaskReplayBuffer)':
            return False  # The shared prefix always uses ordinary Episode ReplayBuffer.
        if any(isinstance(n, (ast.Call, ast.Attribute, ast.NamedExpr, ast.Lambda)) for n in ast.walk(node)):
            raise ValueError('Not a literal baseline expression')
        return eval(compile(ast.Expression(node), '<prefix-path>', 'eval'), {'__builtins__': {}}, self.known)

    def visit_IfExp(self, node):
        try:
            value = self.value(node.test)
        except (ValueError, NameError, KeyError, TypeError, IndexError):
            return self.generic_visit(node)
        return self.visit(node.body if value else node.orelse)

    def visit_Compare(self, node):
        try:
            return ast.copy_location(ast.Constant(self.value(node)), node)
        except (ValueError, NameError, KeyError, TypeError, IndexError):
            return self.generic_visit(node)

    def visit_If(self, node):
        # Evaluate only side-effect-free known operations over this fixed contract.
        try:
            value = self.value(node.test)
        except (ValueError, NameError, KeyError, TypeError, IndexError):
            pass
        else:
            result = []
            for child in node.body if value else node.orelse:
                projected = self.visit(child)
                result.extend(projected if isinstance(projected, list) else [projected])
            return result
        return self.generic_visit(node)


def definitions(tree, names):
    result = []
    for name in names:
        scope = tree.body
        for part in name.split('.'):
            found = [n for n in scope if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
            if len(found) != 1:
                raise ValueError(f'Unrecognized Prefix training dependency: {name}')
            node = found[0]
            scope = node.body
        result.append(copy.deepcopy(node))
    return result


def driver_training(tree):
    fn = definitions(tree, ['train_sequence'])[0]
    attempts = [n for n in fn.body if isinstance(n, ast.Try)]
    if len(attempts) != 1:
        raise ValueError('Unrecognized Prefix driver try boundary; audit training dependencies')
    attempt = attempts[0]
    loops = [n for n in attempt.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'phase']
    if len(loops) != 1:
        raise ValueError('Unrecognized Prefix phase loop; audit training dependencies')
    phase_loop = loops[0]
    whiles = [n for n in phase_loop.body if isinstance(n, ast.While)]
    if len(whiles) != 1 or ast.unparse(whiles[0].test) != 'runner.t_env < end':
        raise ValueError('Unrecognized Prefix TD loop; audit training dependencies')
    phase_loop.body = phase_loop.body[:phase_loop.body.index(whiles[0]) + 1]
    # These are definitions of consumers which the baseline never calls. Creation
    # of the baseline models, replay, seed, budgets, and actual while loop remain.
    consumer_defs = {'record_preference', 'record_gradients', 'prepare_shared_pair_pool',
                     'prepare_selected_history'}
    body = attempt.body[:attempt.body.index(phase_loop) + 1]
    body = [n for n in body if not (isinstance(n, ast.FunctionDef) and n.name in consumer_defs)]
    attempt.body, attempt.handlers, attempt.orelse, attempt.finalbody = body, [], [], [ast.Pass()]
    fn.body = fn.body[:fn.body.index(attempt) + 1]
    return BaselinePath().visit(fn)


def training_implementation(root, cfg):
    """Named dependency components; configuration values are checked by Prefix Spec."""
    if (cfg['learner'], cfg['mac'], cfg['mixer'], cfg['action_selector']) != (
            'q_learner', 'basic_mac', 'vdn', 'epsilon_greedy'):
        raise ValueError('Prefix implementation boundary supports baseline recurrent VDN only')
    agents = {'rnn': ('rnn_agent.py', 'RNNAgent'),
              'single_ff_rnn': ('single_ff_rnn_agent.py', 'SingleFFRNNAgent')}
    if cfg['agent'] not in agents:
        raise ValueError('Unknown Prefix Agent implementation; audit dependencies')
    agent_file, agent_class = agents[cfg['agent']]
    entries = {
        'drivers/continual/action_set_train.py': ['config', 'augmented_scheme', 'isolated_rollouts',
                                                'greedy_runner', 'layout', 'evaluate_task', 'evaluation_settings'],
        'drivers/continual/common/prefix.py': ['base_config'],
        'drivers/continual/common/runtime.py': ['episode_spec', 'rng_state', 'restore_rng', 'td_sample', 'log_task_epsilon'],
        'drivers/continual/common/exploration.py': ['epsilon_on_switch'],
        'src/learners/q_learner.py': ['QLearner.__init__', 'QLearner.train', 'QLearner._update_targets',
                                     'QLearner.cuda', 'QLearner.remove_activation_hooks'],
        'src/controllers/basic_controller.py': ['BasicMAC'],
        'src/modules/agents/' + agent_file: [agent_class],
        'src/modules/layers/act_layer.py': ['ActivateLayer'],
        'src/modules/mixers/vdn.py': ['VDNMixer'],
        'src/runners/episode_runner.py': ['EpisodeRunner'],
        'src/components/episode_buffer.py': None,
        'src/components/transforms.py': ['OneHot'],
        'src/components/task_replay.py': ['start_replay_task'],
        'src/components/action_selectors.py': ['EpsilonGreedyActionSelector'],
        'src/components/epsilon_schedules.py': ['DecayThenFlatSchedule'],
        'src/utils/hook_lifecycle.py': None,
        'drivers/continual/hetero_environment.py': ['ExactTeams', 'ExactProtoss.__init__',
            'ExactProtoss.__getattr__', 'ExactProtoss.set_task', 'ExactProtoss.reset',
            'ExactProtoss.step', 'ExactProtoss.distribution_state'],
        'drivers/continual/environment.py': ['seed_distributions'],
    }
    # Capability/map/distribution definitions are environment dependencies, not methods.
    for path in (root / 'vendor/smacv2/smacv2/env').rglob('*.py'):
        if 'pettingzoo' not in path.parts:
            entries[str(path.relative_to(root))] = None
    components = {}
    for name, symbols in sorted(entries.items()):
        tree = ast.parse((root / name).read_text(), filename=name)
        nodes = definitions(tree, symbols) if symbols is not None else tree.body
        if name in ('src/learners/q_learner.py', 'src/components/task_replay.py'):
            nodes = [BaselinePath().visit(copy.deepcopy(n)) for n in nodes]
        used = {n.id for item in nodes for n in ast.walk(item) if isinstance(n, ast.Name)}
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
                   and any((a.asname or a.name.split('.')[0]) in used for a in n.names)]
        # Only referenced named constants, not unrelated registry entries.
        constants = [n for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                     and any(isinstance(t, ast.Name) and t.id in used
                             for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
        if symbols is not None:
            nodes = imports + constants + nodes
        if name.endswith('/action_set_train.py'):
            # Its method imports are provenance only; they do not execute baseline updates.
            bindings = [n for n in tree.body if isinstance(n, ast.Import)
                        or (isinstance(n, ast.ImportFrom) and n.module in (
                            'hetero_environment', 'components.task_replay', 'common.checkpoints',
                            'common.metrics', 'common.records', 'common.runtime', 'common.exploration',
                            'common.prefix', 'types', 'contextlib'))
                        or isinstance(n, (ast.Assign, ast.AnnAssign))]
            nodes = bindings + definitions(tree, symbols) + [driver_training(tree)]
        if name in ('src/learners/q_learner.py', 'src/components/task_replay.py'):
            # Diagnostic and preference callbacks are inactive during baseline updates.
            nodes = [BaselinePath().visit(copy.deepcopy(n)) for n in nodes]
        payload = ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False)
        components[name] = hashlib.sha256(payload.encode()).hexdigest()
    base_tree = ast.parse((root / 'drivers/continual/train.py').read_text())
    base_bindings = [n for n in base_tree.body if isinstance(n, ast.ImportFrom) and n.module in (
        'components.episode_buffer', 'controllers', 'learners', 'runners.episode_runner', 'envs', 'utils.logging')]
    components['drivers/continual/train.py:factories'] = hashlib.sha256(
        ast.dump(ast.Module(body=base_bindings, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    # Registry wiring is checked separately: unrelated registry entries are irrelevant.
    for name, key in [('src/controllers/__init__.py', 'basic_mac'),
                      ('src/modules/agents/__init__.py', cfg['agent']),
                      ('src/learners/__init__.py', 'q_learner'),
                      ('src/components/action_selectors.py', 'epsilon_greedy')]:
        tree = ast.parse((root / name).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'REGISTRY'
            and isinstance(t.slice, ast.Constant) and t.slice.value == key for t in n.targets)]
        if len(nodes) != 1:
            raise ValueError(f'Unrecognized Prefix registry: {name}:{key}')
        symbol = nodes[0].value
        if not isinstance(symbol, ast.Name):
            raise ValueError(f'Unrecognized Prefix registry target: {name}:{key}')
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                   and any((a.asname or a.name) == symbol.id for a in n.names)]
        local_classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == symbol.id]
        if len(imports) + len(local_classes) != 1:
            raise ValueError(f'Unrecognized Prefix registry import: {name}:{key}')
        payload = ast.dump(ast.Module(body=imports + nodes, type_ignores=[]), include_attributes=False)
        components[name + ':' + key] = hashlib.sha256(payload.encode()).hexdigest()
    payload = '\n'.join(f'{k}:{v}' for k, v in sorted(components.items()))
    return dict(version=1, sha256=hashlib.sha256(payload.encode()).hexdigest(), components=components)
