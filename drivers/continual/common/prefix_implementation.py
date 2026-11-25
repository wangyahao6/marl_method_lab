"""Audited baseline source boundary, independent of preference implementations.

AST normalization ignores comments/formatting. Driver orchestration is specialized
for the actual shared-prefix contract (baseline, no parent, no preferences), ending
at the first phase's training loop. Unknown source layouts fail closed.
"""
import ast
import copy
import hashlib
from pathlib import Path

# Bump whenever extraction boundaries or normalization rules change.
IMPLEMENTATION_VERSION = 2


class BaselinePath(ast.NodeTransformer):
    def __init__(self):
        from action_set_protocol import ASPR_MAX_METHODS, PREFERENCE_RANK_METHODS, HISTORY_RANK_METHODS
        self.known = dict(role='shared-prefix', parent=None, method='baseline', preference=None,
                          training_preference=None, gradient_monitor=None, monitor_interval=0,
                          selected_frpr=False, pair_ablation=False, aspr=False, aspr_group=False,
                          capture_timing=False, check=False, ASPR_MAX_METHODS=ASPR_MAX_METHODS,
                          PREFERENCE_RANK_METHODS=PREFERENCE_RANK_METHODS, HISTORY_RANK_METHODS=HISTORY_RANK_METHODS)

    def value(self, node):
        if ast.unparse(node) == 'args.evaluate_run':
            return False  # Shared-prefix uses --seed, not the final-evaluation entry.
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


def source_nodes_hash(nodes):
    """Canonical executable AST; comments, formatting and docstrings are not code."""
    class WithoutDocstrings(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return self.generic_visit(node)

    tree = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    normalized = WithoutDocstrings().visit(tree)
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def extraction_rules_hash():
    """Identity of the executing extractor, independent of the tree being audited."""
    return source_nodes_hash(ast.parse(Path(__file__).read_text()).body)


def implementation_hash(components, rules_sha256, version=IMPLEMENTATION_VERSION):
    payload = f'{version}:{rules_sha256}\n' + '\n'.join(f'{k}:{v}' for k, v in sorted(components.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def module_initialization(tree):
    """Executable module setup, including statements after function declarations.

    Function bodies are hashed separately when actually used. Main guards are
    omitted for imports; action_set_train's training dispatch is covered by main.
    """
    nodes = []
    for original in tree.body:
        node = copy.deepcopy(original)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Defining a function evaluates defaults/decorators/annotations, but
            # does not execute its standalone training body.
            node.body = [ast.Pass()]
        elif isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            nodes.extend(node.orelse)  # Imported as train, never as __main__.
            continue
        nodes.append(node)
    return nodes


def base_initialization(tree):
    if not {'merge', 'main'}.issubset({n.name for n in tree.body if isinstance(n, ast.FunctionDef)}):
        raise ValueError('Unrecognized train.py module initialization boundary')
    return module_initialization(tree)


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
                                                'greedy_runner', 'layout', 'evaluate_task', 'evaluation_settings', 'main'],
        'drivers/continual/common/prefix.py': ['base_config'],
        'drivers/continual/common/runtime.py': ['episode_spec', 'rng_state', 'restore_rng', 'td_sample', 'log_task_epsilon'],
        'drivers/continual/common/exploration.py': ['epsilon_on_switch'],
        'src/learners/q_learner.py': ['QLearner.__init__', 'QLearner.train', 'QLearner._update_targets',
                                     'QLearner.cuda', 'QLearner.remove_activation_hooks', 'QLearner.save_models'],
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
        'src/utils/logging.py': ['Logger'],
        'drivers/continual/common/records.py': ['write', 'append', 'attach_legacy_metrics'],
        'drivers/continual/common/checkpoints.py': ['cpu_snapshot', 'parameter_hash', 'hash_model_states', 'save_models'],
        'drivers/continual/common/preference_prefix.py': ['full_state_extra'],
        'drivers/continual/common/metrics.py': ['Metrics.__init__', 'Metrics.record', 'Metrics._write',
                                             'Metrics.evaluation', 'Metrics.retention', 'record_evaluation'],
        'drivers/continual/hetero_environment.py': ['ExactTeams', 'ExactProtoss.__init__',
            'ExactProtoss.__getattr__', 'ExactProtoss.set_task', 'ExactProtoss.reset',
            'ExactProtoss.step', 'ExactProtoss.distribution_state'],
        'drivers/continual/environment.py': ['seed_distributions'],
    }
    # Agent constructors install TRHR only for a truthy cfg.trhr. QLearner
    # creates A2Trace only when requested and not in evaluation mode. Even with
    # TRHR off, A2Trace calls capture_activity and valid_a2_rows from trhr.py.
    trhr = bool(cfg.get('trhr'))
    trace = bool(cfg.get('a2_trace')) and not bool(cfg.get('evaluate', False))
    if trhr or trace:
        entries['src/modules/layers/trhr.py'] = ['capture_activity', 'valid_a2_rows']
        if trhr:
            entries['src/modules/layers/trhr.py'].append('TRHRReLU')
    if trace:
        entries['src/utils/a2_trace.py'] = ['A2Trace']
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
            # Entry dispatch and all top-level initialization execute before the
            # first phase. The __main__ guard is represented by the main body.
            nodes = module_initialization(tree) + definitions(tree, symbols) + [driver_training(tree)]
            nodes = [BaselinePath().visit(copy.deepcopy(n)) for n in nodes]
        if name in ('src/learners/q_learner.py', 'src/components/task_replay.py'):
            # Diagnostic and preference callbacks are inactive during baseline updates.
            nodes = [BaselinePath().visit(copy.deepcopy(n)) for n in nodes]
        components[name] = source_nodes_hash(nodes)
    base_tree = ast.parse((root / 'drivers/continual/train.py').read_text())
    # This includes ModuleType('envs'), its REGISTRY, sys.modules publication,
    # sys.path setup, SC2PATH and the final envs/EpisodeRunner factory imports.
    components['drivers/continual/train.py:initialization'] = source_nodes_hash(base_initialization(base_tree))
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
        components[name + ':' + key] = source_nodes_hash(imports + nodes)
    rules_sha256 = extraction_rules_hash()
    return dict(version=IMPLEMENTATION_VERSION, rules_sha256=rules_sha256,
                sha256=implementation_hash(components, rules_sha256), components=components)
