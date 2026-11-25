"""First-stage training identity and immutable shared-prefix artifact metadata."""
import copy
import hashlib
import json
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[3]
STATE_VERSION = 1
MANIFEST_VERSION = 2


def base_config(algorithm):
    def merge(target, values):
        for key, value in values.items():
            target[key] = merge(target.get(key, {}), value) if isinstance(value, dict) else value
        return target

    cfg = {}
    for name in ("src/config/default.yaml", "vendor/smacv2/smacv2/examples/configs/sc2_gen_protoss.yaml",
                 f"src/config/algs/{algorithm}.yaml"):
        merge(cfg, yaml.safe_load((ROOT / name).read_text()))
    return copy.deepcopy(cfg)


def prefix_spec(protocol, seed, effective_config=None):
    """Use resolved training inputs; later stages and preference options never enter the spec."""
    import torch as th
    import numpy as np

    cfg = copy.deepcopy(effective_config if effective_config is not None else base_config(protocol['algorithm']))
    # These are outputs, method consumers, instrumentation, or whole-sequence controls.
    from action_set_protocol import ASPR_MAX_METHODS, PREFERENCE_RANK_METHODS
    excluded = {'method', 'action_set', 'gradient_monitor_interval', 'rank_position_diagnostics',
                't_max', 'local_results_path', 'name', 'device', 'seed', 'algorithm',
                'env_args', 'env', 'epsilon_on_switch', 'replay_on_switch', 'optimizer',
                'preference_role', 'parent', 'preference_weight', 'capture_timing',
                'n_agents', 'n_actions', 'state_shape', 'full_shared_prefix',
                *ASPR_MAX_METHODS, *PREFERENCE_RANK_METHODS}
    training = {key: value for key, value in cfg.items() if key not in excluded}
    training['use_cuda'] = cfg.get('use_cuda', th.cuda.is_available()) if effective_config is not None else th.cuda.is_available()
    training['optimizer'] = protocol.get('optimizer', 'rmsprop')
    # Shared-prefix stage zero always uses ordinary baseline episode replay.
    # FT/ER describes the child's switch behavior, not prefix training.
    training['first_stage_replay'] = 'episode'
    env = copy.deepcopy(cfg['env_args'])
    env.pop('rosters', None)
    env.update(seed=seed, game_version=protocol.get('sc2_game_version', '4.10'))
    from common.prefix_implementation import training_implementation
    implementation = training_implementation(ROOT, cfg)
    cuda = None
    if training['use_cuda']:
        cuda = dict(torch_cuda_build=th.version.cuda, cudnn=th.backends.cudnn.version(),
                    device=th.cuda.get_device_name(0), capability=list(th.cuda.get_device_capability(0)),
                    cudnn_deterministic=th.backends.cudnn.deterministic,
                    cudnn_benchmark=th.backends.cudnn.benchmark,
                    cudnn_tf32=th.backends.cudnn.allow_tf32,
                    matmul_tf32=th.backends.cuda.matmul.allow_tf32,
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    dependencies = {}
    for package in ('pysc2', 's2clientprotocol', 'protobuf', 'absl-py'):
        try:
            dependencies[package] = package_version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    stage = copy.deepcopy(protocol['stages'][0])
    return dict(version=STATE_VERSION, environment='action_set_protoss', env_args=env,
                runtime=dict(environment_packages=dependencies, python=platform.python_version(), python_implementation=platform.python_implementation(),
                             cuda=cuda, torch_cuda_build=th.version.cuda,
                             default_dtype=str(th.get_default_dtype()),
                             float32_matmul_precision=th.get_float32_matmul_precision(),
                             torch_num_threads=th.get_num_threads() if effective_config is not None else 1,
                             torch=str(th.__version__), numpy=np.__version__,
                             deterministic=th.are_deterministic_algorithms_enabled()),
                implementation=implementation,
                algorithm=protocol['algorithm'], seed=seed, stage=stage,
                roster=copy.deepcopy(protocol['tasks'][stage['task']]), training=training)


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def differences(previous, requested, path=''):
    if isinstance(previous, dict) and isinstance(requested, dict):
        rows = []
        for key in sorted(previous.keys() | requested.keys()):
            name = f'{path}.{key}' if path else key
            if key not in previous or key not in requested:
                rows.append(f'{name}: {previous.get(key)!r} -> {requested.get(key)!r}')
            else:
                rows.extend(differences(previous[key], requested[key], name))
        return rows
    return [] if previous == requested else [f'{path}: {previous!r} -> {requested!r}']


def require_compatible(previous, requested):
    rows = differences(previous, requested)
    if rows:
        raise ValueError('Incompatible Prefix Spec:\n' + '\n'.join(rows))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(out, spec, checkpoint, collection):
    out = Path(out)
    launch_path = out / 'launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
        code = {key: launch.get(key) for key in ('git_commit', 'git_status', 'source_sha256')}
    else:
        code = dict(git_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip(),
                    source_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for folder in ('src', 'drivers')
                                   for p in sorted((ROOT / folder).rglob('*.py'))})
    if isinstance(code.get('source_sha256'), dict):
        code['source_sha256'] = {Path(name).as_posix(): digest for name, digest in code['source_sha256'].items()}
    files = [out / 'retention.json', out / 'config.json', out / 'env_info.json', out / 'return_state.th', out / 'prefix_replay.th', out / 'initial_agent.th',
             *sorted(checkpoint.glob('*.th')),
             *sorted((out / 'action_set_memory').rglob('*.th'))]
    manifest = dict(manifest_version=MANIFEST_VERSION, prefix_id='prefix-' + uuid4().hex, state_version=STATE_VERSION,
                    prefix_spec=spec, prefix_spec_sha256=spec_hash(spec), code_version=code,
                    history_collection=collection,
                    files={path.relative_to(out).as_posix(): file_hash(path) for path in files})
    with (out / 'prefix_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return manifest


REQUIRED_ARTIFACTS = frozenset((
    'retention.json', 'config.json', 'env_info.json', 'return_state.th',
    'prefix_replay.th', 'initial_agent.th',
    'action_set_memory/phase0000/success.th', 'action_set_memory/phase0000/failure.th',
))
CHECKPOINT_ARTIFACTS = ('agent.th', 'mixer.th', 'opt.th')


def relative_manifest_path(name):
    """Canonical portable paths only; never interpret an external/parent path."""
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
            or '..' in PurePosixPath(name).parts or name != PurePosixPath(name).as_posix()
            or name == '.'):
        raise ValueError(f'Unsafe Prefix Manifest path: {name!r}')
    return name


def artifact_path(parent, name):
    name = relative_manifest_path(name)
    parent = Path(parent).resolve()
    path = parent / name
    if not path.resolve().is_relative_to(parent):
        raise ValueError(f'Prefix artifact escapes parent directory: {name}')
    return path


def read_manifest_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate Prefix JSON key: {key}')
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(), object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read Prefix JSON {path.name}: {exc}') from exc


def require_fields(value, names, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object')
    missing = sorted(set(names) - value.keys())
    if missing:
        raise ValueError(f'{label} missing required fields: {", ".join(missing)}')


def valid_sha(value, length=64):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def validate_manifest_fields(manifest):
    from common.prefix_implementation import IMPLEMENTATION_VERSION, implementation_hash

    require_fields(manifest, ('prefix_id', 'manifest_version', 'state_version', 'prefix_spec',
                             'prefix_spec_sha256', 'code_version', 'history_collection', 'files'), 'Prefix Manifest')
    if not isinstance(manifest['prefix_id'], str) or not re.fullmatch('prefix-[0-9a-f]{32}', manifest['prefix_id']):
        raise ValueError('Invalid Prefix ID')
    if type(manifest['state_version']) is not int or manifest['state_version'] != STATE_VERSION:
        raise ValueError(f"Unsupported prefix state version: {manifest['state_version']}")
    spec = manifest['prefix_spec']
    require_fields(spec, ('version', 'environment', 'env_args', 'runtime', 'implementation',
                         'algorithm', 'seed', 'stage', 'roster', 'training'), 'Prefix Spec')
    if type(spec['version']) is not int or spec['version'] != STATE_VERSION:
        raise ValueError('Unsupported Prefix Spec version')
    for name in ('env_args', 'runtime', 'roster', 'training'):
        if not isinstance(spec[name], dict) or not spec[name]:
            raise ValueError(f'Prefix Spec {name} must be a nonempty object')
    if (not isinstance(spec['environment'], str) or not spec['environment']
            or not isinstance(spec['algorithm'], str) or not spec['algorithm'] or type(spec['seed']) is not int):
        raise ValueError('Invalid Prefix environment, algorithm or seed')
    require_fields(spec['stage'], ('task', 'steps'), 'Prefix first stage')
    if (not isinstance(spec['stage']['task'], str) or not spec['stage']['task']
            or type(spec['stage']['steps']) is not int or spec['stage']['steps'] <= 0):
        raise ValueError('Invalid Prefix first stage')
    implementation = spec['implementation']
    require_fields(implementation, ('version',), 'Prefix implementation')
    if type(implementation['version']) is not int or implementation['version'] != IMPLEMENTATION_VERSION:
        raise ValueError(f"Unsupported Prefix implementation version: {implementation['version']}; "
                         f'expected {IMPLEMENTATION_VERSION}. No automatic upgrade.')
    require_fields(implementation, ('rules_sha256', 'sha256', 'components'), 'Prefix implementation')
    components = implementation['components']
    if (not isinstance(components, dict) or not components
            or any(not isinstance(key, str) or not key or not valid_sha(value) for key, value in components.items())
            or not valid_sha(implementation['rules_sha256']) or not valid_sha(implementation['sha256'])):
        raise ValueError('Invalid Prefix implementation fingerprint')
    if implementation_hash(components, implementation['rules_sha256']) != implementation['sha256']:
        raise ValueError('Prefix implementation aggregate hash mismatch')
    if not valid_sha(manifest['prefix_spec_sha256']) or spec_hash(spec) != manifest['prefix_spec_sha256']:
        raise ValueError('Prefix Spec hash mismatch')
    code = manifest['code_version']
    require_fields(code, ('git_commit', 'git_status', 'source_sha256'), 'Prefix code_version')
    if not valid_sha(code['git_commit'], 40) or not isinstance(code['git_status'], str):
        raise ValueError('Invalid Prefix code version record')
    sources = code['source_sha256']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Prefix source_sha256 must be a nonempty object')
    for name, digest in sources.items():
        relative_manifest_path(name)
        if not valid_sha(digest):
            raise ValueError(f'Invalid Prefix source hash: {name}')
    collection = manifest['history_collection']
    require_fields(collection, ('seed_offsets', 'success_episodes', 'attempts_per_seed'), 'Prefix history_collection')
    if (not isinstance(collection['seed_offsets'], list) or not collection['seed_offsets']
            or any(type(value) is not int for value in collection['seed_offsets'])
            or any(type(collection[key]) is not int or collection[key] <= 0
                   for key in ('success_episodes', 'attempts_per_seed'))):
        raise ValueError('Invalid Prefix history collection record')
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Prefix files must be a nonempty hash list')
    for name, digest in files.items():
        relative_manifest_path(name)
        if not valid_sha(digest):
            raise ValueError(f'Invalid Prefix artifact hash: {name}')
    missing = REQUIRED_ARTIFACTS - files.keys()
    if missing:
        raise ValueError('Prefix required artifacts missing from files: ' + ', '.join(sorted(missing)))


def verify_manifest(parent, requested):
    path = artifact_path(parent, 'prefix_manifest.json')
    if not path.exists():
        raise ValueError('Legacy Prefix without prefix_manifest.json has no verified training implementation; '
                         'regenerate the Prefix with manifest version 2. Historical artifacts are not modified.')
    manifest = read_manifest_json(path)
    if (not isinstance(manifest, dict) or type(manifest.get('manifest_version', 1)) is not int
            or manifest.get('manifest_version', 1) != MANIFEST_VERSION):
        raise ValueError('Legacy/unsupported Prefix Manifest version: training implementation provenance cannot '
                         'be inferred from source_sha256; regenerate a version-2 Prefix. No automatic upgrade.')
    validate_manifest_fields(manifest)
    require_compatible(manifest['prefix_spec'], requested)
    for name, expected in manifest['files'].items():
        path = artifact_path(parent, name)
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Prefix artifact hash mismatch: {name}')
    report = read_manifest_json(artifact_path(parent, 'retention.json'))
    require_fields(report, ('stages', 'initial_reference', 'before_first_training'), 'Prefix retention')
    stages = report['stages']
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError('Prefix retention must contain exactly its first stage')
    if any(not isinstance(report[key], dict) for key in ('initial_reference', 'before_first_training')):
        raise ValueError('Invalid Prefix retention reference dictionaries')
    entry = stages[0]
    require_fields(entry, ('phase', 'task', 'checkpoint', 'win_rates'), 'Prefix retention stage')
    if type(entry['phase']) is not int or entry['phase'] != 0 or entry['task'] != manifest['prefix_spec']['stage']['task']:
        raise ValueError('Prefix retention first stage does not match its Spec')
    if (not isinstance(entry['win_rates'], dict) or entry['task'] not in entry['win_rates']
            or type(entry['win_rates'][entry['task']]) not in (int, float)):
        raise ValueError('Invalid Prefix retention first-stage win_rates')
    checkpoint = relative_manifest_path(entry['checkpoint'])
    if not artifact_path(parent, checkpoint).is_dir():
        raise ValueError('Prefix retention checkpoint directory is missing')
    for name in CHECKPOINT_ARTIFACTS:
        relative = checkpoint + '/' + name
        if relative not in manifest['files']:
            raise ValueError(f'Prefix checkpoint artifact missing from files: {relative}')
        # Already hashed above; require the exact files used by learner.load_models.
        if not artifact_path(parent, relative).is_file():
            raise ValueError(f'Prefix checkpoint artifact missing: {relative}')
    return manifest


def resolve_prefix(prefix_id, runs=None):
    matches = []
    for path in Path(runs or ROOT / 'runs').glob('**/prefix_manifest.json'):
        if json.loads(path.read_text()).get('prefix_id') == prefix_id:
            matches.append(path.parent.resolve())
    if len(matches) != 1:
        raise ValueError(f'Prefix ID {prefix_id!r} resolved to {len(matches)} runs; use --parent for an external path')
    return matches[0]
