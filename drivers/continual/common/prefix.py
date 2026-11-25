"""First-stage training identity and immutable shared-prefix artifact metadata."""
import copy
import hashlib
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
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
                    source_sha256={str(p.relative_to(ROOT)): file_hash(p) for folder in ('src', 'drivers')
                                   for p in sorted((ROOT / folder).rglob('*.py'))})
    files = [out / 'retention.json', out / 'config.json', out / 'env_info.json', out / 'return_state.th', out / 'prefix_replay.th', out / 'initial_agent.th',
             *sorted(checkpoint.glob('*.th')),
             *sorted((out / 'action_set_memory').rglob('*.th'))]
    manifest = dict(manifest_version=MANIFEST_VERSION, prefix_id='prefix-' + uuid4().hex, state_version=STATE_VERSION,
                    prefix_spec=spec, prefix_spec_sha256=spec_hash(spec), code_version=code,
                    history_collection=collection,
                    files={str(path.relative_to(out)): file_hash(path) for path in files})
    with (out / 'prefix_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return manifest


def verify_manifest(parent, requested):
    path = Path(parent) / 'prefix_manifest.json'
    if not path.exists():
        raise ValueError('Legacy Prefix without prefix_manifest.json has no verified training implementation; '
                         'regenerate the Prefix with manifest version 2. Historical artifacts are not modified.')
    manifest = json.loads(path.read_text())
    if manifest.get('manifest_version', 1) != MANIFEST_VERSION:
        raise ValueError('Legacy/unsupported Prefix Manifest version: training implementation provenance cannot '
                         'be inferred from source_sha256; regenerate a version-2 Prefix. No automatic upgrade.')
    if 'implementation' not in manifest['prefix_spec']:
        raise ValueError('Prefix Manifest lacks its baseline training implementation fingerprint')
    if manifest['state_version'] != STATE_VERSION:
        raise ValueError(f"Unsupported prefix state version: {manifest['state_version']}")
    if spec_hash(manifest['prefix_spec']) != manifest['prefix_spec_sha256']:
        raise ValueError('Prefix Spec hash mismatch')
    require_compatible(manifest['prefix_spec'], requested)
    for name, expected in manifest['files'].items():
        path = Path(parent) / name
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Prefix artifact hash mismatch: {name}')
    return manifest


def resolve_prefix(prefix_id, runs=None):
    matches = []
    for path in Path(runs or ROOT / 'runs').glob('**/prefix_manifest.json'):
        if json.loads(path.read_text()).get('prefix_id') == prefix_id:
            matches.append(path.parent.resolve())
    if len(matches) != 1:
        raise ValueError(f'Prefix ID {prefix_id!r} resolved to {len(matches)} runs; use --parent for an external path')
    return matches[0]
