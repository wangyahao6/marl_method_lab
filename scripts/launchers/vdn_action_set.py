"""Launch plan for the vdn_action_set suite."""
from pathlib import Path
import sys
from .common import ROOT, DEFAULT_PROTOCOL, LEGACY_DEFAULT_PROTOCOL, LaunchSpec, load_protocol, prepare_smacv2


def build(a, p, out, env):
    from action_set_protocol import validate, validate_fork

    if (a.protocol in (DEFAULT_PROTOCOL, LEGACY_DEFAULT_PROTOCOL) or a.map or a.steps is not None or a.set
            or a.ts3_role or a.hetero_role or a.scenario or a.ts3_group != 'N'
            or a.ts4_group != 'N' or a.group != 'N'):
        p.error('vdn_action_set takes its complete sequence and settings from a named --protocol JSON')
    try:
        protocol = validate(load_protocol(a.protocol), a.algorithm, a.method)
        validate_fork(protocol, a.preference_role, a.parent)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    prepare_smacv2(env, out)
    command = [sys.executable, str(ROOT / 'drivers/continual/action_set_train.py'), '--seed', str(a.seed)]
    stages = protocol['stages']
    if a.preference_role:
        command += ['--preference-role', a.preference_role]
        stages = {'prefix': stages[:-1], 'keep': stages[-1:], 'release': stages[-1:],
                  'shared-prefix': stages[:1], 'shared-branch': stages[1:],
                  'pair-reference': stages[:-1]}[a.preference_role]
    if a.capture_timing:
        if protocol[a.method]['k'] != 1:
            p.error('--capture-timing requires TKPR-K1')
        command += ['--capture-timing']
    if a.parent:
        command += ['--parent', str(Path(a.parent).resolve())]
    details = dict(protocol=protocol, steps=sum(stage['steps'] for stage in stages))
    if a.preference_role:
        details.update(preference_role=a.preference_role, parent=a.parent)
    return LaunchSpec(command, env, details)
