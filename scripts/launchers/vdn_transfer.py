"""Launch plans for TS3–TS8, including CHI and TRHR protocols."""
from pathlib import Path
import sys
import yaml

from .common import ROOT, LaunchSpec, load_protocol, prepare_smacv2


def trhr_settings(parser, overrides):
    settings = dict(target_rate=0.05, window_updates=100, low_windows=3,
                    min_condition_samples=128, ema_weight=0.25, gain=0.1, max_offset=0.25)
    for item in overrides:
        key, separator, value = item.partition('=')
        name = key[5:]
        if not separator or not key.startswith('trhr.') or name not in settings:
            parser.error('TRHR overrides require --set trhr.FIELD=VALUE (see docs/TRHR.md)')
        settings[name] = yaml.safe_load(value)
    for name in ('window_updates', 'low_windows', 'min_condition_samples'):
        if type(settings[name]) is not int or settings[name] <= 0:
            parser.error(f'trhr.{name} must be a positive integer')
    for name in ('target_rate', 'ema_weight', 'gain', 'max_offset'):
        value = settings[name]
        if type(value) not in (int, float) or not 0 < value < float('inf'):
            parser.error(f'trhr.{name} must be finite and positive')
    if settings['target_rate'] >= 1 or settings['ema_weight'] > 1:
        parser.error('trhr.target_rate must be below 1 and trhr.ema_weight at most 1')
    return settings


def method_options(a, p):
    trhr_overrides = [item for item in a.set if item.split('=', 1)[0].split('.')[0] == 'trhr']
    trhr = None
    chi_trhr = a.method == 'softprotect_chi_trhr'
    chi = a.method == 'softprotect_chi' or chi_trhr
    factorial = a.suite in ('ts3', 'ts4') and a.algorithm == 'vdn_1ff_128' and not chi
    if a.method == 'softprotect_trhr' and not factorial:
        p.error('combined soft protection and TRHR requires a TS3 or TS4 128-unit protocol')
    if factorial and a.set:
        p.error('Factorial settings belong in the named protocol')
    if a.method == 'trhr' and not factorial:
        if a.algorithm != 'vdn' or a.suite not in ('smac', 'ts3'):
            p.error('TRHR A2 supports two-layer VDN in SMAC and TS3')
        if a.suite == 'ts3' and (a.ts3_role != 'branch' or a.ts3_group != 'N'):
            p.error('TRHR TS3 requires --ts3-role branch and --ts3-group N')
        if a.suite == 'ts3' and len(trhr_overrides) != len(a.set):
            p.error('TS3 TRHR only accepts --set overrides under trhr.*')
        trhr = trhr_settings(p, trhr_overrides)
    elif trhr_overrides:
        p.error('trhr.* overrides require --method trhr')
    chi_control = a.algorithm == 'vdn_1ff' and a.method == 'softprotect' and a.ts3_group == 'A_union'
    return chi, chi_trhr, chi_control, factorial, trhr, trhr_overrides


def build(a, p, out, env, chi, chi_trhr, chi_control, factorial, trhr):
    if chi or chi_control:
        if chi_trhr and ((a.suite, a.algorithm, a.ts3_group) != ('ts4', 'vdn_1ff_128', 'A_chi_union')
                         or a.ts3_role not in ('branch', 'full')):
            p.error('CHI + TRHR requires TS4 128-unit A_chi_union with full or branch')
        if a.suite not in ('ts3', 'ts4', 'ts5', 'ts6', 'ts7', 'ts8') or a.algorithm not in ('vdn', 'vdn_1ff', 'vdn_1ff_128'):
            p.error('CHI requires shared-Agent VDN in TS3–TS8')
        if a.set or a.steps is not None:
            p.error('CHI settings belong in the named protocol')
        groups = ('A1_chi', 'A2_chi', 'A2_chi_union') if a.algorithm == 'vdn' else ('A_chi', 'A_chi_union', 'A_chi_input', 'A_chi_union_input')
        if a.ts3_role not in ('full', 'prefix', 'branch') or bool(a.parent) != (a.ts3_role == 'branch'):
            p.error('CHI requires full, prefix, or branch with --parent')
        if chi_control:
            groups = ('A_union',)
        if a.ts3_group not in (('N',) if a.ts3_role == 'prefix' else groups):
            p.error('CHI uses N for a prefix and a CHI group matching the Agent for full or branch')
        protocol = load_protocol(a.protocol)
        if chi_trhr:
            if not protocol.get('trhr_settings') or a.seed not in protocol.get('seeds', []):
                p.error('use a named TS4 CHI protocol with TRHR settings and the matching seed')
            protocol['trhr'] = protocol['trhr_settings']
            if a.ts3_role == 'full':
                protocol['shared_prefix'] = False
                protocol.pop('chi_prefix_probe', None)
        protocol_id = protocol.get('id', '')
        if not (protocol_id == a.suite or (a.suite == 'ts3' and protocol_id.startswith('ts3_'))
                or (a.suite == 'ts4' and protocol.get('suite') == 'ts4'
                    and protocol.get('algorithm') == a.algorithm and a.ts3_group in protocol['groups'])):
            p.error('suite must match named protocol')
        if a.suite == 'ts8' and (protocol.get('algorithm') != a.algorithm
                                  or a.ts3_group not in protocol['groups']):
            p.error('use the named TS8 protocol and its Agent and group')
        if protocol.get('selection') == 'type_threshold_union':
            p.error('CHI requires overall selection')
        top_k = protocol.get('top_k')
        threshold = protocol.get('threshold_beta', protocol.get('single_ff_detection', {}).get('threshold'))
        if top_k is None and threshold is None:
            threshold = 3
        probe = protocol.get('chi_prefix_probe')
        if a.parent:
            if protocol.get('chi_release_trial'):
                event = Path(a.parent)/'release_event.json'
                if not event.is_file():
                    p.error('CHI release trial requires the parent release event record')
            elif not (ROOT/probe if probe else Path(a.parent)/'chi_probes'/'phase0.th').is_file():
                p.error('CHI branch requires a prefix with a saved fixed probe')
        matched = protocol.get('intervention') == 'single_ff_chi_comparison'
        if chi_control and (not matched or a.ts3_role != 'branch'):
            p.error('the matched soft protection control requires a CHI comparison protocol and prefix')
        intervention = 'single_ff_softprotect' if chi_control else 'vdn_chi'
        screening = ('latest-task threshold selection with historical union' if chi_control
                     else 'CHI fixed-probe reassessment with overall VDN activity')
        protocol.update(algorithm=a.algorithm, groups=['N', *groups], intervention=intervention,
                        chi_comparison=matched, threshold_beta=threshold, screening=screening)
        prepare_smacv2(env, out)
        command=[sys.executable,str(ROOT/'drivers/continual/ts3_train.py'),'--role',a.ts3_role,
                 '--seed',str(a.seed),'--group',a.ts3_group]
        if a.parent:
            command += ['--parent', a.parent]
        thresholds = protocol['thresholds']
        steps = {'prefix': thresholds[0], 'branch': thresholds[-1]-thresholds[0],
                 'full': thresholds[-1]}[a.ts3_role]
        details = dict(protocol=protocol, parent=a.parent, group=a.ts3_group, steps=steps)
    elif a.suite=='ts4' and a.algorithm=='vdn' and a.ts3_role != 'full':
        if a.algorithm!='vdn' or a.method!=('baseline' if a.ts4_group=='N' else 'softprotect'):p.error('ts4 requires VDN and matching group/method')
        protocol=load_protocol(a.protocol)
        prepare_smacv2(env, out)
        command=[sys.executable,str(ROOT/'drivers/continual/ts4_train.py'),'--seed',str(a.seed),'--group',a.ts4_group]
        details={'protocol':protocol,'group':a.ts4_group,'steps':protocol['thresholds'][-1]}
    elif (a.suite in ('ts5', 'ts6', 'ts7', 'ts8')
          or (a.suite == 'ts4' and a.algorithm in ('vdn', 'vdn_1ff'))
          or (a.suite == 'ts3' and a.algorithm == 'vdn_1ff')):
        single_ff = a.algorithm in ('vdn_1ff', 'vdn_1ff_128') and a.suite != 'ts7'
        full_baseline = single_ff or (a.suite == 'ts4' and a.algorithm == 'vdn')
        roles = ('full',) if full_baseline else ('resume',) if a.suite=='ts6' else ('prefix','branch')
        if a.suite in ('ts5', 'ts8') and not single_ff:
            p.error(f'{a.suite} requires a single-feedforward VDN baseline')
        if (not single_ff and a.algorithm!='vdn') or a.ts3_role not in roles:
            p.error(f'{a.suite} requires VDN and one of {roles}')
        if a.ts3_group not in ('N','A2_union') or a.method!=('baseline' if a.ts3_group=='N' else 'softprotect'):
            p.error('group and method must agree')
        if bool(a.parent) != (a.ts3_role in ('branch','resume')):
            p.error('pass --parent only for continuation')
        if a.ts3_role in ('prefix','full') and a.ts3_group!='N':
            p.error('fresh sequence training requires baseline group N')
        protocol=load_protocol(a.protocol)
        if a.suite == 'ts3':
            if (protocol.get('id') != 'ts3_vdn_1ff_chi'
                    or protocol.get('intervention') != 'single_ff_chi_comparison'
                    or a.seed not in protocol['seeds']):
                p.error('use the named TS3 single-feedforward comparison protocol and its seed')
            protocol.pop('intervention')
            protocol.update(top_k=3, single_ff_detection={'interval': 100000,
                                                         'threshold': protocol['threshold_beta']})
        elif a.suite == 'ts4':
            if (protocol.get('suite') != 'ts4'
                    or protocol.get('intervention') != 'single_ff_trhr_factorial'
                    or a.seed not in protocol['seeds']):
                p.error('use the named TS4 comparison protocol and its seed')
            protocol.pop('intervention')
            protocol.pop('trhr_settings')
            protocol.pop('nominal_actual_training_steps', None)
            protocol.update(shared_prefix=False, top_k=3,
                            single_ff_detection={'interval': 100000,
                                                 'threshold': protocol['threshold_beta']})
        elif a.suite == 'ts8':
            if (protocol.get('id') != 'ts8'
                    or protocol.get('intervention') != 'single_ff_chi_comparison'
                    or a.seed not in protocol['seeds']):
                p.error('use the named TS8 single-feedforward comparison protocol and its seed')
            protocol.pop('intervention')
            protocol.update(shared_prefix=False, top_k=3, selection='top_k',
                            single_ff_detection={'interval': 100000,
                                                 'threshold': protocol['threshold_beta']})
        elif protocol.get('id')!=a.suite:
            p.error('suite must match named protocol')
        if full_baseline:
            layers = 'Agent A' if single_ff else 'Agent A1 and A2'
            protocol.update(algorithm=a.algorithm, groups=['N'],
                            screening=f'read-only VDN state-weighted activity at {layers}')
        prepare_smacv2(env, out)
        command=[sys.executable,str(ROOT/'drivers/continual/ts3_train.py'),'--role',a.ts3_role,'--seed',str(a.seed),'--group',a.ts3_group]
        if a.parent:command+=['--parent',a.parent]
        thresholds=protocol['thresholds']
        steps = thresholds[-1]
        if a.ts3_role == 'prefix':
            steps = thresholds[0]
        elif a.ts3_role == 'branch':
            steps -= thresholds[0]
        elif a.ts3_role == 'resume':
            steps -= thresholds[4]
        details={'protocol':protocol,'parent':a.parent,'group':a.ts3_group,'steps':steps}
    elif factorial:
        protocol=load_protocol(a.protocol)
        methods = {'N': 'baseline', 'A_union': 'softprotect',
                   'TRHR': 'trhr', 'A_union_TRHR': 'softprotect_trhr',
                   'A_equal_union': 'softprotect', 'A_union_input': 'softprotect'}
        if (protocol.get('intervention') != 'single_ff_trhr_factorial'
                or protocol.get('suite', 'ts3') != a.suite
                or protocol.get('algorithm') != a.algorithm or a.seed not in protocol['seeds']):
            p.error('use the named 128-unit TRHR protocol for this suite and seed')
        roles = ('prefix', 'branch', 'full') if a.suite == 'ts4' and a.ts3_group == 'TRHR' else ('prefix', 'branch')
        if (a.ts3_role not in roles or bool(a.parent) != (a.ts3_role == 'branch')
                or a.ts3_group not in protocol['groups'] or methods.get(a.ts3_group) != a.method
                or (a.ts3_role == 'prefix' and a.ts3_group != 'N')):
            p.error('use prefix/N, a matching branch with parent, or full/TRHR on TS4')
        prepare_smacv2(env, out)
        command=[sys.executable,str(ROOT/'drivers/continual/ts3_train.py'),'--role',a.ts3_role,
                 '--seed',str(a.seed),'--group',a.ts3_group]
        if a.parent:command+=['--parent',a.parent]
        end=protocol['thresholds']
        branch_start = protocol.get('branch_start_phase', 1)
        details={'protocol':protocol,'parent':a.parent,'group':a.ts3_group,
                 'steps':{'prefix':end[0], 'branch':end[-1]-end[branch_start-1],
                          'full':end[-1]}[a.ts3_role]}
    elif a.suite=='ts3':
        if a.algorithm not in ['qmix','vdn'] or a.ts3_role not in ('prefix','branch','eval_fwt'):p.error('ts3 requires qmix/vdn and --ts3-role')
        if a.ts3_role=='branch' and not a.parent:p.error('ts3 branch requires --parent')
        expected='gem' if a.ts3_group.startswith('GEM_') else ('baseline' if a.ts3_group=='N' else 'softprotect')
        if a.method!=expected and trhr is None:p.error('ts3 group and method must agree')
        if a.ts3_role=='eval_fwt' and (a.algorithm!='vdn' or a.ts3_group!='N' or not a.parent):p.error('FWT evaluation requires VDN baseline and parent')
        protocol = load_protocol(a.protocol)
        if a.suite=='ts3' and a.algorithm!=protocol.get('algorithm','qmix'):p.error('algorithm must match named ts3 protocol')
        if trhr is not None:
            if (protocol['id'] in ('ts5', 'ts6', 'ts7') or 'N' not in protocol['groups']
                    or protocol.get('intervention')):
                p.error('TRHR needs a TS3 VDN protocol with group N and no other intervention')
            protocol['trhr'] = trhr
        elif 'trhr' in protocol:
            p.error('a TS3 TRHR protocol requires --method trhr')
        if a.method=='gem' and protocol.get('intervention')!='a2_local_gem':p.error('gem requires a named a2_local_gem protocol')
        if a.method == 'gem' and a.ts3_group not in protocol['groups']:
            p.error('GEM group must match named TS3 protocol')
        if a.ts3_group == 'GEM_gt3' and (a.ts3_role != 'branch' or protocol.get('gem_score_threshold') != 3):
            p.error('GEM_gt3 requires the threshold-3 branch protocol')
        if a.ts3_role=='eval_fwt' and protocol.get('intervention')!='a2_local_gem':p.error('eval_fwt requires the local GEM protocol')
        prepare_smacv2(env, out)
        command=[sys.executable,str(ROOT/'drivers/continual/ts3_train.py'),'--role',a.ts3_role,'--seed',str(a.seed),'--group',a.ts3_group]
        if a.parent:command+=['--parent',a.parent]
        details={'protocol':protocol,'parent':a.parent,'group':a.ts3_group,'steps':0 if a.ts3_role=='eval_fwt' else (protocol['thresholds'][0] if a.ts3_role=='prefix' else protocol['thresholds'][-1]-protocol['thresholds'][0])}
    return LaunchSpec(command, env, details)
