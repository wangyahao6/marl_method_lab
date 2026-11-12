"""Explicit single-run launcher. No automatic training or implicit queue."""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, tarfile
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def merge(a,b):
    for k,v in b.items():
        if isinstance(v,dict): a[k]=merge(a.get(k,{}),v)
        else: a[k]=v
    return a
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n')
def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['smac','continual','hetero','ts3','ts4','ts5','ts6','ts7','ts8','history','network_similarity'],required=True)
    p.add_argument('--algorithm',choices=['qmix','qplex','dmix','rmix','vdn','vdn_1ff','vdn_1ff_128'],required=True)
    p.add_argument('--method',choices=['baseline','reborn','redo','reset','softprotect','gem','trhr','softprotect_trhr','softprotect_chi','softprotect_chi_trhr'],required=True)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--run-id',required=True)
    p.add_argument('--map')
    p.add_argument('--steps',type=int)
    p.add_argument('--scenario',choices=['task_drift','task_switch'])
    p.add_argument('--protocol',default='configs/continual/knife_fig6.json')
    p.add_argument('--set',action='append',default=[],metavar='KEY=YAML_VALUE')
    p.add_argument('--hetero-role',choices=['prefix','branch'])
    p.add_argument('--parent')
    p.add_argument('--branch-task',choices=['R','M'],default='R')
    p.add_argument('--group',choices=['N','T1','G1','R1','T2','G2','R2','T12','G12','R12'],default='N')
    p.add_argument('--ts3-role',choices=['prefix','branch','eval_fwt','resume','full'])
    p.add_argument('--ts3-group',choices=['N','A1_linear','A1_union','A2_linear','A2_union','GEM_high','GEM_middle','GEM_low','GEM_gt3','A_union','TRHR','A_union_TRHR','A_equal_union','A_union_input','A_chi','A1_chi','A2_chi',
                                             'A_chi_union','A_chi_input','A_chi_union_input','A2_chi_union'])
    p.add_argument('--ts4-group',choices=['N','A1_linear','A2_linear'],default='N')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--experiment-metadata', help='JSON object with conversation and branch metadata')
    a=p.parse_args()
    if a.ts3_group is None:
        if a.method == 'softprotect_chi' and a.ts3_role in ('full', 'branch'):
            a.ts3_group = 'A2_chi_union' if a.algorithm == 'vdn' else 'A_chi_union'
        else:
            a.ts3_group = 'N'

    metadata={}
    if a.experiment_metadata is not None:
        sys.path.insert(0,str(ROOT/'scheduler'))
        from experiment_metadata import fields
        try:
            metadata=fields(json.loads(a.experiment_metadata))
            if 'branch_id' in metadata and 'experiment_group_id' not in metadata:
                raise ValueError('branch_id requires experiment_group_id')
        except ValueError as exc:
            p.error(str(exc))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id): p.error('run-id must be a simple directory name')
    out=ROOT/'runs'/a.run_id
    env=os.environ.copy();env['MARL_RUN_DIR']=str(out)
    env.setdefault('SC2PATH',str(ROOT/'runtime/StarCraftII'))
    env.setdefault('OMP_NUM_THREADS','1');env.setdefault('MKL_NUM_THREADS','1')
    if a.method=='gem' and a.suite!='ts3':p.error('gem currently supports TS3 only')
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
    if chi or chi_control:
        if chi_trhr and (a.suite, a.algorithm, a.ts3_role, a.ts3_group) != (
                'ts4', 'vdn_1ff_128', 'branch', 'A_chi_union'):
            p.error('CHI + TRHR requires the TS4 128-unit A_chi_union branch')
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
        protocol = json.loads((ROOT/a.protocol).read_text())
        if chi_trhr:
            if not protocol.get('trhr_settings') or a.seed not in protocol.get('seeds', []):
                p.error('use a named TS4 CHI protocol with TRHR settings and the matching seed')
            protocol['trhr'] = protocol['trhr_settings']
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
        if a.parent and not (ROOT/probe if probe else Path(a.parent)/'chi_probes'/'phase0.th').is_file():
            p.error('CHI branch requires a prefix with a saved fixed probe')
        matched = protocol.get('intervention') == 'single_ff_chi_comparison'
        if chi_control and (not matched or a.ts3_role != 'branch'):
            p.error('the matched soft protection control requires a CHI comparison protocol and prefix')
        intervention = 'single_ff_softprotect' if chi_control else 'vdn_chi'
        screening = ('latest-task threshold selection with historical union' if chi_control
                     else 'CHI fixed-probe reassessment with overall VDN activity')
        protocol.update(algorithm=a.algorithm, groups=['N', *groups], intervention=intervention,
                        chi_comparison=matched, threshold_beta=threshold, screening=screening)
        env['MARL_PROTOCOL']=str(out/'protocol.json')
        env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/ts3_train.py'),'--role',a.ts3_role,
                 '--seed',str(a.seed),'--group',a.ts3_group]
        if a.parent:
            command += ['--parent', a.parent]
        thresholds = protocol['thresholds']
        steps = {'prefix': thresholds[0], 'branch': thresholds[-1]-thresholds[0],
                 'full': thresholds[-1]}[a.ts3_role]
        details = dict(protocol=protocol, parent=a.parent, group=a.ts3_group, steps=steps)
    elif a.suite=='smac':
        if not a.map or a.steps is None or a.steps<=0: p.error('smac requires --map and positive --steps')
        if a.scenario or a.method=='softprotect': p.error('softprotect currently uses the continual driver')
        suffix={'baseline':'','reborn':'_b','redo':'_r','reset':'_s','trhr':''}[a.method]
        preset=a.algorithm+suffix
        f=ROOT/'src/config/algs'/f'{preset}.yaml'
        if not f.exists():p.error('No upstream preset for this combination; see docs/METHODS.md')
        cfg={}
        for path in ['src/config/default.yaml','src/config/envs/sc2.yaml',str(f.relative_to(ROOT))]:
            merge(cfg,yaml.safe_load((ROOT/path).read_text()))
        # Immutable experiment identity is kept separate from upstream method dispatch name.
        cfg.update(seed=a.seed,t_max=a.steps,local_results_path=str(out),save_model=True)
        cfg['env_args'].update(map_name=a.map,seed=a.seed)
        for item in a.set:
            if item in trhr_overrides:
                continue
            if '=' not in item:p.error('--set requires KEY=VALUE')
            key,value=item.split('=',1);d=cfg
            if key.split('.')[0] in ['name','seed','t_max','local_results_path','unique_token','evaluate','checkpoint_path']:p.error('use the dedicated training arguments; evaluation is a separate command')
            if key in ['env_args.map_name','env_args.seed']:p.error('use --map / --seed')
            keys=key.split('.')
            for k in keys[:-1]: d=d.setdefault(k,{})
            d[keys[-1]]=yaml.safe_load(value)
        if trhr is not None:
            if (cfg['env'], cfg['mac'], cfg['agent'], cfg['learner'], cfg['mixer']) != (
                'sc2', 'basic_mac', 'rnn', 'q_learner', 'vdn'
            ):
                p.error('TRHR requires the standard SMAC VDN controller, agent and learner')
            cfg['trhr'] = trhr
        if cfg.get('a2_trace') and (
            a.method not in ('baseline', 'trhr') or
            (cfg['env'], cfg['mac'], cfg['agent'], cfg['learner'], cfg['mixer'], cfg['runner'])
            != ('sc2', 'basic_mac', 'rnn', 'q_learner', 'vdn', 'episode')
        ):
            p.error('a2_trace requires static SMAC two-layer VDN baseline or TRHR')
        # Supply a fully merged config to Sacred through a simple dedicated entrypoint.
        command=[sys.executable,str(ROOT/'scripts/smac_entry.py'),str(out/'config.json')]
        details={'config':cfg,'preset':preset}
    elif a.suite=='network_similarity':
        if a.algorithm!='vdn' or a.method!='baseline':p.error('network similarity requires frozen VDN baseline models')
        protocol=json.loads((ROOT/a.protocol).read_text())
        env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
        command=[sys.executable,str(ROOT/'scripts/network_similarity.py'),'--protocol',str(out/'protocol.json'),'--output',str(out)]
        details={'protocol':protocol,'steps':0,'mode':'read-only offline analysis of saved models'}
    elif a.suite=='history':
        if a.algorithm!='vdn' or a.method!='baseline':p.error('history requires VDN baseline')
        protocol=json.loads((ROOT/a.protocol).read_text())
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/history_train.py'),'--seed',str(a.seed)]
        if a.parent:command+=['--parent',a.parent]
        details={'protocol':protocol,'parent':a.parent,'steps':2000000}
    elif a.suite=='ts4' and a.algorithm=='vdn':
        if a.algorithm!='vdn' or a.method!=('baseline' if a.ts4_group=='N' else 'softprotect'):p.error('ts4 requires VDN and matching group/method')
        protocol=json.loads((ROOT/a.protocol).read_text())
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/ts4_train.py'),'--seed',str(a.seed),'--group',a.ts4_group]
        details={'protocol':protocol,'group':a.ts4_group,'steps':protocol['thresholds'][-1]}
    elif a.suite in ('ts5','ts6','ts7','ts8') or (a.suite in ('ts3', 'ts4') and a.algorithm == 'vdn_1ff'):
        single_ff = a.algorithm in ('vdn_1ff', 'vdn_1ff_128') and a.suite != 'ts7'
        roles = ('full',) if single_ff else ('resume',) if a.suite=='ts6' else ('prefix','branch')
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
        protocol=json.loads((ROOT/a.protocol).read_text())
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
                p.error('use the named TS4 single-feedforward comparison protocol and its seed')
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
        if single_ff:
            protocol.update(algorithm=a.algorithm, groups=['N'],
                            screening='read-only VDN state-weighted activity at Agent A')
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
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
        protocol=json.loads((ROOT/a.protocol).read_text())
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
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
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
        protocol_path=(ROOT/a.protocol).resolve();protocol=json.loads(protocol_path.read_text())
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
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/ts3_train.py'),'--role',a.ts3_role,'--seed',str(a.seed),'--group',a.ts3_group]
        if a.parent:command+=['--parent',a.parent]
        details={'protocol':protocol,'parent':a.parent,'group':a.ts3_group,'steps':0 if a.ts3_role=='eval_fwt' else (protocol['thresholds'][0] if a.ts3_role=='prefix' else protocol['thresholds'][-1]-protocol['thresholds'][0])}
    elif a.suite=='hetero':
        if a.algorithm!='qmix' or not a.hetero_role:p.error('hetero requires qmix and --hetero-role')
        if a.hetero_role=='branch' and not a.parent:p.error('branch requires --parent')
        protocol_path=(ROOT/a.protocol).resolve();protocol=json.loads(protocol_path.read_text())
        if a.suite=='ts3' and a.algorithm!=protocol.get('algorithm','qmix'):p.error('algorithm must match named ts3 protocol')
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/hetero_train.py'),'--role',a.hetero_role,'--seed',str(a.seed),'--task',a.branch_task,'--group',a.group]
        if a.parent:command+=['--parent',a.parent]
        details={'protocol':protocol,'parent':a.parent,'branch_task':a.branch_task,'group':a.group,'steps':1000000 if a.hetero_role=='prefix' else 1500000}
    else:
        if a.algorithm!='qmix' or a.method not in ['baseline','softprotect']:p.error('continual driver currently supports QMIX baseline/softprotect only')
        if not a.scenario:p.error('continual requires --scenario')
        if a.map or a.steps is not None or a.set:p.error('continual settings belong in a named --protocol JSON')
        protocol_path=(ROOT/a.protocol).resolve()
        protocol=json.loads(protocol_path.read_text())
        env['MARL_PROTOCOL']=str(out/'protocol.json')
        env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/train.py'),'--algorithm',a.algorithm,'--condition',a.method,'--scenario',a.scenario,'--seed',str(a.seed)]
        details={'protocol':protocol,'scenario':a.scenario,'steps':protocol['scenarios'][a.scenario][-1][0]}
    plan={'request':vars(a),'command':command,'output':str(out),'details':details}
    if metadata:
        plan['metadata']=metadata
    if a.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if (ROOT/'runtime/scheduler/enabled').exists():
        sys.path.insert(0,str(ROOT/'scheduler'))
        from service import forward_run
        task_type='evaluation' if a.ts3_role=='eval_fwt' or a.suite=='network_similarity' else 'training'
        scheduled=forward_run(a.run_id,[sys.executable,'scripts/run.py']+sys.argv[1:],ROOT,out,task_type,metadata)
        if scheduled or metadata:
            plan['metadata']=dict(metadata,**scheduled)
    out.mkdir(parents=True,exist_ok=False)
    if os.name == 'nt':
        temp = out/'tmp'
        temp.mkdir()
        env.update(TEMP=str(temp), TMP=str(temp))
    if a.suite=='smac':save(out/'config.json',cfg)
    else:save(out/'protocol.json',protocol)
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    files={x for x in tracked if x and (ROOT/x).is_file()}
    for folder in ['src','scripts','drivers','configs']:
        files.update(str(x.relative_to(ROOT)) for x in (ROOT/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc')
    hashes={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sorted(files)}
    save(out/'launch.json',dict(plan,git_commit=git('rev-parse','HEAD'),git_status=git('status','--porcelain'),source_sha256=hashes,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.executable,sc2path=env['SC2PATH'],temp_dir=env.get('TEMP')))
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for name in sorted(files):tar.add(ROOT/name,arcname=name,recursive=False)
    status={'state':'running','meaning':'process execution only; experiment completion requires result review'}
    save(out/'launcher_status.json',status)
    try:
        with (out/'console.log').open('w') as log:
            code=subprocess.call(command,cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT)
    except BaseException as e:
        save(out/'launcher_status.json',dict(status,state='interrupted',error=repr(e)));raise
    save(out/'launcher_status.json',dict(status,state='exited' if code==0 else 'failed',exit_code=code,ended=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    raise SystemExit(code)
if __name__=='__main__':main()
