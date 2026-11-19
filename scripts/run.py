"""Explicit single-run launcher. No automatic training or implicit queue."""
import argparse, datetime, hashlib, json, os, re, subprocess, sys, tarfile
from pathlib import Path
from launchers import continual, hetero, history, network_similarity, smac, vdn_action_set, vdn_transfer
from launchers.common import DEFAULT_PROTOCOL, ROOT, save


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def main():
    sys.path.insert(0, str(ROOT / 'drivers/continual'))
    from action_set_protocol import ASPR_MAX_METHODS, FORK_ROLES, PREFERENCE_RANK_METHODS, validate
    from common.exploration import epsilon_on_switch
    preference_methods = (*ASPR_MAX_METHODS, *PREFERENCE_RANK_METHODS)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['smac','continual','hetero','ts3','ts4','ts5','ts6','ts7','ts8','history','network_similarity','vdn_action_set'],required=True)
    p.add_argument('--algorithm',choices=['qmix','qplex','dmix','rmix','vdn','vdn_1ff','vdn_1ff_128'],required=True)
    p.add_argument('--method',choices=['baseline','reborn','redo','reset','softprotect','gem','trhr','softprotect_trhr','softprotect_chi','softprotect_chi_trhr',*preference_methods],required=True)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--run-id',required=True)
    p.add_argument('--map')
    p.add_argument('--steps',type=int)
    p.add_argument('--scenario',choices=['task_drift','task_switch'])
    p.add_argument('--protocol',default=DEFAULT_PROTOCOL)
    p.add_argument('--replay-on-switch', choices=('retain', 'ft', 'er'),
                   help='Replay mode for VDN continual training (ER supports baseline and CHI)')
    p.add_argument('--set',action='append',default=[],metavar='KEY=YAML_VALUE')
    p.add_argument('--hetero-role',choices=['prefix','branch'])
    p.add_argument('--parent')
    p.add_argument('--preference-role', choices=FORK_ROLES)
    p.add_argument('--capture-timing', action='store_true')
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
            a.ts3_group = 'A2_chi' if a.algorithm == 'vdn' else 'A_chi'
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
    if a.capture_timing and (a.suite, a.method, a.preference_role) != (
            'vdn_action_set', 'topk_preference', 'shared-branch'):
        p.error('--capture-timing requires a shared TKPR-K1 branch')
    if a.preference_role and a.suite != 'vdn_action_set':
        p.error('--preference-role requires --suite vdn_action_set')
    if a.method in preference_methods and a.suite != 'vdn_action_set':
        p.error(f'{a.method} requires --suite vdn_action_set')
    if a.method=='gem' and a.suite!='ts3':p.error('gem currently supports TS3 only')
    chi, chi_trhr, chi_control, factorial, trhr, trhr_overrides = vdn_transfer.method_options(a, p)
    if a.suite == 'smac':
        spec = smac.build(a, p, out, env, trhr, trhr_overrides)
    elif a.suite in ('ts3', 'ts4', 'ts5', 'ts6', 'ts7', 'ts8'):
        spec = vdn_transfer.build(a, p, out, env, chi, chi_trhr, chi_control, factorial, trhr)
    else:
        builders = {
            'continual': continual.build,
            'hetero': hetero.build,
            'history': history.build,
            'network_similarity': network_similarity.build,
            'vdn_action_set': vdn_action_set.build,
        }
        spec = builders[a.suite](a, p, out, env)
    command, env, details = spec.command, spec.env, spec.details
    protocol = details.get('protocol')
    if a.suite == 'vdn_action_set' and protocol.get('pair_ablation_experiment'):
        out = out.parent / protocol['pair_ablation_experiment'] / out.name
        env['MARL_RUN_DIR'] = str(out)
        env['MARL_PROTOCOL'] = str(out / 'protocol.json')
    if 'protocol' in details and a.suite != 'network_similarity':
        try:
            protocol['epsilon_on_switch'] = epsilon_on_switch(protocol)
        except ValueError as exc:
            p.error(str(exc))
    configurable_replay = (a.suite in ('ts3', 'ts4', 'ts5', 'ts6', 'ts7', 'ts8', 'history')
                           and a.algorithm in ('vdn', 'vdn_1ff', 'vdn_1ff_128')
                           and a.ts3_role != 'eval_fwt')
    replay = a.replay_on_switch
    if replay is None and 'protocol' in details and a.suite != 'network_similarity':
        replay = protocol.get('replay_on_switch')
    if configurable_replay and replay is None:
        replay = 'ft'
    if a.suite == 'vdn_action_set' and replay is not None:
        try:
            protocol = validate(dict(protocol, replay_on_switch=replay), a.algorithm, a.method)
        except ValueError as exc:
            p.error(str(exc))
        details['protocol'] = protocol
    elif replay is not None:
        if replay not in ('retain', 'ft', 'er'):
            p.error('replay_on_switch must be retain, ft or er')
        if not configurable_replay:
            p.error('replay_on_switch is supported for VDN continual training')
        if replay == 'er' and a.method not in ('baseline', 'softprotect_chi'):
            p.error('VDN-ER is supported for VDN continual baselines and CHI')
        if replay == 'er' and a.ts3_role == 'resume':
            p.error('VDN-ER requires historical replay; the TS6 weight-only resume has none')
        protocol['replay_on_switch'] = replay
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
    if a.suite=='smac':save(out/'config.json',details['config'])
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
