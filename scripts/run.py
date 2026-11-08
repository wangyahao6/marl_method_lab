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
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['smac','continual','hetero','ts3','ts4','ts6','ts7','history','network_similarity'],required=True)
    p.add_argument('--algorithm',choices=['qmix','qplex','dmix','rmix','vdn','vdn_1ff','vdn_1ff_128'],required=True)
    p.add_argument('--method',choices=['baseline','reborn','redo','reset','softprotect','gem'],required=True)
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
    p.add_argument('--ts3-role',choices=['prefix','branch','eval_fwt','resume'])
    p.add_argument('--ts3-group',choices=['N','A1_linear','A1_union','A2_linear','A2_union','GEM_high','GEM_middle','GEM_low'],default='N')
    p.add_argument('--ts4-group',choices=['N','A1_linear','A2_linear'],default='N')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.run_id): p.error('run-id must be a simple directory name')
    out=ROOT/'runs'/a.run_id
    env=os.environ.copy();env['MARL_RUN_DIR']=str(out)
    env.setdefault('SC2PATH',str(ROOT/'runtime/StarCraftII'))
    env.setdefault('OMP_NUM_THREADS','1');env.setdefault('MKL_NUM_THREADS','1')
    if a.method=='gem' and a.suite!='ts3':p.error('gem currently supports TS3 only')
    if a.suite=='smac':
        if not a.map or a.steps is None or a.steps<=0: p.error('smac requires --map and positive --steps')
        if a.scenario or a.method=='softprotect': p.error('softprotect currently uses the continual driver')
        suffix={'baseline':'','reborn':'_b','redo':'_r','reset':'_s'}[a.method]
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
            if '=' not in item:p.error('--set requires KEY=VALUE')
            key,value=item.split('=',1);d=cfg
            if key.split('.')[0] in ['name','seed','t_max','local_results_path','unique_token','evaluate','checkpoint_path']:p.error('use the dedicated training arguments; evaluation is a separate command')
            if key in ['env_args.map_name','env_args.seed']:p.error('use --map / --seed')
            keys=key.split('.')
            for k in keys[:-1]: d=d.setdefault(k,{})
            d[keys[-1]]=yaml.safe_load(value)
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
    elif a.suite=='ts4':
        if a.algorithm!='vdn' or a.method!=('baseline' if a.ts4_group=='N' else 'softprotect'):p.error('ts4 requires VDN and matching group/method')
        protocol=json.loads((ROOT/a.protocol).read_text())
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/ts4_train.py'),'--seed',str(a.seed),'--group',a.ts4_group]
        details={'protocol':protocol,'group':a.ts4_group,'steps':protocol['thresholds'][-1]}
    elif a.suite in ('ts6','ts7'):
        roles = ('resume',) if a.suite=='ts6' else ('prefix','branch')
        if a.algorithm!='vdn' or a.ts3_role not in roles:
            p.error(f'{a.suite} requires VDN and one of {roles}')
        if a.ts3_group not in ('N','A2_union') or a.method!=('baseline' if a.ts3_group=='N' else 'softprotect'):
            p.error('group and method must agree')
        if bool(a.parent) != (a.ts3_role!='prefix') or (a.ts3_role=='prefix' and a.ts3_group!='N'):
            p.error('use a shared baseline prefix and pass --parent for continuation')
        protocol=json.loads((ROOT/a.protocol).read_text())
        if protocol.get('id')!=a.suite:p.error('suite must match named protocol')
        env['MARL_PROTOCOL']=str(out/'protocol.json');env['SC2PATH']=str(ROOT/'runtime/SMACv2StarCraftII')
        command=[sys.executable,str(ROOT/'drivers/continual/ts3_train.py'),'--role',a.ts3_role,'--seed',str(a.seed),'--group',a.ts3_group]
        if a.parent:command+=['--parent',a.parent]
        thresholds=protocol['thresholds']
        steps={'prefix':thresholds[0],'branch':thresholds[-1]-thresholds[0],'resume':thresholds[-1]-thresholds[4]}[a.ts3_role]
        details={'protocol':protocol,'parent':a.parent,'group':a.ts3_group,'steps':steps}
    elif a.suite=='ts3':
        if a.algorithm not in ['qmix','vdn'] or a.ts3_role not in ('prefix','branch','eval_fwt'):p.error('ts3 requires qmix/vdn and --ts3-role')
        if a.ts3_role=='branch' and not a.parent:p.error('ts3 branch requires --parent')
        expected='gem' if a.ts3_group.startswith('GEM_') else ('baseline' if a.ts3_group=='N' else 'softprotect')
        if a.method!=expected:p.error('ts3 group and method must agree')
        if a.ts3_role=='eval_fwt' and (a.algorithm!='vdn' or a.ts3_group!='N' or not a.parent):p.error('FWT evaluation requires VDN baseline and parent')
        protocol_path=(ROOT/a.protocol).resolve();protocol=json.loads(protocol_path.read_text())
        if a.suite=='ts3' and a.algorithm!=protocol.get('algorithm','qmix'):p.error('algorithm must match named ts3 protocol')
        if a.method=='gem' and protocol.get('intervention')!='a2_local_gem':p.error('gem requires a named a2_local_gem protocol')
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
    if a.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    if (ROOT/'runtime/scheduler/enabled').exists():
        sys.path.insert(0,str(ROOT/'scheduler'))
        from service import forward_run
        task_type='evaluation' if a.ts3_role=='eval_fwt' or a.suite=='network_similarity' else 'training'
        forward_run(a.run_id,[sys.executable,'scripts/run.py']+sys.argv[1:],ROOT,out,task_type)
    out.mkdir(parents=True,exist_ok=False)
    if a.suite=='smac':save(out/'config.json',cfg)
    else:save(out/'protocol.json',protocol)
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode().split('\0')
    files={x for x in tracked if x and (ROOT/x).is_file()}
    for folder in ['src','scripts','drivers','configs']:
        files.update(str(x.relative_to(ROOT)) for x in (ROOT/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts and x.suffix!='.pyc')
    hashes={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sorted(files)}
    save(out/'launch.json',dict(plan,git_commit=git('rev-parse','HEAD'),git_status=git('status','--porcelain'),source_sha256=hashes,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),python=sys.executable,sc2path=env['SC2PATH']))
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
