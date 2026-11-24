"""Prepare the complete authorized two-seed, four-branch TS5-1M scheduler manifest."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BATCH='ts5_tkpr_cpr_1m_20261124'


def build(repo, batch=BATCH, artifacts=None):
    artifacts=artifacts or repo
    runtime=artifacts/'results'/batch
    python=str(repo/'.venv/bin/python')
    tasks=[]
    config='configs/continual/action_preference/ts5/cpr_shared_1m/'
    prefixes={seed:f'{batch}_A_seed{seed}_attempt001' for seed in (3535,2222)}
    common=dict(experiment_group_id=batch,experiment_group_title='TS5-1M CPR shared-prefix comparison')
    def training(rid,seed,slug,method,title,role,parent=None):
        argv=[python,'scripts/run.py','--suite','vdn_action_set','--algorithm','vdn_1ff',
              '--method',method,'--seed',str(seed),'--run-id',rid,
              '--protocol',config+slug+'.json','--preference-role',role]
        if parent:argv+=['--parent',str(repo/'runs'/parent)]
        return dict(id=rid,argv=argv,cwd=str(repo),run_dir=str(repo/'runs'/rid),type='training',
                    dependencies=[parent] if parent else [],queue=batch,queue_limit=2,
                    resources=dict(gpu=0,gpu_mib=2000,ram_mib=10000 if title=='VDN-ER' else 7500),
                    metadata=dict(common,role=role,method=title,seed=seed,prefix_id=parent or rid,
                                  branch_id=f'{slug}_seed{seed}',branch_title=title+' seed'+str(seed),
                                  attempt=1,results=str(runtime)))
    for seed,rid in prefixes.items():
        tasks.append(training(rid,seed,'prefix','topk_preference','Shared A','shared-prefix'))
    for seed in (3535,2222):
        for slug,method,title in (('vdn_er','baseline','VDN-ER'),('tkpr_k1','topk_preference','TKPR-K1'),
                                  ('tkpr_cpr','tkpr_cpr','TKPR-CPR'),('tkpr_refresh','tkpr_refresh','TKPR-Refresh')):
            rid=f'{batch}_{slug}_seed{seed}_attempt001'
            tasks.append(training(rid,seed,slug,method,title,'shared-branch',prefixes[seed]))
            tasks.append(dict(id=rid+'_independent_final',argv=[python,'scripts/evaluate.py','--run',
                              str(repo/'runs'/rid),'--episodes','128'],cwd=str(repo),type='evaluation',
                              run_dir=str(repo/'runs'/rid/'evaluation'),dependencies=[rid],queue=batch,queue_limit=2,
                              resources=dict(gpu_mib=0,ram_mib=2000),
                              metadata=dict(common,inherit_from=rid,method=title,seed=seed,prefix_id=prefixes[seed],
                                            branch_id=f'{slug}_seed{seed}',branch_title=title+' final recheck',attempt=1,
                                            results=str(runtime))))
    tasks.append(dict(id=batch+'_summary',argv=[python,'scripts/summarize_ts5_cpr_1m.py','--plan',
                      str(runtime/'plan.json'),'--output',str(artifacts/'docs/results/comparisons/ts5-tkpr-cpr-1m')],
                      cwd=str(repo),type='summary',run_dir=str(runtime/'summary_task'),queue=batch,queue_limit=2,
                      dependencies=[t['id'] for t in tasks],dependency_mode='terminal',
                      resources=dict(gpu_mib=0,ram_mib=2000),metadata=dict(common,results=str(runtime))))
    return dict(id=batch,tasks=tasks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,default=ROOT)
    parser.add_argument('--artifacts',type=Path,help='Main artifact repository when execution uses a pinned worktree')
    parser.add_argument('--batch',default=BATCH)
    args=parser.parse_args();repo=args.repository.resolve()
    artifacts=args.artifacts.resolve() if args.artifacts else repo
    output=artifacts/'results'/args.batch
    output.mkdir(parents=True,exist_ok=False)
    plan=build(repo,args.batch,artifacts)
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(output/'plan.json')
    print('2 prefixes + 8 branches + 8 independent-process evaluations + 1 terminal summary')


if __name__=='__main__':main()
