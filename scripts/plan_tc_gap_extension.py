"""Prepare only the ten newly authorized training branches after offline feasibility."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BATCH='ts5_tc_gap_extension_20261125'
OLD='ts5_tc_coverage_20261123'


def build():
    out=ROOT/'results'/BATCH
    feasibility=json.loads((out/'feasibility.json').read_text())
    if not feasibility['passed']:raise ValueError('Offline feasibility has not passed')
    protocol=json.loads((ROOT/'configs/continual/action_preference/tc_coverage/ts5_vdn64_tc_1m.json').read_text())
    gate=BATCH+'_offline_verify'
    tasks=[dict(id=gate,argv=[str(ROOT/'.venv/bin/python'),'scripts/offline_tc_gap_extension.py','--verify',str(out)],
        cwd=str(ROOT),type='analysis',dependencies=[],queue=BATCH,queue_limit=5,resources=dict(gpu_mib=0,ram_mib=2000),
        metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 TC K1 and min_gap 0.5 extension',results=str(out)))]
    evaluations=[]
    for seed in (2222,3535):
        parent=ROOT/'runs/tc_coverage'/('%s_reference_seed%d'%(OLD,seed))
        for gap,tag in ((0,'gap0'),(.5,'gap05')):
            for level in ('k1',) if gap==0 else ('k1','low','medium','high'):
                method='pair_tc_'+level;identity='%s_%s_%s_seed%d'%(BATCH,tag,level,seed)
                settings=dict(protocol,id=identity,tc_mask_root=str(out/'masks'/('seed%d'%seed)/tag))
                settings[method]=dict(weight=1.,temperature=1.,min_gap=gap,q_scale_floor=.1,pair_seed=3535)
                path=ROOT/'configs/continual/action_preference/tc_coverage/gap_extension'/('%s_%s_seed%d.json'%(tag,level,seed))
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(settings,indent=2)+'\n')
                metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 TC K1 and min_gap 0.5 extension',
                              branch_id=level+'-TC',branch_title=('K1-only' if level=='k1' else level+'-TC')+' '+tag,
                              attempt=1,seed=seed,min_gap=gap,results=str(out))
                run_dir=ROOT/'runs/tc_coverage'/identity
                args=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','vdn_action_set','--algorithm','vdn_1ff',
                      '--method',method,'--seed',str(seed),'--protocol',str(path.relative_to(ROOT)),
                      '--run-id',identity,'--preference-role','shared-branch','--parent',str(parent)]
                task=dict(id=identity,argv=args,cwd=str(ROOT),type='training',run_dir=str(run_dir),dependencies=[gate],
                          queue=BATCH,queue_limit=5,resources=dict(gpu=0,gpu_mib=1400,ram_mib=3500),metadata=metadata)
                tasks.append(task)
                evaluation=identity+'_independent_final';evaluations.append(evaluation)
                tasks.append(dict(id=evaluation,argv=[str(ROOT/'.venv/bin/python'),'scripts/evaluate.py','--run',str(run_dir),'--episodes','128'],
                    cwd=str(ROOT),type='evaluation',dependencies=[identity],queue=BATCH,queue_limit=5,
                    resources=dict(gpu_mib=0,ram_mib=2500),metadata=metadata))
    tasks.append(dict(id=BATCH+'_summary',argv=[str(ROOT/'.venv/bin/python'),'scripts/summarize_tc_gap_extension.py',str(out/'plan.json')],
        cwd=str(ROOT),type='summary',dependencies=evaluations,dependency_mode='terminal',queue=BATCH,queue_limit=5,
        resources=dict(gpu_mib=0,ram_mib=3500),metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 TC K1 and min_gap 0.5 extension')))
    return dict(id=BATCH,tasks=tasks,seeds=[2222,3535],reused_original_plan=str(ROOT/'results'/OLD/'plan.json'),
                offline_verdict=str(out/'feasibility.json'),new_training_branches=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'results'/BATCH/'plan.json')
    args=parser.parse_args();plan=build();args.output.write_text(json.dumps(plan,indent=2)+'\n');print(args.output)
