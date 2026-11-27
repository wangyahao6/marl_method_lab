"""Two seeds x two own-policy history methods; no source-history or prefix retraining."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BATCH='ts5_tkpr_own_policy_20261127'


def prepare():
    main=Path('/home/wyh/marl_method_lab');out=main/'results'/BATCH;out.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for seed in (2222,3535):
        parent=main/'runs/tc_coverage'/f'ts5_tc_coverage_20261123_reference_seed{seed}'
        for method,title in (('tkpr_k1_per_task8','TKPR-K1-PerTask8'),('tkpr_win_refresh','TKPR-WinRefresh')):
            rid=f'{BATCH}_{method}_seed{seed}_attempt001';run=ROOT/'runs'/rid
            cfg=f'configs/continual/action_preference/ts5/own_policy/{method}_seed{seed}.json'
            meta=dict(experiment_group_id=BATCH,experiment_group_title='TS5 own-policy PerTask8 vs win-gated replacement',
                      branch_id=f'{method}_seed{seed}',branch_title=f'{title} seed{seed}',method=title,seed=seed,
                      attempt=1,prefix_id=parent.name,results=str(out))
            tasks.append(dict(id=rid,argv=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','vdn_action_set','--algorithm','vdn_1ff',
                          '--method',method,'--seed',str(seed),'--protocol',cfg,'--run-id',rid,'--preference-role','shared-branch','--parent',str(parent)],
                          cwd=str(ROOT),run_dir=str(run),type='training',dependencies=[],queue=BATCH,queue_limit=6,
                          resources=dict(gpu=0,gpu_mib=2000,ram_mib=7500),metadata=meta))
            tasks.append(dict(id=rid+'_independent_final',argv=[str(ROOT/'.venv/bin/python'),'scripts/evaluate.py','--run',str(run),'--episodes','128'],
                          cwd=str(ROOT),type='evaluation',dependencies=[rid],queue=BATCH,queue_limit=6,
                          resources=dict(gpu_mib=0,ram_mib=2500),metadata=meta))
    tasks.append(dict(id=BATCH+'_report',argv=[str(ROOT/'.venv/bin/python'),'scripts/report_k1_per_task8.py',str(out/'plan.json'),'--commit'],
                     cwd=str(ROOT),type='summary',dependencies=[t['id'] for t in tasks],dependency_mode='terminal',queue=BATCH,queue_limit=6,
                     resources=dict(gpu_mib=0,ram_mib=2500),metadata=dict(experiment_group_id=BATCH,results=str(out))))
    plan=dict(id=BATCH,seeds=[2222,3535],own_policy_history=True,tasks=tasks,nominal_new_steps=16000000,
              superseded_batch='ts5_tkpr_k1_per_task8_20261127',cleanup_record=str(main/'results/stopped_per_task8_cleanup_20261127.json'),
              baseline_dataset=str(main/'docs/results/parameter-ablations/ts5-native-tkpr-k1-gap'),
              comparison='Primary: WinRefresh minus corrected own-policy PerTask8. Old fixed-FT-history native K1 is historical context, not budget-only control.')
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(out/'plan.json')


if __name__=='__main__':prepare()
