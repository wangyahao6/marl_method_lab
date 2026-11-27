"""Prepare two budget-only branches from the exact native K1 gap0.5 reference states."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BATCH='ts5_tkpr_k1_per_task8_20261127'


def prepare():
    main=Path('/home/wyh/marl_method_lab');out=main/'results'/BATCH;out.mkdir(parents=True,exist_ok=False)
    tasks=[];parents={}
    for seed in (2222,3535):
        parent=main/'runs/tc_coverage'/f'ts5_tc_coverage_20261123_reference_seed{seed}'
        baseline=main/'runs'/f'ts5_native_tkpr_k1_gap_20261126_gap05_seed{seed}'
        # No SC2 probe: verify the exact completed control and existing reference inputs.
        assert json.loads((baseline/'status.json').read_text())['status']=='completed'
        for phase in range(4):(parent/f'action_set_memory/phase{phase:04d}/success.th').stat()
        (parent/'return_state.th').stat()
        rid=f'{BATCH}_seed{seed}_attempt001';run=ROOT/'runs'/rid
        cfg=f'configs/continual/action_preference/ts5/tkpr_k1_per_task8/seed{seed}.json'
        metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 native K1 budget-only PerTask8',
                      branch_id=f'per_task8_seed{seed}',branch_title=f'TKPR-K1-PerTask8 seed{seed}',
                      method='TKPR-K1-PerTask8',seed=seed,attempt=1,results=str(out),baseline_run_id=baseline.name,prefix_id=parent.name)
        task=dict(id=rid,argv=[str(ROOT/'.venv/bin/python'),'scripts/run.py','--suite','vdn_action_set','--algorithm','vdn_1ff',
                    '--method','tkpr_k1_per_task8','--seed',str(seed),'--protocol',cfg,'--run-id',rid,
                    '--preference-role','shared-branch','--parent',str(parent)],cwd=str(ROOT),run_dir=str(run),type='training',
                    dependencies=[],queue=BATCH,queue_limit=6,resources=dict(gpu=0,gpu_mib=2000,ram_mib=7500),metadata=metadata)
        tasks.append(task)
        tasks.append(dict(id=rid+'_independent_final',argv=[str(ROOT/'.venv/bin/python'),'scripts/evaluate.py','--run',str(run),'--episodes','128'],
                    cwd=str(ROOT),type='evaluation',dependencies=[rid],queue=BATCH,queue_limit=6,
                    resources=dict(gpu_mib=0,ram_mib=2500),metadata=metadata))
        parents[str(seed)]=str(parent)
    tasks.append(dict(id=BATCH+'_report',argv=[str(ROOT/'.venv/bin/python'),'scripts/report_k1_per_task8.py',str(out/'plan.json'),'--commit'],
                 cwd=str(ROOT),type='summary',dependencies=[t['id'] for t in tasks],dependency_mode='terminal',queue=BATCH,queue_limit=6,
                 resources=dict(gpu_mib=0,ram_mib=2500),metadata=dict(experiment_group_id=BATCH,experiment_group_title='TS5 native K1 budget-only PerTask8',results=str(out))))
    plan=dict(id=BATCH,seeds=[2222,3535],tasks=tasks,reference_runs=parents,
              baseline_dataset=str(main/'docs/results/parameter-ablations/ts5-native-tkpr-k1-gap'),
              method='tkpr_k1_per_task8',min_gap=.5,batch_episodes_per_task=8,nominal_new_steps=8000000,
              gradient_monitor_policy='New branches interval100 by user confirmation; baseline interval0 remains missing')
    (out/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');print(out/'plan.json')


if __name__=='__main__':prepare()
