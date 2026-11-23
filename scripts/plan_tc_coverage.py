"""Prepare the complete TS5 TC-coverage dependency plan; no implicit submission."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = 'ts5_tc_coverage_20261123'
SEEDS = (2222, 3535, 4646)
PROTOCOL = 'configs/continual/action_preference/tc_coverage/ts5_vdn64_tc_1m.json'


def build():
    out = ROOT / 'results' / BATCH
    python = str(ROOT / '.venv/bin/python')
    tasks, refs, audits, branches = [], [], [], []
    def task(identity, argv, kind, dependencies, branch):
        item = dict(id=identity, argv=[python, *argv], cwd=str(ROOT), type=kind,
                    dependencies=dependencies, queue=BATCH, queue_limit=5,
                    resources=dict(gpu=0, gpu_mib=1400 if kind=='training' else 0,
                                   ram_mib=3500 if kind=='training' else 3000),
                    metadata=dict(experiment_group_id=BATCH, experiment_group_title='TS5-1M TC Coverage Ablation',
                                  branch_id=branch, branch_title=branch, attempt=1, results=str(out)))
        tasks.append(item)
        return item
    for seed in SEEDS:
        ref = '%s_reference_seed%d' % (BATCH, seed)
        refs.append(ref)
        args = ['scripts/run.py','--suite','vdn_action_set','--algorithm','vdn_1ff',
                '--method','full_rank_preference','--seed',str(seed),'--protocol',PROTOCOL,
                '--run-id',ref,'--preference-role','pair-reference']
        item = task(ref,args,'training',[],'VDN-FT-reference')
        item['run_dir'] = str(ROOT / 'runs' / 'tc_coverage' / ref)
        evaluation = ref + '_independent_final'
        task(evaluation,['scripts/evaluate.py','--run',item['run_dir'],'--episodes','128'],
             'evaluation',[ref],'VDN-FT-reference')
        branches.append(evaluation)
        audit = ref + '_offline'
        audits.append(audit)
        task(audit,['scripts/offline_tc_coverage.py',str(ROOT/'runs'/'tc_coverage'/ref)],'analysis',[ref],'offline-masks')
    gate = BATCH + '_feasibility'
    task(gate,['scripts/offline_tc_coverage.py','--gate',str(out/'plan.json')], 'analysis',audits,'offline-feasibility')
    for seed, ref in zip(SEEDS,refs):
        for level in ('low','medium','high'):
            run = '%s_%s_seed%d' % (BATCH,level,seed)
            item = task(run,['scripts/run.py','--suite','vdn_action_set','--algorithm','vdn_1ff',
                             '--method','pair_tc_'+level,'--seed',str(seed),'--protocol',PROTOCOL,
                             '--run-id',run,'--preference-role','shared-branch','--parent',str(ROOT/'runs'/'tc_coverage'/ref)],
                        'training',[gate],level+'-TC')
            item['run_dir'] = str(ROOT/'runs'/'tc_coverage'/run)
            evaluation = run + '_independent_final'
            task(evaluation,['scripts/evaluate.py','--run',item['run_dir'],'--episodes','128'],
                 'evaluation',[run],level+'-TC')
            branches.append(evaluation)
    final = task(BATCH+'_summary',['scripts/summarize_tc_coverage.py',str(out/'plan.json')],
                 'summary',branches,'summary')
    final['dependency_mode'] = 'terminal'
    return dict(id=BATCH,tasks=tasks,reference_ids=refs,offline_ids=audits,
                protocol=PROTOCOL,algorithm='vdn_1ff',seeds=list(SEEDS))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    plan=build()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(plan,indent=2)+'\n')
    else:
        print(json.dumps(plan,indent=2))
