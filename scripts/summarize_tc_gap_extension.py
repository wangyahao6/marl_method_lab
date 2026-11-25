"""Summarize the two-seed, two-gap comparison without rerunning any completed TC branch."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import sys

from summarize_tc_coverage import summarize, write

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def records(path):return list(csv.DictReader(path.open())) if path.exists() else []


def run(plan_path):
    plan=read(plan_path);out=plan_path.parent;old=read(Path(plan['reused_original_plan']))
    source_stats=records(out/'offline_statistics.csv');original_stats=records(Path(plan['reused_original_plan']).parent/'offline_statistics.csv')
    all_runs=[];all_stages=[];all_diagnostics=[];all_final=[];new_completed=0
    configs={}
    for gap,tag in ((0,'gap0'),(.5,'gap05')):
        directory=out/tag;directory.mkdir(exist_ok=True)
        tasks=[]
        for task in old['tasks']:
            if task['type']!='training':continue
            seed=int(task['argv'][task['argv'].index('--seed')+1])
            if seed not in (2222,3535):continue
            reference=task['id'] in old['reference_ids']
            if reference or gap==0:tasks.append(task)
        new_tasks=[t for t in plan['tasks'] if t['type']=='training' and t['metadata']['min_gap']==gap]
        tasks+=new_tasks
        cohort=dict(id=plan['id']+'_'+tag,tasks=tasks,seeds=[2222,3535],
                    reference_ids=[r for r in old['reference_ids'] if r.endswith(('seed2222','seed3535'))])
        cohort_path=directory/'comparison_plan.json';cohort_path.write_text(json.dumps(cohort,indent=2)+'\n')
        stats=[dict(r,states=r['valid_states']) for r in source_stats if float(r['min_gap'])==gap]
        if gap==0:stats += [r for r in original_stats if int(r['seed']) in (2222,3535)]
        write(directory/'offline_statistics.csv',stats)
        summarize(cohort_path)
        for filename,target in (('runs.csv',all_runs),('stage_metrics.csv',all_stages),('preference_diagnostics.csv',all_diagnostics),('independent_final.csv',all_final)):
            rows=records(directory/filename)
            target += [dict(min_gap=gap,**r) for r in rows]
        configs.update({p.name:read(p) for p in (directory/'configs').glob('*.json')})
        # The generic summarizer provides data; replace its original three-seed prose.
        (directory/'REPORT.md').write_text('# '+tag+' comparison\n\nTwo training seeds, 2222/3535. Read ../REPORT.md for context.\n')
        new_ids={t['id'] for t in new_tasks}
        new_completed += sum(r['status']=='completed' and r['run_id'] in new_ids for r in records(directory/'runs.csv'))
    write(out/'comparison_runs.csv',all_runs);write(out/'comparison_stage_metrics.csv',all_stages)
    write(out/'comparison_preference_diagnostics.csv',all_diagnostics);write(out/'comparison_independent_final.csv',all_final)
    aggregate=[];paired=[];gapeffects=[]
    for gap in (0,.5):
        for method in ('VDN-FT-reference','k1-TC','low-TC','medium-TC','high-TC'):
            final=[r for r in all_stages if float(r['min_gap'])==gap and r['method']==method and int(r['phase'])==4]
            for metric in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct'):
                values=[float(r[metric]) for r in final if r[metric]!='']
                aggregate.append(dict(min_gap=gap,method=method,metric=metric,n=len(values),
                                      mean=mean(values) if values else None,std=stdev(values) if len(values)>1 else None))
        for seed in (2222,3535):
            for phase in range(5):
                rows=[r for r in all_stages if float(r['min_gap'])==gap and int(r['seed'])==seed and int(r['phase'])==phase]
                control=next((r for r in rows if r['method']=='k1-TC'),None)
                if control is None:continue
                for row in rows:
                    if row['method'] not in ('low-TC','medium-TC','high-TC'):continue
                    paired.append(dict(min_gap=gap,seed=seed,phase=phase,method=row['method'],run_id=row['run_id'],control_run_id=control['run_id'],
                         **{k+'_delta':float(row[k])-float(control[k]) if row[k] and control[k] else None for k in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct')}))
    for seed in (2222,3535):
        for method in ('k1-TC','low-TC','medium-TC','high-TC'):
            for phase in range(5):
                a=next((r for r in all_stages if float(r['min_gap'])==0 and int(r['seed'])==seed and r['method']==method and int(r['phase'])==phase),None)
                b=next((r for r in all_stages if float(r['min_gap'])==.5 and int(r['seed'])==seed and r['method']==method and int(r['phase'])==phase),None)
                if a is None or b is None:continue
                gapeffects.append(dict(seed=seed,method=method,phase=phase,gap0_run_id=a['run_id'],gap05_run_id=b['run_id'],
                    **{k+'_delta':float(b[k])-float(a[k]) if a[k] and b[k] else None for k in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct')}))
    write(out/'two_seed_summary.csv',aggregate);write(out/'paired_vs_k1.csv',paired);write(out/'paired_gap05_minus_gap0.csv',gapeffects)
    lines=['# TS5-1M TC: K1-only and min_gap=0.5 extension','',
       'Only training seeds 2222/3535 enter both main comparison cohorts. Seed 4646 remains in the original dataset. No reference retraining and no completed TC gap0 reruns. Original task sequence, 1M per stage, RMSprop/lr, auxiliary weight and evaluation seeds/counts are preserved.',
       '', 'New training completed: %d/10. References and six completed gap0 TC branches are reused. Shared-prefix S metrics are reused, not independent repeats.'%new_completed,
       '[Offline feasibility](OFFLINE_REPORT.md) and offline_statistics.csv specify common states, exact base masks, budgets, distance quotas and Q-gap statistics.',
       'Gap0 K1 has N-1 Top-1 pairs on exactly the old common valid states. Gap05 K1 and TC arms share threshold-filtered Top-1 pairs and the same quota-feasible state set. TC extras retain N-2 total and the original Near/Mid/Far per-state quotas; K1-only has zero extras.',
       'Normalization and filtering directly reuse reference_order/full_rank_pairs. Filtering applies to K1 as well as extras. Some original K1 edges therefore do not survive gap05. Threshold is never lowered and budgets/quotas are never shrunk.',
       '', '| Gap | Method | n | ACC % | BWT pp | FWT pp | E AUC % |','|---:|---|---:|---:|---:|---:|---:|']
    for gap in (0,.5):
        for method in ('VDN-FT-reference','k1-TC','low-TC','medium-TC','high-TC'):
            rows={r['metric']:r for r in aggregate if r['min_gap']==gap and r['method']==method}
            values=[rows[k]['mean'] for k in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct')]
            lines.append('| %s | %s | %d | %s |'%(gap,'K1-only' if method=='k1-TC' else method,rows['acc_pct']['n'],
                ' | '.join('N/A' if v is None else '%.3f'%v for v in values)))
    lines += ['', 'comparison_stage_metrics.csv saves all stages and per-task win rates. paired_vs_k1.csv compares each TC arm against same-seed K1 within each gap. paired_gap05_minus_gap0.csv records cross-gap contrasts, which combine filtering and a change of eligible-state population and do not isolate the threshold effect.',
        'Direct coverage, Micro/Macro TC and Closure Gain are in offline_statistics.csv; exact per-state TC frequencies are in tc_state_histogram.csv; raw per-state selected/base/extra gap and TC distributions are in masks/seed*/gap*/{arm}/phase*_distributions.npz on the experiment host.',
        'Actual configurations and restoration identities are in gap0/configs/ and gap05/configs/. Each terminal formal history mask is checked against its frozen offline hash by the shared summarizer. Incomplete/failed paths are retained without implicit retries or added budgets.',
        'Pair-violation and preference-loss diagnostics count repeated sampled historical episodes; Top-1 flip denominator follows the existing implementation. Fixed-pool TC is not sampling-weighted. Two training seeds cannot establish a general optimal TC interval or a causal effect of scalar TC independent of graph topology and Q-gap differences.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    summary=dict(new_training_completed=new_completed,new_training_total=10,seeds=[2222,3535],
                 reused_gap0_TC_branches=6,reused_reference_runs=2,formal_comparison_rows=len(all_stages))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':run(Path(sys.argv[1]))
