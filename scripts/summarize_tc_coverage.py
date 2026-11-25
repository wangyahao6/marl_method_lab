"""Export TS5 performance, paired comparisons and mechanism records from actual artifacts."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import sys

import numpy as np
import torch as th

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from learners.pair_tc import digest_masks


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def records(path):
    if path.exists():
        for line in path.read_text().splitlines():
            yield json.loads(line)


def write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)


def summarize(plan_path):
    plan=read(plan_path);out=plan_path.parent
    manifests=[];metrics=[];evaluation_rows=[];mechanisms=[];independent=[];configs={}
    refs={s:ROOT/'runs'/'tc_coverage'/r for s,r in zip(plan['seeds'],plan['reference_ids'])}
    for task in plan['tasks']:
        if task['type']!='training':continue
        run=Path(task['run_dir']);identity=dict(run_id=run.name,method=task['metadata']['branch_id'])
        config=read(run/'config.json');protocol=read(run/'protocol.json');launch=read(run/'launch.json')
        seed=config.get('seed',int(run.name.split('seed')[-1]));identity['seed']=seed
        reference=run.name in plan['reference_ids'];parent=None if reference else refs[seed]
        state=read(run/'status.json');retention=read(run/'retention.json');restore=read(run/'branch_restore.json')
        manifest=dict(identity,status=state.get('status','not_started'),commit=launch.get('git_commit'),
                      parent_run_id=parent.name if parent else None,run_dir=str(run),
                      completed_global_steps=state.get('t_env'),
                      trained_steps=(state.get('t_env',0)-(restore.get('t_env',0))) if state else None,
                      replay_on_switch=config.get('replay_on_switch'),epsilon_on_switch=config.get('epsilon_on_switch'),
                      parent_model_hash=restore.get('parent_model_hash'),restore_model_hash=restore.get('model_hash'))
        manifests.append(manifest)
        configs[run.name]=dict(config=config,protocol=protocol,launch_commit=launch.get('git_commit'),branch_restore=restore)
        own=list(records(run/'evaluations.jsonl'))
        evaluations=([r for r in records(parent/'evaluations.jsonl') if r['phase']==0 or r['label']=='initial_reference'] if parent else [])+own
        evaluation_rows += [dict(identity,**{k:r.get(k) for k in ('phase','task','label','t_env','wins','episodes')}) for r in own]
        order=[s['task'] for s in protocol.get('stages',[])]
        initial={r['task']:r['wins']/r['episodes'] for r in evaluations if r['label']=='initial_reference'}
        before={r['task']:r['wins']/r['episodes'] for r in evaluations if r['label']=='next_unseen'}
        previous={}
        for stage in retention.get('stages',[]):
            phase=stage['phase'];training=stage['task'];start=stage['start'];end=stage['t_env']
            rows=[r for r in evaluations if r['phase']==phase and r['label']=='stage_end']
            wr={r['task']:r['wins']/r['episodes'] for r in rows}
            if set(wr)!=set(order[:phase+1]):continue
            bwt=[wr[t]-previous[t] for t in previous if t!=training]
            previous[training]=wr[training]
            fwt=[before[t]-initial[t] for t in order[1:phase+1] if t in before and t in initial]
            curve=sorted([r for r in evaluations if r['phase']==phase and r['task']==training and r['label'] in ('phase_start','periodic','stage_end')],key=lambda r:r['t_env'])
            auc=None
            if curve and curve[0]['t_env']==start and curve[-1]['t_env']==end:
                auc=100*sum((a['wins']/a['episodes']+b['wins']/b['episodes'])*(b['t_env']-a['t_env'])/2 for a,b in zip(curve,curve[1:]))/(end-start)
            metrics.append(dict(identity,phase=phase,training_task=training,acc_pct=100*mean(wr.values()),
                                bwt_pp=100*mean(bwt) if bwt else None,fwt_pp=100*mean(fwt) if len(fwt)==phase and fwt else None,
                                learning_auc_pct=auc,**{t+'_win_pct':100*v for t,v in wr.items()}))
        accum=defaultdict(list)
        method=protocol.get('method','')
        for update in records(run/(method+'_updates.jsonl')):
            for hist in update.get('historical_tasks',[]):
                accum[(update['phase'],hist.get('source_phase'),hist.get('task'))].append(hist)
        for (phase,source,history),rows in accum.items():
            valid=sum(r.get('valid_pairs',0) for r in rows);top=sum(r.get('top1_states',0) for r in rows)
            losses=[r['loss'] for r in rows if r.get('loss') is not None]
            mechanisms.append(dict(identity,phase=phase,source_phase=source,historical_task=history,updates=len(rows),
               top1_flip_rate=sum(r.get('top1_flips',0) for r in rows)/top if top else None,
               pair_violation_rate=sum(r.get('pair_violations',0) for r in rows)/valid if valid else None,
               mean_preference_loss=mean(losses) if losses else None,valid_pairs=valid,top1_states=top))
        for path in (run/'evaluation').glob('*/results.json'):
            result=read(path)
            for row in result.get('tasks',[]):
                independent.append(dict(identity,task=row['task'],wins=row['wins'],episodes=row['episodes']))
        if not reference and state.get('status')=='completed':
            for phase in range(len(order)-1):
                mask_root=Path(protocol['tc_mask_root']) if protocol.get('tc_mask_root') else parent/'offline_tc'
                frozen=th.load(mask_root/protocol['method'].replace('pair_tc_','')/('phase%04d.th'%phase),map_location='cpu')
                archived=th.load(run/'pair_ablation_memory'/('phase%04d'%phase)/'selection.th',map_location='cpu')
                if digest_masks(archived)!=frozen['audit']['pair_mask_sha256']:
                    raise ValueError('training mask differs from offline mask: '+run.name)
    write(out/'runs.csv',manifests);write(out/'stage_metrics.csv',metrics);write(out/'evaluations.csv',evaluation_rows)
    write(out/'preference_diagnostics.csv',mechanisms);write(out/'independent_final.csv',independent)
    (out/'configs').mkdir(exist_ok=True)
    for name,data in configs.items():(out/'configs'/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    final=[r for r in metrics if r['phase']==4];aggregate=[];paired=[]
    for method in sorted({r['method'] for r in final}):
        rows=[r for r in final if r['method']==method]
        for metric in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct'):
            values=[r[metric] for r in rows if r[metric] is not None]
            aggregate.append(dict(method=method,metric=metric,n=len(values),mean=mean(values) if values else None,
                                  std=stdev(values) if len(values)>1 else None))
    for row in final:
        control=next((r for r in final if r['seed']==row['seed'] and r['method']=='VDN-FT-reference'),None)
        if control and row is not control:
            paired.append(dict(run_id=row['run_id'],control_run_id=control['run_id'],seed=row['seed'],method=row['method'],
                   **{metric+'_delta':row[metric]-control[metric] if row[metric] is not None and control[metric] is not None else None for metric in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct')}))
    write(out/'seed_summary.csv',aggregate);write(out/'paired_final.csv',paired)
    tc_rows=[];trends=[]
    offline=list(csv.DictReader((out/'offline_statistics.csv').open())) if (out/'offline_statistics.csv').exists() else []
    for result in metrics:
        if result['method']=='VDN-FT-reference' or result['phase']==0:continue
        level=result['method'].replace('-TC','')
        pools=[r for r in offline if int(r['seed'])==result['seed'] and r['level']==level and int(r['phase'])<result['phase']]
        if not pools:continue
        states=sum(int(r['states']) for r in pools);possible=sum(int(r['possible_pairs']) for r in pools)
        selected=sum(int(r['direct_pairs']) for r in pools)
        tc_rows.append(dict(result,micro_tc=sum(int(r['closure_pairs']) for r in pools)/possible,
                            macro_tc=sum(float(r['macro_tc'])*int(r['states']) for r in pools)/states,
                            qgap_mean=sum(float(r['qgap_mean'])*int(r['direct_pairs']) for r in pools)/selected))
    for seed in plan['seeds']:
        for phase in range(1,5):
            arms={r['method']:r for r in tc_rows if r['seed']==seed and r['phase']==phase and r['method'] in ('low-TC','medium-TC','high-TC')}
            if set(arms)!=set(('low-TC','medium-TC','high-TC')):continue
            for metric in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct'):
                values=[arms[level+'-TC'][metric] for level in ('low','medium','high')]
                if any(v is None for v in values):continue
                increasing=all(a<=b for a,b in zip(values,values[1:]));decreasing=all(a>=b for a,b in zip(values,values[1:]))
                trends.append(dict(seed=seed,phase=phase,metric=metric,low=values[0],medium=values[1],high=values[2],
                   high_minus_low=values[2]-values[0],monotonic='flat' if increasing and decreasing else 'increasing' if increasing else 'decreasing' if decreasing else 'nonmonotonic',
                   high_minus_low_qgap=arms['high-TC']['qgap_mean']-arms['low-TC']['qgap_mean']))
    write(out/'tc_performance.csv',tc_rows);write(out/'within_seed_trends.csv',trends)
    completed=sum(r['status']=='completed' for r in manifests)
    lines=['# TS5-1M Transitive Closure Coverage Ablation','',
           'S → Z → C → 2S3Z → 2Z3C, five 1M stages, seeds 2222/3535/4646. VDN single-FF64, FT replay, per-task epsilon, RMSprop. Three fresh full VDN-FT references and nine branches sharing their seed-specific S boundary and four fixed FT historical success pools.',
           '', 'Completed training runs: %d/12. Full terminal evidence requires completed training, five-stage greedy evaluation matrix, and independent final evaluations.'%completed,
           '', 'Offline selection, feasibility and controls: [OFFLINE_REPORT.md](OFFLINE_REPORT.md). Actual configs and source commits: configs/ and runs.csv.',
           'Performance: stage_metrics.csv (ACC percent, BWT/FWT percentage points, normalized per-stage current-task learning AUC percent), seed_summary.csv, paired_final.csv, evaluations.csv.',
           'Mechanisms: offline_statistics.csv, offline_seed_summary.csv and preference_diagnostics.csv. Offline counts are unique fixed states; training mechanism counts include repeated historical sampling. Top-1 flips use the existing unique-old-top1 decision denominator, which also includes decisions outside the >=4-action TC subset.',
           '', '| Method | Seed | ACC % | BWT pp | FWT pp | E AUC % |','|---|---:|---:|---:|---:|---:|']
    def fmt(x):return 'N/A' if x is None else '%.3f'%x
    lines += ['| %s | %s | %s | %s | %s | %s |'%(r['method'],r['seed'],fmt(r['acc_pct']),fmt(r['bwt_pp']),fmt(r['fwt_pp']),fmt(r['learning_auc_pct'])) for r in final]
    for metric in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct'):
        rows=[r for r in trends if r['phase']==4 and r['metric']==metric]
        if rows:
            lines += ['', '%s final-stage High−Low: %s; mean %.3f. Monotonic patterns: %s.'%(metric, ', '.join('seed%d %+.3f'%(r['seed'],r['high_minus_low']) for r in rows),mean(r['high_minus_low'] for r in rows), ', '.join('seed%d %s'%(r['seed'],r['monotonic']) for r in rows))]
    candidates=[r for r in aggregate if r['metric']=='acc_pct' and r['method']!='VDN-FT-reference' and r['n']==3]
    if candidates:
        best=max(candidates,key=lambda r:r['mean']);lines += ['', 'Highest observed three-seed final ACC: %s (%.3f%%). This identifies an observed arm, not a general optimal TC interval.'%(best['method'],best['mean'])]
    lines += ['', 'tc_performance.csv joins performance to the cumulative unique-state fixed-pool TC and Q-gap statistics available at each stage; these are not exposure-weighted training TC. within_seed_trends.csv retains per-stage paired trends and residual gap differences.', '', 'Interpretation limits: three training seeds are the independent replicates; states/pairs are not extra replicates. The arms preserve K1, edge budgets and per-state distance quotas, while graph topology and TC change together. TC-performance correlation alone does not isolate a causal role for the scalar TC score. Residual Q-gap differences remain possible competing explanations. The experiment contains no separately budgeted full-order training arm; if complete-order coverage is zero throughout, it cannot determine full versus partial ordering superiority. A best observed three-arm setting is not an estimated general optimum.',
              '', 'Shared-prefix AUC and stage-S metrics reuse the reference S trajectory once. References continue FT through all five stages; formal methods replay masks selected on FT reference states, including C/D reference states rather than each branch\'s own history. This is an intentional fixed-reference intervention.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(training_completed=completed,training_total=len(manifests),
               formal_training_complete=sum(r['status']=='completed' and r['method']!='VDN-FT-reference' for r in manifests),
               final_stage_rows=len(final),independent_evaluation_rows=len(independent)),indent=2)+'\n')
    print((out/'summary.json').read_text())


if __name__=='__main__':summarize(Path(sys.argv[1]))
