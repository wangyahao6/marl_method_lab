"""Export TS5-1M full-prefix CPR comparisons from actual run artifacts, without training."""
import argparse
from array import array
from collections import defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = tuple('ABCDE')
METHODS = ('VDN-ER', 'TKPR-K1', 'TKPR-CPR', 'TKPR-Refresh')
GRADIENTS = ('td_grad_norm_agent', 'preference_grad_norm_agent',
             'preference_td_grad_ratio', 'td_preference_grad_cosine')


def clean(value):
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '').replace(str((ROOT/'runs').resolve().parent) + '/', '')
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, dict):
        return {key: clean(x) for key, x in value.items()}
    return value


class Export:
    def __init__(self, plan, out):
        self.plan, self.out = plan, out
        self.sources = {}
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'configs').mkdir(exist_ok=True)

    def read(self, path):
        if not path.exists():
            return {}
        data = path.read_bytes()
        self.sources[str(path)] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        return json.loads(data)

    def lines(self, path):
        if not path.exists():
            return
        digest, n = hashlib.sha256(), 0
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                n += 1
                yield json.loads(line)
        self.sources[str(path)] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest(), lines=n)

    def csv(self, name, rows, fields):
        columns = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
        with (self.out / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    def json(self, name, data):
        (self.out / name).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def statistics(values):
    if not values:
        return dict(n=0, mean=None, median=None, std=None)
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std(ddof=1)) if len(a) > 1 else None)


def auc(rows, stage):
    # Same regular seed block at all points. Do not mix focus/stage-end evaluation seeds.
    points = sorted((r['t_env'], 100 * r['wins'] / r['episodes']) for r in rows
                    if r['phase'] == stage['phase'] and r['task'] == stage['task']
                    and r['label'] in ('phase_start', 'periodic'))
    if not points:
        return None, 0
    end, start = stage['t_env'], stage['start']
    if points[-1][0] < end:
        points.append((end, points[-1][1]))
    return float(np.trapz([p[1] for p in points], [p[0] for p in points]) / (end - start)), len(points) - 1


def export(plan_path, output):
    ex = Export(plan_path, output)
    plan = ex.read(plan_path)
    tasks = [t for t in plan['tasks'] if t['type'] == 'training']
    run_rows, evaluations, stages, curves = [], [], [], []
    preferences, cpr, gradients, memory, costs, independent = [], [], [], [], [], []
    executions, run_data = [], {}
    for task in tasks:
        run, meta = Path(task['run_dir']), task['metadata']
        rid, role, method, seed = task['id'], meta['role'], meta['method'], meta['seed']
        prefix_id = meta.get('prefix_id', rid)
        status, launch = ex.read(run / 'status.json'), ex.read(run / 'launch.json')
        launcher = ex.read(run / 'launcher_status.json')
        retention = ex.read(run / 'retention.json')
        protocol, cfg = ex.read(run / 'protocol.json'), ex.read(run / 'config.json')
        restored = ex.read(run / 'branch_restore.json')
        final_state_manifest = ex.read(run / 'final_state_manifest.json')
        parent = ROOT / 'runs' / prefix_id if role == 'shared-branch' else run
        prefix_manifest = ex.read(parent / 'prefix_state_manifest.json')
        checkpoint_sha = restored.get('prefix_checkpoint_sha256', prefix_manifest.get('checkpoint_sha256'))
        ident = dict(run_id=rid, prefix_id=prefix_id, method=method, seed=seed, role=role)
        complete = (status.get('status') == 'completed' and launcher.get('exit_code') == 0
                    and len(retention.get('stages', [])) == (1 if role == 'shared-prefix' else 5)
                    and status.get('t_env',0) >= (1000000 if role=='shared-prefix' else 5000000))
        executions.append(dict(**ident, status=status.get('status', 'pending'), complete=complete,
                               exit_code=launcher.get('exit_code'), error=status.get('error'),
                               restored_model_matches=restored.get('model_hash') == restored.get('parent_model_hash')
                               if restored else None,
                               initial_memory_matches=restored.get('parent_memory_input_hash') == restored.get('branch_memory_input_hash')
                               if restored else None))
        run_rows.append(dict(**ident, branch_id=rid if role == 'shared-branch' else '',
                             parent_run_id=prefix_id if role == 'shared-branch' else '',
                             commit=launch.get('git_commit'), prefix_checkpoint_sha256=checkpoint_sha,
                             epsilon_on_switch=protocol.get('epsilon_on_switch'),
                             replay_on_switch=protocol.get('replay_on_switch'), algorithm=protocol.get('algorithm'),
                             cumulative_steps=status.get('t_env'),
                             new_steps=status.get('t_env', 0) - (restored.get('t_env', 0) if restored else 0)
                             if status else None,
                             elapsed_seconds=status.get('elapsed_seconds'), complete=complete,
                             status=status.get('status', 'pending'), config='configs/' + rid + '.json'))
        ex.json('configs/' + rid + '.json', dict(launch_git_commit=launch.get('git_commit'),
                                                protocol=protocol, config=cfg, status=status,
                                                branch_restore=restored, prefix_state_manifest=prefix_manifest,
                                                final_state_manifest=final_state_manifest))
        ev = list(ex.lines(run / 'evaluations.jsonl'))
        for row in ev:
            outcomes = row.get('outcomes', [])
            evaluations.append(dict(**ident, **{k: row.get(k) for k in ('phase','task','label','t_env','wins','episodes','win_rate')},
                                    seeds=json.dumps(sorted({o['seed'] for o in outcomes})),
                                    count_check=(sum(o['won'] for o in outcomes) == row['wins']
                                                 and len(outcomes) == row['episodes'])))
            if row['label'] in ('phase_start', 'periodic'):
                curves.append(dict(**ident, **{k: row[k] for k in ('phase','task','label','t_env','wins','episodes','win_rate')}))
        previous, stage_rows = {}, []
        for s in retention.get('stages', []):
            names = list(s['win_rates'])
            previous[s['task']] = s['win_rates'][s['task']]
            if role == 'shared-branch' and s['phase'] == 0:
                continue  # Parent A endpoints are referenced, never recorded as a new evaluation.
            a = 100 * np.mean(list(s['win_rates'].values()))
            old = [s['win_rates'][name] - previous[name] for name in names if name != s['task']]
            before = retention.get('before_first_training', {})
            initial = retention.get('initial_reference', {})
            fw = [before[name] - initial[name] for name in names if name in before and name in initial]
            b, f = 100 * np.mean(old) if old else None, 100 * np.mean(fw) if fw else None
            area, points = auc(ev, s)
            check = all((v is None and s.get(k) is None) or (v is not None and s.get(k) is not None
                        and math.isclose(v, s[k], abs_tol=1e-8))
                        for k,v in [('acc_percent',a),('bwt_pp',b),('fwt_pp',f)])
            sr = dict(**ident, phase=s['phase'], task=s['task'], t_env=s['t_env'], start=s['start'],
                      acc_pct=float(a), bwt_pp=float(b) if b is not None else None,
                      fwt_pp=float(f) if f is not None else None, learning_auc_pct=area,
                      regular_auc_points=points, auc_endpoint='last_regular_carried_to_stage_end',
                      metrics_match_retention=check,
                      **{name+'_win_pct':100*s['win_rates'][name] if name in s['win_rates'] else None for name in TASKS})
            stages.append(sr);stage_rows.append(sr)
        run_data[rid] = dict(ident=ident, complete=complete, stages=stage_rows, retention=retention, ev=ev)
        for event in ex.lines(run / 'tkpr_cpr_updates.jsonl'):
            count = event['decision_states']
            row = dict(**ident, **{k:v for k,v in event.items()
                                  if k not in ('old_reference_confidence','new_reference_confidence','reference_source_counts')})
            row['reference_source_counts'] = json.dumps(event.get('reference_source_counts', {}), sort_keys=True)
            for k in ('updated_states','retained_states','uncertain_states'):
                row[k.replace('_states','_fraction')] = event[k]/count if count else None
            for kind in ('old','new'):
                stats = event.get(kind+'_reference_confidence') or {}
                for k in ('n','nonfinite','mean','median','p10','p90','error'):
                    row[kind+'_confidence_'+k] = stats.get(k)
            cpr.append(row)
        for event in ex.lines(run / 'memory_collection.jsonl'):
            memory.append(dict(**ident, record_type='collection',
                               **{k:v for k,v in event.items() if k != 'blocks'}, blocks=json.dumps(event['blocks'])))
        for event in ex.lines(run / 'memory_inventory.jsonl'):
            for name,size in event['pools'].items():
                memory.append(dict(**ident,record_type='inventory',phase=event['phase'],task=name,
                                   trained_task=event['task'],t_env=event['t_env'],pool_size=size))
        phase_execution = {row['phase']:row for row in ex.lines(run / 'phase_execution.jsonl')}
        pref_name = {'TKPR-K1':'topk_preference','TKPR-Refresh':'tkpr_refresh',
                     'TKPR-CPR':'tkpr_cpr_preference'}.get(method)
        hist = {}
        cost = defaultdict(lambda:defaultdict(lambda:array('d')))
        for event in ex.lines(run / (pref_name+'_updates.jsonl')) if pref_name else ():
            phase = event['phase']
            hs = event['historical_tasks']
            active = sum(h['valid_states'] > 0 for h in hs)
            for key in ('historical_episode_count','preference_loss','td_loss','preference_participation_rate',
                        'preference_participated_update', 'preference_prepare_seconds',
                        'learner_update_seconds','preference_forward_cuda_seconds'):
                if finite(event.get(key)):
                    cost[phase][key].append(event[key])
            for h in hs:
                key = (phase,h['task'])
                g = hist.setdefault(key, dict(updates=0,sampled_episodes=0,valid_states=0,decision_states=0,
                                              valid_pairs=0,raw_decision_states=0,reference_mask_valid_states=0,
                                              values=defaultdict(lambda:array('d'))))
                g['updates'] += 1
                for k in ('sampled_episodes','valid_states','decision_states','valid_pairs',
                          'raw_decision_states','reference_mask_valid_states'):
                    g[k] += h.get(k, 0)
                for k in ('loss','coverage','pair_violation_rate','top1_flip_rate','sampled_episodes',
                          'reference_mask_coverage','effective_coverage_of_raw_decisions'):
                    if finite(h.get(k)):
                        g['values'][k].append(h[k])
                weight = 1/active if h['valid_states'] and active else 0
                g['values']['effective_task_weight'].append(weight)
                g['values']['weighted_loss_contribution'].append(weight * (h.get('loss') or 0))
                g['values']['no_valid_constraint'].append(int(h['valid_states']==0))
        for (phase,name), g in hist.items():
            row = dict(**ident,phase=phase,historical_task=name,logged_updates=g['updates'])
            for k in ('sampled_episodes','valid_states','decision_states','valid_pairs',
                      'raw_decision_states','reference_mask_valid_states'):
                row[k]=g[k] if k not in ('raw_decision_states','reference_mask_valid_states') or method=='TKPR-CPR' else None
            row['constraint_coverage'] = g['valid_states']/g['decision_states'] if g['decision_states'] else None
            row['effective_coverage_of_raw_decisions'] = g['valid_states']/g['raw_decision_states'] if g['raw_decision_states'] else None
            for k,values in g['values'].items():
                row.update({k+'_'+s:v for s,v in statistics(values).items()})
            preferences.append(row)
        for phase, execution in phase_execution.items():
            row = dict(**ident,phase=phase,task=execution['task'],actual_steps=execution['actual_steps'],
                       train_updates=execution['train_updates'],train_episodes=execution['train_episodes'],
                       phase_elapsed_seconds=execution['elapsed_seconds'],
                       replay_task_episodes=json.dumps(execution['replay_task_episodes'],sort_keys=True))
            for key,values in cost[phase].items():
                row.update({key+'_'+s:v for s,v in statistics(values).items()})
                row[key+'_total']=float(sum(values))
            if not cost[phase]:
                row['historical_episode_count_total']=0
                row['auxiliary_update_records']=0
            else:
                row['auxiliary_update_records']=len(cost[phase]['historical_episode_count'])
                row['preference_participation_rate_in_phase'] = statistics(
                    cost[phase]['preference_participated_update'])['mean']
            costs.append(row)
        gd = defaultdict(lambda:dict(total=0,values=array('d')))
        for event in ex.lines(run / 'experiment_metrics.jsonl'):
            if event.get('context',{}).get('event')!='gradient_monitor' or event.get('metric') not in GRADIENTS:
                continue
            entry = gd[event['phase'],event['metric']];entry['total']+=1
            if finite(event.get('value')):
                entry['values'].append(event['value'])
        for (phase,metric),data in gd.items():
            gradients.append(dict(**ident,phase=phase,metric=metric,logged_points=data['total'],
                                  missing_points=data['total']-len(data['values']),**statistics(data['values'])))
        if role=='shared-branch':
            for path in sorted((run/'evaluation').glob('*/results.json'), reverse=True) if (run/'evaluation').exists() else ():
                result=ex.read(path)
                final=retention.get('stages',[{}])[-1]
                evaluation_status = ex.read(path.parent/'launcher_status.json')
                if (result.get('checkpoint')!=final.get('checkpoint')
                        or evaluation_status.get('exit_code') != 0
                        or len(result.get('tasks',[])) != 5
                        or any(row['episodes'] != 128 for row in result['tasks'])):
                    continue
                for row in result.get('tasks',[]):
                    reference = next((e for e in ev if e['label']=='stage_end' and e['phase']==4 and e['task']==row['task']),{})
                    old={(x['seed'],x['index']) for x in reference.get('outcomes',[])}
                    new={(x['seed'],x['index']) for x in row.get('outcomes',[])}
                    independent.append(dict(**ident,phase=4,task=row['task'],t_env=result.get('t_env'),
                                            wins=row['wins'],episodes=row['episodes'],win_rate=row['win_rate'],
                                            checkpoint=result['checkpoint'],evaluation_path=str(path.parent.relative_to(ROOT)),
                                            seeds=json.dumps(sorted({x[0] for x in new})),
                                            repeated_stage_end_seed_index_set=(old==new),
                                            evaluation_kind='independent_process_recheck' if old & new else 'independent_process_distinct_seeds'))
                break
        for path in run.glob('models/*/agent.th'):
            if str(path) not in ex.sources:
                ex.sources[str(path)]=dict(bytes=path.stat().st_size,sha256=file_sha(path))
        for filename in ('prefix_replay.th','final_replay.th'):
            path=run/filename
            if path.exists():
                ex.sources[str(path)]=dict(bytes=path.stat().st_size,
                                         sha256=(final_state_manifest or prefix_manifest).get('files',{}).get(filename),
                                         retained_on_experiment_host=True)
    complete_ids = {r['run_id'] for r in executions if r['complete']}
    final_rows=[r for r in stages if r['role']=='shared-branch' and r['phase']==4 and r['run_id'] in complete_ids]
    paired=[]
    for r in final_rows:
        for control in ('TKPR-K1','TKPR-Refresh','VDN-ER'):
            match=next((x for x in final_rows if x['seed']==r['seed'] and x['method']==control),None)
            if not match or r['method']==control:
                continue
            row=dict(run_id=r['run_id'],control_run_id=match['run_id'],prefix_id=r['prefix_id'],seed=r['seed'],
                     method=r['method'],control=control,same_prefix=r['prefix_id']==match['prefix_id'])
            for key in ('acc_pct','bwt_pp','fwt_pp',*[name+'_win_pct' for name in TASKS]):
                row[key+'_difference']=r[key]-match[key]
            paired.append(row)
    ex.csv('runs.csv',run_rows,['run_id','prefix_id','method','seed','role','commit'])
    ex.csv('evaluations.csv',evaluations,['run_id','prefix_id','method','seed','role','phase','task','label','t_env','wins','episodes','win_rate'])
    ex.csv('stage_metrics.csv',stages,['run_id','prefix_id','method','seed','phase','task','t_env','acc_pct','bwt_pp','fwt_pp','learning_auc_pct'])
    ex.csv('learning_curves.csv',curves,['run_id','prefix_id','method','seed','phase','task','label','t_env','wins','episodes','win_rate'])
    ex.csv('paired_final.csv',paired,['run_id','control_run_id','prefix_id','seed','method','control','acc_pct_difference','bwt_pp_difference','fwt_pp_difference'])
    ex.csv('preference_diagnostics.csv',preferences,['run_id','prefix_id','method','seed','phase','historical_task','logged_updates','valid_states','decision_states','valid_pairs','constraint_coverage'])
    ex.csv('cpr_reference_updates.csv',cpr,['run_id','prefix_id','method','seed','phase','task','current_win_rate','historical_best_win_rate','task_gate_passed','decision_states','updated_states','retained_states','uncertain_states'])
    ex.csv('gradient_diagnostics.csv',gradients,['run_id','prefix_id','method','seed','phase','metric','logged_points','missing_points','n','mean','median','std'])
    ex.csv('memory_collection.csv',memory,['run_id','prefix_id','method','seed','phase','task','record_type','successful','failed','pool_size'])
    ex.csv('sampling_cost.csv',costs,['run_id','prefix_id','method','seed','phase','actual_steps','train_updates','phase_elapsed_seconds','auxiliary_update_records'])
    ex.csv('independent_final.csv',independent,['run_id','prefix_id','method','seed','phase','task','wins','episodes','win_rate','evaluation_kind'])
    summaries=[]
    for method in METHODS:
        rs=[r for r in final_rows if r['method']==method]
        row=dict(method=method,completed_seeds=json.dumps(sorted(r['seed'] for r in rs)),n=len(rs))
        for k in ('acc_pct','bwt_pp','fwt_pp',*[t+'_win_pct' for t in TASKS]):
            row.update({k+'_'+s:v for s,v in statistics([r[k] for r in rs if finite(r[k])]).items()})
        summaries.append(row)
    ex.csv('final_summary.csv',summaries,['method','completed_seeds','n','acc_pct_mean','acc_pct_std','bwt_pp_mean','bwt_pp_std','fwt_pp_mean','fwt_pp_std'])
    all_complete=(len(executions)==10 and all(x['complete'] for x in executions) and len(independent)==40
                  and all(x['episodes']==128 for x in independent))
    ex.json('execution_summary.json',dict(all_complete=all_complete,training_runs=executions,
                                          expected_prefixes=2,expected_branches=8,expected_independent_task_points=40,
                                          actual_independent_task_points=len(independent),
                                          metric_mismatches=[r for r in stages if not r['metrics_match_retention']],
                                          evaluation_count_mismatches=[r for r in evaluations if not r['count_check']]))
    ex.json('source_manifest.json',dict(plan=str(plan_path),exported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                      sources=ex.sources,run_ids=[r['run_id'] for r in run_rows],
                                      prefix_checkpoint_sha256={r['prefix_id']:r['prefix_checkpoint_sha256'] for r in run_rows},
                                      expected_nominal_new_training_steps=34000000,
                                      independent_evaluation_seed_policy='same focus seeds: independent-process recheck, not independent test set'))
    render(ex,run_rows,stages,curves,paired,preferences,cpr,gradients,independent,summaries,all_complete)
    if (all_complete and output.name=='ts5-tkpr-cpr-1m'
            and output.parent.name=='comparisons' and output.parents[1].name=='results'):
        register_completed_result(output.parents[3])
    print(json.dumps(dict(all_complete=all_complete,runs=len(executions),stages=len(stages),
                          independent_task_points=len(independent)),indent=2))


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def register_completed_result(repository):
    """The results catalog contains completed datasets; pending reports remain unlisted."""
    path=repository/'docs/results/catalog.json'
    catalog=json.loads(path.read_text())
    prefix='comparisons/ts5-tkpr-cpr-1m/'
    entry=dict(id='ts5-tkpr-cpr-1m',category='comparisons',report=prefix+'README.md',
               files={name:prefix+name+'.csv' for name in (
                   'runs','evaluations','stage_metrics','learning_curves','paired_final',
                   'preference_diagnostics','cpr_reference_updates','gradient_diagnostics',
                   'memory_collection','sampling_cost','independent_final','final_summary')},
               task_order=list(TASKS),training_seeds=[3535,2222],final_phase=4,
               comparison=dict(kind='shared_prefix_matched_by_seed',match_on=['seed','prefix_id'],
                               primary_control=dict(method='TKPR-K1',role='shared-branch'),
                               additional_control=dict(method='VDN-ER',role='shared-branch')),
               interpretation_limits=[
                   '两个训练种子、单TS5顺序；共同A仅计一次。B之后各方法独立训练。',
                   '最终独立进程复核复用focus评价种子，不是独立测试集。',
                   'K1/CPR总8条，Refresh每任务8条，未单独消融Refresh采样和刷新。',
                   '无本批FT分支，不把旧2M FT作为同前缀配对对照。',
                   '边界c的参考更新影响下一阶段；不能从局部梯度/掩码相关变化推断性能因果。'])
    catalog['experiments']=[x for x in catalog['experiments'] if x['id']!=entry['id']]+[entry]
    path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    path=repository/'docs/results/README.md';text=path.read_text()
    if prefix+'README.md' not in text:
        anchor=text.index('| --- | --- | --- |',text.index('## 方法与基线对照'))
        offset=text.index('\n',anchor)+1
        line='| [TS5-1M · TKPR-CPR共同前缀对照]('+prefix+'README.md) | 单前馈VDN64，seed3535/2222，A共用1M、B–E各1M | ER、K1、CPR、Refresh的保持矩阵、学习AUC、参考更新及梯度诊断。 |\n'
        path.write_text(text[:offset]+line+text[offset:])


def render(ex,runs,stages,curves,paired,pref,cpr,gradients,independent,finals,complete):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(METHODS,('tab:blue','tab:orange','tab:green','tab:red')))
    fig,axs=plt.subplots(5,1,figsize=(10,13),constrained_layout=True)
    for phase,ax in enumerate(axs):
        selected=('Shared A',) if phase==0 else METHODS
        for method in selected:
            rs=[r for r in runs if r['method']==method]
            values=[]
            for run in rs:
                ps=sorted((e['t_env'],100*e['win_rate']) for e in curves if e['run_id']==run['run_id'] and e['phase']==phase)
                sr=next((s for s in stages if s['run_id']==run['run_id'] and s['phase']==phase),None)
                if ps and sr:
                    x=np.linspace(0,1,101);xp=[(t-sr['start'])/(sr['t_env']-sr['start']) for t,_ in ps]
                    values.append(np.interp(x,xp,[v for _,v in ps]))
            if values:
                mean=np.mean(values,axis=0);sd=np.std(values,axis=0,ddof=1) if len(values)>1 else np.zeros_like(mean)
                color=colors.get(method,'black');ax.plot(x,mean,label=f'{method} (n={len(values)})',color=color)
                ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.13)
        ax.set(title=f'Task {TASKS[phase]}: current-task regular evaluation',ylabel='Win rate (%)',ylim=(0,100))
        if ax.lines:ax.legend(fontsize=8,ncol=2)
    axs[-1].set_xlabel('Fraction of actual stage training steps')
    fig.savefig(ex.out/'learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    x=np.arange(5)
    for i,m in enumerate(METHODS):
        f=next(r for r in finals if r['method']==m)
        means=[f.get(t+'_win_pct_mean') for t in TASKS];sd=[f.get(t+'_win_pct_std') or 0 for t in TASKS]
        if all(v is not None for v in means):ax.bar(x+(i-1.5)*.19,means,.19,yerr=sd,label=m,color=colors[m],capsize=2)
    ax.set(xticks=x,xticklabels=TASKS,ylabel='Final win rate (%)',ylim=(0,100),title='Final task retention: seed mean +/- sample SD')
    if ax.patches:ax.legend(fontsize=8)
    fig.savefig(ex.out/'final_retention.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for key,label in (('updated_fraction','Accepted reference update fraction'),('valid_preference_coverage','Valid reference-mask coverage')):
        ys=[]
        for phase in range(1,5):
            rs=[r for r in cpr if r['phase']==phase and finite(r.get(key))]
            ys.append(np.mean([r[key] for r in rs]) if rs else np.nan)
        axs[0].plot(list('BCDE'),ys,'-o',label=label)
    axs[0].set(title='CPR boundary diagnostics (task-point mean)',ylim=(0,1));axs[0].legend(fontsize=7)
    for method in METHODS[1:]:
        ys=[]
        for phase in range(1,5):
            rs=[r for r in pref if r['phase']==phase and r['method']==method]
            numerator=sum(r['valid_states'] for r in rs)
            denominator=sum((r.get('raw_decision_states') or r['decision_states']) for r in rs)
            ys.append(numerator/denominator if denominator else np.nan)
        axs[1].plot(list('BCDE'),ys,'-o',label=method,color=colors[method])
    axs[1].set(title='Sampled constraints / pre-mask decisions',ylim=(0,1));axs[1].legend(fontsize=7)
    for method in METHODS[1:]:
        ys=[]
        for phase in range(1,5):
            rs=[r for r in gradients if r['phase']==phase and r['method']==method and r['metric']=='td_preference_grad_cosine' and r['mean'] is not None]
            ys.append(np.mean([r['mean'] for r in rs]) if rs else np.nan)
        axs[2].plot(list('BCDE'),ys,'-o',label=method,color=colors[method])
    axs[2].set(title='TD / auxiliary gradient cosine',ylim=(-1,1));axs[2].legend(fontsize=7)
    fig.savefig(ex.out/'cpr_mechanisms.png',dpi=150);plt.close(fig)
    lines=['# TS5-1M · TKPR-CPR 共同前缀四方法对照','',
           '**状态：'+('全部训练与独立进程复核完成。' if complete else '等待或部分完成；不是最终实验结论。详见 execution_summary.json。')+'**','',
           '分组：VDN-ER、TKPR-K1、TKPR-CPR、TKPR-Refresh；训练种子3535、2222。统一 vdn_action_set + vdn_1ff，TS5 A→B→C→D→E，每阶段名义1M，累计阈值后的整回合末切换，逐任务 ε 退火。每种子仅训练一次A前缀，四分支分别新增B–E共4M；两前缀加八分支名义新增34M。没有新增FT组；旧2M FT数据不作为本批同前缀配对对照。','',
           '## 前缀与初始化','',
           '共同A用K1入口训练；A没有历史辅助约束。full_shared_prefix 模式保存online/target、optimizer、完整A ReplayBuffer、训练与探索计数、Python/NumPy/Torch/CUDA RNG、环境分布状态、偏好采样RNG及成功Memory。原共享入口未保存ReplayBuffer且不支持ER分支，此模式显式补齐，原模式不变。ER继承A回放并按原均衡规则进入B；K1/CPR/Refresh恢复后按既有FT规则在B清空TD池。每分支单独保存最终ReplayBuffer及各阶段模型，父运行不被写入。','',
           'A成功池仅采集一次，所有分支的obs/actions/old_q_values指纹相同。CPR添加original Q副本、全true有效掩码及来源0；Refresh仅补轨迹来源0元数据，不重新采集A。初始化没有重算或更改共同Q。prefix_id和checkpoint SHA见runs.csv/branch_restore及source_manifest。A评价只在prefix数据记录一次；分支retention中的A条目是父结果引用，不算新评价。','',
           '## 评价与聚合','',
           '周期每25k训练环境步、32局；阶段末已见任务各128局，保留wins/episodes及种子。ACC按已见任务等权；BWT相对各任务最近学完时表现，FWT相对共同初始模型的首次训练前表现。当前任务学习AUC只使用phase_start与periodic的固定regular种子块，梯形积分，末次periodic到实际阶段末保持末值，除以实际阶段步数，不混入focus评价。阶段指标附数值对照结果。','',
           '最终模型通过scripts/evaluate.py在独立进程逐任务128局复核，沿用focus评价种子；因此是独立进程复核，不称独立测试集。independent_final.csv记录实际seed/index集合是否与stage_end重复。两训练种子等权汇总，样本SD(ddof=1)，同种子差值共享同一prefix；两个种子不提供稳健显著性证据。','',
           '## 最终表现','', '| 方法 | 完成种子n | ACC均值±SD (%) | BWT均值±SD (pp) | FWT均值±SD (pp) |', '| --- | ---: | ---: | ---: | ---: |']
    def show(mean,sd):return 'N/A' if mean is None else f'{mean:.2f} ± {sd:.2f}' if sd is not None else f'{mean:.2f} (n=1)'
    for r in finals:
        lines.append('| {} | {} | {} | {} | {} |'.format(r['method'],r['n'],*[show(r.get(k+'_mean'),r.get(k+'_std')) for k in ('acc_pct','bwt_pp','fwt_pp')]))
    lines+=['','![训练曲线](learning_curves.png)','','![最终任务保持](final_retention.png)','','## 逐任务结果与负面差异','','| 方法 | A | B | C | D | E |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in finals:
        lines.append('| '+r['method']+' | '+' | '.join(show(r.get(t+'_win_pct_mean'),r.get(t+'_win_pct_std')) for t in TASKS)+' |')
    lines+=['','CPR同种子配对结果（方法减对照；缺失不补零）：','','| seed | 对照 | ΔACC | ΔBWT | ΔFWT |','| --- | --- | ---: | ---: | ---: |']
    for r in paired:
        if r['method']=='TKPR-CPR':lines.append(f"| {r['seed']} | {r['control']} | {r['acc_pct_difference']:+.2f} | {r['bwt_pp_difference']:+.2f} | {r['fwt_pp_difference']:+.2f} |")
    lines+=['','## CPR机制与实际约束','','![CPR机制](cpr_mechanisms.png)','','状态级有效参考覆盖是完整成功池中参考有效掩码占真实决策的比例；原训练日志coverage的分母已叠加CPR掩码，表示通过min_gap的约束/已接纳决策。新增raw_decision_states使实际约束/掩码前决策也能单独统计，图中使用后者。不能把掩码覆盖率和Pair筛选后的覆盖率混用。CPR新旧置信度在本阶段参考选择前的同一完整状态集合汇总均值、中位数、P10、P90；门控失败不执行新MAC前向，新置信度为空，旧置信度仍可从存储Q统计。诊断不改变筛选规则。','',
              '边界c末更新的参考用于下一阶段c+1训练；E末参考更新不影响已完成的E阶段表现。分析更新比例与保持效果必须按此时序关联，不可用E末更新解释E末胜率。采样预算K1/CPR跨历史任务总8条，Refresh每任务8条；偏好损失仍是任务内有效状态均值再跨有效任务等权，贡献列为实际任务权重×任务loss，不按Pair数量加权。','']
    if cpr:
        passed=sum(r['task_gate_passed'] for r in cpr);count=sum(r['decision_states'] for r in cpr)
        updated=sum(r['updated_states'] for r in cpr);unknown=sum(r['uncertain_states'] for r in cpr)
        lines.append(f'当前已记录{len(cpr)}个历史任务边界点，门控通过{passed}个；更新状态{updated}，不确定状态{unknown}，合计决策{count}。这些计数是阶段/任务状态点，不能当作独立训练重复。')
    else:lines.append('CPR边界数据尚未产生，不预填门控、置信度或覆盖统计。')
    if complete:
        contrasts=[r for r in paired if r['method']=='TKPR-CPR' and r['control']=='TKPR-K1']
        if contrasts:
            lines.append('CPR相对K1的两个种子ΔACC为'+', '.join(f"seed{r['seed']} {r['acc_pct_difference']:+.2f}pp" for r in contrasts)+'。应同时查看各任务差值、阶段AUC和BWT；即使平均ACC上升，也不抹去个别任务/种子的下降。')
        lines.append('参考更新、掩码、覆盖及梯度记录可以描述与性能变化的对应关系；本实验没有移除门控/状态筛选的组件消融，也没有将Refresh采样预算与刷新单独控制，不能据相关变化认定某项机制是原因。')
    lines+=['','## 日志来源与开销口径','','原始逐更新偏好JSONL保存在各实验主机run目录，记录每次任务、episode_ids、采样数量、loss、valid/decision states、pairs、违反率、翻转率和参与率；公开preference_diagnostics.csv按运行/阶段/历史任务精确聚合，保留计数和分母，完整原始日志路径及校验和见source_manifest。gradient_monitor_interval=100：从experiment_metrics的gradient_monitor事件提取实际Agent TD/加权偏好梯度范数、比值和余弦；null按缺失计数，不将baseline无偏好梯度填成零。','',
              'sampling_cost.csv记录实际阶段步数、更新/回合数、回放库存、偏好prepare/update计时的均值/中位数/SD与和。prepare与learner_update有包含关系，GPU异步会影响CPU计时；不能相加当作额外开销。CPR重评与Refresh环境采集另见cpr_reference_updates、memory_collection；run总耗时来自status.elapsed_seconds，阶段耗时包含评价、记忆操作及快照I/O，不是纯训练GPU时间。成功/失败数量指显式记忆采集；baseline不额外采集旧任务成功池，相关值留空而非伪造。','',
              '每方法两个种子、单一顺序。共同A不是每分支的独立重复；保留负面结果与缺失。若任一训练失败，独立分支继续，不自动补种子或预算，最终汇总保留失败状态。','',
              '## 数据与复现','','runs.csv、evaluations.csv、stage_metrics.csv、learning_curves.csv、paired_final.csv、preference_diagnostics.csv、cpr_reference_updates.csv、gradient_diagnostics.csv、memory_collection.csv、sampling_cost.csv、independent_final.csv、final_summary.csv、source_manifest.json、execution_summary.json。空值表示不适用或缺失。模型、完整ReplayBuffer、逐条轨迹与原始日志保留实验主机。','',
              '复算：`python scripts/summarize_ts5_cpr_1m.py --plan results/ts5_tkpr_cpr_1m_20261124/plan.json --output docs/results/comparisons/ts5-tkpr-cpr-1m`。完整训练/评价命令与依赖见plan.json；源配置为configs/continual/action_preference/ts5/cpr_shared_1m/。','']
    (ex.out/'README.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    export(args.plan.resolve(),args.output.resolve())


if __name__=='__main__':main()
