"""Offline terminal export for budget-only K1, paired with completed native gap0.5 runs."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean,stdev
import subprocess

ROOT=Path(__file__).resolve().parents[1]
METRICS=('acc_pct','bwt_pp','fwt_pp','learning_auc_pct','A_win_pct','B_win_pct','C_win_pct','D_win_pct','E_win_pct')
GRADS=('td_grad_norm_agent','preference_grad_norm_agent','preference_td_grad_ratio','td_preference_grad_cosine')


def read(p):return json.loads(p.read_text()) if p.exists() else {}
def records(p):
    if p.exists():
        with p.open() as f:
            for line in f:yield json.loads(line)
def csvread(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def stats(values):
    return dict(n=len(values),mean=mean(values) if values else None,std=stdev(values) if len(values)>1 else None,
                min=min(values) if values else None,max=max(values) if values else None)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def report(plan_path):
    plan=read(plan_path);own=plan.get('own_policy_history',False)
    target=ROOT/'docs/results/parameter-ablations'/('ts5-tkpr-own-policy-refresh' if own else 'ts5-tkpr-k1-per-task8')
    target.mkdir(parents=True,exist_ok=True);(target/'configs').mkdir(exist_ok=True)
    runs=[];stages=[];curves=[];evals=[];sampling=[];prefs=[];gradients=[];independent=[];memory=[];memory_updates=[];sources={}
    for task in plan['tasks']:
        if task['type']!='training':continue
        run=Path(task['run_dir']);seed=task['metadata']['seed'];parent=Path(task['argv'][task['argv'].index('--parent')+1])
        status=read(run/'status.json');restore=read(run/'branch_restore.json');launch=read(run/'launch.json')
        method=task['metadata'].get('method','TKPR-K1-PerTask8')
        method_key=task['argv'][task['argv'].index('--method')+1]
        ident=dict(run_id=run.name,seed=seed,method=method,parent_run_id=parent.name,min_gap=.5)
        runs.append(dict(ident,status=status.get('status','pending'),commit=launch.get('git_commit'),
                         new_training_steps=status.get('t_env',0)-restore.get('t_env',0),elapsed_seconds=status.get('elapsed_seconds'),
                         config='configs/'+run.name+'.json'))
        config=read(run/'config.json');protocol=read(run/'protocol.json')
        (target/'configs'/(run.name+'.json')).write_text(json.dumps(dict(config=config,protocol=protocol,branch_restore=restore,launch=launch),indent=2)+'\n')
        own=list(records(run/'evaluations.jsonl'));parent_rows=[r for r in records(parent/'evaluations.jsonl') if r['phase']==0]
        all_rows=parent_rows+own;ret=read(run/'retention.json')
        for r in own:evals.append(dict(ident,**{k:r[k] for k in ('phase','task','label','t_env','wins','episodes','win_rate')}))
        initial={r['task']:r['wins']/r['episodes'] for r in all_rows if r['label']=='initial_reference'}
        before={r['task']:r['wins']/r['episodes'] for r in all_rows if r['label']=='next_unseen'}
        previous={}
        for s in ret.get('stages',[]):
            phase=s['phase'];task_name=s['task'];row={r['task']:r['wins']/r['episodes'] for r in all_rows if r['label']=='stage_end' and r['phase']==phase}
            if not row:continue
            bw=[v-previous[k] for k,v in row.items() if k in previous and k!=task_name];previous[task_name]=row[task_name]
            fw=[before[k]-initial[k] for k in row if k in before and k in initial]
            points=sorted([r for r in all_rows if r['phase']==phase and r['task']==task_name and r['label'] in ('phase_start','periodic','stage_end')],key=lambda r:r['t_env'])
            area=100*sum((a['wins']/a['episodes']+b['wins']/b['episodes'])*(b['t_env']-a['t_env'])/2 for a,b in zip(points,points[1:]))/(s['t_env']-s['start']) if points else None
            stages.append(dict(ident,phase=phase,training_task=task_name,start=s['start'],t_env=s['t_env'],shared_prefix=phase==0,
                acc_pct=100*mean(row.values()),bwt_pp=100*mean(bw) if bw else None,fwt_pp=100*mean(fw) if fw else None,
                learning_auc_pct=area,**{k+'_win_pct':v*100 for k,v in row.items()}))
            curves.extend(dict(ident,phase=phase,task=task_name,t_env=r['t_env'],stage_steps=r['t_env']-s['start'],label=r['label'],wins=r['wins'],episodes=r['episodes'],win_pct=100*r['wins']/r['episodes'],shared_prefix=phase==0) for r in points)
        byphase=defaultdict(list);hist=defaultdict(lambda:defaultdict(list))
        for event in records(run/(method_key+'_updates.jsonl')):
            phase=event['phase'];byphase[phase].append(sum(h['sampled_episodes'] for h in event['historical_tasks']))
            for h in event['historical_tasks']:
                bucket=hist[phase,h['task']]
                for k in ('sampled_episodes','loss','valid_states','decision_states','valid_pairs','pair_violation_rate','top1_flip_rate'):
                    if h.get(k) is not None:bucket[k].append(h[k])
        for phase,counts in byphase.items():sampling.append(dict(ident,phase=phase,scope='all_historical_tasks',expected=phase*8,violations=sum(n!=phase*8 for n in counts),**stats(counts)))
        for (phase,task_name),values in hist.items():
            sampling.append(dict(ident,phase=phase,historical_task=task_name,scope='task',expected=8,violations=sum(n!=8 for n in values['sampled_episodes']),**stats(values['sampled_episodes'])))
            row=dict(ident,phase=phase,historical_task=task_name)
            for key,v in values.items():row.update({key+'_'+k:n for k,n in stats(v).items()})
            prefs.append(row)
        monitor=defaultdict(list)
        for r in records(run/'experiment_metrics.jsonl'):
            if r.get('context',{}).get('event')=='gradient_monitor' and r.get('metric') in GRADS and r.get('value') is not None:
                monitor[r['phase'],r['metric']].append(r['value'])
        for (phase,metric),values in monitor.items():gradients.append(dict(ident,phase=phase,metric=metric,**stats(values)))
        for r in records(run/'fixed_reference_memory.jsonl'):
            memory.append(dict(ident,**r,source_still_matches=digest(Path(r['source_pool']))==r['source_sha256'],local_still_matches=digest(Path(r['local_pool']))==r['local_sha256']))
        memory_updates.extend(dict(ident,**r) for r in records(run/'own_policy_memory.jsonl'))
        memory_updates.extend(dict(ident,**r) for r in records(run/'tkpr_win_refresh_memory_updates.jsonl'))
        for path in sorted((run/'evaluation').glob('*/results.json'),reverse=True):
            if read(path.parent/'launcher_status.json').get('exit_code')==0:
                independent.extend(dict(ident,**{k:r[k] for k in ('task','wins','episodes','win_rate')}) for r in read(path)['tasks']);break
        for path in (run/'status.json',run/'launch.json',run/'retention.json',run/'evaluations.jsonl',run/(method_key+'_updates.jsonl'),run/'fixed_reference_memory.jsonl',run/'experiment_metrics.jsonl',run/'own_policy_memory.jsonl',run/'tkpr_win_refresh_memory_updates.jsonl'):
            if path.exists():sources[str(path)]=dict(bytes=path.stat().st_size,sha256=digest(path))
    baseline=Path(plan['baseline_dataset'])
    for name,rows in (('runs',runs),('stage_metrics',stages),('learning_curves',curves),('evaluations',evals),('independent_final',independent)):
        for r in csvread(baseline/(name+'.csv')):
            if float(r['min_gap'])==.5:
                r=dict(r,method='TKPR-K1-native',**({'baseline_reused':True}));rows.append(r)
                if name=='runs':
                    import shutil
                    shutil.copy2(baseline/r['config'],target/r['config'])
    paired=[]
    for r in stages:
        if r['method']=='TKPR-K1-native':continue
        control_method='TKPR-K1-PerTask8' if r['method']=='TKPR-WinRefresh' else 'TKPR-K1-native'
        control=next((x for x in stages if x['method']==control_method and int(x['seed'])==r['seed'] and int(x['phase'])==r['phase']),None)
        if control is None:continue
        row=dict(seed=r['seed'],phase=r['phase'],run_id=r['run_id'],control_run_id=control['run_id'],method=r['method'],control=control_method,comparison_kind='matched_own_policy_histories' if r['method']=='TKPR-WinRefresh' else 'historical_control_different_history_source' if own else 'budget_only_matched_fixed_histories',same_parent=r['parent_run_id']==control['parent_run_id'])
        for key in METRICS:row[key+'_delta']=float(r[key])-float(control[key]) if r.get(key) is not None and control.get(key) not in (None,'') else None
        paired.append(row)
    for seed in plan['seeds']:
        for metric in GRADS:gradients.append(dict(seed=seed,method='TKPR-K1-native',metric=metric,n=0,missing_reason='baseline gradient_monitor_interval=0'))
    for name,rows in (('runs',runs),('stage_metrics',stages),('learning_curves',curves),('evaluations',evals),('sampling_budget',sampling),('preference_diagnostics',prefs),('gradient_diagnostics',gradients),('independent_final',independent),('memory_copies',memory),('memory_updates',memory_updates),('paired_by_phase',paired)):
        write(target/(name+'.csv'),rows)
    final=[]
    methods=('TKPR-K1-native','TKPR-K1-PerTask8','TKPR-WinRefresh') if own else ('TKPR-K1-native','TKPR-K1-PerTask8')
    for method in methods:
        rs=[r for r in stages if r['method']==method and int(r['phase'])==4]
        for key in METRICS:final.append(dict(method=method,metric=key,**stats([float(r[key]) for r in rs if r.get(key) not in (None,'') ])))
    write(target/'final_summary.csv',final)
    summary=dict(training_complete=sum(r['status']=='completed' for r in runs if r['method']!='TKPR-K1-native'),new_training_total=4 if own else 2,
                 independent_new_points=sum(r['method']!='TKPR-K1-native' for r in independent),sampling_violations=sum(r['violations'] for r in sampling))
    (target/'execution_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (target/'source_manifest.json').write_text(json.dumps(dict(sources=sources,plan=str(plan_path),baseline=str(baseline),baseline_files={str(p):digest(p) for p in baseline.glob('*.csv')}),indent=2)+'\n')
    lines=['# TS5-1M · '+('Own-policy PerTask8 vs WinRefresh' if own else 'TKPR-K1-PerTask8'),'',f"新训练完成 {summary['training_complete']}/{summary['new_training_total']}；最终独立进程评价 {summary['independent_new_points']}/{summary['new_training_total']*5}。",'',
        '与原生gap0.5 K1使用相同seed2222/3535、相同A checkpoint/target/optimizer/RNG、相同完整固定历史轨迹和old-Q，B–E各1M。唯一算法变化为总8条→每历史任务8条；保持Top-1、min_gap0.5、权重1、温度1、std floor0.1与有效任务等权平均，不刷新、不替换池。按用户确认，仅新分支开启每100更新的既有梯度诊断，旧对照此项缺失，不补造。','',
        'A为共享前缀引用，不是新增训练。Learning AUC沿用原生基准定义：phase_start、periodic和stage_end的胜率对实际步数梯形积分并归一化；保留不同评价块的原始wins/episodes，不与其他AUC定义混用。ACC/BWT/FWT按现有口径。最终128局独立进程复核沿用focus种子，不称独立测试集。','',
        '| 方法 | 指标 | n | 均值 | 样本SD |','| --- | --- | ---: | ---: | ---: |']
    if own:
        lines[4]='两新方法使用相同seed2222/3535与A前缀。B/C/D/E池由各分支自身任务结束策略采集（含rollout参考Q），每任务8条，K1损失、min_gap0.5、权重1与任务等权归约不变。PerTask8不再更新旧池；WinRefresh仅在当前胜率严格超过现池采集时记录的胜率时重新采集最多100成功轨迹并全量替换，不保留旧轨迹，零成功则保留旧池及旧胜率基准。两者梯度监控间隔100。原fixed-FT-history PerTask8试验已停止，不作为完成结果。旧原生K1仅作历史参照：其B/C/D轨迹来自FT参考，故与新PerTask8同时有采样量和数据来源差异，不能作纯预算因果比较。'
    for r in final:
        if r['n']:lines.append(f"| {r['method']} | {r['metric']} | {r['n']} | {r['mean']:.3f} | {r['std'] if r['std'] is not None else 'N/A'} |")
    lines+=['','配对差值及对照类型见paired_by_phase.csv的method/control/comparison_kind字段；各任务AUC与最终胜率见stage_metrics.csv，采样总量和逐任务分配见sampling_budget.csv。诊断含重复采样，不能当独立种子。两个种子不足以证明普遍收益。模型、完整Replay、轨迹与原始日志保留实验主机；不修改原始对照数据。','']
    if curves:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,5,figsize=(18,3.3),sharey=True)
        for phase,ax in enumerate(axs):
            for method,style in (('TKPR-K1-native',':'),('TKPR-K1-PerTask8','--'),*([('TKPR-WinRefresh','-')] if own else [])):
                for seed,color in ((2222,'tab:blue'),(3535,'tab:orange')):
                    rs=sorted([r for r in curves if r['method']==method and int(r['phase'])==phase and int(r['seed'])==seed],key=lambda r:int(r['stage_steps']))
                    if rs:ax.plot([int(r['stage_steps'])/1e6 for r in rs],[float(r['win_pct']) for r in rs],style,color=color,label=f'{method} {seed}')
            ax.set(title='ABCDE'[phase],ylim=(0,100),xlabel='Stage steps (M)')
        axs[0].set_ylabel('Win rate (%)');axs[-1].legend(fontsize=6);fig.tight_layout();fig.savefig(target/'learning_curves.png',dpi=150);plt.close(fig)
        lines+=['![Learning curves](learning_curves.png)','']
    (target/'README.md').write_text('\n'.join(lines))
    if summary['training_complete']==summary['new_training_total'] and summary['independent_new_points']==5*summary['new_training_total']:
        catalog_path=ROOT/'docs/results/catalog.json';catalog=read(catalog_path)
        prefix='parameter-ablations/'+target.name+'/'
        entry=dict(id=target.name,category='parameter-ablations',report=prefix+'README.md',
            files={p.stem:prefix+p.name for p in target.glob('*.csv')},task_order=list('ABCDE'),training_seeds=[2222,3535],final_phase=4,
            comparison=dict(kind='own_policy_history_refresh_ablation' if own else 'native_K1_sampling_budget_same_full_history',match_on=['seed','parent_run_id'],primary_control=dict(method='TKPR-K1-PerTask8' if own else 'TKPR-K1-native',min_gap=.5)),
            interpretation_limits=['两训练种子；A为共享前缀引用。','新两组相同逐任务8采样，仅历史池更新规则不同；旧原生对照的数据来源也不同。' if own else '仅采样预算改变算法，新分支额外启用既有梯度诊断，旧对照梯度缺失。','新实验按任务末自身策略采集；WinRefresh性能严格提高才全量换池，历史原生K1的数据来源不同。' if own else '静态完整历史池按任务边界激活；不刷新，不替换，不重评Q。','独立进程复核沿用focus种子，不是独立测试集。'])
        catalog['experiments']=[e for e in catalog['experiments'] if e['id']!=target.name]+[entry]
        catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
        index=ROOT/'docs/results/README.md';text=index.read_text()
        anchor='| --- | --- | --- |\n'
        if prefix not in text:index.write_text(text.replace(anchor,anchor+'| [TS5-1M · Own-policy PerTask8/Refresh]('+prefix+'README.md) | K1 gap0.5，seed2222/3535，相同A前缀、各任务自身策略采集 | PerTask8静态自身池与胜率提高才全量换池的配对结果、采样与梯度诊断。 |\n',1))
    print(json.dumps(summary))
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);p.add_argument('--commit',action='store_true');a=p.parse_args();target=report(a.plan.resolve())
    if a.commit:
        relative=str(target.relative_to(ROOT))
        paths=[relative,'docs/results/catalog.json','docs/results/README.md']
        subprocess.check_call(['git','add','--',*paths],cwd=str(ROOT))
        subprocess.check_call(['git','diff','--cached','--check'],cwd=str(ROOT))
        subprocess.check_call(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','--only','-m','Publish TKPR own-policy history and win-gated refresh results','--',*paths],cwd=str(ROOT))
