"""Export and verify the four native TKPR-K1 gap-only branches; optionally publish terminal data."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean,stdev
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'drivers/continual'),str(ROOT/'scripts')]
from drivers.continual.memory.fixed_reference import file_sha256
from prepare_native_k1_gap import write

DATASET='ts5-native-tkpr-k1-gap'


def read(path):return json.loads(path.read_text()) if path.exists() else {}
def records(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:yield json.loads(line)


def summarize(plan_path):
    plan=read(plan_path);out=plan_path.parent
    manifests=[];evaluations=[];curves=[];metrics=[];diagnostics=[];independent=[];memory=[];snapshots={};verification=[]
    for task in plan['tasks']:
        if task['type']!='training':continue
        run=Path(task['run_dir']);seed=task['metadata']['seed'];gap=task['metadata']['min_gap']
        parent=ROOT/'runs/tc_coverage'/plan['reference_runs'][str(seed)]
        identity=dict(run_id=run.name,seed=seed,min_gap=gap,method='TKPR-K1-native',parent_run_id=parent.name)
        protocol=read(run/'protocol.json');config=read(run/'config.json');state=read(run/'status.json');launch=read(run/'launch.json');restore=read(run/'branch_restore.json')
        manifests.append(dict(identity,status=state.get('status','not_started'),commit=launch.get('git_commit'),
            completed_global_steps=state.get('t_env'),new_training_steps=state.get('t_env',0)-restore.get('t_env',0) if state else None,
            epsilon_on_switch=config.get('epsilon_on_switch'),replay_on_switch=config.get('replay_on_switch'),
            config='configs/'+run.name+'.json'))
        snapshots[run.name]=dict(config=config,protocol=protocol,environment=read(run/'env_info.json'),
            launch_commit=launch.get('git_commit'),branch_restore=restore)
        own=list(records(run/'evaluations.jsonl'))
        prefix=[r for r in records(parent/'evaluations.jsonl') if r['phase']==0]
        all_eval=prefix+own
        evaluations += [dict(identity,**{k:r.get(k) for k in ('phase','task','label','t_env','wins','episodes')},shared_prefix=False) for r in own]
        evaluations += [dict(identity,**{k:r.get(k) for k in ('phase','task','label','t_env','wins','episodes')},shared_prefix=True) for r in prefix]
        initial={r['task']:r['wins']/r['episodes'] for r in all_eval if r['label']=='initial_reference'}
        before={r['task']:r['wins']/r['episodes'] for r in all_eval if r['label']=='next_unseen'}
        order=[s['task'] for s in protocol.get('stages',[])]
        previous={}
        for stage in read(run/'retention.json').get('stages',[]):
            phase=stage['phase'];task_name=stage['task'];start=stage['start'];end=stage['t_env']
            end_rows=[r for r in all_eval if r['phase']==phase and r['label']=='stage_end']
            endpoints={r['task']:r['wins']/r['episodes'] for r in end_rows}
            if set(endpoints)!=set(order[:phase+1]):continue
            bwt=[endpoints[t]-previous[t] for t in previous if t!=task_name];previous[task_name]=endpoints[task_name]
            fwt=[before[t]-initial[t] for t in order[1:phase+1] if t in before and t in initial]
            curve=sorted([r for r in all_eval if r['phase']==phase and r['task']==task_name and r['label'] in ('phase_start','periodic','stage_end')],key=lambda r:r['t_env'])
            auc=None
            if curve and curve[0]['t_env']==start and curve[-1]['t_env']==end:
                auc=100*sum((a['wins']/a['episodes']+b['wins']/b['episodes'])*(b['t_env']-a['t_env'])/2 for a,b in zip(curve,curve[1:]))/(end-start)
            metric=dict(identity,phase=phase,training_task=task_name,start=start,t_env=end,
                acc_pct=100*mean(endpoints.values()),bwt_pp=100*mean(bwt) if bwt else None,
                fwt_pp=100*mean(fwt) if fwt and len(fwt)==phase else None,learning_auc_pct=auc,
                **{t+'_win_pct':100*v for t,v in endpoints.items()})
            metrics.append(metric)
            for key in ('acc_pct','bwt_pp','fwt_pp'):
                other=stage.get({'acc_pct':'acc_percent'}.get(key,key))
                if other is not None and metric[key] is not None and abs(other-metric[key])>1e-8:
                    raise ValueError('Metric reconstruction differs from trainer retention: '+run.name)
            curves += [dict(identity,phase=phase,task=task_name,t_env=r['t_env'],stage_steps=r['t_env']-start,
                             label=r['label'],wins=r['wins'],episodes=r['episodes'],win_pct=100*r['wins']/r['episodes'],shared_prefix=phase==0) for r in curve]
        aggregates=defaultdict(lambda:dict(updates=0,decision_states=0,states_with_pairs=0,selected_pairs=0,loss_sum=0.,loss_updates=0,violation_weight=0.,violation_pairs=0))
        future=0
        for row in records(run/'topk_preference_updates.jsonl'):
            for h in row['historical_tasks']:
                key=(row['phase'],h['source_phase'],h['task']);a=aggregates[key]
                future+=int(h['source_phase']>=row['phase'])
                a['updates']+=1;a['decision_states']+=h['decision_states'];a['states_with_pairs']+=h['valid_states'];a['selected_pairs']+=h['valid_pairs']
                if h.get('loss') is not None:a['loss_sum']+=h['loss'];a['loss_updates']+=1
                if h.get('pair_violation_rate') is not None:
                    a['violation_weight']+=h['pair_violation_rate']*h['valid_pairs'];a['violation_pairs']+=h['valid_pairs']
        for (phase,source,history),a in aggregates.items():
            diagnostics.append(dict(identity,phase=phase,source_phase=source,historical_task=history,
                updates=a['updates'],decision_states=a['decision_states'],states_with_pairs=a['states_with_pairs'],
                zero_pair_decision_states=a['decision_states']-a['states_with_pairs'],selected_pairs=a['selected_pairs'],
                mean_task_loss=a['loss_sum']/a['loss_updates'] if a['loss_updates'] else None,
                pair_violation_rate=a['violation_weight']/a['violation_pairs'] if a['violation_pairs'] else None))
        attached=list(records(run/'fixed_reference_memory.jsonl'));memory += [dict(identity,**row) for row in attached]
        for path in (run/'evaluation').glob('*/results.json'):
            if read(path.parent/'launcher_status.json').get('exit_code')==0:
                independent += [dict(identity,task=r['task'],wins=r['wins'],episodes=r['episodes']) for r in read(path)['tasks']]
        if state.get('status')=='completed':
            assert config['method']=='topk_preference' and config['topk_preference']['k']==1
            assert config['topk_preference']['min_gap']==gap and protocol['fixed_reference_history']
            assert not protocol.get('pair_ablation_experiment') and not protocol.get('tc_mask_root')
            assert read(run/'launcher_status.json')['exit_code']==0 and len(attached)==4 and future==0
            assert restore['model_hash']==restore['parent_model_hash']
            for row in attached:
                assert row['preference_class']=='RankingPreference' and row['loss_function']=='learners.ranking_preference.topk_preference'
                assert row['source_sha256']==row['local_sha256']==file_sha256(Path(row['source_pool']))==file_sha256(Path(row['local_pool']))
                assert row['state_filter'] is False
            verification.append(dict(identity,passed=True,history_phases=len(attached),future_pool_update_rows=future,
                restore_model_hash=restore['model_hash'],parent_model_hash=restore['parent_model_hash']))
    write(out/'runs.csv',manifests);write(out/'evaluations.csv',evaluations);write(out/'learning_curves.csv',curves)
    write(out/'stage_metrics.csv',metrics);write(out/'training_pair_counts.csv',diagnostics)
    write(out/'independent_final.csv',independent);write(out/'memory_copies.csv',memory);write(out/'verification.csv',verification)
    (out/'configs').mkdir(exist_ok=True)
    for name,snapshot in snapshots.items():(out/'configs'/(name+'.json')).write_text(json.dumps(snapshot,indent=2)+'\n')
    paired=[]
    for seed in plan['seeds']:
        for phase in range(5):
            a=next((r for r in metrics if r['seed']==seed and r['min_gap']==0 and r['phase']==phase),None)
            b=next((r for r in metrics if r['seed']==seed and r['min_gap']==.5 and r['phase']==phase),None)
            if a is None or b is None:continue
            paired.append(dict(seed=seed,phase=phase,gap0_run_id=a['run_id'],gap05_run_id=b['run_id'],
                 **{k+'_delta':b[k]-a[k] if b.get(k) is not None and a.get(k) is not None else None for k in
                     ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct','A_win_pct','B_win_pct','C_win_pct','D_win_pct','E_win_pct')}))
    write(out/'paired_gap05_minus_gap0.csv',paired)
    aggregate=[]
    for gap in (0,.5):
        for phase in range(5):
            for metric in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct'):
                values=[r[metric] for r in metrics if r['min_gap']==gap and r['phase']==phase and r[metric] is not None]
                aggregate.append(dict(min_gap=gap,phase=phase,metric=metric,n=len(values),mean=mean(values) if values else None,std=stdev(values) if len(values)>1 else None))
    write(out/'seed_summary.csv',aggregate)
    final_deltas=[r for r in paired if r['phase']==4]
    summary=dict(training_completed=sum(r['status']=='completed' for r in manifests),training_total=4,
                 terminal_verified=len(verification),stage_metric_rows=len(metrics),independent_model_task_rows=len(independent))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# TS5-1M · 原生 TKPR-K1 Gap 消融','',
       '原生 `topk_preference(k=1)`，min_gap=0/0.5 × seed2222/3535，仅四条新增训练。禁止使用的 pair_tc_k1 未参与此实验。两种 gap 复用同种子原 FT 参考的 S 检查点（在线网络、目标网络、优化器和 RNG）以及完整 S/Z/C/D 历史轨迹与 rollout-stored old Q；不重新训练参考模型。',
       '任务为 S→Z→C→2S3Z→2Z3C，各阶段1M；单层 VDN64、FT、逐任务 epsilon、RMSprop、lr=0.0005、偏好权重1、temperature1、old-Q std floor0.1、历史batch8、评价种子与局数均沿用原实验。分支新增 Z/C/D/E 约4M步，总新增名义16M；S 前缀每种子只训练过一次。',
       '没有 TC 有效状态筛选、Near/Mid/Far 配额或额外边。完整历史池逐阶段接入，训练Z仅见S，C见S/Z，D见S/Z/C，E见S/Z/C/D。仅原生合法/存活/终止掩码、至少两个合法动作及非零 old-Q pair 定义参与原生损失。gap0保留全部原生K1合法非并列动作对；gap0.5仅额外要求绝对归一化历史Q差>=0.5。未通过间隔的动作对不产生偏好损失；零动作对状态与整条轨迹仍留在历史池，RNN仍按完整时序重放。',
       '正常 `src/learners/ranking_preference.py` 与 QLearner 的损失和优化代码未改变。新增功能仅为原检查点兼容和按阶段复制固定历史池，运行记录确认 RankingPreference 调用原生 topk_preference。',
       '', '完成训练：%d/4；终态核验通过：%d；阶段指标：%d；独立模型×任务终点评价：%d。'%(summary['training_completed'],summary['terminal_verified'],summary['stage_metric_rows'],summary['independent_model_task_rows']),
       '', '| Seed | gap | ACC % | BWT pp | FWT pp | E AUC % |','|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        if r['phase']==4:lines.append('| %d | %s | %s |'%(r['seed'],r['min_gap'],' | '.join('N/A' if r[k] is None else '%.3f'%r[k] for k in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct'))))
    if final_deltas:
        lines += ['', '最终 ACC 的 gap0.5−gap0 逐种子差值：'+ '；'.join('seed%d %+.3f pp'%(r['seed'],r['acc_pct_delta']) for r in final_deltas)+'。']
    lines += ['', '## 数据与核验','',
       '- `stage_metrics.csv`：由评价胜局计数重算 ACC/BWT/FWT、每阶段当前任务胜率的归一化梯形 AUC，以及全部已见任务端点胜率。AUC按实际阶段步数归一化，单位百分比；BWT/FWT为百分点。与训练器 retention 记录对照核验。',
       '- `evaluations.csv` / `learning_curves.csv`：原始胜局数、局数和当前任务学习曲线；shared_prefix=true 表示同种子共用S前缀，不计为额外重复。`paired_gap05_minus_gap0.csv` 是同种子逐阶段 gap0.5−gap0 差值。',
       '- `historical_pair_inventory.csv`：完整固定池的状态/轨迹数、池哈希、S检查点与优化器哈希、逐池实际原生动作对数、零对状态数和Q-gap分布。`training_pair_counts.csv` 为实际训练历史抽样的状态/动作对计数，包含重复；两者分母不同。',
       '- `memory_copies.csv` / `verification.csv`：全部历史池逐字节相同、原生实现类/函数、检查点恢复哈希、无未来池参与更新。`configs/` 和 `runs.csv` 保留逐运行真实配置与训练提交。独立终点评价为每任务128局。',
       '只有两个训练种子；不得把历史状态或评价局当训练重复。与旧 TC 子集 K1 不同，本次唯一干预是原生动作对 gap 筛选；两个 gap 不改变历史池或轨迹状态总体。旧 TC 结果不修改、不并入本次主对比。']
    if curves:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,5,figsize=(18,3.4),sharey=True)
        for phase,ax in enumerate(axes):
            for seed,color in ((2222,'#2672b6'),(3535,'#d96531')):
                for gap,style in ((0,'-'),(.5,'--')):
                    data=[r for r in curves if r['seed']==seed and r['min_gap']==gap and r['phase']==phase]
                    data.sort(key=lambda r:r['stage_steps'])
                    if data:ax.plot([r['stage_steps']/1e6 for r in data],[r['win_pct'] for r in data],style,color=color,lw=1.4,label='%d gap%s'%(seed,gap))
            ax.set(title=['S','Z','C','2S3Z','2Z3C'][phase],xlabel='Stage steps (M)',ylim=(-2,102));ax.grid(alpha=.2)
        axes[0].set_ylabel('Greedy win rate (%)');axes[-1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
        lines += ['', '![各阶段学习曲线](learning_curves.png)','S 曲线按种子共享；定期评价与阶段末评价局数不同，保留原始计数。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary));return summary


def relative(value):
    if isinstance(value,str):return value.replace(str(ROOT)+'/', '')
    if isinstance(value,list):return [relative(v) for v in value]
    if isinstance(value,dict):return {k:relative(v) for k,v in value.items()}
    return value


def publish(plan_path):
    import shutil,sqlite3
    summary=summarize(plan_path);out=plan_path.parent;plan=read(plan_path)
    if summary!=dict(training_completed=4,training_total=4,terminal_verified=4,stage_metric_rows=20,independent_model_task_rows=20):
        raise ValueError('Incomplete native-K1 ablation: partial report retained; no publication or extra training')
    implementation=read(out/'implementation_manifest.json')
    for source in read(out/'reference_manifest.json'):
        assert file_sha256(Path(source['source_pool']))==source['pool_sha256']
        assert file_sha256(Path(source['s_checkpoint'])/'agent.th')==source['s_agent_sha256']
        assert file_sha256(Path(source['s_checkpoint'])/'opt.th')==source['s_optimizer_sha256']
        assert file_sha256(ROOT/'runs/tc_coverage'/source['parent_run_id']/'return_state.th')==source['s_return_state_sha256']
    for seed in (2222,3535):
        for phase in range(4):
            subset=[r for r in records_csv(out/'memory_copies.csv') if int(r['seed'])==seed and int(r['source_phase'])==phase]
            assert len(subset)==2 and len({r['local_sha256'] for r in subset})==1
    target=ROOT/'docs/results/parameter-ablations'/DATASET
    if target.exists():raise ValueError('Preserve existing result: '+str(target))
    common=['docs/results/catalog.json','docs/results/README.md']
    for file in common:
        if subprocess.call(['git','diff','--quiet','HEAD','--',file],cwd=str(ROOT)):raise ValueError('Preserve pending index edits: '+file)
    target.mkdir()
    for path in [out/'REPORT.md',out/'reference_manifest.json',out/'implementation_manifest.json',out/'learning_curves.png',out/'summary.json',*out.glob('*.csv')]:
        shutil.copy2(str(path),str(target/path.name))
    shutil.copytree(str(out/'configs'),str(target/'configs'))
    for path in (target/'configs').glob('*.json'):path.write_text(json.dumps(relative(read(path)),indent=2)+'\n')
    (target/'reference_manifest.json').write_text(json.dumps(relative(read(out/'reference_manifest.json')),indent=2)+'\n')
    for name in ('memory_copies.csv','historical_pair_inventory.csv'):
        data=[relative(r) for r in records_csv(target/name)];write(target/name,data)
    manifest={}
    for task in plan['tasks']:
        if task['type']=='training':
            launch=read(Path(task['run_dir'])/'launch.json')
            for path,digest in implementation['source_sha256'].items():
                assert launch['source_sha256'][path]==digest, 'Native loss/optimizer implementation changed'
            manifest[task['id']]=dict(commit=launch['git_commit'],source_sha256=launch['source_sha256'])
    conn=sqlite3.connect(str(ROOT/'runtime/scheduler/state.sqlite3'));conn.row_factory=sqlite3.Row
    execution=[dict(r) for r in conn.execute('select id,state,reason from tasks where id like ?', (plan['id']+'%',))];conn.close()
    (target/'source_manifest.json').write_text(json.dumps(dict(training=manifest,execution=execution),indent=2)+'\n')
    (target/'README.md').write_text((out/'REPORT.md').read_text()+'\n完整模型、固定历史池与运行日志留在实验主机 `results/'+plan['id']+'/` 及 `runs/'+plan['id']+'_*`。本次是独立的原生 K1 gap-only 对照；旧 TC 与子集 K1 数据保持原样。\n')
    index=ROOT/'docs/results/README.md';text=index.read_text();anchor='| [TS5-1M · K1 与 gap0/0.5 扩展]';position=text.index(anchor)
    text=text[:position]+'| [TS5-1M · 原生 TKPR-K1 Gap 消融](parameter-ablations/'+DATASET+'/README.md) | 原生 topk_preference(k=1)，gap0/0.5 × 2222/3535 | 完整共同历史池、无状态筛选、实际动作对计数与配对性能。 |\n'+text[position:];index.write_text(text)
    catalog_path=ROOT/'docs/results/catalog.json';catalog=read(catalog_path);assert not any(e['id']==DATASET for e in catalog['experiments'])
    prefix='parameter-ablations/'+DATASET+'/'
    catalog['experiments'].append(dict(id=DATASET,category='parameter-ablations',report=prefix+'README.md',
        files={p.stem:prefix+p.name for p in target.glob('*.csv')},task_order=['A','B','C','D','E'],training_seeds=[2222,3535],final_phase=4,
        comparison=dict(kind='native_TKPR_K1_gap_only_same_full_history',match_on=['seed','parent_run_id'],min_gap=[0,.5],method='topk_preference',k=1),
        interpretation_limits=['仅四条新增训练；没有参考重训或TC分支重跑。','两个gap使用相同完整历史轨迹；仅原生动作对筛选变化，零对状态仍保留。','原生 RankingPreference/topk_preference，不是 pair_tc_k1；历史池按任务边界激活。','S前缀每种子共享，两训练种子不足以推断普遍最优阈值。']))
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    paths=[str(target.relative_to(ROOT)),*common]
    subprocess.check_call(['git','add','--',*paths],cwd=str(ROOT));subprocess.check_call(['git','diff','--cached','--check'],cwd=str(ROOT))
    subprocess.check_call(['git','-c','user.name=Codex','-c','user.email=codex@openai.com','commit','--only','-m','Publish native TKPR-K1 gap-only two-seed ablation','--',*paths],cwd=str(ROOT))


def records_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path);parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.publish:publish(args.plan.resolve())
    else:summarize(args.plan.resolve())
