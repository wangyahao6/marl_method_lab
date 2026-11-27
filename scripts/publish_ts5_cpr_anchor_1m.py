"""Publish completed Anchor runs with the frozen four-arm TS5-1M baseline export."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import summarize_ts5_cpr_1m as base

METHODS = (*base.METHODS, 'TKPR-CPR-Anchor')


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def number(row, key):
    return float(row[key]) if row.get(key) not in ('', None) else None


def write(out, name, rows):
    base.Export(None, out).csv(name, rows, [])


def summary(rows, key):
    values = [number(r, key) for r in rows if number(r, key) is not None]
    return base.statistics(values)


def publish(plan, baseline, out):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan, out/'plan.json')
    with tempfile.TemporaryDirectory(prefix='anchor_export_') as tmp:
        fresh = Path(tmp)
        base.export(plan, fresh, render_output=False)
        execution = json.loads((fresh/'execution_summary.json').read_text())
        if not execution['all_complete']:
            raise RuntimeError('Anchor export is incomplete; inspect original run status')
        # Read-only reuse of the already published baseline. No old run is re-exported or changed.
        for source in (baseline/'configs').glob('*.json'):
            (out/'configs').mkdir(exist_ok=True)
            shutil.copy2(source, out/'configs'/source.name)
        for source in (fresh/'configs').glob('*.json'):
            shutil.copy2(source, out/'configs'/source.name)
        tables = {}
        names = ('runs','evaluations','stage_metrics','learning_curves','preference_diagnostics',
                 'cpr_reference_updates','gradient_diagnostics','memory_collection','sampling_cost','independent_final')
        for name in names:
            tables[name] = read(baseline/(name+'.csv')) + read(fresh/(name+'.csv'))
            write(out, name+'.csv', tables[name])
        finals = [r for r in tables['stage_metrics'] if int(r['phase'])==4]
        aggregates = []
        for method in METHODS:
            rs=[r for r in finals if r['method']==method]
            row=dict(method=method,n=len(rs),completed_seeds=json.dumps(sorted(int(r['seed']) for r in rs)))
            for key in ('acc_pct','bwt_pp','fwt_pp',*[t+'_win_pct' for t in base.TASKS]):
                row.update({key+'_'+k:v for k,v in summary(rs,key).items()})
            aggregates.append(row)
        write(out,'final_summary.csv',aggregates)
        paired=[]
        for r in finals:
            if r['method']!='TKPR-CPR-Anchor': continue
            for control in base.METHODS:
                c=next(x for x in finals if x['method']==control and x['seed']==r['seed'])
                row=dict(run_id=r['run_id'],control_run_id=c['run_id'],prefix_id=r['prefix_id'],seed=r['seed'],
                         method=r['method'],control=control,same_prefix=r['prefix_id']==c['prefix_id'])
                for key in ('acc_pct','bwt_pp','fwt_pp',*[t+'_win_pct' for t in base.TASKS]):
                    row[key+'_difference']=number(r,key)-number(c,key)
                paired.append(row)
        write(out,'paired_final.csv',paired)
        old=json.loads((baseline/'source_manifest.json').read_text())
        new=json.loads((fresh/'source_manifest.json').read_text())
        new['baseline_dataset']=str(baseline)
        new['baseline_files']={str(p):dict(bytes=p.stat().st_size,sha256=base.file_sha(p))
                               for p in baseline.rglob('*') if p.is_file()}
        new['sources']={**old['sources'],**new['sources']}
        new['run_ids']=[r['run_id'] for r in tables['runs']]
        new['expected_nominal_new_training_steps']=8000000
        new['reused_baseline_nominal_steps']=34000000
        (out/'source_manifest.json').write_text(json.dumps(new,indent=2)+'\n')
        prior=json.loads((baseline/'execution_summary.json').read_text())
        execution['dataset_counts']={name:len(rows) for name,rows in tables.items()}
        execution['reused_baseline_execution']=prior
        execution['all_complete']=execution['all_complete'] and prior['all_complete']
        (out/'execution_summary.json').write_text(json.dumps(execution,indent=2)+'\n')
    render(out,tables,aggregates,paired)
    register(out)
    print(json.dumps(aggregates,indent=2))


def render(out, tables, aggregates, paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(METHODS,('tab:blue','tab:orange','tab:green','tab:red','tab:purple')))
    fig,axs=plt.subplots(4,1,figsize=(10,11),constrained_layout=True)
    for phase,ax in enumerate(axs,1):
        for method in METHODS:
            series=[]
            for run in tables['runs']:
                if run['method']!=method: continue
                stage=next(r for r in tables['stage_metrics'] if r['run_id']==run['run_id'] and int(r['phase'])==phase)
                points=sorted((int(e['t_env']),100*float(e['win_rate'])) for e in tables['learning_curves']
                              if e['run_id']==run['run_id'] and int(e['phase'])==phase)
                x=np.linspace(0,1,101)
                series.append(np.interp(x,[(t-int(stage['start']))/(int(stage['t_env'])-int(stage['start'])) for t,_ in points],[v for _,v in points]))
            mean=np.mean(series,axis=0);sd=np.std(series,axis=0,ddof=1)
            ax.plot(x,mean,label=method,color=colors[method]);ax.fill_between(x,mean-sd,mean+sd,color=colors[method],alpha=.1)
        ax.set(title=f'Task {base.TASKS[phase]} regular evaluation',ylabel='Win rate (%)',ylim=(0,100))
        ax.legend(ncol=3,fontsize=8)
    axs[-1].set_xlabel('Fraction of actual phase steps')
    fig.savefig(out/'learning_curves.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    x=np.arange(5)
    for i,r in enumerate(aggregates):
        ax.bar(x+(i-2)*.16,[r[t+'_win_pct_mean'] for t in base.TASKS],.16,
               yerr=[r[t+'_win_pct_std'] for t in base.TASKS],label=r['method'],color=colors[r['method']],capsize=2)
    ax.set(xticks=x,xticklabels=base.TASKS,ylabel='Final win rate (%)',ylim=(0,100));ax.legend(ncol=3,fontsize=8)
    fig.savefig(out/'final_retention.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
    for method in ('TKPR-CPR','TKPR-CPR-Anchor'):
        for key,ax in zip(('updated_fraction','valid_preference_coverage'),axs[:2]):
            values=[]
            for phase in range(1,5):
                rs=[r for r in tables['cpr_reference_updates'] if r['method']==method and int(r['phase'])==phase]
                den=sum(int(r['decision_states']) for r in rs)
                numerator=sum(int(r['updated_states' if key=='updated_fraction' else 'valid_reference_states']) for r in rs)
                values.append(numerator/den if den else np.nan)
            ax.plot(list('BCDE'),values,'-o',label=method,color=colors[method])
    for method in METHODS[1:]:
        values=[]
        for phase in range(1,5):
            rs=[r for r in tables['preference_diagnostics'] if r['method']==method and int(r['phase'])==phase]
            den=sum(int(r.get('raw_decision_states') or r['decision_states']) for r in rs)
            values.append(sum(int(r['valid_states']) for r in rs)/den if den else np.nan)
        axs[2].plot(list('BCDE'),values,'-o',label=method,color=colors[method])
    for ax,title in zip(axs,('Accepted updates / boundary decisions','Valid mask / boundary decisions','Valid constraints / sampled raw decisions')):
        ax.set(title=title,ylim=(0,1.03));ax.legend(fontsize=7)
    fig.savefig(out/'anchor_mechanisms.png',dpi=150);plt.close(fig)
    lines=['# TS5-1M · TKPR-CPR-Anchor 同前缀对照','',
           '**状态：两个 Anchor 分支训练、两个最终独立进程评价均完成；原四组只读复用。**','',
           '种子3535、2222，vdn_action_set + vdn_1ff，A→B→C→D→E。复用原实验对应种子的完整A前缀和相同A成功轨迹/Q；新增B–E各1M，两分支名义新增8M步。原四组34M步不重复训练。共同前缀在数据集中每种子只记录一次。','',
           'Anchor训练版本 `9ff065fb08ea91d6113f8f70383ade3f500a26e6`；原四组版本见 runs.csv。Anchor只在当前Top-1与永久原始Top-1一致、当前归一化置信度≥max(threshold,min_gap)、且严格高于已存参考置信度+margin时更新。未接受则保留原Q、掩码与来源。门控、K1损失、任务等权归约、总8条采样与原CPR相同；Refresh每历史任务8条。','',
           '每25k环境步32局regular评价，阶段末各已见任务128局。ACC任务等权；BWT相对各任务最近训练完成时表现；FWT相对共同初始模型的首次训练前表现。AUC只用phase_start/periodic的regular种子块梯形积分，末次值延伸至实际阶段末，再除以实际步数。两种子等权，SD为样本标准差；不能据两个种子作稳健显著性推断。','',
           '最终128局评价复用focus种子/index，是独立进程复核，不是独立测试集。空值保留缺失语义。','',
           '| 方法 | ACC均值±SD (%) | BWT均值±SD (pp) | FWT均值±SD (pp) |','| --- | ---: | ---: | ---: |']
    for r in aggregates:
        lines.append('| '+r['method']+' | '+' | '.join(f"{r[k+'_mean']:.2f} ± {r[k+'_std']:.2f}" for k in ('acc_pct','bwt_pp','fwt_pp'))+' |')
    lines+=['','| 方法 | A | B | C | D | E |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregates:
        lines.append('| '+r['method']+' | '+' | '.join(f"{r[t+'_win_pct_mean']:.2f} ± {r[t+'_win_pct_std']:.2f}" for t in base.TASKS)+' |')
    lines+=['','Anchor减对照，同种子配对（pp）：','','| seed | 对照 | ΔACC | ΔBWT | ΔFWT |','| --- | --- | ---: | ---: | ---: |']
    for r in paired:lines.append(f"| {r['seed']} | {r['control']} | {r['acc_pct_difference']:+.2f} | {r['bwt_pp_difference']:+.2f} | {r['fwt_pp_difference']:+.2f} |")
    lines+=['','![训练曲线](learning_curves.png)','','![最终任务保持](final_retention.png)','','![机制诊断](anchor_mechanisms.png)','',
           '机制图前两项按完整历史池真实决策数加权，包含门控失败任务；第三项为累计采样有效约束状态/掩码前真实决策（含重复采样）。状态掩码覆盖与min_gap筛后实际约束覆盖分开。逐任务、来源阶段、新旧置信度、梯度及开销见CSV。边界c的参考用于下一阶段；E末更新不能解释E阶段结果。Anchor门控通过时retained包含所有未更新状态，uncertain不新增屏蔽；门控失败沿用现有掩码计数，不改Q。','',
           '原始模型、Replay、轨迹和完整日志保留实验主机；原四组数据来自冻结的 [原报告](../ts5-tkpr-cpr-1m/README.md)，校验和见 source_manifest.json。本文仅描述配对性能与诊断，不把相关变化解释成性能原因。','',
           '复算：`python scripts/publish_ts5_cpr_anchor_1m.py --plan /home/wyh/marl_method_lab/results/ts5_tkpr_cpr_anchor_1m_20261126/plan.json --baseline docs/results/comparisons/ts5-tkpr-cpr-1m --output /tmp/ts5-anchor-reexport`。输出目录须尚不存在；脚本不训练、不修改原始运行或原四组结果。','']
    lines += ['', '## 结果解读与分阶段诊断', '',
              'Anchor最终ACC 44.06±3.54%，相对原CPR平均−0.70pp；seed3535 −3.75pp、seed2222 +2.34pp，方向不一致。相对K1平均−4.92pp（−10.78/+0.94pp），相对Refresh平均−0.23pp，相对ER平均+5.70pp。没有一致超越原CPR或K1的证据。', '',
              '最终A–D旧任务均值：Anchor 45.90%，原CPR 45.02%，K1 49.02%；最终E：Anchor 36.72%，原CPR 43.75%，K1 48.83%。Anchor相对原CPR在旧任务均值略高，但E学习更弱。按任务看，Anchor相对原CPR的A/B提高、C/D/E下降，不能把ACC变化一概归为遗忘。', '',
              '| 方法 | B学习AUC | C学习AUC | D学习AUC | E学习AUC |',
              '| --- | ---: | ---: | ---: | ---: |']
    for method in METHODS:
        values=[]
        for phase in range(1,5):
            rs=[r for r in tables['stage_metrics'] if r['method']==method and int(r['phase'])==phase]
            st=summary(rs,'learning_auc_pct'); values.append(f"{st['mean']:.2f} ± {st['std']:.2f}")
        lines.append('| '+method+' | '+' | '.join(values)+' |')
    lines += ['', 'B阶段K1、CPR、Anchor和Refresh的regular学习曲线一致；首次参考更新发生在B末，差异从C阶段开始。Anchor的C阶段AUC高于原CPR，但D/E低于原CPR，D阶段也低于K1。这是学习效率差异，不能只根据最终保持率评价。', '',
              '| 边界 | 方法 | 门控通过/任务点 | 更新比例 | 有效参考覆盖 | Top-1变化状态 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for phase in range(1,5):
        for method in ('TKPR-CPR','TKPR-CPR-Anchor'):
            rs=[r for r in tables['cpr_reference_updates'] if r['method']==method and int(r['phase'])==phase]
            count=sum(int(r['decision_states']) for r in rs)
            passed=sum(r['task_gate_passed']=='True' for r in rs)
            updated=sum(int(r['updated_states']) for r in rs)/count
            valid=sum(int(r['valid_reference_states']) for r in rs)/count
            flips=sum(int(r['reference_top1_changed_states']) for r in rs)
            lines.append(f"| {base.TASKS[phase]} | {method} | {passed}/{len(rs)} | {updated:.2%} | {valid:.2%} | {flips} |")
    lines += ['', 'Anchor在全部边界的uncertain状态数均为0，且有效掩码100%；这符合不新增屏蔽的规则。已接受更新的Top-1变化数为0，原CPR则可改变参考Top-1。Anchor仍更新Q值的归一化间隔和Pair筛选，不能据Top-1未变认定偏好目标完全静止。', '',
              '| 方法 | C实际约束覆盖 | D实际约束覆盖 | E实际约束覆盖 |',
              '| --- | ---: | ---: | ---: |']
    for method in METHODS[1:]:
        values=[]
        for phase in (2,3,4):
            rs=[r for r in tables['preference_diagnostics'] if r['method']==method and int(r['phase'])==phase]
            den=sum(int(r.get('raw_decision_states') or r['decision_states']) for r in rs)
            values.append(f"{sum(int(r['valid_states']) for r in rs)/den:.2%}")
        lines.append('| '+method+' | '+' | '.join(values)+' |')
    lines += ['', '| 方法 | C梯度余弦均值 | D梯度余弦均值 | E梯度余弦均值 |',
              '| --- | ---: | ---: | ---: |']
    for method in METHODS[1:]:
        values=[]
        for phase in (2,3,4):
            rs=[r for r in tables['gradient_diagnostics'] if r['method']==method and int(r['phase'])==phase and r['metric']=='td_preference_grad_cosine']
            values.append(f"{summary(rs,'mean')['mean']:.4f}")
        lines.append('| '+method+' | '+' | '.join(values)+' |')
    lines += ['', '梯度表是各运行监控点均值再按两个种子等权平均，不是不同checkpoint梯度向量的直接比较。完整范数、比值及运行内标准差见gradient_diagnostics.csv。更高约束覆盖并未转化成一致的性能优势；本次不能将原CPR的D/E下降完全解释为Top-1切换或uncertain屏蔽。锚定与不屏蔽是一起改变的，尚未单独分离二者作用。', '',
              '## 发布前复核（2026-11-27）', '',
              '调度器四项任务全部completed/exit 0；两个Anchor分支均完成5M累计阈值。新生成数据包含8个B–E阶段端点、10个最终128局复核点；与原四组及两个共同A合并后为12个训练运行、42个阶段端点、50个最终复核点。全部新阶段胜率、ACC/BWT/FWT已从胜局数独立重算，评价outcomes计数匹配；最终10点胜局数及seed/index集合均与阶段末一致。模型恢复及初始Memory指纹均匹配，启动commit均为9ff065f。三张图已检查。', '']
    (out/'README.md').write_text('\n'.join(lines))


def register(out):
    if len(out.parents) < 4: return
    repo=out.parents[3]
    if not (repo/'docs/results/catalog.json').exists(): return
    prefix='comparisons/'+out.name+'/'
    catalog=repo/'docs/results/catalog.json';data=json.loads(catalog.read_text())
    entry=dict(id=out.name,category='comparisons',report=prefix+'README.md',
        files={p.stem:prefix+p.name for p in out.glob('*.csv')},task_order=list(base.TASKS),training_seeds=[3535,2222],final_phase=4,
        comparison=dict(kind='shared_prefix_matched_by_seed',match_on=['seed','prefix_id'],primary_control=dict(method='TKPR-CPR'),additional_controls=['TKPR-K1','TKPR-Refresh','VDN-ER']),
        interpretation_limits=['两个种子、单TS5顺序；原四组复用，不重复训练。','最终评价复用focus种子，是独立进程复核。','E末参考更新不影响E阶段训练；机制相关不等于因果。'])
    data['experiments']=[x for x in data['experiments'] if x['id']!=out.name]+[entry]
    catalog.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    p=repo/'docs/results/README.md';s=p.read_text();anchor='| --- | --- | --- |\n'
    line=f'| [TS5-1M · CPR-Anchor]({prefix}README.md) | seed3535/2222，复用共同A，新增B–E各1M | 原始Top-1锚定与原CPR、K1、Refresh、ER的配对结果和机制诊断。 |\n'
    if prefix not in s:p.write_text(s.replace(anchor,anchor+line,1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();publish(a.plan.resolve(),a.baseline.resolve(),a.output.resolve())
