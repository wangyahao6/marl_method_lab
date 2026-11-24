"""Stream four gzip JSONL logs; retain selected numeric scalars, never complete log rows.
Usage: python analyze_logs.py --input-root PRIVATE_INPUT_ROOT --results-root RESULT_DIR
Requires Python 3 and NumPy. Inputs are read-only; outputs go to RESULT_DIR/analysis.
"""
import argparse
from array import array
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

GRAD = ('preference_grad_norm_agent','td_grad_norm_agent','preference_td_grad_ratio','td_preference_grad_cosine')
TIMING = ('preference_prepare_seconds','learner_update_seconds')
METHODS = {'tkpr-k1': 'topk_preference_updates.jsonl.gz', 'tkpr-refresh': 'tkpr_refresh_updates.jsonl.gz'}

def dump_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n'); writer.writeheader();writer.writerows(rows)

def stat(values, total):
    a=np.asarray(values,dtype=float)
    result=dict(n_records=total,n_valid=len(a),n_missing=total-len(a))
    if len(a):
        p10,p50,p90,p95=np.percentile(a,[10,50,90,95])
        result.update(mean=float(a.mean()),median=float(p50),p10=float(p10),p90=float(p90),p50=float(p50),p95=float(p95),sum=float(a.sum()),minimum=float(a.min()),maximum=float(a.max()),negative_fraction=float((a<0).mean()))
    return result

class Scalars:
    def __init__(self): self.data=defaultdict(lambda:array('d'));self.count=Counter()
    def add(self,key,value):
        self.count[key]+=1
        if isinstance(value,(int,float)) and math.isfinite(value):self.data[key].append(value)
    def stats(self,key):return stat(self.data[key],self.count[key])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--results-root',type=Path,required=True)
    args=parser.parse_args(); out=args.results_root/'analysis';out.mkdir(exist_ok=True)
    expected={x['file']:x for x in json.loads((out/'input_manifest.json').read_text())}
    inventory={}; gradients=Scalars();timings=Scalars();aux=defaultdict(lambda:dict(updates=0,sampled=0,participating=0,counts=Counter(),pools=Counter()))
    bounds={}; run_ids={}
    for r in csv.DictReader((args.results_root/'stage_metrics.csv').open()):
        if r['method'] in ('TKPR-K1','TKPR-Refresh'):
            method='tkpr-k1' if r['method']=='TKPR-K1' else 'tkpr-refresh'
            bounds[method,int(r['phase'])]=int(r['t_env']);run_ids[method]=r['run_id']
    def segments(method,phase,t):
        start=bounds.get((method,phase-1),0);end=bounds[method,phase]
        q=min(3,max(0,int(4*(t-start)/(end-start))))
        return ('all',f'q{q+1}')
    def stream(method,name):
        key=method+'/'+name;path=args.input_root/key; raw_hash=hashlib.sha256();n=0
        fields=Counter();metrics=Counter();events=Counter();nested=Counter()
        with gzip.open(path,'rb') as f:
            for line in f:
                n+=1;raw_hash.update(line);r=json.loads(line)
                fields.update(r.keys())
                if 'metric' in r:
                    metrics[r['metric']]+=1;events[r.get('context',{}).get('event','absent')]+=1
                for h in r.get('historical_tasks',[]):nested.update(h.keys())
                yield r
        e=expected[key]
        if raw_hash.hexdigest()!=e['original_sha256'] or n!=e['lines']:raise ValueError(f'input changed: {key}')
        inventory[key]=dict(lines=n,original_sha256=raw_hash.hexdigest(),fields=dict(fields),metrics=dict(metrics),events=dict(events),historical_task_fields=dict(nested))
    for method,update_name in METHODS.items():
        monitors={};monitor_ids=defaultdict(set);monitor_intervals=Counter()
        # Pass 1: scalar event rows. Loss/timing are NOT assumed to be present here.
        for r in stream(method,'experiment_metrics.jsonl.gz'):
            if r.get('context',{}).get('event')!='gradient_monitor' or r.get('metric') not in GRAD:continue
            phase=r['phase'];update=r['context']['learner_update'];metric=r['metric']
            if update % 100:raise ValueError('unexpected gradient recording interval')
            if (phase,update,metric) in monitors:raise ValueError('duplicate gradient event')
            monitors[phase,update,metric]=r.get('value');monitor_ids[phase].add(update)
            for seg in segments(method,phase,r['t_env']):gradients.add((method,phase,seg,'gradient_monitor',metric),r.get('value'))
        for phase,ids in monitor_ids.items():
            ids=sorted(ids);monitor_intervals.update(b-a for a,b in zip(ids,ids[1:]))
        inventory[method+'/gradient_update_intervals']=dict(monitor_intervals)
        seen=set()
        # Pass 2: one record per learner update, including A's zero-history updates.
        for r in stream(method,update_name):
            phase=r['phase'];update=r['learner_update'];seen.add(update)
            monitored=update in monitor_ids[phase]
            for seg in segments(method,phase,r['t_env']):
                gradients.add((method,phase,seg,'all_updates','preference_loss'),r.get('preference_loss'))
                if monitored:gradients.add((method,phase,seg,'gradient_monitor','preference_loss'),r.get('preference_loss'))
                for scope in ('all_updates', 'gradient_monitor' if monitored else 'without_gradient_monitor'):
                    for metric in TIMING:timings.add((method,phase,seg,scope,metric),r.get(metric))
                key=method,phase,seg,'ALL';g=aux[key];g['updates']+=1;g['participating']+=int(r.get('preference_participated_update',False))
                histories=r.get('historical_tasks',[]);count=sum(h['sampled_episodes'] for h in histories)
                if count!=r['historical_episode_count']:raise ValueError('sample count mismatch')
                g['sampled']+=count;g['counts'][count]+=1
                for h in histories:
                    key=method,phase,seg,h['task'];g=aux[key];g['updates']+=1;g['sampled']+=h['sampled_episodes'];g['counts'][h['sampled_episodes']]+=1;g['pools'][h['pool_size']]+=1
            if monitored:
                if any((phase,update,m) not in monitors for m in GRAD):raise ValueError('incomplete gradient sample')
        if any(u not in seen for _,u,_ in monitors):raise ValueError('gradient sample without update record')
        print(method,'streamed and verified',flush=True)
    grows=[];trows=[];arows=[]
    for key in sorted(gradients.count):
        method,phase,segment,scope,metric=key
        grows.append(dict(method=method,phase=phase,task='ABCDE'[phase],segment=segment,scope=scope,metric=metric,**gradients.stats(key)))
    for key in sorted(timings.count):
        method,phase,segment,scope,metric=key;s=timings.stats(key)
        trows.append(dict(method=method,phase=phase,task='ABCDE'[phase],segment=segment,scope=scope,metric=metric,unit='seconds',**s,total_recorded_seconds=s.get('sum')))
    # Attach phase/segment same-scope mean ratios, not causal runtime estimates.
    lookup={(r['method'],r['phase'],r['segment'],r['scope'],r['metric']):r for r in trows}
    for r in trows:
        base=lookup.get(('tkpr-k1',r['phase'],r['segment'],r['scope'],r['metric']),{})
        if r['method']=='tkpr-refresh' and base.get('mean'):
            r['refresh_over_k1_mean_ratio']=r['mean']/base['mean'];r['refresh_minus_k1_mean_seconds']=r['mean']-base['mean']
            r['refresh_minus_k1_recorded_seconds']=r['sum']-base['sum']
    for key,g in sorted(aux.items()):
        method,phase,segment,source=key;total=aux[method,phase,segment,'ALL']
        arows.append(dict(row_type='sampling',method=method,phase=phase,task='ABCDE'[phase],segment=segment,historical_task=source,recorded_updates=g['updates'],participating_updates=g['participating'] if source=='ALL' else '',sampled_episodes_total=g['sampled'],mean_episodes_per_record=g['sampled']/g['updates'],sample_share=g['sampled']/total['sampled'] if total['sampled'] else None,sample_count_histogram=json.dumps(g['counts'],sort_keys=True),pool_size_histogram=json.dumps(g['pools'],sort_keys=True)))
    # Four supplied gzip files do not contain stage-end pool replacement events.
    # Optional existing published companion file is explicitly labelled external.
    if (args.results_root/'memory_updates.csv').exists():
        for r in csv.DictReader((args.results_root/'memory_updates.csv').open()):
            arows.append(dict(row_type='memory_refresh_external_companion',method='tkpr-refresh',phase=int(r['update_phase']),task='ABCDE'[int(r['update_phase'])],segment='stage_end',historical_task=r['task'],old_count=int(r['old_count']),new_count=int(r['new_count']),replaced=int(r['replaced']),final_size=int(r['final_size']),pool_delta=int(r['final_size'])-int(r['old_count']),source='memory_updates.csv (not one of four gzip inputs)'))
    dump_csv(out/'gradient_metrics_by_phase.csv',grows);dump_csv(out/'timing_metrics_by_phase.csv',trows);dump_csv(out/'aux_updates_by_phase.csv',arows)
    (out/'field_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2)+'\n')
    def grad(method,phase,metric,seg='all',scope='gradient_monitor'):
        return next(r for r in grows if (r['method'],r['phase'],r['segment'],r['scope'],r['metric'])==(method,phase,seg,scope,metric))
    def timing(method,phase,metric,scope='all_updates'):
        return lookup[method,phase,'all',scope,metric]
    notes=['# TKPR-K1 与 TKPR-Refresh：按阶段日志分析','',
    '输入为四份 gzip JSONL，逐行 `gzip.open(..., "rb")` 读取并校验原始 SHA-256/行数；不加载整份日志，不修改输入。内存仅保留所选标量的紧凑 double 数组、梯度事件索引与累计计数；精确分位数使用 NumPy percentile 默认线性插值。原始压缩日志仅保留于实验主机；GitHub 当前树撤下日志，旧上传提交仍留在 Git 历史中。','',
    '## 字段、计算口径与记录频率','',
    '实际字段与出现次数见 field_inventory.json。两方法 experiment_metrics 中均没有 preference_loss、preference_prepare_seconds、learner_update_seconds 这三个 metric；它们在 updates 宽表中逐更新存在。四项梯度指标均取自 gradient_monitor 事件。experiment_metrics 是 metric/value/context 长表；偏好 updates 是每次 learner 更新的一行宽表。两者不是同一粒度，不能将长表各行当作独立更新或重复汇总。',
    '训练提交为 9db0ed1951ded62c00eb1ff27a80c71ae5465dc5。代码定位：src/learners/q_learner.py 的 agent_gradient_metrics、QLearner.train；src/learners/action_set_preference.py 的 __call__、reduce_task_losses、record；drivers/continual/common/metrics.py 的 gradients、preference；drivers/continual/action_set_train.py 的 monitor_interval 与 record_preference。',
    'preference_loss 从 updates 每次读取，是未乘 weight 的任务等权平均排序损失；只对有有效约束的历史任务平均。两方法 weight 均为 1，损失算法和标准化相同，但轨迹来源与采样预算不同，绝对损失不是统一测试集上的质量评估。A 无历史，记录 loss=0 但偏好梯度为 null，不能把 null 当零。',
    '梯度事件每 100 个全局 learner 更新记录一次（实际间距清单见 inventory），相位内不是每 100 环境步。Agent 可训练参数梯度按 float64 展平取 L2 范数；preference_grad_norm_agent 对 weight*preference_loss 求导，td_grad_norm_agent 对 masked TD MSE 求导。均在联合反传、梯度裁剪和 optimizer.step 之前计算。ratio 为偏好范数/TD 范数；cosine 为二者点积除范数乘积，负值表示局部方向冲突。缺损失、零范数等情形可为 null，CSV 分开记录 n_records/n_valid/n_missing。',
    'gradient CSV 同时提供 all_updates 的 preference_loss 和 gradient_monitor 子样本的五项指标，后者通过 phase/learner_update 对齐，便于统一记录频率比较。每阶段汇总加 q1–q4（按该运行实际训练阶段起止环境步的 0–25/25–50/50–75/75–100% 划分）；不是按记录数分箱。分段权重和有效采样数可不同，不将分段平均再平均当成阶段平均。',
    '', '## 计时解释','',
    'preference_prepare_seconds 从 ActionSetPreference.__call__ 开始，到采样、旧轨迹组批、各历史任务前向和损失汇总完成为止的 perf_counter 差。无显式 CUDA 同步，部分 GPU 工作异步；它是 CPU 侧实测经过时间，不能解释为纯 GPU 执行耗时。另有 preference_forward_cuda_seconds，但本次请求的两个指标不可互相替代。',
    'learner_update_seconds 从 QLearner.train 入口 begin_update 到 record 中汇总张量搬 CPU 后计时；包含 TD/偏好前向、联合反传、裁剪、优化、每100次的梯度诊断及其日志、部分同步和主机等待。排除 TD buffer 采样、runner 环境交互、阶段评价/采集刷新，以及 record 之后的偏好JSON写出和 target 更新。两计时互有包含，不能相加。',
    'timing CSV 提供 all_updates、gradient_monitor、without_gradient_monitor 三种范围。total_recorded_seconds 是该范围内实测经过时间的和，不是完整训练墙钟，也不是纯偏好新增成本。Refresh/K1 比值是独立运行记录的描述性比较；更新次数、轨迹长度、并发负载等均可影响计时。四份日志缺少阶段末刷新采集耗时，不能据此估算完整 Refresh 总开销。',
    '', '## 分阶段梯度与辅助更新','',
    '| 阶段 | 方法 | 全部记录更新 | 实际参与偏好更新 | 每更新历史样本 | 梯度采样有效n | 偏好梯度均值 | TD梯度均值 | ratio中位数 | cosine均值 | cosine<0比例 |',
    '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for phase in range(5):
        for method in METHODS:
            a=aux[method,phase,'all','ALL'];g=grad(method,phase,'preference_grad_norm_agent');td=grad(method,phase,'td_grad_norm_agent');ratio=grad(method,phase,'preference_td_grad_ratio');co=grad(method,phase,'td_preference_grad_cosine')
            fmt=lambda v:'N/A' if v is None else f'{v:.4g}'
            notes.append(f"| {'ABCDE'[phase]} | {method} | {a['updates']} | {a['participating']} | {a['sampled']/a['updates']:.2f} | {g['n_valid']} | {fmt(g.get('mean'))} | {fmt(td.get('mean'))} | {fmt(ratio.get('median'))} | {fmt(co.get('mean'))} | {fmt(co.get('negative_fraction'))} |")
    notes+=['','## C、D、E 的计时与阶段内变化','', '| 阶段 | prepare K1均值ms | Refresh均值ms | 比值 | update K1均值ms | Refresh均值ms | 比值 | K1记录总秒 | Refresh记录总秒 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for phase in (2,3,4):
        a=timing('tkpr-k1',phase,TIMING[0]);b=timing('tkpr-refresh',phase,TIMING[0]);c=timing('tkpr-k1',phase,TIMING[1]);d=timing('tkpr-refresh',phase,TIMING[1])
        notes.append(f"| {'ABCDE'[phase]} | {a['mean']*1000:.3f} | {b['mean']*1000:.3f} | {b['mean']/a['mean']:.3f} | {c['mean']*1000:.3f} | {d['mean']*1000:.3f} | {d['mean']/c['mean']:.3f} | {c['sum']:.1f} | {d['sum']:.1f} |")
    for phase in (2,3,4):
        notes+=['',f"### {'ABCDE'[phase]} 阶段：按实际环境步四等分",'', '| 方法 | 区间 | loss均值（监控对齐） | 偏好梯度均值 | ratio中位数 | cosine均值 |', '| --- | --- | ---: | ---: | ---: | ---: |']
        for method in METHODS:
            for seg in ('q1','q2','q3','q4'):
                vals=[grad(method,phase,m,seg).get(k) for m,k in [('preference_loss','mean'),('preference_grad_norm_agent','mean'),('preference_td_grad_ratio','median'),('td_preference_grad_cosine','mean')]]
                notes.append('| '+method+' | '+seg+' | '+' | '.join('N/A' if x is None else f'{x:.5g}' for x in vals)+' |')
    notes+=['', '## 对 C、D、E 的解释', '',
    'C 阶段两方法偏好梯度与 loss 从 q1 到 q4 总体下降；Refresh 的偏好梯度均值略低，方向冲突比例也更低，此时 ACC 较高。但它使用已刷新的 A/B 参考轨迹和更大的样本预算，不能将较高 loss 直接解释为更差的排序保留。',
    'D 阶段两方法偏好梯度均值几乎相同，而 Refresh 的 TD 梯度均值更大，使 ratio 中位数降低。阶段内 K1 的偏好梯度由早段下降后趋稳，Refresh 在 q2 后回升；Refresh 的 loss 从 q1 到 q4 下降，但始终高于 K1。更高的 cosine 与更少的负 cosine 并没有对应更高的阶段末 ACC，说明局部梯度一致性不足以解释保留效果。',
    'E 阶段两方法偏好梯度均值仍接近，Refresh 的 TD 梯度更大，ratio 中位数更低；Refresh 的 cosine 平均更高、冲突比例更低。两方法 loss 均随阶段推进下降，偏好梯度在后段仍可回升。Refresh 的最终 ACC 仍低于 K1，不能由这些观测认定强约束、弱约束或刷新本身是原因。',
    'C/D/E 的每任务采样分别为：K1 4、约2.67、2条；Refresh 一律8条。两方法来源任务的样本占比都约等权（C各50%、D各约1/3、E各25%），采样时池大小均为100。变化主要在总样本量和池内内容，日志没有呈现某个历史任务获得明显更高采样份额。',
    '额外开销随历史任务增多而扩大，但总样本量2/3/4倍并不意味着更新耗时2/3/4倍。损失本来按任务平均、每任务有更多样本也不意味着梯度范数按倍数增长；这里报告的是实测分布与比值，不作线性开销外推。', '',
    '代码证据采用实际训练版本：[梯度计算](https://github.com/wangyahao6/marl_method_lab/blob/9db0ed1951ded62c00eb1ff27a80c71ae5465dc5/src/learners/q_learner.py#L33)、[记录与计时](https://github.com/wangyahao6/marl_method_lab/blob/9db0ed1951ded62c00eb1ff27a80c71ae5465dc5/src/learners/action_set_preference.py#L183)、[每100次更新采样](https://github.com/wangyahao6/marl_method_lab/blob/9db0ed1951ded62c00eb1ff27a80c71ae5465dc5/drivers/continual/action_set_train.py#L674)。', '']
    notes+=['','## 样本与记忆池边界','',
    'aux CSV 的 row_type=sampling：historical_task=ALL 表示更新总预算；其他行分别表示各来源任务。recorded_updates 对 ALL 是阶段 learner 更新次数，对单来源是该任务被采样的更新次数；participating_updates 只在 ALL 有定义。sample_count_histogram 记录每更新样本数分布，pool_size_histogram 记录被使用池的容量分布，sample_share 按实际样本出现次数计数（含重复采样），不是池内独有轨迹比例。',
    '四份 gzip 日志只包含被采池大小和采样episode_ids，没有阶段末新增/淘汰事件，也没有逐轨迹策略来源。不能用相同episode_id推断跨刷新轨迹身份。补充的 memory_refresh_external_companion 行明确来自现有 memory_updates.csv：每池首次100条、其后每次新增50/淘汰50/最终100，新增阶段池净增100，刷新阶段池净增0。此项不是从四份gzip推断出的数据。',
    '所有数值为日志实测值或由其精确聚合/比值派生；没有外推总训练时间。阶段端点已公布：C ACC K1 61.98%/Refresh 65.63%，D 57.81%/49.22%，E 60.78%/54.84%。梯度与计时变化只能辅助描述，Refresh同时改变预算及参考轨迹，单种子日志不建立性能差异的因果解释。',
    '', '## 复现','',
    '依赖 Python 3 + NumPy。从仓库根目录执行：', '', '```sh', 'python docs/results/comparisons/ts5-tkpr-refresh/analysis/analyze_logs.py \\', '  --input-root /home/wyh/marl_method_lab/results/ts5_tkpr_refresh_3535_20261122/analysis_inputs \\', '  --results-root docs/results/comparisons/ts5-tkpr-refresh', '```', '',
    '输入子路径与校验和见 input_manifest.json。输出三份 CSV、field_inventory.json 和本文；原始输入留在实验主机。当前脚本无训练、模型修改或原始日志写入操作。','']
    (out/'analysis_notes.md').write_text('\n'.join(notes))
    print('Wrote',len(grows),'gradient rows,',len(trows),'timing rows,',len(arows),'aux rows',flush=True)
if __name__=='__main__':main()
