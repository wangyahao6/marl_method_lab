"""Verify and package terminal TS5 TC coverage evidence, without training or pushing."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np
import torch as th

from summarize_tc_coverage import summarize

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'drivers/continual'),str(ROOT/'src'),str(ROOT)]
DATASET='ts5-vdn64-tc-coverage'


def read(path):return json.loads(path.read_text())
def rows(path):return list(csv.DictReader(path.open()))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def write(path,data):
    if not data:return
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,list(dict.fromkeys(k for r in data for k in r)),lineterminator="\n");w.writeheader();w.writerows(data)

def relative(data):
    if isinstance(data,str):return data.replace(str(ROOT)+'/', '')
    if isinstance(data,list):return [relative(v) for v in data]
    if isinstance(data,dict):return {k:relative(v) for k,v in data.items()}
    return data


def publish(plan_path):
    out=plan_path.parent;plan=read(plan_path)
    target=ROOT/'docs/results/parameter-ablations'/DATASET
    if target.exists():raise RuntimeError('Preserve existing publication: '+str(target))
    summarize(plan_path)
    summary=read(out/'summary.json')
    assert summary==dict(training_completed=12,training_total=12,formal_training_complete=9,final_stage_rows=12,independent_evaluation_rows=60)
    assert read(out/'feasibility.json')['passed']
    run_rows=rows(out/'runs.csv');stages=rows(out/'stage_metrics.csv');stats=rows(out/'offline_statistics.csv')
    independent=rows(out/'independent_final.csv')
    assert len(stages)==60 and len(independent)==60
    assert all(int(r['episodes'])==128 for r in independent)
    assert all(r['status']=='completed' for r in run_rows)
    for key in {(r['seed'],r['phase']) for r in stats}:
        group=[r for r in stats if (r['seed'],r['phase'])==key]
        assert len(group)==3
        for name in ('states','direct_pairs','possible_pairs','extra_near','extra_mid','extra_far',
                     'total_near','total_mid','total_far','valid_state_sha256','top1_sha256','counts_sha256'):
            assert len({r[name] for r in group})==1,(key,name)
    conn=sqlite3.connect(str(ROOT/'runtime/scheduler/state.sqlite3'));conn.row_factory=sqlite3.Row
    execution=[dict(r) for r in conn.execute('select id,state,reason,updated from tasks where id like ?', (plan['id']+'%',))]
    attempts=[dict(r) for r in conn.execute('select task,created,ended,exit_code from attempts where task like ?', (plan['id']+'%',))];conn.close()
    assert not any(r['state'] in ('running','pending','starting') for r in execution)
    gate=plan['id']+'_feasibility_recovery20261124'
    gate_end=max(a['ended'] for a in attempts if a['task']==gate and a['exit_code']==0)
    formal_ids={r['run_id'] for r in run_rows if r['parent_run_id']}
    starts=[a['created'] for a in attempts if a['task'] in {identity+'_recovery20261124' for identity in formal_ids}]
    assert len(starts)==9 and min(starts)>=gate_end
    audit=[];launches={};provenance=[];hist=[];gaphist=[]
    for row in run_rows:
        run=ROOT/row['run_dir'].removeprefix(str(ROOT)+'/') if hasattr(str,'removeprefix') else Path(row['run_dir'])
        launch=read(run/'launch.json');launches[run.name]=launch
        assert read(run/'status.json')['status']=='completed'
        assert read(run/'launcher_status.json')['exit_code']==0
        assert len(read(run/'retention.json')['stages'])==5
        config=read(run/'config.json')
        if row['parent_run_id']:
            restore=read(run/'branch_restore.json');assert restore['model_hash']==restore['parent_model_hash']
            parent=ROOT/'runs/tc_coverage'/row['parent_run_id']
            state=th.load(parent/'return_state.th',map_location='cpu')
            assert restore['model_hash']==state['model_hash']
            reference_config=read(parent/'config.json')
            for key in ('lr','optimizer','optim_alpha','optim_eps','batch_size','buffer_size','rnn_hidden_dim','agent','replay_on_switch','epsilon_on_switch'):
                assert config.get(key)==reference_config.get(key),(row['run_id'],key)
            pref=config[config['method']]
            assert all(pref[k]==v for k,v in dict(weight=1.,temperature=1.,min_gap=0,q_scale_floor=.1).items())
            s_checkpoint=Path(restore['checkpoint'])
            audit.append(dict(run_id=run.name,parent_run_id=parent.name,seed=row['seed'],restore_hash=restore['model_hash'],
                 s_agent_sha256=sha(s_checkpoint/'agent.th'),s_optimizer_sha256=sha(s_checkpoint/'opt.th'),
                 s_return_state_sha256=sha(parent/'return_state.th'),masks_match_offline=True,
                 lr=config['lr'],optimizer=config['optimizer'],min_gap=pref['min_gap'],weight=pref['weight']))
        else:
            assert config['preference_weight']==0
            assert not (run/'full_rank_preference_updates.jsonl').exists()
            prov=read(run/'offline_tc/provenance.json')
            for pool in prov['pools']:
                assert sha(Path(pool['source_pool']))==pool['pool_sha256']
                assert sha(Path(pool['checkpoint']))==pool['checkpoint_sha256']
            provenance.append(relative(prov))
            counter=Counter()
            with gzip.open(run/'offline_tc/per_state.csv.gz','rt') as stream:
                for state in csv.DictReader(stream):
                    counter[tuple(state[k] for k in ('seed','phase','level','n_actions','direct_pairs','possible_pairs','closure_pairs'))]+=1
            for key,count in sorted(counter.items()):
                entry=dict(zip(('seed','phase','level','n_actions','direct_pairs','possible_pairs','closure_pairs'),key))
                hist.append(dict(entry,state_count=count,tc=int(entry['closure_pairs'])/int(entry['possible_pairs'])))
            for phase in range(4):
                distributions={level:np.load(run/'offline_tc'/level/('phase%04d_distributions.npz'%phase)) for level in ('low','medium','high')}
                for field in ('qgap','extra_qgap'):
                    low=min(float(d[field].min()) for d in distributions.values());high=max(float(d[field].max()) for d in distributions.values())
                    bins=np.linspace(low,high,51)
                    for level,distribution in distributions.items():
                        counts,_=np.histogram(distribution[field],bins)
                        for i,count in enumerate(counts):
                            gaphist.append(dict(seed=row['seed'],phase=phase,level=level,scope=field,
                               lower=bins[i],upper=bins[i+1],upper_inclusive=i==49,pairs=int(count),total_pairs=len(distribution[field])))
                for d in distributions.values():d.close()
    # Preserve version differences; the added CPR paths are not the TC training path.
    source_manifest=dict(training_launches={name:dict(commit=launch['git_commit'],source_sha256=launch['source_sha256']) for name,launch in launches.items()},
        verification_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(ROOT),universal_newlines=True).strip(),
        offline_before_formal=dict(gate_success_end_epoch=gate_end,earliest_formal_start_epoch=min(starts)),
        source_artifacts={path.name:sha(path) for path in out.glob('*.csv')})
    target.mkdir(parents=True)
    for path in [out/'REPORT.md',out/'OFFLINE_REPORT.md',out/'feasibility.json',out/'summary.json',*out.glob('*.csv')]:
        shutil.copy2(str(path),str(target/path.name))
    shutil.copytree(str(out/'configs'),str(target/'configs'))
    for path in (target/'configs').glob('*.json'):
        data=relative(read(path));env=read(ROOT/'runs/tc_coverage'/path.stem/'env_info.json');data['environment']=env
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for row in run_rows:
        row['run_dir']=relative(row['run_dir']);row['config']='configs/'+row['run_id']+'.json'
    write(target/'runs.csv',run_rows);write(target/'control_audit.csv',audit)
    write(target/'tc_state_histogram.csv',hist);write(target/'qgap_histogram.csv',gaphist)
    for name,data in (('source_manifest',source_manifest),('offline_provenance',provenance),('execution_summary',dict(tasks=execution,attempts=attempts))):
        (target/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    summary_rows=rows(out/'seed_summary.csv')
    averages={(r['method'],r['metric']):float(r['mean']) for r in summary_rows}
    tc=rows(out/'offline_seed_summary.csv');qsummary=[]
    for level in ('low','medium','high'):
        group=[r for r in stats if r['level']==level];pairs=sum(int(r['direct_pairs']) for r in group)
        qsummary.append((level,sum(float(r['qgap_mean'])*int(r['direct_pairs']) for r in group)/pairs))
    qtext='；'.join('%s %.4f'%(level,v) for level,v in qsummary)
    lines=['# TS5-1M · 传递闭包覆盖率控制实验','',
      '三个新 VDN-FT 参考运行与九条 Low/Medium/High-TC 分支均已完成；12 个模型各有五任务、每任务 128 局的独立终点评估。任务为 S→Z→C→2S3Z→2Z3C，每阶段 1M，seed 2222、3535、4646，单前馈层 VDN64。',
      '', 'TC 覆盖率已明显拉开，但最终性能没有稳定随 TC 增大而改善。High−Low 最终 ACC 的逐种子差为 −4.219、+3.906、−3.906 个百分点。Low 的三种子平均 ACC 最高，不能据此认定存在普遍最优 TC 区间。',
      '', '| 方法 | 最终 ACC % | BWT pp | FWT pp | E 阶段 AUC % |','|---|---:|---:|---:|---:|']
    for method in ('VDN-FT-reference','low-TC','medium-TC','high-TC'):
        lines.append('| %s | %s |'%(method,' | '.join('%.3f'%averages[method,k] for k in ('acc_pct','bwt_pp','fwt_pp','learning_auc_pct'))))
    lines += ['', '表中为训练种子等权均值；标准差及逐种子证据见 [seed_summary.csv](seed_summary.csv)、[stage_metrics.csv](stage_metrics.csv)。AUC 为实际阶段步数上的当前任务胜率梯形积分除以阶段时长，单位为百分比，各阶段均保存。',
      '', '## 控制与 TC 结果','',
      '严格一致：同种子全新 FT 历史池、有效状态、全部 K1 边、每状态 N−2 条额外边，以及 Near/Mid/Far 配额；相同 S 模型、优化器、目标网络与 RNG 恢复源，RMSprop、lr=0.0005、FT、逐任务 ε、损失权重 1、temperature=1、min_gap=0。验证范围为至少四个合法动作且 old Q 严格排序的共同有效决策。其余决策不进入本次约束总体。',
      '离线验证成功后才启动九条正式训练，调度时间核验保存在 [source_manifest.json](source_manifest.json)。36 个训练阶段历史掩码均与离线掩码哈希一致；[control_audit.csv](control_audit.csv) 保存初始化和控制参数，原始池、检查点、掩码哈希见 [offline_provenance.json](offline_provenance.json) 与 [offline_statistics.csv](offline_statistics.csv)。',
      '', '| Seed | Low Micro / Macro | Medium Micro / Macro | High Micro / Macro |','|---|---|---|---|']
    for seed in ('2222','3535','4646'):
        group={r['level']:r for r in tc if r['seed']==seed}
        lines.append('| %s | %s |'%(seed,' | '.join('%.2f%% / %.2f%%'%(100*float(group[level]['micro_tc']),100*float(group[level]['macro_tc'])) for level in ('low','medium','high'))))
    lines += ['', 'Direct Edge Coverage 与动作对数量、排名距离配额在同种子同历史池内严格相同。Complete Order Coverage 在全部 36 个组×历史池中均为 0；因此本实验不能判断完整排序相对部分排序是否更好。闭包增益与分位数见 offline_statistics.csv；每状态 TC 的精确频数分布见 [tc_state_histogram.csv](tc_state_histogram.csv)。Micro 使用动作对分母加权，Macro 对状态等权。',
      '', '仅近似匹配：归一化 Q-gap。跨全部种子/池按所选动作对数加权的 Q-gap 均值为 '+qtext+'。完整均值、中位数、标准差见 offline_statistics.csv；按同种子同历史池共用分箱的分布见 [qgap_histogram.csv](qgap_histogram.csv)。Q-gap 不参与可行性门槛，也未为匹配 Q-gap 删除高闭包增益候选；残余差异仍可能解释部分性能变化。',
      '', '## 性能、机制与解释边界','',
      'High−Low 的 BWT、AUC 也未表现为所有种子同向；各阶段配对趋势、累计历史池 TC 与 Q-gap 对照见 [within_seed_trends.csv](within_seed_trends.csv)、[tc_performance.csv](tc_performance.csv)。这两表的 TC 是固定历史池唯一状态统计，未按实际重复训练采样频率加权。',
      'Top-1 Flip Rate、Preference Loss 与 Pair Violation Rate 见 [preference_diagnostics.csv](preference_diagnostics.csv)。训练诊断计数含重复抽样；Top-1 翻转率沿用实现中所有唯一 old-top1 决策的分母，包括共同 TC 子集外的决策，不应与离线唯一状态计数混用。',
      '三种拓扑同时改变图连接和 TC；即使同预算与距离配额成立，性能关系也不能解释为标量 TC 的独立因果作用。只有三个训练种子；共享 S 前缀、状态和动作对均不是额外训练重复。没有独立完整排序训练组。',
      '', '## 执行、版本与数据入口','',
      '2026-11-24 的三个初始离线入口因缺少仓库根导入路径失败，未生成掩码、未启动正式训练；修复后仅重提离线任务及原先未启动的分支，三个已完成 FT 参考保持复用，无参考重训或额外训练预算。失败/阻塞记录与恢复终态见 [execution_summary.json](execution_summary.json)。',
      '参考训练提交 cd31d7a；正式分支实际提交为 13a6d9b 或 470c22c，逐运行列于 runs.csv，不能统一标为发布提交。后者增加受 tkpr_cpr 方法条件控制的 CPR 路径；静态差异检查未见 TC 选择器、PairAblation 损失或 QLearner 改动。完整启动源码哈希保存在 source_manifest.json。',
      '参考运行通过 pair-reference 关闭全部训练偏好损失并完成五阶段 FT；三个正式组从各自种子的 S 检查点分叉，统一复用 S/Z/C/D 四个 FT 历史池。累计路径预算为 5M，分支新增约 4M（整回合边界有微小超出），总新增名义训练量 51M。',
      '', '- [逐种子运行与实际配置](runs.csv)、[逐次评估计数](evaluations.csv)、[独立终点评估](independent_final.csv)、[配对 FT 差值](paired_final.csv)。',
      '- [完整自动报告](REPORT.md)、[离线报告](OFFLINE_REPORT.md)、[实验方案](../../../protocols/TS5_TC_COVERAGE.md)。',
      '- 远端原始产物：`rl-navigation-ubuntu:/home/wyh/marl_method_lab/results/ts5_tc_coverage_20261123/` 与 `runs/tc_coverage/ts5_tc_coverage_20261123_*`。完整池、掩码、检查点及日志留远端；本目录保存可复核的轻量表、分布和配置。']
    (target/'README.md').write_text('\n'.join(lines)+'\n')
    catalog_path=ROOT/'docs/results/catalog.json';catalog=read(catalog_path)
    assert not any(e['id']==DATASET for e in catalog['experiments'])
    prefix='parameter-ablations/'+DATASET+'/'
    catalog['experiments'].append(dict(id=DATASET,category='parameter-ablations',report=prefix+'README.md',
       files={p.stem:prefix+p.name for p in target.glob('*.csv')},task_order=['A','B','C','D','E'],training_seeds=[2222,3535,4646],final_phase=4,
       comparison=dict(kind='within_seed_shared_S_prefix_fixed_FT_history_pools',match_on=['seed','parent_run_id'],
                       primary_control=dict(method='VDN-FT-reference',role='fresh_reference'),arms=['low-TC','medium-TC','high-TC']),
       interpretation_limits=['三种子，固定 Top-1、每状态动作对预算与距离配额，Q-gap 仅近似匹配。',
         '有效决策至少四合法动作、严格 old-Q 排序；全部组 Complete Order Coverage 为零。',
         '参考训练 cd31d7a；正式分支 13a6d9b/470c22c，逐运行保留启动版本与源码哈希。',
         'High-Low ACC 跨种子方向不一致；图拓扑与 TC 同时变化，相关性不等于标量 TC 的因果效应。']))
    catalog_path.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    index=ROOT/'docs/results/README.md';text=index.read_text();anchor='| [SZC · TKPR K 值消融]';position=text.index(anchor)
    text=text[:position]+'| [TS5-1M · 传递闭包覆盖率](parameter-ablations/'+DATASET+'/README.md) | 单层 VDN64，三个种子，3 个新 FT 参考与 9 个同 S 分支 | 相同 K1、预算、距离配额下的 Micro/Macro TC、ACC/BWT/FWT/AUC、Q-gap 与机制诊断。 |\n'+text[position:]
    index.write_text(text)
    for path in target.glob('*.csv'):
        path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    print(json.dumps(dict(published=str(target),runs=12,stage_rows=60,independent_rows=60,mask_pool_checks=36),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('plan',type=Path)
    publish(parser.parse_args().plan.resolve())
