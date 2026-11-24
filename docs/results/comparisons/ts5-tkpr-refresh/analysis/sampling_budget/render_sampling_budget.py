# -*- coding: utf-8 -*-
"""Render reproducible sampling-budget figures and report from the analysis CSVs."""
import csv
import inspect
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
BUDGETS=('K1','Refresh-2','Refresh-4','Refresh-8')
def rows(name):return list(csv.DictReader((P/name).open()))
raw=rows('sampling_budget.csv');summary=rows('sampling_budget_summary.csv');pairs=rows('paired_directions.csv')
refs=rows('full_pool_reference.csv');checks=rows('verification_checks.csv');manifest=json.loads((P/'input_manifest.json').read_text())
def select(ph,b,scope='ALL'):return next(x for x in summary if int(x['checkpoint_phase'])==ph and x['budget']==b and x['scope']==scope)
for metric,filename,title,ylabel in [('cosine_full','gradient_cosine.png','Direction relative to fixed full-pool gradient','Cosine similarity'),('grad_norm','gradient_norm.png','Sampled auxiliary gradient norm','Agent gradient L2 norm')]:
 fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
 for ax,ph in zip(axs,(2,3,4)):
  data=[[float(x[metric]) for x in raw if int(x['checkpoint_phase'])==ph and x['budget']==b and x['scope']=='ALL'] for b in BUDGETS]
  label_key='tick_labels' if 'tick_labels' in inspect.signature(ax.boxplot).parameters else 'labels'
  ax.boxplot(data,**{label_key:['K1','R2','R4','R8']},showmeans=True)
  ax.set(title=f'Checkpoint {"ABCDE"[ph]} / {ph} historical tasks',ylabel=ylabel)
  if metric=='cosine_full':ax.set_ylim(-.05,1.05);ax.axhline(1,color='grey',linestyle=':',linewidth=1)
  else:
   ref=float(next(x for x in refs if int(x['checkpoint_phase'])==ph and x['scope']=='ALL')['grad_norm'])
   ax.axhline(ref,color='grey',linestyle='--',label='Full-pool norm');ax.legend(fontsize=8)
 fig.suptitle(title+' (30 draws; fixed parameters and static K1 memory)',fontsize=12)
 fig.savefig(P/filename,dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
for ax,ph in zip(axs,(2,3,4)):
 x=np.arange(4);ss=[select(ph,b) for b in BUDGETS]
 for shift,key,label in [(-.18,'state_coverage_of_full','Effective decision coverage'),(.18,'pair_coverage_of_full','Effective pair coverage')]:
  means=[100*float(r[key+'_mean']) for r in ss];err=[100*float(r[key+'_std']) for r in ss]
  ax.bar(x+shift,means,.36,yerr=err,capsize=2,label=label)
 ax.set(xticks=x,xticklabels=['K1','R2','R4','R8'],ylabel='% of full-pool valid units',title=f'Checkpoint {"ABCDE"[ph]}')
axs[0].legend(fontsize=8);fig.suptitle('Per-draw effective coverage: mean +/- sample SD, 30 draws',fontsize=12)
fig.savefig(P/'coverage.png',dpi=160);plt.close(fig)
L=['# TS5 TKPR 采样预算离线验证','', '**固定同一模型和原始静态历史池后，增加每任务采样量在本实验中降低了梯度向量方差，并改善平均方向；但每任务8条仍不是全量梯度的高精度替代。**','', '## 实验设计与兼容性','', 'TS5 seed3535，使用 TKPR-K1 原始运行 `ts5_3535_tkpr_k1_20261122_attempt001`。选 C、D、E 阶段末模型，分别只回放此前 A/B、A/B/C、A/B/C/D 历史任务，排除当前任务。每任务100条静态成功轨迹，原始引用 checkpoint、完整序列及 old_q_values 均兼容；未使用 Refresh 已覆盖的历史池，也未混入轨迹刷新效果。','', '每个 checkpoint 内模型、历史池、TKPR-K1配置完全固定：weight=1、temperature=1、min_gap=0.5、q_scale_floor=0.1、k=1。只改变采样预算：K1调用 PreferenceHistory.sample，跨所有历史任务合计8条且按原实现随机均分；R2/R4/R8每任务分别2/4/8条。每方案30次；固定分析种子86431（不是新增训练种子），不同阶段使用独立SeedSequence；K1与Refresh预算用独立随机流。每次重复、每任务生成一个无放回8条清单，R2是前2条、R4前4条、R8全部8条，严格嵌套。各原池100条，无需有放回。','', 'C阶段K1与R4同为每任务4条，E阶段K1与R2同为每任务2条，属于相同采样分布；独立随机流下30次样本的均值仍会不同，不把这种差异解释成方法效果。D阶段K1随机分配3/3/2条。全部保存 episode_ids，方便核对嵌套与分配。','', '来源路径、SHA-256、分析基版本与运行版本、随机种子、依赖版本和可训练Agent参数列表见 [input_manifest.json](input_manifest.json)。模型路径分别为原运行 `models/phase0002_6000067/agent.th`、`phase0003_8000011/agent.th`、`phase0004_10000031/agent.th`；Memory为 `action_set_memory/phase0000` 到 `phase0003/success.th`。训练版本为 `9db0ed1951ded62c00eb1ff27a80c71ae5465dc5`，分析基版本为 `'+manifest['analysis_base_commit']+'`。','', '分析在本机CPU、两线程、PyTorch '+manifest['torch']+'、NumPy '+manifest['numpy']+' 上完成；输入副本与远端原件SHA-256核对。没有环境实例化、在线采样、optimizer.step或输入文件写入。360次联合梯度检查和参数不变检查全部通过，结束后再次验证输入文件哈希不变。','', '## 损失、参考梯度与统计定义','', '- **每任务全量参考**：直接调用仓库 RankingPreference.task_loss，一次使用该任务全部100条完整序列与存储old_q_values；每个模型独立从零恢复RNN隐状态，保留历史动作输入。合法动作、alive/filled/terminated过滤、Q归一化、温度、min_gap和状态内Pair均值沿用真实实现。', '- **跨任务聚合**：调用原 ActionSetPreference.reduce_task_losses，先各任务有效状态平均，再对有约束任务等权平均。全量参考先分别计算全池任务梯度，再使用原归约函数的自动微分权重合成；没有合并所有任务Pair平均。采样时直接对原归约函数返回的联合损失求导，并核对它等于任务梯度的有效任务加权和。', '- **参数范围**：与 preference_grad_norm_agent 一致，为 agent.parameters() 中requires_grad参数；autograd.grad计算，未用optimizer。梯度向量转float64后求范数、距离和余弦。', '- **相对误差**：||g_sample−g_full||₂/||g_full||₂。余弦是同一checkpoint、同一scope的采样梯度与全量梯度比较，绝不混合不同参数位置。', '- **向量方差**：trace(Cov(g)) = Σ||g_r−mean(g)||²/(30−1)；normalized_gradient_variance_trace再除以||g_full||²。另给相对偏差||mean(g)−g_full||/||g_full||和相对误差平方均值；这比仅看范数SD更能判断估计稳定性。范数的标准差不必随预算单调降低；这里的相对偏差仅是30次经验均值相对参考的偏移，不能当成已证明的总体估计偏差。', '- **覆盖**：有效状态指至少一个有效TKPR Pair的智能体决策，有效Pair为这些决策内通过合法动作、Top-K及min_gap筛选的Pair计数。state/pair_coverage_of_full是单次采样有效单位数/同任务（或全历史任务）全池有效单位数。每任务无放回，单次没有重复轨迹计数；不是30次采样并集覆盖率或环境空间覆盖率。', '- **任务参与**：单任务有有效状态记1，否则0；ALL为当次有约束任务比例。no_valid_constraint_fraction为其补值。每个任务独立行与汇总均保存，不用ALL行替代任务差异。所有SD使用样本SD（ddof=1）。','', '## 联合辅助梯度结果','', '| checkpoint | 预算 | 总轨迹数 | 范数均值±SD | 对全量cosine均值±SD | 相对误差均值 | 归一化向量方差 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for ph in (2,3,4):
 for b in BUDGETS:
  r=select(ph,b)
  L.append(f'| {"ABCDE"[ph]} | {b} | {float(r["sampled_episodes_mean"]):.0f} | {float(r["grad_norm_mean"]):.4f} ± {float(r["grad_norm_std"]):.4f} | {float(r["cosine_full_mean"]):.3f} ± {float(r["cosine_full_std"]):.3f} | {float(r["relative_gradient_error_mean"]):.3f} | {float(r["normalized_gradient_variance_trace"]):.3f} |')
L+=['','![梯度方向](gradient_cosine.png)','','![梯度范数](gradient_norm.png)','','## 有效约束覆盖与任务参与','','| checkpoint | 预算 | 有效状态均值 | 有效Pair均值 | 状态全池覆盖率 | Pair全池覆盖率 | 有效任务参与率 | 无有效约束任务比例 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for ph in (2,3,4):
 for b in BUDGETS:
  r=select(ph,b);L.append(f'| {"ABCDE"[ph]} | {b} | {float(r["valid_states_mean"]):.1f} | {float(r["valid_pairs_mean"]):.1f} | {float(r["state_coverage_of_full_mean"]):.2%} | {float(r["pair_coverage_of_full_mean"]):.2%} | {float(r["task_participation_mean"]):.0%} | {float(r["no_valid_constraint_fraction_mean"]):.0%} |')
L+=['','![有效覆盖](coverage.png)','','## 保留各历史任务的独立结果','','下表给K1与R8各历史任务的方向和误差对照；所有四档预算的任务级范数、覆盖及参与率见 sampling_budget_summary.csv。','','| checkpoint | 历史任务 | K1 cosine | R8 cosine | K1相对误差 | R8相对误差 |','| --- | --- | ---: | ---: | ---: | ---: |']
for ph in (2,3,4):
 for task in 'ABCD'[:ph]:
  x=select(ph,'K1',task);y=select(ph,'Refresh-8',task)
  L.append(f'| {"ABCDE"[ph]} | {task} | {float(x["cosine_full_mean"]):.3f} | {float(y["cosine_full_mean"]):.3f} | {float(x["relative_gradient_error_mean"]):.3f} | {float(y["relative_gradient_error_mean"]):.3f} |')
L+=['','## 嵌套预算的方向变化','','每一行只比较同一次重复、同一checkpoint的两个采样梯度；角度由arccos(cosine)转换。最后一列是增加预算后对全量参考的cosine反而下降的比例，不能用平均趋势掩盖这些单次变化。','','| checkpoint | 嵌套比较 | 梯度间夹角均值±SD | 最大夹角 | 对全量cosine下降比例 |','| --- | --- | ---: | ---: | ---: |']
for ph in (2,3,4):
 for left,right in [('Refresh-2','Refresh-4'),('Refresh-4','Refresh-8'),('Refresh-2','Refresh-8')]:
  xs=[r for r in pairs if int(r['checkpoint_phase'])==ph and r['budget_a']==left and r['budget_b']==right]
  angles=np.array([float(x['angle_degrees']) for x in xs]);worse=np.mean([float(x['cosine_full_delta'])<0 for x in xs])
  L.append(f'| {"ABCDE"[ph]} | {left}→{right} | {angles.mean():.1f}° ± {angles.std(ddof=1):.1f}° | {angles.max():.1f}° | {worse:.1%} |')
L+=['','## 对重点问题的回答','','1. **增加采样量是否降低估计方差？** 在这三个固定checkpoint/静态池上，R2→R4→R8的联合梯度向量方差逐档降低。R8相对K1的归一化向量方差在C/D/E约降低52%/72%/75%。单个范数SD不保证单调，例如C的R8范数SD高于R4；不能把范数SD等同向量方差。','2. **小预算方向是否已代表全量梯度？** 只能提供部分代表性。K1联合余弦约0.38–0.50，相对误差约1.78–2.43；R8改善后仍只有0.60–0.70，相对误差约0.99–1.28。没有据此设定任意通过门槛，也不能称其高精度复现全量方向。','3. **每任务8条是否提供更有效信息？** 本静态对照中是：比总8条增加有效状态与Pair覆盖，同时平均余弦提高、相对误差及向量方差降低。但所有方案任务参与率都为100%，不能声称它解决了本数据中的历史任务未参与问题；也未计算单位时间的信息效率。','4. **是否只是覆盖增加？** 不只是计数增加，方向与误差指标也改善。覆盖与方向并非同一指标：原目标按任务有效状态和有效任务平均，不能把Pair数倍增解释成梯度范数倍增。','5. **增加预算会否明显改变方向，甚至更远离全量？** 会。嵌套比较夹角及单次余弦下降比例已逐阶段列出；平均改善不保证每次采样改善。这种变化表明新增轨迹带来不同梯度贡献，不足以解释为有害梯度或训练效果恶化。','','## 解释边界与文件','','这是单训练种子、单序列、原始K1静态记忆池上的离线预算对照。30次采样不是30个训练种子；统计SD衡量固定模型/池上的采样波动。结果不证明改用大预算会提高在线胜率，也不解释TKPR-Refresh的D/E下降：真正Refresh同时改变历史轨迹内容和参考目标，历史真实池又已被覆盖。本实验刻意固定这些变量，仅检查采样预算。全量经验池本身是有限样本基准，不是总体环境分布的真实梯度。','','- sampling_budget.csv：每次采样、每个任务及ALL的原始指标和索引。','- sampling_budget_summary.csv：各档预算/任务均值SD、完整向量方差、相对偏差与MSE。','- full_pool_reference.csv：全量基准的任务loss、梯度范数、有效状态和Pair数。','- paired_directions.csv：同次抽样方案之间方向变化。','- verification_checks.csv：360次原任务归约梯度一致性、模型不变与嵌套标记。','- input_manifest.json：输入文件路径和校验和、Git版本与依赖。','','复现（原始输入在Ubuntu运行目录，亦可使用校验一致的本机副本）：','','```sh','PYTHONPATH=src:. python docs/results/comparisons/ts5-tkpr-refresh/analysis/sampling_budget/analyze_sampling_budget.py --repo . --seed 86431 --repeats 30','python docs/results/comparisons/ts5-tkpr-refresh/analysis/sampling_budget/render_sampling_budget.py','```','','依赖PyTorch、NumPy、PyYAML、Matplotlib；跨PyTorch版本不保证逐位一致。原始checkpoint、Memory和训练配置从未修改。','']
(P/'analysis_report.md').write_text('\n'.join(L))
