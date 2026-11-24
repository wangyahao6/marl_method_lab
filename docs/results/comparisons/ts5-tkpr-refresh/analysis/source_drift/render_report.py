# -*- coding: utf-8 -*-
"""Render figures and report from existing source_drift CSV outputs."""
from pathlib import Path
import csv,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent

def rows(n):return list(csv.DictReader((p/n).open()))
r=rows('reference_rank_drift.csv');cs=rows('source_cosine_summary.csv');td=rows('td_loss_by_phase.csv');gc=rows('gradient_reconstruction.csv');qc=rows('stored_q_checks.csv');stats=rows('trajectory_source_stats.csv')
fig,axs=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
for ax,task in zip(axs,'ABC'):
 v=[x for x in r if x['model_method']=='tkpr_refresh' and x['task']==task];m=np.eye(4)
 for x in v:
  a,b=int(x['checkpoint_a']),int(x['checkpoint_b']);m[a,b]=m[b,a]=float(x['top1_agreement'])
 im=ax.imshow(m,vmin=0,vmax=1,cmap='viridis');ax.set(xticks=range(4),yticks=range(4),xticklabels=list('ABCD'),yticklabels=list('ABCD'),title=f'Task {task}: Top-1 agreement',xlabel='Policy checkpoint',ylabel='Policy checkpoint')
 for a,b in itertools.product(range(4),repeat=2):ax.text(b,a,f'{m[a,b]:.2f}',ha='center',va='center',color='white' if m[a,b]<.7 else 'black')
fig.colorbar(im,ax=axs,shrink=.7);fig.savefig(p/'rank_agreement.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(9,7),layout='constrained')
for i,phase in enumerate((3,4)):
 for j,mode in enumerate(('natural_survivors','balanced_decisions')):
  ax=axs[i,j];m=np.eye(phase);ns=np.zeros((phase,phase),dtype=int)
  for x in cs:
   if x['task']=='A' and int(x['checkpoint_phase'])==phase and x['mode']==mode:
    a,b=int(x['source_a']),int(x['source_b']);m[a,b]=m[b,a]=float(x['cosine_mean']);ns[a,b]=ns[b,a]=int(x['cosine_n'])
  im=ax.imshow(m,vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=range(phase),yticks=range(phase),xticklabels=list('ABCD'[:phase]),yticklabels=list('ABCD'[:phase]),title=f'Checkpoint {"ABCDE"[phase]} / Task A\n{mode}',xlabel='Trajectory source',ylabel='Trajectory source')
  for a,b in itertools.product(range(phase),repeat=2):ax.text(b,a,'1' if a==b else f'{m[a,b]:.2f}\nn={ns[a,b]}',ha='center',va='center',fontsize=9)
fig.colorbar(im,ax=axs,shrink=.7);fig.savefig(p/'source_cosine_task_A.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
for method,label in [('tkpr_k1','TKPR-K1'),('tkpr_refresh','TKPR-Refresh')]:
 vals=[next(x for x in td if x['method']==method and x['phase']==str(ph) and x['segment']=='all') for ph in (2,3,4)]
 ax.plot(list('CDE'),[float(x['mean']) for x in vals],'-o',label=label)
ax.set(xlabel='Training phase',ylabel='Mean logged TD loss',title='All logged updates, current-task training data');ax.legend();fig.savefig(p/'td_loss.png',dpi=160);plt.close(fig)
lines=['# TKPR-Refresh：轨迹来源与参考排序漂移离线分析','', 'TS5，训练 seed 3535。仅使用已保存产物，未重新训练、启动 SMAC 或执行新策略采样。','', '**证据边界：Refresh 的各阶段历史池已原地覆盖，真实 D/E 训练池与同次 TD minibatch 不可恢复。梯度结果是最终池留存旧来源的离线探测，不是原训练更新的重放。** 详见 [数据审计](data_audit.md)。','', '## 1. 输入与统计口径','', '- 28份输入校验和已与Ubuntu原始文件一致，500条Refresh来源路径与阶段匹配，13个损失/掩码/梯度函数AST与训练版本一致；复查用 verify_inputs.py。', '- 两方法各五个阶段末 checkpoint 可读，无阶段内 25/50/75% checkpoint。训练版本为 `9db0ed1951ded62c00eb1ff27a80c71ae5465dc5`。分析运行版本、Python/PyTorch/NumPy 与输入 SHA-256 见 [manifest](input_manifest.json)。本机 CPU / PyTorch 2.6.0，与训练 CPU rollout PyTorch 1.12.1 不同。',f'- 每方法/任务/来源抽查一条完整序列，共 {len(qc)} 条，来源模型重新展开 Q 的最大绝对差 {max(float(x["max_abs_error"]) for x in qc):.3g}。固定的绝对误差参考线 2e-5 并非全部满足，详细数值如实保留；跨版本/后端可能贡献误差，未据此覆盖保存 Q，亦不能宣称逐位完全一致。轨迹字段、时间轴和有限值检查通过。', '- Refresh 来源只取逐轨迹 reference_phase，checkpoint 路径与阶段对应关系经审计；pool phase 仅表示池首次创建阶段。K1 的静态池来源依据采集代码与池级 checkpoint。', '- 固定状态面板：A/B/C 各包含 K1 原池 4 条及 Refresh 各来源最多4条完整轨迹，分别24/20/16条。两方法的所有 A–D 模型读取完全相同的面板；每次独立从零隐状态完整展开其自己的 RNN，使用记录中的历史动作输入。包含后来来源的状态用于回溯评估，不代表当时在线分布；选择信息见 fixed_panel.csv。', '- Top-1 并列时取最小合法动作索引；只统计 alive、filled、终止前且至少两个合法动作的决策。Kendall tau-b 在每决策合法动作内计算后等权平均，保留精确同值，分母为0的决策剔除。Pairwise Ranking Agreement 对所有合法无序动作对汇总 sign(gap) 相等比例（含双同值）；同值与严格排序不算一致。', '- Normalized Q-gap = (合法Top1 Q − Top2 Q) / max(合法Q总体标准差, 0.1)，每模型独立归一化；同时给 gap 绝对变化和旧模型对新贪婪动作的归一化 regret。', '- 来源分布统计独立于排序实验：trajectory_source_stats.csv 给最终池各来源数量、长度均值/标准差、有效决策数量、动作频数。coverage 是 obs 四舍五入3位后的独有行数/有效决策数量，仅为记录覆盖代理，不是全环境状态覆盖；source_distribution_distances.csv 给动作频率 JS 散度，不能单凭它推断排序漂移。','', '## 2. 同一状态上的参考排序变化','', 'Refresh 初始任务参考模型对比 D 末模型（最新纳入的参考模型）：','', '| 历史任务 | 模型阶段 | Top-1一致率 | tau-b | 成对一致率 | 归一化gap绝对变化 |','| --- | --- | ---: | ---: | ---: | ---: |']
for task in 'ABC':
 x=next(x for x in r if x['model_method']=='tkpr_refresh' and x['task']==task and int(x['checkpoint_a'])==ord(task)-65 and x['checkpoint_b']=='3')
 lines.append(f'| {task} | {task}→D | {float(x["top1_agreement"]):.3f} | {float(x["kendall_tau_b"]):.3f} | {float(x["pairwise_ranking_agreement"]):.3f} | {float(x["normalized_gap_abs_change"]):.3f} |')

lines+=['', '最终池各来源轨迹数量（pool phase不参与来源推断）：', '', '| 方法 | 历史任务 | 来源阶段:轨迹数 | 各来源平均长度范围 |', '| --- | --- | --- | ---: |']
for method in ('tkpr_k1','tkpr_refresh'):
 for task in 'ABCDE':
  xs=[x for x in stats if x['method']==method and x['task']==task]
  counts=', '.join('ABCDE'[int(x['reference_phase'])]+':'+x['trajectories'] for x in xs)
  lengths=[float(x['length_mean']) for x in xs]
  lines.append(f'| {method} | {task} | {counts} | {min(lengths):.1f}–{max(lengths):.1f} |')

lines+=['', '![固定状态排序一致率](rank_agreement.png)','', '所有阶段对与 K1 策略对照均见 reference_rank_drift.csv。固定状态上的排序确有变化，这证明策略的输出参考可能改变；不是根据不同来源轨迹分布倒推的。K1 自身策略也会变化，但其存储约束目标保持原始模型 Q；Refresh 改变新增轨迹的参考策略。当前数据不足以逐状态追踪“某条原池轨迹被刷新后其目标如何改变”，因为替换的是轨迹本身且旧池快照缺失。','', '## 3. 来源梯度冲突与数值还原','', 'C/D/E 阶段末模型分别独立分析，只用早于该训练阶段的来源及旧任务：C用A/B；D用A/B/C；E用A/B/C/D。Refresh 最终池按 reference_phase < checkpoint phase 筛选后，是有幸存偏差的旧来源集合，不能视为原来的完整历史池。未直接比较不同模型位置的梯度。','', '每模型位置、每设置10次独立抽样。natural_survivors 从该留存子集均匀采8条完整轨迹，来源份额由抽到的轨迹自然决定；不是已经缺失的原始池自然分布。K1 自然设置保持跨历史任务总预算8条与原始随机均分方式。balanced_decisions 每来源采8条完整序列，计算真实 TKPR 有效状态后，各来源随机保留相同数量的损失决策；完整序列及隐状态展开均保留，只对损失决策掩码做条件化筛选。此设置减少数量差异，同时改变状态选择，不能等同真实训练采样。','', '调用仓库 topk_preference，复用合法动作、min_gap=0.5、Q尺度下限0.1、温度1、k=1和状态/动作对归约。先算完整任务损失与 g，再按来源计算 L_s、g_s，权重 w_s = N_s / ΣN_s，其中 N_s 是有至少一个有效动作对的决策数，**不是轨迹数占比**。实际辅助贡献为 λ/H × w_s g_s（λ=1，H为该设置有约束历史任务数）；原始任务损失为 Σ w_s L_s。梯度范围为 agent.parameters() 中 requires_grad 参数，使用 autograd.grad，按 float64 展平范数与余弦，与 preference_grad_norm_agent 一致。没有 optimizer.step。',f'全部 {len(gc)} 个任务/抽样的重构通过：max|Σw_s g_s−g|最大 {max(float(x["max_abs_error"]) for x in gc):.3g}，相对L2误差最大 {max(float(x["relative_l2_error"]) for x in gc):.3g}。固定验收条件是绝对误差≤2e-6或相对误差≤2e-5；未放宽阈值。详见 gradient_reconstruction.csv。', '', '每抽样源未出现或零有效梯度时，余弦记缺失，不能记零；cosine_n 是10次中的可定义数，负余弦比例仅以可定义项为分母。标准差用 ddof=1；逐次数据和汇总均保留。K1 每任务只有一个参考来源，因此没有可定义的同任务跨来源矩阵。加权贡献还提供其在完整任务梯度上的投影比例；范数不可直接相加。','', 'Task A 的 D/E 来源矩阵（颜色为10次内可定义余弦均值，n为有效次数）：','', '![来源梯度矩阵](source_cosine_task_A.png)','', '| checkpoint | task | 设置 | 来源对 | cosine均值±样本SD | 有效次数/10 | 负余弦比例 |','| --- | --- | --- | --- | ---: | ---: | ---: |']
for x in cs:
 if x['checkpoint_phase'] not in ('3','4'):continue
 lines.append(f'| {"ABCDE"[int(x["checkpoint_phase"])]} | {x["task"]} | {x["mode"]} | {"ABCDE"[int(x["source_a"])]}/{"ABCDE"[int(x["source_b"])]} | {float(x["cosine_mean"]):.3f} ± {float(x["cosine_std"]):.3f} | {x["cosine_n"]}/10 | {float(x["negative_mean"]):.1%} |')
lines+=['', '观测到部分抽样下新旧来源梯度方向冲突；来源对之间差异明显。等量控制后的均值大多为正，但仍有负余弦样本（例如 D末/A任务的A–C来源对50%）。自然设置中某些旧来源有效次数很少，例如 E末/A任务A–B对仅2次；不把这种高方差结果解释为稳定的训练规律。来源分组同时改变轨迹状态、行为历史及参考Q，负余弦不能单独归因于参考目标漂移。','', '## 4. TD 关系与训练日志','', 'source_td_gradient.csv 为明确不可用记录。没有当前任务 TD minibatch 与同次参数位置，未用成功轨迹或其他阶段数据代替。已公布的聚合 preference–TD cosine 无来源拆分，不能填补本项。','', '| 阶段 | K1 TD均值 | Refresh TD均值 | 相对变化 | K1中位数 | Refresh中位数 |','| --- | ---: | ---: | ---: | ---: | ---: |']
for ph in (2,3,4):
 x=next(x for x in td if x['method']=='tkpr_k1' and int(x['phase'])==ph and x['segment']=='all');y=next(x for x in td if x['method']=='tkpr_refresh' and int(x['phase'])==ph and x['segment']=='all')
 lines.append(f'| {"ABCDE"[ph]} | {float(x["mean"]):.5f} | {float(y["mean"]):.5f} | {float(y["mean"])/float(x["mean"])-1:+.1%} | {float(x["median"]):.5f} | {float(y["median"]):.5f} |')
lines+=['', '![TD loss](td_loss.png)','', 'TD Loss 取原始 updates JSONL 每次更新的 td_loss（masked TD MSE），并以该运行实际阶段起止环境步分成四个进度段；不是 gradient_monitor 子样本。逐阶段与四分段的 n/mean/median/sample SD 全见 td_loss_by_phase.csv。两组来自不同策略、状态经验和目标网络，较大TD Loss不直接等于学习能力较差，也不构成性能差异的因果解释。','', '## 5. 对五个问题的回答','', '1. **参考动作排序是否变化：是。** 同一固定完整轨迹上，Refresh 初始参考策略与D末策略的Top1一致率约62%–68%；K1策略也变化，但其历史参考目标固定。', '2. **新旧来源有无梯度冲突：部分抽样存在。** 数值拆分已校验；自然与等量设置下冲突比例不同。缺失真实历史池使结果只适用于留存样本探测。', '3. **新来源是否更符合当前TD：无法验证。** 缺少同checkpoint/同更新TD minibatch；不从来源间余弦或TD Loss替代推断。', '4. **参考目标漂移可能削弱知识保护吗：机制上可能，尚未建立因果。** 新策略生成的新轨迹携带不同参考排序，旧策略目标的覆盖也因替换减少；但状态分布变化、每任务采样预算变化、新任务本身的学习差异都是竞争解释。', '5. **是否足以解释D/E下降：不足。** 固定状态排名变化、部分来源冲突与更高TD Loss是可验证现象；缺少训练时真实池、阶段内checkpoint及同次TD数据，无法证明它们导致Refresh在D/E的ACC下降。重复抽样不增加训练种子数量。','', '## 复现与文件','', '从仓库根目录执行（原始输入在实验主机，亦可使用按 manifest 校验过的只读副本）：','', '```sh','PYTHONPATH=src:. python docs/results/comparisons/ts5-tkpr-refresh/analysis/source_drift/analyze_source_drift.py --repo .','python docs/results/comparisons/ts5-tkpr-refresh/analysis/source_drift/render_report.py','```','', '依赖 Python、PyTorch、NumPy、PyYAML 与 Matplotlib。分析随机种子68121，CPU两线程；跨PyTorch版本重现不保证逐位相同。必要输入位置/SHA-256见 manifest，固定状态面板含序列hash。原始模型/轨迹/日志不上传本目录。所有 CSV 的 phase 为0起始，A=0…E=4。空格表示不可用；`source_gradient_metrics.csv`、`source_gradient_cosine.csv` 是逐次记录，相应 summary 文件给均值/标准差及有效次数。','']
(p/'analysis_report.md').write_text('\n'.join(lines))
