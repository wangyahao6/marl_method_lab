# TS5 · TKPR-Refresh 与 TKPR-K1、VDN-FT/ER

比较问题：逐历史任务采样并在阶段末刷新成功轨迹池，是否改善五任务序列的最终表现与保留？

四组均已完成五阶段训练及阶段末独立贪心评价。seed 3535，单前馈层 VDN64 + GRU，RMSprop；TS5 顺序为双方镜像 S5 → Z5 → C5 → S2Z3 → Z2C3，每阶段名义 2M、全程 10M，累计阈值后的整回合末切换，逐任务 ε 退火。所有组使用同一 action_set 驱动、评价种子块与环境接口。周期评价每次 32 局，阶段末与 FWT 评价每任务 128 局；初始化和评价不计入训练步数。

| 方法 | ACC (%) | BWT (pp) | FWT (pp) | 实际训练步数 |
| --- | ---: | ---: | ---: | ---: |
| TKPR-K1 | 60.78 | 0.98 | 14.06 | 10000031 |
| TKPR-Refresh | 54.84 | 0.78 | 13.87 | 10000055 |
| VDN-FT | 27.19 | -50.20 | 16.41 | 10000062 |
| VDN-ER | 35.47 | -32.42 | 14.84 | 10000012 |

该种子下 TKPR-Refresh 的最终 ACC 比 TKPR-K1 低 5.94 个百分点；两者均高于 VDN-FT 与 VDN-ER。Refresh 相对 K1 同时改变采样预算和记忆刷新，不能将差异单独归因于其中一项。只有一个训练种子和一个任务顺序，不据此推断普遍优劣或统计显著性。

TKPR-K1 使用静态成功轨迹池，每次辅助更新跨历史任务总共采 8 条。Refresh 使用同一 K=1 排序损失，每个非当前历史任务各采 8 条，按有约束的任务等权平均；每阶段末评价后初始化新任务最多 100 条成功轨迹，并刷新所有已有任务最多 50 条。保留轨迹的参考 Q 不变，新轨迹携带当前策略 Q 和逐轨迹来源。两个偏好方法都使用 FT 的 TD 回放。VDN-FT 切换时清空 TD 池，VDN-ER 使用按任务均衡回放。

四组从头独立训练，无共享前缀；同种子不保证独立运行轨迹逐步相同。实际训练 commit 和配置以 [runs.csv](runs.csv) 及其快照链接为准，后续仓库版本不反写为训练版本。

## 数据

- [运行清单](runs.csv)与 `configs/`：实际版本、生效参数、维度和终态。
- [阶段评价矩阵与复算指标](stage_metrics.csv)：A–E 胜率、ACC、BWT、FWT。
- [独立贪心评价及学习曲线数据](evaluations.csv)：保留每次评价胜局数与局数。
- [相对 VDN-FT 的最终差值](paired_final.csv)：单位为百分点。
- [历史任务实际采样汇总](sampling_diagnostics.csv)：含重复抽样，各任务每次样本数的最小/最大值。
- [Refresh 更新记录](memory_updates.csv)：旧数量、新数量、替换数量与最终池大小。

已从原始 evaluations.jsonl 的 outcomes 核对 wins/episodes，复算全部 20 个阶段的 ACC/BWT/FWT 并与 retention.json 对照一致。完整原始日志、模型、成功轨迹及逐轨迹 old_q_values 保留在 Ubuntu `runs/<run_id>/`。

## 分阶段日志分析

四份 gzip 原始日志已从当前结果目录撤下，仅在实验主机保留分析输入；此前上传提交仍存在于 Git 历史。

- [分析说明](analysis/analysis_notes.md)：字段、计算方式、记录频率、分段比较与解释边界。
- [梯度统计](analysis/gradient_metrics_by_phase.csv)：A–E 全阶段及阶段内四等分的 mean/median/P10/P90，区分全部更新与每100次更新的梯度监控样本。
- [计时统计](analysis/timing_metrics_by_phase.csv)：各阶段均值、P50/P95、实测记录耗时总和及 Refresh/K1 比值，区分是否包含梯度监控。
- [辅助更新统计](analysis/aux_updates_by_phase.csv)：参与次数、每更新样本数、来源任务分布与池容量；记忆更新补充数据注明来自现有 memory_updates.csv。
- [可复现脚本](analysis/analyze_logs.py)、[输入校验清单](analysis/input_manifest.json)与[实际字段清单](analysis/field_inventory.json)。

## 历史来源与参考漂移离线分析

[机制分析报告](analysis/source_drift/analysis_report.md)与[数据审计](analysis/source_drift/data_audit.md)：固定状态排序、C/D/E末留存来源梯度（每设置10次）、数值重构检查、TD Loss分阶段/进度段统计。Refresh真实历史池快照和同次TD数据缺失，结果不能还原实际D/E训练更新或证明下降原因。

## 固定历史池的采样预算离线验证

[采样预算分析](analysis/sampling_budget/analysis_report.md)：固定原始TKPR-K1的C/D/E末模型与静态池，K1总8条对比每任务2/4/8条，各30次；包含全池参考梯度、任务级统计、嵌套方向变化与覆盖。该对照隔离采样量，不包含Refresh轨迹刷新，不是在线性能实验。
