# 实验结果

这里保存 MARL Method Lab 已核对并公开的实验。自动分析从[机器可读索引](catalog.json)开始，按实验 ID 获取数据文件、关联合并键、对照组与解释边界。每页先给出比较问题、图表与结论，再链接逐种子评估和实际运行版本。指标、字段和后续整理方式见[数据说明](DATA.md)。

## 方法与基线对照

| 实验 | 设置 | 从哪里读起 |
| --- | --- | --- |
| [SZC · 双层 VDN64 约束方法结果](comparisons/szc-vdn64/README.md) | 追猎者 → 狂热者 → 巨像，33 条独立运行，每阶段 2M 步 | 集中查看 FRPR-Raw、RawLoss、Unfiltered、ReLU、TKPR、T1PR、ASPR-Max、CHI 与 VDN-FT/ER 的逐种子指标、同种子差值和诊断 CSV。 |
| [SZC · TKPR-K1 共同前缀对照](comparisons/szc-vdn64-tkpr-k1-shared/README.md) | 双层 VDN64，seed 2222、3535，各 1 个 S 前缀与 5 个 Z→C 分支 | 学习曲线、ACC/BWT/FWT/AUC、同前缀差值、偏好诊断和 Z/C 配对更新时间。 |

## 回访与阶段干预消融

| 实验 | 设置 | 从哪里读起 |
| --- | --- | --- |
| [TS4 · 回访时继续或取消偏好约束](return-ablations/ts4-vdn64-ranking/README.md) | 追猎者 → 狂热者 → 巨像 → 追猎者，种子 3535 | 先看 A₂ 回训曲线及配对差值，再看各任务终点。独立 VDN-FT 单列。 |

## 超参数与组件消融

| 实验 | 设置 | 从哪里读起 |
| --- | --- | --- |
| [SZC · TKPR K 值消融](parameter-ablations/szc-vdn64-tkpr-k/README.md) | 双层 VDN64，seed 1111，K=1、2、3、5 与 FRPR | 阶段末胜率、当前任务曲线、有效 K 分布及梯度诊断。 |
| [SZC-1M · 动作对结构与 Top-1 比例](parameter-ablations/szc-vdn64-pair-structure-ratio/README.md) | 双层 VDN64，seed 3535，共用 S/Z 历史池，7 个结构组与 6 个比例组 | 逐任务胜率、等预算对照、实际比例、传递闭包和偏好梯度诊断。 |

## 机制诊断

用于收录冻结模型屏蔽、筛选稳定性、任务历史以及网络相似性等分析。当前暂无公开结果。

## 策略行为分析

| 实验 | 设置 | 分析内容 |
| --- | --- | --- |
| [SZC · VDN64 贪婪动作一致率](behavioral-analysis/szc-vdn64-greedy-action-agreement/README.md) | 冻结检查点与固定旧策略轨迹 | K1/K2/K3/K5/FRPR 的阶段间动作一致率、置信度和旧任务胜率。 |
| [SZC · VDN64 旧策略估计的动作价值损失](behavioral-analysis/szc-vdn64-action-regret/README.md) | 复用 AGR 的 1,408 局归档轨迹 | 比较动作变化的归一化损失、置信度和 VDN 团队损失。 |
| [SZC · VDN64 高损失动作的环境干预](behavioral-analysis/szc-vdn64-action-intervention/README.md) | 冻结 VDN64，新策略、高损失纠正、全纠正、旧策略各 128 局 | 比较实测胜率、回报和干预量。 |
| [SZC · 单智能体反事实动作干预](behavioral-analysis/szc-vdn64-counterfactual-action/README.md) | 同一 S 胜局三个决策点，冻结旧策略 | 检查 GRU 恢复，比较各合法动作的旧 Q 与干预后胜负。 |
