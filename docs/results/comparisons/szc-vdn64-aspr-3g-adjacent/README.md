# SZC VDN64 · ASPR-3G-Adjacent vs TKPR-K1

协议：Stalker→Zealot→Colossus，各阶段 2M 环境步（累计阈值后结束完整回合）；VDN 双前馈层/GRU64、RMSprop、FT、逐任务 ε；seed 2222、3535。
同种子复用同一条已完成的 S 前缀及历史胜局池，Z/C 各自采集历史数据。两方法 min_gap=0、temperature=1、weight=1、q_scale_floor=0.1，使用相同的归一化 Softplus 与损失聚合。
ASPR-3G-Adjacent 只选择 G1-G2、G2-G3；聚类完全复用确定性一维 K-means。K1 仅选择 Top-1 端点对；gap0 独立入口调用原 K1 损失。原有 2M K1 的 gap0.5 结果不作本次对照。

## 运行核验

| 方法 | Seed | 状态 | 退出码 | 恢复哈希匹配 | 终局独立评价一致 |
| --- | --- | --- | --- | --- | --- |
| ASPR-3G-Adjacent | 2222 | completed | 0 | True | True |
| TKPR-K1 | 2222 | completed | 0 | True | True |
| ASPR-3G-Adjacent | 3535 | completed | 0 | True | True |
| TKPR-K1 | 3535 | completed | 0 | True | True |

## 各阶段结果

| 方法 | Seed | 阶段 | S % | Z % | C % | ACC % | BWT pp | FWT pp | 当前任务 AUC % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared S | 2222 | stalker | 55.47 | N/A | N/A | 55.47 | N/A | N/A | 41.87 |
| ASPR-3G-Adjacent | 2222 | zealot | 52.34 | 69.53 | N/A | 60.94 | -3.12 | 9.38 | 47.64 |
| ASPR-3G-Adjacent | 2222 | colossus | 48.44 | 68.75 | 54.69 | 57.29 | -3.91 | 19.92 | 34.23 |
| TKPR-K1 | 2222 | zealot | 45.31 | 63.28 | N/A | 54.30 | -10.16 | 9.38 | 53.38 |
| TKPR-K1 | 2222 | colossus | 50.78 | 64.84 | 53.91 | 56.51 | -1.56 | 20.70 | 42.58 |
| Shared S | 3535 | stalker | 55.47 | N/A | N/A | 55.47 | N/A | N/A | 38.12 |
| ASPR-3G-Adjacent | 3535 | zealot | 53.12 | 71.88 | N/A | 62.50 | -2.34 | 28.12 | 53.53 |
| ASPR-3G-Adjacent | 3535 | colossus | 51.56 | 60.16 | 53.91 | 55.21 | -7.81 | 28.91 | 33.95 |
| TKPR-K1 | 3535 | zealot | 53.91 | 70.31 | N/A | 62.11 | -1.56 | 28.12 | 51.73 |
| TKPR-K1 | 3535 | colossus | 47.66 | 65.62 | 44.53 | 52.60 | -6.25 | 28.91 | 37.43 |

## 同种子配对差值

差值均为 ASPR-3G-Adjacent 减 TKPR-K1，单位 pp。

| Seed | 阶段 | ΔACC | ΔBWT | ΔFWT | ΔAUC |
| --- | --- | ---: | ---: | ---: | ---: |
| 2222 | zealot | 6.64 | 7.03 | 0.00 | -5.74 |
| 2222 | colossus | 0.78 | -2.34 | -0.78 | -8.35 |
| 3535 | zealot | 0.39 | -0.78 | 0.00 | 1.81 |
| 3535 | colossus | 2.60 | -1.56 | 0.00 | -3.48 |

## 诊断与解释

[实际采样对统计](aspr_adjacent_training_pairs.csv)按训练阶段和历史来源任务累计三类跨组对；[组大小与占比](aspr_group_training_sampled.csv)计入重复采样。组大小分布行重复展示动作对总计，不能逐行相加。
G1-G3 只做离线排序诊断，不参与损失；其 `valid_pairs` 是可统计对数，不计入训练约束总数。违反率分母为对应有效对数，当前 Q 平局也计为违反。[固定探测轨迹](aspr_adjacent_offline_pairs.csv)是独立统计口径，沿用已有探测机制，不与训练采样混合。
[偏好诊断](preference_diagnostics.csv)提供两方法实际有效对数和按对数加权的违反率。


| 方法 | Seed | 训练阶段 | 历史任务 | 实际有效对总数 | 每有效状态对数 | 违反率 % |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ASPR-3G-Adjacent | 2222 | zealot | stalker | 653687708 | 11.42 | 1.10 |
| ASPR-3G-Adjacent | 2222 | colossus | stalker | 234799698 | 11.43 | 0.85 |
| ASPR-3G-Adjacent | 2222 | colossus | zealot | 181803089 | 8.72 | 3.03 |
| TKPR-K1 | 2222 | zealot | stalker | 358462916 | 6.08 | 0.90 |
| TKPR-K1 | 2222 | colossus | stalker | 127849905 | 6.08 | 0.37 |
| TKPR-K1 | 2222 | colossus | zealot | 113627547 | 5.19 | 2.50 |
| ASPR-3G-Adjacent | 3535 | zealot | stalker | 650785458 | 10.85 | 0.89 |
| ASPR-3G-Adjacent | 3535 | colossus | stalker | 228514159 | 10.85 | 0.62 |
| ASPR-3G-Adjacent | 3535 | colossus | zealot | 195118800 | 8.72 | 2.67 |
| TKPR-K1 | 3535 | zealot | stalker | 342194493 | 5.89 | 0.77 |
| TKPR-K1 | 3535 | colossus | stalker | 121238971 | 5.89 | 0.31 |
| TKPR-K1 | 3535 | colossus | zealot | 118306511 | 5.13 | 2.65 |
[全量历史胜局离线对照](aspr_cross_group_offline.csv)对两方法阶段末模型重放各自全部历史池；S 池两方法相同，Z 池来自各自分支。该表不进行更新，也不进入损失。

| 方法 | Seed | 阶段 | 历史任务 | G1-G3 有效对数 | G1-G3 违反率 % |
| --- | --- | --- | --- | ---: | ---: |
| ASPR-3G-Adjacent | 2222 | zealot | stalker | 95210 | 0.00 |
| ASPR-3G-Adjacent | 2222 | colossus | stalker | 95210 | 0.00 |
| ASPR-3G-Adjacent | 2222 | colossus | zealot | 71930 | 0.04 |
| TKPR-K1 | 2222 | colossus | zealot | 74575 | 1.21 |
| ASPR-3G-Adjacent | 3535 | zealot | stalker | 87318 | 0.00 |
| ASPR-3G-Adjacent | 3535 | colossus | stalker | 87318 | 0.00 |
| ASPR-3G-Adjacent | 3535 | colossus | zealot | 77914 | 0.04 |
| TKPR-K1 | 3535 | colossus | zealot | 79876 | 2.45 |
ASPR-3G-Adjacent 两种子均值：acc_pct=56.25、bwt_pp=-5.86、fwt_pp=24.41、learning_auc_pct=34.09。
TKPR-K1 两种子均值：acc_pct=54.56、bwt_pp=-3.91、fwt_pp=24.80、learning_auc_pct=40.00。

相邻组相对 K1 的最终平均差值：acc_pct=1.69、bwt_pp=-1.95、fwt_pp=-0.39、learning_auc_pct=-5.92。
新任务学习需结合 Z/C 当前任务 AUC 与端点，旧任务保持需结合 S/Z 终点与 BWT；不能仅由动作对数量或违反率推断保护效果。仅两个训练种子，不作一般性优劣或显著性结论。

逐种子观察（方法减对照）：
Seed 2222：Z 当前任务 AUC -5.74 pp、Z 末胜率 6.25 pp；C 当前任务 AUC -8.35 pp、C 末胜率 0.78 pp；C 末 S 保持 -2.34 pp、Z 保持 3.91 pp、BWT -2.34 pp。
Seed 3535：Z 当前任务 AUC 1.81 pp、Z 末胜率 1.56 pp；C 当前任务 AUC -3.48 pp、C 末胜率 9.38 pp；C 末 S 保持 3.91 pp、Z 保持 -5.47 pp、BWT -1.56 pp。
zealot 新任务学习 AUC：两种子未显示同方向改善。
colossus 新任务学习 AUC：两种子均更低。

G1-G3 的阶段末离线违反情况：
Seed 2222、colossus 末、历史 zealot：Adjacent 0.04%（27/71930），K1 1.21%（905/74575），差 -1.18 pp。
Seed 3535、colossus 末、历史 zealot：Adjacent 0.04%（30/77914），K1 2.45%（1958/79876），差 -2.41 pp。
若同一状态的两类相邻组排序全部严格成立，G1-G3 顺序可由传递性成立；Softplus 训练并不保证每个相邻对都满足，因此应以实际离线违反数判断跨层级问题，不能仅根据掩码排除了 G1-G3 就断言其排序必然失效。S 池可直接按同样本比较，Z 池还包含分支采集差异。

## 数据与计算口径

[运行与版本](runs.csv)、[协议快照](configs/)、[源文件校验](source_manifest.json)、[核验表](audit.csv)。
[阶段指标](stage_metrics.csv)、[配对差值](paired_metrics.csv)、[逐次评价](evaluations.csv)、[学习曲线](learning_curves.csv)、[终局独立评价](independent_final.csv)。
ACC 为已见任务等权胜率；BWT 为旧任务相对学完时的变化；FWT 使用各自首次训练前评价减初始模型表现。AUC 按实际步数对当前任务贪心胜率作梯形积分并除以阶段长度，定期评价32局、阶段末128局。终局独立评价对每个任务使用相同的128局种子块，只用于核验，不作为新增训练种子。
实现校验完成：必要 Python 语法检查、四条分支配置 dry-run、两种子父协议匹配，以及合成合法/非法/无效决策上的显式相邻组 Softplus 损失与梯度核对。另用一条已有 S 胜局核对了离线循环状态重建与跨组统计入口；没有运行额外验证训练。
原始日志、检查点和历史池留在 rl-navigation-ubuntu:/home/wyh/marl_method_lab/runs/；本批次新增四条 Z/C 分支，名义新增训练16M步，S前缀仅引用一次。

调度执行状态见 [执行汇总](execution_summary.json)。失败与缺项按原样保留，不自动重跑。
