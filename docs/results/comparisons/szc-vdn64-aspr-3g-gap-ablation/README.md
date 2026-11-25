# SZC · ASPR-3G / Adjacent 的 min_gap 2×2 消融

ASPR-3G 与 ASPR-3G-Adjacent 分别比较 min_gap=0 与 0.5。只新增 3G gap0、Adjacent gap0.5 各两个种子分支，复用已有 3G gap0.5 与 Adjacent gap0。未新增 TKPR-K1 或 S 训练。
seed 2222、3535；每个种子共用原 TKPR-K1 的 2M S 检查点及同一历史 S 胜局池；Z/C 各按原累计协议推进到4M/6M，结束完整回合。双层 VDN/GRU64、RMSprop、FT、逐任务 ε、温度/权重/标准化/采样/评价均沿用对应方法原协议。方法实现未改变；两份新增配置相对各自原配置只改 min_gap。

## 运行核验

| 方法 | gap | seed | 新增 | 完成/退出0 | 恢复哈希 | 阶段预算 | 终局独立评价 | 采样对数重建 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASPR-3G | 0.0 | 2222 | True | True | True | True | True | True |
| ASPR-3G | 0.5 | 2222 | False | True | True | True | None | True |
| ASPR-3G-Adjacent | 0.0 | 2222 | False | True | True | True | True | True |
| ASPR-3G-Adjacent | 0.5 | 2222 | True | True | True | True | True | True |
| ASPR-3G | 0.0 | 3535 | True | True | True | True | True | True |
| ASPR-3G | 0.5 | 3535 | False | True | True | True | None | True |
| ASPR-3G-Adjacent | 0.0 | 3535 | False | True | True | True | True | True |
| ASPR-3G-Adjacent | 0.5 | 3535 | True | True | True | True | True | True |

历史 3G 若无额外终局重评，独立评价复核记 N/A；原阶段末128局独立贪心评价仍用于性能指标。新增四条分支均安排终局每任务128局独立重评。[逐运行核验](run_audit.csv)、[实际阶段步数](stage_steps.csv)、[仅 min_gap 差异核对](config_audit.csv)保留细节。

## 阶段指标

| 方法 | gap | seed | 阶段 | S % | Z % | C % | ACC % | BWT pp | FWT pp | 当前任务 AUC % |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared S | None | 2222 | stalker | 55.47 | N/A | N/A | 55.47 | N/A | N/A | 41.87 |
| ASPR-3G | 0.0 | 2222 | zealot | 49.22 | 70.31 | N/A | 59.77 | -6.25 | 9.38 | 48.90 |
| ASPR-3G | 0.0 | 2222 | colossus | 47.66 | 73.44 | 59.38 | 60.16 | -2.34 | 16.41 | 37.42 |
| ASPR-3G | 0.5 | 2222 | zealot | 50.00 | 81.25 | N/A | 65.62 | -5.47 | 9.38 | 59.00 |
| ASPR-3G | 0.5 | 2222 | colossus | 54.69 | 78.12 | 57.81 | 63.54 | -1.95 | 24.61 | 42.14 |
| ASPR-3G-Adjacent | 0.0 | 2222 | zealot | 52.34 | 69.53 | N/A | 60.94 | -3.12 | 9.38 | 47.64 |
| ASPR-3G-Adjacent | 0.0 | 2222 | colossus | 48.44 | 68.75 | 54.69 | 57.29 | -3.91 | 19.92 | 34.23 |
| ASPR-3G-Adjacent | 0.5 | 2222 | zealot | 48.44 | 82.81 | N/A | 65.62 | -7.03 | 9.38 | 57.57 |
| ASPR-3G-Adjacent | 0.5 | 2222 | colossus | 44.53 | 73.44 | 50.78 | 56.25 | -10.16 | 21.09 | 38.48 |
| Shared S | None | 3535 | stalker | 55.47 | N/A | N/A | 55.47 | N/A | N/A | 38.12 |
| ASPR-3G | 0.0 | 3535 | zealot | 50.00 | 63.28 | N/A | 56.64 | -5.47 | 28.12 | 51.29 |
| ASPR-3G | 0.0 | 3535 | colossus | 49.22 | 64.06 | 54.69 | 55.99 | -2.73 | 26.95 | 38.78 |
| ASPR-3G | 0.5 | 3535 | zealot | 52.34 | 82.03 | N/A | 67.19 | -3.12 | 28.12 | 58.05 |
| ASPR-3G | 0.5 | 3535 | colossus | 51.56 | 75.00 | 50.78 | 59.11 | -5.47 | 28.12 | 42.60 |
| ASPR-3G-Adjacent | 0.0 | 3535 | zealot | 53.12 | 71.88 | N/A | 62.50 | -2.34 | 28.12 | 53.53 |
| ASPR-3G-Adjacent | 0.0 | 3535 | colossus | 51.56 | 60.16 | 53.91 | 55.21 | -7.81 | 28.91 | 33.95 |
| ASPR-3G-Adjacent | 0.5 | 3535 | zealot | 49.22 | 74.22 | N/A | 61.72 | -6.25 | 28.12 | 57.49 |
| ASPR-3G-Adjacent | 0.5 | 3535 | colossus | 47.66 | 66.41 | 62.50 | 58.85 | -7.81 | 32.03 | 43.83 |

## 两种子最终均值

| 方法 | gap | ACC % | BWT pp | FWT pp | C AUC % |
| --- | --- | ---: | ---: | ---: | ---: |
| ASPR-3G | 0.0 | 58.07 | -2.54 | 21.68 | 38.10 |
| ASPR-3G | 0.5 | 61.33 | -3.71 | 26.37 | 42.37 |
| ASPR-3G-Adjacent | 0.0 | 56.25 | -5.86 | 24.41 | 34.09 |
| ASPR-3G-Adjacent | 0.5 | 57.55 | -8.98 | 26.56 | 41.16 |

| 对照方向 | 方法 | gap | seed | ΔACC pp | ΔBWT pp | ΔFWT pp | ΔC AUC pp |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| gap0-minus-gap05 | ASPR-3G | 0.0 | 2222 | -3.39 | -0.39 | -8.20 | -4.72 |
| gap0-minus-gap05 | ASPR-3G-Adjacent | 0.0 | 2222 | 1.04 | 6.25 | -1.17 | -4.25 |
| adjacent-minus-3g | ASPR-3G-Adjacent | 0.0 | 2222 | -2.86 | -1.56 | 3.52 | -3.19 |
| adjacent-minus-3g | ASPR-3G-Adjacent | 0.5 | 2222 | -7.29 | -8.20 | -3.52 | -3.66 |
| gap0-minus-gap05 | ASPR-3G | 0.0 | 3535 | -3.12 | 2.73 | -1.17 | -3.82 |
| gap0-minus-gap05 | ASPR-3G-Adjacent | 0.0 | 3535 | -3.65 | 0.00 | -3.12 | -9.88 |
| adjacent-minus-3g | ASPR-3G-Adjacent | 0.0 | 3535 | -0.78 | -5.08 | 1.95 | -4.83 |
| adjacent-minus-3g | ASPR-3G-Adjacent | 0.5 | 3535 | -0.26 | -2.34 | 3.91 | 1.23 |
ASPR-3G：gap0 相对 gap0.5 的两种子最终均值差为 acc_pct -3.26 pp、bwt_pp 1.17 pp、fwt_pp -4.69 pp、learning_auc_pct -4.27 pp。
ASPR-3G-Adjacent：gap0 相对 gap0.5 的两种子最终均值差为 acc_pct -1.30 pp、bwt_pp 3.12 pp、fwt_pp -2.15 pp、learning_auc_pct -7.07 pp。

[逐种子配对差](paired_metrics.csv)同时报告同方法 gap0−gap0.5 与同 gap 的 Adjacent−3G；[交互差值](interaction_metrics.csv)定义为 (Adjacent0.5−Adjacent0)−(3G0.5−3G0)。ACC 与任务端点衡量结果，Z/C AUC 衡量当前任务学习过程，BWT 结合各旧任务端点判断保持。只有两个种子，不作显著性或普遍优劣断言。

## 动作对与 Q-gap

[三类跨组对的阶段末离线违反率](cross_group_offline.csv)重放各自完整历史胜局池，固定旧 Q/分组，使用阶段末当前 Q；当前平局计为违反。non_tie_violation_rate 对所有非平局跨组对统计，便于跨 gap 直接对照；gap_kept_violation_rate 仅统计 |标准化旧 Q 差|≥min_gap 的对；loss_violation_rate 仅用于实际进入该方法损失的对。Adjacent 的 G1-G3 只做诊断，loss_pairs=0。S 池相同；Z 池由各分支独立采集，不视为共同样本因果对照。
[实际采样动作对数](sampled_pairs.csv)从每次更新的 source_phase/episode_ids 重建，重复轨迹重复计数，按 G1-G2/G2-G3/G1-G3 分别保存候选、非平局、间隔通过与实际入损失对数；与日志 valid_pairs 总计核对。旧 3G 日志只记录总违反率，未记录三类对逐更新的违反数，不能恢复训练中逐类违反率；三类比较统一使用阶段末离线口径。
[Q-gap 分布摘要](q_gap_summary.csv)与[直方图](q_gap_histogram.csv)使用实际采样次数加权，分别记录原始旧 Q 差和标准化旧 Q 差，并区分 non_tie、gap_kept、loss 三个总体。分位数使用加权经验分布逆 CDF；标准化复用原 reference_order()/full_rank_pairs()，保留原实现精确平局排除和 ≥min_gap 边界。记录分布不改变训练筛选。

## 来源与复用

[运行清单](runs.csv)、[配置快照](configs/)、[源文件校验](source_manifest.json)、[逐次评价](evaluations.csv)、[学习曲线](learning_curves.csv)、[终局重评](independent_final.csv)、[训练日志汇总](preference_diagnostics.csv)、[采样重建核验](sampling_audit.csv)。
ACC/BWT/FWT 按仓库标准口径，AUC 是各阶段当前任务起点/定期/阶段末贪心胜率对实际步数的梯形积分均值。已有前缀、方法结果和原始日志只读引用；本批次诊断另存 results/本批次/diagnostics/<run-id>/，不覆盖旧诊断。
[已有 ASPR-3G gap0.5](../szc-vdn64-aspr-groups/README.md)；[已有 Adjacent gap0](../szc-vdn64-aspr-3g-adjacent/README.md)。完整原始产物留在 rl-navigation-ubuntu:/home/wyh/marl_method_lab/runs/。


## C 阶段末三类跨组违反率

以下均为全部非平局对的离线违反率，避免 min_gap 改变分母掩盖排序变化。

| 方法 | gap | seed | 历史任务 | G1-G2 % | G2-G3 % | G1-G3 % |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ASPR-3G | 0.0 | 2222 | stalker | 0.67 | 0.68 | 0.01 |
| ASPR-3G | 0.0 | 2222 | zealot | 1.79 | 1.87 | 0.03 |
| ASPR-3G | 0.0 | 3535 | stalker | 0.92 | 0.93 | 0.00 |
| ASPR-3G | 0.0 | 3535 | zealot | 2.37 | 2.57 | 0.06 |
| ASPR-3G | 0.5 | 2222 | stalker | 5.99 | 5.99 | 0.54 |
| ASPR-3G | 0.5 | 2222 | zealot | 4.22 | 4.21 | 0.33 |
| ASPR-3G | 0.5 | 3535 | stalker | 5.34 | 5.01 | 0.24 |
| ASPR-3G | 0.5 | 3535 | zealot | 3.13 | 3.34 | 0.38 |
| ASPR-3G-Adjacent | 0.0 | 2222 | stalker | 0.32 | 0.33 | 0.00 |
| ASPR-3G-Adjacent | 0.0 | 2222 | zealot | 1.32 | 1.27 | 0.04 |
| ASPR-3G-Adjacent | 0.0 | 3535 | stalker | 0.33 | 0.31 | 0.00 |
| ASPR-3G-Adjacent | 0.0 | 3535 | zealot | 1.28 | 1.29 | 0.04 |
| ASPR-3G-Adjacent | 0.5 | 2222 | stalker | 6.70 | 6.67 | 1.58 |
| ASPR-3G-Adjacent | 0.5 | 2222 | zealot | 3.69 | 3.36 | 0.67 |
| ASPR-3G-Adjacent | 0.5 | 3535 | stalker | 5.60 | 5.60 | 0.90 |
| ASPR-3G-Adjacent | 0.5 | 3535 | zealot | 2.61 | 2.49 | 0.32 |

## 实际训练采样约束量

G1-G3 对 Adjacent 的入损失数量始终为 0，其候选与间隔通过计数仍保留在 CSV 供诊断。

| 方法 | gap | seed | 训练阶段 | 历史任务 | G1-G2 入损失 | G2-G3 入损失 | G1-G3 入损失 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| ASPR-3G | 0.0 | 2222 | 1 | stalker | 323403056 | 327340386 | 238466406 |
| ASPR-3G | 0.0 | 2222 | 2 | stalker | 119884312 | 121363964 | 88425013 |
| ASPR-3G | 0.0 | 2222 | 2 | zealot | 92149553 | 102800963 | 69954208 |
| ASPR-3G | 0.0 | 3535 | 1 | stalker | 300582381 | 329126384 | 221330844 |
| ASPR-3G | 0.0 | 3535 | 2 | stalker | 107464151 | 117738969 | 79170871 |
| ASPR-3G | 0.0 | 3535 | 2 | zealot | 89110917 | 99845604 | 70106957 |
| ASPR-3G | 0.5 | 2222 | 1 | stalker | 190289397 | 196166136 | 232862617 |
| ASPR-3G | 0.5 | 2222 | 2 | stalker | 66278429 | 68306977 | 81136795 |
| ASPR-3G | 0.5 | 2222 | 2 | zealot | 67445035 | 80575994 | 65997188 |
| ASPR-3G | 0.5 | 3535 | 1 | stalker | 202478130 | 213938839 | 220931214 |
| ASPR-3G | 0.5 | 3535 | 2 | stalker | 69664853 | 73622142 | 76067378 |
| ASPR-3G | 0.5 | 3535 | 2 | zealot | 66449921 | 79814229 | 61875403 |
| ASPR-3G-Adjacent | 0.0 | 2222 | 1 | stalker | 324865359 | 328822349 | 0 |
| ASPR-3G-Adjacent | 0.0 | 2222 | 2 | stalker | 116680160 | 118119538 | 0 |
| ASPR-3G-Adjacent | 0.0 | 2222 | 2 | zealot | 83693853 | 98109236 | 0 |
| ASPR-3G-Adjacent | 0.0 | 3535 | 1 | stalker | 310646715 | 340138743 | 0 |
| ASPR-3G-Adjacent | 0.0 | 3535 | 2 | stalker | 109057623 | 119456536 | 0 |
| ASPR-3G-Adjacent | 0.0 | 3535 | 2 | zealot | 91329043 | 103789757 | 0 |
| ASPR-3G-Adjacent | 0.5 | 2222 | 1 | stalker | 187407043 | 193190331 | 0 |
| ASPR-3G-Adjacent | 0.5 | 2222 | 2 | stalker | 66571204 | 68609555 | 0 |
| ASPR-3G-Adjacent | 0.5 | 2222 | 2 | zealot | 64860448 | 79646183 | 0 |
| ASPR-3G-Adjacent | 0.5 | 3535 | 1 | stalker | 195462416 | 206528179 | 0 |
| ASPR-3G-Adjacent | 0.5 | 3535 | 2 | stalker | 71036695 | 75068255 | 0 |
| ASPR-3G-Adjacent | 0.5 | 3535 | 2 | zealot | 69983187 | 81809606 | 0 |
调度任务终态见 [执行记录](execution_summary.json)。核验缺项如实保留，不自动增加种子或重跑训练。
