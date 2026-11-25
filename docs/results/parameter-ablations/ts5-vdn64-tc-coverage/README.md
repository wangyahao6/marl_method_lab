# TS5-1M · 传递闭包覆盖率控制实验

三个新 VDN-FT 参考运行与九条 Low/Medium/High-TC 分支均已完成；12 个模型各有五任务、每任务 128 局的独立终点评估。任务为 S→Z→C→2S3Z→2Z3C，每阶段 1M，seed 2222、3535、4646，单前馈层 VDN64。

TC 覆盖率已明显拉开，但最终性能没有稳定随 TC 增大而改善。High−Low 最终 ACC 的逐种子差为 −4.219、+3.906、−3.906 个百分点。Low 的三种子平均 ACC 最高，不能据此认定存在普遍最优 TC 区间。

| 方法 | 最终 ACC % | BWT pp | FWT pp | E 阶段 AUC % |
|---|---:|---:|---:|---:|
| VDN-FT-reference | 26.354 | -36.719 | 17.969 | 57.140 |
| low-TC | 53.750 | 11.914 | 19.727 | 39.730 |
| medium-TC | 52.292 | 12.956 | 17.318 | 38.091 |
| high-TC | 52.344 | 12.565 | 17.708 | 38.990 |

表中为训练种子等权均值；标准差及逐种子证据见 [seed_summary.csv](seed_summary.csv)、[stage_metrics.csv](stage_metrics.csv)。AUC 为实际阶段步数上的当前任务胜率梯形积分除以阶段时长，单位为百分比，各阶段均保存。

## 控制与 TC 结果

严格一致：同种子全新 FT 历史池、有效状态、全部 K1 边、每状态 N−2 条额外边，以及 Near/Mid/Far 配额；相同 S 模型、优化器、目标网络与 RNG 恢复源，RMSprop、lr=0.0005、FT、逐任务 ε、损失权重 1、temperature=1、min_gap=0。验证范围为至少四个合法动作且 old Q 严格排序的共同有效决策。其余决策不进入本次约束总体。
离线验证成功后才启动九条正式训练，调度时间核验保存在 [source_manifest.json](source_manifest.json)。36 个训练阶段历史掩码均与离线掩码哈希一致；[control_audit.csv](control_audit.csv) 保存初始化和控制参数，原始池、检查点、掩码哈希见 [offline_provenance.json](offline_provenance.json) 与 [offline_statistics.csv](offline_statistics.csv)。

| Seed | Low Micro / Macro | Medium Micro / Macro | High Micro / Macro |
|---|---|---|---|
| 2222 | 55.15% / 59.47% | 60.92% / 63.77% | 75.23% / 74.91% |
| 3535 | 55.17% / 59.56% | 60.99% / 63.88% | 75.26% / 74.99% |
| 4646 | 55.73% / 60.45% | 61.36% / 64.49% | 75.14% / 74.86% |

Direct Edge Coverage 与动作对数量、排名距离配额在同种子同历史池内严格相同。Complete Order Coverage 在全部 36 个组×历史池中均为 0；因此本实验不能判断完整排序相对部分排序是否更好。闭包增益与分位数见 offline_statistics.csv；每状态 TC 的精确频数分布见 [tc_state_histogram.csv](tc_state_histogram.csv)。Micro 使用动作对分母加权，Macro 对状态等权。

仅近似匹配：归一化 Q-gap。跨全部种子/池按所选动作对数加权的 Q-gap 均值为 low 0.7371；medium 0.7332；high 0.7265。完整均值、中位数、标准差见 offline_statistics.csv；按同种子同历史池共用分箱的分布见 [qgap_histogram.csv](qgap_histogram.csv)。Q-gap 不参与可行性门槛，也未为匹配 Q-gap 删除高闭包增益候选；残余差异仍可能解释部分性能变化。

## 性能、机制与解释边界

High−Low 的 BWT、AUC 也未表现为所有种子同向；各阶段配对趋势、累计历史池 TC 与 Q-gap 对照见 [within_seed_trends.csv](within_seed_trends.csv)、[tc_performance.csv](tc_performance.csv)。这两表的 TC 是固定历史池唯一状态统计，未按实际重复训练采样频率加权。
Top-1 Flip Rate、Preference Loss 与 Pair Violation Rate 见 [preference_diagnostics.csv](preference_diagnostics.csv)。训练诊断计数含重复抽样；Top-1 翻转率沿用实现中所有唯一 old-top1 决策的分母，包括共同 TC 子集外的决策，不应与离线唯一状态计数混用。
三种拓扑同时改变图连接和 TC；即使同预算与距离配额成立，性能关系也不能解释为标量 TC 的独立因果作用。只有三个训练种子；共享 S 前缀、状态和动作对均不是额外训练重复。没有独立完整排序训练组。

## 执行、版本与数据入口

2026-11-24 的三个初始离线入口因缺少仓库根导入路径失败，未生成掩码、未启动正式训练；修复后仅重提离线任务及原先未启动的分支，三个已完成 FT 参考保持复用，无参考重训或额外训练预算。失败/阻塞记录与恢复终态见 [execution_summary.json](execution_summary.json)。
参考训练提交 cd31d7a；正式分支实际提交为 13a6d9b 或 470c22c，逐运行列于 runs.csv，不能统一标为发布提交。后者增加受 tkpr_cpr 方法条件控制的 CPR 路径；静态差异检查未见 TC 选择器、PairAblation 损失或 QLearner 改动。完整启动源码哈希保存在 source_manifest.json。
参考运行通过 pair-reference 关闭全部训练偏好损失并完成五阶段 FT；三个正式组从各自种子的 S 检查点分叉，统一复用 S/Z/C/D 四个 FT 历史池。累计路径预算为 5M，分支新增约 4M（整回合边界有微小超出），总新增名义训练量 51M。

- [逐种子运行与实际配置](runs.csv)、[逐次评估计数](evaluations.csv)、[独立终点评估](independent_final.csv)、[配对 FT 差值](paired_final.csv)。
- [完整自动报告](REPORT.md)、[离线报告](OFFLINE_REPORT.md)、[实验方案](../../../protocols/TS5_TC_COVERAGE.md)。
- 远端原始产物：`rl-navigation-ubuntu:/home/wyh/marl_method_lab/results/ts5_tc_coverage_20261123/` 与 `runs/tc_coverage/ts5_tc_coverage_20261123_*`。完整池、掩码、检查点及日志留远端；本目录保存可复核的轻量表、分布和配置。
