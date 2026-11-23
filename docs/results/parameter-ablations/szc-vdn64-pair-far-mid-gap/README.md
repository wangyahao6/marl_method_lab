# SZC-1M：Far/Mid 额外动作对的 Q-gap 筛选实验（B）

**直接结果。** 在同一种子、同一 S 前缀和同一组有效历史状态上，只把 Far/Mid 额外动作对的归一化旧 Q-gap 门槛从 0 提到 0.5，确实提高了所选额外对的平均 Q-gap，降低了训练中的偏好损失和选中对违反率。但 C 末性能的配对方向在两个种子相反：seed 2222 的 ACC、BWT 分别提高 1.30、6.25 pp；seed 3535 分别降低 3.39、8.98 pp。因此本批不能认定筛选有稳定的持续学习收益。

## 设计、共同状态与代码版本

本实验共四个分支，分别为 `pair_gap_far_mid_unfiltered` 和 `pair_gap_far_mid_filtered` × seed 2222、3535；复用实验 A 相应种子的两条 TD 参考线。总批次 20 个任务均已完成、退出码为 0。SMACv2 5v5、双层 VDN64、Stalker → Zealot → Colossus、每阶段 100 万环境步，其余训练、历史池和评估协议见[实验总说明](../../../SZC_PAIR_DISTANCE.md)。每个阶段末对所有已见任务分别进行 128 局独立贪心评估。

两组都保留全部 `N−1` 个 Top-1 对。`N≥5` 时额外选 `B=min(N−2,N−3,(N−4)(N−3)/2)` 个非 Top-1 对，Far 约占 2/3、Mid 约占 1/3；`N=4` 时没有额外对。筛选组只要求额外候选对的 `|Q_i−Q_j|/max(σ(Q_legal),0.1)≥0.5`，Top-1 对始终不筛。两组统一仅保留筛选后 Far、Mid 候选均能填满配额的状态，故逐状态有效位置、Top-1 对、Far/Mid 数量及总预算完全相同。选对掩码在历史池构建时用固定 `pair_seed=3535` 生成并保存。

筛选可行性造成明显的共同状态覆盖损失。相对于实验 A 的基本有效状态，seed 2222 的 S/Z 池分别再排除 5,520/6,465 个位置，保留 16,533/15,128 个；seed 3535 再排除 6,293/6,867 个，保留 15,651/15,467 个。约四分之一至近三分之一的基本有效位置因此不进入 B 的任一组。两组在四个历史池合计各选 553,097 对，Top-1 比例 62.5%；逐池有效状态哈希、预算和固定种子掩码均经归档文件与源码重建核对。

两种子 seed 2222 的分支由 `a10c88d`、seed 3535 的分支由 `9db0ed1` 启动，参考线由 `d4f4255` 启动。中间提交增加了独立 TKPR-refresh 和 TS5 结果；动作对选择、损失源码哈希在本批保持相同。TKPR-refresh 的历史更新和采样分支以其专用方法名及设置为条件，本实验未启用；实际训练共享协议字段相同。[运行清单](runs.csv)和[配置快照](configs/)保留所有提交 SHA 和关键源码哈希，不能仅凭不同 SHA 推断本方法效果，也不能忽略版本事实。

## 性能与配对比较

ACC 为 C 末三任务胜率均值，BWT 为旧任务相对各自训练后端点的平均百分点变化。标准差是两种子样本标准差。筛选−不筛选为同种子配对差值，ACC 与 BWT 不相加。

| 额外对规则 | C 末 ACC %，均值±SD | C 末 BWT pp，均值±SD | Z AUC % | C AUC % |
| --- | ---: | ---: | ---: | ---: |
| Q-gap 不筛选 | 48.83±5.71 | 5.66±6.35 | 43.80 | 34.34 |
| 额外对 Q-gap≥0.5 | 47.79±2.39 | 4.30±4.42 | 46.94 | 31.71 |

| Seed | 不筛选 ACC/BWT | 筛选 ACC/BWT | 筛选−不筛选 ΔACC/ΔBWT pp |
| --- | --- | --- | --- |
| 2222 | 44.79/1.17 | 46.09/7.42 | +1.30/+6.25 |
| 3535 | 52.86/10.16 | 49.48/1.17 | −3.39/−8.98 |

两种子平均差为 ACC −1.04 pp、BWT −1.37 pp；其方向都受 seed 3535 影响，样本不足以说明稳定的负作用。两组位于同一有效状态集合，但实验 A 的 K1 基线使用更大的基本有效状态集合；[相对 K1 的差值](paired_metrics.csv)是同种子训练性能参照，不是同状态选择机制的严格配对。严格的筛选对照见 [paired_gap.csv](paired_gap.csv)。[stage_metrics.csv](stage_metrics.csv)提供 S/Z/C 端点、FWT 和各阶段 AUC，[learning_curves.csv](learning_curves.csv)提供逐次评估。

## Q-gap、排序距离与优化

在相同历史池和共同有效位置上，额外对平均归一化 Q-gap 由 seed 2222 的 S/Z 池 **0.910/0.927** 增至 **1.024/1.047**，seed 3535 由 **0.874/0.929** 增至 **0.995/1.041**。每池的额外对数相同；Far/Mid 大类数量相同。筛选也使 Far 内部的具体排名距离略有上移：四池平均距离由约 3.13–3.20 增至 3.19–3.26。因此 B 主要操纵 Q-gap，但并非在每个精确排名距离上完全匹配。逐池分位数、逐状态均值及分距离覆盖见 [pair_diagnostics.csv](pair_diagnostics.csv)和[pair_rank_gap_distribution.csv](pair_rank_gap_distribution.csv)。

C 阶段两种子平均的偏好梯度范数由不筛选的 0.453 降至筛选的 0.367；选中对违反率约由 2.4%–2.5% 降至 1.6%–1.7%，偏好损失也在两个种子均降低。与此同时，seed 2222 的 BWT 提高、seed 3535 的 BWT 降低。**较低的损失和梯度只说明所选约束在当前训练过程中较易满足，不能推出更好的旧任务保持。** Top-1 翻转率、TD 梯度及偏好／TD 梯度余弦见[偏好诊断](preference_diagnostics.csv)和[梯度诊断](gradient_diagnostics.csv)。

## 数据与解释边界

- [runs.csv](runs.csv)含两条参考线和四个分支的实际终态、同种子恢复模型哈希和运行路径；[configs/](configs/)保存实际协议和源码版本信息。
- [evaluations.csv](evaluations.csv)、[summary.csv](summary.csv)、[stage_metrics.csv](stage_metrics.csv)、[learning_curves.csv](learning_curves.csv)、[paired_metrics.csv](paired_metrics.csv)、[paired_gap.csv](paired_gap.csv)保留原始评估局数、均值、标准差及逐种子配对数据。
- [pair_diagnostics.csv](pair_diagnostics.csv)、[pair_rank_gap_distribution.csv](pair_rank_gap_distribution.csv)、[preference_diagnostics.csv](preference_diagnostics.csv)、[gradient_diagnostics.csv](gradient_diagnostics.csv)保留结构与机制诊断。

导出脚本 [`scripts/summarize_szc_pair_distance.py`](../../../../scripts/summarize_szc_pair_distance.py)重建全部掩码、复核预算和排序传递闭包。两个种子的性能方向相反，且筛选改变了 Far 内部的精确排名距离；本批只支持“筛选确实改变选对分布并降低偏好优化量”的观测，**不支持**“Q-gap≥0.5 一定更有利于 ACC 或 BWT”的结论。后续若要区分 Q-gap 与排名距离，应在精确距离层内匹配 Q-gap 或反向匹配距离，并增加独立种子。
