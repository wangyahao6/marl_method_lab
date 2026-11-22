# SZC VDN64 · ASPR-3G / ASPR-4G

两种多组 ASPR 在 seed 2222、3535 上各完成一条 S→Z→C 运行。每个种子沿用同一条已完成的 TKPR-K1 S 前缀，只在新分支目录用父池 `old_q_values` 生成 `group_ids`；S 不重新训练。Z、C 各训练 2M 环境步，RMSprop、FT 回放、逐任务 ε、评估种子和局数均沿用父协议。四条分支的 `status=completed`、启动器退出码 0，恢复后的模型哈希与对应父 S 哈希相同。

## C 阶段结束

| 方法 | seed 2222 ACC % | seed 3535 ACC % | 两种子平均 ACC % | 平均 BWT pp | 平均 FWT pp | 平均 C 阶段 AUC % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASPR-2G（既有共同 S 对照） | 61.98 | 64.32 | 63.15 | -5.47 | 24.61 | 44.63 |
| ASPR-3G | 63.54 | 59.11 | 61.33 | -3.71 | 26.37 | 42.37 |
| ASPR-4G | 55.47 | 64.58 | 60.03 | -2.34 | 24.02 | 38.61 |

3G 相比 2G 的 C 末 ACC 在 seed 2222 高 1.56 pp，在 seed 3535 低 5.21 pp；4G 分别低 6.51 pp、高 0.26 pp。两种子未显示一致的 ACC 改善。4G 的平均 BWT 较高，但 C 阶段当前任务 AUC 较低。逐种子、逐阶段结果和同前缀差值分别见 [阶段指标](stage_metrics.csv) 与 [配对指标](paired_metrics.csv)。样本量只有两个种子，不能据此推断一般性优劣。

## 分组与采样

旧 Q 在有效决策内按现有 `reference_scale()` 标准化，等值动作先合并，再由确定性动态规划求连续一维 K-means 的最小组内平方误差；组中心从高到低编号。最多用 3 或 4 组，实际组数可减少。只约束不同组的合法无序动作对，沿用 `min_gap=0.5`、Softplus、温度 1、偏好权重 1 和原有损失聚合。实现见[方法说明](../../../VDN_ACTION_SET_PREFERENCE.md)。

[组均值、占比、空组率](aspr_group_summary.csv)和[组大小完整分布](aspr_group_distribution.csv)来自各阶段保存的胜局。[训练实际采样分布](aspr_group_training_sampled.csv)按训练阶段、历史任务和组大小汇总，重复抽取的轨迹重复计数；它同时给出跨组候选/有效动作对、保留率和 Pair Violation Rate。保留率的分母为非平局候选对。CSV 中每个组大小行重复展示所属训练阶段与历史任务的动作对总计，聚合动作对数时每组只取一行。

## 数据与边界

- [运行清单](runs.csv)和[有效协议快照](configs/)记录方法、种子、父 S、优化器、回放及恢复哈希；[来源清单](source_manifest.json)保留远端原始小文件 SHA-256。
- [独立评估](evaluations.csv)含胜局数和局数；[贪心学习曲线](learning_curves.csv)按各自父 S 拼接 Z/C，AUC 为阶段当前任务胜率相对实际步数的梯形积分。训练中 ε 探索的胜率没有混入贪心曲线。
- [ASPR-2G/RMSprop 原结果](../szc-vdn64-aspr-adam/README.md)为同种子、同父 S 的历史对照。2G 的实现和结果保持原样；不同分支从 Z 开始各自采集历史轨迹，后续轨迹不相同。
- 公开包保留紧凑 CSV 与配置，不含检查点、历史轨迹池或每次更新的完整 JSONL。远端原始运行目录仍保留这些产物。
