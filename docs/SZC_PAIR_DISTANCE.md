# SZC-1M Top-1 加额外动作对实验

本批使用双层 VDN64、SMACv2 5v5、Stalker → Zealot → Colossus，每阶段 100 万环境步，训练种子 2222、3535。每个种子独立生成一条 TD 共同参考线；同种子的九个偏好分支都从其 S 末快照恢复模型、目标网络、优化器和随机状态，并使用该参考线保存的 S、Z 胜局历史池。当前任务在线轨迹仍随分支策略变化。

[实验 A 配置](../configs/continual/action_preference/pair_distance/szc_vdn64_pair_distance_1m.json)和[实验 B 配置](../configs/continual/action_preference/pair_gap_far_mid/szc_vdn64_pair_gap_far_mid_1m.json)沿用上一批 1M `pair_structure` 协议的优化器、回放、探索和评估设置。偏好权重和 Softplus 温度均为 1，合法旧 Q 的总体标准差下限为 0.1；`min_gap=0`，Top-1 对不经过间隔筛选，不使用状态置信度权重。每状态先平均所选动作对，再在历史任务内平均有效状态，最后等权平均有有效状态的历史任务。原有 TKPR、FRPR 和旧消融方法不变。

## 实验 A：额外对的排名距离

共同有效决策位置要求至少 4 个合法动作，且合法动作旧 Q 无并列。严格按旧 Q 降序确定排名，排名相邻距离为 Near=1、Mid=2、Far≥3。全部方法都保留 `N−1` 个 Top-1 对。`N≥5` 时，六个扩展组各选 `B=min(N−2,N−3,(N−4)(N−3)/2)` 个非 Top-1 对；`N=4` 时 `B=0`。

| 组 | CLI 方法 | 额外动作对 |
| --- | --- | --- |
| K1 | `pair_distance_k1` | 0（基线） |
| K1+Near | `pair_distance_near` | Near 随机 B 个 |
| K1+Mid | `pair_distance_mid` | Mid 随机 B 个 |
| K1+Far | `pair_distance_far` | Far 随机 B 个 |
| K1+Far/Mid | `pair_distance_far_mid` | Far 约 2/3、Mid 约 1/3，合计 B 个 |
| K1+Random-Matched | `pair_distance_random_matched` | 所有非 Top-1 对中随机 B 个 |
| K1+MaxGap | `pair_distance_maxgap` | 归一化旧 Q 差距最大的 B 个 |

K1 是无额外约束基线，所以其总对数为 `N−1`；六个扩展组总对数均为 `N−1+B`，Top-1 对和额外预算逐状态一致。固定 `pair_seed=3535` 在各历史池进入训练前一次性抽样，原始动作索引的上三角布尔掩码存于 `pair_ablation_memory/phase*/selection.th`，更新时直接读取。随机评分在同一池上跨方法一致；MaxGap 同分时按原始动作对索引稳定裁决。

## 实验 B：只对额外 Far/Mid 对筛选

两个方法分别为 `pair_gap_far_mid_unfiltered`（额外对无门槛）和 `pair_gap_far_mid_filtered`（额外对的归一化旧 Q 差距至少 0.5）。两者保留全部 Top-1 对，且使用相同 Far/Mid 配额和 B。共同有效位置要求实验 A 的条件成立，并要求通过 0.5 门槛的 Far、Mid 候选各自足以填满配额；不满足者在两组中统一排除。`N=4` 无额外对，门槛对该位置无影响。历史池摘要记录 `excluded_by_extra_gap_states`，可与实验 A 的状态覆盖率比较。

实验 B 的有效位置是实验 A 的子集，因此实验 B 与实验 A 的 K1 端点差值只可作描述性参考；若要严格在同一位置解释结构作用，应以归档旧 Q 和固定掩码在实验 B 共同位置重算 K1 诊断。

## 启动与归档

批次计划由下列命令生成，包含每个种子 1 个共同参考线和 9 个依赖分支，共 20 个调度任务。先检查 `--dry-run` 生成的计划，再在远端 Ubuntu 调度器提交。不得在本地 Mac 直接训练。

```sh
.venv/bin/python scripts/plan_pair_distance.py szc_pair_distance_2222_3535_20261122 > results/szc_pair_distance_2222_3535_20261122/plan.json
.venv/bin/python scheduler/service.py submit results/szc_pair_distance_2222_3535_20261122/plan.json
```

固定掩码摘要记录有效状态及哈希、候选与选中对数、Top-1 比例、Near/Mid/Far 对数、间隔筛除数、完整排序相邻边覆盖及掩码哈希。逐更新日志记录偏好损失、选中动作对违反率；既有梯度日志记录偏好和 TD 梯度范数及余弦。训练结束并核验全部运行与评估后，实验 A、B 分别归档于 `docs/results/parameter-ablations/` 下独立目录，提供 `README.md`、`runs.csv`、`stage_metrics.csv`、`pair_diagnostics.csv`、`paired_metrics.csv` 与配置快照。报告分别列出 ACC 和 BWT 的逐种子值、均值、标准差、K1 配对差值及 Pareto 关系，不合并成单一分数。

```sh
PYTHONPATH=src:drivers/continual OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
```
