# SZC-1M：Q-gap 匹配的非 Top-1 动作对

本实验只新增 `pair_gap_matched_non_top1` 一个训练分支，复用 seed 3535 的既有 TD 参考线、K1 和 Ratio-0：

- 参考线：`runs/pair_structure/szc_pair_matched_3535_20261119_reference`
- K1：`runs/pair_structure/szc_pair_matched_3535_20261119_pair_structure_k1`
- Ratio-0：`runs/pair_ratio/szc_pair_matched_3535_20261119_pair_ratio_0`

三条旧运行均由 `c6b7002560c9827061e2f543be318f32cafb5622` 启动。K1 与 Ratio-0 的父运行相同，S 末恢复模型哈希均为 `54c9f68d05a423e84bcea663d1d8102dee62382f4e7792bbce0d39bc2d37af0f`；S、Z 池来源分别是参考线的 `action_set_memory/phase0000/phase0001/success.th`。不重新训练或改写这三条运行。

新方法在每个有效 Agent 决策位置保留与 K1 同样的 `N−1` 对预算，只从非 Top-1 候选中选择。用 `reference_order` 的合法旧 Q 总体标准差及下限 0.1 计算归一化 Q-gap。K1 gap 排序后，动态规划从非 Top-1 候选中无放回选出 `N−1` 个，使两个排序序列的绝对差之和最小；相同目标值以原始动作对索引的字典序稳定裁决。平均绝对差即等样本量一维经验 Wasserstein-1 距离。`min_gap=0`，temperature=1，辅助权重=1，`pair_seed=3535` 仅作协议记录，选择本身完全确定。沿用现有动作对 Softplus 损失、状态内对均值、任务内有效状态均值和有效历史任务间等权平均。

预训练核验从原参考线保存的成功轨迹及旧 Q 逐池重建 K1、Ratio-0 与新方法的掩码。S 池和 Z 池有效状态分别为 21,944 和 22,334 个，三组状态哈希逐池完全一致。新方法分别选 124,392 和 118,223 对，均无 Top-1 对，每个有效位置恰为 `N−1` 对。原始池 SHA-256、掩码 SHA-256、逐状态 W1 和相对误差记录在独立批次 `results/szc_pair_gapmatched_3535_20261123/preflight/`，完成后抽取必要表格公开。

| 历史池 | K1 平均 Q-gap | Ratio-0 平均 Q-gap | 匹配后平均 Q-gap | 平均 W1：Ratio-0 → 匹配后 | 匹配后相对误差均值 / P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| S | 1.161 | 0.550 | 0.793 | 0.639 → 0.395 | 0.289 / 0.622 |
| Z | 1.233 | 0.553 | 0.790 | 0.689 → 0.456 | 0.320 / 0.653 |

匹配是**给定非 Top-1 候选集合下的最优解**，并未消除 gap 差距。K1 的较大 gap 尾部通常无法由非 Top-1 候选重现，因此本批不能作为“完全匹配 Q-gap 后仅改变 Top-1 身份”的严格因果对照。后续性能差异仍可能来自残余 Q-gap、排名距离或动作对图结构。两种实验启动提交也不同；当前实现重建旧掩码逐元素一致，旧 K1／Ratio-0 的损失公式与新方法沿用的 `ablation_loss` 核心项相同，新增偏好诊断不改变损失。

[协议](../configs/continual/action_preference/pair_gap_matched/szc_vdn64_pair_gap_matched_1m.json)仅复制旧结构实验的训练、评价与回放设置，并注册一个新方法。批次计划只含一个依赖旧参考线的任务：

```sh
.venv/bin/python scripts/diagnose_pair_gap_matched.py
.venv/bin/python scripts/plan_pair_gap_matched.py szc_pair_gapmatched_3535_20261123 > results/szc_pair_gapmatched_3535_20261123/plan.json
.venv/bin/python scheduler/service.py submit results/szc_pair_gapmatched_3535_20261123/plan.json
```

训练完成后在 `docs/results/parameter-ablations/szc-vdn64-pair-gap-matched/` 归档逐阶段胜率、ACC/BWT/FWT、Z/C AUC、学习曲线、偏好与梯度诊断，以及 S/Z 分池 Q-gap、W1 和排名距离。单种子仅作为探索性比较，不作统计显著性结论。
