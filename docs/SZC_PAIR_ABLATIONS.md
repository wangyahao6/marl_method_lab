# SZC-1M 动作对消融第一版

双层 VDN64，seed 3535，5v5 Protoss，Stalker → Zealot → Colossus。
沿用 SZC-1M 的累计切换阈值 1M／2M／3M、逐任务 ε 重启、FT 清空 TD 回放池和 RMSprop。
共用参考线仅执行 TD 更新，在 S 阶段结束后保存分支快照，并生成 S、Z 两个胜局历史池。
11 个分支在 S 结束处恢复同一套在线网络、目标网络、优化器和随机状态。Z 阶段使用共同的 S 池，
C 阶段使用共同的 S、Z 池。各分支的历史回放采样使用相同的种子和规则。当前任务的在线轨迹会随策略更新而变化。

两个实验均按旧 Q 严格比较筛选动作对。当前 Q 差除以该位置合法旧 Q 的总体标准差，分母下限为 0.1。
Softplus 的 temperature 和辅助权重均为 1，min_gap 为 0。每个状态先均分各选中对的损失，
再对采样批次内的全部有效状态等权求均值，包括来自不同历史任务的状态。

## 实验一 · 动作对结构

[协议](../configs/continual/action_preference/pair_structure/szc_vdn64_pair_structure_1m.json) · 运行目录 `runs/pair_structure/`。

五组共用合法动作数至少为 4 且所有合法旧 Q 均不相等的位置。

| 组 | `--method` | 选择 |
| --- | --- | --- |
| TKPR-K1 | `pair_structure_k1` | 全部与旧 Top-1 相连的对 |
| Adjacent | `pair_structure_adjacent` | 按旧 Q 降序的 N−1 条相邻边 |
| TKPR-K1 + Adjacent | `pair_structure_k1_adjacent` | Top-1 对和相邻对的并集，共 2N−3 对 |
| FRPR | `pair_structure_frpr` | 合法且旧 Q 不相等的全部无序对 |
| Random-N-1 | `pair_structure_random_n_minus_1` | 从候选对无放回抽取 N−1 对 |

对以旧 Q 降序定向的边，任意相邻名次之间没有中间名次可供绕行。
因此，当且仅当选中全部相邻边时，传递闭包包含完整严格排序。历史池摘要记录满足该条件的位置数。
对应单元测试直接计算传递闭包，验证 Adjacent、TKPR-K1 + Adjacent 的闭包覆盖全部严格关系，且 TKPR-K1 缺少非 Top-1 之间的关系。

## 实验二 · Top-1 约束比例

[协议](../configs/continual/action_preference/pair_ratio/szc_vdn64_pair_ratio_1m.json) · 运行目录 `runs/pair_ratio/`。

比例组为 Ratio-0、Ratio-20、Ratio-40、Ratio-60、Ratio-80、Ratio-100。
CLI 标识依次为 `pair_ratio_0`、`pair_ratio_20`、`pair_ratio_40`、`pair_ratio_60`、
`pair_ratio_80`、`pair_ratio_100`。六组共用至少 4 个合法动作、旧 Top-1 唯一且非 Top-1
候选对不少于 N−1 的位置。每个位置均选 N−1 对，Ratio-100 在这套共同状态上与 TKPR-K1 动作对一致。

按固定种子打乱状态顺序，再对累计预算乘以目标比例并取最近整数。相邻累计值之差是当前状态的
Top-1 配额，剩余配额从非 Top-1 候选对中补齐。S 池与 S+Z 的 Top-1 总数都取到目标配额的最近整数。
每个历史池加入训练前生成一次掩码，训练更新只按回合索引读取。

## 诊断

- `pair_ablation_memory/phase*/summary.json` 记录候选、有效状态和选对统计、源历史池及掩码摘要。
  比较各组的 `valid_state_sha256` 检查共同状态。结构组记录 `complete_order_states`，比例组记录 `budget_mismatch_states`。
- `pair_ablation_memory/phase*/selection.th` 按源池回合顺序保存布尔掩码，张量轴为决策时刻、Agent、动作对。
  动作对轴按 `torch.triu_indices(N_total, N_total, 1)` 的原始动作索引排列。
- `<method>_updates.jsonl` 记录每次更新的偏好损失、聚合及分历史任务的有效状态数、覆盖率、全部合法候选对数、
  排除旧 Q 并列后的候选对数、选中对数、Top-1 和非 Top-1 对数、实际比例、每状态平均对数及压缩率。
  覆盖率分母为真实存活 Agent 的决策位置数。压缩率为 1 − 选中对数 / 有效状态中排除并列后的候选对数。
- `experiment_metrics.jsonl` 每 100 个 learner 更新记录加权偏好损失对 Agent 参数的梯度范数
  `preference_grad_norm_agent`。

## 启动

在方法仓库根目录执行。第一条命令只调用 12 次启动预览并写出计划。第二条命令提交共用参考线和两个实验的分支，
所有分支依赖参考线成功结束。每个新批次使用唯一标识。

```sh
.venv/bin/python scripts/plan_pair_ablations.py szc_pair_v1_3535 > /tmp/szc_pair_v1_3535.json
.venv/bin/python scheduler/service.py submit /tmp/szc_pair_v1_3535.json
```

单元测试使用合成 Q 和脚本化 SMAC 接口验证选对、预算、固定掩码、传递闭包、共享恢复和 VDN 梯度。

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p test_pair_ablation.py
```
