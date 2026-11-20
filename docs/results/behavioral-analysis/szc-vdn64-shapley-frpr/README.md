# SZC VDN64 · Shapley-Selected FRPR

用旧任务胜局上的单智能体实际动作分支，估计标准 FRPR 各偏好关系在当前 VDN-FT 下的行为贡献。行为 Pilot 对比 VDN-FT 贪婪、投影全部 FRPR 关系、投影等预算随机关系、投影按回报 Shapley 排序的关系。后续训练使用原始 FRPR 损失，行为投影用于离线评价。

## 配置与运行状态

- SZC 使用双层 VDN64 和镜像 5v5 Protoss。训练检查点种子为 3535 和 2222。每个种子覆盖学完 Z 评 S、学完 C 评 S、学完 C 评 Z。旧任务阶段末 FRPR 模型提供偏好，独立 VDN-FT 的 Z/C 阶段末模型提供当前 Q 和后续策略。检查点见 [pilot.json](pilot.json)。
- S 使用环境／SC2 种子 91101–91104，Z 使用 91201–91204。每个条件最多采样 9 个决策。按回合实际长度三等分，并按原 Action Regret 的归一化损失 [0, 0.1)、[0.1, 0.5)、[0.5, +∞) 分档，每个交叉分层最多一个。每个候选动作在相同初始化下重复两次。区间描述配对重复，在确定性回放下可以退化为零宽度。
- 使用标准 FRPR 参数 min_gap=0.5、q_scale_floor=0.1、损失温度及权重均为 1。投影 margin 为 0，行为 Softmax 温度为 1，行为对照 K 为 3。抽样后记录并跳过 M > 12 的决策，每个获准决策枚举至多 4,096 个联盟。per_stratum 支持独立配额及全量采样。
- Ubuntu 的真实环境 Smoke 与六个 Pilot 任务均已完成，运行组为 szc_vdn64_shapley_frpr_pilot_20261119_attempt001。六组共抽样 54 个决策，43 个进入精确计算并完成，510 个逐动作分支均通过核验。Smoke 复用已归档的 57 步 seed 91101 胜局，配对 VDN-FT seed 2222。后续研究将本 Pilot 的 43 个决策纳入 Discovery，并在留出轨迹上评价冻结选择。 [独立验证](../szc-vdn64-shapley-frpr-validation/README.md)

## 参与者和联盟值

分析调用原 FRPR 的 reference_order 和共用的 full_rank_pairs。仅实际有效决策的合法动作参与，上三角去重，排除旧 Q 并列。合法动作的总体标准差应用原下限，按标准化后的绝对差筛选 min_gap，偏好方向由旧 Q 差的符号确定。

确定性初始化和旧轨迹逐帧核验复用原脚本。在同一历史动作下用 VDN-FT 在线和离线分别展开 GRU，核对 Q、贪婪动作和前后隐状态。全部合法动作分别在独立 SC2 进程中进行单点替换。其他智能体执行 FT 贪婪策略，干预后全队恢复 FT 贪婪。每条分支记录逐步奖励、从干预步起的回报、胜负、局长、终止原因及种子。环境和 GRU 沿用 atol=1e-6、rtol=0 及离散字段精确比较。

基准为 q_base = (q_ft − old_mean) / old_scale。每个联盟用 SLSQP 求最小欧氏距离投影，联盟关系约束 z_preferred − z_dispreferred ≥ projection_margin。求解器状态、原始约束残差和一阶驻点残差均检查。只对合法动作的投影 Q 做 Softmax，非法槽位概率为零。联盟值为所有合法动作概率与真实分支回报或胜负的乘积之和。空联盟 Softmax 期望与 FT 贪婪分支结果分别记录。对全部 2^M 个联盟精确计算两套 Shapley 并核对效率性质。联盟值来自有限动作分支。

交互采用加权二阶差分的 Shapley interaction index。正、负交互分别计入协同及冗余候选。弱单条贡献用同一决策内单条增益绝对值的中位数定义。跨状态汇总使用状态内名次、Top-K 频率及正贡献比例。回报和胜率在任一指标为常数时，状态内秩相关留空。两次重复的 Top-K Jaccard 衡量相同初始化下的复现度。

## 输出与复跑

manifest.json 记录两套检查点、配置、源码快照和运行时信息。逐 case 输出 contexts.csv、strata_inventory.csv、frpr_pairs.csv、action_returns.csv、coalition_values.csv、shapley_values.csv、interactions.csv、summary.csv 和 audit.json。原始轨迹及 trace_pool.th 留在实验主机。历史清点与可重放胜局分开统计。旧档缺少 Q 时，使用其参考检查点和实际动作展开，并核对贪婪动作。

从 Ubuntu 方法仓库根目录用新运行编号复跑：

```sh
.venv/bin/python scripts/plan_shapley_frpr.py \
  --config docs/results/behavioral-analysis/szc-vdn64-shapley-frpr/pilot.json \
  --output results/szc_vdn64_shapley_frpr_pilot_new \
  --smoke-task szc_vdn64_shapley_frpr_smoke_20261119_attempt001
.venv/bin/python scheduler/service.py submit results/szc_vdn64_shapley_frpr_pilot_new/plan.json
```

## 后续训练

对比 baseline（VDN-FT）、full_rank_preference、random_selected_frpr 和 shapley_selected_frpr。沿用 SZC 每阶段 2M、逐任务 ε、切换时清空 TD 池和相同评估种子。按种子共享 S 阶段起点。报告 Z 末 S 保持、C 末 S/Z 保持、当前任务胜率和 ACC/BWT。三组 FRPR 对照使用同一参照历史输入。随机和 Shapley 组每个决策的关系预算一致。

两个新方法均从 scripts/run.py 的 vdn_action_set 入口注册。相应方法块示例：

```json
{
  "shapley_selected_frpr": {
    "shapley_results": ["results/complete_offline_selection/stalker"],
    "selection_k": 3
  }
}
```

selection_ratio 可替代 selection_k，比例按每个决策向上取整。随机组使用同一个结果目录决定支持集，可配置 selection_seed。回合 ID 包含任务及实际偏好训练输入内容哈希，不受批次或回合列表顺序影响。逐状态重新计算并核对完整有向关系集合。可选 trace_pools 将阶段编号映射到 trace_pool.th，阶段、任务、布局和旧检查点哈希须匹配。覆盖不足写入 shapley_coverage.jsonl，在对应阶段 TD 更新前拒绝运行。损失复用原温度、缩放、Softplus 和聚合，唯一算式改动是掩码交集。全选时损失及梯度相同。

Pilot 的采样与枚举上限用于行为诊断。已核对的四个历史池各有 100 个胜局，M 均值为 10.75–11.02，中位数为 10–11，最大值 38。按 M > 0 的决策统计，约 36% 的决策超出 Pilot 上限。正式训练需要完整可重放历史和可执行的全覆盖计划。 [逐 M 计数](historical_inventory.csv) · [输入哈希及总体统计](historical_inventory_manifest.json)
