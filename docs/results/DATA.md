# 公开结果的数据说明

## AI 分析入口

从 [catalog.json](catalog.json) 选择实验 ID。先读该实验的 `comparison` 和 `interpretation_limits`，再用 `run_id` 关联运行清单与指标。优先用 `stage_metrics` 比较端点，分析学习曲线或复算指标时再读 `evaluations`。按需读取配置与诊断，图表是同一数据的阅读视图。

索引中的数据路径相对于 `catalog.json`，运行清单中的配置路径相对于清单所在目录。CSV 的空单元格表示缺失或不适用，保留为 null。对照选择器的字段取自 `runs.csv`，配对分析只在声明的 `match_on` 字段内比较。不同实验按各自的任务、网络和协议独立解释。

## 从运行定位到数据

| 文件 | 内容 |
| --- | --- |
| `runs.csv` | 一行一个实际运行。记录 seed、训练源码 commit、ε 调度、TD 回放模式、直接父运行、累计与新增环境步数，以及配置链接。 |
| `evaluations.csv` | 从 `evaluations.jsonl` 提取的一行一次独立贪心评估。用 `wins / episodes` 重算胜率。阶段编号 `phase` 从 0 开始，`t_env` 是累计训练环境步数。 |
| `stage_metrics.csv` | 由评估计数重算的阶段末逐任务胜率、ACC、BWT、FWT，以及有回访曲线时的回训 AUC。空字段表示该阶段无对应指标。 |
| `preference_diagnostics.csv` | 从偏好回放逐更新记录按运行、当前阶段及历史来源任务汇总。覆盖率为 `valid_states / decision_states`，计数包含重复采样；`valid_pairs` 等字段按实现实际记录。 |
| `paired_final.csv` | SZC 同种子终点差值，`run_id` 是方法运行，`control_run_id` 是其对照运行。差值均为方法减对照，单位为百分点。 |
| `gradient_diagnostics.csv` | 按运行、阶段和指标汇总 Agent 梯度抽样点的均值和中位数。`logged_points` 含未定义点，均值与中位数使用 `defined_points`。抽样间隔见生效配置。 |
| `effective_k_histogram.csv` | TKPR 按历史来源任务的有效 K 计数。每个计数是含至少一个约束对的采样决策，含重复采样。 |
| `replay_samples.csv` | ER + CHI 每个阶段首个 TD 批次的任务分配，粒度为运行、阶段、采样来源任务。 |
| `configs/<run-id>.json` | 运行快照中的配置、任务协议、环境维度与分支恢复记录。实验主机的仓库绝对路径转写为仓库相对路径。`launch_git_commit` 可用于查阅实际训练版本。 |

SZC 的 `runs.csv` 中 `cohort` 标识原始提交批次。`preference_diagnostics.csv` 的筛选率和 Top-1 翻转率由累计计数相除。`mean_task_loss` 是有定义的历史任务损失在各更新间的平均，分母为 `task_loss_updates`。旧日志缺少的细项留空。Raw 与归一化变体的损失尺度不同，比较时同时查看覆盖率、动作对与梯度。

MinAtar 单智能体结果使用原始每局回报：其 `evaluations.csv` 的 `mean_return`、`stage_metrics.csv` 的逐游戏回报及 `learning_curves.csv` 的探索策略最近 100 局回报均不是胜率。三个游戏的奖励尺度不同，不计算跨游戏 ACC。`retention.csv` 的 BWT 为最终 100 局平均回报减首次学完后 20 局平均回报。`preference_diagnostics.csv` 每 1,000 环境步抽样一次；完整逐更新和逐局日志保留在实验主机。

MinAtar 历史动作保持分析的 `action_retention.csv` 以 `(method, game, reference_stage, current_stage)` 为键。`har` 是同一固定观测上两次贪心动作相同的比例，`adr=1-har`；`mean_old_q_regret` 等分布统计来自历史模型对两个动作的 Q 差。`states/*.npz` 为两方法共同使用的原评价种子贪心轨迹观测，`pointwise/*.npz` 保留每个对比的旧 Q 估值与损失。`historical_performance.csv` 复用原来的阶段末贪心回报。Q 差不跨游戏合并，动作保持也不等同于真实回报保持。

SZC 共享前缀联合动作保持分析的 `joint_action_retention.csv` 以 `(seed, method, phase)` 为键，指标是 32 局独立 S 轨迹的逐局等权均值；`episode_metrics.csv` 保留每局分子、分母，`per_agent_retention.csv` 保留五名智能体的逐局汇总。轨迹数据 `historical_trajectories_seed*.th.gz` 包含完整序列；清单记录原检查点和轨迹哈希。`IAR` 仅含存活智能体，`JAA` 与改变人数仅含至少两人存活的时间步，均为固定历史轨迹上的离线动作比较。

评估标签中，`initial_reference` 是初始模型在将来任务上的表现，`next_unseen` 是首次训练该任务前的评估，`phase_start` 和 `periodic` 描述本阶段当前任务的学习过程，`stage_end` 用于比较该阶段末各已见任务。曲线的圆点为阶段末评估。定期与阶段末评估的局数与种子块可不同，见各实验的协议。

## 重算阶段指标

- **ACC** 为该阶段末所有已见任务胜率的等权平均，单位是百分比。
- **BWT** 为除当前任务外，各已见任务的当前胜率减去各自最近一次训练完成时的胜率后取平均，单位是百分点。
- **FWT** 为已进入训练且非首个任务的首次训练前胜率减去初始模型参考胜率后取平均，单位是百分点。同一任务回访沿用其首次进入时的 FWT。
- **回训 AUC** 在回访的实际训练步数上对当前任务胜率做梯形积分，再除以实际回访步数，单位是百分比。

有共享前缀时，子运行的 `evaluations.csv` 记录只覆盖其新增阶段。重算子运行的 BWT、FWT 时按 `parent_run_id` 取父运行数据。多个分支共用的前缀只记录一次，评估局数也只用于估计该模型的表现。多种子报告按训练种子等权。

## 归档后续结果

按实验的主要问题选择目录。方法与基线放入 `comparisons/`，回访与阶段分支放入 `return-ablations/`，超参数与组件放入 `parameter-ablations/`，机制诊断放入 `diagnostics/`。等到该类有公开结果时再建目录。

每组用一个稳定的子目录，报告开头说明问题与实际协议，列出对照、负结果和有解释价值的差异，再链接轻量数据、图与运行记录。用训练时的提交标记旧结果，控制实验尚有版本或驱动差异时在对应报告中说明。同一运行或报告被其他类别引用时，链接原记录。发布时同步维护[机器可读索引](catalog.json)与[阅读索引](README.md)，核对数据路径、运行 ID、配置 commit 和对照／共享前缀关系。新类型可以由索引指向专用数据；沿用既有字段时保留原语义和单位。完整日志、模型、轨迹与调度文件保存在实验主机。
