# MEAL：神经元机制与持续学习设计参考

2026-11-09。用户明确用途：将 MEAL 作为自己的神经元机制和持续学习方法设计参考；对照实验等自己的方法确定并取得初步结果后再开展。本次仅归集参考材料和接入分析，不新增基线、实验配置、队列或训练方法，不改变现有运行。

作者源码固定提交 `8e7857c76eedc44623a7bfe987235f2a9aadb626`；[五个重点源码及许可](../references/meal/README.md)。本仓库接入位置基于 `ef33f9e5a68a39421cb0bae77f451f59ed83778e` 的只读检查。下述“可借鉴”是设计选项，不表示用户已采用、已经实现或已验证效果。

## 最有用的设计启发

| 设计问题 | MEAL 提供的材料 | 在我们这里如何使用 |
| --- | --- | --- |
| 低活动是否等于低效用？ | CBP 将激活幅度与输出权重结合 | 作为状态加权活动、TRHR 激活率之外的解释维度；不直接替代已有筛选 |
| 怎样避免刚生成/刚调整的单元马上再次被干预？ | EMA、成熟期、按 eligible 数量累积替换额度 | 将时间记忆、成熟期、干预速率分别定义；具体数值不照搬 |
| 历史知识应该保留多少？ | importance 的 last/sum/EMA 三种历史规则 | 用来设计衰减、持续保护和释放；历史长度与保护强度分开考虑 |
| 保护应约束参数位置还是更新方向？ | EWC/MAS 参数锚点与 A-GEM 梯度方向 | 与现有更新幅度缩放区分；可用于解释和设计机制，不立即组织对照 |
| 方法状态如何跨任务延续？ | train_state 与 cl_state 分离 | 显式列出统计量、记忆、参数锚点的生命周期与恢复内容 |

## 神经元机制：优先借鉴效用定义与干预时间尺度

[cbp.py](../references/meal/upstream/experiments/continual/cbp.py) 中，单元贡献为 `mean(abs(h_j)) * sum(abs(W_out[j,:]))`（Flax 权重 [in,out]）。效用使用指数平均，年龄随 train 前向增加。只在 age > maturity 的单元中排序，计数器累积 rho × eligible 数量，替换整数个低效用单元，保留小数额度。输入权重重初始化、偏置与输出连接清零、效用和年龄重置。

对当前 TRHR 的启发是把“活动不足”“输出贡献低”“长期持续不足”分开。TRHR 已有有效样本掩码、条件分组、EMA 和连续低活跃窗口，不能仅因 MEAL 也有 EMA 再加入第二套控制器。CBP 的输出权重项可用于后续思考功能贡献，但大权重或大激活都不是因果重要性证明；罕见状态中的关键单元也可能总体低效用。

MEAL 的成熟期意味着暂时不再替换；我们的 TRHR 是补偿而非重新初始化，是否需要调节后的等待期、何时释放保护，应按自己的机制决定。不要将 CBP 的 maturity 或 replacement rate 数值直接套给 TRHR，也不自动把低活动单元改成待删除名单。

实现细节不能照搬：

- CBP 层名写死为 actor_fc1/fc2、critic_fc1/fc2；VDN 的方法表没有 CBP。
- 文件没有实现所有 CBP 文献变体，例如效用 EMA 偏差修正、中心化贡献等；准确称呼为这个仓库的实现。
- 实验 CBP 调用只替换 params/cbp_state，未按被替换位置清理优化器张量状态。PyTorch/RMSprop 移植时必须明确这一点。
- age 的单位是 train 前向调用；在我们的 replay+GRU 中，应另行明确按 learner 更新还是有效样本计时，避免旧任务参考前向、target 和评估污染。
- 清零输出连接同时删除原单元贡献，不是精确函数保持。

若自己的方法最终确实需要替换，当前参数映射可复用：双层 A1 是 net[0] 输入行/bias → net[3] 输出列；双层 A2 是 net[3] 输入行/bias → GRU weight_ih 的输入列（全部门）；单层 A 使用唯一 Linear → GRU。这里是 GRU 前单元，不是直接重置 GRU hidden 单元，不能误改 weight_hh。GRU 输入门权重之和作为输出强度属于我们的适配假设。

## 历史重要性：把“分数累积”与“参数锚点”分开

[ewc.py](../references/meal/upstream/experiments/continual/ewc.py)、[mas.py](../references/meal/upstream/experiments/continual/mas.py) 提供 last、multi、online 状态更新：分别替换、求和、指数平均。三者都将 old_params 改成最近任务末参数；multi 不是保存所有旧任务的独立锚点。

惩罚为 `0.5 * coef * sum(mask * importance * (theta-anchor)^2) / sum(mask)`。这与目前软保护缩小实际增量不同：前者产生拉向锚点的梯度，后者改变更新幅度。重要性、mask、锚点、归一化尺度应独立记录；不要把长期累积分数造成的惩罚增强误认为识别更准确。

关键调用事实：VDN 不使用 EWC 类中的策略 Fisher estimator，而让 EWC 和 MAS 共用 vdn.py 的 make_q_importance_fn：对所有 Agent/动作 Q 平方范数求参数梯度，再平方并平均。它是 Q 输出敏感度平方，不是标准策略 Fisher。MAS 默认还在 update_state 中归一化新重要性；EWC 没有同样步骤。多 Agent 输出先聚合再求梯度平方，与各 Agent 梯度平方求和不同。

该 estimator 的 stride 命中才累积，但分母仍是总采样步数，未归一化时 stride 会改变强度。SMAC 中若借鉴输出敏感度，需明确有效动作/时刻，不能把非法动作屏蔽成极大负值后求 Q 范数。重要性可以按单元参数组汇总用于设计讨论，但这不是已证实的协作重要性，更不是必须采用的新筛选规则。

## 历史梯度：平均约束是一种可调设计思想

[agem.py](../references/meal/upstream/experiments/continual/agem.py) 的 VDN 记忆按任务分槽。旧数据使用当前 online/target 计算 Double-Q TD 目标并 detach；各旧任务梯度等量采样后相加，形成一条参考方向。冲突时做半空间投影。合成约束允许任务间抵消，不同于逐任务约束。

在自己的方法设计中，可以借此区分保护强度、覆盖参数集合、历史任务权重三个因素；不必把完整 A-GEM 当作下一项实验。现有 a2_local_gem.py 已有 episode 记忆、参考 MAC/target 副本、参数切片、实际 RMSprop 增量与逐任务约束，实现另一种方向规则时可沿用这些部件。

保留这些边界：

- MEAL 的 VDN 无条件清零 q_head 的参考梯度；我们的单头网络不能无说明排除输出层。
- MEAL 基础 TD 用 target max，记忆梯度用 Double-Q；两者也未包含我们的完整合法动作处理。应复用本仓库 TD 语义。
- 独立 transition 不能代替 GRU episode 重放，需保留 filled、terminated、avail_actions 等输入。
- MEAL 在自适应优化器前投影梯度；本仓库实际增量投影更直接对应参数位移。局部方向约束也不能限制未保护参数对完整损失的影响。
- TD 约束不是旧任务胜率或协作保持保证；统计下降不能替代独立评价。

## 任务流与状态管理：已有能力优先复用

[vdn.py](../references/meal/upstream/experiments/algos/vdn.py) 保留 online/target，切换时重置 optimizer、n_updates 和 grad_steps，重新开始 epsilon 进度。它没有我们这种长期训练 replay，且支持显式任务 ID/多头。不要为了借鉴 CL 状态就改变执行策略的信息、输出头、优化器或回放规则。

仓库已有 `epsilon_on_switch` 和 `log_task_epsilon` 阶段日志。新实验使用逐任务 ε 退火，历史对照采用各自的启动协议。 [配置索引](../configs/continual/README.md)

当前 QLearner 已在在线 TD 前向使用 capture_activity，TRHR 与 A2 trace 复用统计入口。新统计应在同一有效样本口径下设计，避免另外注册无差别 forward hook。TRHR 按 window_updates 和有效条件处理低活动，详见 [TRHR](TRHR.md)。

MEAL 在线评价当前模型的全部任务，能显示遗忘过程；我们已有阶段胜率矩阵、初始参考、下一任务训练前评价，应继续沿用。论文 §4.4 的 AUC 型 FT 和时间加权遗忘与本仓库 FWT/BWT 口径不同，不直接改名替换。仅保存 params 的 MEAL 检查点也不能替代 continuation/prefix_state。

## 后续设计真正落地时的接入位置

| 当前模块 | 可承接的职责 | 现在的处理 |
| --- | --- | --- |
| src/learners/q_learner.py | TD loss、有效掩码、统计采集、更新后的干预位置 | 保留现状，记录接口 |
| src/modules/layers/trhr.py 与现有 A2 trace | 活动率、条件统计、时间状态 | 优先复用，不增加重复控制器 |
| drivers/continual/ts3_protection.py | 单元到参数切片、保护集合和调度 | 用于后续设计定位，不改筛选 |
| drivers/continual/a2_local_gem.py | 旧任务 episode、参考 TD 梯度、实际位移投影 | 作为现成基础，不新增 A-GEM 分支 |
| drivers/continual/ts3_train.py | 边界处理与状态恢复 | 未来保存新方法必要状态，不改旧协议 |
| Agent 与 checkpoint | 若采用替换，处理输入行/输出列、优化器状态和方法缓冲区 | 机制确定后再实现 |

本次有意把材料保留在 references/docs，不将未选定机制加入 src 或方法注册表。后续以用户自己的方法设计为中心取用这些部件；对照实验在方法确定并获得初步结果后再单独设计。本次没有性能结论、没有新增训练预算或后台任务。
