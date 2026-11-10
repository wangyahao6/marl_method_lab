# 方法与实现范围

算法与神经元干预是两个维度：例如QMIX是算法，ReBorn/ReDo是附加干预。以下为代码存在性与入口整理，不表示已逐项复现或验证论文效果。

| 算法 | baseline | ReBorn | ReDo回收 | 周期重置 |
| --- | --- | --- | --- | --- |
| QMIX | qmix | qmix_b | qmix_r | qmix_s |
| QPLEX | qplex | qplex_b | qplex_r | qplex_s |
| DMIX | dmix | dmix_b | dmix_r | dmix_s |
| RMIX | rmix | rmix_b | rmix_r | rmix_s |
| VDN | vdn | 无上游配置 | vdn_r | vdn_s |

原作者提供baseline和四个`_b`配置。`_r`/`_s`配置由对应baseline仅改变name补齐；不擅自给VDN增加ReBorn配置。learner里有分支不等于每个组合都已实测可用。

- 原训练循环`src/run.py`按name中的`_b`/`_r`/`_s`分别调用`reborn()`/`recycle()`/`reset()`。
- 调用时机是测试间隔计数达到T，原缺省T=2；不能直接说每两次梯度更新干预。检测通常使用最新replay batch。
- QMIX/VDN共用q_learner；QPLEX用dmaq_qatten_learner；DMIX用iqn_learner；RMIX用rmix_learner。
- `reset`只按作者代码作周期重置称呼，不直接等同其他论文的Reset或Shrink-and-Perturb。
- ReDo是ReBorn仓库自己的回收实现，不宣称与ReDo原作者代码逐项相同。KNIFE尚未实现。
- Agent保留ReBorn两层ReLU MLP后接GRU，以及其ActivateLayer，不能默认等于其他PyMARL网络结构。

## 单层前馈 VDN

| 原版 SMAC 静态任务 baseline 配置 | GRU 前的隐藏层 | 前馈与 GRU 宽度 |
| --- | --- | --- |
| `vdn` | ReBorn 的两层 Linear + ReLU | 64（默认） |
| `vdn_1ff` | 单层 Linear + ReLU | 64 |
| `vdn_1ff_128` | 单层 Linear + ReLU | 128 |

单层配置的 Agent 路径为 Linear → ReLU → A 诊断门 → GRUCell → 线性 Q 输出头。A 门为原仓库的冻结全一门，用于采集活动值。两套配置沿用 VDN 的共享控制器、求和 Mixer、QLearner 与其他训练参数。宽度同时作用于前馈层和 GRU，因此 64 与 128 的对比涉及两处容量变化。每种结构使用各自的 checkpoint。

通过 `scripts/run.py --suite smac --method baseline` 使用，`--algorithm` 指定 `vdn_1ff` 或 `vdn_1ff_128`，其余参数沿用同一入口的静态任务用法。TS6 单层基线通过 `scripts/run.py --suite ts6 --method baseline --ts3-role full --protocol configs/continual/ts6.json` 使用。从 A 起训到 J，每阶段 2M，保留连续回放与优化器状态。全程复用 VDN 训练及连续任务独立评估。TS5 单层基线使用 `--suite ts5 --protocol configs/continual/ts5.json`，按 TS7 的 A–E 双边精确阵容从头训练，每阶段 2M，使用逐任务 ε 衰减。按每 100k 步和阶段末取当前任务最近 32 局，用当前模型复算唯一前馈层 A（`agent.net[2]`）的状态加权活动。记录全宽归一化分数、top-3 和严格 >3 名单，不改变训练。

## 用户方法

`drivers/continual/protection.py`、`release_schedule.py`、`detection_common.py`为状态加权Mixer W/V软保护；`neuron_records.py`为名单和频次记录。导入自已停止的latest_ramp工程，不导入其旧队列或结果。

参考配置：`configs/continual/knife_fig6.json`。公式omega=c/mean(c)，A=sum(omega*abs(z))/N；阈值严格大于3，最近32训练回合，实际参数更新系数0.1，最新集合替换，退出后100000环境步线性恢复；未定义检测按原实现记录与处理。仅是已导入实现和参考设置，不固化为所有未来实验要求。

## 环境覆盖

- `--suite smac`：原SMAC静态地图，调用原训练循环；支持上表配置选择。
- `--suite continual`：SMACv2 task_drift/task_switch；目前只接入QMIX baseline/softprotect。
- ReBorn/ReDo/reset尚未接入连续任务驱动，未来做这些比较时必须补接阶段调度并说明口径，不能仅把静态地图入口的可用性当作连续任务支持。

## Configurable soft-protection locations (2026-10-30)

The continual QMIX driver now accepts any configured subset of A/W/V:
- A: `agent.net[2]`, the first shared-agent ReLU output (64 units). Associated parameters: `agent.net[0]` input rows/bias and `agent.net[3]` output columns. This is neither the second Agent ReLU nor the GRU.
- W: `mixer.hyper_w_final[2]`; V: `mixer.V[2]` (64 units each), with the same incoming rows/bias and outgoing columns as the previous W/V implementation.

For A, each state's activity is the mean absolute activation over agent histories. The state weight is the same counterfactual team-Q sensitivity c/mean(c) used for W/V. Thus A_j = sum_state omega(state) * mean_agent abs(z_agent,j) / N. Thresholding remains strictly >3 within each eligible layer. Shared Agent parameters are scaled once, not five times. Only configured layers are screened/protected/recorded. All-one frozen gates and shared output biases stay outside protection; target copies follow the original schedule. Protection scales the post-RMSprop increment, preserving optimizer state. Latest selection replacement and 100k-step linear recovery are unchanged.

`results/location_ablation_seed1111/implementation_validation.json` records offline checks on an existing 32-episode batch: explicit state-agent A statistic, exact old/new W/V equality, no model/RNG/hidden change, and real RMSprop increments at 0.1/0.55/1 for all four configurations. No new SC2 smoke training was run. Assertions in the offline checker are not in the training execution path.

## Optional phase control (2026-10-31)

Continual soft protection accepts zero-based `method.active_phases` and `method.detection_phases`. Outside active_phases, enter() clears both protected and recovering sets immediately (all increments scale 1), while retaining detection identities for provenance. detection_phases limits boundary screening. Omitting these keys preserves existing behavior. The ABA experiment explicitly selects W/V, screens phase 0 and protects phase 1 only; see docs/EXPERIMENTS.md.

## TRHR A2

The standalone TRHR intervention uses the existing two-layer VDN Agent on static SMAC and TS3 VDN branches. It adds bounded per-unit bias compensation at the second ReLU based on sustained low activation in training replay. See [TRHR A2](TRHR.md) for the update rule, conditional screening, checkpoint behavior and launcher options.

## TS3 Agent schedules (2026-11-02)

`ts3_protection.py` adds single-layer A1/A2 actual RMSprop scaling with either existing ReleaseSchedule linear recovery or permanent history union. Shared protection parameters and overall state-weighted screening, no per-type selector in this experiment. See [TS3 protocol](TS3_AGENT_PROTOCOL.md). This does not change old heterogeneous ABA semantics.

## VDN 历史探测集重评估（CHI）

在 TS3、TS5、TS6、TS7 的共享 Agent 序列驱动中，通过 `scripts/run.py --method softprotect_chi` 启用。两层 VDN 使用 `--ts3-group A1_chi` 或 `A2_chi`，单层 VDN 使用 `A_chi`。支持 `--ts3-role full` 从 A 开始训练，也支持用同一方法以 `--ts3-role prefix --ts3-group N` 保存 A 的探测集，再以 `--ts3-role branch --parent <prefix目录>` 从 A 续训。分支引用父运行的固定探测集。仅持有模型权重的旧检查点缺少固定探测集，无法用来还原原训练轨迹及权重。

每个阶段保留最近至多 32 局完整训练轨迹。阶段末用在线 Agent 计算 VDN 单智能体替换动作的最大 Q 差，在有效存活智能体间取均值作为状态协作敏感度 `c`，再按该探测集的有效状态均值计算 `omega = c / mean(c)`。将轨迹及对齐的权重保存为 `chi_probes/phase*.th`，与训练 replay 分离。重评估时逐局从零隐状态开始，按时序重放观测、动作历史和 agent ID，复用固定权重。排除死亡智能体、无效动作、padding 和仅用于 bootstrap 的末状态。

每个任务先计算 `mean_state(omega * mean_live_agent(abs(activation)))`，再除以本任务的层内均值。对本轮各任务的归一化分数逐神经元取最大值。选择沿用配置的全层 top-k 或严格阈值，分数并列时优先较小索引。下一阶段保护名单以本轮结果替换，选中神经元的实际 RMSprop 参数增量系数为 0.1，退出名单后沿用 100000 环境步的线性恢复。本轮无有效分数时使用空名单，探测或重评估异常另记 `chi_errors.jsonl`。每轮逐任务分数、最大值和名单写入 `chi_assessments.jsonl`。

设计借鉴 SPG §3.2 中随网络变化重新评估历史任务重要性的思路。VDN 使用共享 Q 输出和各历史任务自己的固定探测集，以无梯度的响应统计评估。参见 [Parameter-Level Soft-Masking for Continual Learning](https://proceedings.mlr.press/v202/konishi23a.html)。

## TS4 VDN (2026-11-02)

--suite ts4 --algorithm vdn supports baseline and A1/A2 single-layer soft protection; additive-VDN sensitivity replaces QMIX hypernetwork calculation. Other continual suite restrictions remain unchanged. See [protocol](TS4_VDN_PROTOCOL.md).


## TS3 VDN A2 local GEM (2026-11-04)

Authorized existing-baseline seeds1234/4567/3535, only high/middle/low method branches,21A2units each, local actual RMSprop increment projection. Reuse original2M A prefixes, B-E8M each, nine branches72M new nominal steps. Latest-boundary groups, no neuronal history union or 0.1 scaling. Three baseline FWT jobs are evaluation-only. Resource-aware max5new workers; no interference with existing queues. See [protocol](TS3_VDN_A2_LOCAL_GEM_PROTOCOL.md). Queue `scripts/run_ts3_vdn_local_gem.py`, results `results/ts3_vdn_A2_local_gem_20261104/`. Launch/queuing is not completion.

## 方法设计参考

[MEAL 机制参考](MEAL_DESIGN_REFERENCE.md)及[固定版本源码](../references/meal/README.md)用于自己的神经元机制和持续学习设计。EWC/MAS/A-GEM/CBP 在这里是参考材料，不是新增的可运行方法或待执行对照。方法确定并取得初步结果后再考虑对照实验。
