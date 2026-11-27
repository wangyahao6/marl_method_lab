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

通过 `scripts/run.py --suite smac --method baseline` 使用，`--algorithm` 指定 `vdn_1ff` 或 `vdn_1ff_128`，其余参数沿用同一入口的静态任务用法。TS6 单层基线通过 `scripts/run.py --suite ts6 --method baseline --ts3-role full --protocol configs/continual/task_transfer/ts6.json` 使用。从 A 起训到 J，每阶段 2M，保留连续回放与优化器状态。全程复用 VDN 训练及连续任务独立评估。TS5 单层基线使用 `--suite ts5 --protocol configs/continual/task_transfer/ts5.json`，按 TS7 的 A–E 双边精确阵容从头训练，每阶段 2M，使用逐任务 ε 衰减。按每 100k 步和阶段末取当前任务最近 32 局，用当前模型复算唯一前馈层 A（`agent.net[2]`）的状态加权活动。记录全宽归一化分数、top-3 和严格 >3 名单，不改变训练。

## 动作偏好约束方法

### 主体方法（Default Method）

**TKPR-K1** 是正式默认方法。使用 `vdn_action_set` 的 `topk_preference`，显式设置 `k=1`、`min_gap=0.5`。每个任务训练完成后，用本方法的阶段末策略采集完整的贪心成功轨迹，并保存同一策略在轨迹上产生的历史参考 Q（`data.old_q_values`）。新任务训练时从历史任务采样，用当前 Agent 重放这些轨迹。对每个有效决策，取历史合法 Top-1 作为唯一锚点，只比较它与其他合法动作。排除旧 Q 平局及归一化旧 Q 差低于 0.5 的对，再把归一化的成对 Softplus 与当前任务 TD 损失联合优化。

标准历史池随本方法训练逐阶段生成，旧轨迹和参考 Q 在后续任务中保持其采集版本。共享前缀实验的首任务可以复用同一个 K1 前缀及其胜局，后续仍由 K1 自己采集。VDN-FT 固定历史池是专门为控制实验准备的输入。通用 `topk_preference` 配置缺省仍是 `k=2`，使用 [TS5-1M K1 配置](../configs/continual/action_preference/ts5/tkpr_k1_1m.json)或 [SZC K1 配置](../configs/continual/action_preference/szc/szc_vdn64_tkpr_k1.json)可显式选中正式设定。

论文语境下，ASPR 指动作集合偏好回放的框架。K1 可写成单元素首选集合的特例，等价范围是相同锚点、决策掩码、间隔筛选和归约下的动作对与损失。仓库中的 `aspr_preference` 是按 Q 差阈值划分 Good／Bad 的 **ASPR-2G**，其 Good 可以含多个动作。

### 主体方法变体（Variants）

| 方法 | 实现入口 | 实际差异 | 实现提交（均可从 `main` 到达） |
| --- | --- | --- | --- |
| TKPR-K1/K2/K3/K5/K10 | `topk_preference` + 显式 `k` | 选至少一个端点属于旧 Top-K 的动作对，包含 Top-K 内部的对。每个决策钳到 `min(k, 合法动作数 - 1)`；K 足够大时覆盖 FRPR 的全部有效对。 | [Top-K](https://github.com/wangyahao6/marl_method_lab/commit/ef384cf) · [K10 配置](https://github.com/wangyahao6/marl_method_lab/commit/488ebce) |
| TKPR-R2/R3 | `tkpr_r2` / `tkpr_r3` | 只保留与历史第 2／3 名相连的动作对，方向仍由旧 Q 决定；有效对可以要求锚点低于更高排名动作。 | [单排名锚点](https://github.com/wangyahao6/marl_method_lab/commit/c5f8610) |
| TKPR-K1-W | `topk_preference_weighted` | 复用 K1 选对；用旧 Top-1／Top-2 归一化 Q 差重加权有效决策，每个抽样历史任务内平均权重为 1，全零回退为均匀权重。 | [K1-W](https://github.com/wangyahao6/marl_method_lab/commit/2c761c4) |
| TKPR-K1-HR | `topk_preference_high_regret` | 在 K1 损失之外加入高旧 Q 损失的贪心翻转项。 | [K1-HR](https://github.com/wangyahao6/marl_method_lab/commit/2c761c4) |

SZC 的 [K 值对照](results/parameter-ablations/szc-vdn64-tkpr-k/README.md)同批训练 K1/K2/K3/K5 与 FRPR，[K10](results/comparisons/szc-vdn64/README.md)来自后续批次。R2/R3 的同批两条 K1 重跑在 Z 阶段失败，[排名位置报告](results/parameter-ablations/szc-vdn64-tkpr-rank-position/README.md)使用同 S 前缀的历史 K1 终点作参照，训练提交不同。

### 方法扩展（Extensions）

| 方法 | 实现入口 | 参考记忆如何改变 | 分支与实现提交 |
| --- | --- | --- | --- |
| TKPR-Refresh | `tkpr_refresh` | 复用 K1 损失；阶段末在已见任务采集新胜局以刷新轨迹池。新轨迹携带采集时的 Q，保留轨迹沿用原 Q。每历史任务采 8 条，标准 K1 的参考协议是跨任务合计 8 条。 | `main` · [a10c88d](https://github.com/wangyahao6/marl_method_lab/commit/a10c88d) |
| TKPR-CPR | `tkpr_cpr` | 复用 K1 损失及总 8 条采样，保留轨迹。过任务胜率门控后，用当前策略重算旧轨迹上的 Q；可更新有效参考（包括改变 Top-1），也可暂时关闭不确定决策的约束。 | `main` · [470c22c](https://github.com/wangyahao6/marl_method_lab/commit/470c22c) |
| TKPR-CPR-Anchor | `tkpr_cpr_anchor` | 复用 CPR 门控和回放；当前 Top-1 须与永久原始参考的 Top-1 一致，当前置信度须达到 `max(reference_confidence_threshold, min_gap)`，并严格高于已存置信度加 margin。未接受时保留已有 Q、掩码和来源。 | `feature/tkpr-cpr-anchor` · [0e4d330](https://github.com/wangyahao6/marl_method_lab/commit/0e4d330) · [分支结果](https://github.com/wangyahao6/marl_method_lab/blob/e7f1dd3b6f1051932198f5539d44afe7d868179d/docs/results/comparisons/ts5-tkpr-cpr-anchor-1m/README.md) |

### 其他偏好约束方法（Related Methods）

| 方法 | 实现入口 | 动作集合或约束关系 | 实现提交（均可从 `main` 到达） |
| --- | --- | --- | --- |
| ASPR-2G | `aspr_preference` | 以旧 Q 到最优 Q 的绝对差和 δ 划分 Good／Bad。约束通过归一化 gap 筛选的 Good × Bad 对。历史报告也写作 ASPR。 | [6e0c745](https://github.com/wangyahao6/marl_method_lab/commit/6e0c745) |
| ASPR-3G/4G | `aspr_3g_preference` / `aspr_4g_preference` | 合法旧 Q 按确定性一维 K-means 得到至多 3／4 个有序组，同 Q 不拆分；所有不同组之间的合格对都可进入损失。 | [69c671c](https://github.com/wangyahao6/marl_method_lab/commit/69c671c)；训练记录使用 [3bb47ae](https://github.com/wangyahao6/marl_method_lab/commit/3bb47ae)，该提交也是 `origin/aspr-multigroup` 的已核对引用 |
| ASPR-3G-Adjacent | `aspr_3g_adjacent_preference` | 复用 3G 分组，只用 G1–G2、G2–G3 合格对训练；G1–G3 只用于诊断。 | [3fe39a2](https://github.com/wangyahao6/marl_method_lab/commit/3fe39a2) |
| TC（传递闭包覆盖率） | `pair_tc_low` / `pair_tc_medium` / `pair_tc_high` | 在共同有效状态上，用冻结掩码约束 K1 基础边及固定预算的非 Top-1 边。额外边按闭包增益与距离配额选择，不进行 ASPR 动作集合聚类。 | [fa58883](https://github.com/wangyahao6/marl_method_lab/commit/fa58883) · [间隔扩展与 TC K1-only](https://github.com/wangyahao6/marl_method_lab/commit/8b7aa9d) |
| FRPR | `full_rank_preference` | 约束通过筛选的全部合法非并列动作对；另有 Unfiltered、ReLU、Raw、RawLoss，详见实现协议。 | [5366f1e](https://github.com/wangyahao6/marl_method_lab/commit/5366f1e) |
| T1PR | `top1_preference` | 用旧 Top-1／Top-2 差筛选整个决策，并以置信度加权；K1 对各个动作对单独筛选。另有 Max 与 Unfiltered 版本。 | [5366f1e](https://github.com/wangyahao6/marl_method_lab/commit/5366f1e) |
| ASPR-Max | `aspr_max_preference`，旧别名 `action_set_preference` | Good 与 Bad 各取当前最大 Q，用 ReLU hinge。早期 TS3 报告的 ASPR 指此方法。 | [fce5b0e](https://github.com/wangyahao6/marl_method_lab/commit/fce5b0e) · [TS3 历史协议](TS3_ASPR_PROTOCOL.md) |

### 实验对照与消融（Controls and Ablations）

VDN-FT／VDN-ER 是 TD 回放对照。用同一个 FT 参考模型生成历史池，再用不同动作对规则训练分支，可以控制输入来源。启用 `fixed_reference_history` 的原生 K1 与启用冻结掩码的 `pair_tc_k1` 是两种实验对照。标准 K1 在 CPR／Refresh 的对照表中仍是正式方法运行，Anchor 报告引用同一批运行。一份结果对应一个 `run_id`，对照角色可随比较问题变化。详见 [TS5-1M 运行对照表](results/README.md#ts5-1m-的-k1-运行对照)和[实现协议](VDN_ACTION_SET_PREFERENCE.md)。

以上实现提交用于定位代码。逐运行训练版本以结果清单及启动快照为准，分支发布提交用于定位报告。CPR-Anchor 的训练版本为 `9ff065f`，成果发布于 `e7f1dd3`。原四组仍可从 `main` 上的 [TS5-1M 结果](results/comparisons/ts5-tkpr-cpr-1m/README.md)读取。

`feature/single-agent-tkpr-minatar` 和 `feature/single-agent-tkpr-mountaincar` 分别在 [81bbc9b](https://github.com/wangyahao6/marl_method_lab/commit/81bbc9b) 和 [9eb1b5f](https://github.com/wangyahao6/marl_method_lab/commit/9eb1b5f) 引入独立 DQN 路径。其报告按单智能体实验读取。`feature/generic-prefix` 在 [d1bcea5](https://github.com/wangyahao6/marl_method_lab/commit/d1bcea5) 改造前缀复用入口，与本文的偏好规则分类属于不同维度。

### Shapley 与随机筛选 FRPR

shapley_selected_frpr 和 random_selected_frpr 在原 FRPR 有效动作对上应用离线筛选掩码。真实单点动作分支提供精确 Shapley 枚举所用的联盟价值。协议、覆盖要求和调度入口见 [SZC Shapley-FRPR](results/behavioral-analysis/szc-vdn64-shapley-frpr/README.md)。

## 用户方法

`drivers/continual/protection.py`、`release_schedule.py`、`detection_common.py`为状态加权Mixer W/V软保护；`neuron_records.py`为名单和频次记录。导入自已停止的latest_ramp工程，不导入其旧队列或结果。

参考配置：`configs/continual/task_transfer/knife_fig6.json`。公式omega=c/mean(c)，A=sum(omega*abs(z))/N；阈值严格大于3，最近32训练回合，实际参数更新系数0.1，最新集合替换，退出后100000环境步线性恢复；未定义检测按原实现记录与处理。仅是已导入实现和参考设置，不固化为所有未来实验要求。

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

项目默认采用原版 CHI。用当前网络重评估历史探测集，按层内归一化分数严格大于 3 选出本轮名单，再用它替换保护名单。后续实验以单层 VDN 为主，使用 `--algorithm vdn_1ff`（64 单元）或按实验指定宽度，保护唯一前馈层 A。两层 VDN 对应 A2。

通过 `scripts/run.py --method softprotect_chi` 启用。`--ts3-role full` 或 `branch` 省略分组时，单层 VDN 默认使用 `A_chi`，两层 VDN 使用 `A2_chi`。单层参考协议为 `configs/continual/neuron_protection/chi/ts4_vdn_1ff_chi_comparison.json`。使用 `--ts3-role prefix` 保存无保护 A 前缀和固定探测集，再以 `--ts3-role branch --parent <prefix目录>` 续训。

每个阶段保留最近至多 32 局完整训练轨迹。阶段末用在线 Agent 计算 VDN 单智能体替换动作的最大 Q 差，在有效存活智能体间取均值作为状态协作敏感度 `c`，再按该探测集的有效状态均值计算 `omega = c / mean(c)`。将轨迹及对齐的权重保存为 `chi_probes/phase*.th`，与训练 replay 分离。重评估时逐局从零隐状态开始，按时序重放观测、动作历史和 agent ID，复用固定权重。排除死亡智能体、无效动作、padding 和仅用于 bootstrap 的末状态。

每个任务先计算 `mean_state(omega * mean_live_agent(abs(activation)))`，再除以本任务的层内均值。对本轮各任务的归一化分数逐神经元取最大值。选择沿用配置的全层 top-k 或严格阈值，分数并列时优先较小索引。本轮入选单元的输入权重行、偏置和 GRU 输入列的实际 RMSprop 增量系数为 0.1。退出名单的单元在 100000 环境步内线性恢复。本轮无有效分数时使用空名单。显式选择 `A_chi_union` 或 `A2_chi_union` 可运行保留历史入选名单的变体。探测或重评估异常另记 `chi_errors.jsonl`。每轮逐任务分数、最大值和名单写入 `chi_assessments.jsonl`。

设计借鉴 SPG §3.2 中随网络变化重新评估历史任务重要性的思路。VDN 使用共享 Q 输出和各历史任务自己的固定探测集，以无梯度的响应统计评估。参见 [Parameter-Level Soft-Masking for Continual Learning](https://proceedings.mlr.press/v202/konishi23a.html)。

## TS4 VDN (2026-11-02)

--suite ts4 --algorithm vdn supports baseline and A1/A2 single-layer soft protection; additive-VDN sensitivity replaces QMIX hypernetwork calculation. Other continual suite restrictions remain unchanged. See [protocol](TS4_VDN_PROTOCOL.md).


## TS3 VDN A2 local GEM (2026-11-04)

Authorized existing-baseline seeds1234/4567/3535, only high/middle/low method branches,21A2units each, local actual RMSprop increment projection. Reuse original2M A prefixes, B-E8M each, nine branches72M new nominal steps. Latest-boundary groups, no neuronal history union or 0.1 scaling. Three baseline FWT jobs are evaluation-only. Resource-aware max5new workers; no interference with existing queues. See [protocol](TS3_VDN_A2_LOCAL_GEM_PROTOCOL.md). Queue `scripts/run_ts3_vdn_local_gem.py`, results `results/ts3_vdn_A2_local_gem_20261104/`. Launch/queuing is not completion.

## 方法设计参考

[MEAL 机制参考](MEAL_DESIGN_REFERENCE.md)及[固定版本源码](../references/meal/README.md)用于自己的神经元机制和持续学习设计。EWC/MAS/A-GEM/CBP 在这里是参考材料，不是新增的可运行方法或待执行对照。方法确定并取得初步结果后再考虑对照实验。

`configs/continual/neuron_protection/chi/ts3_vdn_1ff_chi.json` 与 `ts5_vdn_1ff_chi.json` 为种子 3535 的 64 宽度配对实验。每套序列共享无干预 A 前缀，前缀保存固定探测集与原版阈值选择，B–E 分别运行 CHI 与原版历史并集软保护。两组严格使用层内分数 >3，各任务 2M 环境步，评价含初始未见任务、阶段末全部已见任务和下一未见任务。Windows 使用其已安装的 SC2 版本，两组在同一主机运行。

TS3 的 64 宽单前馈基线使用 `--suite ts3 --algorithm vdn_1ff --method baseline --ts3-role full --protocol configs/continual/neuron_protection/chi/ts3_vdn_1ff_chi.json`。沿用该协议的任务表、预算、探索和评估，从 A 训练到 E。每 100k 步及阶段末记录 Agent A 的状态加权活动、top-3 和严格阈值名单。

Windows 启动器为每次运行设置独立的 `runs/<run-id>/tmp`，并将 TEMP/TMP 传给训练与评估子进程，隔离 SC2 的 `StarCraft II/TempLaunchMap.SC2Map`。TS3 共享驱动的评估重试覆盖协议断连、启动连接失败与创建地图的 InvalidMapData。重试仍使用同一环境种子，并恢复该种子块起始 RNG。周期评估重试耗尽时记录错误并跳过该次评估，正式阶段评估仍按原规则报告失败。

2026-11-11 Windows 配对实验的 attempt001 前缀分别在 TS3 的 525002 步和 TS5 的 800013 步因上述评估异常失败，四条方法分支尚未开始。阶段内未保存训练检查点，修复后以新 attempt002 从 A 重新训练，失败产物保留。

TS4 的 `ts4_vdn_1ff_128_chi.json` 使用同一个种子 3535 的 A₁ 前缀比较四组。所有组按当前网络重新评估固定探测集，使用严格 `>3`。`A_chi` 和 `A_chi_input` 替换保护名单并对退出单元线性恢复。`A_chi_union` 和 `A_chi_union_input` 将每轮入选单元加入永久保护并集。带 `_input` 的组缩放前馈层输入权重行和偏置，其他两组同时缩放对应的 GRU 输入列。保护中的实际 RMSprop 增量系数均为 0.1。A₁ 探测集从已有前缀的最后 32 局及其阶段末模型离线生成，与前缀原始产物分别保存。

TS4 等数量筛选对照在 `results/ts4_chi_selection_matched_3535_20261111/plan.json`。从已有 A₁ 前缀按普通软保护复跑一次 B 并保存完整续训状态，三组从同一 B 末状态训练 C。保护名单在 C 内固定，分别使用普通 5 单元、CHI 7 单元及普通排序补足的 7 单元。主要比较 C 末 stalker 和 zealot 相对 B 末的胜率变化，同时记录 C 的胜率。

TS4 和 TS8 的 64 单元配对实验使用 `ts4_vdn_1ff_chi_comparison.json` 和 `ts8_vdn_1ff_chi_comparison.json`。各序列共享一个无保护的 A₁ 前缀，普通软保护和 CHI 都使用阈值严格大于 3、历史入选并集及全参数范围保护。TS8 用精确镜像阵容运行 S、Z、C、2S3Z、2Z3C、3S2C、S，累计阈值为 1M、1.5M、2M、2.5M、3M、3.5M、4.5M。A₂ 按同一任务回访统计。

TS8 单前馈基线通过 `--suite ts8 --method baseline --ts3-role full --protocol configs/continual/neuron_protection/chi/ts8_vdn_1ff_chi_comparison.json` 启动，`--algorithm` 支持 `vdn_1ff` 和 `vdn_1ff_128`。两种宽度均从 A₁ 随机初始化，沿用七阶段任务表及逐任务 ε 衰减，每 100k 步和阶段末记录唯一前馈层的状态加权活动、top-3 与严格大于 3 的名单。

双层 64 单元 VDN 的 TS4 A2 CHI 对照使用 `ts4_vdn64_a2_chi_3535.json`。种子 3535，从 A₁ 开始，累计阶段阈值与现有双层无保护基线相同。每阶段固定最近 32 局，按层内归一化分数严格大于 3 更新历史保护并集。保护 A2 的输入权重行、偏置和通往 GRU 的输入权重列。
