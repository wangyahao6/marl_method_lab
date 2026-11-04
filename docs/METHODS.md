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

## TS3 Agent schedules (2026-11-02)

`ts3_protection.py` adds single-layer A1/A2 actual RMSprop scaling with either existing ReleaseSchedule linear recovery or permanent history union. Shared protection parameters and overall state-weighted screening, no per-type selector in this experiment. See [TS3 protocol](TS3_AGENT_PROTOCOL.md). This does not change old heterogeneous ABA semantics.

## TS4 VDN (2026-11-02)

--suite ts4 --algorithm vdn supports baseline and A1/A2 single-layer soft protection; additive-VDN sensitivity replaces QMIX hypernetwork calculation. Other continual suite restrictions remain unchanged. See [protocol](TS4_VDN_PROTOCOL.md).


## TS3 VDN A2 local GEM (2026-11-04)

Authorized existing-baseline seeds1234/4567/3535, only high/middle/low method branches,21A2units each, local actual RMSprop increment projection. Reuse original2M A prefixes, B-E8M each, nine branches72M new nominal steps. Latest-boundary groups, no neuronal history union or 0.1 scaling. Three baseline FWT jobs are evaluation-only. Resource-aware max3new workers; no interference with existing queues. See [protocol](TS3_VDN_A2_LOCAL_GEM_PROTOCOL.md). Queue `scripts/run_ts3_vdn_local_gem.py`, results `results/ts3_vdn_A2_local_gem_20261104/`. Launch/queuing is not completion.
