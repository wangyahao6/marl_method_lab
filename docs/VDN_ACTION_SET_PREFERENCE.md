# VDN 可接受动作集合偏好约束

在连续任务中，用每个已完成任务的阶段末 Agent 给成功轨迹标记可接受动作集合。新阶段在当前任务上执行原 VDN TD 更新，同时重新展开旧任务轨迹，约束当前 Agent 对旧任务动作的排序。

## 启动与配置

```sh
cd /home/wyh/marl_method_lab
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method action_set_preference --seed 3535 --run-id vdn_action_set_example_3535 \
  --protocol configs/continual/vdn_action_set_example.json --dry-run
```

示例只用于检查配置。使用同一协议配合 `--method baseline` 可做同入口对照。支持 `vdn`、`vdn_1ff`、`vdn_1ff_128`。配置按已有连续任务协议放在 JSON 中。

| 字段 | 用途 |
| --- | --- |
| `tasks` | 每个任务独立指定己方和敌方阵容，按 Stalker、Zealot、Colossus 计数。 |
| `stages[].task`、`stages[].steps` | 任意长度的任务顺序与阶段内步数。达到步数后跑完当局，因此实际步数可以超出该阶段预算。 |
| `action_set.weight` | λ，默认 1。设为 0 时禁用记忆采集和偏好损失。示例值尚未校准。 |
| `action_set.margin` | 偏好间隔 m，默认 0。m=0 允许两组的最大 Q 打平。 |
| `action_set.batch_episodes` | 每次 learner 更新采样的历史 episode 总数，默认 8。 |
| `action_set.delta_by_task` | 启用时为每个阶段用到的任务显式设置绝对 Q 差阈值 δ。示例的 0 只保留并列最优动作。 |
| `action_set.delta_candidates_by_task` | 同一批胜局上统计候选 δ 的集合大小、单元素占比和覆盖全部合法动作的占比。正式标记使用配置中的任务级 δ。 |
| `action_set.success_episodes` | 每阶段目标胜局数，默认 100。 |
| `action_set.seed_offsets` | 至少两个采集种子偏移，实际种子为训练种子加偏移后对 2³¹ 取模。 |
| `action_set.attempts_per_seed` | 每个采集种子的最大局数，默认 125。按种子块递增分配累计目标，最后一块可补前面块的不足。 |
| `evaluation.seeds`、`episodes_per_seed`、`interval` | 独立贪心评价使用固定种子和每种子局数。阶段内定期评价当前任务，interval=0 关闭定期评价。 |

每阶段重新分配当前任务的 TD replay 并重置 ε 衰减计数。在线网络、目标网络、优化器和全局步数跨阶段延续。阶段末评价所有已见任务，并在新任务首次训练前保存对应评价；结合初始模型评价计算 ACC、BWT、FWT。最终检查点可用 `scripts/evaluate.py --run <run-dir> --episodes <n>` 独立评价所有已见任务，`n` 是每个任务的总局数。

当前使用同一个 SMACv2 `10gen_protoss` 5v5 布局。仓库环境按 unit type 和位置分配本局 ally slots，Agent ID 的 one-hot 对应本局 slot。动作 0 为死亡单位 no-op，1 为停止，2–5 为移动，6–10 按本局敌方 slot 攻击。slot 表示局内索引，阵容变化会改变该 slot 对应的单位。旧轨迹保留自己的观测和合法动作掩码。真实 5v5 测得 state 130、obs 92、Agent 数 5、动作数 11。改变队伍规模或动作布局需要先定义新的输入及动作映射。

## 阶段末记忆

1. 保存 `models/phaseNNNN_<step>/agent.th`，用该状态创建独立的冻结 CPU Agent。贪心 runner 的随机状态与训练隔离。
2. 在该阶段的任务环境中按多个种子采集完整 episode。胜局进入 `action_set_memory/phaseNNNN/success.th`，失败局进入同目录的 `failure.th`。达到种子尝试上限后继续阶段流程，`summary.json` 记录目标、实际胜局和缺口。
3. 对每条胜局从零隐藏状态展开参考 Agent，用轨迹原动作生成上一动作输入。在每个有效决策中，保存所有满足 `legal[a] and max_legal_Q - Q[a] <= delta` 的动作。
4. 池按阶段编号持久化，包含任务 ID、δ、输入/动作布局、参考 checkpoint 路径和 episode 列表。每条记录保存种子、局号、真实决策长度、胜负和截到最后观测的完整 `EpisodeBatch` 张量。成功记录附加布尔 `good_mask`。旧 Agent 在标记后退出，后续更新只读取轨迹和掩码。

回访会保存一个新阶段快照。当前任务在自己的回访阶段也只受其他任务约束。其他任务若有多次完成记录，采样最近一次非空池，所有较早快照继续留存。零胜局的池保留诊断信息，采样器跳过它。当前入口以一次进程执行整条序列。

## 联合更新与张量

历史任务先随机排列，再在本次 episode 总预算内均衡分配。有 2 个历史任务时各采 4 条，超过 8 个时等概率抽 8 个任务各采一条。每个池内优先无放回采样，池小于分配数时用有放回采样。

每个历史任务独立组 batch，补齐该 batch 内的时间维。当前 GRU 从零隐藏状态依时间顺序展开，使用旧轨迹的上一动作与 Agent ID。以 `E` 为该任务的采样局数，`T` 为含最终观测的时间长度，`N` 为 Agent 数，`A` 为动作数。

| 张量 | 维度 |
| --- | --- |
| obs、上一动作 one-hot | `[E,T,N,obs_dim]`、`[E,T,N,A]` |
| avail_actions、good_mask | `[E,T,N,A]`，单条序列落盘为 `[T,N,A]` |
| filled、terminated | `[E,T,1]` |
| GRU 隐状态 | 初始化 `[E,N,H]`，GRUCell 内部使用 `[E*N,H]` |
| 重算 Q、参与约束的状态掩码 | `[E,T-1,N,A]`、`[E,T-1,N]` |

真实最后一次决策参与约束，包括超时前最后一步。最终 bootstrap 观测、padding、终止后的帧和仅有 no-op 的死亡 Agent 均排除。无合法坏动作的状态只计入集合诊断。参考标记在有效状态遇到空集合时报告数据构建错误。

每个有效状态计算 `relu(m + max_bad(Q) - max_good(Q))`。先对一个任务内的有效状态求平均，再对本次采样中至少有一个有效状态的任务等权平均；没有有效历史状态时偏好项为零。联合目标为 `TD_loss + weight * preference_loss`，一次反传。历史数据只进入偏好项；当前任务的 action selector、VDN Mixer、Double-Q target 和 TD 公式沿用原实现。

`action_set_updates.jsonl` 每次更新记录 TD、偏好、总损失，以及采样任务、源阶段、池大小、episode 编号与数量。任务级记录包括有效状态数、good/bad 的最大 Q 均值、违约比例（hinge 内部大于零）、集合大小和两种占比。没有定义的比率或最大值记 null。另记录历史分支准备耗时、CUDA 历史前向时间和完整 learner 更新耗时；联合反传的边际耗时可用同协议 baseline 对照估计。常规 learner 和 runner 指标沿用 `metrics.jsonl`。

## 代码位置与验证

- `src/learners/action_set_preference.py` 管理历史池采样、GRU 展开、合法/有效掩码、损失和逐更新日志。 `src/learners/q_learner.py` 接收可选偏好项并接入原有单次反传。
- `drivers/continual/action_set_train.py` 驱动任务切换、独立评估、轨迹采集与持久化。 `action_set_protocol.py` 负责协议默认值和结构校验。
- `scripts/run.py` 和 `scripts/evaluate.py` 注册训练与最终评价。 `configs/continual/vdn_action_set_example.json` 是三任务示例。
- `.venv/bin/python tests/test_action_set_preference.py -v` 使用脚本环境和真实 VDN/GRU 验证 T1→T2→T3、联合损失、梯度、采样、掩码和关闭方法的对照。另在实际 SMACv2 的三个示例任务各完成一条冻结 Agent 贪心轨迹并检查掩码；正式持续学习训练及保留效果尚未评估。
