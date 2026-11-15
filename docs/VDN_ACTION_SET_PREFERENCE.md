# 可接受动作集合偏好回放（ASPR）

Action-Set Preference Replay（ASPR）用每个已完成任务的阶段末 Agent 给成功轨迹标记可接受动作集合。新阶段在当前任务上执行原 VDN TD 更新，同时重新展开旧任务轨迹，约束当前 Agent 对旧任务动作的排序。

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
| `action_set.delta_by_task` | 为各任务显式设置绝对 Q 差阈值 δ。设置 0 只保留并列最优动作。 |
| `action_set.second_gap_quantile` | 缺少显式 δ 时，从本阶段成功记忆中取有至少两个合法动作的有效决策的最优－次优 Q 差分位数。阶段末计算一次并冻结。无可用差值时记录并使用 0。 |
| `action_set.probe_episodes_per_seed` | 从首次标记该阶段任务的阶段末评估中每个种子块保存前 N 局作为独立固定轨迹，默认 0。 |
| `action_set.delta_candidates_by_task` | 同一批胜局上统计候选 δ 的集合大小、单元素占比和覆盖全部合法动作的占比。正式标记使用配置中的任务级 δ。 |
| `action_set.success_episodes` | 每阶段目标胜局数，默认 100。 |
| `action_set.seed_offsets` | 至少两个采集种子偏移，实际种子为训练种子加偏移后对 2³¹ 取模。 |
| `action_set.attempts_per_seed` | 每个采集种子的最大局数，默认 125。按种子块递增分配累计目标，最后一块可补前面块的不足。 |
| `evaluation.seeds`、`episodes_per_seed`、`interval` | 独立贪心评价使用固定种子和每种子局数。阶段内定期评价当前任务，interval=0 关闭定期评价。 |

每阶段重新分配当前任务的 TD replay。ε 可逐任务重置（默认）或累计退火。阶段阈值可按本阶段步数或全序列累计步数计算。在线网络、目标网络、优化器和全局步数跨阶段延续。阶段末评价所有已见任务，并在新任务首次训练前保存对应评价；结合初始模型评价计算 ACC、BWT、FWT。最终检查点可用 `scripts/evaluate.py --run <run-dir> --episodes <n>` 独立评价所有已见任务，`n` 是每个任务的总局数。

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

`action_set_updates.jsonl` 每次更新记录 TD、偏好、总损失，以及采样任务、源阶段、池大小、episode 编号与数量。任务级记录包括有效状态数、good/bad 的最大 Q 均值、违约比例（hinge 内部大于零）、合法贪心动作落在集合外的比例、集合大小和两种占比。没有定义的比率或最大值记 null。另记录历史分支准备耗时、CUDA 历史前向时间和完整 learner 更新耗时；联合反传的边际耗时可用同协议 baseline 对照估计。常规 learner 和 runner 指标沿用 `metrics.jsonl`。

## 代码位置与验证

- `src/learners/action_set_preference.py` 管理历史池采样、GRU 展开、合法/有效掩码、损失和逐更新日志。 `src/learners/q_learner.py` 接收可选偏好项并接入原有单次反传。
- `drivers/continual/action_set_train.py` 驱动任务切换、独立评估、轨迹采集与持久化。 `action_set_protocol.py` 负责协议默认值和结构校验。
- `scripts/run.py` 和 `scripts/evaluate.py` 注册训练与最终评价。 `configs/continual/vdn_action_set_example.json` 是三任务示例。
- `.venv/bin/python tests/test_action_set_preference.py -v` 使用脚本环境和真实 VDN/GRU 验证 T1→T2→T3、联合损失、梯度、采样、掩码和关闭方法的对照。另在实际 SMACv2 的三个示例任务各完成一条冻结 Agent 贪心轨迹并检查掩码；正式持续学习训练及保留效果尚未评估。

## 完整动作排序与 Top-1 偏好保持

全排序偏好回放（FRPR）提供 Softplus（`full_rank_preference`）和 ReLU（`full_rank_preference_relu`）两个版本。它们与首选偏好回放（T1PR，`top1_preference`、`top1_max_preference` 和 `top1_unfiltered_preference`）及 ASPR 共用 `vdn_action_set` 驱动。排序方法沿用阶段末贪心胜局采集、历史池目录和任务均衡采样。采集时由阶段末冻结 Agent 从零 GRU 状态逐步前向，胜局在 `success.th` 的每条轨迹 `data.old_q_values` 中保存完整 FP32 动作 Q，包含末尾 bootstrap 观测对应的 Q。该字段与 `avail_actions` 同形状，采样时一起补齐时间轴。回放时直接读取旧 Q，当前 Agent 从零 GRU 状态展开历史观测，沿用轨迹中的上一动作和 Agent ID。排序损失排除非法动作、只有 no-op 的死亡 Agent、bootstrap 观测、终止后的帧和 padding。只有至少两个合法动作的智能体决策进入排序统计。最后一个真实决策参与约束，包括超时前最后一步。

对每个智能体决策，令 `s = max(std(Q_old[合法动作]), q_scale_floor)`，标准差使用总体口径。旧模型的筛选间隔和当前模型的排序间隔均除以同一个冻结尺度 `s`。尺度下限保留了近似平局的识别能力。固定旧尺度使两个合法动作的情况也能通过当前 Q 间隔获得梯度。

- **FRPR Softplus（Exp-1）** 对每个无序合法动作对只计算一次。旧模型的归一化绝对差距达到 `min_gap` 且严格大于零时，按旧差距的符号对当前差距定向，计算 `softplus(-sign(old_gap) * current_gap / temperature)`。明确的非首选动作对也参与约束。
- **FRPR ReLU** 复用同一归一化、动作对和 `min_gap` 筛选，计算 `relu(-sign(old_gap) * current_gap)`。只惩罚排序反转，方向一致或当前 Q 打平时该动作对损失为零。
- **T1PR 第一版（Exp-2，`top1_preference`）** 用旧模型第一与第二名的归一化差距作为置信度 `c`。当 `c >= min_gap` 且 `c > 0`，用旧首选动作与其余每个合法动作计算 `softplus(-(Q_new[首选] - Q_new[其他]) / s / temperature)`。先对该决策的动作对求均值，再乘以 `c`。这使更高的置信度直接加强该决策的约束。

- **T1PR 第二版（`top1_max_preference`）** 使用相同的旧模型置信度门槛。每个有效决策只比较旧首选动作与当前 Q 最高的其他合法动作，损失为 `relu((max_competitor_q - preferred_q) / s + margin)^2`。置信度仅用于筛选。排序梯度只通过旧首选动作与选中的竞争动作传播。竞争动作的当前 Q 并列时取动作索引最小者。

- **T1PR 无阈值版（`top1_unfiltered_preference`）** 使用与第一版相同的成对 Softplus 和置信度权重 `c`。所有至少有两个合法动作且 `c > 0` 的有效决策都参与约束，包括低于第一版 `min_gap` 的决策。旧首选动作并列的决策仍被排除。每个历史任务另记 `mean_constrained_confidence`，即参与约束的决策的平均置信度，零有效状态时记 null。该指标也写入 `experiment_metrics.jsonl`。

成对损失先对每个决策的动作对取均值。各方法再对历史任务内有约束的决策取均值，最后对本次采样中有约束的任务等权平均。历史项使用 `TD_loss + weight * preference_loss` 接入已有的一次反传。历史池的 `reference_checkpoint` 记录 Q 的来源。早期胜局池须离线补齐 `old_q_values` 才能用于新代码的继续约束分支。ASPR 继续使用集合标记、δ 校准与固定集合 probe。每次独立运行的历史池来自该次运行自己的阶段末模型，因此同一采集协议下的实际轨迹可能随训练分化。

配置写入协议 JSON 的同名顶层对象 `full_rank_preference`、`full_rank_preference_relu`、`top1_preference`、`top1_max_preference` 或 `top1_unfiltered_preference`，由 `--method` 独立选择。可以按方法分别覆盖以下默认值。

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `weight` | 1.0 | 偏好损失系数 λ，设为 0 时关闭该方法的记忆采集和偏好更新 |
| `temperature` | 1.0 | Softplus 排序温度，取正值，用于 `full_rank_preference`、`top1_preference` 和 `top1_unfiltered_preference` |
| `margin` | 0.0 | 归一化 Q 间隔，取非负值，仅用于 `top1_max_preference` |
| `min_gap` | 0.5 | 旧模型归一化 Q 差阈值，取非负值，`top1_unfiltered_preference` 配置省略此字段 |
| `q_scale_floor` | 0.1 | 旧模型合法 Q 标准差的下限，单位与原 Q 值一致，取正值 |

例如，可在协议 JSON 中添加以下顶层对象。ReLU 配置仅接受 `weight`、`min_gap` 和 `q_scale_floor`。

```json
{
  "full_rank_preference_relu": {
    "weight": 1.0,
    "min_gap": 0.5,
    "q_scale_floor": 0.1
  }
}
```

采集和采样继续读取 `action_set.success_episodes`、`seed_offsets`、`attempts_per_seed`、`batch_episodes`。默认值用于实现验证，尚未按保留效果校准。

```sh
cd /home/wyh/marl_method_lab
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method full_rank_preference --seed 3535 --run-id vdn_full_rank_example_3535 \
  --protocol configs/continual/vdn_action_set_example.json --dry-run
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method full_rank_preference_relu --seed 3535 --run-id vdn_full_rank_relu_example_3535 \
  --protocol configs/continual/vdn_action_set_example.json --dry-run
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method top1_preference --seed 3535 --run-id vdn_top1_example_3535 \
  --protocol configs/continual/vdn_action_set_example.json --dry-run
```

`<method>_updates.jsonl` 逐更新记录 TD、偏好、总损失及采样信息。每个采样任务记录有效状态数、约束覆盖率和 Top-1 翻转率。成对损失另记有效动作对数。覆盖率的分母是至少两个合法动作的有效决策数。翻转率的分母是这些决策中旧模型有唯一贪心动作的数量，不受训练的 `min_gap` 限制。新模型按合法 Q 贪心选择，当前 Q 并列时沿用 `argmax` 的动作索引顺序。分母为零的比率记 null。

`top1_max_preference_updates.jsonl` 以 `valid_states` 记录有效比较次数，每个有效状态对应一次比较。每个历史任务的 `constrained_top1_flip_rate` 和 `margin_violation_rate` 均以该任务的有效状态数为分母。后者只计入加入 `margin` 后 hinge 输入严格大于零的状态。统一指标文件也记录联合更新的 `preference_loss` 及这些历史任务指标。

`.venv/bin/python -m unittest discover -s tests -v` 验证排序损失和三阶段 CPU 集成。测试使用真实 VDN/GRU 和脚本化 SMAC 风格环境，并检查采集与保存的 Q 一致性、变长轨迹对齐、采样一致性和联合损失。

真实环境检查使用既有 TS3 双层 VDN seed 3535 的 B、C 阶段末检查点。在 SMACv2 5v5 收集一条 B 胜局（73 步）和一条 C 局（49 步），四种方法各执行一次 CPU learner 更新。两种排序方法的 TD loss 同为 0.154165。Exp-1 有 281 个有效决策和 2070 个动作对，Exp-2 有 108 个有效决策和 522 个动作对。四种方法均产生有限梯度并更新当前 GRU。未执行正式训练或保留效果评估。

## TS4 回访配对实验

种子 3535，双层 64 宽共享 VDN。双方 5v5 纯追猎者 → 纯狂热者 → 纯巨像 → 纯追猎者，按累计 1M、1.5M、2M、3M 阈值整局切换。两种方法各训练一次前三阶段；首阶段无历史约束，从 B 开始使用各自默认参数。C 末从同一状态分出两条 A₂ 路径。继续组仍回放 B/C 历史，取消组从回访起只使用 VDN-TD。首次 A 的胜局仍保留在前缀，但 A₂ 时同任务池不进入偏好采样。四条 A₂ 分支与两个前缀合计名义训练量 8M。

使用 `configs/continual/ts4_vdn64_ranking_return_3535.json`。训练入口是 `scripts/run.py --suite vdn_action_set --algorithm vdn`，配合对应 `--method` 和 `--preference-role prefix` 保存前三阶段。两条回访使用 `--preference-role keep` 或 `--preference-role release`，加 `--parent` 指向对应前缀的绝对运行目录。前缀在 C 末保存在线网络、目标网络、优化器、更新计数、累计探索状态、训练及历史采样随机状态、环境分布状态和各历史池。分支重新启动 SC2 进程，恢复可保存的环境分布状态。TD replay 沿用偏好驱动的逐阶段新建协议，A₂ 两组都使用空的当前任务 TD buffer。端点评估每任务 128 局，定期当前任务评估 32 局，每 25k 步一次。前缀的 C 末三任务结果作为回访更新前的共同参考；分支的 `retention.json` 保留该前缀结果。

调度计划存于 `results/ts4_vdn64_ranking_return_3535_20261112/plan.json`，回访任务依赖对应方法前缀成功退出。`branch_restore.json` 保存恢复来源及模型指纹，正式结果按三任务阶段矩阵、A₂ 恢复及 B/C 保留比较。

## SZC 两种子 T1PR 补充

双层 64 宽 VDN，训练种子 3535、2222。使用任务历史实验的镜像 5v5 追猎者 → 狂热者 → 巨像路径，累计阈值 2M、4M、6M，整局切换。两条完整路径合计名义 12M 步。从狂热者阶段开始用 T1PR，超参数、阶段末胜局采集和任务均衡采样沿用 TS4 的 T1PR。当前任务每 25k 步评价 32 局，阶段末每个已见任务评价 128 局，使用 SZC 原实验的任务级种子块。按阶段矩阵报告 ACC、BWT、FWT。

原 SZC 运行跨阶段保留 TD replay；本补充使用 T1PR 驱动的逐阶段重建规则。原同种子 SZC 结果可作参照，组间差值包含 TD replay 协议差异。原 SZC 节点分进程训练，本补充各种子在同一进程内完成整条路径。

配置为 `configs/continual/szc_vdn64_t1pr.json`。调度计划为 `results/szc_vdn64_t1pr_20261113/plan.json`。

## SZC T1PR 第二版协议

`configs/continual/szc_vdn64_t1pr_max.json` 沿用第一版 SZC 协议的任务顺序、预算、累计探索和评估设置。使用同一训练种子可分别预览两种方法。

```sh
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method top1_max_preference --seed 3535 --run-id szc_t1pr_max_3535 \
  --protocol configs/continual/szc_vdn64_t1pr_max.json --dry-run
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method top1_preference --seed 3535 --run-id szc_t1pr_3535 \
  --protocol configs/continual/szc_vdn64_t1pr.json --dry-run
```

## SZC T1PR 无阈值协议

`configs/continual/szc_vdn64_t1pr_unfiltered.json` 使用第一版 SZC 协议的任务顺序、步数、累计探索和评估设置。方法配置接受 `weight`、`temperature` 和 `q_scale_floor`，默认分别为 1.0、1.0 和 0.1。

```sh
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method top1_unfiltered_preference --seed 3535 --run-id szc_t1pr_unfiltered_3535 \
  --protocol configs/continual/szc_vdn64_t1pr_unfiltered.json --dry-run
```

## Top-K 偏好回放（TKPR）

`topk_preference` 按旧模型 Q 值对每个有效智能体决策的合法动作降序排列。同值时动作索引较小者在前。设合法动作数为 N，实际取 `K_eff = min(K, N - 1)`，少于两个合法动作的决策跳过。从 FRPR 的无序合法动作对中保留至少一个端点属于旧模型 Top-K 的对。K=2 时，旧模型排名第一与第二之间的关系以及这两个动作各自与低位动作的关系都受约束。

对保留的动作对，用合法旧 Q 的总体标准差与 `q_scale_floor` 的较大值作为冻结尺度 `s`。筛掉归一化旧 Q 间隔为零或绝对值小于 `min_gap` 的对，然后计算 `softplus(-sign(old_gap) * (Q_new[i] - Q_new[j]) / s / temperature)`。对决策内有效动作对求均值，再对历史任务内有有效对的决策求均值，最后对有约束的历史任务等权平均，与 TD loss 一次联合反传。

TKPR K=1 仍按动作对筛选和均值聚合。原版 T1PR 按旧 Top-1 与 Top-2 的间隔筛选整个决策，并给损失乘以置信度。在 K 覆盖全部合法动作中除最后一名外的排名时，每个有效动作对都被覆盖，TKPR 的损失和梯度等于 FRPR Softplus。

`topk_preference` 配置接受 `weight`、`temperature`、`min_gap`、`q_scale_floor` 和 `k`，默认依次为 1.0、1.0、0.5、0.1 和 2。K 为正整数，运行时上限为环境的 `n_actions - 1`。SZC 协议沿用 `szc_vdn64_t1pr.json` 的任务、预算和评估，每阶段用 `epsilon_on_switch: per_task` 从初始 ε 重新退火。

`topk_preference_updates.jsonl` 和统一指标文件记录有效状态数、覆盖率、有效动作对数、Top-1 翻转率及以下诊断。

- `pair_violation_rate` 以有效动作对为分母，统计当前 Q 反转或打平旧严格排序的比例。
- `mean_effective_k` 和 `k_clip_rate` 以至少两个合法动作的有效决策为分母，包含后来被 `min_gap` 筛掉全部动作对的决策。
- Top-1 翻转率沿用 FRPR 的口径，分母为旧模型有唯一首选动作的有效决策。上述比例在分母为零时在方法日志中记 null。

```sh
.venv/bin/python scripts/run.py --suite vdn_action_set --algorithm vdn \
  --method topk_preference --seed 3535 --run-id szc_tkpr_3535 \
  --protocol configs/continual/szc_vdn64_tkpr.json --dry-run
```
