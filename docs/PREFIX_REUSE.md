# 通用第一阶段 Prefix

Prefix 是独立训练产物。`vdn_action_set` 沿用 `shared-prefix` / `shared-branch` 角色和 `--parent`；新产物另有 `prefix_manifest.json`，无需复制训练驱动。现有协议、结果和运行目录不作迁移。

## 生成与分支

以下命令是用法示例，不代表已经授权或启动实验。每次使用新的 run-id。

```sh
python scripts/run.py --suite vdn_action_set --algorithm vdn --method baseline \
  --seed 3535 --protocol configs/continual/action_preference/szc/szc_vdn64_tkpr_k1_shared.json \
  --preference-role shared-prefix --run-id prefix_S_3535_new

python scripts/run.py --suite vdn_action_set --algorithm vdn --method topk_preference \
  --seed 3535 --protocol <分支自己的完整协议.json> \
  --preference-role shared-branch --parent runs/prefix_S_3535_new --run-id branch_tkpr_new
```

前缀只执行 `stages[0]`，也允许专用的单阶段协议。历史命令中指定的排名方法在 `shared-prefix` 内部统一转成 baseline，不构造偏好消费者，不启用偏好损失、标签和诊断干预。训练默认参数不变。

分支可改用 `--method full_rank_preference`、`aspr_preference` 或其他已支持方法，也可指定不同后续任务及预算。第一阶段、算法、种子等训练输入须匹配。保留 `--parent <绝对或相对路径>`；也可用 Manifest 中的 `--prefix-id prefix-...`，在当前仓库 `runs/` 下查找唯一产物。外部 Prefix 使用 parent 路径。两个选项互斥。

加 `--dry-run` 只检查配置和已有 Manifest、显示计划，不启动环境或新建运行目录。真实训练仍必须有独立授权。

## Prefix Spec

Spec 来自与训练共用的默认配置合并函数及实际生效配置，不比较完整实验协议。它包括：

- 环境参数、SC2 game version 和环境实现源码哈希；第一阶段任务名、双方阵容、预算和训练种子。
- 算法、网络、输入编码、Agent/MAC/Learner/Mixer 设置。
- 学习率、optimizer、TD 折扣、double-Q、梯度裁剪、Target 更新、batch 和 ReplayBuffer 设置。
- ε 起止值和退火策略、训练后端及 PyTorch/NumPy 版本。

整个第一阶段对象参与签名；仅使用该任务的阵容。后续 stages、未参与第一阶段的任务、方法及其损失设置、全序列 t_max、输出路径、评价设置和梯度诊断不参与签名。`epsilon_on_switch` 在 t_env=0 无差异，分支按自己的切换规则运行。

共同前缀第一阶段统一使用普通基线 Episode ReplayBuffer。`replay_on_switch` 表示后续切换行为，不进入第一阶段 Spec；不同分支可选择 FT 或 ER。FT 在切换后使用空池；ER 将同一前缀池转换为已有 TaskReplayBuffer 并保留历史样本。未使用共享前缀的原 ER 训练流程保持原样。

不兼容时列出字段路径及父/分支值，例如 `seed`、`stage.steps`、`roster.ally`、`training.lr`。不通过忽略全部 stages 或全部协议字段放宽校验。启动器在创建目录前核验，驱动按实际生效配置再次核验。

## 产物与恢复

`prefix_manifest.json` 包含 UUID Prefix ID、state version、完整 Spec 及其 SHA-256、实际 launch 的 Git 提交/dirty 状态/源码哈希，以及下列文件哈希：

- 检查点中的 Online Agent、VDN Mixer 和完整 Optimizer 文件。
- `return_state.th`：Target Agent、Learner/Runner 日志与训练计数、RNN online/target hidden、runner episode step、ε/探索起点、Python/NumPy/Torch CPU/CUDA RNG、环境分布及其 RNG。
- `prefix_replay.th`：完整 TD 池；`initial_agent.th`：训练前 Online Agent，供分支自己的 FWT 评价使用。
- 原始历史 success/failure 轨迹及旧 Q；配置、环境信息和第一阶段 retention 报告。

训练边界状态在第一阶段训练结束、评价及历史采集之前保存。历史采集复用独立 CPU greedy runner 和随机数隔离；前缀结束时再次恢复边界 RNG。通用历史不含 Good-set、TKPR 选择或 CPR 更新标签；采集的 seed offsets、目标胜局数与尝试上限单独记录，不改变训练 Spec。

分支先检查 Spec 及文件哈希，再沿用原有模型/Optimizer/Target 恢复链路，恢复计数、hidden、ε、环境分布和 RNG。环境恢复保留分支自己的阵容表。FT 在下一阶段新建空 TD 池；ER 恢复池及 task ownership，使用分支自己的任务顺序应用现有 quota/sampling 规则。训练在完整 episode 边界分支；下一 episode 原有的 hidden reset 保持不变。

排名方法直接消费同一原始胜局和旧 Q；ASPR 在分支内按自己的阈值/分位数派生 Good-set。各分支重新使用自己的评价协议对初始模型和前缀模型评价需要的后续任务，不沿用父协议中不对应的 FWT 引用。

这保存的是现有驱动可恢复的 episode 边界训练状态。SC2 使用新进程，恢复环境分布 RNG，不包含游戏进程的逐帧内存快照，不声称跨进程 SC2 完全确定性。

## 方法范围与旧产物

已支持 baseline FT/ER、TKPR（含 K1、weighted/high-regret）、FRPR 变体、Top1、rank-position、ASPR/ASPR-max 和 ASPR 分组方法。Refresh 在分支内从原始历史补充每 episode 的 reference phase/checkpoint；CPR 在分支内通过已有 initializer 派生 original Q、reference valid/source metadata；它们各自后续更新/刷新与损失公式保持原样。新 CPU 集成测试覆盖这两条分支，以及 ASPR-3G 的标签派生。Selected-FRPR 仍要求其自己的 Shapley/trace 覆盖数据。Pair-ablation 保留 `pair-reference` 专用流程，不将多阶段 reference 误认成第一阶段 Prefix。

原有 `full_shared_prefix: true` 继续使用完整恢复、独立分支记忆、副产物清单和 final replay 保存；新 `shared-prefix` 即使不带该开关也会生成完整通用前缀。`prefix_state_manifest.json` / `final_state_manifest.json` 原有用途保留，新统一身份及校验使用 `prefix_manifest.json`。

旧 `--parent` 产物仍按原有严格协议匹配恢复；没有新 Manifest 的产物不声称已通过新哈希校验，也不能凭空恢复缺失的 TD 池或 hidden。旧数据缺少旧 Q 时仍明确拒绝需要它的方法，不修改历史产物补写元数据。旧 full_shared_prefix 的 FT Prefix → ER 分支路径保留。新 Prefix 的训练边界状态与旧版在采集/方法处理之后保存的状态有区别，版本和文件哈希分别记录，不将旧实验结果改归为新实现。

## CPU 验证

`tests/test_generic_prefix.py` 复用现有 ScriptedProtoss 小型 episode fixture 和真实 VDN/GRU/Optimizer；覆盖跨任务顺序/预算、多方法、字段差异、Manifest 哈希、采集扰动隔离、状态恢复与 FT/ER 池行为。`test_full_preference_prefix.py`、`test_shared_tkpr_k1.py`、`test_tkpr_rank_position.py`、`test_tkpr_cpr.py` 继续验证原共享分支与损失。没有启动 SMAC/SC2，也没有正式训练。

本次验证：67 项相关 CPU 测试通过，`py_compile` 与 `git diff --check` 通过；baseline `shared-prefix --dry-run` 未创建运行目录。测试使用临时依赖目录，未改动仓库运行时。可复查的测试模块：

```sh
python -m unittest discover -s tests -p test_generic_prefix.py
python -m unittest discover -s tests -p test_full_preference_prefix.py
python -m unittest discover -s tests -p test_shared_tkpr_k1.py
```
