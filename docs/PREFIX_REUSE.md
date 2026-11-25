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

兼容策略更新：Generic shared-branch 现在要求 Manifest schema 2。缺少 Manifest、schema 1 或未知 schema 均明确拒绝，不自动推断旧代码依赖，也不改写旧产物。旧 `--parent` 命令接口保留，需指向经过新版本生成的通用 Prefix。`keep/release` 和专用 pair-reference 恢复路径不因此改成 Generic Prefix。旧 full_shared_prefix 的 FT Prefix → ER 行为在新版本生成的 Prefix 上保持，原实验结果不改归为新实现。

## CPU 验证

`tests/test_generic_prefix.py` 复用现有 ScriptedProtoss 小型 episode fixture 和真实 VDN/GRU/Optimizer；覆盖跨任务顺序/预算、多方法、字段差异、Manifest 哈希、采集扰动隔离、状态恢复与 FT/ER 池行为。`test_full_preference_prefix.py`、`test_shared_tkpr_k1.py`、`test_tkpr_rank_position.py`、`test_tkpr_cpr.py` 继续验证原共享分支与损失。没有启动 SMAC/SC2，也没有正式训练。

本次验证：67 项相关 CPU 测试通过，`py_compile` 与 `git diff --check` 通过；baseline `shared-prefix --dry-run` 未创建运行目录。测试使用临时依赖目录，未改动仓库运行时。可复查的测试模块：

```sh
python -m unittest discover -s tests -p test_generic_prefix.py
python -m unittest discover -s tests -p test_full_preference_prefix.py
python -m unittest discover -s tests -p test_shared_tkpr_k1.py
```


## 可靠性修复（2026-11-25）

Manifest schema 更新为 2；状态保存格式仍为 `state_version: 1`，两者分开。Spec 现在记录 Python 版本/实现、PyTorch/NumPy、PySC2/SC2 protocol/Protobuf/absl 包版本、CUDA 构建版本、cuDNN 运行库版本、GPU 名称/计算能力、cuDNN deterministic/benchmark、TF32 和 CUBLAS_WORKSPACE_CONFIG。CPU Prefix 的 CUDA 设备段为空，同时记录默认 dtype、float32 matmul precision 和训练线程数。启动器按现有驱动固定 1 线程的执行约定预检，驱动核验实际线程数。没有探测 NVIDIA 驱动版本，也不声称系统/SC2 的跨进程确定性。

`code_version.source_sha256` 仍是启动时完整源码的溯源记录，不能与分支整个仓库逐文件比较：那会把新增方法、文档或结果变成不兼容。真正执行兼容校验的是 Spec 内独立的 `implementation`，含组件 SHA-256 与汇总哈希。组件边界在 `common/prefix_implementation.py` 明列；配置值继续由 resolved Prefix Spec 校验，配置文件的注释和其他算法 YAML 不进入训练代码指纹。

混合驱动按已核查的 shared-prefix baseline 路径作 AST 提取：保留初始化、工厂绑定、阶段预算、TD while 循环，结束于第一阶段训练边界；静态排除 parent 恢复、偏好消费者定义/分支和训练结束后的 CPR/Refresh 历史处理。Learner 仅提取 baseline 实际使用的构造、TD train、target 更新及设备/observer 方法；具体偏好损失文件不纳入。实际 Agent/MAC、ε、Runner、Episode Replay、预处理、环境和 registry 的选中绑定参与指纹。源码注释/格式不影响 AST 指纹；共享定义内部尚有保守范围，修改这些核心定义须检查边界。无法识别阶段或 TD 循环结构时明确拒绝，不自动猜测依赖。新增环境、网络、动态 import 或运行时 monkeypatch 需要重新人工审核边界；此指纹不是任意 Python 程序的完备依赖分析，也不是对恶意源码的安全认证。

run.py 在创建运行目录前校验；直接训练驱动在创建 Runner/环境前使用同一 `prefix_spec` / `verify_manifest` 校验。schema 1 没有第一阶段实现指纹，无法安全地由整仓库 source_sha256 自动补出；必须重新生成 schema 2，旧文件不修改。

`branch_restore.json` 在实际回放导入操作之后写入：

| 字段 | 含义 |
| --- | --- |
| `prefix_replay_snapshot_episodes` | Prefix 文件里的样本数 |
| `restored_replay_episodes` | 实际导入数量；FT 固定为 0，ER 在导入返回后读取 |
| `discarded_prefix_replay_episodes` | FT 主动丢弃的快照样本数；ER 为 0 |
| `replay_restore_mode` | 实际选择的 ft / er |

TaskReplay 切换记录另含 `replay_episodes_before_switch`、`replay_episodes_after_switch` 和 `replay_switch_evicted_episodes`；ER 配额淘汰不会回写初始导入数。恢复失败时不会提前生成成功记录。FT/ER 的数据导入、配额、抽样和训练顺序不变。

所有本仓库直接清空 PyTorch 私有 Hook 字典的代码已去除。Learner 为自身 online/target 观察器保存独立 handle，需要关闭诊断时只移除这些 handle；target 拷贝仍保留外部观察器及原有原生采集行为。临时分析 unroll 使用 finally，原生训练入口在 finally 移除长期观察器。其他模块注册的 Hook 保留。

### 本次验证与修改范围

78 项相关 CPU 测试通过（Prefix、TKPR、ASPR、CPR、Replay、Hook、来源指纹及 TS3 回归）；源码语法与 `git diff --check` 通过。额外 AST 对比确认四个 Learner 的 train 函数、TaskReplayBuffer 及其 restore/make 实现未改，原生 run_sequential 的训练语句顺序保持不变。CLI dry-run 及旧 schema 拒绝测试未创建运行目录。未启动 SMAC/SC2 或正式训练。

先合并 origin/main `20ac440`，合并提交为 `d0b2966`，无冲突，新增结果完整保留。根与第三方 LICENSE 原文未改。第三方授权待确认项见 [许可证清单](THIRD_PARTY_LICENSES.md)。

实际修改文件（按目录）：

- `docs/PREFIX_REUSE.md`, `docs/PROVENANCE.md`, `docs/THIRD_PARTY_LICENSES.md`
- `drivers/continual/action_set_train.py`, `drivers/continual/common/prefix.py`, `drivers/continual/common/prefix_implementation.py`, `drivers/continual/detection_common.py`, `drivers/continual/hetero_protection.py`, `drivers/continual/hetero_train.py`, `drivers/continual/history_train.py`, `drivers/continual/static_detection.py`, `drivers/continual/train.py`, `drivers/continual/ts3_train.py`, `drivers/continual/ts3_vdn_detection.py`, `drivers/continual/ts3_vdn_type_detection.py`, `drivers/continual/ts4_diagnostics.py`, `drivers/continual/ts4_train.py`
- `scripts/benchmark_szc_tkpr_k1.py`, `scripts/check_a2_local_gem.py`, `scripts/diagnose_all_locations.py`, `scripts/network_similarity.py`, `scripts/type_ablation_winrate.py`, `scripts/type_matched_ablation.py`, `scripts/validate_hetero_agent.py`, `scripts/validate_protection_locations.py`, `scripts/validate_ts4_vdn.py`
- `src/components/task_replay.py`, `src/learners/dmaq_qatten_learner.py`, `src/learners/iqn_learner.py`, `src/learners/q_learner.py`, `src/learners/rmix_learner.py`, `src/run.py`, `src/utils/hook_lifecycle.py`
- `tests/test_full_preference_prefix.py`, `tests/test_generic_prefix.py`, `tests/test_prefix_reliability.py`, `tests/test_ts3_driver_regression.py`
