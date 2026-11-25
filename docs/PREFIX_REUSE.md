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

## 依赖审计与提取规则版本 2（2026-11-25）

Manifest schema 仍为 2，状态格式仍为 1；第一阶段实现提取规则独立升级为 `implementation.version: 2`。本次扩充覆盖边界并将 AST 文档字符串排除，不能继续沿用规则版本 1 的语义。`rules_sha256` 记录实际执行的提取器规范化 AST，汇总哈希同时纳入版本、规则哈希及各组件哈希。测试固定已登记版本的规则哈希；未来调整提取边界或算法必须递增版本并登记新的规则哈希。规则版本 1 的旧 Manifest 不自动重算或升级，明确拒绝；缺失 Manifest、schema 1、未知 schema 的既有拒绝策略不变。

### 实际调用链与维护清单

| 执行位置 | 实际作用与依赖 | 校验范围 |
| --- | --- | --- |
| `scripts/run.py` → `scripts/launchers/vdn_action_set.py` | 读取/验证协议，生成命令、seed、角色及环境变量，保存启动源码/配置；启动新 Python 驱动进程 | 参数与训练输入由驱动实际生效的 Prefix Spec 校验，线程/库/CUDA 等由 runtime 段校验；完整启动源码保存在 code_version。没有把其他 suite 的启动代码整体纳入第一阶段指纹 |
| `action_set_train.py` 模块初始化与 `main` | 导入 `train`，将 `action_set_protoss` 注册为 ExactProtoss；`main` 读启动协议、固定 Torch 线程数、分派训练 | 模块级可执行语句、函数声明时求值的默认值/装饰器/注解、训练模式 main 路径；最终评价分支不执行。第一阶段初始化、phase 预算及 TD while 路径继续单独提取 |
| 导入 `train.py` | 建立 ROOT、sys.path、SC2PATH，读取启动协议；构造 `ModuleType('envs')` 和其 REGISTRY，发布到 `sys.modules['envs']`；之后导入 envs registry 与 EpisodeRunner 等工厂 | **新增完整模块初始化 AST**，包括定义之间/之后的顶层语句；不只记录 import。导入不会调用该文件的独立 QMIX main，故独立训练函数体不进入指纹 |
| EpisodeRunner → env_REGISTRY → ExactProtoss | EpisodeRunner 构造/运行环境，调用 reset、step、get_obs/state/avail_actions；ExactProtoss 通过属性转发使用 SMACv2 wrapper | Runner、ExactTeams、ExactProtoss 构造/转发/任务选择/reset/step/分布快照，以及 seed_distributions；分支专用 restore_distribution 不属于前缀训练路径 |
| SMACv2 wrapper → StarCraft2Env | 分布查找/采样、阵容与初始位置、地图注册与查找、游戏进程接口、观测/奖励/终止条件 | 仓库内 SMACv2 env 的 Python 环境实现、分布、地图和 MultiAgentEnv；不纳入未使用的 PettingZoo 适配层 |
| BasicMAC → 选中 Agent | 根据 registry 构造实际 rnn / single_ff_rnn，拼接 observation/last-action/agent-id，更新 RNN hidden 与 Q 输出 | BasicMAC、实际 Agent、ActivateLayer，以及 controllers/agents 选中 registry 的赋值和定义来源；未选中的 Agent 实现不纳入 |
| Action Selector / 探索 | ε-greedy 抽样，DecayThenFlatSchedule，切换时探索起点 | 选中 registry、EpsilonGreedyActionSelector、schedule、log_task_epsilon / epsilon_on_switch |
| EpisodeBatch / ReplayBuffer → td_sample | 插入完整 episode，抽样、截断到 filled 长度、设备转移与 OneHot 预处理 | 普通 episode replay 与 transforms 实现；start_replay_task 提取普通 ReplayBuffer 路径，ER/FT 后续 task 配额/采样代码不进入第一阶段实现指纹 |
| QLearner → VDNMixer / Optimizer | online/target unroll、TD 目标、损失、backward、clip_grad_norm、step、target 更新 | 实际 QLearner 构造/train/target 更新/cuda/observer 生命周期及模型保存，VDNMixer；RMSprop/Adam、Torch 运算由记录的库版本/运行配置约束。`check=False` 不调用 find，Prefix 无偏好消费者、无 gradient_monitor，故不纳入这些未执行方法/辅助诊断实现 |
| QLearner / Runner 的已调用辅助代码 | 原生 observer 安装/移除，Logger.log_stat 及 metrics 包装；训练过程保存/散列模型及边界状态 | **补入** Logger、records、必要 metrics 方法、checkpoint 快照/哈希/保存及 full_state_extra 的实际实现；Metrics.preference 等方法专用处理仍不纳入 |
| 可选 TRHR | Agent 构造时仅在 cfg.trhr 为真时替换 ReLU；Learner 实际调用 capture_activity、TRHRReLU.observe/_adjust 和 valid_a2_rows | **仅启用 TRHR 时**加入 TRHRReLU 类及实际辅助函数，不因无条件 import 就纳入其控制器实现 |
| 可选 A2Trace | 仅在 cfg.a2_trace 为真且 cfg.evaluate 为假时构造；Learner 调用 before_step、observe，observe 可调用 flush、_norm | **仅实际启用时**加入 A2Trace 及其方法；即使 TRHR 关闭，Trace 仍使用 capture_activity、valid_a2_rows，故这两个辅助实现必须加入。仅 Trace 启用时不纳入未使用的 TRHR 控制器类 |

这是一份沿当前调用路径人工核查、由测试维护的依赖边界，不是仅由 import 扫描得出的自动闭包。cfg 中可选开关的指纹覆盖不代表本次新增了相应 CLI 开关、训练干预或诊断状态恢复能力；默认参数与训练逻辑不变。生效配置继续进入 Prefix Spec；YAML 注释、其他算法配置及方法损失源码不因本次审计被整体纳入。

### Manifest 完整性与路径

校验现在要求顶层身份/版本/Spec/来源/采集配置/文件清单，检查字段类型、SHA-256 形状、Spec 自校验和实现汇总哈希，并拒绝重复 JSON key。必须有非空 `files`，且包含 retention、config、env_info、return_state、prefix_replay、initial_agent、第一阶段 success/failure 历史文件；缺文件或不在哈希清单中均拒绝。

retention 必须只有第一阶段，phase/task 与 Spec 一致。其 checkpoint 必须为 Prefix 内的目录，且 `agent.th`、`mixer.th`、`opt.th` 全部存在并被哈希清单覆盖。所有列出的产物均重新校验文件哈希，不通过只检查存在性替代。Manifest 产物路径、retention checkpoint 路径与来源清单路径拒绝绝对路径、`..`、驱动器路径、反斜杠和非规范别名；产物 symlink 也不能指向 Prefix 目录之外。新写入路径统一使用相对 POSIX 形式。

完整性检查不反序列化 Torch 文件，不宣称仅凭文件存在及哈希就证明张量/优化器在语义上正确；原有 CPU 恢复测试继续负责实际状态回读。历史产物不修改，不通过补写元数据“升级”成当前规则产物。

### 明确局限

- 不对任意 Python 程序自动证明依赖闭包。运行时 monkeypatch、用户插件/回调、预加载模块、动态 import 或 import 时新增副作用需要重新审计；不能把版本和哈希匹配解释为这些动态行为已被验证。
- 第三方包由版本/构建信息约束，不逐字节审计其安装目录；同版本本地被修改的依赖、SC2 二进制/实际部署地图及 NVIDIA 驱动不在本次静态源码指纹的完整证明范围内。
- 核心定义与环境模块按函数/类/模块粒度保守校验；并非任意语义等价修改都能保持指纹。文档、注释、格式以及规范化 AST 的文档字符串不影响指纹，但新增可执行初始化必须被视为代码变化。
- Prefix 历史采集在训练边界之外隔离执行，原始历史文件受产物哈希保护；本指纹不把 TKPR/ASPR/CPR 的后续数据处理视作共同前缀训练依赖。

### 本次验证

83 项相关 CPU 回归通过；最终完整性与路径规范化修改后，42 项重点复核再次通过（包含 11 项可靠性测试及 Prefix、共享前缀、TKPR、ASPR、CPR、Replay 回归）。可靠性测试中，4 项验证规则版本/代码指纹，1 项验证运行信息，3 项验证 Manifest（循环覆盖缺字段、空清单、缺失/未列出的必要产物、checkpoint、危险路径与 symlink 等子案例），其余 3 项为 Hook 回归。`py_compile` 和 `git diff --check` 通过；没有启动 SMAC/SC2 或正式训练。

修改仅在 `common/prefix.py`、`common/prefix_implementation.py`、`tests/test_prefix_reliability.py`、本说明，以及 `action_set_train.py` 的两处相对路径序列化（统一 POSIX 分隔符）。训练数学、优化器、超参数、采样及更新顺序未修改，已有结果与历史产物不改写。
