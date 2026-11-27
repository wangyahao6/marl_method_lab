# VDN 网络骨架索引

[architectures.json](architectures.json)登记共享循环 Agent 与 VDN Mixer 的骨架，复用现有 [Catalog](../results/catalog.json)、[运行索引](../results/run_index.csv)与[环境／任务索引](../environments/README.md)。`build`从运行记录自动更新运行数及种子，不手工维护第二份统计。

| architecture_id | Agent / 主干前向路径 | H | 配置入口 |
| --- | --- | --- | --- |
| `vdn_2ff_h64` | `rnn` / Linear → ReLU → ActivateLayer → Linear → ReLU → GRUCell → Q head | 64 | `src/config/algs/vdn.yaml`；宽度来自 `src/config/default.yaml` |
| `vdn_1ff_h64` | `single_ff_rnn` / Linear → ReLU → ActivateLayer → GRUCell → Q head | 64 | `src/config/algs/vdn_1ff.yaml` |
| `vdn_1ff_h128` | `single_ff_rnn` / Linear → ReLU → ActivateLayer → GRUCell → Q head | 128 | `src/config/algs/vdn_1ff_128.yaml` |

三者的 H 同时是前馈层和 GRU 隐藏维度，Mixer 均为 `VDNMixer`，对 Agent Q 求和。输入维度和动作数由环境决定，不另造骨架 ID。`ActivateLayer`是初始为单位乘法的诊断／干预门；可选 TRHR 及方法的门控／更新行为继续属于方法配置，不增加骨架类别。旧 `RNNAgent` 的未使用 `fc1` 不在实际前向路径上，不能被算成第三个前馈层。

## 架构与方法身份

`run_index.csv`新增 `architecture_id`、`agent_type`、`hidden_dim`、`mixer_type`、`architecture_source`。其中 `architecture_source`为 JSON，记录配置证据位置、记录的代码提交、核对的源码路径和哈希、匹配依据及未知原因。

VDN 的 `algorithm`统一为 `vdn`；原配置选择器 `vdn`／`vdn_1ff`／`vdn_1ff_128`保存在新增的 `algorithm_config`。`method_id`仍表示 baseline、topk_preference、tkpr_refresh 等实际方法；`method`仍保留原公开标签。

`method_identity_version=2`的 `method_variant_id`不再包含 Agent、隐藏维度、Mixer 或网络配置别名。同方法、同已记录训练／方法设置在不同骨架上使用相同 variant。原有偏好参数、K/gap、干预组、优化器／学习率、FT/ER 回放与探索设置仍保留，避免将已有实质差异合并。相同 variant 不意味着协议、预算、源码或环境可比，比较时仍应匹配相应字段。版本 2 会重新生成派生 variant ID；历史 `runs.csv`、配置、协议及结果不改写。

## 历史证据与未知策略

解析顺序是已有生效配置中的 `agent`、`rnn_hidden_dim`、`mixer`、`mac`，再核对该运行记录的 Git 提交中的 Agent 注册绑定、实际前向主干与 Mixer 实现。配置入口仅说明标准入口，不用于猜测历史运行。相同结构换一个 YAML 名称仍是同一骨架；同名配置中字段被覆盖时按生效字段核对。缺失宽度时不按当前默认值补成 64。

解析器版本 1 使用静态 AST 签名：Agent 构造器中直接赋给 `net`、`rnn`、`fc2`、`hidden_dim`的表达式，以及完整 `forward`；Mixer 使用 `forward`。它识别登记并审查过的主干实现，忽略注释、格式、Python 3.12 新增的空 type_params 元数据及未参与该主干的旧兼容张量。登记表固定参考提交和预期签名，不在每次构建时从最新源码自动认可新结构。修改提取边界、算法或未知判定规则时必须升级 `resolver_version`并显式迁移；不把它当作完整训练代码指纹。

缺少必要字段、控制器未知、代码版本不可访问、注册绑定不同、实现签名不认识或同一运行的结构证据冲突时，`architecture_id=unknown`，原因保留在 `architecture_source`。已知的 Agent／宽度／Mixer 字段仍可保留。QMIX 和单智能体 DQN 不属于这三个 VDN 骨架，不强行分配相近的 VDN ID。未解析的 VDN 记录还会进入 `index_audit.json`。

`source_verification=git_commit_only`表示仅有生效配置和启动 Git 版本证据。本次继承的冻结元数据没有保存启动源码哈希／源码归档内容，因此不能据此证明当时没有未提交改动或运行时动态替换。生成器不会把当前工作区源码当作历史源码。新的只读元数据导出器保留启动时的相关网络文件哈希；若它们与提交版本一致，标记 `launch_hashes_match_git`；发现差异时标记未知，不用 Git 版本强行覆盖实际源码证据。历史快照不补写。

这是骨架分类，不是模型权重验证、Prefix 恢复验证或完整动态调用图验证。自定义 MAC、动态 monkey patch、归档外的运行时改动及未登记的新骨架需要补充实际证据，不能用目录名或算法选择器替代。

## 查询与覆盖

在仓库根目录执行，均只读索引，不导入 PyTorch／环境、不启动训练：

```sh
python3 scripts/experiment_index.py build
python3 scripts/experiment_index.py check
python3 scripts/experiment_index.py architectures
python3 scripts/experiment_index.py architectures --architecture vdn_1ff_h128
python3 scripts/experiment_index.py query --architecture vdn_1ff_h64 --method tkpr --seed 3535
python3 scripts/experiment_index.py query --architecture vdn_2ff_h64
python3 scripts/experiment_index.py matrix architecture-method
python3 scripts/experiment_index.py matrix architecture-environment --coverage complete_result
python3 scripts/experiment_index.py matrix seed-coverage --architecture vdn_1ff_h128
```

`architectures`包含定义、全部已记录种子及 run_id；`query --architecture`返回对应完整运行行，包含失败与前缀，可继续组合环境、Benchmark、方法、种子和实验筛选。`show RUN_ID`也显示架构字段和源码证据。

架构矩阵包含登记骨架，即使某格没有运行；未登记的网络保留单独 `unknown`行，不挤入 VDN 骨架。默认排除共享前缀与 pair-reference，可用 `--include-prefix`查看。`reported_completed`计来源报告的完成运行，`complete_result`只计有完整公开端点的运行；每格另列 observed_runs、failed_runs 和完成运行 ID。0 表示索引没有相应记录，不是零性能，也不证明从未执行。
