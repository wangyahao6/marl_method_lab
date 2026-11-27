# 环境、任务与实验查询

[环境登记](environments.json)维护平台名称、代码位置和支持边界；[任务登记](benchmarks.json)由真实配置、源码中的字面量登记及运行记录生成。[Catalog](../results/catalog.json)仍是实验报告的主入口，新增 `global_index` 只描述其索引来源与生成规则。

## 四层身份

| 概念 | 含义 | 例子 |
| --- | --- | --- |
| Environment | 环境平台及其适配实现 | SMAC、SMACv2、MinAtar、MPE、Stag Hunt、Gymnasium Classic Control |
| Task | 一个具体游戏、地图或阵容定义 | SpaceInvaders；MountainCar T1（目标位置 0.45）；指定阵容的 stalker |
| Task Sequence | 有序 Task ID 列表，保留重复与回访 | SpaceInvaders → Freeway → Breakout → Freeway → Freeway → Breakout → Breakout |
| Benchmark | 单任务、任务集合或任务序列对应的评测问题 | SMAC 2s3z；SZC；TS5。通过协议来源关联训练预算、方法和探索设置 |

Task/Benchmark ID 来自结构化定义的内容哈希，包含平台、地图、阵容及可获得的环境设置，不包含种子、方法、训练预算或数据路径。任务名相同、定义不同的 Benchmark 保留为不同 ID；不把历史默认值补写到旧运行。无顺序的任务字典登记为 `task_collection`，不擅自认作已运行的顺序。异质驱动的 A→分支→A 和 sequential 驱动的 scenario 顺序按当前驱动语义及显式配置解析。

代码中的地图表、MinAtar `GAMES`、MountainCar `GOALS`、MPE 场景及 Stag Hunt 模板只提供代码证据。Stag Hunt 的参数化模板未被当作确定的实际任务配置。当前平台登记不等于已经安装依赖或通过环境启动验证。

## 证据与生成文件

- `code_only`：只发现代码或配置声明。不能据此认定曾训练。
- `actual_runs`：有已发布 `runs.csv` 的实际运行记录，或运行元数据中的训练进度／失败进程证据。
- `complete_results`：存在 `reported_complete_result` 运行，即报告完成、关联公开文件，并有该序列所有阶段的端点记录。它是索引层的完整性分类，独立评价质量与可比较范围仍以原报告为准。

`status` 是来源报告的运行状态；`result_status` 独立记录结果归档程度。`reported_prefix_result` 只表示前缀端点，不能充当整个序列的方法结果。分支的阶段覆盖可以关联其直接父运行的已发布端点；缺少证据时保留 `published_partial_or_unverified`。已有进度但没有完成证据时不升级为完成，退出码 0 也不能单独证明完成。缺失字段用 `unknown`，空列表用 `[]`。

[run_index.csv](../results/run_index.csv)每个 `run_id` 一行；重复公开引用合并到 `experiment_ids`、`result_paths`、`source_records`，不增加训练次数。`experiment_id` 是 Catalog 实验身份，`experiment_group_id` 保留 scheduler 批次／会话身份，两者不互相冒充。`method_id` 保留底层方法名，`method`保留公开标签；`method_variant_id`还区分网络、优化器、FT/ER 回放与探索设置等已记录差异。相同 variant 不自动证明实验可比，仍需查看完整配置和协议。

`parent_run_id`、`role`、`scheduler_task_ids`、`scheduler_dependencies`保留前缀树和调度关联。`logical_run_id`仅在元数据显式给出 group、branch、seed 时生成；`attempt`优先取显式元数据。运行名的 `attemptNNN` 只能作为标签记录（`attempt_source=run_id_label`），不会据此猜测 `retry_of`。失败运行保留独立行，只有排队记录且没有运行证据的任务进入审计报告。

[analysis_index.csv](../results/analysis_index.csv)将公开离线分析、scheduler 的评价／汇总任务及运行目录中的分析与训练分开。`input_run_ids`来自显式字段、路径中的完整运行 ID 或调度依赖链；`input_sources`保留输入路径，未能匹配的显式运行 ID 留在 `unresolved_inputs`。引用已删除或未归档的输入不被替换成猜测的运行。

[index_audit.json](../results/index_audit.json)记录无法确认的字段、缺失父运行、冲突和只有 scheduler 证据的任务；不反写修补原始结果。`associated_run_count`表示 Task 出现在运行所选 Benchmark 中，不断言失败运行实际训练过该序列的每个 Task。

## 生成与查询

在仓库根目录使用系统 Python（仅标准库，无须导入 PyTorch、SMAC 或 SC2）：

```sh
python3 scripts/experiment_index.py build
python3 scripts/experiment_index.py check
python3 scripts/experiment_index.py environments
python3 scripts/experiment_index.py benchmarks --environment smacv2 --observed
python3 scripts/experiment_index.py tasks --environment minatar
python3 scripts/experiment_index.py task_sequences --environment minatar
python3 scripts/experiment_index.py query --environment smacv2 --method tkpr --seed 3535
python3 scripts/experiment_index.py matrix seed-coverage --experiment ts5-tkpr-cpr-1m
python3 scripts/experiment_index.py matrix environment-method --coverage complete_result
python3 scripts/experiment_index.py matrix benchmark-seed --environment smacv2
python3 scripts/experiment_index.py show minatar_ft_tkpr_k1_seed3535_20261122_ft_attempt001
```

以上均输出 JSON。`query`支持 environment、benchmark、experiment、method、seed、run-status 组合筛选；method 支持标签、底层名或 variant ID 的子串。`show`包含配置、来源、父运行、子运行、Benchmark 定义和关联分析。矩阵默认排除 prefix/shared-prefix/pair-reference；可用 `--include-prefix`查看。默认 `reported_completed`计数来源报告已完成的运行；`--coverage complete_result`只计有完整归档端点的运行，同时仍输出 observed_runs 和 failed_runs。0 表示当前索引未发现符合条件的记录，不证明从未运行，也不是零性能。环境矩阵包含已登记但没有运行的环境；方法和种子列来自当前筛选结果，不虚构计划种子。其他矩阵的行来自已有运行，代码清单可另用 benchmarks/tasks 查询。

生成器直接复用 `catalog.json`、各实验 `runs.csv`、配置快照、阶段结果和 `configs/`。当前已检查本地 Git refs，将有独有历史记录／配置的分支 SHA 固定在 Catalog 的 `branch_sources`；用 Git 对象只读访问，不切分支、不合并、不执行历史代码。`git:<sha>:<path>`表示该版本中的文件，不是当前检出路径。复现生成需要这些提交对象仍在本地；缺失时明确报错，不静默漏扫。新增分支记录时人工审查并更新此唯一来源列表，再生成即可。

运行元数据输入是 [runtime_records.jsonl](../results/runtime_records.jsonl) 的冻结导出，头记录包含采集时间、主机和目录。本次快照为 2026-11-27，包含五个 Ubuntu 检出及调度任务；不能作为以后的实时运行状态。生成时另只读扫描本机注册 worktree 中可访问的 `runs/` 元数据。未连接远端的目录不声称已扫描。

需要更新运行快照时，在已授权的实验主机使用只读导出器，例如：

```sh
python3 scripts/collect_experiment_metadata.py --host HOST --root /absolute/repo > /tmp/runtime_records.new.jsonl
```

先核对新导出范围，再用新快照更新索引。导出器只读取元数据，SQLite 使用 `mode=ro`；不读取模型权重、不调用 scheduler 的写连接、不提交或启动任务。生成文件是可重建视图，更新时不手工编辑 CSV 或矩阵，也不改历史实验文件。`check`重建并逐字比较生成物；来源数据发生变化时应重新生成。

当前局限：部分早期运行没有 Catalog 实验 ID、明确 parent 或协议 ID；许多旧评价／汇总任务只有批次路径，无法唯一确定输入运行。缺失训练输出的调度任务另列待确认。MinAtar/MountainCar 的源码与结果仍在其原功能分支，由固定 SHA 引用；没有迁移或合并到当前分支。这里只整理可追溯关系，不重算指标、不验证模型恢复等价性，也不提供后台同步。
