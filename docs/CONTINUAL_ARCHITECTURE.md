# 持续学习驱动架构

`scripts/run.py` 以原有参数和协议启动 `drivers/continual/*train.py`。各驱动决定何时切换任务、更新探索、训练、评估、筛选和保存。QMIX 精确步数截断、VDN 完整回合边界、各方法的评估种子、旧任务复测和重试策略继续由对应驱动决定。

| 模块 | 职责 |
| --- | --- |
| `drivers/continual/common/runtime.py` | EpisodeBatch 规格、TD 批次裁切和设备传输、探索切换记录、RNG 快照、阶段与状态记录 |
| `drivers/continual/common/checkpoints.py` | 模型目录、阶段目标网络、延续快照的公共字段、网络和优化器恢复 |
| `drivers/continual/common/evaluation.py` | CPU 贪心评估块和共享的固定种子计算 |
| `drivers/continual/common/records.py` 与 `metrics.py` | 原有 JSON、JSONL 和 learner 日志，以及统一的实验指标 |
| `src/components/episode_buffer.py` 与 `task_replay.py` | TD 训练经验、任务切换时的 FT / ER / retain 策略及其回放恢复 |
| `drivers/continual/memory/` | 方法历史数据及其持久化，与 TD 池独立 |

## 方法历史

- `History` 按阶段或任务键保存对现有载荷的引用。调用方用映射接口注册和读取，使用 `save(path, payload)` 原子保存，使用 `load(path, device='cpu')` 读取可信本地 Torch 文件。固定探测集可在映射中只登记路径，按需读取。
- `PreferenceHistory` 用 `restore_success` 恢复前缀已有的胜局池，用 `sample(current_task, batch_episodes, rng)` 按原策略选取旧任务最近的非空池和回合索引。ASPR 的好动作掩码、T1PR / FRPR 的参考 Q 值与对应轨迹一起保存，参考 Agent 使用阶段检查点路径。失败轨迹与固定评估探测集仍各用既有文件。
- CHI 继续使用 `chi_probes/phase*.th` 的 `batch` 与 `weights`。Local GEM 保持 CPU 历史和计算设备上的只读视图，CPU 执行时共享同一底层张量。GEM 的 `gem_state.th` 保留原有诊断字段，分支的参考轨迹仍从指定前缀的 `screen_input.th` 加载。TS4 的纵向诊断使用内存中的固定阶段探测。
- 探测、标签和约束仍由各方法定义。新方法可以复用历史载荷接口并提供自己的抽样策略。已有 `action_set_memory/phaseNNNN/`、`chi_probes/` 和诊断快照可以直接读取。排名方法恢复仍要求成功池已具备 `old_q_values`。

## 实验指标

新运行在原有日志旁追加 `experiment_metrics.jsonl`。独立评估在对应的 `evaluation/<timestamp>/` 目录记录已执行评估，使用来源训练运行的编号和种子。每行一个有限标量。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前为 1 |
| `run_id`, `algorithm`, `method`, `seed` | 运行身份及训练种子 |
| `category` | `Performance` 或 `Mechanism` |
| `phase`, `task`, `t_env` | 零基阶段、指标对应任务、累计环境步数 |
| `metric`, `value` | 指标名与标量 |
| `context` | 适用的事件、评估局数、当前训练任务、层、神经元索引、历史任务、角色等维度 |

- Performance 写入原评估的 `win_rate`（0–1），以及驱动已计算的 `acc_percent`、`bwt_pp`、`fwt_pp` 等。阶段汇总沿用各驱动既有口径，TS3 系列汇总可用 `context.source_run_id` 追溯共用前缀或恢复来源。
- Mechanism 在既有诊断时点记录每神经元的活动或归一化分数、阶段开始时受保护神经元占该层的比例、GEM 投影与点积诊断、偏好违反率或 Top-1 翻转率等。保护比例以当时的保护名单为分子，恢复中的神经元另由原调度日志记录。偏好指标的历史任务及来源阶段在 `context`。
- 文件按事件写入。相同 `phase/task/t_env/metric` 可能来自不同评估或诊断。汇总时可先按 `context.event` 过滤，再按层、来源阶段或神经元索引分组。缺失和非有限数值省略。只使用已有指标的运行可继续读取 `evaluations.jsonl`、`retention.json`、`metrics.jsonl` 和各方法的日志。

新增指标示例

```python
from drivers.continual.common.metrics import Metrics

metrics = Metrics(run_dir, algorithm, method, seed)
metrics.record("Mechanism", phase, task, t_env, {"projection_residual": residual},
               event="projection_audit")
```

离线回归测试

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```
