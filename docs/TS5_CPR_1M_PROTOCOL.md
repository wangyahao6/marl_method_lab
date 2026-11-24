# TS5-1M TKPR-CPR 共同前缀对照

2026-11-24 用户确认四组：VDN-ER、TKPR-K1、TKPR-CPR、TKPR-Refresh；两个训练种子3535、2222。沿用既有TS5-1M任务、网络、探索及评价协议。每种子训练一个A前缀1M，然后四个分支各训练B/C/D/E各1M，名义新增总量34M。不是四个独立A阶段，不共享B之后的训练。FT不作为本批额外分支，不能把旧2M FT当成同批配对对照。

## 配置与入口

- `configs/continual/action_preference/ts5/tkpr_cpr_1m.json`：从K1-1M派生，仅替换方法块并增加原CPR三个参数。
- `configs/continual/action_preference/ts5/cpr_shared_1m/`：prefix、vdn_er、tkpr_k1、tkpr_cpr、tkpr_refresh。`full_shared_prefix: true`显式启用完整共享状态，旧共享入口行为不变。
- 入口统一 `scripts/run.py --suite vdn_action_set --algorithm vdn_1ff`；prefix用`--method topk_preference --preference-role shared-prefix`，分支用对应方法及`--preference-role shared-branch --parent <prefix>`。
- 顺序A/B/C/D/E是双方镜像S5/Z5/C5/S2Z3/Z2C3；累计阈值1/2/3/4/5M，完整回合后切换。ε逐任务重启，其他设置沿用现有1M配置。K1/CPR/Refresh是FT，ER按原任务均衡TD回放。
- K1/CPR辅助更新跨历史任务总8条；Refresh每任务8条。CPR weight/temperature/min_gap/q_scale_floor/k = 1/1/0.5/0.1/1，置信度阈值0.5、切换间隔0.1、任务门控容差0.05。没有改变筛选、损失或增加辅助目标。

## 完整A状态与分支隔离

审计发现旧`shared-prefix`不保存TD ReplayBuffer，且只允许FT共享分支。新显式模式补齐：

- A阶段模型目录保留online模型和optimizer，`return_state.th`另外直接保存online/target、mixer、optimizer、两MAC隐藏状态、训练计数、目标更新计数、探索率/退火起点、所有RNG、环境分布状态及偏好采样RNG。
- `prefix_replay.th`保存完整原ReplayBuffer，包括数据、容量、填充、环索引和库存。ER分支复用现有`restore_replay_buffer()`把已知单任务A的原缓冲映射到任务槽位；FT类恢复同一A状态后按既有B切换语义丢弃A的TD回放，不改变FT定义。
- prefix阶段末评价只真实执行/记录一次。子retention文件中的A行只是父结果引用；导出evaluations/stage_metrics不重复生成A评价。
- A成功池只采集一次。K1直接使用，CPR增加原始Q副本/全true掩码/来源0，Refresh仅增加来源0元数据；这些初始化不改变obs/actions/初始old_q_values。ER保留该池的独立审计副本，但不将成功池当作辅助损失或TD池。
- 每个子进程从只读父产物恢复，并在自己的run目录复制Memory；没有共享可写tensor对象或B后的训练。`branch_restore.json`记录prefix ID、branch ID、seed、A checkpoint SHA、模型与初始Memory输入指纹、恢复的Replay库存/索引。
- 各分支保留阶段模型、日志、当前历史池、`final_replay.th`与最终`return_state.th`（含target/optimizer/RNG等）及其manifest。本模式没有保存B–D的全部中间TD池快照。

身份/指纹用于审计记录，不添加基于诊断数值的训练中止阈值。输入结构缺失仍按已有正常恢复要求报错。单个运行失败保留状态，不自动重跑，也不停止其他独立分支。

## 数据审计与新增诊断

现有日志已记录：evaluations的wins/episodes/win_rate/phase/task/t_env/outcomes，阶段ACC/BWT/FWT，逐更新历史任务与episode_ids/采样数量、loss、Valid/Decision States、Valid Pairs、coverage、Pair Violation/Top1 Flip Rate、participation，以及每100个learner更新的Agent梯度范数、范数比和余弦。

只补充缺失的记录：

1. CPR旧/新置信度的n、均值、中位数、P10/P90（线性插值，完整池真实合法决策；不含padding/bootstrap）。门控失败不执行新模型重评，因此新置信度缺失，而不是填0；诊断失败不改变选择规则。
2. CPR逐更新`raw_decision_states`、`reference_mask_valid_states`，并导出`effective_coverage_of_raw_decisions`。原coverage的分母已应用有效参考掩码；掩码覆盖与min_gap后的实际约束覆盖必须分开解释。
3. 此实验模式的`memory_collection.jsonl`保存每次初始/Refresh采集的成功、失败、种子块和耗时，`memory_inventory.jsonl`保存每阶段池库存；baseline不额外采集旧任务成功池。
4. `phase_execution.jsonl`记录阶段实际步数、更新/回合数、TD回放库存及阶段墙钟；`status.elapsed_seconds`记录运行总墙钟。

原始逐更新日志保留实验主机，公开CSV按阶段/历史任务精确聚合，记录分母和缺失数。任务损失贡献按有效任务等权计算，不按Pair数量分配权重。梯度参数范围和诊断频率沿用当前实现；不凭空补全baseline的偏好梯度。

## 评价、AUC与汇总

每25k环境步当前任务32局；阶段末全部已见任务各128局。所有组相同评价种子。阶段AUC只使用phase_start+periodic的regular种子块，梯形积分，最后一次periodic到实际阶段末保持末值，不混合focus评价。当前任务AUC和旧任务矩阵同时展示。

每分支训练成功后通过`scripts/evaluate.py --episodes 128`独立进程复核最终模型全部已见任务。它沿用stage_end的focus种子和seed/index集合，因此标为独立进程复核，不称独立测试集。核验实际outcomes后记录是否重复。

CPR在边界c后更新的参考影响c+1训练；E末更新不影响已经完成的E表现。不能把参考更新比例、局部覆盖/梯度相似性与胜率的相关变化直接解释为因果。只有两个训练种子，不把多个阶段、任务或评价回合当作额外训练重复。

实际执行冻结在独立Git worktree；其 `.venv` 和SC2均只读复用既有运行时，`runs/`、`results/`链接主仓库产物，scheduler状态仍为同一个全局数据库。这样排队期间主分支的新改动不改变本批代码。具体固定提交和cwd保留在plan/launch中。

运行前生成完整计划（`--repository`可指定固定worktree，`--artifacts`指定主仓库）：

```sh
.venv/bin/python scripts/prepare_ts5_cpr_1m.py
.venv/bin/python scheduler/service.py submit results/ts5_tkpr_cpr_1m_20261124/plan.json
```

完整计划包含2个prefix、8个训练branch、8个独立评价和1个terminal汇总任务。保持全局6槽；本批最多同时占2槽。每个分支依赖自己的prefix，独立评价依赖对应分支成功；汇总等待所有任务终态，失败也如实报告。调度器只按已有资源规则派发，不增加训练预算或验证训练。

结果写入`docs/results/comparisons/ts5-tkpr-cpr-1m/`。`scripts/summarize_ts5_cpr_1m.py`生成所需CSV/manifest/执行摘要、训练曲线、最终任务保持和CPR机制图，按两个训练种子等权报告均值/样本SD与同种子差值。待完成前只显示pending/部分完成，不生成预期数值。
