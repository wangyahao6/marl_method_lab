# VDN task-specific scores and training-history analysis

Authorized 2026-11-03: VDN baseline, seeds3535/2222/1111, mirror5v5 Stalker(S), Zealot(Z), Colossus(C),2M environment steps/stage, all six permutations. No neuron intervention. Existing TS4 state-weighted activation score, shared Agent A1=net[2], A2=net[4],64 units each; do not modify existing scoring implementation.

## Comparisons and computation

Each seed trains a prefix tree:3 single tasks,6 ordered pairs,6 ordered triples. Each node adds nominal2M steps, totaling30M/seed and90M actual training across three seeds (full-path equivalent108M). Single nodes are scratch controls; pair nodes are A->B controls. Reused nodes are a single observation, not extra replicates. Full orders:SZC,SCZ,ZSC,ZCS,CSZ,CZS. Root networks initialize with the same seed within each seed block. All nodes retain shared network shape and three-unit-type encoding.

Cumulative thresholds2/4/6M at whole-episode boundaries; actual stage lengths and overshoot recorded. Transfers retain online/target models, RMSprop optimizer,5000-episode replay, target-update counter, episode and environment-step counts, cumulative epsilon, Python/NumPy/Torch and recoverable environment distribution RNG. Branches create new SC2 processes; engine state is not bitwise continuous. No task-ID input, separate heads, replay reset or exploration restart. Reward denominator recomputed at each reset.

## Measurement

At every stage end, use last32 complete current-task training episodes, excluding padding/bootstrap states, and existing `ts4_diagnostics.score`. Sensitivity c=mean_agent(max_legal_alternative|Q_i(alt)-Q_i(executed)|); weight c/mean(c); weighted absolute activation averaged over agents and states, normalized by within-layer mean. Preserve existing zero-mean/undefined handling. Strict score>3 selections and full64-vector stored for both layers, alongside CPU trajectories and model. Diagnostic errors recorded without cancelling training or other nodes.

Summary: score heatmaps, distributions (Wasserstein), indexed Spearman/mean absolute changes, threshold-set counts/Jaccard and fixed top3 overlap/Jaccard. Top3 ties use ascending index. Empty threshold union is undefined. Compare all histories ending in the same task within the same seed; compare scratch tasks across S/Z/C. Aggregate only descriptive seed statistics; no cross-seed neuron index alignment or inflated replicate count.

Different task scratch scores compare tasks under common initializations. Different predecessor orders at stage3 have equal total budgets. Transfer-vs-scratch matches current-task exposure, but total training/replay/exploration differ: interpret as the combined existing continual-training protocol, not isolated weight-memory causation. On-policy data and state weights can change, so score differences include data-distribution changes. Neuron indices across separately learned models are descriptive labels, not guaranteed functional matches. No causal importance or protection benefit claims.

Current task32 greedy evaluation episodes at phase start/every25k steps; stage end all seen tasks128 each, separate CPU environments, fixed per-task evaluation seed blocks, training RNG restored, no replay writes. Stage-end wins and periodic curves provide context for whether tasks were learned. No performance gates or automatic budget increases.

## Execution and artifacts

Resource snapshot at setup:24 logical CPUs,87GiB RAM available,480GiB disk free; six existing GPU jobs use7731/12053MiB, GPU utilization82%. Run at most TWO new training processes, one Torch/OMP/MKL thread each. Existing jobs unchanged. Queue `scripts/run_history.py` starts each node through `scripts/run.py --suite history`. A failed node marks its dependents unavailable; unrelated nodes continue, no silent retries.

Protocol `configs/continual/history.json`; immutable per-node config in result folder and each run. Source/Git snapshots via standard launcher. Runs `runs/vdnh_<prefix>_seed<seed>_20261103_attempt001`; main output `results/vdn_task_history_20261103/`. Continuation checkpoints at single/pair nodes include replay and remain remote; terminal nodes save model/score inputs. Summary at queue termination via `scripts/summarize_history.py`; training completion and diagnostic completeness reported separately. Queue completion requires later review; no chat notification or recurring monitoring configured.

## Queue concurrency override

The 2026-11-03 queue was handed over to the resumable scheduler with six parallel slots. Existing launchers are adopted by PID and run ID. Dependencies and protocols continue to come from the original queue snapshot. The snapshot records the effective slot count. For a handover, stop the previous scheduler alone and verify that its child launchers match the running entries before starting the replacement.

Adjust the live limit with `.venv/bin/python scripts/run_history.py --set-parallel N`. The scheduler reloads this setting each scheduling cycle. Lowering it lets existing training finish and prevents new starts until slots are available. Raising it admits eligible pending tasks in queue order.

New History queues default to seven parallel slots.

## SZC 可接受动作集合偏好回放（ASPR）

2026-11-13 补充 seed 3535、2222，各一条完整 SZC 路径。按原路径使用镜像 5v5 Stalker → Zealot → Colossus、双层前馈 64→64 + GRU 64 的 `vdn`、累计 ε 退火、累计 2/4/6M 步阈值，整局结束后切换。名义新增 12M 步。沿用每次 32 局当前任务评估、阶段末每个已见任务 128 局及对应种子块。

沿用 [ASPR 首跑](TS3_ASPR_PROTOCOL.md) 的方法和诊断。λ=1、margin=0、每更新采 8 条历史轨迹。各阶段目标 100 胜局，用最优与次优合法动作 Q 差的中位数固定 δ。评估中留出最多 16 局固定轨迹，记录后续阶段的贪心动作越集合比例及同期胜率。各路径从头训练，采用精确同构阵容适配器，训练环境在同一进程跨阶段延续，TD replay 随任务切换重建。

原 `vdnh_SZC_seed3535_20261103_attempt001` 与 `vdnh_SZC_seed2222_20261103_attempt001` 已完成，可作同种子描述性参照。原路径跨节点保留 TD replay，阶段切换会新建 SC2 进程，并通过单一权重为 1 的阵容生成器抽取镜像队伍。这些差异影响因果归因。原路径最终 S/Z 均为 0/128 胜，C 分别为 88/128 和 90/128。

协议为 `configs/continual/history_szc_aspr.json`，统一调度计划为 `results/vdnh_szc_aspr_20261113/plan.json`。运行目录名为 `vdnh_SZC_ASPR_seed<seed>_20261113_attempt001`。
