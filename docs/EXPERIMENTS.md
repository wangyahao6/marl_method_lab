# 新实验的使用方式

## 持续学习的 ε 调度（2026-11-09）

持续学习默认使用 `epsilon_on_switch: per_task`。训练任务开始时记录当前全局 `t_env` 作为探索计数起点，随后按回合开始时的任务内训练步数计算原线性衰减。训练使用原有 `epsilon_start`、`epsilon_finish` 和 `epsilon_anneal_time`。同一规则用于 baseline 与各干预组。跨进程分支和阶段末续训在进入下一任务时重新设置起点，模型、优化器、经验池、目标网络及全局计数沿用各协议的恢复流程。

`runs/<run-id>/task_epsilon.jsonl` 按任务记录 `start`、`end`、阶段编号、任务、全局步数、探索计数及训练 ε。结束值按实际阶段终点计算，短于衰减周期的阶段可能尚未到达 `epsilon_finish`。评估使用贪心动作，评估回合不计入探索步数。

`scripts/run.py` 将缺省的 `per_task` 写入 `--dry-run` 预览和新运行的 `protocol.json`，并沿用协议显式指定的值。从旧协议派生新实验时，默认在新协议中使用 `per_task`。复现或续跑既有实验时沿用该实验的生效协议。

已提交的 TS6 单层 VDN 和 TS3 GEM 阈值大于 3 批次继续使用提交时的全局调度。对应协议 `ts6.json` 和 `ts3_vdn_A2_local_gem_gt3_20261108.json` 显式固定 `cumulative_no_restart`。后续使用相同任务表做逐任务衰减时，新建协议并删除这一兼容字段或设为 `per_task`。

以后可以直接说：“在我的marl_method_lab仓库中……”。Codex先读取本仓库README、AGENTS、方法清单和相关方案。

一次实验说明：目的、方法/比较对象、环境或任务序列、种子、每次预算；其余可明确“沿用某个方案”。只有方案设计就说“只设计，不运行”；需要执行就说“在远端运行”。缺少决定性设置时确认缺项，不猜测种子或预算。

## VDN 持续学习经验回放

TS3–TS8 和 history 的新 VDN 持续学习实验默认使用 `ft`，包括基线和受支持的
神经元保护与约束方法。VDN 基线需要均衡回放时，在协议 JSON 中显式设置
`"replay_on_switch": "er"`，或通过 `scripts/run.py --replay-on-switch er` 覆盖协议。
共享前缀和续训分支应使用同一回放模式。复现旧的共享池协议时显式指定
`--replay-on-switch retain`。
`vdn_action_set` 的偏好约束仍在每个阶段重建 TD 经验池，历史参考数据由各约束方法管理。
各入口沿用原有的模型、TD 更新、评估与任务预算。

- **VDN-FT** 在任务切换后从空池开始，积满一个批次的当前任务完整回合再更新。
- **VDN-ER** 按任务标识保留完整回合，按已存经验的任务和当前任务分配
  `buffer_size` 总容量。新任务加入时按新配额淘汰各池最旧的超额回合。
  有历史经验时，每批一半取自当前任务，一半在其他任务之间均分。
  除不尽的名额随机分配。历史任务回合过少时先取其不同回合，再从该任务重复抽样。
  仍无历史经验时用一个完整批次的当前经验。切换后等当前任务积够所需回合再更新。
  回访时，同一任务此前的经验也算当前经验。训练批次大小须为偶数，且每个任务的存储配额
  须能容纳半个训练批次。
- ER 的跨进程分支沿用带任务索引的回放快照，也支持能确定来源的单任务旧前缀。
  TS6 双层 VDN 的权重续训入口没有历史回放，ER 应走能从头训练或恢复回放的入口。

例如用 TS4 双层 VDN 基线协议生成两份启动预览。实际执行时去掉
`--dry-run`，并使用各自的唯一运行编号。

```sh
.venv/bin/python scripts/run.py --suite ts4 --algorithm vdn --method baseline \
  --protocol configs/continual/ts4_vdn_4242.json --seed 4242 \
  --run-id ts4_vdn_ft_4242 --replay-on-switch ft --dry-run
.venv/bin/python scripts/run.py --suite ts4 --algorithm vdn --method baseline \
  --protocol configs/continual/ts4_vdn_4242.json --seed 4242 \
  --run-id ts4_vdn_er_4242 --replay-on-switch er --dry-run
```

`transitions.jsonl` 的 `replay_task_episodes` 记录切换完成后的各任务库存。
`replay_samples.jsonl` 记录每阶段首个更新批次的任务计数。检查 FT 切换后库存为零，
首批全来自当前任务。检查 ER 在已有旧经验时两半各占一半，且历史各任务计数差至多一。
回放单元测试可独立运行，无需启动 SC2。

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_task_replay.py
```

## 静态SMAC示例（命令只是示例，未执行训练）

```sh
cd /home/wyh/marl_method_lab
.venv/bin/python scripts/run.py --suite smac --algorithm qmix --method reborn --map 3s_vs_5z --seed 1 --steps 2050000 --run-id example_qmix_reborn_seed1 --dry-run
```

去掉`--dry-run`才启动。方法可选baseline/reborn/redo/reset，组合支持见方法清单。通过`--set test_interval=10000`等覆盖超参数；原始name保留作方法分支选择，不作为运行编号。完整配置落到各run目录。当前保留原训练循环整回合结束逻辑，steps是预算阈值，最终可能整回合超出；不要与连续任务精确截断混为一谈。原默认模型保存周期较长，若实验需要阶段模型，应在方案明确设置save_model_interval。

## 连续任务示例（参考配置，不是已授权队列）

```sh
.venv/bin/python scripts/run.py --suite continual --algorithm qmix --method softprotect --scenario task_switch --seed 1111 --run-id example_switch_softprotect_1111 --protocol configs/continual/knife_fig6.json --dry-run
```

种子由命令显式指定，阶段和训练设置来自参考方案及驱动；需要变化时新建有名称的方案文件，不反写旧运行配置。导入参考方案清除了旧队列顺序、重试编号和计划总运行数，不会恢复旧实验。连续任务当前保留原驱动的batch/replay/lr等设置，修改训练超参数时需同步驱动读取方式，不声称可用任意JSON键覆盖。

## 记录与结果

每次run-id唯一，目录已存在时不覆盖。启动保留config/protocol、launch.json、源码SHA-256与source.tar.gz、console.log；进程终态由launcher_status.json记录，exited只代表进程退出。连续任务另有原训练器status.json与逐次评价/检测记录。

独立评价：`.venv/bin/python scripts/evaluate.py --run /home/wyh/marl_method_lab/runs/<run-id> --episodes 32`。静态SMAC用`--step`选择最接近的已有模型步数（0选择最大），保留上游选择逻辑；连续任务评价最终模型。评价结果另存该运行evaluation目录。

多方法/多种子由后续具体实验的方案定义，必要时编写简单队列；某run失败不连带取消剩余已授权任务。不默认新增验证训练、种子或中止条件。汇总图表和REPORT按具体研究问题生成，不预设所有实验都用同一CI或图。

## 可以这样下达任务

- “在我的仓库中，设计QMIX、QMIX+ReBorn、QMIX+本方法在……上的比较方案，先不运行。”
- “在我的仓库中，按configs/……方案运行远端实验，种子……，每次……步；原始产物留远端，结束后整理独立评价、胜率曲线和报告。”
- “在我的仓库中修改本方法的……，其余设置沿用……；这次只改代码，不启动训练。”
- “检查我的仓库实验……的进度，只读，不重启。”

## Authorized location ablation (2026-10-30)

User request: soft-protection QMIX, 5gen_protoss task-switch, seed 1111, locations A / W / V / A+W+V. Each location starts from scratch for the inherited full 4M-step sequence (Stalker/Zealot alternating every 500k). Four runs, total 16M steps. Preserve replay, optimizer, target state and cumulative exploration across switches. Same 32-episode tests every 10k steps and at phase boundaries; same final independent 32-episode Zealot evaluation. Logical 5v5 continues to use the existing SMACv2 10gen_protoss map registration with capability n_units=5, as in the preceding task-switch experiment.

Named protocols: `configs/continual/location_{A,W,V,AWV}_switch_1111.json`. Only location-related fields differ across them. Per-layer selection count is determined by the existing threshold; this is a location ablation, not an equal protected-parameter-budget experiment. No additional baseline or seed is scheduled.

Queue: `.venv/bin/python scripts/run_location_ablation.py`, two concurrent runs, order A, W, V, AWV. It invokes the public scripts/run.py, preserving each run's effective protocol, config, source snapshot and local Git identity. Failed runs are recorded without cancelling the remaining requested locations, and never silently retried. The old stopped experiment queue remains stopped.

Artifacts: `runs/qmix_switch_softprotect_<location>_seed1111_20261030_attempt002/`; summary/status in `results/location_ablation_seed1111/`. After training, the queue runs independent final evaluation, then writes explicit learning curves, phase-end comparisons, per-run state, final evaluation tables and REPORT.md. Raw checkpoints and logs remain remote. Four groups contain only one training seed each; no across-training-seed confidence interval or significance claim is appropriate. No desktop notification/automatic chat wakeup is configured.

Startup repair: initial attempts stopped at 0 training steps because the repository's static-SMAC runtime lacked SMACv2's 32x32_flat.SC2Map. Initial runs/logs are retained; queue startup records are in results/location_ablation_seed1111/startup_attempt001. Continual train/evaluate now explicitly use the read-only runtime/SMACv2StarCraftII link to the preceding task-switch runtime; static SMAC retains its original runtime. Corrected runs are named attempt002. No extra training, shortened budgets, or model resumption was introduced.

## Authorized ABA comparison (2026-10-31)

User-requested seed 1111: QMIX baseline versus QMIX + W/V soft protection. Each starts independently from scratch: Stalker mirror 5v5 for 1M steps, Zealot mirror 5v5 for 500k, then Stalker for 500k (2M each, 4M total). Protocol: `configs/continual/aba_WV_switch_1111.json`. Inherited map registration is `10gen_protoss`, with capability n_units=n_enemies=5, representing logical 5gen_protoss.

Screen W+V once at the end of A1, using the existing latest-32-episode statistic and strict >3 threshold. Only B applies 0.1 to selected post-RMSprop parameter increments. A2 immediately clears both protected sets and outstanding release ramps, so all update scales are 1 from the first update. Method active_phases=[1] and detection_phases=[0] are zero-based. Reference recovery fields remain for compatibility but no recovery ramp applies in this protocol. No B screening, Agent protection, optimizer reset, replay reset, target reset or epsilon restart. Existing protocols without these phase options keep their previous behavior.

Run: `.venv/bin/python scripts/run_aba.py` (two concurrent independent runs). Unique run IDs: `qmix_ABA_baseline_seed1111_20261031`, `qmix_ABA_softprotect_seed1111_20261031`. It uses scripts/run.py with saved config/protocol, source archive/hashes and local Git revision. Failed runs are recorded without cancelling the other method or automatically retrying.

Primary return-A observation: evaluations.jsonl phase=2, t_env=1500000, evaluation=phase_start, 32 greedy episodes before any A2 training; followed by the inherited roughly 10k-step evaluation cadence and exact endpoint. Queue performs independent final 32-episode A evaluation and writes `results/aba_WV_seed1111_20261031/{REPORT.md,learning_and_recovery.png,evaluations.csv,return_A_recovery.csv,independent_final.csv,status.json}`. One training seed per method; no across-seed significance claim. No desktop notification or chat wakeup configured.

Direct verification before launch: Python syntax; dry-run effective protocol; CPU real RMSprop update scaling 0.1 in B, exactly 1 in A2; immediate removal of both selected sets and a synthetic outstanding ramp. No extra SC2 training or changed seeds/budgets.

## Authorized extended-return ABA, seeds 3535 / 2222 (2026-10-31)

New runs from scratch: QMIX baseline and W+V soft protection for each seed 3535, 2222 (four runs). Stalker 1M → Zealot 500k → Stalker 1M, 2.5M steps/run, 10M total. All method, environment, replay/optimizer/target retention, cumulative exploration and evaluation settings inherit the seed1111 ABA experiment. Detection only at A1 end; protection only in B; immediate all-one scaling on return A.

Protocol: `configs/continual/aba_WV_A2_1M_3535_2222.json`. Queue: `.venv/bin/python scripts/run_aba_two_seeds.py`; maximum two concurrent runs, paired methods for 3535 then 2222. Single-run failures do not cancel the remaining requested runs; no automatic retries. Unique IDs: `qmix_ABA_A2_1M_{baseline,softprotect}_seed{3535,2222}_20261031`.

Results/status: `results/aba_WV_A2_1M_seeds3535_2222_20261031/`. After training, independent final A evaluation (32 episodes/run) and per-seed curves/CSV/reports are generated under seed3535/ and seed2222/. The seed1111 2M schedule remains separate. Reusable run_aba.summarize now reads the seed and final phase budget from actual run config, rather than hardcoding the previous schedule. Direct checks: syntax and dry-run protocol; training/method code unchanged. No extra verification training or chat notification.

## Authorized TS3 seed3535 (2026-11-02)

Five stages, 1M each; five comparison paths (common baseline, A1_linear/A1_union/A2_linear/A2_union), shared A prefix. Up to five branches concurrently after prefix. Queue `.venv/bin/python scripts/run_ts3_agent.py` uses `scripts/run.py --suite ts3`; unique attempt001 IDs. See [TS3 protocol](TS3_AGENT_PROTOCOL.md) for budget, evaluation, parameter mappings and state-restoration limits. Direct checks used real RMSprop offline and five exact-team resets without policy training; shapes obs92/state130/actions11 for all tasks. No performance gate or added training budget.

## Authorized TS4 VDN seed4242 (2026-11-02)

VDN baseline, A1_linear, A2_linear; three full TS4 paths, 1M/500k/500k/1M each, single new training process at a time. User delegated seed choice and specified separate A1/A2 arms and old-task win-rate retests. See [TS4 protocol](TS4_VDN_PROTOCOL.md). Queue: .venv/bin/python scripts/run_ts4_vdn.py through scripts/run.py --suite ts4. Results: results/ts4_vdn_agent_4242_20261102/. No chat wakeup configured. Offline arithmetic/RMSprop and pure-team environment reset checks passed without validation training.


2026-11-02 21:50 +08:00 scheduling update: user authorized parallel execution. Resource check: GPU 1402/12053MiB, available RAM116GiB, CPU load0.93. Max concurrency changed from1 to3. Existing baseline training/launcher preserved and adopted by replacement supervisor; A1/A2 each starts its own original full3M path. Seed4242, threshold>3, training, screening, protection and evaluation unchanged. Original run snapshots remain immutable; results/ts4_vdn_agent_4242_20261102/parallel_transition_20261102.json records handover.

## TS3 VDN A2 top-3 (2026-11-02)

Authorized seeds1234/4567, each stage2M, baseline and history-union protection only. [Protocol](TS3_VDN_TOP3_PROTOCOL.md), queue `scripts/run_ts3_vdn_top3.py`. Shared unprotected A per seed, nominal36M actual steps (40M path-equivalent), max4 new processes. Detection uses exact VDN sensitivity and overall A2 top3, no threshold. Direct offline check: explicit summed-Q counterfactual equality, real VDN learner, exact top3, unchanged model/optimizer/hidden/RNG during screening and union/slice mappings. No extra environment training.

## TS3 VDN matched screening-rule completion (2026-11-04)

Authorized three missing branches only:3535 overall top3,1234/4567 overall strict score>3; A2 historical union, same original2M prefix per seed,24M nominal new steps. Queue `scripts/run_ts3_vdn_selection_matched.py`, serial alongside existing workloads. [Protocol](TS3_VDN_SELECTION_MATCHED_PROTOCOL.md); results `results/ts3_vdn_selection_matched_20261104/`. Existing baselines and opposite-rule paths reused. Initial A selections are recomputed from saved overall scores, with immutable original prefixes and explicit provenance. No validation training or extra budgets.


## TS3 VDN A2 local GEM (2026-11-04)

Authorized existing-baseline seeds1234/4567/3535, only high/middle/low method branches,21A2units each, local actual RMSprop increment projection. Reuse original2M A prefixes, B-E8M each, nine branches72M new nominal steps. Latest-boundary groups, no neuronal history union or 0.1 scaling. Three baseline FWT jobs are evaluation-only. Resource-aware max5new workers; no interference with existing queues. See [protocol](TS3_VDN_A2_LOCAL_GEM_PROTOCOL.md). Queue `scripts/run_ts3_vdn_local_gem.py`, results `results/ts3_vdn_A2_local_gem_20261104/`. Launch/queuing is not completion.
