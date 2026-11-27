# MARL Method Lab

MARL 方法研究代码库，基于 ReBorn 固定提交建立。开发主版本位于 `rl-navigation-ubuntu:/home/wyh/marl_method_lab`。方法代码完成验证和提交后，可按 AGENTS.md 同步至公开的 [GitHub 仓库](https://github.com/wangyahao6/marl_method_lab)。

## 入口

- [实验结果](docs/results/README.md) · 按实验问题浏览报告、逐种子数据与实际训练版本。
- [环境与任务索引](docs/environments/README.md) · 查询平台、地图／任务序列、方法、种子、共享前缀及结果路径；覆盖矩阵从真实记录生成。

- [方法清单](docs/METHODS.md)：算法、神经元干预、原始配置与实际支持范围。
- [MEAL 设计参考](docs/MEAL_DESIGN_REFERENCE.md)：神经元效用、历史重要性与梯度约束的参考材料；未接入训练，对照实验待自己的方法确定并取得初步结果后再设计。
- [持续学习驱动架构](docs/CONTINUAL_ARCHITECTURE.md)
- [实验用法](docs/EXPERIMENTS.md)：以后如何描述实验、运行与评价入口。
- 新持续学习实验统一使用 `epsilon_on_switch: per_task`，每次任务切换从 `epsilon_start` 重新退火。 [配置与历史复现说明](configs/continual/README.md)
- [来源与修改](docs/PROVENANCE.md)：上游、导入的方法、许可证差异、运行时依赖。

- [TS5-1M 传递闭包覆盖率控制实验](docs/protocols/TS5_TC_COVERAGE.md)：三个新 FT 参考与九个正式分支已完成；[结果与控制核验](docs/results/parameter-ablations/ts5-vdn64-tc-coverage/README.md)。

## 目录职责

| 路径 | 内容 |
| --- | --- |
| `src/` | 公共训练底座、Agent/Mixer、各 learner、上游算法/环境配置 |
| `drivers/continual/` | 现有 SMACv2 连续任务驱动、状态加权软保护及检测记录 |
| `drivers/continual/common/` | 公共阶段操作、评估、检查点与实验指标 |
| `drivers/continual/memory/` | 独立于 TD 经验池的方法历史轨迹、参考输出和固定探测集 |
| `configs/continual/` | 可显式选用的连续任务参考方案；不是待执行队列 |
| `scripts/run.py` | 统一单次训练入口，`--dry-run`只显示生效配置，不建运行目录 |
| `scripts/evaluate.py` | 对已有运行独立评价，结果另存运行目录 |
| `runs/<run-id>/` | 生效配置、启动身份、源码快照、日志、模型和评价；不入Git |
| `docs/` | 方法与管理说明、来源和运行时版本 |
| `.venv`、`runtime/` | 本机运行时链接，不属于可公开源码 |

新实验共用本仓库代码，用配置与运行编号区分。旧复现工程、已停止实验及产物保留原路径，不搬迁或认作新仓库结果。已核对的公开实验见[结果索引](docs/results/README.md)。

## 2026-10-30 location ablation

A/W/V soft protection is configurable; see docs/METHODS.md and docs/EXPERIMENTS.md. The explicitly authorized seed-1111 task-switch queue has four locations A, W, V, A+W+V, each 4M steps. Its live state and generated report are in `results/location_ablation_seed1111/`; this is separate from all previously stopped queues. Launch is not a completed result.

## 2026-10-31 ABA comparison

Authorized seed-1111 QMIX baseline versus W+V soft protection: Stalker 1M → Zealot 500k → Stalker 500k, 5v5. Protection applies in B only, with immediate full release on return A. Protocol and execution details: docs/EXPERIMENTS.md. Queue: scripts/run_aba.py; result directory: results/aba_WV_seed1111_20261031/. Launch is not completion evidence.

2026-10-31 extended ABA authorized: Stalker 1M → Zealot 500k → Stalker 1M, seeds 3535 and 2222, QMIX baseline and W+V soft protection (four new runs). Entry: scripts/run_aba_two_seeds.py; results: results/aba_WV_A2_1M_seeds3535_2222_20261031/. See docs/EXPERIMENTS.md.

## 2026-11-02 TS3 single-layer protection

Authorized seed3535 five-task retention experiment: baseline, A1/A2 each with latest-selection linear recovery or permanent historical union. Shared unprotected A prefix 1M; five 4M branches, nominal actual 21M steps. Protocol, evaluation and limits: [TS3 protocol](docs/TS3_AGENT_PROTOCOL.md). Queue `scripts/run_ts3_agent.py`, live result directory `results/ts3_agent_3535_20261102/`. Start/completion are distinct.

## 2026-11-02 TS4 VDN

Authorized VDN baseline, A1 and A2 separate soft-protection paths, seed4242. One training process at a time; 3M per path. Both-layer longitudinal detection and old-task win-rate retention. See [protocol](docs/TS4_VDN_PROTOCOL.md), queue scripts/run_ts4_vdn.py, results results/ts4_vdn_agent_4242_20261102/.

## 2026-11-02 TS3 VDN top-3

New authorized run: seeds1234/4567, five2M stages, VDN baseline versus A2 top3 historical union. See [protocol](docs/TS3_VDN_TOP3_PROTOCOL.md). Results `results/ts3_vdn_A2_top3_1234_4567_20261102/`; max4 new processes, other running workloads unchanged.

## 2026-11-03 VDN task/history analysis

Authorized three seeds3535/2222/1111, mirror S/Z/C,2M per stage, all six permutations. Shared-prefix tree90M actual nominal steps; max2 new workers. See [protocol](docs/HISTORY_PROTOCOL.md), queue scripts/run_history.py, results results/vdn_task_history_20261103/. No intervention; A1/A2 endpoint scores. Launch is not completion.

## 2026-11-04 independent network similarity

Offline CPU comparison of completed VDN S/Z/C endpoints, scratch/one-prior/two-prior/prior-only. See [protocol](docs/NETWORK_SIMILARITY_PROTOCOL.md). No new training. Results runs/vdn_network_similarity_20261104_attempt001/.


## TS3 VDN A2 local GEM (2026-11-04)

Authorized existing-baseline seeds1234/4567/3535, only high/middle/low method branches,21A2units each, local actual RMSprop increment projection. Reuse original2M A prefixes, B-E8M each, nine branches72M new nominal steps. Latest-boundary groups, no neuronal history union or 0.1 scaling. Three baseline FWT jobs are evaluation-only. Resource-aware max5new workers; no interference with existing queues. See [protocol](docs/TS3_VDN_A2_LOCAL_GEM_PROTOCOL.md). Queue `scripts/run_ts3_vdn_local_gem.py`, results `results/ts3_vdn_A2_local_gem_20261104/`. Launch/queuing is not completion.

## Unified dispatch (2026-11-05)

Server-wide admission now uses the [unified scheduler](docs/SCHEDULER.md), default six active tasks. Current Local GEM and Selection Matched plans are imported in full. All ordinary run.py launches share this admission limit.
