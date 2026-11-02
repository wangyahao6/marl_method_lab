# MARL Method Lab

用户自己的 MARL 方法研究代码库。2026-10-30 建立，仅保存于 `rl-navigation-ubuntu:/home/wyh/marl_method_lab`，未上传、未配置 Git 网络远端。代码以 ReBorn 固定提交为基础；本仓库建立不启动任何训练、不恢复已终止队列。

## 入口

- [方法清单](docs/METHODS.md)：算法、神经元干预、原始配置与实际支持范围。
- [实验用法](docs/EXPERIMENTS.md)：以后如何描述实验、运行与评价入口。
- [来源与修改](docs/PROVENANCE.md)：上游、导入的方法、许可证差异、运行时依赖。

## 目录职责

| 路径 | 内容 |
| --- | --- |
| `src/` | 公共训练底座、Agent/Mixer、各 learner、上游算法/环境配置 |
| `drivers/continual/` | 现有 SMACv2 连续任务驱动、状态加权软保护及检测记录 |
| `configs/continual/` | 可显式选用的连续任务参考方案；不是待执行队列 |
| `scripts/run.py` | 统一单次训练入口，`--dry-run`只显示生效配置，不建运行目录 |
| `scripts/evaluate.py` | 对已有运行独立评价，结果另存运行目录 |
| `runs/<run-id>/` | 生效配置、启动身份、源码快照、日志、模型和评价；不入Git |
| `docs/` | 方法与管理说明、来源和运行时版本 |
| `.venv`、`runtime/` | 本机运行时链接，不属于可公开源码 |

新实验共用本仓库代码，用配置与运行编号区分。旧复现工程、已停止实验及产物保留原路径，不搬迁或认作新仓库结果。当前状态是代码整理和静态核对完成；没有新仓库真实训练或性能结论。

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
