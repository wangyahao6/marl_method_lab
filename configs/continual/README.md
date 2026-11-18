# 持续学习实验配置

项目的标准探索策略是 `epsilon_on_switch: per_task`。省略该字段时，启动器、协议校验和训练驱动都采用此值。每次任务开始时从 `epsilon_start` 重新退火，并按该任务的训练步数在原有 `epsilon_anneal_time` 内降至 `epsilon_finish`。显式 `cumulative_no_restart` 供历史复现。通过 `scripts/run.py --dry-run` 核对生效协议。

| SZC 用途 | 配置 | ε 调度 |
| --- | --- | --- |
| 新实验的通用 TKPR 示例，未发现启动记录 | `szc_vdn64_tkpr.json` | 逐任务 |
| 新实验的共享 S 前缀和 K1、T1PR、K1-Weighted、K1-High-Regret、VDN-FT 分支 | `szc_vdn64_tkpr_k1_shared.json` | 逐任务 |
| 新实验可选的独立 T1PR-Max 和 T1PR-Unfiltered 示例，未发现对应协议 ID 的运行 | `szc_vdn64_t1pr_max.json`、`szc_vdn64_t1pr_unfiltered.json` | 逐任务 |
| 已登记的 SZC-FT、FRPR 消融、K 值消融、共享 S 前缀和 CHI 协议，仍符合新实验的探索标准 | `szc_vdn64_ft_preferences.json`、`szc_vdn64_frpr*.json`、`szc_vdn64_tkpr_k{1,2,3,5,10}.json`、`szc_vdn64_tkpr_k1_shared.json`、`szc_vdn64_{ft,er}_a2_chi.json` | 逐任务 |
| 历史首版 T1PR、SZC ASPR | `szc_vdn64_t1pr.json`、`history_szc_aspr.json` | 累计 |

复现或续训时，以原运行的 `protocol.json`、`config.json` 和 `launch.json` 为准。早期快照缺少该字段时，按启动版本确认原行为。从历史配置派生新实验时，使用独立协议 ID 并使用 `per_task`。共享前缀与分支使用同一协议版本和 ε 调度。

配置文件供选用。实际提交的实验以调度计划与运行记录为准。 [实验用法](../../docs/EXPERIMENTS.md)
