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

## 其他任务协议

| 用途 | 主要入口 |
| --- | --- |
| 当前方法和任务入口 | TS3、TS4、TS5、TS6、TS7、TS8 由方法文档和实验计划指定协议。复用已登记协议时沿用其参数，从历史协议派生新实验时使用 `per_task`。 [方法与协议](../../docs/METHODS.md) |
| 历史实验及其现存启动脚本 | `aba_WV*.json`、`hetero_agent_softprotect_ABA.json`、`location_*_switch_1111.json`、`knife_fig6.json`、`history.json`、`network_similarity.json` 供历史对照、工具或脚本复跑。 [实验用法](../../docs/EXPERIMENTS.md) |
| 指定历史平台或方法 | `ts4_vdn_1ff_128_softprotect_windows.json` 已用于 Windows TS4 两条运行。各 TS 的 GEM、TRHR、CHI、ASPR 研究配置可用来定位原设定。 |

已退出常用入口且查无现行引用或运行的旧参考见[归档索引](../archive/continual/README.md)。归档依据是协议的使用记录与引用。显式 `per_task` 的在用协议继续保留原路径。
