# 持续学习协议

| 目录 | 用途 |
| --- | --- |
| [action_preference/szc](action_preference/szc) | SZC 的 FRPR、T1PR、TKPR、`tkpr_refresh`、ASPR、ASPR-Max 及对照。共享前缀使用 `szc_vdn64_tkpr_k1_shared.json`。 |
| [action_preference/ts5](action_preference/ts5) | TS5 seed 3535 的 TKPR-K1、TKPR-Refresh、VDN-FT、VDN-ER 四组独立协议；五阶段各 2M，使用同一单前馈层 VDN。 |
| [action_preference/pair_structure](action_preference/pair_structure) | SZC-1M 动作对结构消融。 |
| [action_preference/pair_ratio](action_preference/pair_ratio) | SZC-1M Top-1 约束比例消融。 |
| [action_preference/pair_distance](action_preference/pair_distance) | SZC-1M Top-1 加 Near、Mid、Far、随机和 MaxGap 额外对结构比较。 |
| [action_preference/pair_gap_far_mid](action_preference/pair_gap_far_mid) | SZC-1M Far/Mid 额外动作对的 Q-gap 0／0.5 筛选比较。 |
| [action_preference/pair_gap_matched](action_preference/pair_gap_matched) | SZC-1M K1 与归一化 Q-gap 最优匹配的非 Top-1 对比较。 |
| [action_preference/examples](action_preference/examples) | 动作集合协议示例。 |
| [action_preference/historical](action_preference/historical) | 历史首跑和回访协议，包括累计 ε 的早期 SZC T1PR 与 ASPR-Max。 |
| [neuron_protection/softprotect](neuron_protection/softprotect) | A/W/V、异质 Agent、TS3/TS4 软保护与历史对照。Windows 128 单元软保护协议仍可定位。 |
| [neuron_protection/chi](neuron_protection/chi) | SZC、TS3–TS8 的 CHI 与 CHI + TRHR。 |
| [neuron_protection/trhr](neuron_protection/trhr) | TS3/TS4 的 TRHR 析因协议。 |
| [gradient_constraint/gem](gradient_constraint/gem) | TS3 的 A2 局部 GEM 两种协议。 |
| [task_transfer](task_transfer) | 通用 continual 协议、TS5–TS7 和任务历史训练。 |
| [analysis](analysis) | 使用冻结检查点的网络相似性分析。 |

新实验按研究方向选择协议，向 `scripts/run.py --protocol` 传入仓库相对的新路径。用 `--dry-run` 核对生效值与实际驱动。默认连续学习协议为 `task_transfer/knife_fig6.json`。项目标准探索调度为 `epsilon_on_switch: per_task`，历史协议可显式使用 `cumulative_no_restart`。共享前缀与分支使用同一协议版本。

[path_aliases.json](path_aliases.json) 将老的平铺路径映射到新位置。统一启动入口会解析旧路径，历史调度命令可沿用原值。这次迁移保留已有 JSON 的内容与协议 ID。复现请以对应运行的 `protocol.json`、`config.json`、`launch.json` 和源码版本为依据。历史快照中的旧路径原样保留。已退役参考继续保存在[原归档目录](../archive/continual/README.md)。本次归入上述研究目录，未另建 `legacy/` 迁移已知历史实验协议。

[方法说明](../../docs/METHODS.md) · [实验用法](../../docs/EXPERIMENTS.md)

## SZC-1M

2026-11-19 新增每任务 1M 的 S→Z→C 版本，累计阈值为 1M/2M/3M，完整路径名义 3M 训练环境步，在阈值后的完整回合结束切换。

- 偏好回放与基线入口使用 `action_preference/szc/szc_vdn64_ft_preferences_1m.json`，从同目录 `szc_vdn64_ft_preferences.json` 派生。
- 双层 VDN A2 CHI 入口使用 `neuron_protection/chi/szc_vdn64_ft_a2_chi_1m.json`，从同目录 `szc_vdn64_ft_a2_chi.json` 派生。

两份协议沿用各自母版的模型、方法参数、逐任务探索与评价设置。方法专用协议如 TKPR 的 K 设置应以所选方法协议为依据，再派生相同阶段预算。此次配置创建仅设置预算版本，具体训练方法与种子由运行指令选择。

两组 seed 3535 动作对消融的设置、诊断及启动命令见 [SZC-1M 动作对消融](../../docs/SZC_PAIR_ABLATIONS.md)。

双种子 Top-1 加额外对结构与 Far/Mid Q-gap 消融见 [SZC-1M 动作对距离实验](../../docs/SZC_PAIR_DISTANCE.md)。

复用 seed 3535 旧参考线的 Gap-Matched-NonTop1 单分支见 [SZC-1M 非 Top-1 Q-gap 匹配实验](../../docs/SZC_PAIR_GAP_MATCHED.md)。

双层 VDN64 的 TKPR-K1／R2／R3 单排名消融使用 `action_preference/szc/szc_vdn64_tkpr_rank_position.json`，沿用 2M × 3 的共同前缀协议。
