# 持续学习协议

| 目录 | 用途 |
| --- | --- |
| [action_preference/szc](action_preference/szc) | SZC 的 FRPR、T1PR、TKPR、ASPR、ASPR-Max 及对照。共享前缀使用 `szc_vdn64_tkpr_k1_shared.json`。 |
| [action_preference/pair_structure](action_preference/pair_structure) | SZC-1M 动作对结构消融。 |
| [action_preference/pair_ratio](action_preference/pair_ratio) | SZC-1M Top-1 约束比例消融。 |
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

两组 seed 3535 动作对消融的设置、诊断及启动命令见 [SZC-1M 动作对消融](../../docs/SZC_PAIR_ABLATIONS.md)。
