# TS5-1M TKPR-CPR-Anchor 扩展实验

2026-11-26 授权启动。方法 `tkpr_cpr_anchor`，入口 `vdn_action_set`，网络 `vdn_1ff`，种子 3535、2222。

复用 `ts5_tkpr_cpr_1m_20261124` 对应种子的完整 A 阶段共同前缀，不重新训练 A。从原 Online/Target MAC、优化器、Replay、计数、探索状态、RNG 及同一 A 成功池恢复，独立训练 B–E 各 1M 步（名义新增总计 8M 步）。各分支独立保存产物，不写入共同前缀或原四组运行。

配置：[tkpr_cpr_anchor.json](../configs/continual/action_preference/ts5/cpr_shared_1m/tkpr_cpr_anchor.json)。除方法标识与参考选择规则外，与原共同前缀 CPR 配置相同。K1 损失、跨历史任务合计 8 条、成功池 100、任务门控及所有训练评价设置沿用原实验。每 25k 步当前任务评价 32 局，每阶段末已见任务各 128 局，梯度监控间隔 100。

完整调度计划位于主仓库 `results/ts5_tkpr_cpr_anchor_1m_20261126/plan.json`，包含两项训练及各自依赖训练成功的最终 128 局独立进程复核。复核使用原评价种子，不能称为独立测试集。不重跑原 ER、K1、CPR、Refresh。

运行代码固定在独立工作区 `/home/wyh/marl_method_lab_cpr_anchor`；配置、源码快照、启动版本及共同前缀 SHA 由现有启动与恢复记录保留。训练开始不代表实验完成；完成后再比较逐种子结果、曲线及参考更新诊断。
