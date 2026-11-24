# TS5-1M · TKPR-CPR 共同前缀四方法对照

状态：2026-11-24 已提交完整19任务调度计划；提交时两个A前缀因全局6槽已满而等待，其余任务按依赖自动执行。尚无本批训练结果。

用户确认四组：VDN-ER、TKPR-K1、TKPR-CPR、TKPR-Refresh；训练种子3535、2222。每种子独立训练A前缀1M，再分出四个互相独立的B–E分支，每阶段1M，名义新增34M。使用 `vdn_action_set`、`vdn_1ff`、逐任务探索与原TS5-1M评价。

[执行协议与数据审计](../../../TS5_CPR_1M_PROTOCOL.md)记录完整前缀状态、Replay恢复及FT/ER切换语义、初始A成功池一致性、诊断字段与评价种子。执行工作树为 `/home/wyh/marl_method_lab_ts5_cpr_1m_20261124`，固定代码提交 `8ba02fe065e89cce0044abee78999b4ad3b0a971`。全部启动参数保存在运行快照与 `results/ts5_tkpr_cpr_1m_20261124/plan.json` 中。

终态汇总任务使用 `scripts/summarize_ts5_cpr_1m.py`，生成本目录的运行/评价/阶段指标、学习AUC、配对差、偏好/CPR/梯度诊断、记忆和计算开销、独立进程最终复核，以及来源与执行摘要和三类图表。未完成前不填入预期指标；最终复核复用focus评价种子，属于独立进程复核，不是独立测试集。原始模型、Replay、Memory和逐更新日志保留实验主机。
