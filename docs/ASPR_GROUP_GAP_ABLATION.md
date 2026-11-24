# ASPR-3G 的 min_gap 2×2 消融

只新增 seed 2222、3535 的 ASPR-3G gap0 与 ASPR-3G-Adjacent gap0.5，共四条 Z/C 分支。
已有 ASPR-3G gap0.5、Adjacent gap0、两种 gap 的 TKPR-K1 及所有 S 前缀只读引用，不重复训练。

每种方法从原配置复制一份独立配置，只改变本方法的 min_gap。RMSprop、FT、逐任务 ε、共享 S、Z/C 各2M累计阶段协议、损失和评价口径保持原样；不改方法实现。

生成完整队列：

```sh
.venv/bin/python scripts/plan_aspr_group_gap_ablation.py <batch> > results/<batch>/plan.json
.venv/bin/python scheduler/service.py submit results/<batch>/plan.json
```

计划包含四条新训练、四条终局独立评价、八条只读诊断和一条终态汇总。原运行与原诊断不写回；本批次诊断另存 `results/<batch>/diagnostics/<run-id>/`。
三类跨组违反率按阶段末模型在完整历史池的统一离线口径计算，另报告间隔筛选后的违反率。
实际有效对数及原始/标准化 Q-gap 分布从逐更新 `source_phase/episode_ids` 重建，含重复抽样；用原 `reference_order/full_rank_pairs` 保持筛选边界完全一致，并与日志 valid_pairs 总数核对。
旧 3G 日志只有总违反率，不能恢复逐更新的三类违反率，报告不把离线数值称为训练时逐类违反率。

所有核验只用于结果审查；辅助计数差异不停止训练或取消其余分支。两种子按共同前缀配对，分别比较方法、gap与二者交互，不把同一前缀的评价当独立训练重复。
