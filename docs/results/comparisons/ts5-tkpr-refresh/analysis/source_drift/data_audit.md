# 数据审计（先于数值实验）

审计对象：TS5 seed3535 TKPR-K1 / TKPR-Refresh，训练版本 9db0ed1951ded62c00eb1ff27a80c71ae5465dc5。

- `raw_logs/` 在当前公开树不存在；原始 updates/experiment_metrics JSONL 在运行目录可读，压缩归档路径见上级 input_manifest.json，不能把公开路径缺失解释为日志整体缺失。
- 两方法均有 A–E 五个阶段末 agent checkpoint；未发现 D/E 25%、50%、75% 模型。
- K1 原池每阶段独立保存，源码确认引用当阶段模型。Refresh 每任务仅一个原地覆盖的 success.th，现存是 E 末刷新后的池；未发现刷新前后版本快照。池创建 phase 与 reference_phase 不同。
- Refresh 每条轨迹明确携带 reference_phase/reference_checkpoint/old_q_values；完整 obs、state、actions、actions_onehot、avail_actions、reward、filled、terminated 与 bootstrap 时间轴可读。字段和时间维度检查已执行。
- 来源只读取 Refresh 逐轨迹标记；K1 由静态采集代码及池级 checkpoint 确认，未按索引推断。每来源抽查一个完整序列，重新展开其来源 checkpoint，Q 误差见 stored_q_checks.csv。
- 刷新丢弃轨迹不可恢复，episode_ids 不具备跨刷新稳定身份。**D/E 实际训练池及自然来源分布无法重建。**日志只提供采样索引，无法恢复被覆盖的轨迹/Q。
- 未发现当前任务 TD ReplayBuffer、同次更新 minibatch 或含参数位置的训练快照。成功轨迹不能替代当前 TD 数据。实验3跳过，source_td_gradient.csv 明确记录不可用原因。

可执行范围：固定留存轨迹上的不同 checkpoint 排序；C/D/E 阶段末模型位置的留存旧来源梯度；原始日志 TD Loss。梯度样本仅允许 reference_phase < 当前阶段，且排除当前及未来任务；来自最终池的留存子集有幸存偏差，**不是历史真实池**。natural_survivors 按该子集实际轨迹比例每任务采8条（K1跨任务总8条）；balanced_decisions 每来源采8条完整序列，再随机屏蔽多余有效决策以等量，所有 RNN 输入序列完整。后者是条件化对照，不是原训练采样。每设置10次。

源码依据：memory/__init__.py 的 sample/sample_per_task/refresh；action_set_train.py 的 reference_record/collect_memory 与阶段末刷新；ranking_preference.py 的 topk_preference/_full_rank_loss；action_set_preference.py 的 decision_masks/unroll/reduce_task_losses；q_learner.py 的 agent_gradient_metrics。当前 TKPR 函数与训练版本相同（其他方法新增不影响其路径）。不启动环境，不调用 optimizer.step，不修改原始产物。

补充核验：28份输入SHA-256已与Ubuntu原始文件逐一对照一致，500条Refresh轨迹的reference_checkpoint路径与reference_phase相符；13个损失/掩码/展开/梯度函数AST与训练版本相同。复查脚本为verify_inputs.py。
