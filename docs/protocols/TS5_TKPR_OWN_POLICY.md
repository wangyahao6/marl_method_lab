# TS5-1M · Own-policy PerTask8 / WinRefresh

用户于2026-11-27要求终止并删除原固定FT池PerTask8实验，改为各任务阶段末自身策略采集成功轨迹，并增加胜率提高才全量换池的方法。原批次两训练及后续评价/报告已停止，旧产物已删除；简要删除记录在主仓库 `results/stopped_per_task8_cleanup_20261127.json`。原生K1历史结果、原共享前缀和代码版本保留。

## 两组设置

- `tkpr_k1_per_task8`：A池只复用A前缀自身策略采集的轨迹；B/C/D/E各阶段完成评价后，由该分支当时策略采集最多100条成功轨迹，保存对应rollout old-Q，以后不更新旧池。
- `tkpr_win_refresh`：首次见任务同样初始化自身池。每阶段末复用现有已见任务评价row。对每任务，当前胜率严格大于当前池采集策略在其采集阶段的胜率，才用当前策略重新采集最多100条成功轨迹，全部替换该任务池；不存在旧新混合。胜率相等/下降不采集。采到1–99条则最终池仅这些新轨迹；零成功则保留旧池及其胜率基准。采纳更新后更新该任务的参考胜率、采集阶段和checkpoint。该基准是当前池的采集胜率，不是另设的历史全局最佳胜率。

两组均复用K1的Top-1损失、min_gap0.5、weight1、temperature1、q_scale_floor0.1及有效任务等权归约。按用户确认，每历史任务8条，B/C/D/E总8/16/24/32条；不使用CPR状态掩码或50%替换。每池只有一个生效条目，新轨迹和自己的rollout Q一起保存。重采使用既有isolated_rollouts，不扰动偏好采样RNG。原K1/CPR/Refresh行为不变。

## 实验与来源

TS5 A→B→C→D→E，VDN单FF64，种子2222/3535，RMSprop、FT TD replay、逐任务epsilon；每阶段1M累计阈值后整回合切换；常规评价25k/32局，阶段末各已见任务128局；两新组梯度监控间隔100。复用各种子 `ts5_tc_coverage_20261123_reference_seed*` 的A末online/target/optimizer/counters/RNG和同一自身A成功池。不读取/激活该FT参考的B/C/D未来池；新分支B–E共4M，四分支新增名义16M。所有旧中断训练不作为本批前缀。

配置：`configs/continual/action_preference/ts5/own_policy/{tkpr_k1_per_task8,tkpr_win_refresh}_seed{2222,3535}.json`。新启动使用显式 `own_policy_history=true` 的兼容恢复路径；`fixed_reference_history` 未启用，后续池由各分支自己采集。原生K1 gap0.5旧结果使用固定FT历史池，故与修正后PerTask8同时存在预算和数据来源差异，只作历史参照，不能称纯采样消融。两新组的直接配对比较用于分析胜率门控全量替换。

执行工作区 `/home/wyh/marl_method_lab_own_policy_refresh`，独立分支 `experiment/tkpr-own-policy-refresh`。准备脚本 `scripts/prepare_tkpr_own_policy.py`；完整计划主仓库 `results/ts5_tkpr_own_policy_20261127/plan.json` 包含4训练+4最终独立进程128局复核+1终态报告。最终复核沿用focus种子，不是独立测试集。新run-id与中断批次不同。

`own_policy_memory.jsonl`记录初始池来源；`tkpr_win_refresh_memory_updates.jsonl`记录每阶段/任务的现胜率、原池胜率、门控、旧/新/替换/最终数量、实际参考阶段和checkpoint；`memory_collection.jsonl`记录成功/失败与采集耗时。原逐更新采样、偏好和梯度日志照常保存。终态报告输出 `docs/results/parameter-ablations/ts5-tkpr-own-policy-refresh/`，保留逐任务AUC/胜率、ACC/BWT/FWT、采样预算、梯度和来源记录。仅两个种子，不据重复采样/评价局数推断训练显著性。
