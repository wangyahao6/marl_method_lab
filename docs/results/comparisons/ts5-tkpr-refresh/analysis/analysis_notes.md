# TKPR-K1 与 TKPR-Refresh：按阶段日志分析

输入为四份 gzip JSONL，逐行 `gzip.open(..., "rb")` 读取并校验原始 SHA-256/行数；不加载整份日志，不修改输入。内存仅保留所选标量的紧凑 double 数组、梯度事件索引与累计计数；精确分位数使用 NumPy percentile 默认线性插值。原始压缩日志仅保留于实验主机；GitHub 当前树撤下日志，旧上传提交仍留在 Git 历史中。

## 字段、计算口径与记录频率

实际字段与出现次数见 field_inventory.json。两方法 experiment_metrics 中均没有 preference_loss、preference_prepare_seconds、learner_update_seconds 这三个 metric；它们在 updates 宽表中逐更新存在。四项梯度指标均取自 gradient_monitor 事件。experiment_metrics 是 metric/value/context 长表；偏好 updates 是每次 learner 更新的一行宽表。两者不是同一粒度，不能将长表各行当作独立更新或重复汇总。
训练提交为 9db0ed1951ded62c00eb1ff27a80c71ae5465dc5。代码定位：src/learners/q_learner.py 的 agent_gradient_metrics、QLearner.train；src/learners/action_set_preference.py 的 __call__、reduce_task_losses、record；drivers/continual/common/metrics.py 的 gradients、preference；drivers/continual/action_set_train.py 的 monitor_interval 与 record_preference。
preference_loss 从 updates 每次读取，是未乘 weight 的任务等权平均排序损失；只对有有效约束的历史任务平均。两方法 weight 均为 1，损失算法和标准化相同，但轨迹来源与采样预算不同，绝对损失不是统一测试集上的质量评估。A 无历史，记录 loss=0 但偏好梯度为 null，不能把 null 当零。
梯度事件每 100 个全局 learner 更新记录一次（实际间距清单见 inventory），相位内不是每 100 环境步。Agent 可训练参数梯度按 float64 展平取 L2 范数；preference_grad_norm_agent 对 weight*preference_loss 求导，td_grad_norm_agent 对 masked TD MSE 求导。均在联合反传、梯度裁剪和 optimizer.step 之前计算。ratio 为偏好范数/TD 范数；cosine 为二者点积除范数乘积，负值表示局部方向冲突。缺损失、零范数等情形可为 null，CSV 分开记录 n_records/n_valid/n_missing。
gradient CSV 同时提供 all_updates 的 preference_loss 和 gradient_monitor 子样本的五项指标，后者通过 phase/learner_update 对齐，便于统一记录频率比较。每阶段汇总加 q1–q4（按该运行实际训练阶段起止环境步的 0–25/25–50/50–75/75–100% 划分）；不是按记录数分箱。分段权重和有效采样数可不同，不将分段平均再平均当成阶段平均。

## 计时解释

preference_prepare_seconds 从 ActionSetPreference.__call__ 开始，到采样、旧轨迹组批、各历史任务前向和损失汇总完成为止的 perf_counter 差。无显式 CUDA 同步，部分 GPU 工作异步；它是 CPU 侧实测经过时间，不能解释为纯 GPU 执行耗时。另有 preference_forward_cuda_seconds，但本次请求的两个指标不可互相替代。
learner_update_seconds 从 QLearner.train 入口 begin_update 到 record 中汇总张量搬 CPU 后计时；包含 TD/偏好前向、联合反传、裁剪、优化、每100次的梯度诊断及其日志、部分同步和主机等待。排除 TD buffer 采样、runner 环境交互、阶段评价/采集刷新，以及 record 之后的偏好JSON写出和 target 更新。两计时互有包含，不能相加。
timing CSV 提供 all_updates、gradient_monitor、without_gradient_monitor 三种范围。total_recorded_seconds 是该范围内实测经过时间的和，不是完整训练墙钟，也不是纯偏好新增成本。Refresh/K1 比值是独立运行记录的描述性比较；更新次数、轨迹长度、并发负载等均可影响计时。四份日志缺少阶段末刷新采集耗时，不能据此估算完整 Refresh 总开销。

## 分阶段梯度与辅助更新

| 阶段 | 方法 | 全部记录更新 | 实际参与偏好更新 | 每更新历史样本 | 梯度采样有效n | 偏好梯度均值 | TD梯度均值 | ratio中位数 | cosine均值 | cosine<0比例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | tkpr-k1 | 35912 | 0 | 0.00 | 0 | N/A | 1.271 | N/A | N/A | N/A |
| A | tkpr-refresh | 35912 | 0 | 0.00 | 0 | N/A | 1.271 | N/A | N/A | N/A |
| B | tkpr-k1 | 32827 | 32827 | 8.00 | 328 | 0.06696 | 1.293 | 0.04162 | 0.1021 | 0.06707 |
| B | tkpr-refresh | 32827 | 32827 | 8.00 | 328 | 0.06696 | 1.293 | 0.04162 | 0.1021 | 0.06707 |
| C | tkpr-k1 | 23053 | 23053 | 8.00 | 230 | 0.1387 | 1.376 | 0.09708 | 0.05756 | 0.1652 |
| C | tkpr-refresh | 23404 | 23404 | 16.00 | 234 | 0.1325 | 1.441 | 0.08246 | 0.08046 | 0.0812 |
| D | tkpr-k1 | 29155 | 29155 | 8.00 | 292 | 0.2805 | 1.689 | 0.1798 | 0.08569 | 0.137 |
| D | tkpr-refresh | 29443 | 29443 | 24.00 | 294 | 0.2817 | 2.354 | 0.1295 | 0.1165 | 0.09524 |
| E | tkpr-k1 | 24350 | 24350 | 8.00 | 243 | 0.2998 | 1.311 | 0.2571 | 0.07873 | 0.1811 |
| E | tkpr-refresh | 24080 | 24080 | 32.00 | 241 | 0.3013 | 1.497 | 0.2007 | 0.1508 | 0.07054 |

## C、D、E 的计时与阶段内变化

| 阶段 | prepare K1均值ms | Refresh均值ms | 比值 | update K1均值ms | Refresh均值ms | 比值 | K1记录总秒 | Refresh记录总秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C | 18.429 | 20.404 | 1.107 | 80.881 | 82.123 | 1.015 | 1864.6 | 1922.0 |
| D | 36.759 | 41.851 | 1.139 | 110.627 | 120.099 | 1.086 | 3225.3 | 3536.1 |
| E | 48.528 | 60.788 | 1.253 | 140.796 | 160.659 | 1.141 | 3428.4 | 3868.7 |

### C 阶段：按实际环境步四等分

| 方法 | 区间 | loss均值（监控对齐） | 偏好梯度均值 | ratio中位数 | cosine均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| tkpr-k1 | q1 | 0.036627 | 0.22666 | 0.12609 | 0.042851 |
| tkpr-k1 | q2 | 0.013353 | 0.1413 | 0.098545 | 0.055399 |
| tkpr-k1 | q3 | 0.0071894 | 0.093181 | 0.095399 | 0.064053 |
| tkpr-k1 | q4 | 0.0051015 | 0.086514 | 0.064628 | 0.069125 |
| tkpr-refresh | q1 | 0.050008 | 0.19973 | 0.1289 | 0.065264 |
| tkpr-refresh | q2 | 0.019171 | 0.13915 | 0.083728 | 0.090588 |
| tkpr-refresh | q3 | 0.011124 | 0.11146 | 0.08224 | 0.089584 |
| tkpr-refresh | q4 | 0.007555 | 0.074582 | 0.060476 | 0.077121 |

### D 阶段：按实际环境步四等分

| 方法 | 区间 | loss均值（监控对齐） | 偏好梯度均值 | ratio中位数 | cosine均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| tkpr-k1 | q1 | 0.076332 | 0.3941 | 0.22314 | 0.038165 |
| tkpr-k1 | q2 | 0.044652 | 0.27114 | 0.15035 | 0.10491 |
| tkpr-k1 | q3 | 0.034865 | 0.22742 | 0.18504 | 0.095572 |
| tkpr-k1 | q4 | 0.029186 | 0.23304 | 0.1656 | 0.10338 |
| tkpr-refresh | q1 | 0.14437 | 0.32037 | 0.1236 | 0.084061 |
| tkpr-refresh | q2 | 0.086103 | 0.25673 | 0.13984 | 0.1273 |
| tkpr-refresh | q3 | 0.06092 | 0.26715 | 0.1399 | 0.14004 |
| tkpr-refresh | q4 | 0.053049 | 0.28237 | 0.12529 | 0.11378 |

### E 阶段：按实际环境步四等分

| 方法 | 区间 | loss均值（监控对齐） | 偏好梯度均值 | ratio中位数 | cosine均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| tkpr-k1 | q1 | 0.06386 | 0.37196 | 0.38762 | 0.026941 |
| tkpr-k1 | q2 | 0.043944 | 0.28321 | 0.23316 | 0.07802 |
| tkpr-k1 | q3 | 0.037928 | 0.26864 | 0.19622 | 0.10259 |
| tkpr-k1 | q4 | 0.034996 | 0.27522 | 0.27588 | 0.10688 |
| tkpr-refresh | q1 | 0.10869 | 0.35218 | 0.21099 | 0.11875 |
| tkpr-refresh | q2 | 0.07713 | 0.31388 | 0.2331 | 0.15699 |
| tkpr-refresh | q3 | 0.065857 | 0.23467 | 0.16527 | 0.1654 |
| tkpr-refresh | q4 | 0.059155 | 0.3045 | 0.20983 | 0.16195 |

## 对 C、D、E 的解释

C 阶段两方法偏好梯度与 loss 从 q1 到 q4 总体下降；Refresh 的偏好梯度均值略低，方向冲突比例也更低，此时 ACC 较高。但它使用已刷新的 A/B 参考轨迹和更大的样本预算，不能将较高 loss 直接解释为更差的排序保留。
D 阶段两方法偏好梯度均值几乎相同，而 Refresh 的 TD 梯度均值更大，使 ratio 中位数降低。阶段内 K1 的偏好梯度由早段下降后趋稳，Refresh 在 q2 后回升；Refresh 的 loss 从 q1 到 q4 下降，但始终高于 K1。更高的 cosine 与更少的负 cosine 并没有对应更高的阶段末 ACC，说明局部梯度一致性不足以解释保留效果。
E 阶段两方法偏好梯度均值仍接近，Refresh 的 TD 梯度更大，ratio 中位数更低；Refresh 的 cosine 平均更高、冲突比例更低。两方法 loss 均随阶段推进下降，偏好梯度在后段仍可回升。Refresh 的最终 ACC 仍低于 K1，不能由这些观测认定强约束、弱约束或刷新本身是原因。
C/D/E 的每任务采样分别为：K1 4、约2.67、2条；Refresh 一律8条。两方法来源任务的样本占比都约等权（C各50%、D各约1/3、E各25%），采样时池大小均为100。变化主要在总样本量和池内内容，日志没有呈现某个历史任务获得明显更高采样份额。
额外开销随历史任务增多而扩大，但总样本量2/3/4倍并不意味着更新耗时2/3/4倍。损失本来按任务平均、每任务有更多样本也不意味着梯度范数按倍数增长；这里报告的是实测分布与比值，不作线性开销外推。

代码证据采用实际训练版本：[梯度计算](https://github.com/wangyahao6/marl_method_lab/blob/9db0ed1951ded62c00eb1ff27a80c71ae5465dc5/src/learners/q_learner.py#L33)、[记录与计时](https://github.com/wangyahao6/marl_method_lab/blob/9db0ed1951ded62c00eb1ff27a80c71ae5465dc5/src/learners/action_set_preference.py#L183)、[每100次更新采样](https://github.com/wangyahao6/marl_method_lab/blob/9db0ed1951ded62c00eb1ff27a80c71ae5465dc5/drivers/continual/action_set_train.py#L674)。


## 样本与记忆池边界

aux CSV 的 row_type=sampling：historical_task=ALL 表示更新总预算；其他行分别表示各来源任务。recorded_updates 对 ALL 是阶段 learner 更新次数，对单来源是该任务被采样的更新次数；participating_updates 只在 ALL 有定义。sample_count_histogram 记录每更新样本数分布，pool_size_histogram 记录被使用池的容量分布，sample_share 按实际样本出现次数计数（含重复采样），不是池内独有轨迹比例。
四份 gzip 日志只包含被采池大小和采样episode_ids，没有阶段末新增/淘汰事件，也没有逐轨迹策略来源。不能用相同episode_id推断跨刷新轨迹身份。补充的 memory_refresh_external_companion 行明确来自现有 memory_updates.csv：每池首次100条、其后每次新增50/淘汰50/最终100，新增阶段池净增100，刷新阶段池净增0。此项不是从四份gzip推断出的数据。
所有数值为日志实测值或由其精确聚合/比值派生；没有外推总训练时间。阶段端点已公布：C ACC K1 61.98%/Refresh 65.63%，D 57.81%/49.22%，E 60.78%/54.84%。梯度与计时变化只能辅助描述，Refresh同时改变预算及参考轨迹，单种子日志不建立性能差异的因果解释。

## 复现

依赖 Python 3 + NumPy。从仓库根目录执行：

```sh
python docs/results/comparisons/ts5-tkpr-refresh/analysis/analyze_logs.py \
  --input-root /home/wyh/marl_method_lab/results/ts5_tkpr_refresh_3535_20261122/analysis_inputs \
  --results-root docs/results/comparisons/ts5-tkpr-refresh
```

输入子路径与校验和见 input_manifest.json。输出三份 CSV、field_inventory.json 和本文；原始输入留在实验主机。当前脚本无训练、模型修改或原始日志写入操作。
