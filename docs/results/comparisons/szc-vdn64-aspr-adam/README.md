# SZC VDN64 · ASPR 与 Adam 优化器对照

追猎者 S → 狂热者 Z → 巨像 C，每阶段 2M 步，FT 回放，逐任务 ε，阶段末每个已见任务独立贪心评估 128 局。种子为 2222 和 3535。

## C 阶段结束结果

| 方法 | 2222 ACC % | 3535 ACC % | 平均 ACC % | 平均 BWT（百分点） | 平均 FWT（百分点） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adam TKPR-K1-W | 72.40 | 74.22 | 73.31 | 1.56 | 28.32 |
| Adam TKPR-K1 | 70.31 | 69.53 | 69.92 | -2.34 | 27.34 |
| Adam FRPR | 65.62 | 73.70 | 69.66 | -4.30 | 28.71 |
| RMSprop ASPR | 61.98 | 64.32 | 63.15 | -5.47 | 24.61 |
| RMSprop FRPR | 63.28 | 63.54 | 63.41 | 1.17 | 24.22 |
| RMSprop TKPR-K1 | 63.28 | 64.06 | 63.67 | 0.00 | 24.02 |
| RMSprop TKPR-K1-W | 65.10 | 63.54 | 64.32 | -0.59 | 20.51 |
| RMSprop VDN-FT | 21.61 | 22.40 | 22.01 | -65.82 | 25.98 |

Adam TKPR-K1-W 在 Adam 组内两个种子的 ACC 均最高。RMSprop ASPR 和 FRPR 的平均 ACC 接近。ASPR 的 C 末巨像胜率在两个种子上更高，追猎者胜率更低。

## 配对口径

- RMSprop ASPR、FRPR、TKPR-K1、TKPR-K1-W 与 VDN-FT 按种子复用同一 S 检查点。
- Adam TKPR-K1、TKPR-K1-W 与 FRPR 按种子共用 Adam S 检查点。优化器对照从 S 起分化，训练提交也有差异。新增的 RMSprop 3535 分支启动于 b7c6860，其余新任务启动于 efc41e6。
- S 前缀各记一次。按 parent_run_id 拼接 S 和 Z/C 得到完整曲线。训练胜率包含 ε 探索，贪心评估使用独立局数。
- learning_auc_pct 是每阶段当前任务的贪心胜率在该阶段实际步数上的归一化梯形积分。paired_metrics.csv 的 AUC 差对应 C 阶段。

## 数据

- [运行清单](runs.csv) · [配置快照](configs/) · [独立评估](evaluations.csv) · [训练曲线标量](training_curves.csv)
- [阶段指标](stage_metrics.csv) · [C 末配对差](paired_metrics.csv) · [新增分支偏好诊断](preference_diagnostics.csv)
