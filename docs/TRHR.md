# TRHR A2

目标激活率驱动的神经元稳态调节（Target-Rate Homeostatic Regulation）。首版是两层共享 RNN Agent 上的独立 VDN 干预，接入原版 SMAC 单任务和 TS3 的 VDN 后缀分支。

## 前向与统计

- A2 是第二个 Linear（net[3]）后的 ReLU（net[4]）。TRHR 启用时，A2 输出为 ReLU(z + c)，其中 z 包含 Linear 原有的可训练偏置，c 是逐单元非负补偿。初始 c=0。A2 的模块位置不变。
- QLearner 只采集在线网络在 TD 训练回放前向中的激活符号 1[z+c>0]，沿用 TD 有效掩码，并用下一时刻的 filled 去掉每局最后一个仅用于 bootstrap 的观测。只允许 no-op 或没有可用动作的行不计入统计。目标网络、环境采样、诊断和评估前向不采集。
- 整体率用有效行加权汇总。条件率按智能体槽位与是否有可用目标动作（标准 SMAC/TS3 动作索引 ≥6）交叉分组。各组在本窗口有足够样本才参与条件判定。槽位区分不同局部输入流。它不是兵种标签，也不覆盖所有稀有情境。
- 每个窗口计算整体和有效条件的激活率，并更新各自 EMA。每项 EMA 在首次有足够数据时以实际观测初始化。当前窗口内样本不足的组不参与否决。没有受支持条件的窗口不增加补偿。

## 更新规则

某单元在当前窗口的整体率和 EMA 都低于目标，且任何受支持条件的当前率和 EMA 均未达到目标时，计入连续低活跃窗口。条件不再满足时清零连续计数。达到所需窗口数后，在正常 RMSprop 更新之后执行：

    c[i] ← min(max_offset, c[i] + gain × max(target_rate − EMA[i], 0))

达到目标或发现条件活跃时，停止增加并保留已有补偿。控制器只更新缓冲区，输入权重、原偏置和 GRU 连线继续通过 TD 损失学习。在线网络下一次前向即使用新补偿，目标网络按原有时点同步。

| 配置字段 | 初始值 | 含义 |
| --- | ---: | --- |
| target_rate | 0.05 | 激活率下限 |
| window_updates | 100 | 一个统计窗口内的实际 learner 更新次数 |
| low_windows | 3 | 开始补偿所需的连续低活跃窗口 |
| min_condition_samples | 128 | 单个槽位与动作条件组合的最低窗口样本行数 |
| ema_weight | 0.25 | 新窗口激活率在 EMA 中的权重 |
| gain | 0.1 | 激活率缺口到补偿增量的比例 |
| max_offset | 0.25 | 补偿上限，以预激活单位计 |

这些是实现的起始值。窗口以回放中的样本为准，同一转移可能重复采样。TS3 延续前缀回放，所以跨任务时窗口统计会混合新旧分布。

## 配置与检查点

使用统一入口，给定已有匹配的 TS3 前缀，或指定静态地图与训练步数。下面只生成预览：

    .venv/bin/python scripts/run.py --suite smac --algorithm vdn --method trhr \
      --map MAP --steps STEPS --seed SEED --run-id RUN_ID --dry-run

    .venv/bin/python scripts/run.py --suite ts3 --algorithm vdn --method trhr \
      --ts3-role branch --ts3-group N \
      --protocol configs/continual/neuron_protection/softprotect/ts3_vdn_A2_top3_1234_4567.json \
      --parent PREFIX_DIR --seed SEED --run-id RUN_ID --dry-run

可用重复的 --set trhr.FIELD=VALUE 覆盖上表字段。TRHR 分支只复用父分支的训练状态，独立调节低活跃单元。方法当前独立运行，和软保护的组合留待验证后接入。

agent.th（以及 TS3 的 targets.th）保存补偿、EMA、低活跃计数与未完成窗口统计。同一配置加载时恢复这些状态。旧基线检查点缺少 A2 控制器缓冲区时，以零状态加载，已有网络权重照常严格校验。TS3 的分支模型哈希包含新缓冲区，因此即使旧权重完全相同，模型哈希也会变化。评估使用保存的补偿，前向不会更新控制器。

现有日志按窗口记录 trhr_* 指标，包括有效行数、受支持的条件数、均值激活率、被条件否决／符合调节／实际补偿的单元数和补偿上限情况。实现验证使用合成回放和历史 A 阶段模型。激活率、任务梯度恢复与胜率改善需要分别检验。
