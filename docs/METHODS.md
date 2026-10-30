# 方法与实现范围

算法与神经元干预是两个维度：例如QMIX是算法，ReBorn/ReDo是附加干预。以下为代码存在性与入口整理，不表示已逐项复现或验证论文效果。

| 算法 | baseline | ReBorn | ReDo回收 | 周期重置 |
| --- | --- | --- | --- | --- |
| QMIX | qmix | qmix_b | qmix_r | qmix_s |
| QPLEX | qplex | qplex_b | qplex_r | qplex_s |
| DMIX | dmix | dmix_b | dmix_r | dmix_s |
| RMIX | rmix | rmix_b | rmix_r | rmix_s |
| VDN | vdn | 无上游配置 | vdn_r | vdn_s |

原作者提供baseline和四个`_b`配置。`_r`/`_s`配置由对应baseline仅改变name补齐；不擅自给VDN增加ReBorn配置。learner里有分支不等于每个组合都已实测可用。

- 原训练循环`src/run.py`按name中的`_b`/`_r`/`_s`分别调用`reborn()`/`recycle()`/`reset()`。
- 调用时机是测试间隔计数达到T，原缺省T=2；不能直接说每两次梯度更新干预。检测通常使用最新replay batch。
- QMIX/VDN共用q_learner；QPLEX用dmaq_qatten_learner；DMIX用iqn_learner；RMIX用rmix_learner。
- `reset`只按作者代码作周期重置称呼，不直接等同其他论文的Reset或Shrink-and-Perturb。
- ReDo是ReBorn仓库自己的回收实现，不宣称与ReDo原作者代码逐项相同。KNIFE尚未实现。
- Agent保留ReBorn两层ReLU MLP后接GRU，以及其ActivateLayer，不能默认等于其他PyMARL网络结构。

## 用户方法

`drivers/continual/protection.py`、`release_schedule.py`、`detection_common.py`为状态加权Mixer W/V软保护；`neuron_records.py`为名单和频次记录。导入自已停止的latest_ramp工程，不导入其旧队列或结果。

参考配置：`configs/continual/knife_fig6.json`。公式omega=c/mean(c)，A=sum(omega*abs(z))/N；阈值严格大于3，最近32训练回合，实际参数更新系数0.1，最新集合替换，退出后100000环境步线性恢复；未定义检测按原实现记录与处理。仅是已导入实现和参考设置，不固化为所有未来实验要求。

## 环境覆盖

- `--suite smac`：原SMAC静态地图，调用原训练循环；支持上表配置选择。
- `--suite continual`：SMACv2 task_drift/task_switch；目前只接入QMIX baseline/softprotect。
- ReBorn/ReDo/reset尚未接入连续任务驱动，未来做这些比较时必须补接阶段调度并说明口径，不能仅把静态地图入口的可用性当作连续任务支持。
