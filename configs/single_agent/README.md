# 单智能体 MinAtar DQN/TKPR

安装 `requirements.txt` 中的依赖后，正式训练入口为：

```sh
PYTHONPATH=src python -m single_agent.train --method DQN-TKPR-K1 --seed 3535
```

还支持 `DQN-FT`、`DQN-ER`、`DQN-TKPR-K1-W`、`DQN-TKPR-K2`。同一 seed 的五种方法由独立 RNG 生成相同任务序列，并在 `task_schedule.json` 中保存。默认 7 个区间、每区间 500,000 环境步，总计 3,500,000 步。每区间三个游戏各做 20 局贪心评价，最终阶段另做各 100 局独立评价。完整协议参数见 [minatar_tkpr.json](minatar_tkpr.json)。本次只准备实现和短流程测试，没有启动正式训练。

环境使用 [MinAtar 官方包 v1.0.15](https://github.com/kenjyoung/MinAtar/tree/4241f24685622cfb52f843cab9bbcf385a53484a)的 Gymnasium `v0` 全六动作接口；Breakout、Freeway、Space Invaders 原始观测通道为 4、7、6，补零后统一为 float32 的 `[7,10,10]`。不做帧堆叠或 task ID 输入，`sticky_action_prob=0.1`、`difficulty_ramping=False`。官方 `v1` 是最小动作集合，本实验不用。

CNN-DQN 参考 Anand 与 Precup 的 [NeurIPS 2023 论文附录 C.1.6](https://papers.neurips.cc/paper_files/paper/2023/file/c94bbbef466ab1b2cfa100e41413b3a8-Paper-Conference.pdf#page=32)及其[公开实现提交 `41669d5`](https://github.com/NishanthVAnand/prediction-and-control-in-continual-reinforcement-learning/tree/41669d570541b80dbfba7bc6a8fc800554413a7e/control/minatar_crl)：借鉴 `DQN.py` 的固定 ε=0.1、逐步 Adam/MSE 更新与 1000 步目标同步，`model.py` 实际执行的 32 卷积核、256 隐层 CNN，`replay.py` 的 100k TD replay，以及 `CL_envs.py`、`run_minatar.sh` 的随机切换与 3.5M/500k 预算。论文附录 C.1.6 写 16 个卷积核和 7 个动作，但公开 DQN 代码实际使用 32 卷积核；官方 MinAtar 完整动作集合为 6。这里以公开 DQN 代码的 CNN 和官方环境为准。论文研究的是 permanent/transient 值函数，**本实现的 TKPR 不是其论文算法的复现**；此处还改为前三个区间保证三个游戏各出现一次，并固定相同 seed 的任务序列。

FT 与 TKPR 在不同游戏切换时清空 TD replay；ER 保留。相邻区间游戏相同时保留 replay，环境仍按公开 DQN 代码在区间边界重新初始化。在线网络、目标网络、优化器与全局步数均持续。三个独立随机流分别用于 ε-greedy 动作、TD replay 采样和历史记忆采样；相同 seed 的各方法从相同动作随机流开始。TD replay 以 `uint8` 存 MinAtar 二值状态，组 batch 后恢复 `float32` 输入 DQN。FT/ER 不采集无用的参考记忆。TKPR 直接复用 `learners.ranking_preference` 的 `topk_preference` 和 `topk_preference_weighted`；K1/K1-W 最多选 5 对，K2 最多选 9 对。六动作共有 15 个无序动作对，因此 K2 不等价于 FRPR。每个游戏首次训练后，冻结当前模型贪心交互 10,000 步，把所有访问状态及完整旧 Q 存入独立历史记忆；后续不覆盖，训练时排除当前游戏并在历史游戏间均衡采样。记忆另记去重后的状态数与比例，但保留全部重复状态。

运行目录保存配置、源码快照、Git 提交、任务序列、各区间检查点、参考记忆、逐局回报、训练曲线、每 1000 环境步抽样的 TKPR 与梯度诊断、逐游戏评价和每个游戏的 BWT/Forgetting。区间边界未结束的回合以 `complete=false` 单列。`PYTHONPATH=src python -m single_agent.evaluate --run-dir RUN_DIR` 可从已存检查点独立复算评价。BWT 为最终平均回报减首次学习完成后的平均回报，Forgetting 为该下降量的非负部分；不同游戏的原始回报不合并成 ACC。

MountainCar 是[历史初步实验](../../docs/results/comparisons/mountaincar-single-agent-tkpr/README.md)，原结果、CSV 和配置保留。其旧入口为 `single_agent.mountaincar_train` 与 `single_agent.mountaincar_evaluate`，不属于当前 MinAtar 协议。
