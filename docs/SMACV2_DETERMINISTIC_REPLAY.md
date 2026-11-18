# SMACv2 ExactProtoss 确定性重放验证

在 SC2 4.10.0（build 75689）下，使用同一初始配置、SC2 随机种子和记录的动作，在三个全新的 SC2 进程中重放一条历史策略生成的胜利轨迹。**3 次重放通过，0 次失败**。57 个动作步与最后一帧均匹配。比较了状态、观测、合法动作、回报、终止、胜负和 raw observation 可读取的双方单位信息。

## 条件与方法

- 地图与任务取自已运行的 SZC ExactProtoss 配置，`10gen_protoss`，S 任务为双方各五名追猎者。原始轨迹由 seed 1111 的 FRPR 运行 S 末 Agent（2,000,051 步）生成。检查点 SHA-256 为 `158e651caa6b659f2d6f45994afe8e9df40ab8145382dba285de491f8629f477`。
- 环境 seed 和 SC2 建局 seed 均为 **91101**。复核通过 PySC2 建局请求的实际 `random_seed` 字段。每个实例都只建局一次，强制重启计数为零。
- 原始采集采用 ExactProtoss 的正常分布采样，对每步 Q 屏蔽非法动作后贪婪执行。记录实际送入底层的完整 `episode_config`，包含队伍与双方出生位置。重放时先用相同环境 seed 核对分布能生成完全一样的配置，再从归档读回并直接注入已保存的配置。注入只作用于脚本创建的实例。三个重放环境只执行记录的联合动作。
- 每步保存动作前的 `state`、全体 `obs`、`avail_actions`、游戏时钟、双方 raw unit 字段及智能体槽位到单位 tag 的映射，然后保存该联合动作产生的 reward、终止标志和胜负信息。另存最后一步后的观察帧。单位字段包括位置、生命值、护盾和可读取的指令等。
- 运行时确认 `stochastic_attack=false`、`stochastic_health=false`、`prob_obs_enemy=1.0`。出生位置由 ExactProtoss 使用环境 seed 的 `SeedSequence` 初始化 `surrounded_and_reflect` 及嵌套 PCG64 生成器。归档记录采样前后的生成器状态，以及 Python `random` 和 NumPy 全局 RNG 在重置前、重置后和轨迹结束时的完整状态。脚本在每次重放恢复原始重置前后相应的全局状态，并比较重新初始化后记录的状态。

## 实际结果

原始轨迹在第 57 个动作步获胜，累计环境奖励为 **20.1333333333**。游戏时钟从 2 到 458。原始进程 PID 为 230605。离散值要求完全一致，数值比较用绝对容差 `1e-6` 与相对容差 0。

| 重放 | 新 SC2 PID | 动作步数 | 结局 | state 最大误差 | obs 最大误差 | reward 最大误差 | raw unit 数值最大误差 | 首次差异 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 230697 | 57 | 胜利 | 0 | 0 | 0 | 0 | 无 |
| 2 | 230787 | 57 | 胜利 | 0 | 0 | 0 | 0 | 无 |
| 3 | 230880 | 57 | 胜利 | 0 | 0 | 0 | 0 | 无 |

联合动作、合法动作掩码、终止、胜负、单位 tag 和槽位映射全部相同。三个重放分别复现了出生配置和记录的 RNG 状态。独立核对解压后的逐步数据与最后一帧完全相等，并通过内存扰动核对了比较器对超容差差异和离散掩码差异的定位。

这次实验支持从同样的建局条件和动作历史，重建**该轨迹的可观测状态**，也覆盖了 SC2 raw API 返回的单位层状态。证据范围是本任务、该配置、版本、种子和轨迹。SC2 的内部 RNG、引擎队列和其他未暴露的状态没有可比较的完整快照。

## 复跑与证据

从 Ubuntu 方法仓库根目录执行下列命令。正式运行使用统一调度器，计划文件为 `results/smacv2_exact_replay_20261118_attempt001/plan.json`。复跑使用新的输出目录。

```sh
run=runs/szc_vdn64_frpr_seed1111_attempt001
.venv/bin/python scripts/verify_smacv2_replay.py \
  --run "$run" --checkpoint "$run/models/phase0000_2000051/agent.th" \
  --output results/smacv2_exact_replay_20261118_attempt001/evidence \
  --task stalker --env-seed 91101 --sc2-seed 91101
```

- [验证脚本](../scripts/verify_smacv2_replay.py)
- 实验主机结果目录的 `evidence/metadata.json` 记录命令、配置与哈希，`evidence/summary.json` 记录逐次比较和首次差异。`original.json.gz` 与 `replay_{1,2,3}.json.gz` 含完整逐步证据。
- 执行时仓库基线为 `6e0c745`。实际脚本与批次源码快照的 SHA-256 均为 `64cab7468a9c76d2463ed1e21ad492035dee948c9c5a3f0728cb0a84ddf28750`。
