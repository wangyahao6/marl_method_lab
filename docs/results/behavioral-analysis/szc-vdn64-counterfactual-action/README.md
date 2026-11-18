# SZC · 单智能体反事实动作干预

同一条 57 步 S 任务胜利轨迹的三个决策点上，**14 次单智能体非贪婪替换全部获胜**。三个独立进程中的单点无干预基准也都复现了原胜局。此结果验证了固定初始条件和历史动作下的分支执行，以及较低旧 Q 动作在冻结旧策略继续执行时仍能保胜的现象。

## 方法

- 复用 [SMACv2 确定性重放](../../../SMACV2_DETERMINISTIC_REPLAY.md) 的轨迹与 ExactProtoss 初始化。SC2 seed 和环境 seed 均为 91101，任务为镜像 5v5 追猎者。旧策略为 FRPR seed 1111 的 S 阶段末 VDN64 Agent。轨迹和检查点的 SHA-256 均与前次归档匹配。
- 离线使用真实 `BasicMAC.forward` 和动作选择器，按归档数据逐步填入 EpisodeBatch，由 MAC 组装观测、上一实际动作的 one-hot 和 Agent ID。在线基准调用 `BasicMAC.select_actions`，在该次 forward 中捕获 Q，并保存前后 hidden state。两边各自从 `init_hidden` 开始。
- 在零基的第 9、28、47 步各选一个点，取存活且合法动作超过一个的最小 Agent 槽位。逐一枚举目标智能体的全部合法动作，每个条件只运行一次。每个动作使用独立 SC2 进程和归档 `episode_config`，重放此前的原始联合动作，并逐步对照状态、观测、掩码、单位字段、奖励和终止信息。
- 目标步装入离线展开得到的该步 forward **之前**的 hidden state，再用真实 MAC 计算当前贪婪联合动作。仅替换目标槽位。写回实际执行的联合动作后，从下一步起所有智能体都按干预后自身观测继续执行同一冻结策略。
- 胜利记 Good，终局失败记 Bad，超时、异常或无效分支记 Uncertain。在线重建、目标步 GRU 一致性与三个无干预基准都通过后才执行干预。所有尝试均写入逐动作记录。数值比较使用 `atol=1e-6, rtol=0`，离散字段精确比较。

## 结果

完整在线基准和离线展开均覆盖 57 步。**Q、forward 前 hidden state、forward 后 hidden state 的最大绝对误差均为 0**，各步的贪婪联合动作也完全相同。17 个逐动作条件的前缀和目标 GRU 校验均通过，首次差异均为空。

| 决策点 | Agent 槽位 | 旧贪婪动作 | 合法动作数 | Good / Bad / Uncertain | 可保胜的最大旧 Q 差 | 该动作 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 早期 t=9 | 0 | 2 | 6 | 6 / 0 / 0 | 0.130409 | 5 |
| 中期 t=28 | 0 | 1 | 6 | 6 / 0 / 0 | 0.146952 | 4 |
| 后期 t=47 | 1 | 9 | 5 | 5 / 0 / 0 | 0.181544 | 5 |

每行含一个无干预基准。三个基准各使用独立新进程，均重现原始 57 步动作序列、终止帧与胜利结果。反事实干预的局长为 56–64 步。后期例子中，动作 5 的旧 Q 约为 **1.775187**，旧贪婪动作 9 的 Q 约为 **1.956731**。替换为动作 5 后在第 59 步结束并获胜。

### 与 ASPR Q 分组的关系

仓库中的 ASPR 按旧模型合法动作的原始 Q 差生成动作集合。这个实验按一次干预后原冻结策略是否获胜分组。三个决策点都有多种较低 Q 的保胜动作。在各自决策点，若归档 ASPR 的 Q 差阈值小于表中最大保胜 Q 差，相应动作会落在 Q 分组之外。当前检查点没有可直接配对的 ASPR 校准归档，本次保留真实 Q 差供比较。全部动作都获胜，因此这批数据只验证了保胜动作的识别，区分 Good 与 Bad 的能力还缺少失败样本。观察范围为一个成功轨迹、三个槽位决策及固定后续策略。

## 运行与产物

正式运行号为 `smacv2_counterfactual_20261118_attempt001`，通过 Ubuntu 统一调度器完成。复跑使用独立输出目录。

```sh
.venv/bin/python scripts/verify_smacv2_counterfactual.py \
  --source results/smacv2_exact_replay_20261118_attempt001/evidence \
  --output results/smacv2_counterfactual_new/evidence
```

- [逐动作结果](branches.json) 包含 Q、干预前后联合动作、胜负、终止原因、局长、核验误差和首次差异。 [目标状态与合法动作](sites.json)
- [GRU 核验摘要](rnn_summary.json)和[逐步在线／离线 Q 与 hidden state](rnn_trace.json.gz)保留完整对照。 [命令、检查点与哈希](manifest.json)
- [实验脚本](../../../../scripts/verify_smacv2_counterfactual.py)复用[确定性重放模块](../../../../scripts/verify_smacv2_replay.py)的建局、配置注入、状态采样和比较逻辑。实验冻结原策略，新增训练步数为零。
