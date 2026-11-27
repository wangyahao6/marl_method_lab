# 研究代码仓库约定

- Ubuntu 仓库是方法开发与新实验的代码主版本，公开副本见 README。
- 研究设计、代码整理不自动授权训练。用户明确授权实验后按指定方法、环境、种子和预算执行。
- 不添加非必要训练断言、中止阈值或前置训练验证；辅助诊断差异记录为现象，不阻断训练或整个队列。
- 公共算法在 src，连续任务驱动在 drivers/continual，suite 启动计划在 scripts/launchers，实验方案按研究方向组织在 configs。不要为每个实验复制整个工程。
- 用 scripts/run.py 启动新运行，协议使用新目录路径，旧平铺路径由 configs/continual/path_aliases.json 兼容；每次唯一 run-id，不覆盖旧产物。保留启动时配置、源码快照和版本记录。
- 修改已有方法保持原来源及语义；方法变化用配置/版本说明，不把旧结果归为新实现结果。
- 某一运行失败只记录该运行；剩余已授权任务按原计划继续，不增加种子或预算。
- 完成声明须依据训练终态、独立评价和必要曲线/汇总，启动进程与退出码本身不是实验结论。
- 本机 .venv 与 SC2 是已有运行时的只读复用链接，不在原工程安装、升级、删除文件。
- 原 ReBorn README 与 LICENSE 的许可证表述不一致，保留原文及第三方声明，适用许可仍待上游确认。

- 代码有实质变更后更新相关说明并提交 Git 版本。方法代码通过相应验证后，可在任务收尾时从 Mac 用仓库专用部署密钥同步 Ubuntu 的 `main` 到 GitHub，并核对两端提交号。原始实验产物和运行时保留在各实验主机。启动实验时留存实际源码快照。公开结果按[结果数据规范](docs/results/DATA.md)分类整理。随结果更新机器索引与阅读索引，保留逐种子证据、实际训练版本和可比较范围。

- 已授权实验统一经 scheduler/service.py 的完整计划或 scripts/run.py 入口提交，全局默认 6 个槽位；新批次应提前发布完整任务及依赖。调度运行记录在 runtime/scheduler/state.sqlite3，操作见 docs/SCHEDULER.md。
- 查询已有实验优先使用[环境与任务索引](docs/environments/README.md)及[架构索引](docs/architectures/README.md)，按环境、Benchmark、骨架、方法和种子定位原报告。归档时更新 Catalog 与来源记录，再按[实验用法](docs/EXPERIMENTS.md#查询与更新实验索引)运行默认 `python3 scripts/experiment_index.py build`和 `python3 scripts/experiment_index.py check`；不手工维护生成的 CSV。索引是离线快照，实时状态以 scheduler 和运行产物为准；索引核对不作为训练启动门槛或队列中止条件。
