# 来源、修改和运行时

- ReBorn作者仓库：https://github.com/xmu-rl-3dv/ReBorn ，固定提交865249f974c291a3e3df06878d8dbf1a74cb66bf。
- 导入源：/home/wyh/reborn_figure1/upstream/src；导入时逐文件记录SHA-256。原README和requirements保存在docs，原LICENSE保留根目录。
- SMACv2源码复制自/home/wyh/stagnant_figure2/vendor/smacv2，项目既有固定版本577ab5a2cff2391f8df582da5731ea9cd6adf3c6；现存副本可能含已有环境适配，精确内容以source_imports.json为准，不能据提交名宣称未修改。
- 用户软保护与连续任务相关模块复制自/home/wyh/qmix_mixer_knife_fig6_latest_ramp/scripts；保留原实现，仅适配仓库根路径、src路径、协议/输出入口，独立评价CLI。原队列、验证门槛、旧结果不导入。
- src/main.py的PyYAML加载和collections.Mapping适配；结果输出路径隔离到run目录；src/run.py的TensorBoard路径同样隔离；runner应用2026-10-30已有的测试日志分支修复。初始差异见initial_changes.patch。
- scripts/smac_entry.py沿用原run.run，增加显式Python随机数种子、CUDA种子和单线程Torch，完整配置通过Sacred保存；这些与原main入口的差异应在对照时保留说明。
- 不修改learner损失、神经元干预实现、网络结构或默认干预频率。新增ReDo/reset YAML只改变方法name。

## 运行时

代码独立复制；本机依赖暂复用已有环境：.venv链接至/home/wyh/stagnant_figure2/.venv，SC2链接至/home/wyh/reborn_figure1/runtime/StarCraftII。不修改原环境。精确pip包版本记录在runtime_packages.txt。SMAC作为该环境内依赖，SMACv2源码在本仓库vendor内。此处记录的是实验主机配置，可移植安装说明待整理。实验主机的绝对路径链接由本机维护。

## 许可证

该固定版本的原README称Apache2.0，但LICENSE正文为GPLv3，存在不一致。本仓库保留上游原文及第三方版权和声明。适用条款仍待上游确认。独立仓库名称或新Git历史不改变代码来源。SMACv2自身LICENSE随vendor保留。

公开副本于 2026-11-14 首次从 Ubuntu 主仓库推送，首推版本为 `d96055f`。后续方法代码同步规则见 [AGENTS.md](../AGENTS.md)。

Continuous SMACv2 runs use runtime/SMACv2StarCraftII -> /home/wyh/stagnant_figure2/runtime/StarCraftII (existing SC2 4.10 task-switch runtime including SMACv2 maps). The static SMAC runtime/StarCraftII remains unchanged. Both are read-only reused dependencies.

## MEAL 设计参考材料（2026-11-09）

`references/meal/` 收录 TTomilin/MEAL 固定提交 `8e7857c76eedc44623a7bfe987235f2a9aadb626` 的五个重点源码文件，仅作静态参考，不被训练导入。原作者 MIT 许可保存在该目录 LICENSE；来源及 SHA-256 见 SOURCE.json。不改变本仓库既有来源和许可记录，不声称完成 MEAL 复现或实现这些方法。设计用途及当前模块对应关系见 [MEAL 设计参考](MEAL_DESIGN_REFERENCE.md)。
