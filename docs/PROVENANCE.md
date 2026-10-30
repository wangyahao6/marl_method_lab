# 来源、修改和运行时

- ReBorn作者仓库：https://github.com/xmu-rl-3dv/ReBorn ，固定提交865249f974c291a3e3df06878d8dbf1a74cb66bf。
- 导入源：/home/wyh/reborn_figure1/upstream/src；导入时逐文件记录SHA-256。原README和requirements保存在docs，原LICENSE保留根目录。
- SMACv2源码复制自/home/wyh/stagnant_figure2/vendor/smacv2，项目既有固定版本577ab5a2cff2391f8df582da5731ea9cd6adf3c6；现存副本可能含已有环境适配，精确内容以source_imports.json为准，不能据提交名宣称未修改。
- 用户软保护与连续任务相关模块复制自/home/wyh/qmix_mixer_knife_fig6_latest_ramp/scripts；保留原实现，仅适配仓库根路径、src路径、协议/输出入口，独立评价CLI。原队列、验证门槛、旧结果不导入。
- src/main.py的PyYAML加载和collections.Mapping适配；结果输出路径隔离到run目录；src/run.py的TensorBoard路径同样隔离；runner应用2026-10-30已有的测试日志分支修复。初始差异见initial_changes.patch。
- scripts/smac_entry.py沿用原run.run，增加显式Python随机数种子、CUDA种子和单线程Torch，完整配置通过Sacred保存；这些与原main入口的差异应在对照时保留说明。
- 不修改learner损失、神经元干预实现、网络结构或默认干预频率。新增ReDo/reset YAML只改变方法name。

## 运行时

代码独立复制；本机依赖暂复用已有环境：.venv链接至/home/wyh/stagnant_figure2/.venv，SC2链接至/home/wyh/reborn_figure1/runtime/StarCraftII。不修改原环境。精确pip包版本记录在runtime_packages.txt。SMAC作为该环境内依赖，SMACv2源码在本仓库vendor内。以后公开时应整理安装说明；现有绝对路径链接不作为可移植安装方案，也不提交Git。

## 许可证

该固定版本的原README称Apache2.0，但LICENSE正文为GPLv3，存在不一致。本仓库保留双方原文，不自行指定新的整体许可证；公开前需确认来源许可口径并保留第三方版权和声明。独立仓库名称或新Git历史不改变代码来源。SMACv2自身LICENSE随vendor保留。

当前仅远端本地Git，无origin、无推送。旧实验工程未迁移、未改写。此次建立只做静态核对，不产生训练结果。
