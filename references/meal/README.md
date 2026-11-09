# MEAL 静态设计参考

保留作者固定提交的五个重点文件，供神经元机制和持续学习方法设计时查阅。此目录不是可运行工程、不是已接入的方法实现；不添加 Python 搜索路径，不安装 JAX/Flax。

- [VDN 任务流](upstream/experiments/algos/vdn.py)
- [A-GEM 记忆与投影](upstream/experiments/continual/agem.py)
- [EWC 重要性状态与惩罚](upstream/experiments/continual/ewc.py)
- [MAS 重要性状态与惩罚](upstream/experiments/continual/mas.py)
- [CBP 效用与替换](upstream/experiments/continual/cbp.py)

来源版本及逐文件 SHA-256：[SOURCE.json](SOURCE.json)。作者许可：[LICENSE](LICENSE)。这些文件只是一组原样源码摘选，缺少上游依赖文件，不能当成完整发行版。没有运行或复现 MEAL，也没有确认当前提交等同论文实验版本。

阅读时结合[方法设计笔记](../../docs/MEAL_DESIGN_REFERENCE.md)，尤其注意 VDN 的 EWC/MAS 实际调用和 CBP 的接入范围。
