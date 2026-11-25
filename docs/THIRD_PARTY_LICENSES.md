# 第三方代码、许可证与待确认事项

本页依据当前仓库保留的源码、许可证、README 和来源清单整理（2026-11-25），不是对授权争议的裁决。不修改根 LICENSE，也不替代各第三方目录中的原文。实验结果、论文引用和代码使用许可分别记录。

| 代码范围 | 来源及本地证据 | 本地许可证/版权声明 | 分发时应保留的材料与待确认项 |
| --- | --- | --- | --- |
| 导入的 ReBorn 底座，主要为 `src/`；本仓库后续修改 | [xmu-rl-3dv/ReBorn](https://github.com/xmu-rl-3dv/ReBorn)，固定提交 `865249f974c291a3e3df06878d8dbf1a74cb66bf`；[导入清单](source_imports.json)、[原 README](REBORN_README.md)、[来源说明](PROVENANCE.md) | 根 [LICENSE](../LICENSE) 是 GPLv3 正文；原 README 的 License 段声称 Apache License v2.0。**二者冲突，实际适用授权待上游/权利人确认。** | 保留根 LICENSE、原 README、已有文件头和来源/修改记录。根文本开头的 `Copyright (C) 2007 Free Software Foundation, Inc.` 是许可证文本声明，不能据此认定为 ReBorn 源码的版权持有人。未找到可独立消除冲突的 ReBorn NOTICE 或 Apache-2.0 完整许可文本，不擅自补写版权主体或改根许可证。 |
| ReBorn 声明的 PyMARL/PyMARL2 衍生关系 | 原 [ReBorn README](REBORN_README.md) 开头明确说基于 PyMARL2、PyMARL；具体逐文件谱系未由本地导入清单证明 | 本仓库未单独保存足以归属每个继承文件的 PyMARL/PyMARL2 许可证/NOTICE 证据 | **待人工确认**上游具体版本、对应文件、版权声明及再分发要求；不能用 ReBorn README 的一句 Apache 声明替代继承代码授权核验。 |
| `vendor/smacv2/` | [oxwhirl/smacv2](https://github.com/oxwhirl/smacv2)；来源基准 `577ab5a2cff2391f8df582da5731ea9cd6adf3c6`，本地已有环境适配，内容以导入清单及文件为准 | [随附 LICENSE](../vendor/smacv2/LICENSE)：MIT；`Copyright (c) 2019 whirl`；`setup.py` 也声明 MIT | 包含副本或实质部分时保留完整版权、许可授予、条件和免责文字。本地没有发现另外的 NOTICE 文件；保留已有源码/地图相关说明。MIT 文本不表示 StarCraft II 游戏、Blizzard API/资源或其他依赖也自动获 MIT 授权。 |
| `references/meal/` 的五个静态参考文件 | [TTomilin/MEAL](https://github.com/TTomilin/MEAL)，固定提交 `8e7857c76eedc44623a7bfe987235f2a9aadb626`；[SOURCE.json](../references/meal/SOURCE.json) 记录逐文件 SHA-256；[用途说明](../references/meal/README.md) | [随附 LICENSE](../references/meal/LICENSE)：MIT；`Copyright (c) 2025 Tristan Tomilin` | 保留该版权声明及完整 MIT 文本。此处是源码摘选，未接入训练，不是完整 MEAL 发行版。当前 `references/` 只有此参考目录；没有发现独立 NOTICE 文件。 |

## NOTICE 与分发边界

- MIT 代码的版权和许可条件已由原目录 LICENSE 保存，本页只提供定位；复制代码时不能只复制本表而遗漏原文。
- ReBorn 的 Apache/GPL 冲突未解决。本地没有确认过的 NOTICE 文件可供转录，也不能把“未找到 NOTICE”解释为已经核实上游没有 NOTICE 要求。确认 Apache 适用时还需核查该版本上游是否附带 NOTICE、文件归属和修改说明；确认 GPL 适用时需按确定的授权关系核对源码和再分发安排。
- SMACv2 README 说明其使用 SMAC、PySC2 和 StarCraft II API。运行时依赖、游戏程序与地图不因纳入实验环境而改变各自授权；本次只盘点仓库实际保留的代码声明，没有对未随仓库分发的安装包作全面许可证审计。
- 新增 Generic Prefix、Hook 和恢复日志代码不改变导入代码的来源，也不自动解决整仓库的许可关系。根许可证、第三方许可证、原结果和原始版权声明均保持不变。
