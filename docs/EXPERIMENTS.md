# 新实验的使用方式

以后可以直接说：“在我的marl_method_lab仓库中……”。Codex先读取本仓库README、AGENTS、方法清单和相关方案。

一次实验说明：目的、方法/比较对象、环境或任务序列、种子、每次预算；其余可明确“沿用某个方案”。只有方案设计就说“只设计，不运行”；需要执行就说“在远端运行”。缺少决定性设置时确认缺项，不猜测种子或预算。

## 静态SMAC示例（命令只是示例，未执行训练）

```sh
cd /home/wyh/marl_method_lab
.venv/bin/python scripts/run.py --suite smac --algorithm qmix --method reborn --map 3s_vs_5z --seed 1 --steps 2050000 --run-id example_qmix_reborn_seed1 --dry-run
```

去掉`--dry-run`才启动。方法可选baseline/reborn/redo/reset，组合支持见方法清单。通过`--set test_interval=10000`等覆盖超参数；原始name保留作方法分支选择，不作为运行编号。完整配置落到各run目录。当前保留原训练循环整回合结束逻辑，steps是预算阈值，最终可能整回合超出；不要与连续任务精确截断混为一谈。原默认模型保存周期较长，若实验需要阶段模型，应在方案明确设置save_model_interval。

## 连续任务示例（参考配置，不是已授权队列）

```sh
.venv/bin/python scripts/run.py --suite continual --algorithm qmix --method softprotect --scenario task_switch --seed 1111 --run-id example_switch_softprotect_1111 --protocol configs/continual/knife_fig6.json --dry-run
```

种子由命令显式指定，阶段和训练设置来自参考方案及驱动；需要变化时新建有名称的方案文件，不反写旧运行配置。导入参考方案清除了旧队列顺序、重试编号和计划总运行数，不会恢复旧实验。连续任务当前保留原驱动的batch/replay/lr等设置，修改训练超参数时需同步驱动读取方式，不声称可用任意JSON键覆盖。

## 记录与结果

每次run-id唯一，目录已存在时不覆盖。启动保留config/protocol、launch.json、源码SHA-256与source.tar.gz、console.log；进程终态由launcher_status.json记录，exited只代表进程退出。连续任务另有原训练器status.json与逐次评价/检测记录。

独立评价：`.venv/bin/python scripts/evaluate.py --run /home/wyh/marl_method_lab/runs/<run-id> --episodes 32`。静态SMAC用`--step`选择最接近的已有模型步数（0选择最大），保留上游选择逻辑；连续任务评价最终模型。评价结果另存该运行evaluation目录。

多方法/多种子由后续具体实验的方案定义，必要时编写简单队列；某run失败不连带取消剩余已授权任务。不默认新增验证训练、种子或中止条件。汇总图表和REPORT按具体研究问题生成，不预设所有实验都用同一CI或图。

## 可以这样下达任务

- “在我的仓库中，设计QMIX、QMIX+ReBorn、QMIX+本方法在……上的比较方案，先不运行。”
- “在我的仓库中，按configs/……方案运行远端实验，种子……，每次……步；原始产物留远端，结束后整理独立评价、胜率曲线和报告。”
- “在我的仓库中修改本方法的……，其余设置沿用……；这次只改代码，不启动训练。”
- “检查我的仓库实验……的进度，只读，不重启。”
