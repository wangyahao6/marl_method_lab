"""Export publication-ready results for the shared-prefix SZC comparison."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean, median

TASKS = ("stalker", "zealot", "colossus")
METHODS = ("VDN-FT", "TKPR-K1", "T1PR", "TKPR-K1-W", "TKPR-K1-HR")
RUN_FIELDS = ("run_id", "method", "seed", "role", "parent_run_id")
END_METRICS = ("acc_pct", "bwt_pp", "fwt_pp", "learning_auc_pct")
FIELDS = {
    "runs": (*RUN_FIELDS, "commit", "status", "exit_code", "replay_on_switch", "epsilon_on_switch",
             "completed_global_steps", "trained_steps", "restore_model_hash", "prefix_model_hash", "config"),
    "evaluations": (*RUN_FIELDS, "phase", "task", "label", "t_env", "wins", "episodes"),
    "stage_metrics": (*RUN_FIELDS, "phase", "training_task", "start", "t_env",
                      *(f"{task}_win_pct" for task in TASKS), *END_METRICS),
    "learning_curves": (*RUN_FIELDS, "phase", "task", "label", "t_env", "phase_steps", "wins", "episodes",
                        "win_pct"),
    "paired_metrics": ("method", "control_method", "seed", "phase", "training_task", "run_id",
                       "control_run_id", "parent_run_id", *(f"{metric}_delta" for metric in END_METRICS)),
    "preference_diagnostics": (*RUN_FIELDS, "phase", "training_task", "source_phase", "historical_task",
                               "updates", "decision_states", "valid_states", "valid_pairs", "coverage",
                               "top1_flips", "top1_states", "top1_flip_rate", "pair_violation_rate",
                               "mean_task_loss", "mean_state_confidence", "zero_confidence_states",
                               "high_regret_states", "high_regret_fraction", "high_regret_loss",
                               "mean_high_regret"),
    "gradient_diagnostics": (*RUN_FIELDS, "phase", "training_task", "metric", "logged_points",
                             "defined_points", "mean", "median"),
    "timing_summary": ("phase", "task", "seed", "snapshots", "pairs", "td_mean_ms", "td_tkpr_mean_ms",
                       "extra_mean_ms", "relative_overhead_pct"),
}


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def records(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)


def write_csv(out, name, rows):
    path = out / f"{name}.csv"
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS[name], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def win_percent(row):
    return 100 * row["wins"] / row["episodes"]


def stage_rows(identity, retention, evaluations, own_evaluations):
    references = {label: {row["task"]: win_percent(row) for row in evaluations if row["label"] == label}
                  for label in ("initial_reference", "next_unseen")}
    old = {}
    stage_metrics, curves = [], []
    for stage in retention.get("stages", []):
        phase, task, start, end = (stage[key] for key in ("phase", "task", "start", "t_env"))
        endpoints = {row["task"]: win_percent(row) for row in evaluations
                     if row["phase"] == phase and row["label"] == "stage_end"}
        if set(endpoints) != set(TASKS[:phase + 1]):
            break
        bwt = [endpoints[name] - win for name, win in old.items()]
        fwt = [references["next_unseen"][name] - references["initial_reference"][name]
               for name in TASKS[1:phase + 1]
               if name in references["next_unseen"] and name in references["initial_reference"]]
        old[task] = endpoints[task]
        curve = [row for row in own_evaluations if row["phase"] == phase and row["task"] == task
                 and row["label"] in ("phase_start", "periodic", "stage_end")]
        if not curve:
            continue  # The parent supplies its own stage and curve.
        curve.sort(key=lambda row: row["t_env"])
        area = sum((win_percent(a) + win_percent(b)) * (b["t_env"] - a["t_env"]) / 2
                   for a, b in zip(curve, curve[1:]))
        auc = ratio(area, end - start) if curve[0]["t_env"] == start and curve[-1]["t_env"] == end else None
        stage_metrics.append(dict(identity, phase=phase, training_task=task, start=start, t_env=end,
                                  **{f"{name}_win_pct": win for name, win in endpoints.items()},
                                  acc_pct=mean(endpoints.values()), bwt_pp=mean(bwt) if bwt else None,
                                  fwt_pp=mean(fwt) if len(fwt) == phase and fwt else None,
                                  learning_auc_pct=auc))
        for row in curve:
            curves.append(dict(identity, phase=phase, task=task, label=row["label"], t_env=row["t_env"],
                               phase_steps=row["t_env"] - start, wins=row["wins"], episodes=row["episodes"],
                               win_pct=win_percent(row)))
    return stage_metrics, curves


def preference_rows(identity, run, method):
    counts, seen_fields = defaultdict(Counter), defaultdict(set)
    for row in records(run / f"{method}_updates.jsonl"):
        for item in row["historical_tasks"]:
            key = row["phase"], row["task"], item["source_phase"], item["task"]
            totals = counts[key]
            totals["updates"] += 1
            for field in ("decision_states", "valid_states", "valid_pairs", "top1_flips", "top1_states",
                          "zero_confidence_states", "high_regret_states"):
                if field in item:
                    totals[field] += item[field]
                    seen_fields[key].add(field)
            for field, weight in (("loss", 1), ("pair_violation_rate", item.get("valid_pairs", 0)),
                                  ("mean_state_confidence", item.get("valid_states", 0)),
                                  ("mean_high_regret", item.get("high_regret_states", 0)),
                                  ("high_regret_loss", item.get("high_regret_states", 0))):
                if item.get(field) is not None:
                    totals[field + "_sum"] += item[field] * weight
                    totals[field + "_weight"] += weight
    for (phase, task, source, historical), total in sorted(counts.items()):
        key = phase, task, source, historical
        yield dict(identity, phase=phase, training_task=task, source_phase=source, historical_task=historical,
                   updates=total["updates"], **{field: total[field] for field in seen_fields[key]},
                   coverage=ratio(total["valid_states"], total["decision_states"]),
                   top1_flip_rate=ratio(total["top1_flips"], total["top1_states"]),
                   pair_violation_rate=ratio(total["pair_violation_rate_sum"], total["pair_violation_rate_weight"]),
                   mean_task_loss=ratio(total["loss_sum"], total["loss_weight"]),
                   mean_state_confidence=ratio(total["mean_state_confidence_sum"], total["mean_state_confidence_weight"]),
                   high_regret_fraction=(ratio(total["high_regret_states"], total["decision_states"])
                                         if "high_regret_states" in seen_fields[key] else None),
                   mean_high_regret=ratio(total["mean_high_regret_sum"], total["mean_high_regret_weight"]),
                   high_regret_loss=ratio(total["high_regret_loss_sum"], total["high_regret_loss_weight"]))


def gradient_rows(identity, run):
    path = run / "experiment_metrics.jsonl"
    if not path.exists():
        return
    count, values = Counter(), defaultdict(list)
    with path.open() as stream:
        for line in stream:
            if '"gradient_monitor"' not in line:
                continue
            row = json.loads(line)
            if row["context"].get("event") != "gradient_monitor":
                continue
            key = row["phase"], row["task"], row["metric"]
            count[key] += 1
            if row["value"] is not None:
                values[key].append(row["value"])
    for (phase, task, metric), points in sorted(count.items()):
        vals = values[phase, task, metric]
        yield dict(identity, phase=phase, training_task=task, metric=metric,
                   logged_points=points, defined_points=len(vals), mean=mean(vals) if vals else None,
                   median=median(vals) if vals else None)


def paired_rows(stages, runs):
    completed = {row["run_id"] for row in runs if row["status"] == "completed" and row["exit_code"] == 0}
    pairs = {(row["parent_run_id"], row["seed"], row["phase"], row["method"]): row
             for row in stages if row["parent_run_id"] and row["run_id"] in completed}
    for key, row in pairs.items():
        parent, seed, phase, method = key
        for control in ("VDN-FT", "TKPR-K1"):
            other = pairs.get((parent, seed, phase, control))
            if other is None or method == control or (method == "VDN-FT" and control == "TKPR-K1"):
                continue
            yield {**{key: row[key] for key in ("method", "seed", "phase", "training_task", "run_id",
                                                "parent_run_id")}, "control_method": control,
                   "control_run_id": other["run_id"],
                   **{f"{metric}_delta": (row[metric] - other[metric])
                      if row[metric] is not None and other[metric] is not None else None for metric in END_METRICS}}


def timing_rows(path):
    if not path.exists():
        return []
    grouped = defaultdict(list)
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            for seed in (row["seed"], "all"):
                grouped[int(row["phase"]), row["task"], seed].append(row)
    rows = []
    for (phase, task, seed), samples in sorted(grouped.items()):
        td = mean(float(row["td_ms"]) for row in samples)
        tkpr = mean(float(row["td_tkpr_ms"]) for row in samples)
        rows.append(dict(phase=phase, task=task, seed=seed, snapshots=len({r["snapshot_sha256"] for r in samples}),
                         pairs=len(samples), td_mean_ms=td, td_tkpr_mean_ms=tkpr, extra_mean_ms=tkpr - td,
                         relative_overhead_pct=100 * (tkpr - td) / td))
    return rows


def plot_curves(out, curves):
    if not curves:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    seeds = sorted({row["seed"] for row in curves})
    fig, axes = plt.subplots(len(seeds), 3, squeeze=False, sharey=True, figsize=(12, 3.2 * len(seeds)))
    colors = dict(zip(("Shared S", *METHODS), ("#555555", "#a5a8b0", "#2267b6", "#36966e", "#b44769", "#d98a22")))
    series = defaultdict(list)
    for row in curves:
        series[row["seed"], row["phase"], row["method"]].append(row)
    handles = {}
    for row_number, seed in enumerate(seeds):
        for phase, task in enumerate(TASKS):
            ax = axes[row_number, phase]
            for method, color in colors.items():
                rows = series[seed, phase, method]
                if not rows:
                    continue
                line, = ax.plot([r["phase_steps"] / 1e6 for r in rows], [r["win_pct"] for r in rows],
                                color=color, linewidth=1.15, alpha=.9, label=method)
                last = rows[-1]
                ax.scatter(last["phase_steps"] / 1e6, last["win_pct"], color=color, s=15, zorder=3)
                handles[method] = line
            ax.set_title(f"Seed {seed} · {task.capitalize()}")
            ax.set_ylim(0, 100)
            ax.grid(axis="y", alpha=.2)
            if phase == 0:
                ax.set_ylabel("Greedy win rate (%)")
            if row_number == len(seeds) - 1:
                ax.set_xlabel("Stage environment steps (M)")
    fig.legend(handles.values(), handles.keys(), loc="upper center", frameon=False, ncol=len(handles))
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    return True


def report(out, tables, plot):
    runs, stages, timing = (tables[name] for name in ("runs", "stage_metrics", "timing_summary"))
    branches = [row for row in runs if row["role"] == "shared-branch"]
    prefixes = [row for row in runs if row["role"] == "shared-prefix"]
    completed = {row["run_id"] for row in runs if row["status"] == "completed" and row["exit_code"] == 0}
    completed_branches = {row["run_id"] for row in branches} & completed
    completed_prefixes = {row["run_id"] for row in prefixes} & completed
    matched = sum(bool(row["restore_model_hash"]) and row["restore_model_hash"] == row["prefix_model_hash"]
                  for row in branches)
    lines = [
        "# SZC 双层 VDN64 · TKPR-K1 共同前缀对照", "",
        f"已完成 {len(completed_prefixes)}/{len(prefixes)} 个 S 前缀和 "
        f"{len(completed_branches)}/{len(branches)} 个 Z→C 分支。"
        f"{matched}/{len(branches)} 个分支记录的恢复模型指纹与父运行一致。", "",
        "镜像 5v5 追猎者 → 狂热者 → 巨像，累计阈值 2M／4M／6M，逐任务 ε 和 FT。"
        "每个 seed 的五个分支共用 S 末的在线／目标网络、RMSprop 状态、计数和环境分布随机状态。"
        "偏好分支共用 S 阶段的冻结旧 Q 胜局记忆。Z 末各自采集记忆，C 阶段的 Z 来源因此可以不同。"
        "分支重启 SC2 进程。配对以 seed 和父运行 ID 为键。", "",
        "## 逐种子结果", "",
        "| 方法 | Seed | C 末 S/Z/C 胜率 % | ACC % | BWT pp | FWT pp | Z AUC % | C AUC % |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    final = {(x["run_id"], x["phase"]): x for x in stages}
    def show(value):
        return "—" if value is None else f"{value:.2f}"

    for run in branches:
        end = final.get((run["run_id"], 2))
        z = final.get((run["run_id"], 1))
        if end is None or run["run_id"] not in completed_branches:
            continue
        win = " / ".join(show(end.get(f"{task}_win_pct")) for task in TASKS)
        numbers = [end[key] for key in END_METRICS[:3]]
        numbers.extend([z["learning_auc_pct"] if z else None, end["learning_auc_pct"]])
        lines.append(f"| {run['method']} | {run['seed']} | {win} | " +
                     " | ".join(show(value) for value in numbers) + " |")
    lines += ["", "### 同种子配对差值", "",
              "| 方法 | 对照 | Seed | ΔACC pp | ΔBWT pp | ΔFWT pp | ΔC AUC pp |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    comparisons = defaultdict(list)
    for row in tables["paired_metrics"]:
        if row["phase"] == 2:
            comparisons[row["method"], row["control_method"]].append(row)
    for (method, control), rows in comparisons.items():
        seeds = ", ".join(str(seed) for seed in sorted({row["seed"] for row in rows}))
        means = []
        for metric in END_METRICS:
            vals = [row[f"{metric}_delta"] for row in rows if row[f"{metric}_delta"] is not None]
            means.append(show(mean(vals) if vals else None))
        lines.append(f"| {method} | {control} | {seeds} | " + " | ".join(means) + " |")
    lines += ["", "AUC 按实际阶段步数，将当前任务的起点、定期点和阶段末胜率做梯形积分后归一化。"
              "定期点使用 32 局，阶段末使用 128 局。跨种子汇总使用训练 seed 等权。", ""]
    if plot:
        lines += ["![分阶段当前任务学习曲线](learning_curves.png)", ""]
    lines += ["## 配对更新时间", "",
              "两组每次使用同一在线／目标网络、优化器状态和已搬运到 GPU 的 TD batch。"
              "每个快照分别预热两个条件，交替测量顺序。计时前后同步 GPU。"
              "TD+TKPR-K1 包含历史采样、历史前向、损失、联合反传和优化步骤。"
              "输出不计文件写入、环境 rollout、TD 采样和梯度诊断。", "",
              "| 阶段 | 快照数 | 配对数 | TD ms/update | TD+K1 ms/update | 额外 ms/update | 额外比例 % |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in timing:
        if row["seed"] == "all":
            lines.append(f"| {row['task']} | {row['snapshots']} | {row['pairs']} | " +
                         " | ".join(show(row[key]) for key in ("td_mean_ms", "td_tkpr_mean_ms", "extra_mean_ms",
                                                               "relative_overhead_pct")) + " |")
    if not timing:
        lines.append("| 待计时或计时失败 | 0 | 0 | — | — | — | — |")
    lines += ["", "## 数据", "",
              "- [运行清单](runs.csv)与[配置快照](configs/)提供父子关系和实际训练版本。"
              "S 的原始评价仅保存在父运行下，计算整个分支的阶段指标时按父运行关联。",
              "- [逐次评价](evaluations.csv)、[学习曲线](learning_curves.csv)、"
              "[阶段指标](stage_metrics.csv)和[同前缀差值](paired_metrics.csv)。"
              "差值为方法减对照，ACC／BWT／FWT／AUC 的差值单位均为百分点。",
              "- [偏好与机制](preference_diagnostics.csv)和[梯度](gradient_diagnostics.csv)。"
              "置信度、HR 损失及 regret 的汇总使用对应状态数作为权重，计数包含历史重复采样。",
              "- [计时汇总](timing_summary.csv)记录分阶段和分种子结果。"]
    if (out / "timing_samples.csv").exists():
        lines.append("- [逐次配对计时](timing_samples.csv)和[计时配置](timing_protocol.json)。")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def export(plan_file):
    out = plan_file.parent
    config_dir = out / "configs"
    config_dir.mkdir(exist_ok=True)
    plan = read(plan_file)
    tasks = [task for task in plan["tasks"] if task["type"] == "training"]
    tables = {name: [] for name in FIELDS}
    evaluations = {}
    for task in tasks:
        run, metadata = Path(task["run_dir"]), task["metadata"]
        identity = dict(run_id=run.name, method=metadata["method"], seed=metadata["seed"],
                        role=metadata["role"], parent_run_id=metadata.get("parent_run_id"))
        own = list(records(run / "evaluations.jsonl"))
        evaluations[run.name] = own
        tables["evaluations"].extend(dict(identity, **{key: row[key] for key in
                                     ("phase", "task", "label", "t_env", "wins", "episodes")}) for row in own)
        state, launcher, restore = (read(run / key) for key in
                                     ("status.json", "launcher_status.json", "branch_restore.json"))
        config, protocol, launch = (read(run / key) for key in ("config.json", "protocol.json", "launch.json"))
        config_name = None
        if config:
            config_name = f"configs/{run.name}.json"
            snapshot = dict(run_id=run.name, launch_git_commit=launch.get("git_commit"), config=config,
                            protocol=protocol, env_info=read(run / "env_info.json"), branch_restore=restore)
            text = json.dumps(snapshot, ensure_ascii=False, indent=2)
            (out / config_name).write_text(text.replace(task["cwd"] + "/", "") + "\n")
        steps = state.get("t_env")
        replay, epsilon = protocol.get("replay_on_switch"), protocol.get("epsilon_on_switch")
        tables["runs"].append(dict(identity, commit=launch.get("git_commit"),
                                   status=state.get("status", "no_run_artifacts"), exit_code=launcher.get("exit_code"),
                                   completed_global_steps=steps,
                                   trained_steps=steps - restore.get("t_env", 0) if steps is not None else None,
                                   restore_model_hash=restore.get("model_hash"),
                                   prefix_model_hash=restore.get("parent_model_hash"), config=config_name,
                                   replay_on_switch=replay, epsilon_on_switch=epsilon))
        lineage = [*evaluations.get(identity["parent_run_id"], []), *own]
        metrics, curves = stage_rows(identity, read(run / "retention.json"), lineage, own)
        tables["stage_metrics"].extend(metrics)
        tables["learning_curves"].extend(curves)
        tables["preference_diagnostics"].extend(preference_rows(identity, run, protocol.get("method")))
        tables["gradient_diagnostics"].extend(gradient_rows(identity, run))
    tables["paired_metrics"] = list(paired_rows(tables["stage_metrics"], tables["runs"]))
    tables["timing_summary"] = timing_rows(out / "timing_samples.csv")
    for name, data in tables.items():
        write_csv(out, name, data)
    report(out, tables, plot_curves(out, tables["learning_curves"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    export(parser.parse_args().plan)
