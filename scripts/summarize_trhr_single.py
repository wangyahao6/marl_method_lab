"""Create figures, seed results and a mechanism report for paired TRHR runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def curve(run):
    data = {r["t_env"]: r["value"] for r in rows(run / "metrics.jsonl")
            if r["key"] == "test_battle_won_mean"}
    steps = sorted(data)
    return np.asarray(steps), np.asarray([data[k] for k in steps])


def sustained(data):
    x, y = data
    return next((int(x[i]) for i in range(2, len(y)) if np.all(y[i-2:i+1] >= .5)), None)


def mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def stats(windows, unit):
    out = {"activation": mean([r["activation_rate"][unit] for r in windows if r["activation_rate"] is not None])}
    for key in ("gradient_l2", "update_l2", "gradient_nonzero_fraction", "update_nonzero_fraction"):
        for i, side in enumerate(("input", "gru")):
            out[key + "_" + side] = mean([r[key][i][unit] for r in windows])
    return out


def matched(windows, targets):
    steps = np.asarray([w["t_env"] for w in windows])
    return [windows[int(np.argmin(abs(steps - r["t_env"])))] for r in targets] if len(steps) else []


def plot(curves, trace, out):
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 4.3*n), 4), sharey=True, squeeze=False)
    for ax, (seed, groups) in zip(axes[0], curves.items()):
        for group, (x, y) in groups.items():
            if len(x):
                ax.plot(x / 1e6, 100*y, label=group, linewidth=1.15)
        ax.set(title=f"Seed {seed}", xlabel="Training environment steps (M)", ylim=(-2, 102))
        ax.grid(alpha=.25)
    axes[0, 0].set_ylabel("Periodic greedy win rate (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(out / "performance.png", dpi=170)
    plt.close(fig)

    panels = (
        ("activation_rate", None, "Activation rate (%)", 100),
        ("offset", None, "Bias compensation", 1),
        ("gradient_l2", 0, "Incoming gradient L2", 1),
        ("gradient_l2", 1, "GRU input gradient L2", 1),
        ("update_l2", 0, "Incoming update L2", 1),
        ("update_l2", 1, "GRU input update L2", 1),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (key, index, title, scale) in zip(axes.flat, panels):
        ax.set(title=title, xlabel="Training environment steps (M)")
        ax.grid(alpha=.25)
        for color, (seed, (units, groups)) in enumerate(trace.items()):
            if not units:
                continue
            for group, style in (("baseline", "--"), ("trhr", "-")):
                available = [r for r in groups[group] if r[key] is not None]
                if available:
                    values = [np.mean(np.asarray(r[key] if index is None else r[key][index])[units]) * scale
                              for r in available]
                    ax.plot([r["t_env"]/1e6 for r in available], values, linestyle=style,
                            label=f"{seed} {group}", color=f"C{color}", linewidth=1)
        if key == "activation_rate":
            ax.axhline(5, color="grey", lw=.8, ls=":")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3)
    else:
        fig.text(.5, .96, "No adjusted units in available evaluated pairs", ha="center")
    fig.tight_layout(rect=(0, 0, 1, .89))
    fig.savefig(out / "mechanism.png", dpi=170)
    plt.close(fig)


def pct(value):
    return "N/A" if value is None else f"{100*value:.2f}%"


def difference(result, a, b):
    rates = result["independent"]
    if a in rates and b in rates:
        return 100*(rates[a]["win_rate"] - rates[b]["win_rate"])
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    plan_path = p.parse_args().plan
    out = plan_path.parent
    plan = read(plan_path)
    results, units, curves, plot_rows = {}, [], {}, {}
    for pair in plan["pairs"]:
        seed = pair["seed"]
        runs = {g: Path(pair[g]) for g in ("baseline", "trhr")}
        evaluation = Path(pair["evaluation"])
        final = read(evaluation / "final.json").get("conditions", {})
        selection = read(evaluation / "selection.json")
        adjusted, recovered = (selection.get(k, {}) for k in ("adjusted", "recovered"))
        fixed = read(evaluation / "fixed_replay_stats.json")
        trace = {g: rows(r / "a2_windows.jsonl") for g, r in runs.items()}
        curves[seed] = {g: curve(r) for g, r in runs.items()}
        plot_rows[seed] = (list(map(int, adjusted)), trace)
        actual = {g: read(r / "training_complete.json").get("t_env") for g, r in runs.items()}
        hashes = {g: (r / "initial_parameters.sha256").read_text().strip()
                  for g, r in runs.items() if (r / "initial_parameters.sha256").exists()}
        result = dict(actual_steps=actual, initial_parameter_hashes=hashes,
                      initialization_match=(len(set(hashes.values())) == 1) if len(hashes) == 2 else None,
                      steps_to_three_tests_ge50={g: sustained(data) for g, data in curves[seed].items()},
                      adjusted=len(adjusted) if selection else None,
                      recovered=len(recovered) if selection else None, independent=final)
        if all(len(x) >= 2 for x, y in curves[seed].values()):
            low = max(int(x[0]) for x, y in curves[seed].values())
            high = min(int(x[-1]) for x, y in curves[seed].values())
            if high > low:
                points = np.linspace(low, high, 1001)
                result["auc_range_steps"] = [low, high]
                result["normalized_win_auc"] = {
                    g: float(np.trapz(np.interp(points, x, y), points)/(high-low))
                    for g, (x, y) in curves[seed].items()
                }
        results[seed] = result

        for key, first in adjusted.items():
            unit = int(key)
            wake = recovered.get(key)
            pre = [r for r in trace["trhr"] if r["t_env"] <= first][-3:]
            post = [r for r in trace["trhr"] if r["t_env"] <= wake][-3:] if wake else []
            entry = dict(seed=seed, unit=unit, first_adjustment=first, sustained_recovery=wake,
                         final_offset=trace["trhr"][-1]["offset"][unit])
            for label, window in (("trhr_pre", pre), ("trhr_post", post),
                                  ("baseline_pre", matched(trace["baseline"], pre)),
                                  ("baseline_post", matched(trace["baseline"], post))):
                entry.update({label + "_" + k: v for k, v in stats(window, unit).items()})
            for setting, metrics in fixed.items():
                for metric in ("activation_rate", "std"):
                    entry[f"{setting}_{metric}"] = metrics[metric][unit]
            units.append(entry)

    plot(curves, plot_rows, out)
    if units:
        keys = list(dict.fromkeys(k for entry in units for k in entry))
        with (out / "units.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(units)
    (out / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    lines = [
        "# TRHR 单任务实验一", "",
        f"原版 SMAC 5m_vs_6m，两层前馈 VDN，{len(results)} 组配对训练。独立终点评价每条件 128 局，四个环境种子块。", "",
        "| 种子 | 实际步数 B / T | 初始化相同 | 调节 / 唤醒单元 | 独立终点胜率 B / T | 胜率 AUC B / T | 连续三次测试 ≥50% 的步数 B / T |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    final_deltas, auc_deltas, ablation_deltas = [], [], []
    for seed, r in results.items():
        b, t = (r["independent"].get(g, {}).get("win_rate") for g in ("baseline", "trhr"))
        auc = r.get("normalized_win_auc", {})
        steps = r["steps_to_three_tests_ge50"]
        final_deltas.append(difference(r, "trhr", "baseline"))
        ablation_deltas.append(difference(r, "trhr_mean_control", "trhr_mean_recovered"))
        if auc:
            auc_deltas.append(100*(auc["trhr"]-auc["baseline"]))
        lines.append(f'| {seed} | {r["actual_steps"]["baseline"]} / {r["actual_steps"]["trhr"]} | '
                     f'{r["initialization_match"]} | {r["adjusted"]} / {r["recovered"]} | {pct(b)} / {pct(t)} | '
                     f'{pct(auc.get("baseline"))} / {pct(auc.get("trhr"))} | '
                     f'{steps["baseline"] or "N/A"} / {steps["trhr"] or "N/A"} |')
    links = "[胜率曲线](performance.png) · [A2 曲线](mechanism.png) · [逐种子结果](results.json)"
    if units:
        links += " · [逐单元表](units.csv)"
    lines += ["", links, "", "A2 图在各个种子内对最终被调节单元取平均，基线用同编号作描述性参照。"
              "唤醒指首次补偿后连续三个完整窗口的实测激活率达到 5%。"]
    final_deltas = [x for x in final_deltas if x is not None]
    if final_deltas:
        lines += ["", f"终点配对差值（TRHR 减基线）为 {', '.join(f'{x:+.2f}' for x in final_deltas)} 个百分点，"
                  f"已完成 {len(final_deltas)}/{len(results)} 对，均值 {np.mean(final_deltas):+.2f} 个百分点。"]
    if auc_deltas:
        lines += ["", f"共同覆盖区间的归一化胜率 AUC 配对差均值为 {np.mean(auc_deltas):+.2f} 个百分点。"]
    if any(entry["sustained_recovery"] is not None for entry in units):
        lines += ["", "以下按种子汇总唤醒单元在首次补偿前及达到唤醒时的三个窗口均值。"
                  "两类梯度和更新依次对应 A2 输入行及 GRU 三门输入列。", "",
                  "| 种子 | 唤醒 | 梯度 L2 前 → 后 | 实际更新 L2 前 → 后 | 固定回放最终激活率 有补偿 / 去补偿 | 唤醒 / 对照均值替换胜率 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for seed, r in results.items():
            selected = [u for u in units if u["seed"] == seed and u["sustained_recovery"] is not None]
            if not selected:
                continue
            def change(key):
                vals = [[mean([u[f"trhr_{period}_{key}_{side}"] for u in selected]) for side in ("input", "gru")]
                        for period in ("pre", "post")]
                return " → ".join("/".join(f"{v:.3g}" if v is not None else "N/A" for v in x) for x in vals)
            rates = " / ".join(pct(mean([u.get(k) for u in selected])) for k in
                               ("trhr_final_activation_rate", "trhr_final_zero_offset_activation_rate"))
            ablation = " / ".join(pct(r["independent"].get(k, {}).get("win_rate")) for k in
                                  ("trhr_mean_recovered", "trhr_mean_control"))
            lines.append(f'| {seed} | {len(selected)} | {change("gradient_l2")} | {change("update_l2")} | {rates} | {ablation} |')
    if mean(ablation_deltas) is not None:
        lines += ["", f"相对匹配对照，均值替换唤醒单元后的胜率差平均为 {mean(ablation_deltas):+.2f} 个百分点"
                  "（对照胜率减唤醒单元替换胜率）。固定回放均值替换用于考察输入相关变化对终点策略的贡献。"]
    lines += ["", "训练回放分布会随策略改变，逐单元基线编号仅为同初始化参照。"
              "激活、梯度和更新、固定回放的输入响应、独立胜率应分别判断。"]
    if len(results) == 1:
        lines += ["", "本轮只有种子 1234 的配对结果，种子间稳定性无法评估。"]
    if len(final_deltas) < len(results):
        lines += ["", "部分训练或独立评价暂无完整结果。缺失项不计为零。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
