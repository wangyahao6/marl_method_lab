"""Export and audit ASPR-3G-Adjacent versus matched gap0 TKPR-K1; never trains."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from summarize_szc_tkpr_k1_shared import read, records, stage_rows, preference_rows, END_METRICS, TASKS


def csv_write(out, name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def show(x):
    return "N/A" if x is None else f"{x:.2f}"


def export(plan_file):
    plan = read(plan_file)
    out = plan_file.parent
    (out / "configs").mkdir(exist_ok=True)
    tables = defaultdict(list)
    manifest = []
    parents = {}

    def capture(path):
        if path.exists():
            manifest.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def collect(run, identity, parent=None):
        own = list(records(run / "evaluations.jsonl"))
        all_eval = list(records(parent / "evaluations.jsonl")) + own if parent else own
        retention = read(run / "retention.json")
        stages, curves = stage_rows(identity, retention, all_eval, own)
        tables["stage_metrics"].extend(stages)
        tables["learning_curves"].extend(curves)
        tables["evaluations"].extend(dict(identity, **{key: row[key] for key in
            ("phase", "task", "label", "t_env", "wins", "episodes")}) for row in own)
        status, launch, launcher, restore, protocol, config = [read(run / name) for name in
            ("status.json", "launch.json", "launcher_status.json", "branch_restore.json", "protocol.json", "config.json")]
        snapshot = dict(identity, launch_git_commit=launch.get("git_commit"), protocol=protocol,
                        config=config, branch_restore=restore, env_info=read(run / "env_info.json"))
        config_path = f"configs/{run.name}.json"
        (out / config_path).write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
        parent_state = read(parent / "status.json") if parent else {}
        tables["runs"].append(dict(identity, commit=launch.get("git_commit"), status=status.get("status"),
            exit_code=launcher.get("exit_code"), replay_on_switch=protocol.get("replay_on_switch"),
            epsilon_on_switch=protocol.get("epsilon_on_switch"), optimizer=protocol.get("optimizer", "rmsprop"),
            min_gap=protocol.get(protocol.get("method"), {}).get("min_gap"),
            completed_global_steps=status.get("t_env"),
            trained_steps=(status["t_env"] - parent_state.get("t_env", 0)) if "t_env" in status else None,
            config=config_path, restore_model_hash=restore.get("model_hash"),
            prefix_model_hash=restore.get("parent_model_hash")))
        for name in ("status.json", "launch.json", "launcher_status.json", "branch_restore.json",
                     "protocol.json", "retention.json", "evaluations.jsonl"):
            capture(run / name)
        if parent:
            method = protocol.get("method")
            tables["preference_diagnostics"].extend(preference_rows(identity, run, method))
            for filename in ("aspr_group_summary", "aspr_group_training_sampled",
                             "aspr_adjacent_training_pairs", "aspr_adjacent_offline_pairs", "aspr_cross_group_offline"):
                path = run / f"{filename}.csv"
                if path.exists():
                    with path.open() as stream:
                        tables[filename].extend(dict(identity, **row) for row in csv.DictReader(stream))
                    capture(path)
            # The separate evaluator reuses the final greedy evaluation seed blocks.
            for path in sorted((run / "evaluation").glob("*/results.json")):
                result = read(path)
                evaluation_status = read(path.parent / "launcher_status.json")
                for row in result.get("tasks", []):
                    tables["independent_final"].append(dict(identity, task=row["task"], wins=row["wins"],
                        episodes=row["episodes"], win_pct=100 * row["wins"] / row["episodes"],
                        exit_code=evaluation_status.get("exit_code"), t_env=result["t_env"]))
                capture(path)
                capture(path.parent / "launcher_status.json")
        return stages

    for task in plan["tasks"]:
        if task["type"] != "training":
            continue
        meta = task["metadata"]
        parent = Path(task["argv"][task["argv"].index("--parent") + 1])
        if parent.name not in parents:
            identity = dict(run_id=parent.name, method="Shared S", seed=meta["seed"],
                            role="shared-prefix", parent_run_id=None)
            parents[parent.name] = collect(parent, identity)
        run = Path(task["run_dir"])
        identity = dict(run_id=run.name, method=meta["method"], seed=meta["seed"],
                        role="shared-branch", parent_run_id=parent.name)
        collect(run, identity, parent)
    branches = {(r["method"], r["seed"], r["phase"]): r for r in tables["stage_metrics"] if r["parent_run_id"]}
    for (method, seed, phase), row in branches.items():
        if method != "ASPR-3G-Adjacent":
            continue
        control = branches.get(("TKPR-K1", seed, phase))
        if control:
            tables["paired_metrics"].append(dict(method=method, control_method="TKPR-K1", seed=seed,
                phase=phase, training_task=row["training_task"], run_id=row["run_id"],
                control_run_id=control["run_id"], parent_run_id=row["parent_run_id"],
                **{f"{k}_delta": row[k] - control[k] if row[k] is not None and control[k] is not None else None
                   for k in (*END_METRICS, *(f"{t}_win_pct" for t in TASKS[:phase + 1]))}))
    audits = []
    for row in tables["runs"]:
        if not row["parent_run_id"]:
            continue
        final = branches.get((row["method"], row["seed"], 2))
        independent = [r for r in tables["independent_final"] if r["run_id"] == row["run_id"]]
        match = bool(final and len(independent) == 3 and all(r["exit_code"] == 0 and
            abs(r["win_pct"] - final[f'{r["task"]}_win_pct']) < 1e-8 for r in independent))
        audits.append(dict(run_id=row["run_id"], completed=row["status"] == "completed",
            launcher_success=row["exit_code"] == 0, final_matrix_present=bool(final),
            restored_model_matches_prefix=bool(row["restore_model_hash"] and
                row["restore_model_hash"] == row["prefix_model_hash"]), independent_final_matches=match))
    tables["audit"] = audits
    for name, rows in tables.items():
        csv_write(out, name, rows)
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# SZC VDN64 · ASPR-3G-Adjacent vs TKPR-K1", "",
        "协议：Stalker→Zealot→Colossus，各阶段 2M 环境步（累计阈值后结束完整回合）；"
        "VDN 双前馈层/GRU64、RMSprop、FT、逐任务 ε；seed 2222、3535。",
        "同种子复用同一条已完成的 S 前缀及历史胜局池，Z/C 各自采集历史数据。"
        "两方法 min_gap=0、temperature=1、weight=1、q_scale_floor=0.1，使用相同的归一化 Softplus 与损失聚合。",
        "ASPR-3G-Adjacent 只选择 G1-G2、G2-G3；聚类完全复用确定性一维 K-means。"
        "K1 仅选择 Top-1 端点对；gap0 独立入口调用原 K1 损失。原有 2M K1 的 gap0.5 结果不作本次对照。", "",
        "## 运行核验", "", "| 方法 | Seed | 状态 | 退出码 | 恢复哈希匹配 | 终局独立评价一致 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for row, audit in zip([r for r in tables["runs"] if r["parent_run_id"]], audits):
        lines.append(f'| {row["method"]} | {row["seed"]} | {row["status"]} | {row["exit_code"]} | '
                     f'{audit["restored_model_matches_prefix"]} | {audit["independent_final_matches"]} |')
    lines += ["", "## 各阶段结果", "", "| 方法 | Seed | 阶段 | S % | Z % | C % | ACC % | BWT pp | FWT pp | 当前任务 AUC % |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in tables["stage_metrics"]:
        lines.append(f'| {row["method"]} | {row["seed"]} | {row["training_task"]} | ' +
            " | ".join(show(row.get(k)) for k in (*(f"{t}_win_pct" for t in TASKS), *END_METRICS)) + " |")
    lines += ["", "## 同种子配对差值", "", "差值均为 ASPR-3G-Adjacent 减 TKPR-K1，单位 pp。", "",
              "| Seed | 阶段 | ΔACC | ΔBWT | ΔFWT | ΔAUC |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in tables["paired_metrics"]:
        lines.append(f'| {row["seed"]} | {row["training_task"]} | ' +
            " | ".join(show(row[f"{k}_delta"]) for k in END_METRICS) + " |")
    lines += ["", "## 诊断与解释", "",
        "[实际采样对统计](aspr_adjacent_training_pairs.csv)按训练阶段和历史来源任务累计三类跨组对；"
        "[组大小与占比](aspr_group_training_sampled.csv)计入重复采样。组大小分布行重复展示动作对总计，不能逐行相加。",
        "G1-G3 只做离线排序诊断，不参与损失；其 `valid_pairs` 是可统计对数，"
        "不计入训练约束总数。违反率分母为对应有效对数，当前 Q 平局也计为违反。"
        "[固定探测轨迹](aspr_adjacent_offline_pairs.csv)是独立统计口径，沿用已有探测机制，不与训练采样混合。",
        "[偏好诊断](preference_diagnostics.csv)提供两方法实际有效对数和按对数加权的违反率。", ""]
    if tables["aspr_cross_group_offline"]:
        lines += ["[全量历史胜局离线对照](aspr_cross_group_offline.csv)对两方法阶段末模型重放各自全部历史池；"
                  "S 池两方法相同，Z 池来自各自分支。该表不进行更新，也不进入损失。", "",
                  "| 方法 | Seed | 阶段 | 历史任务 | G1-G3 有效对数 | G1-G3 违反率 % |",
                  "| --- | --- | --- | --- | ---: | ---: |"]
        for row in tables["aspr_cross_group_offline"]:
            if row["pair"] == "G1-G3":
                rate = float(row["pair_violation_rate"]) * 100 if row["pair_violation_rate"] else None
                lines.append(f'| {row["method"]} | {row["seed"]} | {row["training_task"]} | '
                             f'{row["historical_task"]} | {row["valid_pairs"]} | {show(rate)} |')
    final_deltas = [r for r in tables["paired_metrics"] if r["phase"] == 2]
    if len(final_deltas) == 2:
        for method in ("ASPR-3G-Adjacent", "TKPR-K1"):
            rows = [r for r in tables["stage_metrics"] if r["phase"] == 2 and r["method"] == method]
            lines.append(f'{method} 两种子均值：' + "、".join(f'{k}={show(mean(r[k] for r in rows))}' for k in END_METRICS) + "。")
        lines += ["", "相邻组相对 K1 的最终平均差值：" + "、".join(
            f'{k}={show(mean(r[f"{k}_delta"] for r in final_deltas))}' for k in END_METRICS) + "。",
            "新任务学习需结合 Z/C 当前任务 AUC 与端点，旧任务保持需结合 S/Z 终点与 BWT；"
            "不能仅由动作对数量或违反率推断保护效果。仅两个训练种子，不作一般性优劣或显著性结论。"]
        lines += ["", "逐种子观察（方法减对照）："]
        for seed in (2222, 3535):
            z = next(r for r in tables["paired_metrics"] if r["seed"] == seed and r["phase"] == 1)
            c = next(r for r in final_deltas if r["seed"] == seed)
            lines.append(f'Seed {seed}：Z 当前任务 AUC {show(z["learning_auc_pct_delta"])} pp、'
                         f'Z 末胜率 {show(z["zealot_win_pct_delta"])} pp；'
                         f'C 当前任务 AUC {show(c["learning_auc_pct_delta"])} pp、'
                         f'C 末胜率 {show(c["colossus_win_pct_delta"])} pp；'
                         f'C 末 S 保持 {show(c["stalker_win_pct_delta"])} pp、'
                         f'Z 保持 {show(c["zealot_win_pct_delta"])} pp、BWT {show(c["bwt_pp_delta"])} pp。')
        for phase in (1, 2):
            deltas = [r["learning_auc_pct_delta"] for r in tables["paired_metrics"] if r["phase"] == phase]
            direction = ("两种子均更高" if all(v > 0 for v in deltas) else
                         "两种子均更低" if all(v < 0 for v in deltas) else "两种子未显示同方向改善")
            lines.append(f'{TASKS[phase]} 新任务学习 AUC：{direction}。')
        offline = {(r["method"], int(r["seed"]), int(r["phase"]), int(r["source_phase"])): r
                   for r in tables["aspr_cross_group_offline"] if r["pair"] == "G1-G3"}
        lines += ["", "G1-G3 的阶段末离线违反情况："]
        for key, row in sorted(offline.items()):
            method, seed, phase, source = key
            if method != "ASPR-3G-Adjacent":
                continue
            control = offline.get(("TKPR-K1", seed, phase, source))
            if control and row["pair_violation_rate"] and control["pair_violation_rate"]:
                a, b = float(row["pair_violation_rate"]), float(control["pair_violation_rate"])
                lines.append(f'Seed {seed}、{TASKS[phase]} 末、历史 {TASKS[source]}：'
                             f'Adjacent {100*a:.2f}%（{row["pair_violations"]}/{row["valid_pairs"]}），'
                             f'K1 {100*b:.2f}%（{control["pair_violations"]}/{control["valid_pairs"]}），'
                             f'差 {100*(a-b):+.2f} pp。')
        lines.append("若同一状态的两类相邻组排序全部严格成立，G1-G3 顺序可由传递性成立；"
                     "Softplus 训练并不保证每个相邻对都满足，因此应以实际离线违反数判断跨层级问题，"
                     "不能仅根据掩码排除了 G1-G3 就断言其排序必然失效。S 池可直接按同样本比较，Z 池还包含分支采集差异。")
    else:
        lines.append("实验尚未全部完成；当前表只包含已有评价，不能据此宣称方法优劣。")
    lines += ["", "## 数据与计算口径", "",
        "[运行与版本](runs.csv)、[协议快照](configs/)、[源文件校验](source_manifest.json)、[核验表](audit.csv)。",
        "[阶段指标](stage_metrics.csv)、[配对差值](paired_metrics.csv)、[逐次评价](evaluations.csv)、"
        "[学习曲线](learning_curves.csv)、[终局独立评价](independent_final.csv)。",
        "ACC 为已见任务等权胜率；BWT 为旧任务相对学完时的变化；FWT 使用各自首次训练前评价减初始模型表现。"
        "AUC 按实际步数对当前任务贪心胜率作梯形积分并除以阶段长度，定期评价32局、阶段末128局。"
        "终局独立评价对每个任务使用相同的128局种子块，只用于核验，不作为新增训练种子。",
        "原始日志、检查点和历史池留在 rl-navigation-ubuntu:/home/wyh/marl_method_lab/runs/；"
        "本批次新增四条 Z/C 分支，名义新增训练16M步，S前缀仅引用一次。", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    export(parser.parse_args().plan)
