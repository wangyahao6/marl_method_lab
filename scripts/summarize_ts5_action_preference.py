"""Summarize the four seed-3535 TS5 action-preference runs after they terminate."""
import argparse
import json
from pathlib import Path

TASKS = ("A", "B", "C", "D", "E")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(task):
    run = Path(task["run_dir"])
    status = read_json(run / "status.json")
    launcher = read_json(run / "launcher_status.json")
    retention = read_json(run / "retention.json")
    stages = retention.get("stages", [])
    complete = (status.get("status") == "completed" and launcher.get("exit_code") == 0
                and status.get("t_env", 0) >= 10000000 and len(stages) == len(TASKS)
                and all(row.get("task") == name and set(row.get("win_rates", {})) == set(TASKS[:i + 1])
                        for i, (row, name) in enumerate(zip(stages, TASKS))))
    return dict(method=task["metadata"]["branch_id"], seed=3535, run_dir=str(run),
                commit=read_json(run / "launch.json").get("git_commit"),
                status=status, exit_code=launcher.get("exit_code"), complete=complete,
                stages=stages, initial_reference=retention.get("initial_reference", {}),
                before_first_training=retention.get("before_first_training", {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    plan = parser.parse_args().plan
    runs = {task["id"]: summarize(task) for task in read_json(plan)["tasks"]
            if task["type"] == "training"}
    output = plan.with_name("summary.json")
    output.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# TS5 seed 3535 四方法实验结果", "",
             "四组均为独立训练路径，单一训练种子；下表只描述观测值，不作显著性推断。", "",
             "| 方法 | 状态 | 实际步数 | 最终 ACC (%) | BWT (pp) | FWT (pp) |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for run in runs.values():
        final = run["stages"][-1] if run["complete"] else {}
        values = [final.get("acc_percent"), final.get("bwt_pp"), final.get("fwt_pp")]
        metrics = ["—" if value is None else f"{value:.2f}" for value in values]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            run["method"], "completed" if run["complete"] else run["status"].get("status", "missing"),
            run["status"].get("t_env", "—"), *metrics))
    lines.extend(["", "逐阶段评价矩阵、实际配置及源码版本见 `summary.json` 与各 run 目录。",
                  "若某组不完整，保留其原始状态，不自动补跑。", ""])
    plan.with_name("REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
