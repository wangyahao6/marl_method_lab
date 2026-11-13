"""Collect stage evaluation matrices and final metrics for the four SZC methods."""
import argparse
import json
from pathlib import Path
from statistics import mean

TASKS = ("stalker", "zealot", "colossus")
METRICS = ("acc_percent", "bwt_pp", "fwt_pp")


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize_run(task):
    run = Path(task["run_dir"])
    status = read_json(run / "status.json")
    launcher = read_json(run / "launcher_status.json")
    retention = read_json(run / "retention.json")
    if task["metadata"]["method"] == "CHI":
        labels = ("S", "Z", "C")
        matrix = retention.get("stage_win_rates", {})
        metrics = retention.get("metrics", {})
        stages = [
            dict(task=name, win_rates=matrix[label], **metrics[label])
            for name, label in zip(TASKS, labels) if label in matrix
        ]
    else:
        stages = retention.get("stages", [])
    complete = (status.get("status") == "completed" and launcher.get("exit_code") == 0
                and status.get("t_env", 0) >= 6000000 and len(stages) == len(TASKS)
                and all(stage["task"] == name and set(stage["win_rates"]) == set(TASKS[:i+1])
                        for i, (stage, name) in enumerate(zip(stages, TASKS))))
    return dict(method=task["metadata"]["method"], seed=task["metadata"]["seed"],
                run_dir=str(run), commit=read_json(run / "launch.json").get("git_commit"),
                status=status, exit_code=launcher.get("exit_code"), complete=complete,
                stages=stages, initial_reference=retention.get("initial_reference", {}),
                before_first_training=retention.get("before_first_training", {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    plan_path = parser.parse_args().plan
    runs = {task["id"]: summarize_run(task) for task in read_json(plan_path)["tasks"]
            if task["type"] == "training"}
    by_method = {}
    for method in sorted({run["method"] for run in runs.values()}):
        completed = [run for run in runs.values() if run["method"] == method and run["complete"]]
        final = [run["stages"][-1] for run in completed]
        metrics = {}
        for key in METRICS:
            values = [stage[key] for stage in final if stage.get(key) is not None]
            metrics[key] = dict(mean=mean(values) if values else None, n=len(values))
        by_method[method] = dict(completed_seeds=sorted(run["seed"] for run in completed), metrics=metrics)
    output = plan_path.with_name("summary.json")
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(runs=runs, final_by_method=by_method), indent=2, allow_nan=False) + "\n")
    temp.replace(output)


if __name__ == "__main__":
    main()
