"""Export the finished R2/R3 runs and audit the failed in-batch K1 controls."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from summarize_szc_tkpr_k1_shared import records, stage_rows


TASKS = ("stalker", "zealot", "colossus")
BATCH = "szc_tkpr_rank_position_20261120"
METHODS = {"tkpr_k1": "TKPR-K1", "tkpr_r2": "TKPR-R2", "tkpr_r3": "TKPR-R3"}


def load(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(out, name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("empty table: " + name)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ratio(num, den):
    return num / den if den else None


def run_export(root, out):
    plan = load(root / "results" / BATCH / "plan.json")
    if len(plan["tasks"]) != 6:
        raise ValueError("expected six in-batch tasks")
    old = root / "docs/results/comparisons/szc-vdn64-tkpr-k1-shared"
    old_runs = {int(r["seed"]): r for r in csv_rows(old / "runs.csv")
                if r["method"] == "TKPR-K1" and int(r["seed"]) in (2222, 3535)}
    old_stages = {(int(r["seed"]), int(r["phase"])): r for r in csv_rows(old / "stage_metrics.csv")
                  if r["method"] == "TKPR-K1" and int(r["seed"]) in (2222, 3535)}
    if len(old_runs) != 2 or len(old_stages) != 4:
        raise ValueError("missing historical K1 controls")
    out.mkdir(parents=True, exist_ok=True)
    (out / "configs").mkdir(exist_ok=True)
    run_rows, eval_rows, stages, curves, preferences, gradients = ([] for _ in range(6))
    expected = {"tkpr_k1": "failed", "tkpr_r2": "completed", "tkpr_r3": "completed"}
    for task in plan["tasks"]:
        run_id = task["id"]
        path = Path(task["run_dir"])
        label = next(key for key in METHODS if ("_" + key + "_seed") in run_id)
        method = METHODS[label]
        seed = int(task["argv"][task["argv"].index("--seed") + 1])
        parent_id = task["metadata"]["parent_run_id"]
        parent = root / "runs" / parent_id
        status, launcher, launch, protocol = (load(path / name) for name in
                                               ("status.json", "launcher_status.json", "launch.json", "protocol.json"))
        restore = load(path / "branch_restore.json")
        if status["status"] != expected[label] or (launcher["exit_code"] == 0) != (label != "tkpr_k1"):
            raise ValueError("unexpected terminal state: " + run_id)
        if launch["git_commit"] != "c5f86109747c3c6206b68f273ccc40fa48a79498" or launch["git_status"]:
            raise ValueError("launch version differs: " + run_id)
        if restore["model_hash"] != restore["parent_model_hash"] or task["dependencies"] != [parent_id]:
            raise ValueError("parent restore differs: " + run_id)
        if protocol["stages"] != load(parent / "protocol.json")["stages"]:
            raise ValueError("stage protocol differs from parent: " + run_id)
        old_parent = old_runs[seed]["parent_run_id"]
        if old_parent != parent_id:
            raise ValueError("historical K1 uses another parent: " + run_id)
        identity = dict(run_id=run_id, method=method, seed=seed, role="shared-branch",
                        parent_run_id=parent_id, cohort="rank-position-20261120")
        own = list(records(path / "evaluations.jsonl"))
        eval_rows.extend(dict(identity, **{k: row[k] for k in
                                        ("phase", "task", "label", "t_env", "wins", "episodes")}) for row in own)
        if label == "tkpr_k1":
            if status["phase"] != 1 or status["t_env"] >= 4000000:
                raise ValueError("K1 failure location changed: " + run_id)
            failure = "SC2 connection/restart failure in Zealot phase"
        else:
            if status["phase"] != 2 or status["t_env"] < 6000000:
                raise ValueError("branch has not reached full budget: " + run_id)
            retention = load(path / "retention.json")
            parent_evals = [r for r in records(parent / "evaluations.jsonl") if r["phase"] == 0]
            stage, curve = stage_rows(identity, retention, parent_evals + own, own)
            if len(stage) != 2 or len(curve) < 4:
                raise ValueError("incomplete stage metrics: " + run_id)
            for phase in (1, 2):
                ends = [r for r in own if r["phase"] == phase and r["label"] == "stage_end"]
                if {r["task"] for r in ends} != set(TASKS[:phase + 1]) or any(r["episodes"] != 128 for r in ends):
                    raise ValueError("missing independent stage-end evaluation: " + run_id)
            stages.extend(dict(row, cohort=identity["cohort"]) for row in stage)
            curves.extend(dict(row, cohort=identity["cohort"]) for row in curve)
            failure = ""
        run_rows.append(dict(identity, commit=launch["git_commit"], status=status["status"],
                             exit_code=launcher["exit_code"], completed_global_steps=status["t_env"],
                             new_steps=status["t_env"] - restore["t_env"],
                             restore_model_hash=restore["model_hash"],
                             parent_model_hash=restore["parent_model_hash"],
                             failure=failure, run_dir=str(path.relative_to(root)),
                             config="configs/" + run_id + ".json"))
        (out / "configs" / (run_id + ".json")).write_text(json.dumps(
            dict(protocol=protocol, launch_git_commit=launch["git_commit"],
                 source_sha256={k: launch["source_sha256"][k] for k in
                                ("src/learners/ranking_preference.py", "drivers/continual/action_set_train.py",
                                 "drivers/continual/action_set_protocol.py")}, branch_restore=restore),
            ensure_ascii=False, indent=2) + "\n")
        if label == "tkpr_k1":
            continue
        grouped, histories = defaultdict(list), defaultdict(list)
        for row in records(path / (label + "_updates.jsonl")):
            grouped[row["phase"]].append(row)
            for item in row["historical_tasks"]:
                histories[(row["phase"], item["source_phase"], item["task"])].append(item)
        if set(grouped) != {1, 2} or set(source for _, source, _ in histories) != {0, 1}:
            raise ValueError("missing preference update phase: " + run_id)
        for (phase, source, historical), rows in sorted(histories.items()):
            totals = {name: sum(row.get(name, 0) for row in rows) for name in
                      ("decision_states", "valid_states", "candidate_pairs", "eligible_pairs",
                       "filtered_pairs", "valid_pairs", "pair_violations", "anchor_available_states")}
            histogram = Counter()
            for row in rows:
                histogram.update(row["anchor_new_rank_histogram"])
            states = totals["valid_states"]
            if sum(histogram.values()) != states:
                raise ValueError("anchor rank histogram does not cover valid states: " + run_id)
            preferences.append(dict(identity, phase=phase, training_task=TASKS[phase],
                                    cohort=identity["cohort"],
                                    source_phase=source, historical_task=historical, updates=len(rows), **totals,
                                    coverage=ratio(states, totals["decision_states"]),
                                    mean_valid_pairs=ratio(totals["valid_pairs"], states),
                                    pair_filter_rate=ratio(totals["filtered_pairs"], totals["eligible_pairs"]),
                                    pair_violation_rate=ratio(totals["pair_violations"], totals["valid_pairs"]),
                                    anchor_old_rank=2 if label == "tkpr_r2" else 3,
                                    anchor_mean_new_rank=ratio(sum(int(rank)*count for rank,count in histogram.items()),
                                                               states),
                                    anchor_rank_retained_rate=ratio(histogram[str(2 if label == "tkpr_r2" else 3)],
                                                                    states),
                                    anchor_new_rank_histogram=json.dumps(histogram, sort_keys=True),
                                    mean_task_loss=mean(row["loss"] for row in rows if row.get("loss") is not None)))
        counts, values = Counter(), defaultdict(list)
        for row in records(path / "experiment_metrics.jsonl"):
            if row["context"].get("event") != "gradient_monitor":
                continue
            key = row["phase"], row["task"], row["metric"]
            counts[key] += 1
            if row["value"] is not None:
                values[key].append(row["value"])
        for (phase, task_name, metric), n in sorted(counts.items()):
            vals = values[phase, task_name, metric]
            gradients.append(dict(identity, phase=phase, training_task=task_name, metric=metric,
                                  cohort=identity["cohort"],
                                  logged_points=n, defined_points=len(vals),
                                  mean=mean(vals) if vals else None, median=median(vals) if vals else None))

    for seed in (2222, 3535):
        old_run = old_runs[seed]
        run_rows.append(dict(run_id=old_run["run_id"], method="TKPR-K1", seed=seed,
                             role="shared-branch", parent_run_id=old_run["parent_run_id"],
                             cohort="historical-20261117", commit=old_run["commit"], status="completed",
                             exit_code=0, completed_global_steps=old_run["completed_global_steps"],
                             new_steps=old_run["trained_steps"], restore_model_hash=old_run["restore_model_hash"],
                             parent_model_hash=old_run["prefix_model_hash"], failure="",
                             run_dir=str(Path("runs") / old_run["run_id"]),
                             config="../../comparisons/szc-vdn64-tkpr-k1-shared/" + old_run["config"]))
        for phase in (1, 2):
            stages.append(dict(old_stages[seed, phase], cohort="historical-20261117"))
    for row in csv_rows(old / "preference_diagnostics.csv"):
        if row["method"] != "TKPR-K1" or int(row["seed"]) not in (2222, 3535):
            continue
        row["cohort"] = "historical-20261117"
        row["mean_valid_pairs"] = ratio(int(row["valid_pairs"]), int(row["valid_states"]))
        preferences.append(row)
    for row in csv_rows(old / "gradient_diagnostics.csv"):
        if row["method"] == "TKPR-K1" and int(row["seed"]) in (2222, 3535):
            row["cohort"] = "historical-20261117"
            gradients.append(row)
    by_key = {(str(r["seed"]), str(r["phase"]), r["method"]): r for r in stages}
    paired = []
    for seed in (2222, 3535):
        for phase in (1, 2):
            for treatment, control in (("TKPR-R2", "TKPR-R3"), ("TKPR-R2", "TKPR-K1"),
                                       ("TKPR-R3", "TKPR-K1")):
                a = by_key[str(seed), str(phase), treatment]
                b = by_key[str(seed), str(phase), control]
                keys = ("stalker_win_pct", "zealot_win_pct", "colossus_win_pct", "acc_pct",
                        "bwt_pp", "fwt_pp", "learning_auc_pct")
                paired.append(dict(seed=seed, phase=phase, treatment=treatment, control=control,
                                   treatment_run_id=a["run_id"], control_run_id=b["run_id"],
                                   comparison="same batch" if control == "TKPR-R3" else "historical control",
                                   **{key + "_delta": float(a[key]) - float(b[key])
                                      if a.get(key) not in (None, "") and b.get(key) not in (None, "") else None
                                      for key in keys}))
    for name, rows in (("runs", run_rows), ("evaluations", eval_rows), ("stage_metrics", stages),
                       ("learning_curves", curves), ("preference_diagnostics", preferences),
                       ("gradient_diagnostics", gradients), ("paired_metrics", paired)):
        write_csv(out, name, rows)
    print("exported", len(run_rows), "runs (including 2 historical controls),", len(stages), "stage rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_export(args.root, args.out)
