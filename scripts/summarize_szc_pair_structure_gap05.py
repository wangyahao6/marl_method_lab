"""Export verified SZC pair-structure gap-0.5 results from finished scheduler runs."""
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from summarize_szc_tkpr_k1_shared import gradient_rows, records, stage_rows

ROOT = Path(os.environ.get("MARL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
import torch as th
from learners.pair_ablation import select_pairs


BATCH = "szc_pair_structure_gap05_3535_20261121"
RUNS = ROOT / "runs" / "pair_structure_gap05"
OUT = Path(os.environ.get("SZC_GAP05_OUT", ROOT / "docs/results/parameter-ablations/szc-vdn64-pair-structure-gap05"))
OLD = ROOT / "docs/results/parameter-ablations/szc-vdn64-pair-structure-ratio"
NAMES = {
    "k1": "TKPR-K1", "adjacent": "Adjacent", "k1_adjacent": "TKPR-K1 + Adjacent",
    "frpr": "FRPR", "random_n_minus_1": "Random-N-1",
    "random_2n_minus_3": "Random-(2N-3)",
    "random_2n_minus_3_matched": "Random-(2N-3)-Matched",
}


def read(path):
    return json.loads(path.read_text())


def ratio(a, b):
    return a / b if b else None


def write(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("empty " + name)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def check_masks(plan, paths, parent, pools):
    """Verify archived masks and compute true transitive-closure coverage."""
    closure_rows = []
    for phase in (0, 1):
        source = th.load(parent / "action_set_memory" / ("phase%04d" % phase) / "success.th",
                         map_location="cpu")["episodes"]
        masks = {task["id"]: th.load(paths[task["id"]] / "pair_ablation_memory" /
                                    ("phase%04d" % phase) / "selection.th", map_location="cpu")
                 for task in plan["tasks"][1:]}
        tally = defaultdict(lambda: defaultdict(int))
        for index, episode in enumerate(source):
            length = episode["length"]
            old = episode["data"]["old_q_values"][:length]
            legal = episode["data"]["avail_actions"][:length].bool()
            decisions = legal[..., 1:].any(dim=-1)
            filtered, valid, _, _ = select_pairs(old, legal, decisions, "frpr", 3535,
                                                 min_gap=.5, q_scale_floor=.1)
            count = legal.sum(dim=-1)
            n = legal.size(-1)
            left, right = th.triu_indices(n, n, 1)
            old_pairs = legal[..., left] & legal[..., right] & (old[..., left] != old[..., right])
            full = old_pairs.sum(dim=-1)
            top = old.masked_fill(~legal, -th.inf).argmax(dim=-1)
            top_pairs = (left == top[..., None]) | (right == top[..., None])
            union_id = BATCH + "_pair_structure_k1_adjacent"
            matched_id = BATCH + "_pair_structure_random_2n_minus_3_matched"
            union, matched = masks[union_id][index], masks[matched_id][index]
            if not th.equal(union & top_pairs, matched & top_pairs):
                raise ValueError("matched Top-1 edges differ at pool %d episode %d" % (phase, index))
            if not th.equal((union & ~top_pairs).sum(-1), (matched & ~top_pairs).sum(-1)):
                raise ValueError("matched non-Top-1 budget differs")
            if not th.equal(union.sum(-1), matched.sum(-1)):
                raise ValueError("matched total budget differs")
            for task in plan["tasks"][1:]:
                run_id = task["id"]
                selected = masks[run_id][index]
                if selected.shape != old_pairs.shape or (selected & ~old_pairs).any():
                    raise ValueError("illegal archived pair: " + run_id)
                if (selected & ~filtered).any():
                    raise ValueError("archived pair failed min_gap: " + run_id)
                if (selected & ~valid[..., None]).any():
                    raise ValueError("archived pair outside common states: " + run_id)
                suffix = run_id[len(BATCH) + len("_pair_structure_"):]
                if suffix == "random_n_minus_1" and not th.equal(selected.sum(-1)[valid], (count-1)[valid]):
                    raise ValueError("Random-N-1 budget differs")
                if suffix == "random_2n_minus_3" and not th.equal(selected.sum(-1)[valid], (2*count-3)[valid]):
                    raise ValueError("Random-(2N-3) budget differs")
                reach = th.zeros((*old.shape[:-1], n, n), dtype=th.bool)
                reach[..., left, right] = selected & (old[..., left] > old[..., right])
                reach[..., right, left] = selected & (old[..., right] > old[..., left])
                for middle in range(n):
                    reach |= reach[..., :, middle, None] & reach[..., None, middle, :]
                reached = (reach[..., left, right] | reach[..., right, left]) & old_pairs
                totals = tally[run_id]
                totals["valid_states"] += int(valid.sum())
                totals["strict_pairs"] += int((full * valid).sum())
                totals["closure_pairs"] += int((reached.sum(dim=-1) * valid).sum())
                totals["complete_order_states"] += int((valid & (reached.sum(dim=-1) == full)).sum())
        for task in plan["tasks"][1:]:
            run_id = task["id"]
            total = tally[run_id]
            pool = next(row for row in pools if row["run_id"] == run_id and row["source_phase"] == phase)
            if total["valid_states"] != pool["valid_states"] or \
                    total["complete_order_states"] != pool["complete_order_states"]:
                raise ValueError("closure audit differs from archived summary: " + run_id)
            closure_rows.append(dict(run_id=run_id, method=NAMES[run_id[len(BATCH)+len("_pair_structure_"):]],
                                     source_phase=phase, historical_task=("stalker", "zealot")[phase],
                                     **total, closure_coverage=ratio(total["closure_pairs"], total["strict_pairs"]),
                                     complete_order_coverage=ratio(total["complete_order_states"],
                                                                   total["valid_states"])))
    return closure_rows


def main():
    plan = read(ROOT / "results" / BATCH / "plan.json")
    if len(plan["tasks"]) != 8:
        raise ValueError("expected one reference and seven branches")
    paths = {task["id"]: Path(task["run_dir"]) for task in plan["tasks"]}
    reference_id = BATCH + "_reference"
    parent = paths[reference_id]
    for task in plan["tasks"]:
        path = paths[task["id"]]
        if not path.exists() or read(path / "status.json").get("status") != "completed":
            raise ValueError("run unfinished: " + task["id"])
        if read(path / "launcher_status.json").get("exit_code") != 0:
            raise ValueError("run exited abnormally: " + task["id"])
        if read(path / "launch.json").get("git_commit") != read(parent / "launch.json")["git_commit"]:
            raise ValueError("launch commits differ: " + task["id"])
        if task["id"] != reference_id:
            restore = read(path / "branch_restore.json")
            if restore["model_hash"] != restore["parent_model_hash"]:
                raise ValueError("parent restore differs: " + task["id"])
            if task["dependencies"] != [reference_id]:
                raise ValueError("unexpected scheduler dependency: " + task["id"])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configs").mkdir(exist_ok=True)
    parent_evals = list(records(parent / "evaluations.jsonl"))
    runs, evaluations, stages, curves, pools, updates, histories, gradients = ([] for _ in range(8))
    for task in plan["tasks"]:
        run_id = task["id"]
        path = paths[run_id]
        method = ("TD reference" if run_id == reference_id else
                  NAMES[run_id[len(BATCH) + len("_pair_structure_"):]])
        role = "pair-reference" if run_id == reference_id else "shared-branch"
        parent_id = "" if run_id == reference_id else reference_id
        identity = dict(run_id=run_id, method=method, seed=3535, role=role, parent_run_id=parent_id)
        launch, status, retention = (read(path / name) for name in
                                     ("launch.json", "status.json", "retention.json"))
        own = list(records(path / "evaluations.jsonl"))
        expected_phases = (0, 1) if role == "pair-reference" else (1, 2)
        if [stage["phase"] for stage in retention["stages"]] != list(range(expected_phases[-1] + 1)):
            raise ValueError("missing retention stage: " + run_id)
        for phase in expected_phases:
            targets = ("stalker", "zealot", "colossus")[:phase + 1]
            rows = [r for r in own if r["phase"] == phase and r["label"] == "stage_end"]
            if {r["task"] for r in rows} != set(targets) or any(r["episodes"] != 128 for r in rows):
                raise ValueError("missing independent stage-end evaluation: " + run_id)
        joined = own if role == "pair-reference" else [r for r in parent_evals if r["phase"] == 0] + own
        stage, curve = stage_rows(identity, retention, joined, own)
        if len(stage) != len(expected_phases):
            raise ValueError("incomplete stage metrics: " + run_id)
        stages.extend(stage)
        curves.extend(curve)
        evaluations.extend(dict(identity, **{key: row[key] for key in
                                  ("phase", "task", "label", "t_env", "wins", "episodes")}) for row in own)
        restore = read(path / "branch_restore.json") if role == "shared-branch" else {}
        runs.append(dict(identity, commit=launch["git_commit"], status=status["status"], exit_code=0,
                         replay_on_switch=read(path / "protocol.json")["replay_on_switch"],
                         epsilon_on_switch=read(path / "protocol.json")["epsilon_on_switch"],
                         completed_global_steps=status["t_env"],
                         trained_steps=status["t_env"] - restore.get("t_env", 0),
                         run_dir=str(path.relative_to(ROOT)),
                         restore_model_hash=restore.get("model_hash", ""),
                         prefix_model_hash=restore.get("parent_model_hash", ""),
                         config="configs/" + run_id + ".json"))
        (OUT / "configs" / (run_id + ".json")).write_text(json.dumps(
            dict(protocol=read(path / "protocol.json"), launch_git_commit=launch["git_commit"],
                 source_sha256={key: launch["source_sha256"][key] for key in
                                ("src/learners/pair_ablation.py", "src/learners/ranking_preference.py",
                                 "drivers/continual/action_set_protocol.py",
                                 "drivers/continual/action_set_train.py")},
                 branch_restore=restore),
            indent=2, ensure_ascii=False) + "\n")
        if role == "pair-reference":
            continue
        for phase in (0, 1):
            summary = read(path / "pair_ablation_memory" / ("phase%04d" % phase) / "summary.json")
            if summary.get("pair_seed") != 3535 or not summary.get("pair_mask_sha256"):
                raise ValueError("missing fixed pair mask: " + run_id)
            pools.append(dict(identity, source_phase=phase,
                              historical_task=("stalker", "zealot")[phase], **summary,
                              post_gap_pair_coverage=ratio(summary["valid_pairs"], summary["gap_eligible_pairs"]),
                              post_gap_compression_rate=(1 - ratio(summary["valid_pairs"],
                                                                   summary["gap_eligible_pairs"])
                                                         if summary["gap_eligible_pairs"] else None)))
        grouped = defaultdict(list)
        history = defaultdict(list)
        for row in records(path / (run_id[len(BATCH)+1:] + "_updates.jsonl")):
            grouped[row["phase"]].append(row)
            for item in row["historical_tasks"]:
                history[(row["phase"], item["source_phase"], item["task"])].append(item)
        for phase, rows in grouped.items():
            participating = [row for row in rows if row.get("preference_loss") is not None]
            updates.append(dict(identity, phase=phase, training_task=("stalker", "zealot", "colossus")[phase],
                                updates=len(rows), participating_updates=len(participating),
                                mean_preference_loss=mean(r["preference_loss"] for r in participating)
                                if participating else None))
        for (phase, source, historical), rows in sorted(history.items()):
            total = {field: sum(row.get(field, 0) for row in rows) for field in
                     ("decision_states", "valid_states", "candidate_pairs", "eligible_pairs",
                      "valid_pairs", "top1_pairs", "non_top1_pairs", "top1_flips", "top1_states")}
            valid = total["valid_states"]
            def weighted(field):
                return ratio(sum(row[field] * row["valid_states"] for row in rows
                                 if row.get(field) is not None),
                             sum(row["valid_states"] for row in rows if row.get(field) is not None))
            histories.append(dict(identity, phase=phase, training_task=("stalker", "zealot", "colossus")[phase],
                                  source_phase=source, historical_task=historical, updates=len(rows), **total,
                                  coverage=ratio(valid, total["decision_states"]),
                                  top1_ratio=ratio(total["top1_pairs"], total["valid_pairs"]),
                                  pair_compression_rate=1-ratio(total["valid_pairs"], total["eligible_pairs"]),
                                  mean_task_loss=mean(row["loss"] for row in rows if row.get("loss") is not None),
                                  kendall_tau_a=weighted("kendall_tau_a"),
                                  mean_normalized_regret=weighted("mean_normalized_regret"),
                                  selected_top1_retention=weighted("selected_top1_retention")))
        gradients.extend(gradient_rows(identity, path))

    hashes = defaultdict(set)
    for row in pools:
        hashes[row["source_phase"]].add(row["valid_state_sha256"])
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("structure methods do not share the same valid states")
    closure = check_masks(plan, paths, parent, pools)
    old_rows = list(csv.DictReader((OLD / "stage_metrics.csv").open()))
    old_map = {(r["method"], int(r["phase"])): r for r in old_rows if r["role"] == "shared-branch"}
    comparisons = []
    for row in stages:
        if row["role"] != "shared-branch":
            continue
        old = old_map.get((row["method"], row["phase"]))
        if old is None:
            raise ValueError("missing original method comparison: " + row["method"])
        metrics = ("stalker_win_pct", "zealot_win_pct", "colossus_win_pct", "acc_pct",
                   "bwt_pp", "fwt_pp", "learning_auc_pct")
        comparisons.append(dict(run_id=row["run_id"], method=row["method"], phase=row["phase"],
                                old_run_id=old["run_id"],
                                **{key + "_delta": (float(row[key]) - float(old[key]))
                                   if row.get(key) is not None and old.get(key) else None for key in metrics}))
    for name, rows in (("runs", runs), ("evaluations", evaluations), ("stage_metrics", stages),
                       ("learning_curves", curves), ("pool_diagnostics", pools),
                       ("update_diagnostics", updates), ("preference_diagnostics", histories),
                       ("gradient_diagnostics", gradients), ("paired_old_gap0", comparisons)):
        write(name, rows)
    write("closure_diagnostics", closure)
    print("verified", len(runs), "runs,", len(stages), "stage rows,", len(pools), "fixed masks")


if __name__ == "__main__":
    main()
