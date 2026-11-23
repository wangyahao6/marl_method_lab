"""Verify completed SZC pair-distance runs and export the two ablation tables."""
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(os.environ.get("MARL_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from learners.pair_distance import select_distance_pairs
from learners.ranking_preference import reference_order
from summarize_szc_tkpr_k1_shared import gradient_rows, records, stage_rows

BATCH = "szc_pair_distance_2222_3535_20261122"
DEST = ROOT / "docs/results/parameter-ablations"
GROUPS = {
    "pair_distance": ("szc-vdn64-pair-distance", {
        "k1": "K1", "near": "K1+Near", "mid": "K1+Mid", "far": "K1+Far",
        "far_mid": "K1+Far/Mid", "random_matched": "K1+Random-Matched", "maxgap": "K1+MaxGap",
    }),
    "pair_gap_far_mid": ("szc-vdn64-pair-far-mid-gap", {
        "far_mid_unfiltered": "K1+Far/Mid gap=0",
        "far_mid_filtered": "K1+Far/Mid extra gap>=0.5",
    }),
}


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"empty {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ratio(a, b):
    return a / b if b else None


def finite_distribution(prefix, values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {prefix + name: None for name in ("mean", "std", "median", "p25", "p75", "p90", "p95", "p99", "min", "max")}
    return dict(zip((prefix + name for name in ("mean", "std", "median", "p25", "p75", "p90", "p95", "p99", "min", "max")),
                    (float(a.mean()), float(a.std()), *[float(np.quantile(a, q)) for q in (.5, .25, .75, .9, .95, .99)],
                     float(a.min()), float(a.max()))))


def selector_data(episodes):
    fields = ["old_q_values", "avail_actions", "filled", "terminated"]
    data = {key: pad_sequence([ep["data"][key] for ep in episodes], batch_first=True)
            for key in fields}
    old = data["old_q_values"][:, :-1]
    legal = data["avail_actions"][:, :-1].bool()
    filled = data["filled"].bool()
    terminal = data["terminated"][:, :-1].bool()
    after = th.cat((th.zeros_like(terminal[:, :1]), terminal[:, :-1]), dim=1)
    decisions = (filled[:, :-1] & filled[:, 1:] & (after.cumsum(dim=1) == 0))
    decisions = decisions & legal[..., 1:].any(dim=-1)
    return old, legal, decisions


def check_pool_masks(parent, task, source_phase, summary):
    """Rebuild the exact padded-batch mask and gather selected-pair diagnostics."""
    run = Path(task["run_dir"])
    source = th.load(parent / "action_set_memory" / f"phase{source_phase:04d}" / "success.th",
                     map_location="cpu")["episodes"]
    archived = th.load(run / "pair_ablation_memory" / f"phase{source_phase:04d}" / "selection.th",
                       map_location="cpu")
    if len(source) != len(archived):
        raise ValueError("history length and fixed mask length differ")
    old, legal, decisions = selector_data(source)
    method = task["id"].split("_seed")[0].split(BATCH + "_")[1]
    group = "pair_gap_far_mid" if method.startswith("pair_gap_far_mid") else "pair_distance"
    selection = method[len(group) + 1:]
    if group == "pair_gap_far_mid":
        selection = "far_mid_" + selection
    threshold = .5 if selection == "far_mid_filtered" else 0
    rebuilt, valid, counts, audit = select_distance_pairs(
        old, legal, decisions, selection, 3535, min_gap=0, q_scale_floor=.1,
        extra_min_gap=threshold)
    valid_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
    for ep, one, a, v in zip(source, archived, rebuilt, valid):
        length = ep["length"]
        if not th.equal(one, a[:length]):
            raise ValueError("archived mask differs from fixed-seed rebuild: " + task["id"])
        valid_hash.update(v[:length].numpy().tobytes())
        mask_hash.update(one.numpy().tobytes())
    if (summary["valid_state_sha256"] != valid_hash.hexdigest()
            or summary["pair_mask_sha256"] != mask_hash.hexdigest()
            or summary["valid_pairs"] != int(rebuilt.sum())
            or summary["valid_states"] != int(valid.sum())):
        raise ValueError("archived pair summary differs from rebuilt mask: " + task["id"])

    size = legal.size(-1)
    left, right = th.triu_indices(size, size, 1)
    ranks = th.sort(old.masked_fill(~legal, -th.inf), dim=-1, stable=True,
                    descending=True).indices.argsort(dim=-1)
    distance = (ranks[..., left] - ranks[..., right]).abs()
    top = old.masked_fill(~legal, -th.inf).argmax(dim=-1)
    top_pair = (left == top[..., None]) | (right == top[..., None])
    _, scale, _, _ = reference_order(old, legal, .1)
    gap = (old[..., left] - old[..., right]).abs() / scale
    eligible = legal[..., left] & legal[..., right] & (old[..., left] != old[..., right]) & valid[..., None]
    reach = th.zeros((*old.shape[:-1], size, size), dtype=th.bool)
    reach[..., left, right] = rebuilt & (old[..., left] > old[..., right])
    reach[..., right, left] = rebuilt & (old[..., right] > old[..., left])
    for middle in range(size):
        reach |= reach[..., :, middle, None] & reach[..., None, middle, :]
    reached = (reach[..., left, right] | reach[..., right, left]) & eligible
    closure_coverage = ratio(int(reached.sum()), int(eligible.sum()))
    closure_complete = int((valid & (reached.sum(dim=-1) == eligible.sum(dim=-1))).sum())
    if closure_complete != summary["complete_order_states"]:
        raise ValueError("transitive closure differs from archived complete ordering")
    category = {"all": rebuilt, "top1": rebuilt & top_pair, "extra": rebuilt & ~top_pair}
    candidate_category = {"all": eligible, "top1": eligible & top_pair, "extra": eligible & ~top_pair}
    rows = []
    distance_rows = []
    for name, mask in category.items():
        values = gap[mask].numpy()
        per_state = []
        for state_gap, state_mask in zip(gap[valid], mask[valid]):
            subset = state_gap[state_mask].numpy()
            if subset.size:
                per_state.append(float(subset.mean()))
        d = distance[mask].numpy()
        rows.append(dict(category=name, pair_count=int(mask.sum()),
                         mean_pairs_per_valid_state=ratio(int(mask.sum()), int(valid.sum())),
                         near_pairs=int((mask & (distance == 1)).sum()),
                         mid_pairs=int((mask & (distance == 2)).sum()),
                         far_pairs=int((mask & (distance >= 3)).sum()),
                         mean_rank_distance=float(d.mean()) if len(d) else None,
                         transitive_closure_coverage=closure_coverage,
                         **finite_distribution("gap_", values),
                         **finite_distribution("state_mean_gap_", per_state)))
        for rank in sorted(np.unique(distance[candidate_category[name]].numpy())):
            subset = values[d == rank]
            candidate_count = int((candidate_category[name] & (distance == int(rank))).sum())
            distance_rows.append(dict(category=name, rank_distance=int(rank), pair_count=len(subset),
                                      candidate_count=candidate_count,
                                      rank_pair_coverage=ratio(len(subset), candidate_count),
                                      fraction_of_category=ratio(len(subset), len(values)),
                                      **finite_distribution("gap_", subset)))
    return rows, distance_rows, audit


def verify_task(task, parent):
    run = Path(task["run_dir"])
    if not run.exists() or read(run / "status.json").get("status") != "completed":
        raise ValueError("unfinished run: " + task["id"])
    if read(run / "launcher_status.json").get("exit_code") != 0:
        raise ValueError("nonzero launcher exit: " + task["id"])
    launch = read(run / "launch.json")
    parent_launch = read(parent / "launch.json")
    # Unrelated TKPR-refresh and TS5 changes landed while this batch was queued.
    # The pair selector and pair loss files themselves must match byte for byte.
    for source in ("src/learners/pair_ablation.py", "src/learners/pair_distance.py"):
        if launch["source_sha256"][source] != parent_launch["source_sha256"][source]:
            raise ValueError("reference/branch pair source differs: " + task["id"])
    if task["dependencies"]:
        shared = ("tasks", "stages", "evaluation", "action_set", "full_rank_preference",
                  "replay_on_switch", "epsilon_on_switch", "gradient_monitor_interval",
                  "phase_boundary", "sc2_game_version")
        branch_protocol, parent_protocol = read(run / "protocol.json"), read(parent / "protocol.json")
        if any(branch_protocol.get(key) != parent_protocol.get(key) for key in shared):
            raise ValueError("reference/branch shared protocol differs: " + task["id"])
        restore = read(run / "branch_restore.json")
        if restore.get("model_hash") != restore.get("parent_model_hash"):
            raise ValueError("S checkpoint restore mismatch: " + task["id"])
    return run


def export():
    plan = read(ROOT / "results" / BATCH / "plan.json")
    if len(plan["tasks"]) != 20:
        raise ValueError("expected 20 tasks")
    tasks = {t["id"]: t for t in plan["tasks"]}
    for task in plan["tasks"]:
        parent = Path(tasks[task["dependencies"][0]]["run_dir"]) if task["dependencies"] else Path(task["run_dir"])
        verify_task(task, parent)
    all_runs, all_stages, all_pairs, all_dist, all_grads = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    all_curves, all_updates, all_evaluations = defaultdict(list), defaultdict(list), defaultdict(list)
    pool_hashes = defaultdict(set)
    reference_rows = defaultdict(list)
    for task in plan["tasks"]:
        run = Path(task["run_dir"])
        parts = task["id"].split("_seed")
        seed = int(parts[-1])
        key = parts[0][len(BATCH) + 1:]
        if key == "reference":
            parent_identity = dict(run_id=task["id"], group="pair_distance", method="TD reference",
                                   seed=seed, role="pair-reference", parent_run_id="")
            parent_stage, parent_curve = stage_rows(parent_identity, read(run / "retention.json"),
                                                    list(records(run / "evaluations.jsonl")),
                                                    list(records(run / "evaluations.jsonl")))
            if not parent_stage or parent_stage[0]["phase"] != 0:
                raise ValueError("reference S stage missing: " + task["id"])
            for experiment in GROUPS:
                reference_rows[experiment].append(dict(parent_identity, group=experiment,
                                                       status="completed", exit_code=0,
                                                       git_commit=read(run / "launch.json")["git_commit"],
                                                       run_dir=str(run.relative_to(ROOT)),
                                                       restore_model_hash="", parent_model_hash="", parent_run_dir=""))
                all_stages[experiment].append(dict(parent_stage[0], group=experiment))
                all_curves[experiment].extend(dict(row, group=experiment)
                                              for row in parent_curve if row["phase"] == 0)
                all_evaluations[experiment].extend(
                    dict(parent_identity, group=experiment,
                         **{name: row[name] for name in ("phase", "task", "label", "t_env", "wins", "episodes")})
                    for row in records(run / "evaluations.jsonl"))
            continue
        group = "pair_gap_far_mid" if key.startswith("pair_gap_far_mid") else "pair_distance"
        suffix = key[len(group) + 1:]
        selection = "far_mid_" + suffix if group == "pair_gap_far_mid" else suffix
        method = GROUPS[group][1][selection]
        parent_id = task["dependencies"][0]
        parent = Path(tasks[parent_id]["run_dir"])
        identity = dict(run_id=task["id"], group=group, method=method, seed=seed,
                        role="shared-branch", parent_run_id=parent_id)
        own = list(records(run / "evaluations.jsonl"))
        parent_evals = list(records(parent / "evaluations.jsonl"))
        for phase in (1, 2):
            expected = {"stalker", "zealot", "colossus"} if phase == 2 else {"stalker", "zealot"}
            actual = [r for r in own if r["phase"] == phase and r["label"] == "stage_end"]
            if {r["task"] for r in actual} != expected or any(r["episodes"] != 128 for r in actual):
                raise ValueError("missing 128 episode stage-end evaluation: " + task["id"])
        joined = [r for r in parent_evals if r["phase"] == 0] + own
        stage, curve = stage_rows(identity, read(run / "retention.json"), joined, own)
        if len(stage) != 2 or any(r["learning_auc_pct"] is None for r in stage):
            raise ValueError("incomplete stage or AUC: " + task["id"])
        all_stages[group].extend(stage)
        all_curves[group].extend(curve)
        all_evaluations[group].extend(
            dict(identity, **{name: row[name] for name in ("phase", "task", "label", "t_env", "wins", "episodes")})
            for row in own)
        all_grads[group].extend(gradient_rows(identity, run))
        restore = read(run / "branch_restore.json")
        all_runs[group].append(dict(identity, status="completed", exit_code=0,
                                    git_commit=read(run / "launch.json")["git_commit"],
                                    run_dir=str(run.relative_to(ROOT)),
                                    restore_model_hash=restore["model_hash"],
                                    parent_model_hash=restore["parent_model_hash"],
                                    parent_run_dir=str(parent.relative_to(ROOT))))
        for source_phase, historical_task in enumerate(("stalker", "zealot")):
            summary = read(run / "pair_ablation_memory" / f"phase{source_phase:04d}" / "summary.json")
            pool_hashes[seed, group, source_phase].add(summary["valid_state_sha256"])
            diag, dist, audit = check_pool_masks(parent, task, source_phase, summary)
            for row in diag:
                all_pairs[group].append(dict(identity, source_phase=source_phase,
                                             historical_task=historical_task,
                                             valid_states=summary["valid_states"],
                                             decision_states=summary["decision_states"],
                                             state_coverage=summary["coverage"],
                                             candidate_pairs=summary["candidate_pairs"],
                                             eligible_pairs=summary["eligible_pairs"],
                                             pair_mask_sha256=summary["pair_mask_sha256"],
                                             valid_state_sha256=summary["valid_state_sha256"],
                                             excluded_by_extra_gap_states=summary.get("excluded_by_extra_gap_states"),
                                             top1_ratio=summary["top1_ratio"],
                                             complete_order_coverage=summary.get("complete_order_coverage"),
                                             ordering_edge_coverage=summary.get("ordering_edge_coverage"),
                                             extra_budget_pairs=summary.get("extra_budget_pairs"),
                                             extra_selected_pairs=summary.get("extra_selected_pairs"),
                                             target_far_pairs=summary.get("target_far_pairs"),
                                             target_mid_pairs=summary.get("target_mid_pairs"),
                                             **row))
            all_dist[group].extend(dict(identity, source_phase=source_phase,
                                        historical_task=historical_task, **row) for row in dist)
        updates = list(records(run / (key + "_updates.jsonl")))
        for phase in (1, 2):
            rows = [r for r in updates if r["phase"] == phase]
            if not rows:
                raise ValueError("missing preference updates: " + task["id"])
            valid_loss = [r["preference_loss"] for r in rows if r.get("preference_loss") is not None]
            history = [item for r in rows for item in r["historical_tasks"]]
            pairs = sum(item["valid_pairs"] for item in history)
            states = sum(item["valid_states"] for item in history)
            violation_count = sum(item.get("pair_violations", 0) for item in history)
            retention_sum = sum(item["selected_top1_retention"] * item["valid_states"]
                                for item in history if item.get("selected_top1_retention") is not None)
            retention_states = sum(item["valid_states"] for item in history
                                   if item.get("selected_top1_retention") is not None)
            all_updates[group].append(dict(identity, phase=phase,
                                           mean_preference_loss=mean(valid_loss) if valid_loss else None,
                                           sampled_valid_states=states, sampled_pairs=pairs,
                                           selected_pair_violation_rate=ratio(violation_count, pairs),
                                           top1_flip_rate=(1 - ratio(retention_sum, retention_states)
                                                           if retention_states else None),
                                           updates=len(rows)))
    for key, hashes in pool_hashes.items():
        if len(hashes) != 1:
            raise ValueError("methods in a group do not share valid states: " + str(key))
    for group, (folder, names) in GROUPS.items():
        out = DEST / folder
        out.mkdir(parents=True, exist_ok=True)
        config_dir = out / "configs"
        config_dir.mkdir(exist_ok=True)
        for row in all_runs[group]:
            run = ROOT / row["run_dir"]
            launch = read(run / "launch.json")
            sources = ("src/learners/pair_ablation.py", "src/learners/pair_distance.py",
                       "src/learners/ranking_preference.py", "src/learners/action_set_preference.py",
                       "drivers/continual/action_set_train.py", "drivers/continual/action_set_protocol.py")
            (config_dir / (row["run_id"] + ".json")).write_text(json.dumps(
                dict(protocol=read(run / "protocol.json"), launch_git_commit=launch["git_commit"],
                     source_sha256={name: launch["source_sha256"][name] for name in sources},
                     branch_restore=read(run / "branch_restore.json")), ensure_ascii=False, indent=2) + "\n")
        for row in reference_rows[group]:
            run = ROOT / row["run_dir"]
            launch = read(run / "launch.json")
            (config_dir / (row["run_id"] + ".json")).write_text(json.dumps(
                dict(protocol=read(run / "protocol.json"), launch_git_commit=launch["git_commit"],
                     source_sha256={name: launch["source_sha256"][name] for name in sources}),
                ensure_ascii=False, indent=2) + "\n")
        write_csv(out / "runs.csv", reference_rows[group] + all_runs[group])
        write_csv(out / "evaluations.csv", all_evaluations[group])
        write_csv(out / "stage_metrics.csv", all_stages[group])
        write_csv(out / "pair_diagnostics.csv", all_pairs[group])
        write_csv(out / "pair_rank_gap_distribution.csv", all_dist[group])
        write_csv(out / "learning_curves.csv", all_curves[group])
        write_csv(out / "gradient_diagnostics.csv", all_grads[group])
        write_csv(out / "preference_diagnostics.csv", all_updates[group])
        k1 = {(r["seed"], r["phase"]): r for r in all_stages["pair_distance"] if r["method"] == "K1"}
        paired = []
        for row in all_stages[group]:
            if row["method"] in ("K1", "TD reference"):
                continue
            baseline = k1[row["seed"], row["phase"]]
            metrics = ("acc_pct", "bwt_pp", "fwt_pp", "stalker_win_pct", "zealot_win_pct",
                       "colossus_win_pct", "learning_auc_pct")
            paired.append(dict(run_id=row["run_id"], method=row["method"], seed=row["seed"], phase=row["phase"],
                               k1_run_id=baseline["run_id"],
                               **{name + "_delta": (row[name] - baseline[name]
                                                       if row.get(name) is not None and baseline.get(name) is not None else None)
                                  for name in metrics}))
        write_csv(out / "paired_metrics.csv", paired)
        if group == "pair_gap_far_mid":
            by_seed_phase = {(r["seed"], r["phase"], r["method"]): r
                             for r in all_stages[group] if r["method"] != "TD reference"}
            gap_pairs = []
            for seed in (2222, 3535):
                for phase in (1, 2):
                    no = by_seed_phase[seed, phase, "K1+Far/Mid gap=0"]
                    yes = by_seed_phase[seed, phase, "K1+Far/Mid extra gap>=0.5"]
                    gap_pairs.append(dict(seed=seed, phase=phase, unfiltered_run_id=no["run_id"],
                                          filtered_run_id=yes["run_id"],
                                          **{name + "_filtered_minus_unfiltered":
                                             (yes[name] - no[name] if yes.get(name) is not None
                                              and no.get(name) is not None else None)
                                             for name in ("acc_pct", "bwt_pp", "fwt_pp", "learning_auc_pct",
                                                          "stalker_win_pct", "zealot_win_pct", "colossus_win_pct")}))
            write_csv(out / "paired_gap.csv", gap_pairs)
        summary = []
        for method in names.values():
            rows = [r for r in all_stages[group] if r["method"] == method and r["phase"] == 2]
            if len(rows) != 2:
                raise ValueError("missing paired seeds: " + method)
            baseline = [r for r in paired if r["method"] == method and r["phase"] == 2]
            metrics = ("acc_pct", "bwt_pp", "fwt_pp", "stalker_win_pct", "zealot_win_pct",
                       "colossus_win_pct", "learning_auc_pct")
            summary.append(dict(method=method, seeds="2222;3535",
                                **{name + "_mean": mean(r[name] for r in rows)
                                   if all(r.get(name) is not None for r in rows) else None for name in metrics},
                                **{name + "_std": stdev(r[name] for r in rows)
                                   if all(r.get(name) is not None for r in rows) else None for name in metrics},
                                **{name + "_paired_k1_delta_mean": mean(r[name + "_delta"] for r in baseline)
                                   if baseline else None for name in ("acc_pct", "bwt_pp")}))
        write_csv(out / "summary.csv", summary)
        print(group, len(all_runs[group]), "runs", len(all_stages[group]), "stages", len(all_pairs[group]), "pair rows")


if __name__ == "__main__":
    export()
