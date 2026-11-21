"""Compact group diagnostics for ASPR-3G/4G runs."""
import csv
import json
from collections import defaultdict

from learners.ranking_preference import group_diagnostics


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_group_results(out, history, scheme, groups, group_count, method):
    summary, distribution = [], []
    for phase, pool in sorted(history.items()):
        stats = group_diagnostics(pool["episodes"], scheme, groups, group_count)
        for name in stats["group_mean_sizes"]:
            common = dict(source_phase=phase, task=pool["task"], group=name,
                          stored_episodes=len(pool["episodes"]),
                          decision_states=stats["group_decision_states"])
            summary.append(dict(common, mean_actions=stats["group_mean_sizes"][name],
                                action_ratio=stats["group_action_ratios"][name],
                                empty_ratio=stats["group_empty_ratios"][name]))
            for size, count in stats["group_size_distribution"][name].items():
                distribution.append(dict(common, group_size=size, state_count=count))
    summary_fields = ("source_phase", "task", "group", "stored_episodes", "decision_states",
                      "mean_actions", "action_ratio", "empty_ratio")
    distribution_fields = ("source_phase", "task", "group", "stored_episodes", "decision_states",
                           "group_size", "state_count")
    write_csv(out / "aspr_group_summary.csv", summary, summary_fields)
    write_csv(out / "aspr_group_distribution.csv", distribution, distribution_fields)

    sampled = defaultdict(lambda: dict(sampled_episodes=0, decision_states=0, legal_actions=0,
                                       action_count=0, empty_states=0, candidate_pairs=0,
                                       valid_pairs=0, pair_violations=0, distribution=defaultdict(int)))
    log = out / f"{method}_updates.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            update = json.loads(line)
            for task in update["historical_tasks"]:
                for group, histogram in task["group_size_distribution"].items():
                    key = (update["phase"], update["task"], task["source_phase"], task["task"], group)
                    row = sampled[key]
                    row["sampled_episodes"] += task["sampled_episodes"]
                    row["decision_states"] += task["group_decision_states"]
                    row["legal_actions"] += task["group_legal_actions"]
                    row["candidate_pairs"] += task["candidate_pairs"]
                    row["valid_pairs"] += task["valid_pairs"]
                    row["pair_violations"] += task["pair_violations"]
                    for size, count in histogram.items():
                        row["distribution"][size] += count
                        row["action_count"] += int(size) * count
                    row["empty_states"] += histogram.get("0", 0)
    rows = []
    for key, values in sorted(sampled.items()):
        phase, training_task, source_phase, historical_task, group = key
        states, legal = values["decision_states"], values["legal_actions"]
        candidates, valid = values["candidate_pairs"], values["valid_pairs"]
        for size, count in sorted(values["distribution"].items(), key=lambda item: int(item[0])):
            rows.append(dict(training_phase=phase, training_task=training_task,
                             source_phase=source_phase, historical_task=historical_task, group=group,
                             sampled_episodes=values["sampled_episodes"], decision_states=states,
                             mean_actions=values["action_count"] / states if states else None,
                             action_ratio=values["action_count"] / legal if legal else None,
                             empty_ratio=values["empty_states"] / states if states else None,
                             group_size=size, state_count=count, candidate_pairs=candidates,
                             valid_pairs=valid, pair_keep_rate=valid / candidates if candidates else None,
                             pair_violation_rate=values["pair_violations"] / valid if valid else None))
    fields = ("training_phase", "training_task", "source_phase", "historical_task", "group",
              "sampled_episodes", "decision_states", "mean_actions", "action_ratio", "empty_ratio",
              "group_size", "state_count", "candidate_pairs", "valid_pairs", "pair_keep_rate",
              "pair_violation_rate")
    write_csv(out / "aspr_group_training_sampled.csv", rows, fields)
