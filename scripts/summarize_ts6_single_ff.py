"""Summarize the planned single-feedforward VDN runs after they reach a terminal state."""
import argparse
import json
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def read_rows(path):
    if path.exists():
        for line in path.read_text().splitlines():
            yield json.loads(line)


def summarize_run(run, stage_labels):
    rows = list(read_rows(run / 'neuron_detections.jsonl'))
    failures = list(read_rows(run / 'neuron_detection_errors.jsonl'))
    stages = {}
    for phase, label in enumerate(stage_labels):
        checks = [row for row in rows if row['phase'] == phase]
        selected = [set(row['sets']['above_threshold']) if row['sets']['above_threshold'] is not None
                    else None for row in checks]
        valid = [ids for ids in selected if ids is not None]
        overlap = [len(a & b) / len(a | b) for a, b in zip(selected, selected[1:])
                   if a is not None and b is not None and a | b]
        stage_end = next((row for row in checks if row['stage_end']), None)
        stages[label] = dict(
            snapshots=len(checks),
            failures=sum(row['phase'] == phase for row in failures),
            defined=len(valid),
            mean_selected_fraction=mean(len(ids) / checks[0]['width'] for ids in valid) if valid else None,
            mean_adjacent_jaccard=mean(overlap) if overlap else None,
            stage_end=stage_end,
        )
    return dict(status=read_json(run / 'status.json'),
                launcher=read_json(run / 'launcher_status.json'),
                detection=stages, retention=read_json(run / 'retention.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    plan_path = parser.parse_args().plan
    plan = read_json(plan_path)
    stage_labels = plan.get('stage_labels', plan['order'])
    runs = {}
    for task in plan['tasks']:
        if task['type'] != 'training':
            continue
        run = Path(task['run_dir'])
        record = summarize_run(run, stage_labels)
        runs[task['id']] = dict(record, **task['metadata'])
    means = {}
    for algorithm in sorted({run['algorithm'] for run in runs.values()}):
        completed = [run for run in runs.values()
                     if run['algorithm'] == algorithm and run['status'].get('status') == 'completed'
                     and run['launcher'].get('exit_code') == 0]
        final = [run['retention'].get('metrics', {}).get(stage_labels[-1], {}) for run in completed]
        metrics = {}
        for key in ('acc_percent', 'bwt_pp', 'fwt_pp'):
            values = [row[key] for row in final if row.get(key) is not None]
            metrics[key] = dict(mean=mean(values) if values else None, n=len(values))
        means[algorithm] = dict(completed_seeds=sorted(run['seed'] for run in completed), metrics=metrics)
    dest = plan_path.with_name('summary.json')
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(runs=runs, final_by_algorithm=means), indent=2, allow_nan=False) + '\n')
    tmp.replace(dest)


if __name__ == '__main__':
    main()
