"""Summarize the observed TS8 CHI release event and its paired trials."""
import json
import sys
from pathlib import Path
from statistics import mean

plan_file = Path(sys.argv[1])
plan = json.loads(plan_file.read_text())
summary = {}
for task in plan['tasks']:
    if task['type'] != 'training':
        continue
    meta = task['metadata']
    width = summary.setdefault(meta['algorithm'], {'trials': {}})
    run = Path(task['run_dir'])
    status_file = run / 'status.json'
    status = json.loads(status_file.read_text()) if status_file.exists() else {}
    arm = meta['release_arm']
    if arm == 'search':
        event_file = run / 'release_event.json'
        width['search'] = status
        width['event'] = json.loads(event_file.read_text()) if event_file.exists() else None
        continue
    trial = width['trials'][arm] = {'status': status}
    if status.get('status') != 'completed':
        continue
    rows = [json.loads(line) for line in (run / 'evaluations.jsonl').read_text().splitlines()]
    current = {0: next(row['win_rate'] for row in rows if row['label'] == 'phase_start')}
    current.update({row['reference_phase']: row['win_rate'] for row in rows
                    if row['label'] == 'release_window'})
    trial['current_win_rates'] = current
    for horizon in (100000, 200000):
        steps = list(range(0, horizon + 1, 25000))
        trial[f'auc_{horizon}_percent'] = (
            100 * mean((current[a] + current[b]) / 2 for a, b in zip(steps, steps[1:]))
            if all(step in current for step in steps) else None)
    old = {}
    for row in rows:
        if row['label'] == 'release_old':
            old.setdefault(row['reference_phase'], {})[row['task']] = row['win_rate']
    trial['old_win_rates'] = old
    trial['retention'] = json.loads((run / 'retention.json').read_text())

(plan_file.parent / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
