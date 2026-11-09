"""Collect the four TS3 paths and summarize available stage and neuron evidence."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def neuron_summary(path):
    phases = {}
    if not path.is_file():
        return phases
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            phase = row['phase']
            n = row['window_updates']
            entry = phases.setdefault(phase, dict(sampled_episodes=[0]*5, adjusted=set(),
                newly_woken=set(), historical=row['historical_units'], protected=row['protected_units'],
                groups=defaultdict(lambda: dict(unit_updates=0, sums=[0.]*4))))
            for i, count in enumerate(row['source_episodes']):
                entry['sampled_episodes'][i] += count
            entry['adjusted'].update(row['adjusted_this_window'])
            entry['newly_woken'].update(row['newly_woken'])
            for name, units in (
                ('historical', entry['historical']),
                ('outside_history', set(range(len(row['offset']))) - set(entry['historical'])),
            ):
                stats = entry['groups'][name]
                stats['unit_updates'] += n * len(units)
                values = row['gradient_l2'] + row['update_l2']
                for i, values_i in enumerate(values):
                    stats['sums'][i] += n * sum(values_i[j] for j in units)
    for entry in phases.values():
        entry['adjusted'] = sorted(entry['adjusted'])
        entry['newly_woken'] = sorted(entry['newly_woken'])
        for stats in entry['groups'].values():
            size = stats['unit_updates']
            stats['mean_grad_in_grad_gru_delta_in_delta_gru'] = (
                [v / size for v in stats.pop('sums')] if size else None
            )
            stats.pop('sums', None)
    return phases


def show(value):
    return 'N/A' if value is None else f'{value:.2f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    args = p.parse_args()
    plan = read(args.plan)
    output = args.plan.parent
    out = {}
    for group, path in plan['branches'].items():
        run = Path(path)
        out[group] = dict(status=read(run/'status.json', {}), retention=read(run/'retention.json', {}),
                          neuron_summary=neuron_summary(run/'a_windows.jsonl'), run=str(run))
    prefix = Path(plan['prefix'])
    bundle = dict(groups=out, shared_prefix=dict(run=str(prefix), status=read(prefix/'status.json', {}),
        neuron_summary=neuron_summary(prefix/'a_windows.jsonl')))
    (output/'results.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + '\n')
    lines = ['# TS3 VDN 1FF 128 · TRHR × 软保护', '',
             '种子 3535。四组共享无干预 A 前缀，B–E 应用各自方法。', '',
             '| 组别 | 阶段 | A 胜率 | B 胜率 | C 胜率 | D 胜率 | E 胜率 | 旧任务均值 | ACC | BWT | FWT | 当前任务 AUC |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for group, result in out.items():
        retention = result['retention']
        if not retention:
            lines.append(f'| {group} | {result["status"].get("status", "无终态结果")} | | | | | | | | | | |')
            continue
        for stage, m in retention['metrics'].items():
            row = retention['stage_win_rates'].get(stage, {})
            rates = [show(100 * row[t]) if t in row else 'N/A' for t in 'ABCDE']
            keys = ['old_mean_percent', 'acc_percent', 'bwt_pp', 'fwt_pp', 'learning_auc_percent']
            lines.append('| ' + ' | '.join([group, stage, *rates, *(show(m.get(k)) for k in keys)]) + ' |')
    lines += ['', '胜率、ACC、旧任务均值和 AUC 使用百分数，BWT/FWT 使用百分点。',
              '训练和评价原始日志见计划各 run。逐阶段神经元与回放汇总见 results.json。', '',
              '组合组需同时对照单独软保护和单独 TRHR。单种子结果不能估计跨种子稳定性。']
    (output/'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
