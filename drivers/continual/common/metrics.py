"""Versioned scalar metrics written alongside the original driver logs."""
import json
import logging
import math

from .records import append


class Metrics:
    def __init__(self, out, algorithm, method, seed, *, run_id=None, **run_context):
        self.path = out / 'experiment_metrics.jsonl'
        self.identity = dict(schema_version=1, run_id=out.name if run_id is None else run_id,
                             algorithm=algorithm, method=method, seed=seed)
        self.run_context = run_context
        self.warned = False

    def record(self, category, phase, task, t_env, values, **context):
        self._write(category, phase, task, t_env, ((name, value, context) for name, value in values.items()))

    def _write(self, category, phase, task, t_env, measurements):
        lines = []
        for name, value, context in measurements:
            if value is None or not math.isfinite(value):
                continue
            row = dict(self.identity, category=category, phase=int(phase), task=task,
                       t_env=int(t_env), metric=name, value=float(value), context=dict(self.run_context, **context))
            lines.append(json.dumps(row, allow_nan=False))
        if lines:
            try:
                with self.path.open('a', encoding='utf-8') as stream:
                    stream.write('\n'.join(lines) + '\n')
            except OSError:
                if not self.warned:
                    logging.getLogger(__name__).warning('Could not append to %s', self.path, exc_info=True)
                    self.warned = True

    def evaluation(self, row):
        context = {key: row[key] for key in ('focus', 'trained_task', 'reference_phase') if key in row}
        self.record('Performance', row['phase'], row.get('task', row.get('unit')), row['t_env'],
                    {'win_rate': row['win_rate']}, event=row.get('label', row.get('evaluation')),
                    episodes=row['episodes'], **context)

    def retention(self, phase, task, t_env, row, **context):
        keys = ('acc_percent', 'bwt_pp', 'fwt_pp', 'old_mean_percent', 'learning_auc_percent',
                'before_revisit_percent', 'revisit_gain_pp')
        self.record('Performance', phase, task, t_env, {key: row.get(key) for key in keys},
                    event='stage_summary', **context)

    def neurons(self, phase, task, t_env, layer, values, **context):
        def rows():
            for name, vector in values.items():
                if vector is not None:
                    for index, value in enumerate(vector):
                        yield name, value, dict(context, layer=layer, neuron_index=index)

        self._write('Mechanism', phase, task, t_env, rows())

    def protection(self, phase, task, t_env, schedule):
        for layer, scales in schedule['unit_scales'].items():
            if scales:
                self.record('Mechanism', phase, task, t_env,
                            {'protected_neuron_fraction': len(schedule['protected'][layer]) / len(scales)},
                            event='phase_start', layer=layer)

    def preference(self, row, event):
        fields = ('loss', 'violation_ratio', 'greedy_outside_ratio', 'top1_flip_rate', 'coverage')

        def measurements():
            for item in row.get('historical_tasks', [row]):
                context = dict(event=event, source_task=item['task'], source_phase=item['source_phase'])
                for name in fields:
                    yield name, item.get(name), context

        self._write('Mechanism', row['phase'], row['task'], row['t_env'], measurements())


def record_evaluation(out, metrics, row):
    append(out / 'evaluations.jsonl', row)
    metrics.evaluation(row)


def method_for_group(group, *, chi=False, trhr=False):
    if group.startswith('GEM_'):
        return 'gem'
    if group in ('N', 'TRHR'):
        return 'trhr' if trhr else 'baseline'
    if chi:
        return 'softprotect_chi_trhr' if trhr else 'softprotect_chi'
    return 'softprotect_trhr' if trhr else 'softprotect'
