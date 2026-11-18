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

    def _write(self, category, phase, task, t_env, measurements, *, keep_null=False):
        lines = []
        for name, value, context in measurements:
            if isinstance(value, dict):
                metric_value = value
            elif value is not None and math.isfinite(value):
                metric_value = float(value)
            elif keep_null:
                metric_value = None
            else:
                continue
            row = dict(self.identity, category=category, phase=int(phase), task=task,
                       t_env=int(t_env), metric=name, value=metric_value, context=dict(self.run_context, **context))
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

    def gradients(self, phase, task, t_env, update, values):
        context = dict(event='gradient_monitor', learner_update=update)
        self._write('Mechanism', phase, task, t_env,
                    ((name, value, context) for name, value in values.items()), keep_null=True)

    def preference(self, row, event):
        fields = ('loss', 'violation_ratio', 'greedy_outside_ratio', 'top1_flip_rate', 'coverage',
                  'valid_states', 'sampled_episodes', 'decision_states', 'nonzero_loss_states')
        if 'pair_filter_stats' not in row:
            fields += ('valid_pairs',)
        record_top1_max = self.identity['method'] == 'top1_max_preference'
        if record_top1_max:
            fields += ('constrained_top1_flip_rate', 'margin_violation_rate')
        elif self.identity['method'] == 'top1_unfiltered_preference':
            fields += ('mean_constrained_confidence',)
        elif self.identity['method'] in ('topk_preference', 'topk_preference_weighted',
                                           'topk_preference_high_regret'):
            fields += ('pair_violation_rate', 'mean_effective_k', 'k_clip_rate',
                       'effective_k_histogram', 'mean_state_confidence', 'zero_confidence_states',
                       'nonzero_weight_states', 'nonzero_weight_coverage', 'nonzero_weight_fraction_of_valid',
                       'high_regret_states', 'high_regret_fraction', 'high_regret_loss', 'mean_high_regret',
                       'base_preference_loss', 'weighted_high_regret_loss', 'high_regret_loss_share', 'weight_histogram')

        context = dict(event=event)
        if 'learner_update' in row:
            context['learner_update'] = row['learner_update']

        def measurements():
            if record_top1_max:
                yield 'preference_loss', row['preference_loss'], context
            for name in ('preference_loss_nonzero', 'historical_episode_count', 'decision_states',
                         'constrained_states', 'nonzero_loss_states', 'preference_participated_update',
                         'preference_recorded_updates', 'preference_participating_updates',
                         'preference_participation_rate'):
                yield name, row.get(name), dict(context, scope='update')
            if 'pair_filter_stats' not in row:
                yield 'valid_pairs', row.get('valid_pairs'), dict(context, scope='update')
            if 'effective_k_histogram' in row:
                yield 'effective_k_histogram', row['effective_k_histogram'], dict(context, scope='update')
            for item in row.get('historical_tasks', [row]):
                source = dict(context, source_task=item['task'], source_phase=item['source_phase'])
                for name in fields:
                    yield name, item.get(name), source

        self._write('Mechanism', row['phase'], row['task'], row['t_env'], measurements())
        if 'pair_filter_stats' in row:
            totals = row['pair_filter_stats']

            def pair_measurements():
                for name, value in totals.items():
                    yield name, value, dict(context, scope='update')
                for item in row['historical_tasks']:
                    source = dict(context, scope='historical_task', source_task=item['task'],
                                  source_phase=item['source_phase'])
                    for name in totals:
                        yield name, item[name], source

            self._write('Mechanism', row['phase'], row['task'], row['t_env'],
                        pair_measurements(), keep_null=True)


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
