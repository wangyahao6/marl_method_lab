"""Replace an entire task pool only when stage-end performance beats its collector."""


def initialize_own_pool(pool, phase, win_rate):
    pool['collection_phase'] = phase
    pool['reference_win_rate'] = win_rate
    for episode in pool['episodes']:
        episode['reference_phase'] = phase
        episode['reference_checkpoint'] = pool['reference_checkpoint']
    return pool


def update_history(history, win_rates, phase, collect, *, capacity=100):
    """Use existing evaluation rows; collect owns isolated rollouts and rollout old-Q."""
    updates = []
    for task, current in win_rates.items():
        existing = next(((key, pool) for key, pool in history.items() if pool['task'] == task), None)
        key, old = existing if existing is not None else (phase, None)
        previous = old['reference_win_rate'] if old is not None else None
        gate = old is None or current > previous
        count = 0
        changed = False
        old_count = len(old['episodes']) if old is not None else 0
        if gate:
            candidate = collect(task)
            candidate['episodes'] = candidate['episodes'][:capacity]
            count = len(candidate['episodes'])
            # No mix of retained and new trajectories. No-success refresh keeps the old pool.
            if old is None or count:
                initialize_own_pool(candidate, phase, current)
                candidate['phase'] = key
                history[key] = candidate
                changed = True
        pool = history[key]
        updates.append(dict(task=task, update_phase=phase, pool_phase=key, initialized=old is None,
                            current_win_rate=current, previous_reference_win_rate=previous,
                            gate_passed=gate, collection_attempted=gate, pool_changed=changed,
                            old_count=old_count, new_success_count=count,
                            replaced_count=old_count if changed and old is not None else 0,
                            final_size=len(pool['episodes']), reference_win_rate=pool['reference_win_rate'],
                            reference_phase=pool['collection_phase'], reference_checkpoint=pool['reference_checkpoint']))
    return updates
