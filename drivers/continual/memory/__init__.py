"""Method-owned history with no dependency on the TD replay policy."""
from collections import Counter
from pathlib import Path
import tempfile

import torch as th


class History(dict):
    """Ordered keyed payloads. The method owns the payload and replacement policy."""

    @staticmethod
    def save(path, value):
        """Write a payload at its existing artifact path without a new wrapper schema."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
        try:
            th.save(value, temp)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    @staticmethod
    def load(path, *, device='cpu'):
        """Load a trusted local method artifact."""
        return th.load(path, map_location=device)


class PreferenceHistory(History):
    """Success pools keyed by phase, with reference labels beside their trajectories."""

    @staticmethod
    def phase_path(root, phase):
        return Path(root) / 'action_set_memory' / f'phase{phase:04d}'

    def restore_success(self, parent, phases, *, require_old_q=False, require_good=False,
                        require_group=False):
        for phase in phases:
            path = self.phase_path(parent, phase) / 'success.th'
            pool = self.load(path)
            if require_old_q and any('old_q_values' not in ep['data'] for ep in pool['episodes']):
                raise ValueError(f'{path} predates stored old_q_values. Convert the historical pool offline before continuing.')
            if require_good:
                for ep in pool['episodes']:
                    shape = ep['data']['avail_actions'].shape
                    if 'good_mask' not in ep or ep['good_mask'].shape != shape:
                        raise ValueError(f'{path} requires a good_mask aligned with avail_actions for ASPR')
                    if require_old_q and ep['data']['old_q_values'].shape != shape:
                        raise ValueError(f'{path} requires old_q_values aligned with avail_actions for ASPR')
            if require_group:
                for ep in pool['episodes']:
                    shape = ep['data']['avail_actions'].shape
                    if 'group_ids' not in ep or ep['group_ids'].shape != shape:
                        raise ValueError(f'{path} requires group_ids aligned with avail_actions')
                    if ep['data']['old_q_values'].shape != shape:
                        raise ValueError(f'{path} requires old_q_values aligned with avail_actions')
            pool['reference_checkpoint'] = str(parent / pool['reference_checkpoint'])
            self[phase] = pool

    def sample(self, current_task, batch_episodes, rng):
        # Preserve the latest nonempty pool per old task, including on revisits.
        pools = {pool['task']: pool for pool in self.values()
                 if pool['task'] != current_task and pool['episodes']}
        if not pools:
            return []
        task_ids = list(pools)
        slots = rng.permutation(len(task_ids)).tolist()
        allocation = Counter(slots[i % len(slots)] for i in range(batch_episodes))
        result = []
        for index, count in allocation.items():
            pool = pools[task_ids[index]]
            size = len(pool['episodes'])
            episode_ids = rng.choice(size, count, replace=count > size).tolist()
            result.append((pool, episode_ids))
        return result
