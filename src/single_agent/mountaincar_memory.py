"""Frozen-reference preference states, separate from TD replay."""

import copy
import random

import numpy as np
import torch


class PreferenceMemory:
    def __init__(self):
        self.tasks = {}

    def add_successful_trajectories(self, task, trajectories, model):
        frozen = copy.deepcopy(model).eval().requires_grad_(False)
        states = [state for trajectory in trajectories for state in trajectory]
        if not states:
            self.tasks[task] = []
            return 0
        observations = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        with torch.no_grad():
            old_q = frozen(observations).cpu()
        self.tasks[task] = [(state.copy(), q.clone()) for state, q in zip(states, old_q)]
        return len(states)

    def sample(self, batch_size, rng=random):
        available = [records for records in self.tasks.values() if records]
        if not available:
            return None
        # Equal task contribution; rotate the remainder to avoid a fixed favored task.
        offset = rng.randrange(len(available))
        picks = [rng.choice(available[(i + offset) % len(available)])
                 for i in range(batch_size)]
        states, old_q = zip(*picks)
        return (torch.as_tensor(np.asarray(states), dtype=torch.float32),
                torch.stack(old_q))
