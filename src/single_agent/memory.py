"""First-visit frozen policy references, independent of TD replay."""

import copy
import random

import numpy as np
import torch

from single_agent.minatar_env import make_env


class PreferenceMemory:
    def __init__(self):
        self.tasks = {}

    def add_reference(self, game, states, old_q_values, *, collection_seed, model_stage):
        if game in self.tasks:
            return 0
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32).detach().cpu().clone()
        old_q_values = torch.as_tensor(np.asarray(old_q_values), dtype=torch.float32).detach().cpu().clone()
        if states.ndim != 4 or states.shape[1:] != (7, 10, 10):
            raise ValueError("Reference states must have shape [N, 7, 10, 10]")
        if old_q_values.shape != (len(states), 6):
            raise ValueError("Reference Q must have shape [N, 6]")
        self.tasks[game] = dict(states=states, old_q_values=old_q_values,
                                collection_seed=collection_seed, model_stage=model_stage)
        return len(states)

    def sample(self, batch_size, current_game, rng=random):
        games = [game for game, record in self.tasks.items()
                 if game != current_game and len(record["states"])]
        if not games:
            return None
        rng.shuffle(games)
        states, old_q, sources = [], [], []
        for index in range(batch_size):
            game = games[index % len(games)]
            record = self.tasks[game]
            position = rng.randrange(len(record["states"]))
            states.append(record["states"][position])
            old_q.append(record["old_q_values"][position])
            sources.append(game)
        return torch.stack(states), torch.stack(old_q), sources

    def save(self, game, path):
        torch.save(self.tasks[game], path)


def collect_reference(game, model, steps, seed, stage):
    """Greedy interaction under one frozen task-end model; keep every visited state."""
    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    device = next(frozen.parameters()).device
    env = make_env(game)
    states, old_q_values = [], []
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(steps):
            with torch.no_grad():
                q = frozen(torch.as_tensor(observation, dtype=torch.float32,
                                           device=device).unsqueeze(0))[0]
            states.append(observation.copy())
            old_q_values.append(q.cpu().clone())
            observation, _, terminated, truncated, _ = env.step(int(q.argmax().item()))
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
    return dict(game=game, states=np.stack(states),
                old_q_values=torch.stack(old_q_values),
                collection_seed=seed, model_stage=stage)
