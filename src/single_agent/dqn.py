"""Vanilla CNN-DQN for the shared six-action MinAtar interface."""

from collections import Counter, deque
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from single_agent.preference import preference_result, preference_statistics


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(7, 32, kernel_size=3, stride=1)
        self.hidden = nn.Linear(2048, 256)
        self.output = nn.Linear(256, 6)

    def forward(self, states):
        features = F.relu(self.conv(states))
        features = F.relu(self.hidden(features.flatten(start_dim=1)))
        return self.output(features)


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def clear(self):
        self.data.clear()

    def add(self, state, action, reward, next_state, done):
        self.data.append((np.array(state, dtype=np.float32, copy=True), int(action),
                          float(reward), np.array(next_state, dtype=np.float32, copy=True), bool(done)))

    def sample(self, batch_size, rng):
        return tuple(zip(*rng.sample(self.data, batch_size)))


def gradient_geometry(td_loss, preference_loss, parameters):
    td_grads = torch.autograd.grad(td_loss, parameters, retain_graph=True, allow_unused=True)
    pref_grads = torch.autograd.grad(preference_loss, parameters, retain_graph=True, allow_unused=True)
    td_squared = td_loss.new_zeros(())
    pref_squared = td_loss.new_zeros(())
    dot = td_loss.new_zeros(())
    for td, pref in zip(td_grads, pref_grads):
        if td is not None:
            td_squared = td_squared + td.square().sum()
        if pref is not None:
            pref_squared = pref_squared + pref.square().sum()
        if td is not None and pref is not None:
            dot = dot + (td * pref).sum()
    td_norm = td_squared.sqrt().item()
    pref_norm = pref_squared.sqrt().item()
    return dict(td_grad_norm=td_norm, preference_grad_norm=pref_norm,
                preference_td_grad_ratio=pref_norm / td_norm if td_norm else None,
                td_preference_grad_cosine=(dot.item() / (td_norm * pref_norm)
                                           if td_norm and pref_norm else None))


class DQNLearner:
    METHODS = ("DQN-FT", "DQN-ER", "DQN-TKPR-K1", "DQN-TKPR-K1-W", "DQN-TKPR-K2")

    def __init__(self, config, method, seed=0, device=None):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.config = config
        self.method = method
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork().to(self.device)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["learning_rate"])
        self.replay = ReplayBuffer(config["replay_capacity"])
        self.rng = random.Random(seed)
        self.steps = 0
        self.current_game = None

    def switch_game(self, game):
        changed = self.current_game is not None and game != self.current_game
        if changed and self.method != "DQN-ER":
            self.replay.clear()
        self.current_game = game
        return changed

    def act(self, state, explore=True):
        if explore and self.rng.random() < self.config["epsilon"]:
            return self.rng.randrange(6)
        with torch.no_grad():
            values = self.online(torch.as_tensor(state, dtype=torch.float32,
                                                  device=self.device).unsqueeze(0))
        return int(values.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, terminated, truncated):
        self.replay.add(state, action, reward, next_state, terminated or truncated)
        self.steps += 1
        if self.steps % self.config["target_update_steps"] == 0:
            self.target.load_state_dict(self.online.state_dict())

    def update(self, memory=None, *, diagnostics=False):
        if len(self.replay) < self.config["batch_size"]:
            return None
        states, actions, rewards, next_states, done = self.replay.sample(
            self.config["batch_size"], self.rng)
        states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        selected_q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = rewards + (1 - done) * self.config["gamma"] * self.target(next_states).max(1).values
        td_loss = F.mse_loss(selected_q, targets)
        historical = (memory.sample(self.config["batch_size"], self.current_game, self.rng)
                      if memory is not None and self.method.startswith("DQN-TKPR") else None)
        pref_loss = td_loss.new_zeros(())
        summary = None
        if historical is not None:
            old_states, old_q, sources = historical
            current_q = self.online(old_states.to(self.device))
            pref_loss, _, summary = preference_result(
                self.method, current_q, old_q.to(self.device), self.config["preference"])
        total_loss = td_loss + self.config["preference"]["weight"] * pref_loss
        metrics = dict(total_loss=total_loss.item(), td_loss=td_loss.item(),
                       preference_loss=pref_loss.item(),
                       historical_source_counts=dict(Counter(sources)) if historical is not None else {})
        if diagnostics:
            stats = preference_statistics(self.method, summary) if summary is not None else {}
            metrics.update(constrained_states=stats.get("valid_states", 0),
                           valid_pairs=stats.get("valid_pairs", 0),
                           pair_violation_rate=stats.get("pair_violation_rate"),
                           top1_flip_rate=stats.get("top1_flip_rate"))
            if summary is not None:
                metrics.update(gradient_geometry(td_loss, pref_loss, tuple(self.online.parameters())))
            else:
                metrics.update(td_grad_norm=None, preference_grad_norm=None,
                               preference_td_grad_ratio=None, td_preference_grad_cosine=None)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return metrics
