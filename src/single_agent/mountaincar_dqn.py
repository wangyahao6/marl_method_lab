"""Vanilla DQN with an optional historical preference term."""

from collections import deque
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from single_agent.preference import preference_loss


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 128),
                                 nn.ReLU(), nn.Linear(128, 3))

    def forward(self, observations):
        return self.net(observations)


class DQNLearner:
    def __init__(self, config, method, seed=0):
        self.config = config
        self.method = method
        self.online = QNetwork()
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["learning_rate"])
        self.replay = deque(maxlen=config["replay_capacity"])
        self.steps = 0
        self.task_steps = 0
        self.rng = random.Random(seed)

    def switch_task(self):
        self.replay.clear()
        self.task_steps = 0

    def epsilon(self):
        fraction = min(1.0, self.task_steps / self.config["epsilon_decay_steps"])
        return 1.0 + fraction * (self.config["epsilon_final"] - 1.0)

    def act(self, observation, explore=True):
        if explore and self.rng.random() < self.epsilon():
            return self.rng.randrange(3)
        with torch.no_grad():
            q = self.online(torch.as_tensor(observation, dtype=torch.float32))
        return int(q.argmax().item())

    def observe(self, state, action, reward, next_state, terminated, truncated):
        self.replay.append((np.asarray(state, dtype=np.float32), action, reward,
                            np.asarray(next_state, dtype=np.float32), terminated, truncated))
        self.steps += 1
        self.task_steps += 1
        if self.steps % self.config["target_update_steps"] == 0:
            self.target.load_state_dict(self.online.state_dict())

    def update(self, memory=None):
        if len(self.replay) < self.config["batch_size"] or self.task_steps < self.config["learning_starts"]:
            return None
        batch = self.rng.sample(self.replay, self.config["batch_size"])
        states, actions, rewards, next_states, terminated, _ = zip(*batch)
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.float32)
        q = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target = rewards + self.config["gamma"] * (1 - terminated) * self.target(next_states).max(1).values
        td_loss = F.smooth_l1_loss(q, target)
        historical = memory.sample(self.config["batch_size"], self.rng) if memory and self.method != "DQN-FT" else None
        pref_loss = (preference_loss(self.method, self.online(historical[0]), historical[1],
                                     self.config["preference"]) if historical else td_loss.new_zeros(()))
        loss = td_loss + self.config["preference"]["weight"] * pref_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.config["gradient_clip"])
        self.optimizer.step()
        return {"loss": loss.item(), "td_loss": td_loss.item(), "preference_loss": pref_loss.item()}
