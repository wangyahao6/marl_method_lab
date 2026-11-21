"""MountainCar tasks with a shared three-dimensional observation."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


GOALS = {"T1": 0.45, "T2": 0.50, "T3": 0.55}


class GoalMountainCar(gym.ObservationWrapper):
    def __init__(self, task):
        if task not in GOALS:
            raise ValueError(f"Unknown task: {task}")
        env = gym.make("MountainCar-v0", max_episode_steps=200)
        env.unwrapped.goal_position = GOALS[task]
        env.unwrapped.goal_velocity = 0.0
        env.unwrapped.force = 0.001
        env.unwrapped.gravity = 0.0025
        super().__init__(env)
        low = np.append(env.observation_space.low, -1.0).astype(np.float32)
        high = np.append(env.observation_space.high, 1.0).astype(np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.task = task

    def observation(self, observation):
        return np.array([observation[0], observation[1],
                         (GOALS[self.task] - 0.50) / 0.05], dtype=np.float32)
