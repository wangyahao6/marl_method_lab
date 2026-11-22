"""Official MinAtar Gymnasium games with a shared CHW observation."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minatar.gym import register_envs


GAMES = ("Breakout", "Freeway", "SpaceInvaders")
ENV_IDS = {game: f"MinAtar/{game}-v0" for game in GAMES}

if ENV_IDS[GAMES[0]] not in gym.envs.registry:
    register_envs()


class PaddedMinAtar(gym.ObservationWrapper):
    def __init__(self, game):
        if game not in ENV_IDS:
            raise ValueError(f"Unknown MinAtar game: {game}")
        env = gym.make(ENV_IDS[game], sticky_action_prob=0.1,
                       difficulty_ramping=False)
        super().__init__(env)
        self.game_name = game
        self.observation_space = spaces.Box(0.0, 1.0, shape=(7, 10, 10),
                                            dtype=np.float32)

    def observation(self, observation):
        channels = np.asarray(observation, dtype=np.float32).transpose(2, 0, 1)
        padded = np.zeros((7, 10, 10), dtype=np.float32)
        padded[:channels.shape[0]] = channels
        return padded


def make_env(game):
    return PaddedMinAtar(game)
