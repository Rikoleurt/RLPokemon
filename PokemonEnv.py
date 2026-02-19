import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

    def reset(self, seed=None, options=None):
        return None

    def step(self, action):
        return None
