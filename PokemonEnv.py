import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # Define MDP?

    def reset(self, seed=None, options=None):
        # How do I reset
        return None

    def step(self, action):
        # What is a step?
        return None
