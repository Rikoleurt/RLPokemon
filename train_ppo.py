import gymnasium as gym
from sb3_contrib import MaskablePPO
from env import PokemonEnv

gym.register(
    id="gymnasium_env/Pokemon-v0",
    entry_point="env:PokemonEnv",
    max_episode_steps=300,
)

env = gym.make("gymnasium_env/Pokemon-v0")

model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# model.save("pokemon_maskable_ppo")