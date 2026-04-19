import numpy as np
from env import PokemonEnv

env = PokemonEnv(host="localhost", port=5001)

obs, info = env.reset()
done = False
truncated = False

while not done and not truncated:
    mask = info["action_mask"]
    valid_actions = np.flatnonzero(mask)
    action = np.random.choice(valid_actions)

    obs, reward, done, truncated, info = env.step(action)

print("Masked rollout OK")
env.close()