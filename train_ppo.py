from importlib.util import find_spec

import gymnasium as gym
from sb3_contrib import MaskablePPO
from env import PokemonEnv
from plot import plot

MODEL_PATH = "/Users/condreajason/Repositories/RLPokemon/models/switch_new_ppo"
TOTAL_TIMESTEPS = 100_000
TB_LOG_NAME = "pokemon_switch_run_1"
PLOT_DIR = "/Users/condreajason/Repositories/RLPokemon/plots"
TENSORBOARD_LOG_DIR = "./tensorboard_logs/"

def main() -> None:
    gym.register(
        id="gymnasium_env/Pokemon-v0",
        entry_point="env:PokemonEnv",
        max_episode_steps=300,
    )

    env = PokemonEnv()
    try:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG_DIR,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=TB_LOG_NAME)
        model.save(MODEL_PATH)
        plot(env, PLOT_DIR)
    finally:
        env.close()


if __name__ == "__main__":
    main()
