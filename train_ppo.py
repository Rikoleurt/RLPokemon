import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from env import PokemonEnv
from data import get_attack_names

gym.register(
    id="gymnasium_env/Pokemon-v0",
    entry_point="env:PokemonEnv",
    max_episode_steps=300,
)

MODEL_PATH = "/Users/condreajason/Repositories/RLPokemon/models/pikachu_salameche_duel2_ppo"
TOTAL_TIMESTEPS = 100_000
TB_LOG_NAME = "pokemon_run_1"
PLOT_DIR = "/Users/condreajason/Repositories/RLPokemon/plots"


def plot(env: PokemonEnv, plot_dir: str = PLOT_DIR) -> None:
    """
    Save cumulative and moving attack usage / win rate plots collected during training.
    """

    os.makedirs(plot_dir, exist_ok=True)

    episodes = np.arange(1, len(env.winrate_history) + 1)

    attack_usage_path = os.path.join(plot_dir, "attack_usage_cumulative.png")
    attack_usage_moving_path = os.path.join(plot_dir, "attack_usage_moving.png")
    winrate_path = os.path.join(plot_dir, "winrate_cumulative.png")
    winrate_moving_path = os.path.join(plot_dir, "winrate_moving.png")

    attack_labels = [f"Attack {i}" for i in range(4)]
    if env.last_msg is not None:
        attack_labels = get_attack_names(env.last_msg)

    plt.figure(figsize=(10, 6))
    for action_id in range(4):
        plt.plot(
            episodes,
            env.attack_usage_history[action_id],
            label=attack_labels[action_id],
        )
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative usage (%)")
    plt.title("Attack usage over episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(attack_usage_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Usage glissant
    plt.figure(figsize=(10, 6))
    for action_id in range(4):
        plt.plot(
            episodes,
            env.attack_usage_moving_history[action_id],
            label=attack_labels[action_id],
        )
    plt.xlabel("Episodes")
    plt.ylabel(f"Moving usage over last {env.window_size} episodes (%)")
    plt.title("Moving attack usage over episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(attack_usage_moving_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Winrate cumulé
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, env.winrate_history, label="Win rate")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative win rate (%)")
    plt.title("Win rate over episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(winrate_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Winrate glissant
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, env.winrate_moving_history, label="Moving win rate")
    plt.xlabel("Episodes")
    plt.ylabel(f"Moving win rate over last {env.window_size} episodes (%)")
    plt.title("Moving win rate over episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(winrate_moving_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Graph saved: {attack_usage_path}")
    print(f"Graph saved: {attack_usage_moving_path}")
    print(f"Graph saved: {winrate_path}")
    print(f"Graph saved: {winrate_moving_path}")


if __name__ == "__main__":
    env = PokemonEnv()

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=TB_LOG_NAME)
    model.save(MODEL_PATH)

    plot(env)
    env.close()
