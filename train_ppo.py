import os
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from env import PokemonEnv


def make_env():
    env = PokemonEnv(host="localhost", port=5001, max_turns=200)
    env = Monitor(env)
    return env


def main():
    run_name = time.strftime("ppo_pokemon_%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    env = make_env()

    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=log_dir,
        name_prefix="ppo_pokemon",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    model.set_logger(new_logger)

    total_timesteps = 200_000

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    finally:
        env.close()

    model_path = os.path.join(log_dir, "ppo_pokemon_final")
    model.save(model_path)
    print("Saved model to:", model_path)

    test_env = make_env()
    obs, info = test_env.reset()
    done = False
    ep_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        ep_reward += reward
        done = terminated or truncated
    print("Test episode reward:", ep_reward)
    test_env.close()


if __name__ == "__main__":
    main()