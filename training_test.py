from env import PokemonEnv


def train(num_episodes=10):
    env = PokemonEnv(host="localhost", port=5001)

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs, info = env.reset()
        print(f"reset obs shape: {obs.shape}")
        print(f"reset action_mask: {info['action_mask']}")

        total_reward = 0
        step_count = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action_mask = info['action_mask']
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            import random

            rng = random.Random(123)
            action = rng.choice(valid_actions) if valid_actions else 0

            print(f"\nstep {step_count} sending action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)

            turn = info['raw'].get('turn', -1)
            o = info['raw']['opponent_infos']['opponent_team'][0]
            p = info['raw']['player_infos']['player_team'][0]

            print(f"turn: {turn} | agent(opponent) HP: {o['HP']} / {o['maxHP']} status: {o['status']} | " +
                  f"enemy(player) HP: {p['HP']} / {p['maxHP']} status: {p['status']} | " +
                  f"reward: {reward} | mask: {info['action_mask']}")

            total_reward += reward
            step_count += 1

        status = "terminated" if terminated else "truncated"
        print(f"\nfinished: {status} steps: {step_count} total_reward: {total_reward}")

    env.close()
    print("\n=== Training Complete ===")


if __name__ == "__main__":
    train(num_episodes=5)
