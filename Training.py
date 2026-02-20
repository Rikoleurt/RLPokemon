import numpy as np
from PokemonAgent import PokemonEnv

# Test
def main():
    env = PokemonEnv(host="localhost", port=5001, max_turns=200)

    obs, info = env.reset()
    print("reset obs shape:", obs.shape)
    print("reset action_mask:", info["action_mask"])

    done = False
    total_reward = 0.0
    step_n = 0

    while not done:
        mask = info["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            print("No valid actions -> forcing 0")
            action = 0
        else:
            action = int(np.random.choice(valid_actions))

        print(f"\nstep {step_n} sending action:", action)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_n += 1
        done = terminated or truncated

        raw = info["raw"]
        p = raw["player_infos"]["player_team"][0]
        o = raw["opponent_infos"]["opponent_team"][0]
        print("turn:", raw["turn"],
              "| agent(opponent) HP:", o["HP"], "/", o["maxHP"], "status:", o["status"],
              "| enemy(player) HP:", p["HP"], "/", p["maxHP"], "status:", p["status"],
              "| reward:", reward,
              "| mask:", info["action_mask"])

    print("\nfinished:", "terminated" if terminated else "truncated", "steps:", step_n, "total_reward:", total_reward)
    env.close()

if __name__ == "__main__":
    main()