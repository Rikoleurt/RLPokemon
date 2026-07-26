from __future__ import annotations

from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from data import action_name_for_id
from env import PokemonEnv


MODEL_PATH = Path(
    "/Users/condreajason/Repositories/RLPokemon/"
    "models/global_random_matchups_ppo"
)

HOST = "localhost"
PORT = 5001
DETERMINISTIC = True


def print_agent_action(raw_state: dict, action_id: int) -> None:
    action_name = action_name_for_id(raw_state, action_id)

    print(
        f"[Agent] action={action_id} | choice={action_name}"
    )


def print_battle_result(raw_state: dict) -> None:
    player_infos = raw_state.get("player_infos", {})
    opponent_infos = raw_state.get("opponent_infos", {})

    player_alive = int(player_infos.get("healthy_pokemons", 0))
    agent_alive = int(opponent_infos.get("healthy_pokemons", 0))

    print("\n=== Fight results ===")

    if player_alive <= 0 < agent_alive:
        print("Agent wins")
    elif agent_alive <= 0 < player_alive:
        print("Player wins")
    else:
        print("Battle interrupted")


def main() -> None:
    env = PokemonEnv(
        host=HOST,
        port=PORT,
        max_turns=200,
    )

    try:
        print(f"Loading model at : {MODEL_PATH}")

        model = MaskablePPO.load(
            str(MODEL_PATH),
            env=env,
        )

        print("Server connection...")
        observation, info = env.reset()

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action_mask = env.action_masks()

            action, _state = model.predict(
                observation,
                action_masks=action_mask,
                deterministic=DETERMINISTIC,
            )

            action_id = int(np.asarray(action).item())

            raw_state = info["raw"]
            print_agent_action(raw_state, action_id)

            observation, _reward, terminated, truncated, info = env.step(
                action_id
            )

        print_battle_result(info["raw"])
    except KeyboardInterrupt:
        print("\nKeyboard interruption.")

    finally:
        env.close()


if __name__ == "__main__":
    main()