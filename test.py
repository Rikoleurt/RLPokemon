from copy import deepcopy

import numpy as np

from battle_stat_tracker import BattleStatsTracker
from data import build_action_mask, get_moves_data_from_json, json_to_obs

try:
    from env import PokemonEnv, build_observation_bounds, observation_size
except ModuleNotFoundError:
    PokemonEnv = None
    visible_pokemon_slots = 3
    pokemon_feature_count = 14
    battle_scalar_count = 4
    attack_actions = 4
    move_feature_count = 13
    observation_size = (
        visible_pokemon_slots * pokemon_feature_count
        + battle_scalar_count
        + attack_actions * move_feature_count
    )

    def build_observation_bounds(max_turns: int):
        type_upper = 18.0
        status_upper = 10.0
        mode_upper = 2.0
        target_upper = 2.0
        statistic_upper = 5.0

        pokemon_low = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            -1.0, -1.0, -1.0, -1.0, -1.0,
        ]
        pokemon_high = [
            1.0, type_upper, type_upper, status_upper,
            1.5, 1.5, 1.5, 1.5, 1.5,
            1.0, 1.0, 1.0, 1.0, 1.0,
        ]
        battle_low = [0.0] * battle_scalar_count
        battle_high = [1.0, float(max_turns), 6.0, 6.0]
        move_low = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            -1.0, 0.0,
        ]
        move_high = [
            255.0, type_upper, mode_upper,
            1.0, 1.0, 1.0, 1.0,
            status_upper, 1.0, target_upper, statistic_upper,
            1.0, 1.0,
        ]
        low = np.array(
            pokemon_low * visible_pokemon_slots
            + battle_low
            + move_low * attack_actions,
            dtype=np.float32,
        )
        high = np.array(
            pokemon_high * visible_pokemon_slots
            + battle_high
            + move_high * attack_actions,
            dtype=np.float32,
        )
        return low, high


sample_msg = {
    "turn": 7,
    "Priority": {"name": "rl-agent"},
    "action_feedback": {},
    "player_infos": {
        "name": "player",
        "healthy_pokemons": 2,
        "player_team": [
            {
                "name": "Bulbasaur",
                "HP": 30,
                "maxHP": 45,
                "level": 25,
                "type": "grass",
                "type2": "poison",
                "status": "normal",
                "stats": {
                    "atk": 49,
                    "def": 49,
                    "atkSpe": 65,
                    "defSpe": 65,
                    "speed": 45,
                    "statisticsStages": {
                        "atk": -2,
                        "def": 0,
                        "atkSpe": 0,
                        "defSpe": 0,
                        "speed": 1,
                    },
                },
                "attacks": [
                    {
                        "slot": 0,
                        "id": 11,
                        "name": "Vine Whip",
                        "type": "grass",
                        "Mode": "physical",
                        "Power": 45,
                        "Precision": 100,
                        "PP": 10,
                        "maxPP": 10,
                        "isSTAB": True,
                    }
                ],
            },
            {
                "name": "Pidgey",
                "HP": 20,
                "maxHP": 40,
                "level": 22,
                "type": "normal",
                "type2": "flying",
                "status": "normal",
                "stats": {
                    "atk": 45,
                    "def": 40,
                    "atkSpe": 35,
                    "defSpe": 35,
                    "speed": 56,
                    "statisticsStages": {},
                },
                "attacks": [],
            },
        ],
    },
    "opponent_infos": {
        "name": "rl-agent",
        "healthy_pokemons": 2,
        "opponent_team": [
            {
                "name": "Charmander",
                "HP": 28,
                "maxHP": 39,
                "level": 25,
                "type": "fire",
                "type2": None,
                "status": "normal",
                "stats": {
                    "atk": 52,
                    "def": 43,
                    "atkSpe": 60,
                    "defSpe": 50,
                    "speed": 65,
                    "statisticsStages": {
                        "atk": 2,
                        "def": 0,
                        "atkSpe": 1,
                        "defSpe": 0,
                        "speed": 0,
                    },
                },
                "attacks": [
                    {
                        "slot": 0,
                        "id": 52,
                        "name": "Ember",
                        "type": "fire",
                        "Mode": "special",
                        "Power": 40,
                        "Precision": 75,
                        "PP": 25,
                        "maxPP": 25,
                        "isSTAB": True,
                        "Status": "burned",
                        "ChanceOfApplyingStatus": 0.4,
                    },
                    {
                        "slot": 1,
                        "id": 14,
                        "name": "Swords Dance",
                        "type": "normal",
                        "Mode": "status",
                        "PP": 20,
                        "maxPP": 20,
                        "isSTAB": False,
                        "Target": "self",
                        "Statistic": "atk",
                        "StageDelta": 2,
                    },
                    {
                        "slot": 2,
                        "id": 86,
                        "name": "Thunder Wave",
                        "type": "electric",
                        "Mode": "status",
                        "Precision": 90,
                        "PP": 20,
                        "maxPP": 20,
                        "isSTAB": False,
                        "Status": "paralyzed",
                    },
                ],
            },
            {
                "name": "Squirtle",
                "HP": 44,
                "maxHP": 44,
                "level": 24,
                "type": "water",
                "type2": None,
                "status": "normal",
                "stats": {
                    "atk": 48,
                    "def": 65,
                    "atkSpe": 50,
                    "defSpe": 64,
                    "speed": 43,
                    "statisticsStages": {},
                },
                "attacks": [],
            },
        ],
    },
}


def run_observation_smoke_test():
    obs = json_to_obs(sample_msg)
    low, high = build_observation_bounds(max_turns=200)
    full_action_mask = build_action_mask(sample_msg)
    _, attack_mask, _, move_features = get_moves_data_from_json(sample_msg)

    assert obs.shape == (observation_size,)
    assert low.shape == obs.shape == high.shape
    assert move_features.shape == (4, 13)
    assert attack_mask.tolist() == [1, 1, 1, 0]
    assert full_action_mask.tolist() == [1, 1, 1, 0, 1, 1]

    assert np.isclose(obs[9], -2.0 / 6.0)
    assert np.isclose(obs[13], 1.0 / 6.0)
    assert np.isclose(obs[23], 2.0 / 6.0)
    assert np.isclose(obs[25], 1.0 / 6.0)

    assert np.isclose(move_features[0, 7], 2.0)
    assert np.isclose(move_features[0, 8], 0.3)
    assert np.isclose(move_features[1, 4], 1.0)
    assert np.isclose(move_features[1, 9], 1.0)
    assert np.isclose(move_features[1, 10], 1.0)
    assert np.isclose(move_features[1, 11], 2.0 / 6.0)
    assert np.isclose(move_features[1, 12], 1.0)
    assert np.isclose(move_features[2, 8], 0.9)

    print("Observation smoke test OK")


def run_action_mask_fallback_smoke_test():
    exhausted_msg = deepcopy(sample_msg)
    exhausted_front = exhausted_msg["opponent_infos"]["opponent_team"][0]

    exhausted_msg["opponent_infos"]["opponent_team"] = [exhausted_front]
    exhausted_msg["opponent_infos"]["healthy_pokemons"] = 1
    exhausted_front["HP"] = exhausted_front["maxHP"]

    for move in exhausted_front["attacks"]:
        move["PP"] = 0

    fallback_mask = build_action_mask(exhausted_msg)
    assert fallback_mask.tolist() == [1, 1, 1, 0, 0, 0]

    print("Action mask fallback smoke test OK")


def run_tracker_smoke_test():
    tracker = BattleStatsTracker(
        window_size=10,
        attack_actions=4,
        total_actions=6,
        switch_action=4,
        item_action=5,
    )

    class DummyMatchupEvaluator:
        @staticmethod
        def effectiveness_to_string(move_type: str, defender_type1: str, defender_type2: str | None = None) -> str:
            return "neutral"

    tracker.record_action(0)
    tracker.record_attack_context(sample_msg, 0, DummyMatchupEvaluator())

    status_current_msg = deepcopy(sample_msg)
    status_current_msg["player_infos"]["player_team"][0]["status"] = "paralyzed"
    tracker.record_action(2)
    tracker.record_attack_context(sample_msg, 2, DummyMatchupEvaluator())
    tracker.record_move_outcome(sample_msg, status_current_msg, 2)

    setup_current_msg = deepcopy(sample_msg)
    setup_current_msg["opponent_infos"]["opponent_team"][0]["stats"]["statisticsStages"]["atk"] = 4
    tracker.record_action(1)
    tracker.record_attack_context(sample_msg, 1, DummyMatchupEvaluator())
    tracker.record_move_outcome(sample_msg, setup_current_msg, 1)

    tracker.finalize_episode(did_win=False, fight_length=5, terminal_msg=setup_current_msg)

    assert tracker.move_category_counter["attack"] == 1
    assert tracker.move_category_counter["status"] == 1
    assert tracker.move_category_counter["setup"] == 1
    assert tracker.status_inflicted_counter["paralyzed"] == 1
    assert tracker.setup_stat_boost_counter["atk"] == 1
    assert tracker.move_category_history_by_episode[0]["attack"] == 1
    assert tracker.move_category_history_by_episode[0]["status"] == 1
    assert tracker.move_category_history_by_episode[0]["setup"] == 1

    print("Tracker smoke test OK")


def run_live_rollout():
    if PokemonEnv is None:
        print("Live rollout skipped: gymnasium is not installed")
        return

    env = PokemonEnv(host="localhost", port=5001)

    try:
        obs, info = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            mask = info["action_mask"]
            valid_actions = np.flatnonzero(mask)
            action = np.random.choice(valid_actions)

            obs, reward, done, truncated, info = env.step(action)

        print("Masked rollout OK")
    except OSError as exc:
        print(f"Live rollout skipped: {exc}")
    finally:
        env.close()


if __name__ == "__main__":
    run_observation_smoke_test()
    run_action_mask_fallback_smoke_test()
    run_tracker_smoke_test()
    run_live_rollout()
