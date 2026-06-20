import json
import socket

import numpy as np


# region constants
types = {
    "normal": 0, "fire": 1, "water": 2, "grass": 3, "electric": 4, "ice": 5,
    "fighting": 6, "poison": 7, "ground": 8, "flying": 9, "psychic": 10,
    "bug": 11, "rock": 12, "ghost": 13, "dragon": 14, "dark": 15,
    "steel": 16, "fairy": 17,
}

modes = {"physical": 0, "special": 1, "status": 2}

status = {
    "normal": 0,
    "KO": 1,
    "burned": 2,
    "paralyzed": 3,
    "freeze": 4,
    "asleep": 5,
    "poisoned": 6,
    "badlyPoisoned": 7,
    "confused": 8,
    "attracted": 9,
    "cursed": 10,
}

targets = {
    "none": 0,
    "self": 1,
    "opponent": 2,
}

statistics = {
    "none": 0,
    "atk": 1,
    "def": 2,
    "atkSpe": 3,
    "defSpe": 4,
    "speed": 5,
}

locked_id = 255
max_moves = 4
switch_action = 4
item_action = 5
n_actions = 6
# endregion
# region getters
# This functions centralizes the JSON access.

# ---------- Root getters ----------

def get_turn(msg: dict) -> int:
    return int(msg.get("turn", 0))


def get_priority_name(msg: dict) -> str:
    return msg.get("Priority", {}).get("name", "")


def get_action_feedback(msg: dict) -> dict:
    return msg.get("action_feedback", {})


def get_opponent_invalid_action(msg: dict) -> bool:
    return bool(get_action_feedback(msg).get("opponent_invalid", False))


# ---------- Trainer getters ----------

def get_player_infos(msg: dict) -> dict:
    return msg.get("player_infos", {})


def get_agent_infos(msg: dict) -> dict:
    """
    In the current Java JSON, the RL agent is stored in opponent_infos.
    """
    return msg.get("opponent_infos", {})


def get_enemy_infos(msg: dict) -> dict:
    """
    Compatibility alias: the enemy is currently the player side in the JSON.
    """
    return get_player_infos(msg)


def get_player_name(msg: dict) -> str:
    return get_player_infos(msg).get("name", "player")


def get_agent_name(msg: dict) -> str:
    return get_agent_infos(msg).get("name", "opponent")


def get_enemy_name(msg: dict) -> str:
    return get_player_name(msg)


def get_player_team(msg: dict) -> list[dict]:
    return get_player_infos(msg).get("player_team", [])


def get_agent_team(msg: dict) -> list[dict]:
    return get_agent_infos(msg).get("opponent_team", [])


def get_enemy_team(msg: dict) -> list[dict]:
    return get_player_team(msg)


def get_player_healthy_pokemons(msg: dict) -> int:
    return int(get_player_infos(msg).get("healthy_pokemons", 0))


def get_agent_healthy_pokemons(msg: dict) -> int:
    return int(get_agent_infos(msg).get("healthy_pokemons", 0))


def get_enemy_healthy_pokemons(msg: dict) -> int:
    return get_player_healthy_pokemons(msg)


# ---------- Active / benched Pokémon getters ----------

def get_player_front(msg: dict) -> dict | None:
    team = get_player_team(msg)
    return team[0] if len(team) > 0 else None


def get_agent_front(msg: dict) -> dict | None:
    team = get_agent_team(msg)
    return team[0] if len(team) > 0 else None


def get_enemy_front(msg: dict) -> dict | None:
    """
    Compatibility alias for previous code.
    """
    return get_player_front(msg)


def get_player_back(msg: dict, index: int = 1) -> dict | None:
    team = get_player_team(msg)
    return team[index] if len(team) > index else None


def get_agent_back(msg: dict, index: int = 1) -> dict | None:
    team = get_agent_team(msg)
    return team[index] if len(team) > index else None


def get_enemy_back(msg: dict, index: int = 1) -> dict | None:
    return get_player_back(msg, index)


def get_active_matchup(msg: dict) -> tuple[dict | None, dict | None]:
    """
    Returns the active battle matchup in the historical order:
        enemy_front, agent_front
    """
    return get_enemy_front(msg), get_agent_front(msg)


# ---------- Pokémon generic getters ----------

def get_pokemon_name(pokemon: dict | None) -> str:
    if pokemon is None:
        return "unknown"
    return pokemon.get("name") or pokemon.get("species") or "unknown"


def get_pokemon_hp(pokemon: dict | None) -> float:
    if pokemon is None:
        return 0.0
    return float(pokemon.get("HP", 0.0))


def get_pokemon_max_hp(pokemon: dict | None) -> float:
    if pokemon is None:
        return 1.0
    return float(max(1.0, pokemon.get("maxHP", 1.0)))


def get_pokemon_hp_ratio(pokemon: dict | None) -> float:
    return get_pokemon_hp(pokemon) / get_pokemon_max_hp(pokemon)


def get_pokemon_level(pokemon: dict | None) -> int:
    if pokemon is None:
        return 0
    return int(pokemon.get("level", 0))


def get_pokemon_type_1(pokemon: dict | None) -> str | None:
    if pokemon is None:
        return None
    return pokemon.get("type")


def get_pokemon_type_2(pokemon: dict | None) -> str | None:
    if pokemon is None:
        return None
    return pokemon.get("type2")


def get_pokemon_status(pokemon: dict | None) -> str:
    if pokemon is None:
        return "KO"
    return pokemon.get("status", "normal")


def get_pokemon_attacks(pokemon: dict | None) -> list[dict]:
    if pokemon is None:
        return []
    return pokemon.get("attacks", [])


def is_pokemon_alive(pokemon: dict | None) -> bool:
    return pokemon is not None and get_pokemon_status(pokemon) != "KO"


# ---------- Pokémon stat getters ----------

def get_stat_block(pokemon: dict | None) -> dict:
    if pokemon is None:
        return {}
    return pokemon.get("stats", {})


def get_pokemon_atk(pokemon: dict | None) -> float:
    stats = get_stat_block(pokemon)
    return float(stats.get("atk", pokemon.get("atk", 0.0) if pokemon else 0.0))


def get_pokemon_def(pokemon: dict | None) -> float:
    stats = get_stat_block(pokemon)
    return float(stats.get("def", pokemon.get("def", 0.0) if pokemon else 0.0))


def get_pokemon_atk_spe(pokemon: dict | None) -> float:
    stats = get_stat_block(pokemon)
    return float(stats.get("atkSpe", pokemon.get("atkSpe", 0.0) if pokemon else 0.0))


def get_pokemon_def_spe(pokemon: dict | None) -> float:
    stats = get_stat_block(pokemon)
    return float(stats.get("defSpe", pokemon.get("defSpe", 0.0) if pokemon else 0.0))


def get_pokemon_speed(pokemon: dict | None) -> float:
    stats = get_stat_block(pokemon)
    return float(stats.get("speed", pokemon.get("speed", 0.0) if pokemon else 0.0))


def get_statistics_stages(pokemon: dict | None) -> dict:
    return get_stat_block(pokemon).get("statisticsStages", {})


def get_stat_stage(pokemon: dict | None, statistic: str | None) -> int:
    if pokemon is None or statistic is None:
        return 0

    stages = get_statistics_stages(pokemon)
    return int(stages.get(str(statistic), 0))


# ---------- Move list getters ----------

def get_agent_moves(msg: dict) -> list[dict]:
    return get_pokemon_attacks(get_agent_front(msg))


def get_player_moves(msg: dict) -> list[dict]:
    return get_pokemon_attacks(get_player_front(msg))


def get_enemy_moves(msg: dict) -> list[dict]:
    return get_player_moves(msg)


def get_move_by_slot(moves: list[dict], slot: int) -> dict | None:
    for move in moves:
        if move.get("slot") == slot:
            return move
    return None


def get_agent_move_by_slot(msg: dict, slot: int) -> dict | None:
    return get_move_by_slot(get_agent_moves(msg), slot)


def get_player_move_by_slot(msg: dict, slot: int) -> dict | None:
    return get_move_by_slot(get_player_moves(msg), slot)


def get_enemy_move_by_slot(msg: dict, slot: int) -> dict | None:
    return get_player_move_by_slot(msg, slot)


def move_for_action_id(msg: dict, action_id: int, maximum: int = max_moves) -> dict | None:
    """
    Compatibility function used by existing files.
    Returns the agent move associated with an action id.
    """
    if not (0 <= action_id < maximum):
        return None
    return get_agent_move_by_slot(msg, action_id)


def get_agent_attack_names(msg: dict, maximum: int = max_moves) -> list[str]:
    names = [f"Attack {i}" for i in range(maximum)]

    for move in get_agent_moves(msg):
        slot = get_move_slot(move)
        if 0 <= slot < maximum:
            names[slot] = get_move_name(move) or f"Attack {slot}"

    return names


def get_attack_names(msg: dict, maximum: int = max_moves) -> list[str]:
    """
    Compatibility alias for previous code.
    """
    return get_agent_attack_names(msg, maximum)


# ---------- Move generic getters ----------

def get_move_slot(move: dict | None) -> int:
    if move is None:
        return -1
    return int(move.get("slot", -1))


def get_move_id(move: dict | None, default: int = locked_id) -> int:
    if move is None:
        return default
    return int(move.get("id", default))


def get_move_name(move: dict | None) -> str:
    if move is None:
        return ""
    return move.get("name", "")


def get_move_type(move: dict | None) -> str:
    if move is None:
        return "normal"
    return move.get("type", "normal")


def get_move_mode(move: dict | None) -> str:
    if move is None:
        return "status"
    return move.get("Mode", "status")


def get_move_power(move: dict | None) -> float:
    if move is None:
        return 0.0
    return float(move.get("Power", 0.0))


def get_move_precision(move: dict | None) -> float:
    if move is None:
        return 0.0
    return float(move.get("Precision", 0.0))


def get_move_pp(move: dict | None) -> float:
    if move is None:
        return 0.0
    return float(move.get("PP", 0.0))


def get_move_max_pp(move: dict | None) -> float:
    if move is None:
        return 1.0
    return float(max(1.0, move.get("maxPP", 1.0)))


def get_move_pp_ratio(move: dict | None) -> float:
    return get_move_pp(move) / get_move_max_pp(move)


def get_move_is_stab(move: dict | None) -> bool:
    if move is None:
        return False
    return bool(move.get("isSTAB", False))


# ---------- Move status getters ----------

def get_move_status(move: dict | None) -> str | None:
    if move is None:
        return None
    return move.get("Status")


def get_move_status_chance(move: dict | None) -> float:
    if move is None:
        return 0.0
    return float(move.get("ChanceOfApplyingStatus", 0.0))


def move_has_status_effect(move: dict | None) -> bool:
    return get_move_status(move) is not None


# ---------- Move stat stage getters ----------

def get_move_target(move: dict | None) -> str:
    if move is None:
        return "none"
    return str(move.get("Target", "none")).lower()


def get_move_statistic(move: dict | None) -> str | None:
    if move is None:
        return None
    return move.get("Statistic")


def get_move_stage_delta(move: dict | None) -> int:
    if move is None:
        return 0
    return int(move.get("StageDelta", 0))


def move_has_stage_effect(move: dict | None) -> bool:
    return get_move_statistic(move) is not None and get_move_stage_delta(move) != 0


# ---------- Action name getters ----------

def action_name_for_id(msg: dict, action_id: int) -> str:
    if action_id == switch_action:
        return "Switch"

    if action_id == item_action:
        return "Item"

    if 0 <= action_id < max_moves:
        move = move_for_action_id(msg, action_id)
        if move is not None:
            return get_move_name(move) or f"Attack {action_id}"
        return f"Attack {action_id}"

    return f"Attack {action_id}"


def get_action_name(msg: dict, action_id: int) -> str:
    """
    New explicit name, kept alongside action_name_for_id.
    """
    return action_name_for_id(msg, action_id)
# endregion
# region helpers
def type_id(value: str | None) -> float:
    if value is None:
        return float(len(types))
    return float(types.get(str(value).lower(), len(types)))


def status_id(value: str | None) -> float:
    if value is None:
        return 0.0
    return float(status.get(str(value), 0))


def target_id(value: str | None) -> float:
    if value is None:
        return float(targets["none"])
    return float(targets.get(str(value).lower(), targets["none"]))


def statistic_id(value: str | None) -> float:
    if value is None:
        return float(statistics["none"])
    return float(statistics.get(str(value), statistics["none"]))


def stat_norm(value: float, denominator: float = 255.0) -> float:
    return float(value) / denominator


def normalized_stage(value: float | int | None) -> float:
    """
    Stages Pokémon: -6 to +6.
    Normalized to [-1, 1].
    """
    if value is None:
        return 0.0
    return float(value) / 6.0


def move_target(move: dict | None) -> str:
    """
    Returns the target of a move.

    Attack and StatusAttack target the opponent by convention.
    SetUpMove exposes Target directly in JSON.
    """
    if move is None:
        return "none"

    explicit_target = get_move_target(move)
    if explicit_target != "none":
        return explicit_target

    if get_move_status(move) is not None:
        return "opponent"

    mode = str(get_move_mode(move)).lower()
    power = get_move_power(move)

    if mode in ("physical", "special") or power > 0:
        return "opponent"

    return "none"


def move_status_effect(move: dict | None) -> str | None:
    value = get_move_status(move)
    if value is None:
        return None
    return str(value)


def move_status_probability(move: dict | None) -> float:
    """
    Approximate probability that the move applies its status.

    StatusAttack:
        chance = Precision / 100

    Attack with secondary status:
        chance = Precision / 100 * ChanceOfApplyingStatus
    """
    if not move_has_status_effect(move):
        return 0.0

    precision = get_move_precision(move) / 100.0
    mode = str(get_move_mode(move)).lower()

    if mode == "status":
        return precision

    return precision * get_move_status_chance(move)


def can_apply_status(move: dict | None) -> bool:
    """
    True if the move can apply a major status.
    """
    return move_status_effect(move) is not None


def move_statistic(move: dict | None) -> str | None:
    value = get_move_statistic(move)
    if value is None:
        return None
    return str(value)


def move_stage_delta(move: dict | None) -> int:
    return get_move_stage_delta(move)


def modifies_stat_stage(move: dict | None) -> bool:
    """
    True if the move modifies a stat stage.
    """
    return move_statistic(move) is not None and move_stage_delta(move) != 0


def stage_move_target_pokemon(
    move: dict | None,
    agent_front: dict | None,
    enemy_front: dict | None,
) -> dict | None:
    target = move_target(move)

    if target == "self":
        return agent_front

    if target == "opponent":
        return enemy_front

    return None


def would_stage_move_have_effect(
    move: dict | None,
    agent_front: dict | None,
    enemy_front: dict | None,
) -> bool:
    """
    False if the stat stage is already capped at -6 or +6.
    """
    if not modifies_stat_stage(move):
        return False

    statistic = move_statistic(move)
    delta = move_stage_delta(move)
    target_pokemon = stage_move_target_pokemon(move, agent_front, enemy_front)
    current_stage = get_stat_stage(target_pokemon, statistic)

    if delta > 0 and current_stage >= 6:
        return False

    if delta < 0 and current_stage <= -6:
        return False

    return True
# endregion
# region action_mask
def build_action_mask(
    msg: dict,
    total_actions: int = n_actions,
    attack_actions: int = max_moves,
    switch_slot: int = switch_action,
    item_slot: int = item_action,
) -> np.ndarray:
    """
    Builds the full action mask:
        0-3: move actions
        4: switch action
        5: item action
    """
    mask = np.zeros((total_actions,), dtype=np.int8)

    agent_team = get_agent_team(msg)
    agent_front = get_agent_front(msg)
    if agent_front is None:
        return mask

    for move in get_agent_moves(msg):
        slot = get_move_slot(move)
        if 0 <= slot < attack_actions:
            mask[slot] = 1 if get_move_pp(move) > 0 else 0

    if len(agent_team) > 1:
        for pokemon in agent_team[1:]:
            if is_pokemon_alive(pokemon):
                mask[switch_slot] = 1
                break

    if get_pokemon_hp(agent_front) < get_pokemon_max_hp(agent_front):
        mask[item_slot] = 1

    return mask
# endregion
# region data processing for env.py
# This region process the data to later create the obs, truncated and terminated variables.
def pokemon_features(pokemon: dict | None) -> list[float]:
    """
    Get the statistics of a Pokémon.

    :param pokemon: JSON block representing a Pokémon.
    :return: a list of floats that describes the Pokémon.
    """
    if pokemon is None:
        return [
            0.0,
            type_id("normal"),
            type_id(None),
            status_id("KO"),
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]

    return [
        get_pokemon_hp_ratio(pokemon),
        type_id(get_pokemon_type_1(pokemon) or "normal"),
        type_id(get_pokemon_type_2(pokemon)),
        status_id(get_pokemon_status(pokemon)),
        stat_norm(get_pokemon_atk(pokemon)),
        stat_norm(get_pokemon_def(pokemon)),
        stat_norm(get_pokemon_atk_spe(pokemon)),
        stat_norm(get_pokemon_def_spe(pokemon)),
        stat_norm(get_pokemon_speed(pokemon)),
    ]


def get_moves_data_from_json(
    msg: dict,
    maximum: int = max_moves,
    identification: int = locked_id,
):
    attacks = get_agent_moves(msg)

    move_ids = np.full((maximum,), identification, dtype=np.int32)
    action_mask = np.zeros((maximum,), dtype=np.int8)
    move_names = [""] * maximum

    # [id, type_id, mode_id, power_norm, precision_norm, pp_norm, is_stab]
    move_features = np.zeros((maximum, 7), dtype=np.float32)
    move_features[:, 0] = float(identification)

    for attack in attacks:
        slot = get_move_slot(attack)
        if not (0 <= slot < maximum):
            continue

        move_id = get_move_id(attack, identification)
        move_name = get_move_name(attack)
        move_type = types.get(str(get_move_type(attack)).lower(), len(types))
        move_mode = modes.get(str(get_move_mode(attack)).lower(), modes["status"])

        power_norm = get_move_power(attack) / 150.0
        precision_norm = get_move_precision(attack) / 100.0
        pp_norm = get_move_pp_ratio(attack)
        is_stab = 1.0 if get_move_is_stab(attack) else 0.0

        move_ids[slot] = move_id
        move_names[slot] = move_name
        move_features[slot] = np.array(
            [
                float(move_id),
                float(move_type),
                float(move_mode),
                power_norm,
                precision_norm,
                pp_norm,
                is_stab,
            ],
            dtype=np.float32,
        )
        action_mask[slot] = 1 if get_move_pp(attack) > 0 else 0

    compact_names = [name for name, mask_value in zip(move_names, action_mask) if mask_value == 1]
    return move_ids, action_mask, compact_names, move_features
# endregion
# region env.py public API
# These functions are the most important for the env.py file. It gathers the preprocessed data into the obs, truncated
# and terminated variables.
def json_to_obs(msg: dict) -> np.ndarray:
    enemy_front = get_enemy_front(msg)
    agent_front = get_agent_front(msg)
    agent_back = get_agent_back(msg)

    agent_first = json_to_agent_first(msg)
    turn = float(get_turn(msg))
    enemy_healthy = float(get_enemy_healthy_pokemons(msg))
    agent_healthy = float(get_agent_healthy_pokemons(msg))

    _, _, _, move_features = get_moves_data_from_json(msg)

    obs = np.array(
        pokemon_features(enemy_front)
        + pokemon_features(agent_front)
        + pokemon_features(agent_back)
        + [agent_first, turn, enemy_healthy, agent_healthy]
        + move_features.flatten().tolist(),
        dtype=np.float32,
    )

    return obs


def json_to_agent_first(msg: dict) -> float:
    return 1.0 if get_priority_name(msg) == get_agent_name(msg) else 0.0


def json_to_action_mask(msg: dict) -> np.ndarray:
    """
    Historical action mask used by the current environment.
    It only returns the 4 attack actions to avoid changing dependencies.

    Use build_action_mask(msg) when a full 6-action mask is needed.
    """
    _, action_mask, _, _ = get_moves_data_from_json(msg)
    return action_mask


def json_to_terminated(msg: dict) -> bool:
    enemy_alive = get_enemy_healthy_pokemons(msg) > 0
    agent_alive = get_agent_healthy_pokemons(msg) > 0
    return (not enemy_alive) or (not agent_alive)


def json_to_truncated(msg: dict) -> bool:
    """
    Gymnasium-compatible truncated flag.

    The Java battle JSON currently exposes terminal battle state,
    but no explicit time-limit / truncation signal.
    """
    return False


def json_to_invalid_action_flag(msg: dict) -> float:
    return 1.0 if get_opponent_invalid_action(msg) else 0.0
# endregion
# region client
def main():
    host = "localhost"
    port = 5001

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print("Connected to server")

            f = s.makefile("r", encoding="utf-8")
            while True:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    json_obj = json.loads(line)
                    print("Received :", json_obj)
                    print(json_to_obs(json_obj))
                    print("Mask:", json_to_action_mask(json_obj))
                    print("Invalid:", get_action_feedback(json_obj))

    except ConnectionRefusedError:
        print("Impossible to connect to server : Server unavailable")
    except ConnectionResetError:
        print("Connection has been closed")
    except KeyboardInterrupt:
        print("Closing python client")
# endregion
if __name__ == "__main__":
    main()
