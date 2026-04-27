import json
import socket
import numpy as np

types = {
    "normal": 0, "fire": 1, "water": 2, "grass": 3, "electric": 4, "ice": 5,
    "fighting": 6, "poison": 7, "ground": 8, "flying": 9, "psychic": 10,
    "bug": 11, "rock": 12, "ghost": 13, "dragon": 14, "dark": 15,
    "steel": 16, "fairy": 17
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

locked_id = 255
max_moves = 4


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
                    print("Invalid:", json_obj.get("action_feedback", {}))
    except ConnectionRefusedError:
        print("Impossible to connect to server : Server unavailable")
    except ConnectionResetError:
        print("Connection has been closed")
    except KeyboardInterrupt:
        print("Closing python client")

#region helpers
def type_id(value: str | None) -> float:
    if value is None:
        return float(len(types))
    return float(types.get(str(value).lower(), len(types)))


def status_id(value: str | None) -> float:
    if value is None:
        return 0.0
    return float(status.get(str(value), 0))


def stat_norm(value: float, denominator: float = 255.0) -> float:
    return float(value) / denominator

def get_attack_names(msg: dict, maximum: int = max_moves) -> list[str]:
    o = msg["opponent_infos"]["opponent_team"][0]
    attacks = o.get("attacks", [])

    attack_names = [f"Attack {i}" for i in range(maximum)]

    for a in attacks:
        slot = a.get("slot", None)
        if isinstance(slot, int) and 0 <= slot < maximum:
            attack_names[slot] = a.get("name", f"Attack {slot}")

    return attack_names

#endregion
#region json_data_extract
def pokemon_features(pokemon: dict | None) -> list[float]:
    """
    Get the statistics of a Pokémon.
    :param pokemon: JSON block representing a Pokémon
    :return: a list of floats that describes the statistics of a Pokémon
    """
    if pokemon is None:
        return [
            0.0,
            type_id("normal"),
            type_id(None),
            status_id("KO"),
            0.0, 0.0, 0.0, 0.0, 0.0,
        ]

    hp = float(pokemon.get("HP", 0.0))
    max_hp = float(max(1.0, pokemon.get("maxHP", 1.0)))

    stats = pokemon.get("stats", {})
    atk = stat_norm(stats.get("atk", pokemon.get("atk", 0.0)))
    defe = stat_norm(stats.get("def", pokemon.get("def", 0.0)))
    atk_spe = stat_norm(stats.get("atkSpe", pokemon.get("atkSpe", 0.0)))
    def_spe = stat_norm(stats.get("defSpe", pokemon.get("defSpe", 0.0)))
    speed = stat_norm(stats.get("speed", pokemon.get("speed", 0.0)))

    return [
        hp / max_hp,
        type_id(pokemon.get("type", "normal")),
        type_id(pokemon.get("type2")),
        status_id(pokemon.get("status", "normal")),
        atk,
        defe,
        atk_spe,
        def_spe,
        speed,
    ]

def get_moves_data_from_json(msg: dict, maximum: int = max_moves, identification: int = locked_id):
    o = msg["opponent_infos"]["opponent_team"][0]
    attacks = o.get("attacks", [])

    move_ids = np.full((maximum,), identification, dtype=np.int32)
    action_mask = np.zeros((maximum,), dtype=np.int8)
    move_names = [""] * maximum

    # [id, type_id, mode_id, power_norm, precision_norm, pp_norm, is_stab]
    move_features = np.zeros((maximum, 7), dtype=np.float32)
    move_features[:, 0] = float(identification)

    for a in attacks:
        slot = a.get("slot", None)
        if not (isinstance(slot, int) and 0 <= slot < maximum):
            continue

        mid = int(a.get("id", identification))
        name = a.get("name", "")

        move_ids[slot] = mid
        move_names[slot] = name

        move_type = types.get(str(a.get("type", "normal")).lower(), len(types))
        move_mode = modes.get(str(a.get("Mode", "status")).lower(), 2)

        power = float(a.get("Power", 0.0))
        precision = float(a.get("Precision", 0.0))

        pp = float(a.get("PP", 0.0))
        max_pp = float(a.get("maxPP", 1.0))
        pp_norm = pp / max(1.0, max_pp)

        power_norm = power / 150.0
        precision_norm = precision / 100.0
        is_stab = 1.0 if a.get("isSTAB", False) else 0.0

        move_features[slot] = np.array(
            [float(mid), float(move_type), float(move_mode), power_norm, precision_norm, pp_norm, is_stab],
            dtype=np.float32
        )

        action_mask[slot] = 1 if pp > 0 else 0

    compact_names = [n for n, m in zip(move_names, action_mask) if m == 1]
    return move_ids, action_mask, compact_names, move_features

def json_to_obs(msg: dict) -> np.ndarray:
    p_front = msg["player_infos"]["player_team"][0]

    o_team = msg["opponent_infos"]["opponent_team"]
    o_front = o_team[0]
    o_back = o_team[1] if len(o_team) > 1 and o_team[1] is not None else None

    agent_first = json_to_agent_first(msg)
    turn = float(msg.get("turn", 0))
    enemy_healthy = float(msg["player_infos"].get("healthy_pokemons", 0))
    agent_healthy = float(msg["opponent_infos"].get("healthy_pokemons", 0))

    _, _, _, move_features = get_moves_data_from_json(msg)

    obs = np.array(
        pokemon_features(p_front)
        + pokemon_features(o_front)
        + pokemon_features(o_back)
        + [agent_first, turn, enemy_healthy, agent_healthy]
        + move_features.flatten().tolist(),
        dtype=np.float32
    )

    return obs


def json_to_agent_first(msg: dict) -> float:
    prio = msg.get("Priority", {}).get("name", "")
    agent_name = msg.get("opponent_infos", {}).get("name", "opponent")
    return 1.0 if prio == agent_name else 0.0


def json_to_action_mask(msg: dict) -> np.ndarray:
    _, action_mask, _, _ = get_moves_data_from_json(msg)
    return action_mask


def json_to_terminated(msg: dict) -> bool:
    p_alive = msg["player_infos"].get("healthy_pokemons", 0) > 0
    o_alive = msg["opponent_infos"].get("healthy_pokemons", 0) > 0
    return (not p_alive) or (not o_alive)


def json_to_invalid_action_flag(msg: dict) -> float:
    feedback = msg.get("action_feedback", {})
    return 1.0 if feedback.get("opponent_invalid", False) else 0.0
#endregion

if __name__ == "__main__":
    main()