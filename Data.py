import json, socket
import numpy as np


# Type representation
types = {"normal": 0, "fire": 1, "water": 2, "grass": 3, "electric": 4, "ice": 5, "fighting": 6,
         "poison": 7, "ground": 8, "flying": 9, "psychic": 10, "bug": 11, "rock": 12, "ghost": 13,
         "dragon": 14, "dark": 15, "steel": 16, "fairy": 17}

# Mode representation
modes = {"physical": 0, "special": 1, "status": 2}

# Status representation
status = {"normal": 0, "KO": 1}

locked_id = 255
max_moves = 4

# TCP connection between Python & Java
def main():
    host = 'localhost'
    port = 5001
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            print("Connected to server")
            f = s.makefile('r', encoding='utf-8')
            while True:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    json_obj = json.loads(line)
                    print("Received : ", json_obj)
                    print(json_to_obs(json_obj))

    except ConnectionRefusedError:
        print("Impossible to connect to server : Server unavailable")
    except ConnectionResetError:
        print("Connection has been closed")
    except KeyboardInterrupt:
        print("Closing python client")


def json_to_obs(msg: dict) -> np.ndarray:
    p = msg["player_infos"]["player_team"][0]      # enemy
    o = msg["opponent_infos"]["opponent_team"][0]  # agent

    p_hp = p["HP"] / max(1, p["maxHP"])
    o_hp = o["HP"] / max(1, o["maxHP"])

    p_type = types.get(p.get("type", "normal"), len(types))
    o_type = types.get(o.get("type", "normal"), len(types))

    p_status = status.get(p.get("status", "normal"), 0)
    o_status = status.get(o.get("status", "normal"), 0)

    agent_first = json_to_agent_first(msg)
    turn = float(msg.get("turn", 0))

    _, _, _, move_features = extract_moves(msg)

    obs = np.array(
        [o_hp, o_type, o_status, p_hp, p_type, p_status, agent_first, turn]
        + move_features.flatten().tolist(),
        dtype=np.float32
    )
    return obs


def json_to_agent_first(msg: dict) -> float:
    """
    Returns 1 if the agent has the priority else 0
    """
    prio = msg.get("Priority", {}).get("name", "")
    agent_name = msg.get("opponent_infos", {}).get("name", "opponent")
    agent_first = 1.0 if prio == agent_name else 0.0
    return agent_first


def json_to_action_mask(msg: dict) -> np.ndarray:
    _, action_mask, _, _ = extract_moves(msg)
    return action_mask


def json_to_terminated(msg: dict) -> bool:
    p_alive = msg["player_infos"].get("healthy_pokemons", 0) > 0
    o_alive = msg["opponent_infos"].get("healthy_pokemons", 0) > 0
    return (not p_alive) or (not o_alive)


def extract_moves(msg: dict, maximum: int = max_moves, identification: int = locked_id):
    """
    Extract the attacks of the agent.
    attacks = [{"slot":0,"id":0,"name":"Tackle"}, ...]
    Retourne:
      - move_ids: (maximum,) int32, padding=identification
      - action_mask: (maximum,) int8
      - compact_names: list[str] des moves valides
    """
    o = msg["opponent_infos"]["opponent_team"][0]
    attacks = o.get("attacks", [])

    move_ids = np.full((maximum,), identification, dtype=np.int32)
    action_mask = np.zeros((maximum,), dtype=np.int8)
    move_names = [""] * maximum

    # move_features: [id, type_id, mode_id, power_norm, precision_norm, pp_norm]
    move_features = np.zeros((maximum, 6), dtype=np.float32)

    # padding explicite: id = locked_id pour les slots vides
    move_features[:, 0] = float(identification)

    for a in attacks:
        slot = a.get("slot", None)
        if not (isinstance(slot, int) and 0 <= slot < maximum):
            continue

        mid = int(a.get("id", identification))
        name = a.get("name", "")

        move_ids[slot] = mid
        action_mask[slot] = 1
        move_names[slot] = name

        move_type = types.get(a.get("type", "normal"), len(types))
        move_mode = modes.get(str(a.get("Mode", "status")).lower(), 2)

        power = float(a.get("Power", 0.0))
        precision = float(a.get("Precision", 0.0))

        pp = float(a.get("PP", 0.0))
        max_pp = float(a.get("maxPP", 1.0))
        pp_norm = pp / max(1.0, max_pp)

        power_norm = power / 150.0
        precision_norm = precision / 100.0

        move_features[slot] = np.array(
            [float(mid), float(move_type), float(move_mode), power_norm, precision_norm, pp_norm],
            dtype=np.float32
        )

    compact_names = [n for n, m in zip(move_names, action_mask) if m == 1]
    return move_ids, action_mask, compact_names, move_features


if __name__ == "__main__":
    main()