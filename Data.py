import json, socket
import numpy as np

# Type representation
types = {"normal": 0, "fire": 1, "water": 2, "grass": 3, "electric": 4, "ice": 5, "fighting": 6,
         "poison": 7, "ground": 8, "flying": 9, "psychic": 10, "bug": 11, "rock": 12, "ghost": 13,
         "dragon": 14, "dark": 15, "steel": 16, "fairy": 17}

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
    except KeyboardInterrupt:
        print("Closing python client")

def json_to_obs(msg: dict) -> np.ndarray:
    p = msg["player_infos"]["player_team"][0]
    o = msg["opponent_infos"]["opponent_team"][0]

    p_hp = p["HP"] / max(1, p["maxHP"])
    o_hp = o["HP"] / max(1, o["maxHP"])

    p_type = types.get(p.get("type", "normal"), len(types))
    o_type = types.get(o.get("type", "normal"), len(types))

    p_status = status.get(p.get("status", "normal"), 0)
    o_status = status.get(o.get("status", "normal"), 0)

    priority = 1.0 if msg.get("Priority", {}).get("name", False) else 0.0
    turn = float(msg.get("turn", 0))

    opp_move_ids, action_mask, _ = extract_moves(msg)


    return np.array(
        [p_hp, p_type, p_status, o_hp, o_type, o_status, priority, turn] + opp_move_ids.tolist(),
        dtype=np.float32
    )

def json_to_terminated(msg: dict) -> bool:
    p_alive = msg["player_infos"].get("healthy_pokemons", 0) > 0
    o_alive = msg["opponent_infos"].get("healthy_pokemons", 0) > 0
    return (not p_alive) or (not o_alive)


def extract_moves(msg: dict, maximum: int = max_moves, identification: int = locked_id):
    o = msg["opponent_infos"]["opponent_team"][0]
    attacks = o.get("attacks", [])

    move_ids = np.full((maximum,), identification, dtype=np.int32)
    action_mask = np.zeros((maximum,), dtype=np.int8)
    move_names = [""] * maximum

    for a in attacks:
        slot = a.get("slot")
        mid = a.get("id", identification)
        name = a.get("name", "")

        if isinstance(slot, int) and 0 <= slot < maximum:
            move_ids[slot] = int(mid)
            action_mask[slot] = 1
            move_names[slot] = name

    compact_names = [n for n, m in zip(move_names, action_mask) if m == 1]
    return move_ids, action_mask, compact_names

if __name__ == "__main__":
    main()
