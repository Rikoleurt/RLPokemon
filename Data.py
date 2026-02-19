import json, socket
import numpy as np

# Type representation
types = {"normal": 0, "fire": 1, "water": 2, "grass": 3, "electric": 4, "ice": 5, "fighting": 6,
         "poison": 7, "ground": 8, "flying": 9, "psychic": 10, "bug": 11, "rock": 12, "ghost": 13,
         "dragon": 14, "dark": 15, "steel": 16, "fairy": 17}

# Status representation
status = {"normal": 0, "KO": 1}

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
                    jsonObj = json.loads(line)
                    print("Received : ", jsonObj)
                    print(json_to_obs(jsonObj))
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

    return np.array([p_hp, p_type, p_status, o_hp, o_type, o_status, priority, turn])

if __name__ == "__main__":
    main()
