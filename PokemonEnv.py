import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json, socket

from Data import json_to_obs, json_to_terminated


class PokemonEnv(gym.Env):
    def __init__(self, host="localhost", port=5001, max_turns=200):
        super().__init__()
        self.host = host
        self.port = port
        self.max_turns = max_turns

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # exemple: 4 attaques possibles

        self.sock = None
        self.f = None
        self.turns = 0

    def _connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.f = self.sock.makefile("r", encoding="utf-8")

    def _recv_msg(self) -> dict:
        while True:
            line = self.f.readline()
            if not line:
                raise ConnectionError("Server closed connection")
            line = line.strip()
            if line:
                return json.loads(line)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sock is None:
            self._connect()

        self.turns = 0

        # Selon ton protocole: soit le serveur envoie un état tout seul,
        # soit tu dois envoyer une commande "RESET" ici.
        msg = self._recv_msg()
        obs = json_to_obs(msg)
        info = {"raw": msg}
        return obs, info

    def step(self, action):
        # Selon ton protocole : tu dois envoyer l'action au serveur Java
        # Exemple simple si ton serveur attend une ligne JSON :
        payload = {"action": int(action)}
        self.sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

        msg = self._recv_msg()
        obs = json_to_obs(msg)

        terminated = json_to_terminated(msg)
        self.turns += 1
        truncated = self.turns >= self.max_turns

        # Reward (exemple basique, à adapter)
        p = msg["player_infos"]["player_team"][0]
        o = msg["opponent_infos"]["opponent_team"][0]
        reward = (1 - o["HP"]/o["maxHP"]) - (1 - p["HP"]/p["maxHP"])

        info = {"raw": msg}
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            if self.f: self.f.close()
            if self.sock: self.sock.close()
        finally:
            self.f = None
            self.sock = None
