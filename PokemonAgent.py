import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json, socket

from Data import json_to_obs, json_to_terminated, json_to_action_mask


class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=5001, max_turns=200):
        super().__init__()
        self.host = host
        self.port = port
        self.max_turns = max_turns

        # obs = 32 (8 state + 24 moves)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

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

    def _send_action(self, action: int):
        # Java attend juste "0\n", "1\n", ...
        self.sock.sendall((str(int(action)) + "\n").encode("utf-8"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sock is None:
            self._connect()

        self.turns = 0

        # IMPORTANT:
        # Java envoie un premier state via sendState(initialState) au moment du start()
        msg = self._recv_msg()
        msg = self._recv_msg()
        obs = json_to_obs(msg)

        info = {
            "raw": msg,
            "action_mask": json_to_action_mask(msg),
        }
        return obs, info

    def step(self, action):
        # 1) envoyer action au moteur Java (fightLoop2 attend ici)
        self._send_action(action)

        # 2) recevoir l'état suivant (envoyé par le prochain sendStateWaitForAction ou le sendState final)
        msg = self._recv_msg()
        obs = json_to_obs(msg)
        assert obs.shape == (32,)

        terminated = json_to_terminated(msg)
        self.turns += 1
        truncated = self.turns >= self.max_turns

        # reward simple
        # Ici: on pénalise les dégâts subis par l’agent (opponent) et on récompense ceux infligés au player
        p = msg["player_infos"]["player_team"][0]      # enemy
        o = msg["opponent_infos"]["opponent_team"][0]  # agent
        reward = (1 - p["HP"] / max(1, p["maxHP"])) - (1 - o["HP"] / max(1, o["maxHP"]))

        info = {
            "raw": msg,
            "action_mask": json_to_action_mask(msg),
        }
        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            if self.f:
                self.f.close()
            if self.sock:
                self.sock.close()
        finally:
            self.f = None
            self.sock = None