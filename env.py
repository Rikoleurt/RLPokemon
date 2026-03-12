import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json, socket

from data import json_to_obs, json_to_terminated, json_to_action_mask


def ko_points(p, o):
    p_status = p["status"]
    o_status = o["status"]
    if p_status == "KO":
        return -4
    elif o_status == "KO":
        return 4
    else:
        return 0


class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=5001, max_turns=200):
        super().__init__()
        self.current_action_mask = np.ones(4, dtype=bool)
        self.host = host
        self.port = port
        self.max_turns = max_turns

        low = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 4,
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 18.0, 1.0, 1.0, 18.0, 1.0, 1.0, float(max_turns)]
            + [255.0, 18.0, 2.0, 2.0, 1.0, 1.0] * 4,
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.sock = None
        self.f = None
        self.turns = 0
        print("ACTION SPACE:", self.action_space)

    def _send_cmd(self, cmd: str):
        self.sock.sendall((cmd.strip() + "\n").encode("utf-8"))

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
        self.sock.sendall((str(int(action)) + "\n").encode("utf-8"))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.sock is None:
            self._connect()

        if seed is None:
            self._send_cmd("RESET")
            print("RESET signal sent")
        else:
            self._send_cmd(f"RESET {int(seed)}")
            print(f"RESET signal sent with seed {int(seed)}")

        self.turns = 0

        msg = self._recv_msg()
        obs = json_to_obs(msg)
        assert obs.shape == (32,)

        self.current_action_mask = json_to_action_mask(msg).astype(bool)
        print("MASK AFTER RESET:", self.current_action_mask)

        info = {"raw": msg, "action_mask": self.current_action_mask.copy()}
        return obs, info

    def step(self, action):
        action = int(action)

        if not self.current_action_mask[action]:
            raise ValueError(
                f"Invalid action {action} with mask {self.current_action_mask}"
            )

        self._send_action(action)

        msg = self._recv_msg()
        obs = json_to_obs(msg)
        assert obs.shape == (32,)

        terminated = json_to_terminated(msg)
        self.turns += 1
        truncated = self.turns >= self.max_turns

        p = msg["player_infos"]["player_team"][0]      # player
        o = msg["opponent_infos"]["opponent_team"][0]  # agent

        reward = p["HP"] / max(1, p["maxHP"]) + ko_points(o, p)

        # IMPORTANT: mise à jour du masque pour l'état suivant
        self.current_action_mask = json_to_action_mask(msg).astype(bool)

        info = {
            "raw": msg,
            "action_mask": self.current_action_mask.copy(),
        }

        print("ACTION SENT TO JAVA:", action)
        print("NEW MASK:", self.current_action_mask)

        return obs, float(reward), terminated, truncated, info

    def action_masks(self):
        return self.current_action_mask

    def close(self):
        try:
            if self.sock:
                self._send_cmd("DONE")
                print("DONE signal sent")
        except:
            pass
        finally:
            try:
                if self.f:
                    self.f.close()
                if self.sock:
                    self.sock.close()
            finally:
                self.f = None
                self.sock = None