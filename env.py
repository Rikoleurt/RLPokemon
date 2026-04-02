import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json, socket
from collections import deque

from data import json_to_obs, json_to_terminated, json_to_action_mask


def compute_reward(prev_p, prev_o, new_p, new_o):
    p_max = max(1, new_p["maxHP"])
    o_max = max(1, new_o["maxHP"])

    damage_to_player = (prev_p["HP"] - new_p["HP"]) / p_max
    damage_to_opponent = (prev_o["HP"] - new_o["HP"]) / o_max

    reward = damage_to_player - 0.5 * damage_to_opponent - 0.01

    if new_p["status"] == "KO":
        reward += 1.0
    if new_o["status"] == "KO":
        reward -= 1.0

    return reward


class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=5001, max_turns=200, window_size=100):
        super().__init__()
        self.current_action_mask = np.ones(4, dtype=bool)
        self.host = host
        self.port = port
        self.last_msg = None
        self.max_turns = max_turns
        self.window_size = window_size

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

        self.nb_action0 = 0
        self.nb_action1 = 0
        self.nb_action2 = 0
        self.nb_action3 = 0

        self.ep_action_counts = np.zeros(4, dtype=np.int32)

        self.win = 0
        self.total_fights = 0

        self.attack_usage_history = [[] for _ in range(4)]
        self.winrate_history = []

        self.attack_usage_moving_history = [[] for _ in range(4)]
        self.winrate_moving_history = []

        self.recent_episode_actions = deque(maxlen=self.window_size)
        self.recent_episode_wins = deque(maxlen=self.window_size)

        self.sock = None
        self.f = None
        self.turns = 0

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
        if action == 0:
            self.nb_action0 += 1
        elif action == 1:
            self.nb_action1 += 1
        elif action == 2:
            self.nb_action2 += 1
        elif action == 3:
            self.nb_action3 += 1

        self.ep_action_counts[action] += 1
        self.sock.sendall((str(int(action)) + "\n").encode("utf-8"))

    def _update_episode_stats(self, did_win: bool):
        # winrate cumulé
        self.winrate_history.append(100.0 * self.win / max(1, self.total_fights))

        # usage cumulé global
        global_counts = np.array(
            [self.nb_action0, self.nb_action1, self.nb_action2, self.nb_action3],
            dtype=np.int64,
        )
        total_actions = np.sum(global_counts)

        for i in range(4):
            usage = 100.0 * global_counts[i] / max(1, total_actions)
            self.attack_usage_history[i].append(usage)

        # MAJ des buffers glissants
        self.recent_episode_actions.append(self.ep_action_counts.copy())
        self.recent_episode_wins.append(1 if did_win else 0)

        # winrate glissant
        moving_winrate = 100.0 * np.mean(self.recent_episode_wins)
        self.winrate_moving_history.append(moving_winrate)

        # usage glissant
        action_sum = np.sum(np.array(self.recent_episode_actions), axis=0)
        total_recent_actions = np.sum(action_sum)

        for i in range(4):
            moving_usage = 100.0 * action_sum[i] / max(1, total_recent_actions)
            self.attack_usage_moving_history[i].append(moving_usage)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.sock is None:
            self._connect()

        if seed is None:
            self._send_cmd("RESET")
        else:
            self._send_cmd(f"RESET {int(seed)}")

        self.turns = 0
        self.ep_action_counts[:] = 0

        msg = self._recv_msg()
        self.last_msg = msg

        obs = json_to_obs(msg)
        assert obs.shape == (32,)

        self.current_action_mask = json_to_action_mask(msg).astype(bool)

        info = {"raw": msg, "action_mask": self.current_action_mask.copy()}
        return obs, info

    def step(self, action):
        action = int(action)

        if not self.current_action_mask[action]:
            raise ValueError(
                f"Invalid action {action} with mask {self.current_action_mask}"
            )

        prev_msg = self.last_msg
        prev_p = prev_msg["player_infos"]["player_team"][0]
        prev_o = prev_msg["opponent_infos"]["opponent_team"][0]

        self._send_action(action)

        msg = self._recv_msg()
        obs = json_to_obs(msg)
        assert obs.shape == (32,)

        new_p = msg["player_infos"]["player_team"][0]
        new_o = msg["opponent_infos"]["opponent_team"][0]

        reward = compute_reward(prev_p, prev_o, new_p, new_o)

        terminated = json_to_terminated(msg)
        self.turns += 1
        truncated = self.turns >= self.max_turns

        if terminated or truncated:
            self.total_fights += 1
            did_win = new_p["status"] == "KO"

            if did_win:
                self.win += 1

            self._update_episode_stats(did_win)

        self.current_action_mask = json_to_action_mask(msg).astype(bool)

        info = {
            "raw": msg,
            "action_mask": self.current_action_mask.copy(),
        }

        self.last_msg = msg

        return obs, float(reward), terminated, truncated, info

    def action_masks(self):
        return self.current_action_mask

    def close(self):
        try:
            if self.sock:
                self._send_cmd("DONE")
        except Exception:
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