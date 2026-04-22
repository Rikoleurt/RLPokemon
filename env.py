import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import socket
from collections import deque, defaultdict, Counter
from data import json_to_obs, json_to_terminated, get_attack_names

TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "electric": {"water": 2.0, "grass": 0.5, "electric": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "grass": 0.5, "electric": 2.0, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}

INVALID_ACTION_PENALTY = 0.25
N_ACTIONS = 6
ATTACK_ACTIONS = 4
SWITCH_ACTION = 4
ITEM_ACTION = 5


def get_pokemon_name(pokemon: dict) -> str:
    return pokemon.get("name") or pokemon.get("species") or "unknown"


def move_for_action(msg: dict, action: int) -> dict | None:
    if not (0 <= action < ATTACK_ACTIONS):
        return None

    attacks = msg["opponent_infos"]["opponent_team"][0].get("attacks", [])
    for attack in attacks:
        if attack.get("slot") == action:
            return attack
    return None


def effectiveness_multiplier(move_type: str, defender_type1: str, defender_type2: str | None = None) -> float:
    move_type = str(move_type).lower()
    defender_type1 = str(defender_type1).lower()

    mult = TYPE_CHART.get(move_type, {}).get(defender_type1, 1.0)

    if defender_type2 is not None:
        defender_type2 = str(defender_type2).lower()
        mult *= TYPE_CHART.get(move_type, {}).get(defender_type2, 1.0)

    return mult


def effectiveness_bucket(move_type: str, defender_type1: str, defender_type2: str | None = None) -> str:
    mult = effectiveness_multiplier(move_type, defender_type1, defender_type2)
    if mult > 1.0:
        return "super"
    if mult < 1.0:
        return "not_very"
    return "neutral"


def move_name_for_action(msg: dict, action: int) -> str:
    if action == SWITCH_ACTION:
        return "Switch"
    if action == ITEM_ACTION:
        return "Item"

    attack_names = get_attack_names(msg)
    if 0 <= action < len(attack_names):
        return attack_names[action]
    return f"Attack {action}"


def invalid_action_penalty(msg: dict) -> float:
    feedback = msg.get("action_feedback", {})
    return INVALID_ACTION_PENALTY if feedback.get("opponent_invalid", False) else 0.0


def compute_reward(prev_p, prev_o, new_p, new_o, msg):
    p_max = max(1, new_p["maxHP"])
    o_max = max(1, new_o["maxHP"])

    damage_to_player = (prev_p["HP"] - new_p["HP"]) / p_max
    damage_to_opponent = (prev_o["HP"] - new_o["HP"]) / o_max

    reward = damage_to_player - 0.5 * damage_to_opponent - 0.01

    if new_p["status"] == "KO":
        reward += 1.0
    if new_o["status"] == "KO":
        reward -= 1.0

    reward -= invalid_action_penalty(msg)
    return reward


def json_to_action_mask(msg: dict) -> np.ndarray:
    mask = np.zeros((N_ACTIONS,), dtype=np.int8)

    opponent_team = msg["opponent_infos"].get("opponent_team", [])
    if not opponent_team:
        return mask

    front = opponent_team[0]
    attacks = front.get("attacks", [])

    for attack in attacks:
        slot = attack.get("slot", None)
        if isinstance(slot, int) and 0 <= slot < ATTACK_ACTIONS:
            pp = float(attack.get("PP", 0.0))
            mask[slot] = 1 if pp > 0 else 0

    # Switch possible si un autre Pokémon vivant est disponible
    if len(opponent_team) > 1:
        for p in opponent_team[1:]:
            if p is not None and p.get("status") != "KO":
                mask[SWITCH_ACTION] = 1
                break

    # Item possible : on infère depuis le JSON retourné par Java
    # Comme le sac n'est pas exposé, on utilise le signal métier suivant :
    # si GameState autorise itemChoice(front), alors l'action sera valide côté Java
    # Ici on active l'item seulement si le front n'est pas full HP.
    # Si tu exposes plus tard l'inventaire / un booléen "can_use_item", il faudra le lire ici.
    if float(front.get("HP", 0.0)) < float(front.get("maxHP", 1.0)):
        mask[ITEM_ACTION] = 1

    return mask


class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=5001, max_turns=200, window_size=100):
        super().__init__()
        self.current_action_mask = np.ones(N_ACTIONS, dtype=bool)
        self.host = host
        self.port = port
        self.last_msg = None
        self.max_turns = max_turns
        self.window_size = window_size

        # Observation = 3 Pokémon visibles * 9 features + 4 scalars + 4 attaques * 7 features = 59
        type_upper = float(len(TYPE_CHART))
        status_upper = 10.0

        low = np.array(
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ] + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * ATTACK_ACTIONS,
            dtype=np.float32,
        )

        high = np.array(
            [
                1.0, type_upper, type_upper, status_upper, 1.5, 1.5, 1.5, 1.5, 1.5,
                1.0, type_upper, type_upper, status_upper, 1.5, 1.5, 1.5, 1.5, 1.5,
                1.0, type_upper, type_upper, status_upper, 1.5, 1.5, 1.5, 1.5, 1.5,
                1.0, float(max_turns), 6.0, 6.0,
            ] + [255.0, type_upper, 2.0, 1.0, 1.0, 1.0, 1.0] * ATTACK_ACTIONS,
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.nb_action0 = 0
        self.nb_action1 = 0
        self.nb_action2 = 0
        self.nb_action3 = 0
        self.ep_action_counts = np.zeros(N_ACTIONS, dtype=np.int32)

        self.switch_count = 0
        self.item_count = 0
        self.invalid_action_count = 0

        self.win = 0
        self.total_fights = 0

        self.attack_usage_history = [[] for _ in range(ATTACK_ACTIONS)]
        self.winrate_history = []

        self.attack_usage_moving_history = [[] for _ in range(ATTACK_ACTIONS)]
        self.winrate_moving_history = []

        self.recent_episode_actions = deque(maxlen=self.window_size)
        self.recent_episode_wins = deque(maxlen=self.window_size)

        self.fight_length_history = []
        self.fight_length_moving_history = []
        self.recent_episode_lengths = deque(maxlen=self.window_size)

        self.effectiveness_counts = {"super": 0, "neutral": 0, "not_very": 0}

        self.pokemon_move_name_counts = defaultdict(Counter)
        self.matchup_move_name_counts = defaultdict(Counter)
        self.global_move_name_counts = Counter()
        self.ep_pokemon_move_name_counts = defaultdict(Counter)
        self.episode_pokemon_move_name_counts_history = []

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
        elif action == SWITCH_ACTION:
            self.switch_count += 1
        elif action == ITEM_ACTION:
            self.item_count += 1

        self.ep_action_counts[action] += 1
        self.sock.sendall((str(int(action)) + "\n").encode("utf-8"))

    def _record_action_context(self, msg: dict, action: int):
        if action >= ATTACK_ACTIONS:
            return

        agent_front = msg["opponent_infos"]["opponent_team"][0]
        enemy_front = msg["player_infos"]["player_team"][0]

        agent_name = get_pokemon_name(agent_front)
        enemy_name = get_pokemon_name(enemy_front)
        matchup_name = f"{agent_name} vs {enemy_name}"

        move_name = move_name_for_action(msg, action)

        self.pokemon_move_name_counts[agent_name][move_name] += 1
        self.matchup_move_name_counts[matchup_name][move_name] += 1
        self.global_move_name_counts[move_name] += 1
        self.ep_pokemon_move_name_counts[agent_name][move_name] += 1

        move = move_for_action(msg, action)
        if move is not None:
            move_type = move.get("type", "normal")
            defender_type1 = enemy_front.get("type", "normal")
            defender_type2 = enemy_front.get("type2")
            bucket = effectiveness_bucket(move_type, defender_type1, defender_type2)
            self.effectiveness_counts[bucket] += 1

    def _update_episode_stats(self, did_win: bool, fight_length: int):
        self.winrate_history.append(100.0 * self.win / max(1, self.total_fights))

        global_counts = np.array(
            [self.nb_action0, self.nb_action1, self.nb_action2, self.nb_action3],
            dtype=np.int64,
        )
        total_actions = np.sum(global_counts)

        for i in range(ATTACK_ACTIONS):
            usage = 100.0 * global_counts[i] / max(1, total_actions)
            self.attack_usage_history[i].append(usage)

        self.recent_episode_actions.append(self.ep_action_counts.copy())
        self.recent_episode_wins.append(1 if did_win else 0)

        moving_winrate = 100.0 * np.mean(self.recent_episode_wins)
        self.winrate_moving_history.append(moving_winrate)

        recent_actions = np.array(self.recent_episode_actions)
        action_sum = np.sum(recent_actions, axis=0)
        total_recent_attack_actions = np.sum(action_sum[:ATTACK_ACTIONS])

        for i in range(ATTACK_ACTIONS):
            moving_usage = 100.0 * action_sum[i] / max(1, total_recent_attack_actions)
            self.attack_usage_moving_history[i].append(moving_usage)

        self.fight_length_history.append(fight_length)
        self.recent_episode_lengths.append(fight_length)
        self.fight_length_moving_history.append(float(np.mean(self.recent_episode_lengths)))
        self.episode_pokemon_move_name_counts_history.append(
            {
                pokemon: Counter(counter)
                for pokemon, counter in self.ep_pokemon_move_name_counts.items()
            }
        )

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
        self.ep_pokemon_move_name_counts = defaultdict(Counter)

        msg = self._recv_msg()
        self.last_msg = msg

        obs = json_to_obs(msg)
        assert obs.shape == (59,), f"Expected obs shape (59,), got {obs.shape}"

        self.current_action_mask = json_to_action_mask(msg).astype(bool)

        info = {
            "raw": msg,
            "action_mask": self.current_action_mask.copy(),
            "opponent_invalid_action": bool(msg.get("action_feedback", {}).get("opponent_invalid", False)),
            "opponent_invalid_reason": msg.get("action_feedback", {}).get("opponent_invalid_reason", ""),
        }
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

        # Comptage tactique uniquement pour les attaques
        if action < ATTACK_ACTIONS:
            self._record_action_context(prev_msg, action)

        # Comptage global + envoi socket
        self._send_action(action)

        msg = self._recv_msg()
        obs = json_to_obs(msg)
        assert obs.shape == (59,), f"Expected obs shape (59,), got {obs.shape}"

        new_p = msg["player_infos"]["player_team"][0]
        new_o = msg["opponent_infos"]["opponent_team"][0]

        if bool(msg.get("action_feedback", {}).get("opponent_invalid", False)):
            self.invalid_action_count += 1

        reward = compute_reward(prev_p, prev_o, new_p, new_o, msg)

        terminated = json_to_terminated(msg)
        self.turns += 1
        truncated = self.turns >= self.max_turns

        if terminated or truncated:
            self.total_fights += 1
            did_win = new_p["status"] == "KO"

            if did_win:
                self.win += 1

            self._update_episode_stats(did_win, self.turns)

        self.current_action_mask = json_to_action_mask(msg).astype(bool)

        info = {
            "raw": msg,
            "action_mask": self.current_action_mask.copy(),
            "opponent_invalid_action": bool(msg.get("action_feedback", {}).get("opponent_invalid", False)),
            "opponent_invalid_reason": msg.get("action_feedback", {}).get("opponent_invalid_reason", ""),
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