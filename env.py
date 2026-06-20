import gymnasium as gym
from gymnasium import spaces
import numpy as np
from battle_stat_tracker import BattleStatsTracker
from client import Client
from data import (
    build_action_mask,
    get_active_matchup,
    get_enemy_front,
    json_to_obs,
    json_to_terminated,
)
from matchup_evaluator import MatchupEvaluator
from reward_calculator import RewardCalculator
#region Constant
type_chart = {
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

invalid_action_penalty = 0.25
n_actions = 6
attack_actions = 4
switch_action = 4
item_action = 5

visible_pokemon_slots = 3
pokemon_feature_count = 9
battle_scalar_count = 4
move_feature_count = 7
observation_size = (
        visible_pokemon_slots * pokemon_feature_count
        + battle_scalar_count
        + attack_actions * move_feature_count
)

enemy_damage_reward_weight = 1.0
agent_damage_penalty_weight = 0.5
step_reward_offset = 0.01
enemy_ko_reward = 1.0
agent_ko_penalty = 1.0
switch_offense_weight = 0.04
switch_threat_weight = 0.05
switch_action_cost = 0.05
bad_switch_penalty = 0.20

#endregion

def build_observation_bounds(max_turns: int) -> tuple[np.ndarray, np.ndarray]:
    type_upper = float(len(type_chart))
    status_upper = 10.0

    pokemon_low = [0.0] * pokemon_feature_count
    pokemon_high = [1.0, type_upper, type_upper, status_upper, 1.5, 1.5, 1.5, 1.5, 1.5]

    battle_low = [0.0] * battle_scalar_count
    battle_high = [1.0, float(max_turns), 6.0, 6.0]

    move_low = [0.0] * move_feature_count
    move_high = [255.0, type_upper, 2.0, 1.0, 1.0, 1.0, 1.0]

    low = np.array(
        pokemon_low * visible_pokemon_slots
        + battle_low
        + move_low * attack_actions,
        dtype=np.float32,
    )
    high = np.array(
        pokemon_high * visible_pokemon_slots
        + battle_high
        + move_high * attack_actions,
        dtype=np.float32,
    )
    return low, high


def validate_observation_shape(obs: np.ndarray) -> None:
    expected_shape = (observation_size,)
    assert obs.shape == expected_shape, f"Expected obs shape {expected_shape}, got {obs.shape}"


#region env
class PokemonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=5001, max_turns=200, window_size=100):
        super().__init__()
        self.current_action_mask = np.ones(n_actions, dtype=bool)
        self.host = host
        self.port = port
        self.client = Client(host=host, port=port)
        self.last_msg = None
        self.max_turns = max_turns
        self.window_size = window_size
        self.matchup_evaluator = MatchupEvaluator(type_chart)
        self.reward_calculator = RewardCalculator(
            self.matchup_evaluator,
            invalid_action_penalty=invalid_action_penalty,
            switch_action=switch_action,
            enemy_damage_reward_weight=enemy_damage_reward_weight,
            agent_damage_penalty_weight=agent_damage_penalty_weight,
            step_reward_offset=step_reward_offset,
            enemy_ko_reward=enemy_ko_reward,
            agent_ko_penalty=agent_ko_penalty,
            switch_offense_weight=switch_offense_weight,
            switch_threat_weight=switch_threat_weight,
            switch_action_cost=switch_action_cost,
            bad_switch_penalty=bad_switch_penalty,
        )
        self.stats_tracker = BattleStatsTracker(
            window_size,
            attack_actions=attack_actions,
            total_actions=n_actions,
            switch_action=switch_action,
            item_action=item_action,
        )

        low, high = build_observation_bounds(max_turns)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(n_actions)

        self.turns = 0

    # region internal helpers
    def _send_action(self, action: int):
        self.stats_tracker.record_action(action)
        self.client.send_action(action)

    def _build_info(self, msg: dict) -> dict:
        feedback = msg.get("action_feedback", {})
        return {
            "raw": msg,
            "action_mask": self.current_action_mask.copy(),
            "opponent_invalid_action": bool(feedback.get("opponent_invalid", False)),
            "opponent_invalid_reason": feedback.get("opponent_invalid_reason", ""),
        }

    def _reset_episode_tracking(self):
        self.turns = 0
        self.stats_tracker.reset_episode()

    def _set_action_mask(self, msg: dict):
        self.current_action_mask = build_action_mask(msg).astype(bool)

    def _obs_from_message(self, msg: dict) -> np.ndarray:
        obs = json_to_obs(msg)
        validate_observation_shape(obs)
        return obs

    def _validate_action(self, action_id: int):
        if not 0 <= action_id < n_actions:
            raise ValueError(f"Action {action_id} out of bounds for action space size {n_actions}")
        if not self.current_action_mask[action_id]:
            raise ValueError(f"Invalid action {action_id} with mask {self.current_action_mask}")

    def _record_action_context(self, msg: dict, action_id: int):
        if action_id >= attack_actions:
            return

        self.stats_tracker.record_attack_context(
            msg,
            action_id,
            self.matchup_evaluator,
        )

    def _finalize_episode(self, msg: dict):
        did_win = get_enemy_front(msg)["status"] == "KO"
        self.stats_tracker.finalize_episode(did_win, self.turns, msg)

    # endregion
    # region gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.client.connect()
        self.client.reset(seed)
        self._reset_episode_tracking()

        msg = self.client.receive_message()
        self.last_msg = msg

        obs = self._obs_from_message(msg)
        self._set_action_mask(msg)
        return obs, self._build_info(msg)

    def step(self, action):
        action_id = int(action)
        self._validate_action(action_id)

        previous_msg = self.last_msg
        previous_enemy_front, previous_agent_front = get_active_matchup(previous_msg)

        if action_id < attack_actions:
            self._record_action_context(previous_msg, action_id)

        self._send_action(action_id)

        current_msg = self.client.receive_message()
        obs = self._obs_from_message(current_msg)
        current_enemy_front, current_agent_front = get_active_matchup(current_msg)

        if bool(current_msg.get("action_feedback", {}).get("opponent_invalid", False)):
            self.stats_tracker.record_invalid_action()

        reward = self.reward_calculator.compute_step_reward(
            previous_enemy_front,
            previous_agent_front,
            current_enemy_front,
            current_agent_front,
            current_msg,
            prev_msg=previous_msg,
            action=action_id,
        )

        terminated = json_to_terminated(current_msg)
        self.turns += 1
        truncated = self.turns >= self.max_turns

        if terminated or truncated:
            self._finalize_episode(current_msg)

        self._set_action_mask(current_msg)
        self.last_msg = current_msg

        return obs, float(reward), terminated, truncated, self._build_info(current_msg)

    def action_masks(self):
        return self.current_action_mask

    def close(self):
        self.client.close()
    # endregion
#endregion
