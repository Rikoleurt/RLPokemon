from collections import Counter, defaultdict, deque

import numpy as np

from data import (
    action_name_for_id,
    get_agent_front,
    get_enemy_front,
    get_pokemon_name,
    move_for_action_id,
)


def get_move_usage_context(
        msg: dict,
    action_id: int,
    agent_front: dict | None,
    enemy_front: dict | None,
) -> tuple[str, str, str, str]:
    agent_name = get_pokemon_name(agent_front)
    enemy_name = get_pokemon_name(enemy_front)
    matchup_name = f"{agent_name} vs {enemy_name}"
    move_name = action_name_for_id(msg, action_id)

    return agent_name, enemy_name, matchup_name, move_name


class BattleStatsTracker:
    def __init__(
        self,
        window_size: int,
        attack_actions: int,
        total_actions: int,
        switch_action: int,
        item_action: int,
    ):
        self.window_size = window_size
        self.attack_actions = attack_actions
        self.total_actions = total_actions
        self.switch_action = switch_action
        self.item_action = item_action

        self.global_attack_counts = np.zeros(attack_actions, dtype=np.int64)
        self.episode_action_counts = np.zeros(total_actions, dtype=np.int32)

        self.switch_count = 0
        self.item_count = 0
        self.invalid_action_count = 0

        self.win = 0
        self.total_fights = 0

        self.attack_usage_history = [[] for _ in range(attack_actions)]
        self.winrate_history = []

        self.attack_usage_moving_history = [[] for _ in range(attack_actions)]
        self.winrate_moving_history = []

        self.recent_episode_actions = deque(maxlen=window_size)
        self.recent_episode_wins = deque(maxlen=window_size)

        self.fight_length_history = []
        self.fight_length_moving_history = []
        self.recent_episode_lengths = deque(maxlen=window_size)

        self.effectiveness_counter = {"super": 0, "neutral": 0, "not_very": 0}

        self.pokemon_move_counter = defaultdict(Counter)
        self.matchup_move_counter = defaultdict(Counter)
        self.global_move_counter = Counter()
        self.episode_pokemon_move_counter = defaultdict(Counter)
        self.pokemon_move_counter_history_by_episode = []

        self.final_history = []

    # region Public tracking API
    def reset_episode(self):
        self.episode_action_counts[:] = 0
        self.episode_pokemon_move_counter = defaultdict(Counter)

    def record_action(self, action_id: int):
        if 0 <= action_id < self.attack_actions:
            self.global_attack_counts[action_id] += 1
        elif action_id == self.switch_action:
            self.switch_count += 1
        elif action_id == self.item_action:
            self.item_count += 1

        if 0 <= action_id < self.total_actions:
            self.episode_action_counts[action_id] += 1

    def record_invalid_action(self):
        self.invalid_action_count += 1

    def record_attack_context(
        self,
        msg: dict,
        action_id: int,
        matchup_evaluator,
    ):
        agent_front = get_agent_front(msg)
        enemy_front = get_enemy_front(msg)
        agent_name, enemy_name, matchup_name, move_name = get_move_usage_context(
            msg,
            action_id,
            agent_front,
            enemy_front,
        )

        self._record_move_usage(agent_name, matchup_name, move_name)
        self._record_move_effectiveness(msg, action_id, enemy_front, matchup_evaluator)

    def finalize_episode(self, did_win: bool, fight_length: int, terminal_msg: dict):
        self.total_fights += 1
        if did_win:
            self.win += 1

        self.final_history.append(terminal_msg)
        self._update_metrics(did_win, fight_length)
    # endregion

    # region Context of attack usage

    def _record_move_usage(
        self,
        agent_name: str,
        matchup_name: str,
        move_name: str,
    ):
        self.pokemon_move_counter[agent_name][move_name] += 1
        self.matchup_move_counter[matchup_name][move_name] += 1
        self.global_move_counter[move_name] += 1
        self.episode_pokemon_move_counter[agent_name][move_name] += 1

    def _record_move_effectiveness(
        self,
        msg: dict,
        action_id: int,
        enemy_front: dict | None,
        matchup_evaluator,
    ):
        move = move_for_action_id(msg, action_id)
        if move is None or enemy_front is None:
            return

        bucket = matchup_evaluator.effectiveness_to_string(
            move.get("type", "normal"),
            enemy_front.get("type", "normal"),
            enemy_front.get("type2"),
        )
        self.effectiveness_counter[bucket] += 1
    # endregion

    # region Episode outcome
    def _update_metrics(self, did_win: bool, fight_length: int):
        self._record_cumulative_winrate()
        self._record_cumulative_attack_usage()
        self._record_moving_metrics(did_win)
        self._record_fight_length(fight_length)
        self._save_current_episode_move_usage()

    def _record_cumulative_winrate(self):
        self.winrate_history.append(100.0 * self.win / max(1, self.total_fights))

    def _record_cumulative_attack_usage(self):
        total_attack_actions = np.sum(self.global_attack_counts)
        for action_id in range(self.attack_actions):
            usage = 100.0 * self.global_attack_counts[action_id] / max(1, total_attack_actions)
            self.attack_usage_history[action_id].append(usage)

    def _record_moving_metrics(self, did_win: bool):
        self.recent_episode_actions.append(self.episode_action_counts.copy())
        self.recent_episode_wins.append(1 if did_win else 0)
        self.winrate_moving_history.append(100.0 * np.mean(self.recent_episode_wins))

        recent_actions = np.array(self.recent_episode_actions)
        recent_action_sum = np.sum(recent_actions, axis=0)
        total_recent_attack_actions = np.sum(recent_action_sum[:self.attack_actions])

        for action_id in range(self.attack_actions):
            moving_usage = 100.0 * recent_action_sum[action_id] / max(1, total_recent_attack_actions)
            self.attack_usage_moving_history[action_id].append(moving_usage)

    def _record_fight_length(self, fight_length: int):
        self.fight_length_history.append(fight_length)
        self.recent_episode_lengths.append(fight_length)
        self.fight_length_moving_history.append(float(np.mean(self.recent_episode_lengths)))

    def _save_current_episode_move_usage(self):
        self.pokemon_move_counter_history_by_episode.append(
            {
                pokemon: Counter(counter)
                for pokemon, counter in self.episode_pokemon_move_counter.items()
            }
        )
    # endregion
